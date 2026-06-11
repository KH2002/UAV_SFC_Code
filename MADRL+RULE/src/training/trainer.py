# -*- coding: utf-8 -*-
"""MAPPO 训练器（对接当前单Agent轮转环境）。"""

from __future__ import annotations

import math
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
import yaml

from models import MAPPOPolicy, collate_obs_batch, obs_to_tensors


@dataclass
class MAPPOTrainerConfig:
    # PPO
    lr: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    mini_batch_size: int = 128

    # rollout / training
    rollout_steps: int = 1024
    total_timesteps: int = 200_000
    normalize_advantages: bool = True
    use_reward_norm: bool = False
    use_lr_schedule: bool = False
    lr_schedule: str = "cosine_warmup"
    lr_warmup_ratio: float = 0.1
    lr_warmup_init_ratio: float = 0.2
    lr_min_ratio: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_each_step: bool = False


def load_trainer_config(yaml_path: Optional[str] = None) -> MAPPOTrainerConfig:
    """从 YAML 加载训练配置（优先使用 DRL/training/config.yaml 的 ppo/training 段）。"""
    cfg = MAPPOTrainerConfig()
    yaml_path = yaml_path or os.path.join("DRL", "training", "config.yaml")

    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}

        ppo = user_cfg.get("ppo", {}) if isinstance(user_cfg.get("ppo"), dict) else {}
        training = user_cfg.get("training", {}) if isinstance(user_cfg.get("training"), dict) else {}
        logging_cfg = user_cfg.get("logging", {}) if isinstance(user_cfg.get("logging"), dict) else {}

        cfg.lr = float(ppo.get("lr", cfg.lr))
        cfg.gamma = float(ppo.get("gamma", cfg.gamma))
        cfg.gae_lambda = float(ppo.get("lambda", cfg.gae_lambda))
        cfg.clip_eps = float(ppo.get("epsilon", cfg.clip_eps))
        cfg.value_coef = float(ppo.get("value_loss_coef", cfg.value_coef))
        cfg.entropy_coef = float(ppo.get("entropy_coef", cfg.entropy_coef))
        cfg.max_grad_norm = float(ppo.get("max_grad_norm", cfg.max_grad_norm))
        cfg.ppo_epochs = int(ppo.get("ppo_epochs", cfg.ppo_epochs))
        cfg.mini_batch_size = int(ppo.get("batch_size", cfg.mini_batch_size))

        # 可选覆盖：rollout_steps / max_step(total_timesteps) / normalize_advantages / device
        if training.get("rollout_steps") is not None:
            cfg.rollout_steps = int(training.get("rollout_steps"))

        if training.get("max_step") is not None:
            cfg.total_timesteps = int(training.get("max_step"))
        elif training.get("total_timesteps") is not None:
            cfg.total_timesteps = int(training.get("total_timesteps"))
        else:
            # 兼容旧配置：由 total_episodes * rollout_steps 粗估
            total_episodes = int(training.get("total_episodes", 0))
            if total_episodes > 0:
                cfg.total_timesteps = max(cfg.total_timesteps, total_episodes * cfg.rollout_steps)

        if training.get("normalize_advantages") is not None:
            cfg.normalize_advantages = bool(training.get("normalize_advantages"))
        if training.get("use_reward_norm") is not None:
            cfg.use_reward_norm = bool(training.get("use_reward_norm"))
        if training.get("use_lr_schedule") is not None:
            cfg.use_lr_schedule = bool(training.get("use_lr_schedule"))
        if isinstance(training.get("lr_schedule"), str) and training.get("lr_schedule").strip():
            cfg.lr_schedule = str(training.get("lr_schedule")).strip().lower()
            if cfg.lr_schedule not in {"none", "constant"}:
                cfg.use_lr_schedule = True
        if training.get("lr_warmup_ratio") is not None:
            cfg.lr_warmup_ratio = float(training.get("lr_warmup_ratio"))
        if training.get("lr_warmup_init_ratio") is not None:
            cfg.lr_warmup_init_ratio = float(training.get("lr_warmup_init_ratio"))
        if training.get("lr_min_ratio") is not None:
            cfg.lr_min_ratio = float(training.get("lr_min_ratio"))
        if isinstance(training.get("device"), str) and training.get("device").strip():
            cfg.device = training.get("device").strip()
        cfg.log_each_step = bool(logging_cfg.get("step_log", cfg.log_each_step))

    return cfg


class RolloutBuffer:
    """按时间步缓存 on-policy 轨迹。"""

    def __init__(self):
        self.obs: List[Dict[str, object]] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []

    def add(
        self,
        obs: Dict[str, object],
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self.obs.append(deepcopy(obs))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(1.0 if done else 0.0)

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.actions)


class RewardNormalizer:
    """奖励归一化器（Running Mean & Std）- Welford 算法。"""

    def __init__(self, eps: float = 1e-8):
        self.mean = 0.0
        self.m2 = 0.0
        self.count = 0
        self.eps = eps

    def update(self, reward: float) -> None:
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.m2 += delta * delta2

    def normalize(self, reward: float) -> float:
        if self.count < 1:
            return reward
        std = float(np.sqrt(self.m2 / self.count) + self.eps)
        return float((reward - self.mean) / std)


class MAPPOTrainer:
    def __init__(
        self,
        env,
        policy: MAPPOPolicy,
        config: Optional[MAPPOTrainerConfig] = None,
    ):
        self.env = env
        self.cfg = config or MAPPOTrainerConfig()
        self.device = torch.device(self.cfg.device)

        self.policy = policy.to(self.device)
        self.base_lr = float(self.cfg.lr)
        self.optimizer = Adam(self.policy.parameters(), lr=self.base_lr)
        self.reward_normalizer = RewardNormalizer() if self.cfg.use_reward_norm else None

        self.rollout = RolloutBuffer()
        self.current_obs = self.env.reset()

        self._running_ep_return = 0.0
        self._running_ep_len = 0
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_success_rates: List[float] = []
        self.total_episodes_done = 0
        self.total_env_steps = 0

    def _set_optimizer_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def _compute_lr_multiplier(self, update_idx: int, num_updates: int) -> float:
        schedule = str(self.cfg.lr_schedule).lower().strip()
        if not self.cfg.use_lr_schedule or schedule in {"none", "constant"} or num_updates <= 1:
            return 1.0

        # 先升后降：warmup线性上升 + cosine衰减
        warmup_updates = int(round(self.cfg.lr_warmup_ratio * num_updates))
        warmup_updates = min(max(warmup_updates, 1), num_updates - 1)
        warmup_start = float(np.clip(self.cfg.lr_warmup_init_ratio, 0.0, 1.0))
        min_ratio = float(np.clip(self.cfg.lr_min_ratio, 0.0, 1.0))

        if update_idx <= warmup_updates:
            if warmup_updates <= 1:
                return 1.0
            progress = float(update_idx - 1) / float(warmup_updates - 1)
            return warmup_start + (1.0 - warmup_start) * progress

        decay_total = max(num_updates - warmup_updates, 1)
        decay_step = update_idx - warmup_updates - 1  # 首个decay步从0开始
        if decay_total <= 1:
            cosine = 1.0
        else:
            progress = float(decay_step) / float(decay_total - 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> (np.ndarray, np.ndarray):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def collect_rollout(
        self,
        rollout_steps: Optional[int] = None,
        step_callback: Optional[Callable[[Dict[str, float]], None]] = None,
        episode_callback: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> Dict[str, np.ndarray]:
        steps = rollout_steps or self.cfg.rollout_steps
        self.rollout.clear()
        episodes_done_in_rollout = 0
        last_info: Dict[str, object] = {}

        for step_idx in range(steps):
            obs_t = obs_to_tensors(self.current_obs, device=self.device)
            current_agent_id = int(self.current_obs["current_agent_id"])
            action_mask = np.asarray(self.current_obs.get("action_mask", []), dtype=np.float32)
            action_mask_density = float(action_mask.mean()) if action_mask.size > 0 else 0.0

            with torch.no_grad():
                act_out = self.policy.act(obs_t, deterministic=False)
                value = self.policy.get_value(obs_t)
                deploy_score_table = self.policy.score_vnfs_for_deployment(obs_t)[0].detach().cpu().numpy()

            action = int(act_out["action"][0].item())
            log_prob = float(act_out["log_prob"][0].item())
            value_scalar = float(value[0, 0].item())

            next_obs, reward, done, info = self.env.step(
                current_agent_id,
                action,
                deploy_score_table=deploy_score_table,
            )
            raw_reward = float(reward)
            train_reward = raw_reward
            if self.cfg.use_reward_norm and self.reward_normalizer is not None:
                self.reward_normalizer.update(raw_reward)
                train_reward = self.reward_normalizer.normalize(raw_reward)
            self.total_env_steps += 1
            last_info = info

            self.rollout.add(
                obs=self.current_obs,
                action=action,
                log_prob=log_prob,
                value=value_scalar,
                reward=train_reward,
                done=bool(done),
            )

            self._running_ep_return += raw_reward
            self._running_ep_len += 1
            if step_callback is not None:
                step_callback(
                    {
                        "env_step": float(self.total_env_steps),
                        "episode": float(self.total_episodes_done),
                        "episode_step": float(self._running_ep_len),
                        "rollout_step": float(step_idx + 1),
                        "agent_id": float(current_agent_id),
                        "action": float(action),
                        "reward": raw_reward,
                        "invalid_count": float(info.get("invalid_count", 0)),
                        "action_mask_density": action_mask_density,
                        "done": float(int(bool(done))),
                        "current_time_slot": float(info.get("current_slot", np.nan)),
                        "current_round": float(info.get("current_round", np.nan)),
                        "turn_in_round": float(info.get("turn_in_round", np.nan)),
                        "pending_count": float(info.get("pending_count", np.nan)),
                        "success_rate": float(info.get("success_rate", 0.0)),
                    }
                )
            if self.cfg.log_each_step:
                print(
                    f"[STEP] env_step={self.total_env_steps} "
                    f"rollout_step={step_idx + 1}/{steps} "
                    f"agent={current_agent_id} action={action} "
                    f"reward={raw_reward:.4f} done={int(bool(done))} "
                    f"slot={int(info.get('current_slot', -1))} "
                    f"round={int(info.get('current_round', -1))} "
                    f"turn={int(info.get('turn_in_round', -1))} "
                    f"pending={int(info.get('pending_count', -1))} "
                    f"succ={float(info.get('success_rate', 0.0)):.4f}",
                    flush=True,
                )

            if done:
                self.episode_returns.append(self._running_ep_return)
                self.episode_lengths.append(self._running_ep_len)
                ep_success_rate = float(info.get("success_rate", 0.0))
                self.episode_success_rates.append(ep_success_rate)
                self.total_episodes_done += 1
                episodes_done_in_rollout += 1
                if episode_callback is not None:
                    ep_len = float(self.episode_lengths[-1])
                    ep_ret = float(self.episode_returns[-1])
                    episode_callback(
                        {
                            "episode": float(self.total_episodes_done),
                            "episode_length": ep_len,
                            "episode_return": ep_ret,
                            "avg_reward": ep_ret / max(ep_len, 1.0),
                            "success_rate": ep_success_rate,
                            "total_completed": float(info.get("completed_count", np.nan)),
                            "pending_count": float(info.get("pending_count", np.nan)),
                            "final_time_slot": float(info.get("current_slot", np.nan)),
                            "env_step": float(self.total_env_steps),
                        }
                    )
                if self.cfg.log_each_step:
                    print(
                        f"[EPISODE] done env_step={self.total_env_steps} "
                        f"ep_return={self.episode_returns[-1]:.4f} "
                        f"ep_len={self.episode_lengths[-1]} "
                        f"ep_success_rate={ep_success_rate:.4f}",
                        flush=True,
                    )
                self._running_ep_return = 0.0
                self._running_ep_len = 0
                self.current_obs = self.env.reset()
            else:
                self.current_obs = next_obs

        last_done = self.rollout.dones[-1] if len(self.rollout) > 0 else 1.0
        if last_done >= 1.0:
            last_value = 0.0
        else:
            with torch.no_grad():
                last_obs_t = obs_to_tensors(self.current_obs, device=self.device)
                last_value = float(self.policy.get_value(last_obs_t)[0, 0].item())

        rewards = np.asarray(self.rollout.rewards, dtype=np.float32)
        values = np.asarray(self.rollout.values, dtype=np.float32)
        dones = np.asarray(self.rollout.dones, dtype=np.float32)
        advantages, returns = self._compute_gae(rewards, values, dones, last_value)

        return {
            "advantages": advantages,
            "returns": returns,
            "rewards": rewards,
            "values": values,
            "dones": dones,
            "episodes_done_in_rollout": episodes_done_in_rollout,
            "last_info": last_info,
        }

    def _slice_obs_batch(self, obs_batch: Dict[str, torch.Tensor], idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in obs_batch.items()}

    def update(self, rollout_meta: Dict[str, np.ndarray]) -> Dict[str, float]:
        n = len(self.rollout)
        if n == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "loss": 0.0}

        obs_batch = collate_obs_batch(self.rollout.obs, device=self.device)
        actions = torch.as_tensor(self.rollout.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(self.rollout.log_probs, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(rollout_meta["advantages"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(rollout_meta["returns"], dtype=torch.float32, device=self.device)

        if self.cfg.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        total_kl = 0.0
        updates = 0

        batch_size = self.cfg.mini_batch_size

        for _ in range(self.cfg.ppo_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                mb_idx = perm[start:start + batch_size]
                mb_obs = self._slice_obs_batch(obs_batch, mb_idx)
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                eval_out = self.policy.evaluate_actions(mb_obs, mb_actions)
                new_log_probs = eval_out["log_prob"]
                entropy = eval_out["entropy"]
                values = eval_out["value"]

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                approx_kl = (mb_old_log_probs - new_log_probs).mean()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, mb_returns)
                entropy_loss = entropy.mean()

                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy_loss.item())
                total_loss += float(loss.item())
                total_kl += float(approx_kl.item())
                updates += 1

        return {
            "policy_loss": total_policy_loss / max(updates, 1),
            "value_loss": total_value_loss / max(updates, 1),
            "entropy": total_entropy / max(updates, 1),
            "kl_div": total_kl / max(updates, 1),
            "loss": total_loss / max(updates, 1),
            "avg_reward": float(np.mean(self.rollout.rewards)) if len(self.rollout.rewards) > 0 else 0.0,
            "avg_return_10ep": float(np.mean(self.episode_returns[-10:])) if len(self.episode_returns) > 0 else 0.0,
            "avg_episode_length_10ep": (
                float(np.mean(self.episode_lengths[-10:])) if len(self.episode_lengths) > 0 else 0.0
            ),
            "avg_success_rate_10ep": (
                float(np.mean(self.episode_success_rates[-10:])) if len(self.episode_success_rates) > 0 else 0.0
            ),
            "total_episodes_done": float(self.total_episodes_done),
        }

    def train(
        self,
        total_timesteps: Optional[int] = None,
        rollout_steps: Optional[int] = None,
        log_interval_updates: int = 10,
        update_callback: Optional[Callable[[Dict[str, float]], None]] = None,
        step_callback: Optional[Callable[[Dict[str, float]], None]] = None,
        episode_callback: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> List[Dict[str, float]]:
        total_steps = total_timesteps or self.cfg.total_timesteps
        r_steps = rollout_steps or self.cfg.rollout_steps

        num_updates = max(total_steps // r_steps, 1)
        logs: List[Dict[str, float]] = []

        for update_idx in range(1, num_updates + 1):
            lr_mult = self._compute_lr_multiplier(update_idx, num_updates)
            self._set_optimizer_lr(self.base_lr * lr_mult)
            rollout_meta = self.collect_rollout(
                r_steps,
                step_callback=step_callback,
                episode_callback=episode_callback,
            )
            stats = self.update(rollout_meta)
            stats["update"] = float(update_idx)
            stats["env_steps"] = float(self.total_env_steps)
            stats["episodes_done_in_rollout"] = float(rollout_meta.get("episodes_done_in_rollout", 0))
            last_info = rollout_meta.get("last_info", {}) if isinstance(rollout_meta.get("last_info", {}), dict) else {}
            stats["current_time_slot"] = float(last_info.get("current_slot", np.nan))
            stats["completed_count"] = float(last_info.get("completed_count", np.nan))
            stats["pending_count"] = float(last_info.get("pending_count", np.nan))
            stats["lr"] = float(self.optimizer.param_groups[0]["lr"])
            logs.append(stats)
            if update_callback is not None:
                update_callback(stats)

            if update_idx % log_interval_updates == 0 or update_idx == 1 or update_idx == num_updates:
                print(
                    f"[MAPPO] update={update_idx}/{num_updates} "
                    f"env_steps={int(update_idx * r_steps)} "
                    f"loss={stats['loss']:.4f} "
                    f"pi={stats['policy_loss']:.4f} "
                    f"v={stats['value_loss']:.4f} "
                    f"ent={stats['entropy']:.4f} "
                    f"kl={stats['kl_div']:.6f} "
                    f"lr={stats['lr']:.6g} "
                    f"rew={stats['avg_reward']:.4f} "
                    f"ep_ret10={stats['avg_return_10ep']:.4f} "
                    f"ep_len10={stats['avg_episode_length_10ep']:.2f} "
                    f"ep_succ10={stats['avg_success_rate_10ep']:.4f}",
                    flush=True,
                )

        return logs
