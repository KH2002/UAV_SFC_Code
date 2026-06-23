# -*- coding: utf-8 -*-
"""MPopLoc behavior cloning warm-start, then MAPPO fine-tuning."""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import importlib.util
import io
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_TRAINING_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_TRAINING_DIR, ".."))
_MADRL_DIR = os.path.abspath(os.path.join(_SRC_DIR, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_MADRL_DIR, ".."))
for _path in (_SRC_DIR, _MADRL_DIR, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config as env_config


def _load_module_from_file(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_DATASET_MODULE = _load_module_from_file(
    "DRL.training.dataset",
    os.path.join(_REPO_ROOT, "DRL", "training", "dataset.py"),
)
_LOGGER_MODULE = _load_module_from_file(
    "DRL.training.logger",
    os.path.join(_REPO_ROOT, "DRL", "training", "logger.py"),
)
create_or_load_dataset = _DATASET_MODULE.create_or_load_dataset
TrainingLogger = _LOGGER_MODULE.TrainingLogger
from envs.env import MAPPOSFCEnv
from models import MAPPOPolicy, collate_obs_batch, obs_to_tensors
from training import MAPPOTrainer, load_trainer_config
from mpoploc_teacher import MPopLocSolver


def _load_yaml(path: str) -> Dict[str, object]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_output_dir(base_dir: str, config_path: str) -> str:
    if os.path.isabs(base_dir):
        return base_dir
    config_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else os.getcwd()
    return os.path.abspath(os.path.join(config_dir, base_dir))


def _resolve_output_root(logging_cfg: Dict[str, object], config_path: str) -> str:
    output_dir_cfg = logging_cfg.get("output_dir")
    if isinstance(output_dir_cfg, str) and output_dir_cfg.strip():
        return _resolve_output_dir(output_dir_cfg, config_path)
    return os.path.join(_MADRL_DIR, "output")


def _infer_dims(obs: Dict[str, object], device: torch.device) -> Tuple[int, int, int]:
    obs_t = obs_to_tensors(obs, device=device)
    self_dim = int(obs_t["agent_self"].shape[-1])
    task_dim = int(obs_t["task_matrix"].shape[-1])
    context_dim = int(obs_t["context"].shape[-1])
    return self_dim, task_dim, context_dim


def _build_policy(obs: Dict[str, object], env: MAPPOSFCEnv, yaml_cfg: Dict[str, object], device: torch.device) -> MAPPOPolicy:
    self_dim, task_dim, context_dim = _infer_dims(obs, device)
    network_cfg = yaml_cfg.get("network", {}) if isinstance(yaml_cfg.get("network"), dict) else {}
    actor_mlp_hidden_dim_cfg = network_cfg.get("actor_mlp_hidden_dim", None)
    actor_mlp_hidden_dim = None if actor_mlp_hidden_dim_cfg is None else int(actor_mlp_hidden_dim_cfg)
    policy = MAPPOPolicy(
        self_dim=self_dim,
        task_dim=task_dim,
        context_dim=context_dim,
        action_dim=env.cfg.action_dim,
        hidden_dim=int(network_cfg.get("hidden_dim", 256)),
        num_heads=int(network_cfg.get("num_heads", 4)),
        dropout=float(network_cfg.get("dropout", 0.1)),
        num_actor_agent_blocks=int(network_cfg.get("num_encoder_layers", 1)),
        num_critic_uav_blocks=int(network_cfg.get("num_encoder_layers", 1)),
        num_critic_task_blocks=int(network_cfg.get("num_encoder_layers", 1)),
        actor_type=str(network_cfg.get("actor_type", "attn")).lower(),
        actor_mlp_hidden_dim=actor_mlp_hidden_dim,
        actor_mlp_num_layers=int(network_cfg.get("actor_mlp_num_layers", 2)),
    )
    return policy.to(device)


def _run_mpoploc_teacher(episode_data, num_time_slots: int):
    ep = episode_data.copy()
    solver = MPopLocSolver(ep.uavs, ep.requests, ep.locations, num_time_slots=num_time_slots)
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve()
    return solver


def _teacher_action_for_agent(solver: MPopLocSolver, env: MAPPOSFCEnv, agent_id: int) -> Optional[int]:
    slot_details = getattr(solver, "slot_uav_details", {}) or {}
    current_slot = int(env.current_time_slot)
    agent_detail = slot_details.get(current_slot, {}).get(int(agent_id))
    if not agent_detail:
        return None

    location_id = int(agent_detail.get("location_id", 0))
    if location_id <= 0 or location_id > env.cfg.num_locations:
        return None
    action = location_id - 1
    mask = np.asarray(env.get_action_mask(agent_id), dtype=np.int32)
    if action < 0 or action >= mask.shape[0] or int(mask[action]) <= 0:
        return None
    return int(action)


def _greedy_action_for_agent(env: MAPPOSFCEnv, obs: Dict[str, object]) -> Optional[int]:
    agent_id = int(obs["current_agent_id"])
    actions = np.flatnonzero(np.asarray(obs["action_mask"], dtype=np.int32) > 0)
    if actions.size <= 0:
        return None

    best_score = None
    best_actions: List[int] = []
    for action in actions:
        location_id = int(env.decode_action(int(action)))
        potential = float(env._count_location_sfc_potential(agent_id, location_id))
        popularity = 0.0
        for req_idx in range(env._actionable_request_count()):
            req = env.requests[req_idx]
            if req.is_serviced:
                continue
            popularity += sum(1 for vnf in req.vnfs[:2] if int(vnf.location_id) == location_id)
        score = (potential, popularity)
        if best_score is None or score > best_score:
            best_score = score
            best_actions = [int(action)]
        elif score == best_score:
            best_actions.append(int(action))

    return int(min(best_actions)) if best_actions else None


def collect_teacher_bc_samples(
    dataset,
    config_path: str,
    num_time_slots: int,
    seed: int,
    teacher: str,
    max_episodes: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    samples: List[Dict[str, object]] = []
    episode_rows: List[Dict[str, object]] = []
    num_eps = len(dataset) if max_episodes is None else min(int(max_episodes), len(dataset))
    teacher = str(teacher).lower()

    for ep_idx in range(num_eps):
        episode_data = dataset[ep_idx]
        solver = _run_mpoploc_teacher(episode_data, num_time_slots=num_time_slots) if teacher == "mpoploc" else None
        env = MAPPOSFCEnv(config_yaml_path=config_path, episode_data=episode_data, seed=seed + ep_idx)
        env.set_request_shuffle_on_reset(False)
        obs = env.reset()
        done = False
        used = 0
        skipped = 0
        ep_return = 0.0
        last_info: Dict[str, object] = {}

        while not done:
            agent_id = int(obs["current_agent_id"])
            if teacher == "greedy":
                action = _greedy_action_for_agent(env, obs)
            elif teacher == "mpoploc":
                action = _teacher_action_for_agent(solver, env, agent_id) if solver is not None else None
            else:
                raise ValueError(f"Unsupported teacher: {teacher}")

            if action is None:
                skipped += 1
                mask = np.asarray(obs["action_mask"], dtype=np.int32)
                valid_actions = np.flatnonzero(mask > 0)
                action = int(valid_actions[0]) if valid_actions.size > 0 else 0
            else:
                samples.append({"obs": copy.deepcopy(obs), "action": int(action)})
                used += 1

            obs, reward, done, info = env.step(agent_id, int(action), deploy_score_table=None)
            ep_return += float(reward)
            last_info = info

        episode_rows.append(
            {
                "episode_index": int(ep_idx),
                "dataset_episode_id": int(getattr(episode_data, "episode_id", ep_idx)),
                "dataset_seed": int(getattr(episode_data, "seed", -1)),
                "teacher": teacher,
                "bc_samples": int(used),
                "skipped_steps": int(skipped),
                "teacher_success_rate": float(last_info.get("success_rate", 0.0)),
                "teacher_completed_count": float(last_info.get("completed_count", 0.0)),
                "teacher_return": float(ep_return),
            }
        )

    return samples, episode_rows


def pretrain_actor_bc(
    policy: MAPPOPolicy,
    samples: List[Dict[str, object]],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    max_grad_norm: float,
    log_path: str,
    teacher_name: str = "teacher",
) -> List[Dict[str, float]]:
    if not samples:
        raise RuntimeError(f"No valid {teacher_name} BC samples were collected.")

    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=lr)
    rows: List[Dict[str, float]] = []
    indices = np.arange(len(samples))
    policy.train()

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "epoch", "loss", "accuracy", "samples", "lr"],
        )
        writer.writeheader()

        for epoch in range(1, int(epochs) + 1):
            np.random.shuffle(indices)
            total_loss = 0.0
            total_correct = 0
            total_seen = 0

            for start in range(0, len(indices), int(batch_size)):
                batch_idx = indices[start:start + int(batch_size)]
                obs_batch = [samples[int(i)]["obs"] for i in batch_idx]
                actions = torch.as_tensor([int(samples[int(i)]["action"]) for i in batch_idx], dtype=torch.long, device=device)
                obs_t = collate_obs_batch(obs_batch, device=device)

                out = policy.actor.forward(
                    agent_self=obs_t["agent_self"],
                    task_matrix=obs_t["task_matrix"],
                    vnf_location_ids=obs_t.get("vnf_location_ids"),
                    context=obs_t["context"],
                    current_agent_id=obs_t["current_agent_id"],
                    action_mask=obs_t.get("action_mask"),
                    agent_avail_mask=obs_t.get("agent_avail_mask"),
                )
                logits = out["logits"]
                loss = F.cross_entropy(logits, actions)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), float(max_grad_norm))
                optimizer.step()

                total_loss += float(loss.item()) * int(actions.numel())
                total_correct += int((logits.argmax(dim=-1) == actions).sum().item())
                total_seen += int(actions.numel())

            row = {
                "timestamp": datetime.now().isoformat(),
                "epoch": float(epoch),
                "loss": float(total_loss / max(total_seen, 1)),
                "accuracy": float(total_correct / max(total_seen, 1)),
                "samples": float(total_seen),
                "lr": float(lr),
            }
            writer.writerow(row)
            f.flush()
            rows.append(row)
            print(
                f"[{teacher_name}-BC] epoch={epoch}/{epochs} loss={row['loss']:.6f} acc={row['accuracy']:.4f} samples={total_seen}",
                flush=True,
            )

    return rows


def _run_deterministic_eval(
    trainer: MAPPOTrainer,
    eval_env: MAPPOSFCEnv,
    eval_dataset,
    eval_episodes: int,
    device: torch.device,
) -> Dict[str, float]:
    was_training = trainer.policy.training
    trainer.policy.eval()
    total_reward = 0.0
    total_success = 0.0
    total_ep_len = 0.0
    total_completed = 0.0

    with torch.no_grad():
        for i in range(eval_episodes):
            ep_data = eval_dataset[i % len(eval_dataset)]
            eval_env.set_episode_data(ep_data)
            obs = eval_env.reset()
            done = False
            ep_reward = 0.0
            ep_len = 0
            last_info: Dict[str, object] = {}
            while not done:
                obs_t = obs_to_tensors(obs, device=device)
                act_out = trainer.policy.act(obs_t, deterministic=True)
                deploy_score_table = trainer.policy.score_vnfs_for_deployment(obs_t)[0].detach().cpu().numpy()
                agent_id = int(obs["current_agent_id"])
                action = int(act_out["action"][0].item())
                obs, reward, done, info = eval_env.step(agent_id, action, deploy_score_table=deploy_score_table)
                ep_reward += float(reward)
                ep_len += 1
                last_info = info
            total_reward += ep_reward
            total_ep_len += float(ep_len)
            total_success += float(last_info.get("success_rate", 0.0))
            total_completed += float(last_info.get("completed_count", 0.0))

    if was_training:
        trainer.policy.train()
    denom = float(max(eval_episodes, 1))
    return {
        "eval_avg_reward": total_reward / denom,
        "eval_avg_success_rate": total_success / denom,
        "eval_avg_episode_length": total_ep_len / denom,
        "eval_avg_completed_count": total_completed / denom,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPopLoc warm-start + MAPPO fine-tuning.")
    parser.add_argument("--config", type=str, default=os.path.join(_MADRL_DIR, "config", "config_small_mpoploc_warmstart.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--bc-epochs", type=int, default=None)
    parser.add_argument("--bc-batch-size", type=int, default=None)
    parser.add_argument("--bc-lr", type=float, default=None)
    parser.add_argument("--bc-max-episodes", type=int, default=None)
    parser.add_argument("--teacher", choices=["mpoploc", "greedy"], default=None)
    parser.add_argument("--skip-bc", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    yaml_cfg = _load_yaml(args.config)
    logging_cfg = yaml_cfg.get("logging", {}) if isinstance(yaml_cfg.get("logging"), dict) else {}
    training_cfg = yaml_cfg.get("training", {}) if isinstance(yaml_cfg.get("training"), dict) else {}
    dataset_cfg = yaml_cfg.get("dataset", {}) if isinstance(yaml_cfg.get("dataset"), dict) else {}
    scene_cfg = yaml_cfg.get("scene", {}) if isinstance(yaml_cfg.get("scene"), dict) else {}
    warm_cfg = yaml_cfg.get("warmstart", {}) if isinstance(yaml_cfg.get("warmstart"), dict) else {}

    output_root = _resolve_output_root(logging_cfg, args.config)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"madrl2stages_mpoploc_warmstart_seed{args.seed}_{stamp}"
    experiment_root = os.path.join(output_root, experiment_name)
    logger = TrainingLogger(log_dir=experiment_root, experiment_name="log")
    logger.experiment_name = experiment_name
    ckpt_dir = os.path.join(experiment_root, "checkpoint")
    log_dir = logger.log_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    dataset_base_seed = int(dataset_cfg.get("base_seed", args.seed))
    dataset_num_episodes = int(dataset_cfg.get("num_episodes", 100))
    dataset_data_dir = str(dataset_cfg.get("data_dir", "./data"))
    eval_cfg = dataset_cfg.get("eval", {}) if isinstance(dataset_cfg.get("eval"), dict) else {}
    eval_base_seed = int(eval_cfg.get("base_seed", dataset_base_seed + 100000))
    eval_num_episodes = int(eval_cfg.get("num_episodes", 100))
    eval_num_eval_episodes = int(eval_cfg.get("num_eval_episodes", eval_num_episodes))
    eval_data_dir = str(eval_cfg.get("data_dir", "./eval_data"))
    eval_interval_steps = int(training_cfg.get("eval_interval_steps", 0) or 0)
    shuffle_requests_on_reset = bool(training_cfg.get("shuffle_requests_on_reset", True))

    full_cfg_to_log = dict(yaml_cfg)
    full_cfg_to_log.update(
        {
            "runtime": {
                "seed": args.seed,
                "device_override": args.device,
                "experiment_name": experiment_name,
                "bc_epochs_override": args.bc_epochs,
                "bc_batch_size_override": args.bc_batch_size,
                "bc_lr_override": args.bc_lr,
                "bc_max_episodes_override": args.bc_max_episodes,
                "skip_bc": bool(args.skip_bc),
            }
        }
    )
    logger.log_config(full_cfg_to_log)

    eval_dataset = create_or_load_dataset(
        base_seed=eval_base_seed,
        num_episodes=eval_num_episodes,
        data_dir=eval_data_dir,
        force_regenerate=bool(eval_cfg.get("force_regenerate", False)),
        num_locations=int(scene_cfg.get("num_locations", env_config.NUM_LOCATIONS)),
        area_size=float(scene_cfg.get("area_size", env_config.AREA_SIZE)),
        num_uavs=int(scene_cfg.get("num_uavs", env_config.NUM_UAVS)),
        num_requests=int(scene_cfg.get("num_requests", env_config.NUM_REQUESTS)),
    )
    eval_num_eval_episodes = max(1, min(eval_num_eval_episodes, len(eval_dataset)))

    dataset = create_or_load_dataset(
        base_seed=dataset_base_seed,
        num_episodes=dataset_num_episodes,
        data_dir=dataset_data_dir,
        force_regenerate=bool(dataset_cfg.get("force_regenerate", False)),
        num_locations=int(scene_cfg.get("num_locations", env_config.NUM_LOCATIONS)),
        area_size=float(scene_cfg.get("area_size", env_config.AREA_SIZE)),
        num_uavs=int(scene_cfg.get("num_uavs", env_config.NUM_UAVS)),
        num_requests=int(scene_cfg.get("num_requests", env_config.NUM_REQUESTS)),
    )

    env = MAPPOSFCEnv(config_yaml_path=args.config, episode_data=dataset[0], seed=args.seed)
    env.set_request_shuffle_on_reset(shuffle_requests_on_reset)
    raw_env_reset = env.reset
    dataset_idx = {"value": 0}

    def _reset_with_dataset_rotation():
        ep_data = dataset[dataset_idx["value"] % len(dataset)]
        env.set_episode_data(ep_data)
        dataset_idx["value"] += 1
        return raw_env_reset()

    env.reset = _reset_with_dataset_rotation
    first_obs = env.reset()

    trainer_cfg = load_trainer_config(args.config)
    if args.device:
        trainer_cfg.device = args.device
    device = torch.device(trainer_cfg.device)
    policy = _build_policy(first_obs, env=env, yaml_cfg=yaml_cfg, device=device)

    bc_rows: List[Dict[str, float]] = []
    teacher_rows: List[Dict[str, object]] = []
    teacher_name = str(args.teacher or warm_cfg.get("teacher", "mpoploc")).lower()
    if not args.skip_bc:
        samples, teacher_rows = collect_teacher_bc_samples(
            dataset=dataset,
            config_path=args.config,
            num_time_slots=int(scene_cfg.get("num_time_slots", env_config.NUM_TIME_SLOTS)),
            seed=args.seed,
            teacher=teacher_name,
            max_episodes=args.bc_max_episodes if args.bc_max_episodes is not None else warm_cfg.get("bc_max_episodes"),
        )
        teacher_path = os.path.join(log_dir, f"{teacher_name}_teacher_episodes.csv")
        with open(teacher_path, "w", newline="", encoding="utf-8") as f:
            if teacher_rows:
                writer = csv.DictWriter(f, fieldnames=list(teacher_rows[0].keys()))
                writer.writeheader()
                writer.writerows(teacher_rows)

        print(f"[{teacher_name}-BC] collected samples={len(samples)} episodes={len(teacher_rows)}", flush=True)
        bc_rows = pretrain_actor_bc(
            policy=policy,
            samples=samples,
            device=device,
            epochs=int(args.bc_epochs if args.bc_epochs is not None else warm_cfg.get("bc_epochs", 20)),
            batch_size=int(args.bc_batch_size if args.bc_batch_size is not None else warm_cfg.get("bc_batch_size", 256)),
            lr=float(args.bc_lr if args.bc_lr is not None else warm_cfg.get("bc_lr", 0.0003)),
            max_grad_norm=float(warm_cfg.get("bc_max_grad_norm", 0.5)),
            log_path=os.path.join(log_dir, "bc_metrics.csv"),
            teacher_name=teacher_name,
        )
        bc_ckpt_path = os.path.join(ckpt_dir, f"policy_{teacher_name}_bc.pt")
        torch.save(
            {
                "policy_state_dict": policy.state_dict(),
                "seed": args.seed,
                "config_path": os.path.abspath(args.config),
                "experiment_name": experiment_name,
                "bc_epochs": int(len(bc_rows)),
                "bc_samples": int(len(samples)),
                "teacher": teacher_name,
            },
            bc_ckpt_path,
        )
        print(f"[{teacher_name}-BC] checkpoint={bc_ckpt_path}", flush=True)

    trainer = MAPPOTrainer(env=env, policy=policy, config=trainer_cfg)
    eval_env = MAPPOSFCEnv(config_yaml_path=args.config, episode_data=eval_dataset[0], seed=eval_base_seed)
    eval_env.set_request_shuffle_on_reset(False)

    update_metrics_path = os.path.join(log_dir, "update_metrics.csv")
    episode_metrics_path = os.path.join(log_dir, "episode_metrics.csv")
    eval_metrics_path = os.path.join(log_dir, "eval_metrics.csv")
    update_fields = [
        "timestamp", "update", "env_steps", "episodes_done_in_rollout", "total_episodes_done",
        "loss", "policy_loss", "value_loss", "entropy", "kl_div",
        "value_explained_variance", "value_return_corr", "lr",
        "avg_reward", "avg_return_10ep", "avg_episode_length_10ep", "avg_success_rate_10ep",
        "current_time_slot", "completed_count", "pending_count",
    ]
    episode_fields = [
        "timestamp", "episode", "env_step", "episode_length", "episode_return",
        "avg_reward", "success_rate", "total_completed", "pending_count", "final_time_slot",
    ]
    eval_fields = [
        "timestamp", "update", "env_steps", "eval_step_target", "eval_episodes",
        "eval_avg_reward", "eval_avg_success_rate", "eval_avg_episode_length", "eval_avg_completed_count",
    ]
    update_file = open(update_metrics_path, "w", newline="", encoding="utf-8")
    episode_file = open(episode_metrics_path, "w", newline="", encoding="utf-8")
    eval_file = open(eval_metrics_path, "w", newline="", encoding="utf-8")
    update_writer = csv.DictWriter(update_file, fieldnames=update_fields)
    episode_writer = csv.DictWriter(episode_file, fieldnames=episode_fields)
    eval_writer = csv.DictWriter(eval_file, fieldnames=eval_fields)
    update_writer.writeheader()
    episode_writer.writeheader()
    eval_writer.writeheader()

    save_step = int(training_cfg.get("save_step", 0) or 0)
    next_periodic_save_step = save_step if save_step > 0 else 0
    next_eval_step = eval_interval_steps if eval_interval_steps > 0 else 0

    def _on_update(stats: Dict[str, float]) -> None:
        nonlocal next_periodic_save_step, next_eval_step
        logger.step_count = int(stats.get("env_steps", 0))
        logger.log_episode(
            episode=int(stats.get("total_episodes_done", 0)),
            metrics={
                "reward": round(float(stats.get("avg_reward", 0.0)), 6),
                "success_rate": round(float(stats.get("avg_success_rate_10ep", 0.0)), 6),
                "episode_length": round(float(stats.get("avg_episode_length_10ep", 0.0)), 4),
                "policy_loss": round(float(stats.get("policy_loss", 0.0)), 6),
                "value_loss": round(float(stats.get("value_loss", 0.0)), 6),
                "entropy": round(float(stats.get("entropy", 0.0)), 6),
            },
        )
        update_writer.writerow({"timestamp": datetime.now().isoformat(), **{k: stats.get(k, "") for k in update_fields if k != "timestamp"}})
        update_file.flush()

        if eval_interval_steps > 0:
            current_env_steps = int(stats.get("env_steps", 0))
            while next_eval_step > 0 and current_env_steps >= next_eval_step:
                eval_stats = _run_deterministic_eval(trainer, eval_env, eval_dataset, eval_num_eval_episodes, device)
                eval_row = {
                    "timestamp": datetime.now().isoformat(),
                    "update": int(stats.get("update", 0)),
                    "env_steps": current_env_steps,
                    "eval_step_target": int(next_eval_step),
                    "eval_episodes": int(eval_num_eval_episodes),
                    **eval_stats,
                }
                eval_writer.writerow(eval_row)
                eval_file.flush()
                print(
                    f"[MAPPO][EVAL] env_steps={current_env_steps} target={next_eval_step} "
                    f"episodes={eval_num_eval_episodes} succ={eval_stats['eval_avg_success_rate']:.4f} "
                    f"ret={eval_stats['eval_avg_reward']:.4f}",
                    flush=True,
                )
                next_eval_step += eval_interval_steps

        if save_step > 0:
            current_env_steps = int(stats.get("env_steps", 0))
            while next_periodic_save_step > 0 and current_env_steps >= next_periodic_save_step:
                periodic_path = os.path.join(ckpt_dir, f"policy_step{next_periodic_save_step}.pt")
                torch.save(
                    {
                        "policy_state_dict": trainer.policy.state_dict(),
                        "trainer_cfg": trainer_cfg.__dict__,
                        "seed": args.seed,
                        "config_path": os.path.abspath(args.config),
                        "experiment_name": experiment_name,
                        "env_steps": current_env_steps,
                        "update": int(stats.get("update", 0)),
                    },
                    periodic_path,
                )
                next_periodic_save_step += save_step

    def _on_episode(ep: Dict[str, float]) -> None:
        episode_writer.writerow({"timestamp": datetime.now().isoformat(), **{k: ep.get(k, "") for k in episode_fields if k != "timestamp"}})
        episode_file.flush()

    max_step_from_cfg = training_cfg.get("max_step", None)
    train_total_timesteps = args.total_timesteps if args.total_timesteps is not None else (
        int(max_step_from_cfg) if max_step_from_cfg is not None else None
    )
    logs = trainer.train(
        total_timesteps=train_total_timesteps,
        rollout_steps=args.rollout_steps,
        log_interval_updates=args.log_interval,
        update_callback=_on_update,
        episode_callback=_on_episode,
    )

    ckpt_path = os.path.join(ckpt_dir, "policy_final.pt")
    torch.save(
        {
            "policy_state_dict": trainer.policy.state_dict(),
            "trainer_cfg": trainer_cfg.__dict__,
            "seed": args.seed,
            "config_path": os.path.abspath(args.config),
            "experiment_name": experiment_name,
        },
        ckpt_path,
    )
    logger.log_checkpoint(
        episode=int(trainer.total_episodes_done),
        checkpoint_path=ckpt_path,
        metrics={
            "avg_success_rate_10ep": float(logs[-1]["avg_success_rate_10ep"]) if logs else 0.0,
            "avg_return_10ep": float(logs[-1]["avg_return_10ep"]) if logs else 0.0,
            "env_steps": int(trainer.total_env_steps),
        },
    )

    with open(os.path.join(log_dir, "train_update_log.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(logs, f, allow_unicode=True, sort_keys=False)
    summary = {
        "experiment_name": experiment_name,
        "total_episodes": int(trainer.total_episodes_done),
        "total_steps": int(trainer.total_env_steps),
        "bc_epochs": int(len(bc_rows)),
        "bc_final_loss": float(bc_rows[-1]["loss"]) if bc_rows else None,
        "bc_final_accuracy": float(bc_rows[-1]["accuracy"]) if bc_rows else None,
        "teacher_avg_success_rate": float(np.mean([r["teacher_success_rate"] for r in teacher_rows])) if teacher_rows else None,
        "best_success_rate": float(max((x.get("avg_success_rate_10ep", 0.0) for x in logs), default=0.0)),
    }
    with open(os.path.join(log_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    update_file.close()
    episode_file.close()
    eval_file.close()
    logger.close()

    print(f"[{teacher_name}-WARM] training finished. checkpoint={ckpt_path}", flush=True)
    print(f"[{teacher_name}-WARM] summary={os.path.join(log_dir, 'summary.json')}", flush=True)


if __name__ == "__main__":
    main()
