# -*- coding: utf-8 -*-
"""MAPPO 训练入口脚本。"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import yaml

# 兼容 `python MAPPO/train.py` 直接启动
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as env_config
from DRL.training.logger import TrainingLogger
from DRL.training.dataset import create_or_load_dataset
from envs.env import MAPPOSFCEnv
from models import MAPPOPolicy, obs_to_tensors
from training import MAPPOTrainer, load_trainer_config


def _load_yaml(path: str) -> Dict[str, object]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_output_dir(base_dir: str, config_path: str) -> str:
    if os.path.isabs(base_dir):
        return base_dir

    config_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else os.getcwd()
    return os.path.abspath(os.path.join(config_dir, base_dir))


def _resolve_output_root(logging_cfg: Dict[str, object], config_path: str) -> str:
    """解析运行输出根目录，最终结构为 output/<experiment_name>/(log|checkpoint)。"""
    output_dir_cfg = logging_cfg.get("output_dir")
    if isinstance(output_dir_cfg, str) and output_dir_cfg.strip():
        return _resolve_output_dir(output_dir_cfg, config_path)

    log_dir_cfg = str(logging_cfg.get("log_dir", os.path.join("MAPPO", "output", "logs")))
    resolved_log_dir = _resolve_output_dir(log_dir_cfg, config_path)
    if os.path.basename(resolved_log_dir) in {"logs", "log"}:
        return os.path.dirname(resolved_log_dir)
    return resolved_log_dir


def _infer_dims(obs: Dict[str, object], device: torch.device) -> Tuple[int, int, int]:
    obs_t = obs_to_tensors(obs, device=device)
    self_dim = int(obs_t["agent_self"].shape[-1])
    task_dim = int(obs_t["task_matrix"].shape[-1])
    context_dim = int(obs_t["context"].shape[-1])
    return self_dim, task_dim, context_dim


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MAPPO for UAV-SFC scheduling.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("MADRL+RULE", "config_small.yaml"),
        help="训练配置 YAML 路径。",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--device", type=str, default=None, help="覆盖训练设备，如 cpu/cuda。")
    parser.add_argument("--total-timesteps", type=int, default=None, help="覆盖训练总环境步数。")
    parser.add_argument("--rollout-steps", type=int, default=None, help="覆盖每次 rollout 步数。")
    parser.add_argument("--log-interval", type=int, default=10, help="打印日志间隔（按 update 计）。")
    parser.add_argument("--step-log", action="store_true", help="实时输出每个环境 step 日志。")
    parser.add_argument("--save-name", type=str, default=None, help="checkpoint 文件名（可选）。")
    parser.add_argument("--experiment-name", type=str, default=None, help="实验名（默认自动生成）。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    yaml_cfg = _load_yaml(args.config)
    logging_cfg = yaml_cfg.get("logging", {}) if isinstance(yaml_cfg.get("logging"), dict) else {}
    training_cfg = yaml_cfg.get("training", {}) if isinstance(yaml_cfg.get("training"), dict) else {}
    dataset_cfg = yaml_cfg.get("dataset", {}) if isinstance(yaml_cfg.get("dataset"), dict) else {}
    scene_cfg = yaml_cfg.get("scene", {}) if isinstance(yaml_cfg.get("scene"), dict) else {}
    output_root = _resolve_output_root(logging_cfg, args.config)

    # 训练集配置
    dataset_base_seed = int(dataset_cfg.get("base_seed", args.seed))
    dataset_num_episodes = int(dataset_cfg.get("num_episodes", 100))
    dataset_data_dir = str(dataset_cfg.get("data_dir", "./data"))

    # 评估集配置（按配置生成到 eval_data，存在则直接加载）
    eval_cfg = dataset_cfg.get("eval", {}) if isinstance(dataset_cfg.get("eval"), dict) else {}
    eval_base_seed = int(eval_cfg.get("base_seed", dataset_base_seed + 100000))
    eval_num_episodes = int(eval_cfg.get("num_episodes", 100))
    eval_num_eval_episodes = int(eval_cfg.get("num_eval_episodes", eval_num_episodes))
    eval_data_dir = str(eval_cfg.get("data_dir", "./eval_data"))
    eval_force_regenerate = bool(eval_cfg.get("force_regenerate", False))
    eval_interval_steps = int(training_cfg.get("eval_interval_steps", 0) or 0)
    shuffle_requests_on_reset = bool(training_cfg.get("shuffle_requests_on_reset", True))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"mappo_seed{args.seed}_{stamp}"
    experiment_root = os.path.join(output_root, experiment_name)
    logger = TrainingLogger(log_dir=experiment_root, experiment_name="log")
    logger.experiment_name = experiment_name

    runtime_cfg = {
        "runtime": {
            "seed": args.seed,
            "device_override": args.device,
            "total_timesteps_override": args.total_timesteps,
            "rollout_steps_override": args.rollout_steps,
            "max_step_cfg": training_cfg.get("max_step"),
            "save_step_cfg": training_cfg.get("save_step"),
            "log_interval_updates": args.log_interval,
            "step_log": bool(args.step_log),
            "dataset_base_seed": dataset_base_seed,
            "dataset_num_episodes": dataset_num_episodes,
            "dataset_data_dir": dataset_data_dir,
            "eval_base_seed": eval_base_seed,
            "eval_num_episodes": eval_num_episodes,
            "eval_num_eval_episodes": eval_num_eval_episodes,
            "eval_data_dir": eval_data_dir,
            "eval_force_regenerate": eval_force_regenerate,
            "eval_interval_steps": eval_interval_steps,
            "shuffle_requests_on_reset": shuffle_requests_on_reset,
            "record_step_metrics": bool(logging_cfg.get("record_step_metrics", False)),
            },
        "uav_resources": {
            "battery_capacity": env_config.UAV_BATTERY_CAPACITY,
            "computation_capacity": env_config.UAV_COMPUTATION_CAPACITY,
            "max_speed": env_config.UAV_MAX_SPEED,
            "transmit_power": env_config.UAV_TRANSMIT_POWER,
            "travel_time": env_config.UAV_TRAVEL_TIME,
            "blade_profile_power": env_config.UAV_BLADE_PROFILE_POWER,
            "induced_power": env_config.UAV_INDUCED_POWER,
        },
    }
    full_cfg_to_log = dict(yaml_cfg)
    full_cfg_to_log.update(runtime_cfg)
    logger.log_config(full_cfg_to_log)

    # 每次实验前准备评估集（存在则直接加载）
    eval_dataset = create_or_load_dataset(
        base_seed=eval_base_seed,
        num_episodes=eval_num_episodes,
        data_dir=eval_data_dir,
        force_regenerate=eval_force_regenerate,
        num_locations=int(scene_cfg.get("num_locations", env_config.NUM_LOCATIONS)),
        area_size=float(scene_cfg.get("area_size", env_config.AREA_SIZE)),
        num_uavs=int(scene_cfg.get("num_uavs", env_config.NUM_UAVS)),
        num_requests=int(scene_cfg.get("num_requests", env_config.NUM_REQUESTS)),
    )
    print(
        f"[MAPPO] eval dataset ready: episodes={len(eval_dataset)} "
        f"seed={eval_base_seed} dir={eval_data_dir}",
        flush=True,
    )
    eval_num_eval_episodes = max(1, min(eval_num_eval_episodes, len(eval_dataset)))

    # 与 DRL 共享同一套固定种子训练集（同参数同路径 -> 同数据文件）

    dataset = create_or_load_dataset(
        base_seed=dataset_base_seed,
        num_episodes=dataset_num_episodes,
        data_dir=dataset_data_dir,
        num_locations=int(scene_cfg.get("num_locations", env_config.NUM_LOCATIONS)),
        area_size=float(scene_cfg.get("area_size", env_config.AREA_SIZE)),
        num_uavs=int(scene_cfg.get("num_uavs", env_config.NUM_UAVS)),
        num_requests=int(scene_cfg.get("num_requests", env_config.NUM_REQUESTS)),
    )

    env = MAPPOSFCEnv(config_yaml_path=args.config, episode_data=dataset[0], seed=args.seed)
    env.set_request_shuffle_on_reset(shuffle_requests_on_reset)

    # 每次 episode reset 前切换到下一条数据，确保轮换同一训练集
    _raw_env_reset = env.reset
    dataset_idx = {"value": 0}

    def _reset_with_dataset_rotation():
        ep_data = dataset[dataset_idx["value"] % len(dataset)]
        env.set_episode_data(ep_data)
        dataset_idx["value"] += 1
        return _raw_env_reset()

    env.reset = _reset_with_dataset_rotation
    first_obs = env.reset()

    trainer_cfg = load_trainer_config(args.config)
    if args.device:
        trainer_cfg.device = args.device
    if args.step_log:
        trainer_cfg.log_each_step = True

    device = torch.device(trainer_cfg.device)
    self_dim, task_dim, context_dim = _infer_dims(first_obs, device)

    network_cfg = yaml_cfg.get("network", {}) if isinstance(yaml_cfg.get("network"), dict) else {}
    hidden_dim = int(network_cfg.get("hidden_dim", 256))
    num_heads = int(network_cfg.get("num_heads", 4))
    num_blocks = int(network_cfg.get("num_encoder_layers", 1))
    dropout = float(network_cfg.get("dropout", 0.1))
    actor_type = str(network_cfg.get("actor_type", "attn")).lower()
    actor_mlp_hidden_dim_cfg = network_cfg.get("actor_mlp_hidden_dim", None)
    actor_mlp_hidden_dim = None if actor_mlp_hidden_dim_cfg is None else int(actor_mlp_hidden_dim_cfg)
    actor_mlp_num_layers = int(network_cfg.get("actor_mlp_num_layers", 2))

    policy = MAPPOPolicy(
        self_dim=self_dim,
        task_dim=task_dim,
        context_dim=context_dim,
        action_dim=env.cfg.action_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        num_actor_agent_blocks=num_blocks,
        num_critic_uav_blocks=num_blocks,
        num_critic_task_blocks=num_blocks,
        actor_type=actor_type,
        actor_mlp_hidden_dim=actor_mlp_hidden_dim,
        actor_mlp_num_layers=actor_mlp_num_layers,
    )

    trainer = MAPPOTrainer(env=env, policy=policy, config=trainer_cfg)
    eval_env = MAPPOSFCEnv(config_yaml_path=args.config, episode_data=eval_dataset[0], seed=eval_base_seed)
    eval_env.set_request_shuffle_on_reset(False)
    ckpt_dir = os.path.join(experiment_root, "checkpoint")
    log_dir = logger.log_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    save_step = int(training_cfg.get("save_step", 0) or 0)
    next_periodic_save_step = save_step if save_step > 0 else 0
    next_eval_step = eval_interval_steps if eval_interval_steps > 0 else 0

    record_step_metrics = bool(logging_cfg.get("record_step_metrics", False))
    step_metrics_path = os.path.join(logger.log_dir, "step_metrics.csv")
    update_metrics_path = os.path.join(logger.log_dir, "update_metrics.csv")
    episode_metrics_path = os.path.join(logger.log_dir, "episode_metrics.csv")
    eval_metrics_path = os.path.join(logger.log_dir, "eval_metrics.csv")

    step_fields = [
        "timestamp", "env_step", "episode", "episode_step", "rollout_step",
        "agent_id", "action", "reward", "invalid_count", "action_mask_density",
        "done", "current_time_slot", "current_round", "turn_in_round",
        "pending_count", "success_rate",
    ]
    update_fields = [
        "timestamp", "update", "env_steps", "episodes_done_in_rollout", "total_episodes_done",
        "loss", "policy_loss", "value_loss", "entropy", "kl_div", "lr",
        "avg_reward", "avg_return_10ep", "avg_episode_length_10ep", "avg_success_rate_10ep",
        "current_time_slot", "completed_count", "pending_count",
    ]
    episode_fields = [
        "timestamp", "episode", "env_step", "episode_length", "episode_return",
        "avg_reward", "success_rate", "total_completed", "pending_count", "final_time_slot",
    ]
    eval_fields = [
        "timestamp", "update", "env_steps", "eval_step_target", "eval_episodes",
        "eval_avg_reward", "eval_avg_success_rate", "eval_avg_episode_length",
        "eval_avg_completed_count",
    ]

    step_file = open(step_metrics_path, "w", newline="", encoding="utf-8") if record_step_metrics else None
    update_file = open(update_metrics_path, "w", newline="", encoding="utf-8")
    episode_file = open(episode_metrics_path, "w", newline="", encoding="utf-8")
    eval_file = open(eval_metrics_path, "w", newline="", encoding="utf-8")
    step_writer = csv.DictWriter(step_file, fieldnames=step_fields) if record_step_metrics else None
    update_writer = csv.DictWriter(update_file, fieldnames=update_fields)
    episode_writer = csv.DictWriter(episode_file, fieldnames=episode_fields)
    eval_writer = csv.DictWriter(eval_file, fieldnames=eval_fields)
    if record_step_metrics:
        step_writer.writeheader()
    update_writer.writeheader()
    episode_writer.writeheader()
    eval_writer.writeheader()
    if record_step_metrics:
        step_file.flush()
    update_file.flush()
    episode_file.flush()
    eval_file.flush()

    def _run_deterministic_eval() -> Dict[str, float]:
        was_training = trainer.policy.training
        trainer.policy.eval()

        total_reward = 0.0
        total_success = 0.0
        total_ep_len = 0.0
        total_completed = 0.0

        with torch.no_grad():
            for i in range(eval_num_eval_episodes):
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
                    obs, reward, done, info = eval_env.step(
                        agent_id,
                        action,
                        deploy_score_table=deploy_score_table,
                    )
                    ep_reward += float(reward)
                    ep_len += 1
                    last_info = info

                total_reward += ep_reward
                total_ep_len += float(ep_len)
                total_success += float(last_info.get("success_rate", 0.0))
                total_completed += float(last_info.get("completed_count", 0.0))

        if was_training:
            trainer.policy.train()

        denom = float(max(eval_num_eval_episodes, 1))
        return {
            "eval_avg_reward": total_reward / denom,
            "eval_avg_success_rate": total_success / denom,
            "eval_avg_episode_length": total_ep_len / denom,
            "eval_avg_completed_count": total_completed / denom,
        }

    def _on_update(stats: Dict[str, float]) -> None:
        nonlocal next_periodic_save_step, next_eval_step
        # 对齐 DRL/training/logs 下 training_log.csv 字段风格
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
                "current_time_slot": int(stats.get("current_time_slot", -1))
                if not np.isnan(stats.get("current_time_slot", np.nan)) else "",
                "completed_count": int(stats.get("completed_count", -1))
                if not np.isnan(stats.get("completed_count", np.nan)) else "",
                "pending_count": int(stats.get("pending_count", -1))
                if not np.isnan(stats.get("pending_count", np.nan)) else "",
            },
        )
        update_writer.writerow({"timestamp": datetime.now().isoformat(), **{k: stats.get(k, "") for k in update_fields if k != "timestamp"}})
        update_file.flush()

        # 周期性 deterministic 评估（按 env_step）
        if eval_interval_steps > 0:
            current_env_steps = int(stats.get("env_steps", 0))
            while next_eval_step > 0 and current_env_steps >= next_eval_step:
                eval_stats = _run_deterministic_eval()
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
                    f"[MAPPO][EVAL] env_steps={current_env_steps} "
                    f"target={next_eval_step} "
                    f"episodes={eval_num_eval_episodes} "
                    f"succ={eval_stats['eval_avg_success_rate']:.4f} "
                    f"ret={eval_stats['eval_avg_reward']:.4f}",
                    flush=True,
                )
                next_eval_step += eval_interval_steps

        # 周期保存 checkpoint（按 env_step）
        if save_step > 0:
            current_env_steps = int(stats.get("env_steps", 0))
            while next_periodic_save_step > 0 and current_env_steps >= next_periodic_save_step:
                periodic_name = f"policy_step{next_periodic_save_step}.pt"
                periodic_path = os.path.join(ckpt_dir, periodic_name)
                torch.save(
                    {
                        "policy_state_dict": trainer.policy.state_dict(),
                        "trainer_cfg": trainer_cfg.__dict__,
                        "seed": args.seed,
                        "config_path": os.path.abspath(args.config),
                        "experiment_name": experiment_name,
                        "env_steps": int(stats.get("env_steps", 0)),
                        "update": int(stats.get("update", 0)),
                    },
                    periodic_path,
                )
                logger.log_checkpoint(
                    episode=int(trainer.total_episodes_done),
                    checkpoint_path=periodic_path,
                    metrics={
                        "type": "periodic",
                        "save_step": next_periodic_save_step,
                        "env_steps": int(stats.get("env_steps", 0)),
                        "update": int(stats.get("update", 0)),
                        "avg_success_rate_10ep": float(stats.get("avg_success_rate_10ep", 0.0)),
                        "avg_return_10ep": float(stats.get("avg_return_10ep", 0.0)),
                    },
                )
                next_periodic_save_step += save_step

    def _on_step(step: Dict[str, float]) -> None:
        if not record_step_metrics:
            return
        step_writer.writerow({"timestamp": datetime.now().isoformat(), **{k: step.get(k, "") for k in step_fields if k != "timestamp"}})
        step_file.flush()

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
        step_callback=_on_step,
        episode_callback=_on_episode,
    )

    ckpt_name = args.save_name or "policy_final.pt"
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
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

    log_path = os.path.join(log_dir, "train_update_log.yaml")
    with open(log_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(logs, f, allow_unicode=True, sort_keys=False)
    if record_step_metrics:
        step_file.close()
    update_file.close()
    episode_file.close()
    eval_file.close()
    logger.close()

    print(f"[MAPPO] training finished. checkpoint={ckpt_path}")
    print(f"[MAPPO] training_log={os.path.join(log_dir, 'training_log.csv')}")
    if record_step_metrics:
        print(f"[MAPPO] step_metrics={step_metrics_path}")
    print(f"[MAPPO] update_metrics={update_metrics_path}")
    print(f"[MAPPO] episode_metrics={episode_metrics_path}")
    print(f"[MAPPO] eval_metrics={eval_metrics_path}")
    print(f"[MAPPO] summary={os.path.join(log_dir, 'summary.json')}")
    print(f"[MAPPO] logs saved to {log_path}")


if __name__ == "__main__":
    main()
