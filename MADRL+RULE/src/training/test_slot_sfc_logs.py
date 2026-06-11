# -*- coding: utf-8 -*-
"""Run MAPPO on test_data and export step actions plus per-slot completed SFC sets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# Allow `python MADRL+RULE/test_slot_sfc_logs.py` from repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as env_config
from DRL.training.dataset import EpisodeData, create_or_load_dataset
from envs.env import MAPPOSFCEnv
from models import MAPPOPolicy, obs_to_tensors


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


def _resolve_path(path: str, config_path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(config_path)), path))


def _infer_dims(obs: Dict[str, object], device: torch.device) -> Tuple[int, int, int]:
    obs_t = obs_to_tensors(obs, device=device)
    return (
        int(obs_t["agent_self"].shape[-1]),
        int(obs_t["task_matrix"].shape[-1]),
        int(obs_t["context"].shape[-1]),
    )


def _infer_actor_type_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    keys = list(state_dict.keys())
    if any(k.startswith("actor.agent_blocks.") for k in keys):
        return "attn"
    return "mlp" if any(k.startswith("actor.policy_head.") for k in keys) else "attn"


def _looks_like_legacy_vnf_checkpoint(state_dict: Dict[str, torch.Tensor]) -> bool:
    keys = set(state_dict.keys())
    legacy_keys = {
        "actor.query_proj.weight",
        "actor.key_proj.weight",
        "actor.end_head.layers.0.weight",
    }
    if keys.intersection(legacy_keys):
        return True

    task_weight = state_dict.get("actor.task_encoder.layers.0.weight")
    return bool(task_weight is not None and task_weight.ndim == 2 and int(task_weight.shape[1]) == 14)


def _raise_incompatible_checkpoint(checkpoint_path: str) -> None:
    raise RuntimeError(
        "\n".join(
            [
                "Checkpoint 与当前 MADRL+RULE location-rule policy 不兼容。",
                f"checkpoint: {os.path.abspath(checkpoint_path)}",
                "检测到该 checkpoint 更像旧版 VNF/END 动作空间模型：",
                "  - actor.query_proj / actor.key_proj / actor.end_head",
                "  - actor.task_encoder 输入维度为 14",
                "而当前测试脚本使用的是 location 动作空间：",
                "  - action 表示 location_id = action + 1",
                "  - task_matrix 输入维度为 11",
                "  - actor 会把每个 VNF score 加和到 location logits",
                "处理方式：",
                "  1. 用当前 MADRL+RULE 代码重新训练一个 checkpoint 后再运行该脚本；或",
                "  2. 如果你要测试旧版 VNF/END checkpoint，请使用对应的 MAPPO/test_checkpoint.py 和旧环境。",
            ]
        )
    )


def _build_policy(
    obs: Dict[str, object],
    device: torch.device,
    yaml_cfg: Dict[str, object],
    actor_type_override: Optional[str] = None,
) -> MAPPOPolicy:
    self_dim, task_dim, context_dim = _infer_dims(obs, device)
    network_cfg = yaml_cfg.get("network", {}) if isinstance(yaml_cfg.get("network"), dict) else {}

    actor_type = str(actor_type_override or network_cfg.get("actor_type", "attn")).lower()
    hidden_dim = int(network_cfg.get("hidden_dim", 256))
    num_heads = int(network_cfg.get("num_heads", 4))
    num_blocks = int(network_cfg.get("num_encoder_layers", 1))
    dropout = float(network_cfg.get("dropout", 0.1))
    actor_mlp_hidden_dim_cfg = network_cfg.get("actor_mlp_hidden_dim", None)
    actor_mlp_hidden_dim = None if actor_mlp_hidden_dim_cfg is None else int(actor_mlp_hidden_dim_cfg)
    actor_mlp_num_layers = int(network_cfg.get("actor_mlp_num_layers", 2))

    action_mask = np.asarray(obs["action_mask"], dtype=np.int32)
    action_dim = int(action_mask.shape[-1])

    return MAPPOPolicy(
        self_dim=self_dim,
        task_dim=task_dim,
        context_dim=context_dim,
        action_dim=action_dim,
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


def _load_policy(
    checkpoint_path: str,
    obs: Dict[str, object],
    device: torch.device,
    yaml_cfg: Dict[str, object],
) -> MAPPOPolicy:
    checkpoint_path = os.path.abspath(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["policy_state_dict"] if isinstance(ckpt, dict) and "policy_state_dict" in ckpt else ckpt
    if _looks_like_legacy_vnf_checkpoint(state_dict):
        _raise_incompatible_checkpoint(checkpoint_path)

    policy = _build_policy(obs, device=device, yaml_cfg=yaml_cfg)
    try:
        policy.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        actor_type = _infer_actor_type_from_state_dict(state_dict)
        policy = _build_policy(obs, device=device, yaml_cfg=yaml_cfg, actor_type_override=actor_type)
        try:
            policy.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint 加载失败。请确认 checkpoint 是由当前 MADRL+RULE location-rule 代码训练得到的，"
                "并且 config 的 scene/network 参数与训练时一致。"
            ) from exc

    policy = policy.to(device)
    policy.eval()
    return policy


def _request_summary(env: MAPPOSFCEnv, request_id: int) -> Dict[str, object]:
    for idx, req in enumerate(env.requests):
        if int(req.id) != int(request_id):
            continue
        return {
            "request_idx": int(idx),
            "request_id": int(req.id),
            "vnfs": [
                {
                    "vnf_idx": int(vnf_idx),
                    "vnf_id": int(vnf.id),
                    "location_id": int(vnf.location_id),
                    "workload": float(vnf.workload),
                    "cpu_freq": float(vnf.cpu_freq),
                }
                for vnf_idx, vnf in enumerate(req.vnfs)
            ],
            "communication_demand": float(sum(req.communication_demands.values())) if req.communication_demands else 0.0,
        }
    return {"request_idx": None, "request_id": int(request_id), "vnfs": [], "communication_demand": 0.0}


def _selected_locations(env: MAPPOSFCEnv) -> Dict[int, Optional[int]]:
    return {
        int(agent_id): (
            None
            if env.agent_slot_location.get(agent_id) is None
            else int(env.agent_slot_location[agent_id])
        )
        for agent_id in env.agent_ids
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test MAPPO on test_data and log step actions / slot SFC completions.")
    parser.add_argument("--config", type=str, default=os.path.join("MADRL+RULE", "config_small.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda；默认读取配置，cuda 不可用时自动回退 cpu。")
    parser.add_argument("--test-data-dir", type=str, default=None, help="默认读取 dataset.test.data_dir，否则 ./test_data。")
    parser.add_argument("--num-test-episodes", type=int, default=None)
    parser.add_argument("--test-base-seed", type=int, default=None)
    parser.add_argument("--force-regenerate-test", action="store_true")
    parser.add_argument("--stochastic", action="store_true", help="默认 deterministic；加上后按策略分布采样。")
    parser.add_argument("--output-dir", type=str, default=None, help="默认写到配置 logging.output_dir 下。")
    parser.add_argument("--experiment-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    yaml_cfg = _load_yaml(args.config)
    scene_cfg = yaml_cfg.get("scene", {}) if isinstance(yaml_cfg.get("scene"), dict) else {}
    dataset_cfg = yaml_cfg.get("dataset", {}) if isinstance(yaml_cfg.get("dataset"), dict) else {}
    test_cfg = dataset_cfg.get("test", {}) if isinstance(dataset_cfg.get("test"), dict) else {}
    eval_cfg = dataset_cfg.get("eval", {}) if isinstance(dataset_cfg.get("eval"), dict) else {}
    logging_cfg = yaml_cfg.get("logging", {}) if isinstance(yaml_cfg.get("logging"), dict) else {}
    training_cfg = yaml_cfg.get("training", {}) if isinstance(yaml_cfg.get("training"), dict) else {}

    device_name = args.device or str(training_cfg.get("device", "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    dataset_base_seed = int(dataset_cfg.get("base_seed", args.seed))
    dataset_num_episodes = int(dataset_cfg.get("num_episodes", 100))
    test_base_seed = int(
        args.test_base_seed
        if args.test_base_seed is not None
        else test_cfg.get("base_seed", eval_cfg.get("base_seed", dataset_base_seed + 200000))
    )
    test_num_episodes = int(
        args.num_test_episodes
        if args.num_test_episodes is not None
        else test_cfg.get("num_episodes", eval_cfg.get("num_episodes", dataset_num_episodes))
    )
    test_data_dir = str(args.test_data_dir or test_cfg.get("data_dir", "./test_data"))
    force_regenerate = bool(args.force_regenerate_test or test_cfg.get("force_regenerate", False))

    test_dataset = create_or_load_dataset(
        base_seed=test_base_seed,
        num_episodes=test_num_episodes,
        data_dir=test_data_dir,
        force_regenerate=force_regenerate,
        num_locations=int(scene_cfg.get("num_locations", env_config.NUM_LOCATIONS)),
        area_size=float(scene_cfg.get("area_size", env_config.AREA_SIZE)),
        num_uavs=int(scene_cfg.get("num_uavs", env_config.NUM_UAVS)),
        num_requests=int(scene_cfg.get("num_requests", env_config.NUM_REQUESTS)),
    )

    env = MAPPOSFCEnv(config_yaml_path=args.config, episode_data=test_dataset[0], seed=args.seed)
    first_obs = env.reset()
    policy = _load_policy(args.checkpoint, first_obs, device=device, yaml_cfg=yaml_cfg)

    output_root_cfg = str(args.output_dir or logging_cfg.get("output_dir", "./output"))
    output_root = _resolve_path(output_root_cfg, args.config)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"test_slot_sfc_seed{args.seed}_{stamp}"
    log_dir = os.path.join(output_root, experiment_name, "log")
    os.makedirs(log_dir, exist_ok=True)

    step_log_path = os.path.join(log_dir, "step_actions.csv")
    slot_log_path = os.path.join(log_dir, "slot_sfc_completions.csv")
    summary_path = os.path.join(log_dir, "summary.json")

    step_fields = [
        "episode_index",
        "dataset_episode_id",
        "dataset_seed",
        "step_index",
        "slot_id",
        "round_id",
        "turn_id",
        "agent_id",
        "action",
        "location_id",
        "valid_action",
        "valid_location_count",
        "reward",
        "slot_ended",
        "done",
        "slot_deployed_vnf",
        "slot_completed_count",
        "completed_count",
        "pending_count",
        "success_rate",
    ]
    slot_fields = [
        "episode_index",
        "dataset_episode_id",
        "dataset_seed",
        "slot_id",
        "steps_in_slot",
        "selected_locations_json",
        "completed_sfc_ids_json",
        "completed_sfc_count",
        "completed_sfc_details_json",
        "total_completed_count",
        "pending_count",
        "success_rate",
    ]

    deterministic = not args.stochastic
    episode_returns: List[float] = []
    episode_success_rates: List[float] = []
    episode_completed_counts: List[int] = []

    with open(step_log_path, "w", newline="", encoding="utf-8") as step_file, open(
        slot_log_path, "w", newline="", encoding="utf-8"
    ) as slot_file:
        step_writer = csv.DictWriter(step_file, fieldnames=step_fields)
        slot_writer = csv.DictWriter(slot_file, fieldnames=slot_fields)
        step_writer.writeheader()
        slot_writer.writeheader()

        for ep_idx in range(len(test_dataset)):
            ep_data: EpisodeData = test_dataset[ep_idx]
            env.set_episode_data(ep_data)
            obs = env.reset()

            done = False
            step_idx = 0
            ep_return = 0.0
            slot_step_count: Dict[int, int] = {}
            last_info: Dict[str, object] = {}

            while not done:
                slot_before = int(env.current_time_slot)
                round_before = int(env.current_round)
                turn_before = int(env.turn_in_round)
                agent_id = int(obs["current_agent_id"])
                completed_before = set(int(x) for x in env.completed_request_ids)
                selected_locations_before_step = _selected_locations(env)

                obs_t = obs_to_tensors(obs, device=device)
                with torch.no_grad():
                    act_out = policy.act(obs_t, deterministic=deterministic)
                    deploy_score_table = policy.score_vnfs_for_deployment(obs_t)[0].detach().cpu().numpy()
                action = int(act_out["action"][0].item())
                location_id = int(env.decode_action(action))
                selected_locations_for_slot = dict(selected_locations_before_step)
                selected_locations_for_slot[agent_id] = location_id
                valid_action = bool(np.asarray(obs["action_mask"], dtype=np.int32)[action] == 1)
                valid_location_count = int(np.asarray(obs["action_mask"], dtype=np.int32).sum())

                obs, reward, done, info = env.step(
                    agent_id,
                    action,
                    deploy_score_table=deploy_score_table,
                )
                completed_after = set(int(x) for x in env.completed_request_ids)
                completed_this_slot = sorted(completed_after - completed_before)

                step_idx += 1
                ep_return += float(reward)
                last_info = info
                slot_step_count[slot_before] = int(slot_step_count.get(slot_before, 0) + 1)

                step_writer.writerow(
                    {
                        "episode_index": int(ep_idx),
                        "dataset_episode_id": int(ep_data.episode_id),
                        "dataset_seed": int(ep_data.seed),
                        "step_index": int(step_idx),
                        "slot_id": slot_before,
                        "round_id": round_before,
                        "turn_id": turn_before,
                        "agent_id": agent_id,
                        "action": action,
                        "location_id": location_id,
                        "valid_action": int(valid_action),
                        "valid_location_count": valid_location_count,
                        "reward": float(reward),
                        "slot_ended": int(bool(info.get("slot_ended", False))),
                        "done": int(bool(done)),
                        "slot_deployed_vnf": int(info.get("slot_deployed_vnf", 0)),
                        "slot_completed_count": int(info.get("slot_completed", 0)),
                        "completed_count": int(info.get("completed_count", 0)),
                        "pending_count": int(info.get("pending_count", 0)),
                        "success_rate": float(info.get("success_rate", 0.0)),
                    }
                )

                if bool(info.get("slot_ended", False)):
                    completed_details = [_request_summary(env, rid) for rid in completed_this_slot]
                    slot_writer.writerow(
                        {
                            "episode_index": int(ep_idx),
                            "dataset_episode_id": int(ep_data.episode_id),
                            "dataset_seed": int(ep_data.seed),
                            "slot_id": slot_before,
                            "steps_in_slot": int(slot_step_count.get(slot_before, 0)),
                            "selected_locations_json": json.dumps(selected_locations_for_slot, ensure_ascii=False),
                            "completed_sfc_ids_json": json.dumps(completed_this_slot, ensure_ascii=False),
                            "completed_sfc_count": int(len(completed_this_slot)),
                            "completed_sfc_details_json": json.dumps(completed_details, ensure_ascii=False),
                            "total_completed_count": int(info.get("completed_count", 0)),
                            "pending_count": int(info.get("pending_count", 0)),
                            "success_rate": float(info.get("success_rate", 0.0)),
                        }
                    )

            ep_success = float(last_info.get("success_rate", 0.0))
            ep_completed = int(last_info.get("completed_count", 0))
            episode_returns.append(float(ep_return))
            episode_success_rates.append(ep_success)
            episode_completed_counts.append(ep_completed)

            print(
                f"[TEST] episode={ep_idx + 1}/{len(test_dataset)} "
                f"return={ep_return:.4f} success={ep_success:.4f} completed={ep_completed}",
                flush=True,
            )

    denom = float(max(len(episode_returns), 1))
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "device": str(device),
        "deterministic": bool(deterministic),
        "test_data": {
            "data_dir": test_data_dir,
            "base_seed": int(test_base_seed),
            "num_episodes": int(len(test_dataset)),
            "force_regenerate": bool(force_regenerate),
        },
        "metrics": {
            "avg_return": float(sum(episode_returns) / denom),
            "avg_success_rate": float(sum(episode_success_rates) / denom),
            "avg_completed_count": float(sum(episode_completed_counts) / denom),
        },
        "logs": {
            "step_actions": step_log_path,
            "slot_sfc_completions": slot_log_path,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[TEST] step_actions={step_log_path}")
    print(f"[TEST] slot_sfc_completions={slot_log_path}")
    print(f"[TEST] summary={summary_path}")


if __name__ == "__main__":
    main()
