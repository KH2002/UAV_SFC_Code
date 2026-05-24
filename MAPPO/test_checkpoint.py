# -*- coding: utf-8 -*-
"""使用指定 checkpoint 在 test_data 上做 MAPPO 测试，并导出逐样本细节。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# 兼容 `python MAPPO/test_checkpoint.py` 直接启动
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as env_config
from DRL.training.dataset import EpisodeData, create_or_load_dataset
from MAPPO.envs.env import MAPPOSFCEnv
from MAPPO.models import MAPPOPolicy, obs_to_tensors
from MAPPO.training import load_trainer_config


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
    output_dir_cfg = logging_cfg.get("output_dir")
    if isinstance(output_dir_cfg, str) and output_dir_cfg.strip():
        return _resolve_output_dir(output_dir_cfg, config_path)

    log_dir_cfg = str(logging_cfg.get("log_dir", os.path.join("MAPPO", "output", "logs")))
    resolved_log_dir = _resolve_output_dir(log_dir_cfg, config_path)
    if os.path.basename(resolved_log_dir) in {"logs", "log"}:
        return os.path.dirname(resolved_log_dir)
    return resolved_log_dir


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _infer_dims(obs: Dict[str, object], device: torch.device) -> Tuple[int, int, int]:
    obs_t = obs_to_tensors(obs, device=device)
    self_dim = int(obs_t["agent_self"].shape[-1])
    task_dim = int(obs_t["task_matrix"].shape[-1])
    context_dim = int(obs_t["context"].shape[-1])
    return self_dim, task_dim, context_dim


def _infer_actor_type_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    keys = list(state_dict.keys())
    if any(k.startswith("actor.agent_blocks.") for k in keys):
        return "attn"
    if any(k.startswith("actor.policy_head.") for k in keys):
        return "mlp"
    return "attn"


def _snapshot_uavs(uavs: List[Any]) -> List[Dict[str, Any]]:
    return [
        {
            "uav_id": int(uav.id),
            "location_id": int(uav.location_id),
            "x": float(uav.location[0]),
            "y": float(uav.location[1]),
            "energy": float(uav.energy),
            "cpu_capacity": float(uav.cpu_capacity),
            "is_busy": bool(uav.is_busy),
        }
        for uav in uavs
    ]


def _decode_action_detail(env: MAPPOSFCEnv, action: int) -> Dict[str, Any]:
    if int(action) == int(env.end_token):
        return {
            "action_type": "end",
            "request_idx": None,
            "request_id": None,
            "vnf_idx": None,
            "vnf_id": None,
            "vnf_location_id": None,
        }

    req_idx, vnf_idx = env.decode_action(int(action))
    detail = {
        "action_type": "claim",
        "request_idx": int(req_idx),
        "request_id": None,
        "vnf_idx": int(vnf_idx),
        "vnf_id": None,
        "vnf_location_id": None,
    }
    if 0 <= req_idx < len(env.requests):
        req = env.requests[req_idx]
        detail["request_id"] = int(req.id)
        if 0 <= vnf_idx < len(req.vnfs):
            vnf = req.vnfs[vnf_idx]
            detail["vnf_id"] = int(vnf.id)
            detail["vnf_location_id"] = int(vnf.location_id)
    return detail


@torch.no_grad()
def _extract_first_step_attention(
    policy: MAPPOPolicy,
    obs: Dict[str, object],
    device: torch.device,
) -> Dict[str, Any]:
    """提取当前观测下 actor cross-attention 得分（仅 attention actor 可用）。"""
    obs_t = obs_to_tensors(obs, device=device)
    actor = policy.actor

    if not hasattr(actor, "cross_attn"):
        return {
            "attention_available": False,
            "reason": "actor_has_no_cross_attention",
            "current_agent_id": int(obs["current_agent_id"]),
            "num_heads": 0,
            "attention_mean_scores": [],
            "attention_per_head_scores": [],
            "task_valid_mask": [],
        }

    agent_tokens = actor._encode_agents(obs_t["agent_self"], obs_t.get("agent_avail_mask"))  # [B,N,H]
    selected_agent = actor._select_current_agent_token(agent_tokens, obs_t["current_agent_id"])  # [B,H]
    task_tokens = actor.task_encoder(obs_t["task_matrix"])  # [B,P,H]
    context_token = actor.context_encoder(obs_t["context"])  # [B,H]

    query = (selected_agent + context_token).unsqueeze(1)  # [B,1,H]
    norm_query = actor.cross_query_norm(query)
    norm_task_tokens = actor.cross_kv_norm(task_tokens)
    task_padding_mask = (obs_t["task_matrix"].abs().sum(dim=-1) <= 0)  # [B,P]

    _, attn_weights = actor.cross_attn(
        query=norm_query,
        key=norm_task_tokens,
        value=norm_task_tokens,
        key_padding_mask=task_padding_mask,
        need_weights=True,
        average_attn_weights=False,
    )
    # attn_weights: [B, num_heads, target_len(=1), source_len(P)]
    head_scores = attn_weights[0, :, 0, :].detach().cpu().numpy()
    mean_scores = head_scores.mean(axis=0)
    valid_mask = (~task_padding_mask[0]).detach().cpu().numpy().astype(int)

    return {
        "attention_available": True,
        "reason": "",
        "current_agent_id": int(obs["current_agent_id"]),
        "num_heads": int(head_scores.shape[0]),
        "attention_mean_scores": mean_scores.tolist(),
        "attention_per_head_scores": head_scores.tolist(),
        "task_valid_mask": valid_mask.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test MAPPO checkpoint on test_data dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("MAPPO", "config_small.yaml"),
        help="配置文件路径。",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="待测试 checkpoint 路径。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--device", type=str, default=None, help="覆盖设备，如 cpu/cuda。")
    parser.add_argument(
        "--num-test-episodes",
        type=int,
        default=None,
        help="覆盖测试集 episode 数；不填则从配置读取。",
    )
    parser.add_argument(
        "--test-base-seed",
        type=int,
        default=None,
        help="覆盖测试集 base_seed；不填则从配置读取。",
    )
    parser.add_argument(
        "--force-regenerate-test",
        action="store_true",
        help="强制重建 test_data 测试集。",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="使用随机采样动作（默认 deterministic 贪心动作）。",
    )
    parser.add_argument("--experiment-name", type=str, default=None, help="输出实验名。")
    return parser.parse_args()


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

    action_mask = np.asarray(obs.get("action_mask", []), dtype=np.int32)
    action_dim = int(action_mask.shape[-1]) if action_mask.size > 0 else 41

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


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    yaml_cfg = _load_yaml(args.config)
    logging_cfg = yaml_cfg.get("logging", {}) if isinstance(yaml_cfg.get("logging"), dict) else {}
    dataset_cfg = yaml_cfg.get("dataset", {}) if isinstance(yaml_cfg.get("dataset"), dict) else {}
    scene_cfg = yaml_cfg.get("scene", {}) if isinstance(yaml_cfg.get("scene"), dict) else {}

    trainer_cfg = load_trainer_config(args.config)
    if args.device:
        trainer_cfg.device = args.device
    device = torch.device(trainer_cfg.device)

    dataset_base_seed = int(dataset_cfg.get("base_seed", args.seed))
    dataset_num_episodes = int(dataset_cfg.get("num_episodes", 100))
    eval_cfg = dataset_cfg.get("eval", {}) if isinstance(dataset_cfg.get("eval"), dict) else {}

    test_cfg = dataset_cfg.get("test", {}) if isinstance(dataset_cfg.get("test"), dict) else {}
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
    test_data_dir = str(test_cfg.get("data_dir", "./test_data"))
    test_force_regenerate = bool(test_cfg.get("force_regenerate", False)) or bool(args.force_regenerate_test)

    test_dataset = create_or_load_dataset(
        base_seed=test_base_seed,
        num_episodes=test_num_episodes,
        data_dir=test_data_dir,
        force_regenerate=test_force_regenerate,
        num_locations=int(scene_cfg.get("num_locations", env_config.NUM_LOCATIONS)),
        area_size=float(scene_cfg.get("area_size", env_config.AREA_SIZE)),
        num_uavs=int(scene_cfg.get("num_uavs", env_config.NUM_UAVS)),
        num_requests=int(scene_cfg.get("num_requests", env_config.NUM_REQUESTS)),
    )
    print(
        f"[MAPPO][TEST] test dataset ready: episodes={len(test_dataset)} "
        f"seed={test_base_seed} dir={test_data_dir}",
        flush=True,
    )

    env = MAPPOSFCEnv(config_yaml_path=args.config, episode_data=test_dataset[0], seed=args.seed)
    first_obs = env.reset()

    checkpoint_path = os.path.abspath(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["policy_state_dict"] if isinstance(ckpt, dict) and "policy_state_dict" in ckpt else ckpt

    policy = _build_policy(first_obs, device=device, yaml_cfg=yaml_cfg)
    try:
        policy.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        inferred_actor_type = _infer_actor_type_from_state_dict(state_dict)
        policy = _build_policy(first_obs, device=device, yaml_cfg=yaml_cfg, actor_type_override=inferred_actor_type)
        policy.load_state_dict(state_dict, strict=True)
        print(
            f"[MAPPO][TEST] actor_type from config 与 checkpoint 不一致，已自动切换为 '{inferred_actor_type}'。",
            flush=True,
        )
        print(f"[MAPPO][TEST] 原始报错：{e}", flush=True)

    policy = policy.to(device)
    policy.eval()

    output_root = _resolve_output_root(logging_cfg, args.config)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"mappo_test_seed{args.seed}_{stamp}"
    experiment_root = os.path.join(output_root, experiment_name)
    log_dir = os.path.join(experiment_root, "log")
    samples_dir = os.path.join(log_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    step_csv_path = os.path.join(log_dir, "test_step_details.csv")
    slot_csv_path = os.path.join(log_dir, "test_slot_details.csv")
    first_step_attn_csv_path = os.path.join(log_dir, "test_first_step_attention.csv")
    episode_csv_path = os.path.join(log_dir, "test_episode_summary.csv")
    summary_json_path = os.path.join(log_dir, "test_summary.json")

    step_fields = [
        "episode_index",
        "dataset_episode_id",
        "dataset_seed",
        "step_index",
        "slot_before_step",
        "round_before_step",
        "turn_before_step",
        "agent_id",
        "action",
        "action_type",
        "request_idx",
        "request_id",
        "vnf_idx",
        "vnf_id",
        "vnf_location_id",
        "valid_claim_count",
        "invalid_count",
        "reward",
        "done",
        "slot_ended",
        "completed_count",
        "pending_count",
        "success_rate",
    ]
    slot_fields = [
        "episode_index",
        "dataset_episode_id",
        "dataset_seed",
        "slot_id",
        "uav_id",
        "init_location_id",
        "init_x",
        "init_y",
        "init_energy",
        "init_cpu_capacity",
        "init_is_busy",
        "end_location_id",
        "end_x",
        "end_y",
        "end_energy",
        "end_cpu_capacity",
        "end_is_busy",
        "actions_in_slot_json",
        "executed_vnfs_json",
    ]
    episode_fields = [
        "episode_index",
        "dataset_episode_id",
        "dataset_seed",
        "episode_return",
        "episode_length",
        "success_rate",
        "completed_count",
        "pending_count",
        "final_time_slot",
    ]
    first_step_attn_fields = [
        "episode_index",
        "dataset_episode_id",
        "dataset_seed",
        "slot_id",
        "current_agent_id",
        "attention_available",
        "num_heads",
        "attention_mean_scores_json",
        "attention_per_head_scores_json",
        "task_valid_mask_json",
        "reason",
    ]

    step_file = open(step_csv_path, "w", newline="", encoding="utf-8")
    slot_file = open(slot_csv_path, "w", newline="", encoding="utf-8")
    first_step_attn_file = open(first_step_attn_csv_path, "w", newline="", encoding="utf-8")
    episode_file = open(episode_csv_path, "w", newline="", encoding="utf-8")
    step_writer = csv.DictWriter(step_file, fieldnames=step_fields)
    slot_writer = csv.DictWriter(slot_file, fieldnames=slot_fields)
    first_step_attn_writer = csv.DictWriter(first_step_attn_file, fieldnames=first_step_attn_fields)
    episode_writer = csv.DictWriter(episode_file, fieldnames=episode_fields)
    step_writer.writeheader()
    slot_writer.writeheader()
    first_step_attn_writer.writeheader()
    episode_writer.writeheader()

    deterministic = not args.stochastic
    episode_returns: List[float] = []
    episode_success_rates: List[float] = []
    episode_lengths: List[int] = []
    episode_completed_counts: List[int] = []

    for ep_idx in range(len(test_dataset)):
        ep_data: EpisodeData = test_dataset[ep_idx]
        env.set_episode_data(ep_data)
        obs = env.reset()
        first_step_attn = _extract_first_step_attention(policy=policy, obs=obs, device=device)
        first_step_attn_writer.writerow(
            {
                "episode_index": int(ep_idx),
                "dataset_episode_id": int(ep_data.episode_id),
                "dataset_seed": int(ep_data.seed),
                "slot_id": int(env.current_time_slot),
                "current_agent_id": int(first_step_attn["current_agent_id"]),
                "attention_available": int(bool(first_step_attn["attention_available"])),
                "num_heads": int(first_step_attn["num_heads"]),
                "attention_mean_scores_json": json.dumps(first_step_attn["attention_mean_scores"], ensure_ascii=False),
                "attention_per_head_scores_json": json.dumps(first_step_attn["attention_per_head_scores"], ensure_ascii=False),
                "task_valid_mask_json": json.dumps(first_step_attn["task_valid_mask"], ensure_ascii=False),
                "reason": str(first_step_attn.get("reason", "")),
            }
        )
        done = False
        step_idx = 0
        ep_return = 0.0
        last_info: Dict[str, Any] = {}

        slot_records: Dict[int, Dict[str, Any]] = {}
        current_slot = int(env.current_time_slot)
        slot_records[current_slot] = {
            "slot_id": current_slot,
            "initial_uavs": _snapshot_uavs(env.uavs),
            "final_uavs": None,
            "uav_actions": {int(aid): [] for aid in env.agent_ids},
            "uav_executed_vnfs": {int(aid): [] for aid in env.agent_ids},
        }
        step_details: List[Dict[str, Any]] = []

        with torch.no_grad():
            while not done:
                slot_before = int(env.current_time_slot)
                round_before = int(env.current_round)
                turn_before = int(env.turn_in_round)
                agent_id = int(obs["current_agent_id"])

                obs_t = obs_to_tensors(obs, device=device)
                act_out = policy.act(obs_t, deterministic=deterministic)
                action = int(act_out["action"][0].item())

                action_detail = _decode_action_detail(env, action)
                next_obs, reward, done, info = env.step(agent_id, action)
                ep_return += float(reward)
                step_idx += 1
                last_info = info

                rec = slot_records.get(slot_before)
                if rec is None:
                    rec = {
                        "slot_id": slot_before,
                        "initial_uavs": _snapshot_uavs(env.uavs),
                        "final_uavs": None,
                        "uav_actions": {int(aid): [] for aid in env.agent_ids},
                        "uav_executed_vnfs": {int(aid): [] for aid in env.agent_ids},
                    }
                    slot_records[slot_before] = rec

                action_trace = {
                    "step_index": int(step_idx),
                    "agent_id": int(agent_id),
                    "action": int(action),
                    "action_type": str(action_detail["action_type"]),
                    "request_id": action_detail["request_id"],
                    "vnf_idx": action_detail["vnf_idx"],
                    "vnf_id": action_detail["vnf_id"],
                    "vnf_location_id": action_detail["vnf_location_id"],
                    "valid_claim_count": int(info.get("valid_claim_count", 0)),
                    "invalid_count": int(info.get("invalid_count", 0)),
                    "reward": float(reward),
                }
                rec["uav_actions"][agent_id].append(action_trace)

                if int(info.get("valid_claim_count", 0)) > 0 and action_detail["action_type"] == "claim":
                    rec["uav_executed_vnfs"][agent_id].append(
                        {
                            "step_index": int(step_idx),
                            "request_id": action_detail["request_id"],
                            "vnf_idx": action_detail["vnf_idx"],
                            "vnf_id": action_detail["vnf_id"],
                            "vnf_location_id": action_detail["vnf_location_id"],
                        }
                    )

                step_row = {
                    "episode_index": int(ep_idx),
                    "dataset_episode_id": int(ep_data.episode_id),
                    "dataset_seed": int(ep_data.seed),
                    "step_index": int(step_idx),
                    "slot_before_step": slot_before,
                    "round_before_step": round_before,
                    "turn_before_step": turn_before,
                    "agent_id": agent_id,
                    "action": int(action),
                    "action_type": str(action_detail["action_type"]),
                    "request_idx": action_detail["request_idx"],
                    "request_id": action_detail["request_id"],
                    "vnf_idx": action_detail["vnf_idx"],
                    "vnf_id": action_detail["vnf_id"],
                    "vnf_location_id": action_detail["vnf_location_id"],
                    "valid_claim_count": int(info.get("valid_claim_count", 0)),
                    "invalid_count": int(info.get("invalid_count", 0)),
                    "reward": float(reward),
                    "done": int(bool(done)),
                    "slot_ended": int(bool(info.get("slot_ended", False))),
                    "completed_count": int(info.get("completed_count", 0)),
                    "pending_count": int(info.get("pending_count", 0)),
                    "success_rate": float(info.get("success_rate", 0.0)),
                }
                step_writer.writerow(step_row)
                step_details.append(step_row)

                if bool(info.get("slot_ended", False)):
                    rec["final_uavs"] = _snapshot_uavs(env.uavs)
                    next_slot = int(env.current_time_slot)
                    if (not done) and (next_slot <= int(env.cfg.num_time_slots)) and (next_slot not in slot_records):
                        slot_records[next_slot] = {
                            "slot_id": next_slot,
                            "initial_uavs": _snapshot_uavs(env.uavs),
                            "final_uavs": None,
                            "uav_actions": {int(aid): [] for aid in env.agent_ids},
                            "uav_executed_vnfs": {int(aid): [] for aid in env.agent_ids},
                        }

                obs = next_obs

        # 若最后一个时隙没有触发 slot_ended，也补上结束状态快照
        for slot_id, rec in slot_records.items():
            if rec["final_uavs"] is None:
                rec["final_uavs"] = _snapshot_uavs(env.uavs)

            init_map = {int(u["uav_id"]): u for u in rec["initial_uavs"]}
            end_map = {int(u["uav_id"]): u for u in rec["final_uavs"]}
            for uav_id in env.agent_ids:
                iu = init_map[int(uav_id)]
                eu = end_map[int(uav_id)]
                slot_writer.writerow(
                    {
                        "episode_index": int(ep_idx),
                        "dataset_episode_id": int(ep_data.episode_id),
                        "dataset_seed": int(ep_data.seed),
                        "slot_id": int(slot_id),
                        "uav_id": int(uav_id),
                        "init_location_id": int(iu["location_id"]),
                        "init_x": float(iu["x"]),
                        "init_y": float(iu["y"]),
                        "init_energy": float(iu["energy"]),
                        "init_cpu_capacity": float(iu["cpu_capacity"]),
                        "init_is_busy": int(bool(iu["is_busy"])),
                        "end_location_id": int(eu["location_id"]),
                        "end_x": float(eu["x"]),
                        "end_y": float(eu["y"]),
                        "end_energy": float(eu["energy"]),
                        "end_cpu_capacity": float(eu["cpu_capacity"]),
                        "end_is_busy": int(bool(eu["is_busy"])),
                        "actions_in_slot_json": json.dumps(rec["uav_actions"][int(uav_id)], ensure_ascii=False),
                        "executed_vnfs_json": json.dumps(rec["uav_executed_vnfs"][int(uav_id)], ensure_ascii=False),
                    }
                )

        ep_success = float(last_info.get("success_rate", 0.0))
        ep_completed = int(last_info.get("completed_count", 0))
        ep_pending = int(last_info.get("pending_count", 0))
        ep_final_slot = int(last_info.get("current_slot", env.current_time_slot))

        episode_writer.writerow(
            {
                "episode_index": int(ep_idx),
                "dataset_episode_id": int(ep_data.episode_id),
                "dataset_seed": int(ep_data.seed),
                "episode_return": float(ep_return),
                "episode_length": int(step_idx),
                "success_rate": ep_success,
                "completed_count": ep_completed,
                "pending_count": ep_pending,
                "final_time_slot": ep_final_slot,
            }
        )

        sample_json = {
            "episode_index": int(ep_idx),
            "dataset_episode_id": int(ep_data.episode_id),
            "dataset_seed": int(ep_data.seed),
            "episode_return": float(ep_return),
            "episode_length": int(step_idx),
            "success_rate": ep_success,
            "completed_count": ep_completed,
            "pending_count": ep_pending,
            "final_time_slot": ep_final_slot,
            "steps": step_details,
            "slots": [
                {
                    "slot_id": int(k),
                    "initial_uavs": v["initial_uavs"],
                    "final_uavs": v["final_uavs"],
                    "uav_actions": v["uav_actions"],
                    "uav_executed_vnfs": v["uav_executed_vnfs"],
                }
                for k, v in sorted(slot_records.items(), key=lambda kv: kv[0])
            ],
        }
        sample_path = os.path.join(samples_dir, f"sample_{ep_idx:04d}.json")
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(sample_json, f, ensure_ascii=False, indent=2)

        episode_returns.append(float(ep_return))
        episode_success_rates.append(ep_success)
        episode_lengths.append(int(step_idx))
        episode_completed_counts.append(ep_completed)

        print(
            f"[MAPPO][TEST] episode={ep_idx + 1}/{len(test_dataset)} "
            f"return={ep_return:.4f} success={ep_success:.4f} "
            f"len={step_idx} completed={ep_completed}",
            flush=True,
        )

    step_file.close()
    slot_file.close()
    first_step_attn_file.close()
    episode_file.close()

    denom = float(max(len(episode_returns), 1))
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config_path": os.path.abspath(args.config),
        "checkpoint_path": checkpoint_path,
        "deterministic": bool(deterministic),
        "device": str(device),
        "test_dataset": {
            "base_seed": int(test_base_seed),
            "num_episodes": int(len(test_dataset)),
            "data_dir": str(test_data_dir),
            "force_regenerate": bool(test_force_regenerate),
        },
        "metrics": {
            "avg_return": float(sum(episode_returns) / denom),
            "avg_success_rate": float(sum(episode_success_rates) / denom),
            "avg_episode_length": float(sum(episode_lengths) / denom),
            "avg_completed_count": float(sum(episode_completed_counts) / denom),
            "num_episodes": int(len(episode_returns)),
        },
        "artifacts": {
            "episode_csv": episode_csv_path,
            "step_csv": step_csv_path,
            "slot_csv": slot_csv_path,
            "first_step_attention_csv": first_step_attn_csv_path,
            "samples_dir": samples_dir,
        },
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[MAPPO][TEST] finished. summary={summary_json_path}")
    print(f"[MAPPO][TEST] episode_csv={episode_csv_path}")
    print(f"[MAPPO][TEST] step_csv={step_csv_path}")
    print(f"[MAPPO][TEST] slot_csv={slot_csv_path}")
    print(f"[MAPPO][TEST] first_step_attention_csv={first_step_attn_csv_path}")
    print(f"[MAPPO][TEST] samples_dir={samples_dir}")
    print(
        f"[MAPPO][TEST] avg_success_rate={summary['metrics']['avg_success_rate']:.6f} "
        f"avg_return={summary['metrics']['avg_return']:.6f}",
    )


if __name__ == "__main__":
    main()
