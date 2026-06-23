# -*- coding: utf-8 -*-
"""Evaluate location-selection baselines on the MADRL2Stages environment.

Baselines:
- random: uniformly sample one valid location action from the current mask
- greedy: choose the valid location with the largest immediate SFC potential
- trained: load a MAPPO checkpoint and choose deterministic actor actions

By default all three baselines use the environment's rule-based VNF deployment, so the
comparison isolates the quality of location selection. Use --trained-deploy-score
policy to also pass the learned VNF deployment score table for the trained policy.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml

_TRAINING_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_TRAINING_DIR, ".."))
_MADRL_DIR = os.path.abspath(os.path.join(_SRC_DIR, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_MADRL_DIR, ".."))
for _path in (_SRC_DIR, _MADRL_DIR, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config as env_config


def _load_dataset_module():
    """Load DRL/training/dataset.py without importing DRL.training.__init__."""
    dataset_path = os.path.join(_REPO_ROOT, "DRL", "training", "dataset.py")
    spec = importlib.util.spec_from_file_location("DRL.training.dataset", dataset_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load dataset module from {dataset_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["DRL.training.dataset"] = module
    spec.loader.exec_module(module)
    return module


create_or_load_dataset = _load_dataset_module().create_or_load_dataset
from envs.env import MAPPOSFCEnv
from models import MAPPOPolicy, obs_to_tensors
from training import load_trainer_config


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
    action_dim = int(action_mask.shape[-1]) if action_mask.size > 0 else int(env_config.NUM_LOCATIONS)

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
    first_obs: Dict[str, object],
    device: torch.device,
    yaml_cfg: Dict[str, object],
) -> MAPPOPolicy:
    ckpt = torch.load(os.path.abspath(checkpoint_path), map_location="cpu")
    state_dict = ckpt["policy_state_dict"] if isinstance(ckpt, dict) and "policy_state_dict" in ckpt else ckpt

    policy = _build_policy(first_obs, device=device, yaml_cfg=yaml_cfg)
    try:
        policy.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        actor_type = _infer_actor_type_from_state_dict(state_dict)
        policy = _build_policy(first_obs, device=device, yaml_cfg=yaml_cfg, actor_type_override=actor_type)
        policy.load_state_dict(state_dict, strict=True)

    policy = policy.to(device)
    policy.eval()
    return policy


def _valid_actions(mask: np.ndarray) -> np.ndarray:
    actions = np.flatnonzero(np.asarray(mask, dtype=np.int32) > 0)
    if actions.size <= 0:
        actions = np.arange(mask.shape[-1], dtype=np.int64)
    return actions.astype(np.int64)


def _select_random_action(obs: Dict[str, object], rng: np.random.Generator) -> int:
    actions = _valid_actions(np.asarray(obs["action_mask"], dtype=np.int32))
    return int(rng.choice(actions))


def _select_greedy_action(env: MAPPOSFCEnv, obs: Dict[str, object], rng: np.random.Generator) -> int:
    agent_id = int(obs["current_agent_id"])
    actions = _valid_actions(np.asarray(obs["action_mask"], dtype=np.int32))

    best_score = None
    best_actions: List[int] = []
    for action in actions:
        location_id = int(env.decode_action(int(action)))
        potential = float(env._count_location_sfc_potential(agent_id, location_id))
        # Tie-breaker: prefer locations that currently appear more often in pending VNFs.
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

    return int(rng.choice(np.asarray(best_actions, dtype=np.int64)))


@torch.no_grad()
def _select_trained_action(
    policy: MAPPOPolicy,
    obs: Dict[str, object],
    device: torch.device,
    use_policy_deploy_score: bool,
) -> Tuple[int, Optional[np.ndarray]]:
    obs_t = obs_to_tensors(obs, device=device)
    act_out = policy.act(obs_t, deterministic=True)
    action = int(act_out["action"][0].item())
    deploy_score_table = None
    if use_policy_deploy_score:
        deploy_score_table = policy.score_vnfs_for_deployment(obs_t)[0].detach().cpu().numpy()
    return action, deploy_score_table


def _run_episode(
    env: MAPPOSFCEnv,
    baseline: str,
    rng: np.random.Generator,
    policy: Optional[MAPPOPolicy],
    device: torch.device,
    use_policy_deploy_score: bool,
) -> Dict[str, Any]:
    obs = env.reset()
    done = False
    step_count = 0
    total_reward = 0.0
    invalid_count = 0
    valid_claim_count = 0
    slot_completed_total = 0
    slot_deployed_vnf_total = 0
    last_info: Dict[str, Any] = {
        "completed_count": 0,
        "pending_count": len(env.requests),
        "success_rate": 0.0,
        "current_slot": env.current_time_slot,
    }

    while not done:
        agent_id = int(obs["current_agent_id"])
        deploy_score_table = None
        if baseline == "random":
            action = _select_random_action(obs, rng)
        elif baseline == "greedy":
            action = _select_greedy_action(env, obs, rng)
        elif baseline == "trained":
            if policy is None:
                raise ValueError("trained baseline requires a checkpoint policy")
            action, deploy_score_table = _select_trained_action(
                policy=policy,
                obs=obs,
                device=device,
                use_policy_deploy_score=use_policy_deploy_score,
            )
        else:
            raise ValueError(f"Unknown baseline: {baseline}")

        obs, reward, done, info = env.step(agent_id, action, deploy_score_table=deploy_score_table)
        step_count += 1
        total_reward += float(reward)
        invalid_count += int(info.get("invalid_count", 0))
        valid_claim_count += int(info.get("valid_claim_count", 0))
        slot_completed_total += int(info.get("slot_completed", 0))
        slot_deployed_vnf_total += int(info.get("slot_deployed_vnf", 0))
        last_info = info

    return {
        "steps": int(step_count),
        "total_reward": float(total_reward),
        "completed_count": int(last_info.get("completed_count", 0)),
        "pending_count": int(last_info.get("pending_count", 0)),
        "success_rate": float(last_info.get("success_rate", 0.0)),
        "slot_completed_total": int(slot_completed_total),
        "slot_deployed_vnf_total": int(slot_deployed_vnf_total),
        "invalid_count": int(invalid_count),
        "valid_claim_count": int(valid_claim_count),
        "final_slot": int(last_info.get("current_slot", env.current_time_slot)),
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    baselines = sorted({str(row["baseline"]) for row in rows})
    summary: List[Dict[str, Any]] = []
    for baseline in baselines:
        subset = [row for row in rows if str(row["baseline"]) == baseline]
        summary.append(
            {
                "baseline": baseline,
                "episodes": len(subset),
                "avg_success_rate": _mean(row["success_rate"] for row in subset),
                "avg_completed_count": _mean(row["completed_count"] for row in subset),
                "avg_pending_count": _mean(row["pending_count"] for row in subset),
                "avg_steps": _mean(row["steps"] for row in subset),
                "avg_total_reward": _mean(row["total_reward"] for row in subset),
                "avg_slot_deployed_vnf": _mean(row["slot_deployed_vnf_total"] for row in subset),
                "avg_invalid_count": _mean(row["invalid_count"] for row in subset),
            }
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate random/greedy/trained location baselines.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(_MADRL_DIR, "config", "config_small.yaml"),
        help="配置文件路径。",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(
            _MADRL_DIR,
            "output",
            "madrl2stages_split_clip_retnorm_seed42_20260618",
            "checkpoint",
            "policy_final.pt",
        ),
        help="trained baseline 使用的 checkpoint。",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--device", type=str, default=None, help="覆盖设备，如 cpu/cuda。")
    parser.add_argument("--num-episodes", type=int, default=None, help="评估 episode 数；默认读 dataset.eval.num_eval_episodes。")
    parser.add_argument("--baselines", nargs="+", default=["random", "greedy", "trained"], choices=["random", "greedy", "trained"])
    parser.add_argument("--force-regenerate-eval", action="store_true", help="强制重建 eval_data。")
    parser.add_argument(
        "--trained-deploy-score",
        choices=["none", "policy"],
        default="none",
        help="trained baseline 是否把 actor 的 VNF 分数传给部署阶段；默认 none 以隔离 location 选择。",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录；默认写到 logging.output_dir/baseline_eval_<time>。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    yaml_cfg = _load_yaml(args.config)
    scene_cfg = yaml_cfg.get("scene", {}) if isinstance(yaml_cfg.get("scene"), dict) else {}
    dataset_cfg = yaml_cfg.get("dataset", {}) if isinstance(yaml_cfg.get("dataset"), dict) else {}
    eval_cfg = dataset_cfg.get("eval", {}) if isinstance(dataset_cfg.get("eval"), dict) else {}
    logging_cfg = yaml_cfg.get("logging", {}) if isinstance(yaml_cfg.get("logging"), dict) else {}

    trainer_cfg = load_trainer_config(args.config)
    if args.device:
        trainer_cfg.device = args.device
    device = torch.device(trainer_cfg.device)

    num_episodes = int(args.num_episodes if args.num_episodes is not None else eval_cfg.get("num_eval_episodes", eval_cfg.get("num_episodes", 10)))
    dataset_num_episodes = max(int(eval_cfg.get("num_episodes", num_episodes)), num_episodes)
    eval_dataset = create_or_load_dataset(
        base_seed=int(eval_cfg.get("base_seed", int(dataset_cfg.get("base_seed", args.seed)) + 100000)),
        num_episodes=dataset_num_episodes,
        data_dir=str(eval_cfg.get("data_dir", "./eval_data")),
        force_regenerate=bool(eval_cfg.get("force_regenerate", False)) or bool(args.force_regenerate_eval),
        num_locations=int(scene_cfg.get("num_locations", env_config.NUM_LOCATIONS)),
        area_size=float(scene_cfg.get("area_size", env_config.AREA_SIZE)),
        num_uavs=int(scene_cfg.get("num_uavs", env_config.NUM_UAVS)),
        num_requests=int(scene_cfg.get("num_requests", env_config.NUM_REQUESTS)),
    )

    first_env = MAPPOSFCEnv(config_yaml_path=args.config, episode_data=eval_dataset[0], seed=args.seed)
    first_env.set_request_shuffle_on_reset(False)
    first_obs = first_env.reset()

    policy = None
    if "trained" in args.baselines:
        policy = _load_policy(args.checkpoint, first_obs=first_obs, device=device, yaml_cfg=yaml_cfg)

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_root_cfg = str(logging_cfg.get("output_dir", os.path.join(_MADRL_DIR, "output")))
        output_root = _resolve_output_dir(output_root_cfg, args.config)
        output_dir = os.path.join(output_root, f"baseline_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for baseline in args.baselines:
        for episode_index in range(num_episodes):
            episode_data = eval_dataset[episode_index]
            env = MAPPOSFCEnv(config_yaml_path=args.config, episode_data=episode_data, seed=args.seed + episode_index)
            env.set_request_shuffle_on_reset(False)
            rng = np.random.default_rng(args.seed + episode_index * 1009 + hash(baseline) % 997)
            metrics = _run_episode(
                env=env,
                baseline=baseline,
                rng=rng,
                policy=policy,
                device=device,
                use_policy_deploy_score=(baseline == "trained" and args.trained_deploy_score == "policy"),
            )
            row = {
                "baseline": baseline,
                "episode_index": int(episode_index),
                "dataset_episode_id": int(getattr(episode_data, "episode_id", episode_index)),
                "dataset_seed": int(getattr(episode_data, "seed", -1)),
                **metrics,
            }
            rows.append(row)
            print(
                f"[{baseline}] episode={episode_index + 1}/{num_episodes} "
                f"success={row['success_rate']:.4f} completed={row['completed_count']} "
                f"deployed_vnf={row['slot_deployed_vnf_total']} invalid={row['invalid_count']}",
                flush=True,
            )

    summary = _summarize(rows)

    detail_csv = os.path.join(output_dir, "baseline_episode_metrics.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_csv = os.path.join(output_dir, "baseline_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    summary_json = os.path.join(output_dir, "baseline_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "summary": summary}, f, ensure_ascii=False, indent=2)

    print("\n=== Baseline summary ===", flush=True)
    for item in summary:
        print(
            f"{item['baseline']}: success={item['avg_success_rate']:.4f}, "
            f"completed={item['avg_completed_count']:.2f}, deployed_vnf={item['avg_slot_deployed_vnf']:.2f}, "
            f"invalid={item['avg_invalid_count']:.2f}",
            flush=True,
        )
    print(f"\nSaved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
