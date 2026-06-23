# -*- coding: utf-8 -*-
"""在指定 config 的评估集上测试 MADRL2Stages（MAPPO checkpoint）。

每个 episode 记录：部署成功率(success_rate) 与 电量消耗总量(energy_consumed)。
结果输出到 Test/output/<scene_tag>/MADRL2Stages/。

用法:
    python Test/test_madrl2stages.py \
        --config MADRL2Stages/config/config_large.yaml \
        --checkpoint MADRL2Stages/output/<run>/checkpoint/policy_final.pt

能量统计说明: 环境内部没有总能耗计数器，UAV 在基站会被充满电。
本脚本在每个 step 前后快照各 UAV 的 energy，仅累加“下降量”（充电造成的上升量忽略），
因此 energy_consumed = 全程实际消耗的电量总和。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# 目录解析：Test/ 的上一级是仓库根
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_TEST_DIR, ".."))
_MADRL_DIR = os.path.join(_REPO_ROOT, "MADRL2Stages")
_SRC_DIR = os.path.join(_MADRL_DIR, "src")
for _path in (_SRC_DIR, _MADRL_DIR, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config as env_config


def _load_dataset_module():
    """加载 DRL/training/dataset.py，绕过 DRL.training.__init__（避免 gym 依赖）。"""
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


def _load_yaml(path: str) -> Dict[str, object]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _scene_tag(scene: Dict[str, object]) -> str:
    num_uavs = int(scene.get("num_uavs", 8))
    num_locations = int(scene.get("num_locations", 20))
    num_requests = int(scene.get("num_requests", 30))
    num_time_slots = int(scene.get("num_time_slots", 4))
    return f"{num_uavs}UAV_{num_locations}location_{num_requests}SFC_{num_time_slots}slot"


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

    # 兼容旧 checkpoint：清理后的 attn actor 删除了 task_encoder/task_blocks/cross_attn/policy_head，
    # 旧权重里仍含这些键。按当前模型结构过滤后非严格加载；同时拦截真正缺失的关键权重。
    model_keys = set(policy.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    dropped = [k for k in state_dict.keys() if k not in model_keys]
    result = policy.load_state_dict(filtered, strict=False)

    missing = [k for k in result.missing_keys]
    if missing:
        # 形状不匹配（如 mlp/attn 不一致）时回退重建后再加载
        actor_type = _infer_actor_type_from_state_dict(state_dict)
        policy = _build_policy(first_obs, device=device, yaml_cfg=yaml_cfg, actor_type_override=actor_type)
        model_keys = set(policy.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in model_keys}
        result = policy.load_state_dict(filtered, strict=False)
        missing = list(result.missing_keys)
        if missing:
            raise RuntimeError(f"加载 checkpoint 后仍缺失关键权重: {missing}")

    if dropped:
        print(f"[load] 忽略 {len(dropped)} 个已废弃权重键（policy_head/cross_attn/task_encoder 等）。", flush=True)

    policy = policy.to(device)
    policy.eval()
    return policy


@torch.no_grad()
def _select_action(
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
    policy: MAPPOPolicy,
    device: torch.device,
    use_policy_deploy_score: bool,
) -> Dict[str, Any]:
    obs = env.reset()
    done = False
    step_count = 0
    energy_consumed = 0.0
    last_info: Dict[str, Any] = {
        "completed_count": 0,
        "pending_count": len(env.requests),
        "success_rate": 0.0,
    }

    while not done:
        agent_id = int(obs["current_agent_id"])
        # step 前快照各 UAV 电量
        energy_before = [float(u.energy) for u in env.uavs]

        action, deploy_score_table = _select_action(
            policy=policy, obs=obs, device=device, use_policy_deploy_score=use_policy_deploy_score
        )
        obs, reward, done, info = env.step(agent_id, action, deploy_score_table=deploy_score_table)

        # 仅累加下降量（充电造成的上升忽略）
        for u, before in zip(env.uavs, energy_before):
            drop = before - float(u.energy)
            if drop > 0.0:
                energy_consumed += drop

        step_count += 1
        last_info = info

    total_requests = max(len(env.requests), 1)
    return {
        "steps": int(step_count),
        "completed_count": int(last_info.get("completed_count", 0)),
        "total_requests": int(total_requests),
        "success_rate": float(last_info.get("success_rate", 0.0)),
        "energy_consumed": float(energy_consumed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test MADRL2Stages checkpoint on the eval dataset defined by config.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(_MADRL_DIR, "config", "config_large.yaml"),
        help="配置文件路径（YAML），决定场景规模、网络结构与评估集。",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="MAPPO checkpoint 路径 (policy_*.pt)。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--device", type=str, default=None, help="覆盖设备，如 cpu/cuda。")
    parser.add_argument("--num-episodes", type=int, default=None, help="评估 episode 数；默认读 dataset.eval.num_eval_episodes。")
    parser.add_argument(
        "--deploy-score",
        choices=["none", "policy"],
        default="policy",
        help="是否把 actor 的 VNF 分数传给部署阶段；default=policy（与训练一致）。",
    )
    parser.add_argument("--output-root", type=str, default=os.path.join(_TEST_DIR, "output"), help="测试结果输出根目录。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = _load_yaml(args.config)
    scene_cfg = cfg.get("scene", {}) if isinstance(cfg.get("scene"), dict) else {}
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
    eval_cfg = dataset_cfg.get("eval", {}) if isinstance(dataset_cfg.get("eval"), dict) else {}

    num_locations = int(scene_cfg.get("num_locations", env_config.NUM_LOCATIONS))
    area_size = float(scene_cfg.get("area_size", env_config.AREA_SIZE))
    num_uavs = int(scene_cfg.get("num_uavs", env_config.NUM_UAVS))
    num_requests = int(scene_cfg.get("num_requests", env_config.NUM_REQUESTS))
    num_time_slots = int(scene_cfg.get("num_time_slots", 4))

    base_seed = int(dataset_cfg.get("base_seed", args.seed))
    eval_seed = int(eval_cfg.get("base_seed", base_seed + 100000))
    cfg_eval_eps = int(eval_cfg.get("num_episodes", 10))
    requested = int(args.num_episodes) if args.num_episodes is not None else int(
        eval_cfg.get("num_eval_episodes", cfg_eval_eps)
    )
    dataset_num_episodes = max(cfg_eval_eps, requested)
    eval_data_dir = str(eval_cfg.get("data_dir", os.path.join(_REPO_ROOT, "eval_data")))

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_dataset = create_or_load_dataset(
        base_seed=eval_seed,
        num_episodes=dataset_num_episodes,
        data_dir=eval_data_dir,
        force_regenerate=bool(eval_cfg.get("force_regenerate", False)),
        num_locations=num_locations,
        area_size=area_size,
        num_uavs=num_uavs,
        num_requests=num_requests,
    )
    num_eps = min(requested, len(eval_dataset))

    first_env = MAPPOSFCEnv(config_yaml_path=args.config, episode_data=eval_dataset[0], seed=args.seed)
    first_env.set_request_shuffle_on_reset(False)
    first_obs = first_env.reset()
    policy = _load_policy(args.checkpoint, first_obs=first_obs, device=device, yaml_cfg=cfg)

    scene_tag = _scene_tag(scene_cfg)
    output_dir = os.path.join(os.path.abspath(args.output_root), scene_tag, "MADRL2Stages")
    os.makedirs(output_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for i in range(num_eps):
        ep = eval_dataset[i]
        env = MAPPOSFCEnv(config_yaml_path=args.config, episode_data=ep, seed=args.seed + i)
        env.set_request_shuffle_on_reset(False)
        start = time.perf_counter()
        metrics = _run_episode(
            env=env,
            policy=policy,
            device=device,
            use_policy_deploy_score=(args.deploy_score == "policy"),
        )
        metrics["runtime_sec"] = float(time.perf_counter() - start)
        row = {
            "episode_index": int(i),
            "dataset_seed": int(getattr(ep, "seed", -1)),
            **metrics,
        }
        rows.append(row)
        print(
            f"[MADRL2Stages] episode={i + 1}/{num_eps} success={row['success_rate']:.4f} "
            f"completed={int(row['completed_count'])} energy={row['energy_consumed']:.2f}",
            flush=True,
        )

    n = max(len(rows), 1)
    summary = {
        "method": "MADRL2Stages",
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "scene_tag": scene_tag,
        "deploy_score": args.deploy_score,
        "episodes": len(rows),
        "avg_success_rate": float(sum(r["success_rate"] for r in rows) / n),
        "avg_completed_count": float(sum(r["completed_count"] for r in rows) / n),
        "avg_energy_consumed": float(sum(r["energy_consumed"] for r in rows) / n),
        "avg_runtime_sec": float(sum(r["runtime_sec"] for r in rows) / n),
    }

    detail_csv = os.path.join(output_dir, "episode_metrics.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode_index", "dataset_seed", "success_rate", "completed_count",
                "total_requests", "energy_consumed", "steps", "runtime_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_json = os.path.join(output_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== MADRL2Stages summary ===", flush=True)
    print(
        f"episodes={summary['episodes']} avg_success={summary['avg_success_rate']:.4f} "
        f"avg_energy={summary['avg_energy_consumed']:.2f}",
        flush=True,
    )
    print(f"detail={detail_csv}", flush=True)
    print(f"summary={summary_json}", flush=True)


if __name__ == "__main__":
    main()
