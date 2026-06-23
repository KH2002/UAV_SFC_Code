# -*- coding: utf-8 -*-
"""在指定 config 的评估集上测试 MPopLoc baseline。

每个 episode 记录：部署成功率(success_rate) 与 电量消耗总量(energy_consumed)。
结果输出到 Test/output/<scene_tag>/MPopLoc/。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from typing import Dict, List

# 项目根目录（Test 的上一级）
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_TEST_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml


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


_dataset_mod = _load_dataset_module()
EpisodeData = _dataset_mod.EpisodeData
create_or_load_dataset = _dataset_mod.create_or_load_dataset
from mpoploc import MPopLocSolver


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


def _run_episode(ep: EpisodeData, num_time_slots: int) -> Dict[str, float]:
    start = time.perf_counter()
    solver = MPopLocSolver(ep.uavs, ep.requests, ep.locations, num_time_slots=num_time_slots)
    timeline = solver.solve()
    elapsed = time.perf_counter() - start

    total_serviced = sum(len(v) for v in timeline.values())
    total_requests = max(len(ep.requests), 1)
    return {
        "success_rate": float(total_serviced) / float(total_requests),
        "completed_count": float(total_serviced),
        "total_requests": float(total_requests),
        "energy_consumed": float(getattr(solver, "total_energy_consumed", 0.0)),
        "runtime_sec": float(elapsed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test MPopLoc on the eval dataset defined by config.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(_REPO_ROOT, "MADRL2Stages", "config", "config_large.yaml"),
        help="配置文件路径（YAML），决定场景规模与评估集。",
    )
    parser.add_argument("--num-episodes", type=int, default=None, help="评估 episode 数；默认读 dataset.eval.num_eval_episodes。")
    parser.add_argument("--output-root", type=str, default=os.path.join(_TEST_DIR, "output"), help="测试结果输出根目录。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config)

    scene_cfg = cfg.get("scene", {}) if isinstance(cfg.get("scene"), dict) else {}
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
    eval_cfg = dataset_cfg.get("eval", {}) if isinstance(dataset_cfg.get("eval"), dict) else {}

    num_locations = int(scene_cfg.get("num_locations", 20))
    area_size = float(scene_cfg.get("area_size", 1000.0))
    num_uavs = int(scene_cfg.get("num_uavs", 8))
    num_requests = int(scene_cfg.get("num_requests", 30))
    num_time_slots = int(scene_cfg.get("num_time_slots", 4))

    base_seed = int(dataset_cfg.get("base_seed", 42))
    eval_seed = int(eval_cfg.get("base_seed", base_seed + 100000))
    cfg_eval_eps = int(eval_cfg.get("num_episodes", 10))
    requested = int(args.num_episodes) if args.num_episodes is not None else int(
        eval_cfg.get("num_eval_episodes", cfg_eval_eps)
    )
    dataset_num_episodes = max(cfg_eval_eps, requested)
    eval_data_dir = str(eval_cfg.get("data_dir", os.path.join(_REPO_ROOT, "eval_data")))

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

    scene_tag = _scene_tag(scene_cfg)
    output_dir = os.path.join(os.path.abspath(args.output_root), scene_tag, "MPopLoc")
    os.makedirs(output_dir, exist_ok=True)

    rows: List[Dict[str, float]] = []
    for i in range(num_eps):
        ep = eval_dataset[i]
        metrics = _run_episode(ep, num_time_slots=num_time_slots)
        row = {
            "episode_index": int(i),
            "dataset_seed": int(getattr(ep, "seed", -1)),
            **metrics,
        }
        rows.append(row)
        print(
            f"[MPopLoc] episode={i + 1}/{num_eps} success={row['success_rate']:.4f} "
            f"completed={int(row['completed_count'])} energy={row['energy_consumed']:.2f}",
            flush=True,
        )

    n = max(len(rows), 1)
    summary = {
        "method": "MPopLoc",
        "config": os.path.abspath(args.config),
        "scene_tag": scene_tag,
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
                "total_requests", "energy_consumed", "runtime_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_json = os.path.join(output_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== MPopLoc summary ===", flush=True)
    print(
        f"episodes={summary['episodes']} avg_success={summary['avg_success_rate']:.4f} "
        f"avg_energy={summary['avg_energy_consumed']:.2f}",
        flush=True,
    )
    print(f"detail={detail_csv}", flush=True)
    print(f"summary={summary_json}", flush=True)


if __name__ == "__main__":
    main()
