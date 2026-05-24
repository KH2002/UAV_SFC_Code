# -*- coding: utf-8 -*-
"""使用 MPopLoc 在训练集/验证集上批量评测并保存结果。"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Tuple

# 兼容 `python MAPPO/run_mpoploc_eval.py` 直接启动
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from DRL.training.dataset import EpisodeData, create_or_load_dataset
from mpoploc import MPopLocSolver
import yaml


def _load_yaml(path: str) -> Dict[str, object]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _format_area_value(area_size: float) -> str:
    area_val = float(area_size)
    return str(int(area_val)) if area_val.is_integer() else format(area_val, "g").replace(".", "p")


def _dataset_stem(
    base_seed: int,
    num_episodes: int,
    num_locations: int,
    area_size: float,
    num_uavs: int,
    num_requests: int,
) -> str:
    area_str = _format_area_value(area_size)
    return (
        f"dataset_seed{base_seed}_ep{num_episodes}"
        f"_loc{num_locations}_area{area_str}_uav{num_uavs}_req{num_requests}"
    )


def _run_single_episode(ep: EpisodeData, num_time_slots: int) -> Dict[str, float]:
    start = time.perf_counter()
    solver = MPopLocSolver(ep.uavs, ep.requests, ep.locations, num_time_slots=num_time_slots)
    timeline = solver.solve()
    elapsed = time.perf_counter() - start

    total_serviced = sum(len(v) for v in timeline.values())
    total_requests = max(len(ep.requests), 1)
    success_rate = float(total_serviced) / float(total_requests)

    return {
        "total_serviced": float(total_serviced),
        "total_requests": float(total_requests),
        "success_rate": success_rate,
        "runtime_sec": float(elapsed),
        "energy_consumed": float(getattr(solver, "total_energy_consumed", 0.0)),
    }


def _evaluate_split(
    split_name: str,
    dataset,
    num_time_slots: int,
    out_path: str,
) -> Tuple[List[Dict[str, float]], Dict[str, float], str]:
    rows: List[Dict[str, float]] = []
    num_eps = len(dataset)
    for i in range(num_eps):
        ep = dataset[i]
        metrics = _run_single_episode(ep, num_time_slots=num_time_slots)
        row = {
            "split": split_name,
            "episode_idx": float(i),
            "seed": float(ep.seed),
            **metrics,
        }
        rows.append(row)
        if (i + 1) % 10 == 0 or (i + 1) == num_eps:
            print(f"[MPopLoc] {split_name}: {i + 1}/{num_eps} episodes finished", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "split", "episode_idx", "seed", "total_serviced", "total_requests",
            "success_rate", "runtime_sec", "energy_consumed",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n = max(len(rows), 1)
    summary = {
        "split": split_name,
        "episodes": len(rows),
        "avg_success_rate": float(sum(r["success_rate"] for r in rows) / n),
        "avg_serviced": float(sum(r["total_serviced"] for r in rows) / n),
        "avg_runtime_sec": float(sum(r["runtime_sec"] for r in rows) / n),
        "avg_energy_consumed": float(sum(r["energy_consumed"] for r in rows) / n),
    }

    # 在结果文件末尾追加汇总信息，便于一次性查看。
    with open(out_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("# summary\n")
        f.write(f"# split,{summary['split']}\n")
        f.write(f"# episodes,{summary['episodes']}\n")
        f.write(f"# avg_success_rate,{summary['avg_success_rate']:.8f}\n")
        f.write(f"# avg_serviced,{summary['avg_serviced']:.8f}\n")
        f.write(f"# avg_runtime_sec,{summary['avg_runtime_sec']:.8f}\n")
        f.write(f"# avg_energy_consumed,{summary['avg_energy_consumed']:.8f}\n")

    return rows, summary, out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MPopLoc on train/eval datasets defined by config.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("MAPPO", "config_small.yaml"),
        help="配置文件路径（YAML）。",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="输出实验名（默认自动生成）。",
    )
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

    train_seed = int(dataset_cfg.get("base_seed", 42))
    train_eps = int(dataset_cfg.get("num_episodes", 100))
    train_data_dir = str(dataset_cfg.get("data_dir", "./data"))
    train_force_regenerate = bool(dataset_cfg.get("force_regenerate", False))

    eval_seed = int(eval_cfg.get("base_seed", train_seed + 100000))
    eval_eps = int(eval_cfg.get("num_episodes", 100))
    eval_data_dir = str(eval_cfg.get("data_dir", "./eval_data"))
    eval_force_regenerate = bool(eval_cfg.get("force_regenerate", False))

    print("[MPopLoc] preparing train dataset...", flush=True)
    train_dataset = create_or_load_dataset(
        base_seed=train_seed,
        num_episodes=train_eps,
        data_dir=train_data_dir,
        force_regenerate=train_force_regenerate,
        num_locations=num_locations,
        area_size=area_size,
        num_uavs=num_uavs,
        num_requests=num_requests,
    )
    print(
        f"[MPopLoc] train dataset ready: episodes={len(train_dataset)} seed={train_seed} dir={train_data_dir}",
        flush=True,
    )

    print("[MPopLoc] preparing eval dataset...", flush=True)
    eval_dataset = create_or_load_dataset(
        base_seed=eval_seed,
        num_episodes=eval_eps,
        data_dir=eval_data_dir,
        force_regenerate=eval_force_regenerate,
        num_locations=num_locations,
        area_size=area_size,
        num_uavs=num_uavs,
        num_requests=num_requests,
    )
    print(
        f"[MPopLoc] eval dataset ready: episodes={len(eval_dataset)} seed={eval_seed} dir={eval_data_dir}",
        flush=True,
    )

    train_output_dir = os.path.abspath(train_data_dir)
    eval_output_dir = os.path.abspath(eval_data_dir)
    train_dataset_stem = _dataset_stem(
        train_seed,
        train_eps,
        num_locations,
        area_size,
        num_uavs,
        num_requests,
    )
    eval_dataset_stem = _dataset_stem(
        eval_seed,
        eval_eps,
        num_locations,
        area_size,
        num_uavs,
        num_requests,
    )
    train_metrics_path = os.path.join(train_output_dir, f"MpopLoc_{train_dataset_stem}.csv")
    eval_metrics_path = os.path.join(eval_output_dir, f"MpopLoc_{eval_dataset_stem}.csv")

    print(
        f"[MPopLoc] saving train results to: {train_metrics_path}\n"
        f"[MPopLoc] saving eval results to: {eval_metrics_path}",
        flush=True,
    )

    train_summary = None
    eval_summary = None
    if os.path.exists(train_metrics_path):
        print(f"[MPopLoc] skip train: already exists -> {train_metrics_path}", flush=True)
    else:
        _, train_summary, _ = _evaluate_split("train", train_dataset, num_time_slots, train_metrics_path)

    if os.path.exists(eval_metrics_path):
        print(f"[MPopLoc] skip eval: already exists -> {eval_metrics_path}", flush=True)
    else:
        _, eval_summary, _ = _evaluate_split("eval", eval_dataset, num_time_slots, eval_metrics_path)

    if train_summary is not None:
        print(f"[MPopLoc] train_success={train_summary['avg_success_rate']:.4f}", flush=True)
    if eval_summary is not None:
        print(f"[MPopLoc] eval_success={eval_summary['avg_success_rate']:.4f}", flush=True)

    print(f"[MPopLoc] train_metrics={train_metrics_path}", flush=True)
    print(f"[MPopLoc] eval_metrics={eval_metrics_path}", flush=True)


if __name__ == "__main__":
    main()
