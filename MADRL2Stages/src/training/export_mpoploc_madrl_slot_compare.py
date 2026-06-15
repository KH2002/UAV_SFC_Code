# -*- coding: utf-8 -*-
"""Export per-slot UAV locations and deployed VNF ids for MPopLoc and MADRL+RULE."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as env_config
from DRL.training.dataset import create_or_load_dataset
from mpoploc import MPopLocSolver


def _load_yaml(path: str) -> Dict[str, object]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_test_settings(config_path: str, args: argparse.Namespace) -> Dict[str, object]:
    cfg = _load_yaml(config_path)
    scene_cfg = cfg.get("scene", {}) if isinstance(cfg.get("scene"), dict) else {}
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
    eval_cfg = dataset_cfg.get("eval", {}) if isinstance(dataset_cfg.get("eval"), dict) else {}
    test_cfg = dataset_cfg.get("test", {}) if isinstance(dataset_cfg.get("test"), dict) else {}

    dataset_base_seed = int(dataset_cfg.get("base_seed", 42))
    dataset_num_episodes = int(dataset_cfg.get("num_episodes", 100))

    return {
        "num_locations": int(scene_cfg.get("num_locations", env_config.NUM_LOCATIONS)),
        "area_size": float(scene_cfg.get("area_size", env_config.AREA_SIZE)),
        "num_uavs": int(scene_cfg.get("num_uavs", env_config.NUM_UAVS)),
        "num_requests": int(scene_cfg.get("num_requests", env_config.NUM_REQUESTS)),
        "num_time_slots": int(scene_cfg.get("num_time_slots", env_config.NUM_TIME_SLOTS)),
        "test_base_seed": int(
            args.test_base_seed
            if args.test_base_seed is not None
            else test_cfg.get("base_seed", eval_cfg.get("base_seed", dataset_base_seed + 200000))
        ),
        "test_num_episodes": int(
            args.num_test_episodes
            if args.num_test_episodes is not None
            else test_cfg.get("num_episodes", eval_cfg.get("num_episodes", dataset_num_episodes))
        ),
        "test_data_dir": str(args.test_data_dir or test_cfg.get("data_dir", "./test_data")),
    }


def _deployed_vnf_ids_from_json(value: str) -> List[int]:
    if not value:
        return []
    try:
        items = json.loads(value)
    except json.JSONDecodeError:
        return []
    ids = []
    for item in items:
        if isinstance(item, dict) and item.get("vnf_id") is not None:
            ids.append(int(item["vnf_id"]))
        else:
            ids.append(int(item))
    return ids


def _read_madrl_rows(slot_csv_path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(slot_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "algorithm": "MADRL+RULE",
                    "episode_index": int(float(row["episode_index"])),
                    "slot_id": int(float(row["slot_id"])),
                    "uav_id": int(float(row["uav_id"])),
                    "location_id": int(float(row.get("end_location_id") or row.get("selected_location_id") or 0)),
                    "deployed_vnf_ids": _deployed_vnf_ids_from_json(row.get("deployed_vnf_ids_json", "")),
                }
            )
    return rows


def _run_mpoploc_rows(settings: Dict[str, object]) -> List[Dict[str, object]]:
    dataset = create_or_load_dataset(
        base_seed=int(settings["test_base_seed"]),
        num_episodes=int(settings["test_num_episodes"]),
        data_dir=str(settings["test_data_dir"]),
        force_regenerate=False,
        num_locations=int(settings["num_locations"]),
        area_size=float(settings["area_size"]),
        num_uavs=int(settings["num_uavs"]),
        num_requests=int(settings["num_requests"]),
    )

    rows: List[Dict[str, object]] = []
    for ep_idx in range(len(dataset)):
        ep = dataset[ep_idx]
        ep_copy = ep.copy()
        solver = MPopLocSolver(
            ep_copy.uavs,
            ep_copy.requests,
            ep_copy.locations,
            num_time_slots=int(settings["num_time_slots"]),
        )
        solver.solve()

        for slot_id, uav_map in sorted(solver.slot_uav_details.items()):
            for uav_id, detail in sorted(uav_map.items()):
                rows.append(
                    {
                        "algorithm": "MPopLoc",
                        "episode_index": int(ep_idx),
                        "slot_id": int(slot_id),
                        "uav_id": int(uav_id),
                        "location_id": int(detail["location_id"]),
                        "deployed_vnf_ids": [int(v) for v in detail.get("deployed_vnf_ids", [])],
                    }
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MPopLoc and MADRL+RULE slot-level UAV locations/VNF deployments.")
    parser.add_argument("--config", type=str, default=os.path.join("MADRL+RULE", "config_small.yaml"))
    parser.add_argument("--madrl-slot-csv", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--test-data-dir", type=str, default=None)
    parser.add_argument("--test-base-seed", type=int, default=None)
    parser.add_argument("--num-test-episodes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = _load_test_settings(args.config, args)
    madrl_rows = _read_madrl_rows(args.madrl_slot_csv)
    mpoploc_rows = _run_mpoploc_rows(settings)

    out_path = args.output
    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.abspath(args.madrl_slot_csv)), "mpoploc_madrl_slot_compare.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["algorithm", "episode_index", "slot_id", "uav_id", "location_id", "deployed_vnf_ids"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in mpoploc_rows + madrl_rows:
            writer.writerow(
                {
                    **row,
                    "deployed_vnf_ids": json.dumps(row["deployed_vnf_ids"], ensure_ascii=False),
                }
            )

    print(f"[COMPARE] output={out_path}")
    print(f"[COMPARE] mpoploc_rows={len(mpoploc_rows)} madrl_rows={len(madrl_rows)}")


if __name__ == "__main__":
    main()
