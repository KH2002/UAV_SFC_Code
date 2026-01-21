# -*- coding: utf-8 -*-
"""
为现有的结果文件生成一份副本：
- 保留 MILP 相关数据不变
- 对 MPopLoc 和 RandomOrder 算法进行多次随机实验，取平均值后写回

默认处理 results 目录下的三个分类子文件夹 (vary_uav_sfc / vary_requests / vary_area)。
新的平均结果将保存为 detailed_results_avg_YYYYMMDD_HHMMSS.csv，存放在原子目录中。
"""
import os
import glob
from datetime import datetime
from typing import Dict, Tuple, List

import pandas as pd

import config
from test_algorithms import (
    setup_scenario,
    run_mpoploc_test,
    run_random_test,
)

RESULTS_ROOT = "results"
CATEGORIES = ["vary_uav_sfc", "vary_requests", "vary_area"]
NUM_TRIALS = 20

MP_KEYS = ("success", "total_serviced", "total_reward", "runtime")
RAND_KEYS = MP_KEYS

MP_COLUMNS = [
    "mpoploc_success",
    "mpoploc_serviced",
    "mpoploc_reward",
    "mpoploc_runtime",
    "mpoploc_status",
]
RAND_COLUMNS = [
    "random_success",
    "random_serviced",
    "random_reward",
    "random_runtime",
    "random_status",
]


def find_latest_csv(folder: str) -> str:
    pattern = os.path.join(folder, "detailed_results_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"未在 {folder} 中找到 detailed_results_*.csv")
    return max(files, key=os.path.getmtime)


def aggregate(results: List[Dict[str, float]]) -> Dict[str, float]:
    if not results:
        return {
            "success": 0.0,
            "total_serviced": 0.0,
            "total_reward": 0.0,
            "runtime": 0.0,
        }
    count = len(results)
    success_rate = sum(1.0 if r["success"] else 0.0 for r in results) / count
    total_serviced = sum(r["total_serviced"] for r in results) / count
    total_reward = sum(r["total_reward"] for r in results) / count
    avg_runtime = sum(r["runtime"] for r in results) / count
    return {
        "success": success_rate,
        "total_serviced": total_serviced,
        "total_reward": total_reward,
        "runtime": avg_runtime,
    }


def run_trials_for_case(row: pd.Series) -> Tuple[Dict[str, float], Dict[str, float]]:
    num_uavs = int(row["num_uavs"])
    num_requests = int(row["num_requests"])
    area_size = float(row["area_size"])
    num_locations = int(row.get("num_locations", config.NUM_LOCATIONS))
    vnfs_per_request = int(row.get("vnfs_per_request", config.VNFS_PER_REQUEST))
    time_slots = int(row.get("time_slots", config.NUM_TIME_SLOTS))
    base_seed = int(row.get("seed", 0))

    # 备份需要修改的 config 值
    original_config = {
        "NUM_UAVS": config.NUM_UAVS,
        "AREA_SIZE": config.AREA_SIZE,
        "BASE_STATION_LOCATION": config.BASE_STATION_LOCATION,
        "NUM_LOCATIONS": config.NUM_LOCATIONS,
        "VNFS_PER_REQUEST": config.VNFS_PER_REQUEST,
        "NUM_TIME_SLOTS": config.NUM_TIME_SLOTS,
        "NUM_REQUESTS": getattr(config, "NUM_REQUESTS", None),
    }

    config.NUM_UAVS = num_uavs
    config.AREA_SIZE = area_size
    config.BASE_STATION_LOCATION = (area_size / 2.0, area_size / 2.0)
    config.NUM_LOCATIONS = num_locations
    config.VNFS_PER_REQUEST = vnfs_per_request
    config.NUM_TIME_SLOTS = time_slots
    if hasattr(config, "NUM_REQUESTS"):
        config.NUM_REQUESTS = num_requests

    mp_results: List[Dict[str, float]] = []
    rand_results: List[Dict[str, float]] = []

    try:
        for trial_idx in range(NUM_TRIALS):
            seed = base_seed + trial_idx
            uavs, requests, locations = setup_scenario(
                num_uavs=num_uavs,
                num_requests=num_requests,
                area_size=area_size,
                num_locations=num_locations,
                vnfs_per_request=vnfs_per_request,
                seed=seed,
            )

            mp_results.append(run_mpoploc_test(uavs, requests, locations))
            rand_results.append(run_random_test(uavs, requests, locations))
    finally:
        # 恢复 config
        config.NUM_UAVS = original_config["NUM_UAVS"]
        config.AREA_SIZE = original_config["AREA_SIZE"]
        config.BASE_STATION_LOCATION = original_config["BASE_STATION_LOCATION"]
        config.NUM_LOCATIONS = original_config["NUM_LOCATIONS"]
        config.VNFS_PER_REQUEST = original_config["VNFS_PER_REQUEST"]
        config.NUM_TIME_SLOTS = original_config["NUM_TIME_SLOTS"]
        if original_config["NUM_REQUESTS"] is not None:
            config.NUM_REQUESTS = original_config["NUM_REQUESTS"]

    mp_agg = aggregate(mp_results)
    rand_agg = aggregate(rand_results)

    return mp_agg, rand_agg


def update_row_with_averages(row: pd.Series, mp_avg: Dict[str, float], rand_avg: Dict[str, float]) -> pd.Series:
    new_row = row.copy()

    # 更新 MPopLoc 数据
    new_row["mpoploc_success"] = mp_avg["success"]
    new_row["mpoploc_serviced"] = mp_avg["total_serviced"]
    new_row["mpoploc_reward"] = mp_avg["total_reward"]
    new_row["mpoploc_runtime"] = mp_avg["runtime"]
    new_row["mpoploc_status"] = "avg_20_runs"

    # 更新 RandomOrder 数据
    new_row["random_success"] = rand_avg["success"]
    new_row["random_serviced"] = rand_avg["total_serviced"]
    new_row["random_reward"] = rand_avg["total_reward"]
    new_row["random_runtime"] = rand_avg["runtime"]
    new_row["random_status"] = "avg_20_runs"

    # 重新计算比较指标
    milp_serviced = float(new_row["milp_serviced"])
    milp_reward = float(new_row["milp_reward"])
    milp_runtime = float(new_row["milp_runtime"])

    mp_serviced = new_row["mpoploc_serviced"]
    mp_reward = new_row["mpoploc_reward"]
    mp_runtime = new_row["mpoploc_runtime"]

    rand_serviced = new_row["random_serviced"]
    rand_reward = new_row["random_reward"]
    rand_runtime = new_row["random_runtime"]

    new_row["serviced_diff"] = milp_serviced - mp_serviced
    new_row["reward_diff"] = milp_reward - mp_reward
    new_row["runtime_ratio"] = mp_runtime / milp_runtime if milp_runtime > 0 else 0.0

    new_row["milp_vs_random_serviced_diff"] = milp_serviced - rand_serviced
    new_row["mpoploc_vs_random_serviced_diff"] = mp_serviced - rand_serviced
    new_row["milp_vs_random_reward_diff"] = milp_reward - rand_reward
    new_row["mpoploc_vs_random_reward_diff"] = mp_reward - rand_reward

    new_row["random_vs_milp_runtime_ratio"] = rand_runtime / milp_runtime if milp_runtime > 0 else 0.0
    new_row["random_vs_mpoploc_runtime_ratio"] = rand_runtime / mp_runtime if mp_runtime > 0 else 0.0

    return new_row


def process_category(category: str):
    folder = os.path.join(RESULTS_ROOT, category)
    csv_path = find_latest_csv(folder)
    df = pd.read_csv(csv_path)

    updated_rows = []
    for idx, row in df.iterrows():
        print(f"[{category}] 重新计算案例 {idx + 1}/{len(df)} ...")
        mp_avg, rand_avg = run_trials_for_case(row)
        updated_rows.append(update_row_with_averages(row, mp_avg, rand_avg))

    df_updated = pd.DataFrame(updated_rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(folder, f"detailed_results_avg_{timestamp}.csv")
    df_updated.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[{category}] 平均结果已保存到: {output_path}")


def main():
    for category in CATEGORIES:
        try:
            process_category(category)
        except FileNotFoundError as exc:
            print(exc)
        except Exception as exc:
            print(f"处理 {category} 时发生错误: {exc}")


if __name__ == "__main__":
    main()
