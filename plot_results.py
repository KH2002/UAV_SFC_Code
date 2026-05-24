# -*- coding: utf-8 -*-
"""
读取 results 目录下三个子类文件夹的测试结果，
并绘制三张折线图（分别对应 vary_uav_sfc、vary_requests、vary_area）。
横轴为各类测试的变量，纵轴为部署成功的 SFC 数量。
输出图片保存到 results/plots 目录。
"""
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# 使 matplotlib 能够正确显示中文和负号（直接覆盖默认设置）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "STSong", "STFangsong", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False

RESULTS_ROOT = "results"
OUTPUT_DIR = os.path.join(RESULTS_ROOT, "plots")

CATEGORIES = {
    "vary_uav_sfc": {
        "x_col": "num_uavs",
        "xlabel": "UAV 数量 (按比例调整 SFC 数量)",
        "title": "UAV 与 SFC 等比例变化下的部署数量"
    },
    "vary_requests": {
        "x_col": "num_requests",
        "xlabel": "SFC 请求数量",
        "title": "不同 SFC 请求数量下的部署数量"
    },
    "vary_area": {
        "x_col": "area_size",
        "xlabel": "区域大小 (m)",
        "title": "不同区域大小下的部署数量"
    }
}

ALGORITHMS = [
    ("milp_serviced", "MILP"),
    ("mpoploc_serviced", "MPopLoc"),
    ("random_serviced", "RandomOrder")
]


def find_latest_csv(folder: str) -> str:
    """在指定文件夹中查找最新的 detailed_results_*.csv 文件。"""
    pattern = os.path.join(folder, "detailed_results_avg*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"未在 {folder} 中找到 {os.path.basename(pattern)}")
    latest = max(files, key=os.path.getmtime)
    return latest


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_category(category: str, info: dict):
    folder = os.path.join(RESULTS_ROOT, category)
    csv_path = find_latest_csv(folder)
    df = pd.read_csv(csv_path)

    x_col = info["x_col"]
    xlabel = info["xlabel"]
    title = info["title"]

    # 对多次运行的结果取平均值
    grouped = df.groupby(x_col)[[col for col, _ in ALGORITHMS]].mean().sort_index()

    plt.figure(figsize=(8, 5))
    for col, label in ALGORITHMS:
        if col not in grouped.columns:
            continue
        plt.plot(grouped.index, grouped[col], marker="o", label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("部署成功的 SFC 数量")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f"{category}_sfc_served.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"已保存: {output_path}")


def main():
    ensure_output_dir()
    for category, info in CATEGORIES.items():
        try:
            plot_category(category, info)
        except FileNotFoundError as exc:
            print(exc)
        except Exception as exc:  # 简明提示其他异常
            print(f"处理 {category} 时发生错误: {exc}")


if __name__ == "__main__":
    main()
