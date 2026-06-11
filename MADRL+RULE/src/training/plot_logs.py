#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot MAPPO training metrics from one or multiple log directories."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(y, kernel, mode="valid")


def _series_for_plot(x: np.ndarray, y: np.ndarray, smooth_window: int) -> tuple[np.ndarray, np.ndarray]:
    y_s = _smooth(y, smooth_window)
    if len(y_s) == len(y):
        return x, y_s
    return x[: len(y_s)], y_s


def _read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    return df


def plot_success_rate(runs: List[Dict[str, object]], out_path: str, smooth: int, show_raw: bool) -> None:
    plt.figure(figsize=(9, 5.5))
    for run in runs:
        episode_df = run["episode_df"]
        label = str(run["label"])
        x = episode_df["episode"].to_numpy(dtype=float)
        y = episode_df["success_rate"].to_numpy(dtype=float)
        xs, ys = _series_for_plot(x, y, smooth)
        if show_raw:
            plt.plot(x, y, alpha=0.15)
        plt.plot(xs, ys, linewidth=2, label=label)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.title("Success Rate vs Episode")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_avg_reward(runs: List[Dict[str, object]], out_path: str, smooth: int, show_raw: bool) -> None:
    plt.figure(figsize=(9, 5.5))
    for run in runs:
        episode_df = run["episode_df"]
        label = str(run["label"])
        x = episode_df["episode"].to_numpy(dtype=float)
        y = episode_df["avg_reward"].to_numpy(dtype=float)
        xs, ys = _series_for_plot(x, y, smooth)
        if show_raw:
            plt.plot(x, y, alpha=0.15)
        plt.plot(xs, ys, linewidth=2, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.title("Average Reward vs Episode")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_losses(runs: List[Dict[str, object]], out_path: str, smooth: int) -> None:
    plt.figure(figsize=(9, 5.5))
    for run in runs:
        update_df = run["update_df"]
        label = str(run["label"])
        x = update_df["update"].to_numpy(dtype=float)
        loss = update_df["loss"].to_numpy(dtype=float)
        pi = update_df["policy_loss"].to_numpy(dtype=float)
        v = update_df["value_loss"].to_numpy(dtype=float)

        xl, yl = _series_for_plot(x, loss, smooth)
        xp, yp = _series_for_plot(x, pi, smooth)
        xv, yv = _series_for_plot(x, v, smooth)

        plt.plot(xl, yl, linewidth=2, label=f"{label}: loss")
        plt.plot(xp, yp, linewidth=2, linestyle="--", label=f"{label}: policy")
        plt.plot(xv, yv, linewidth=2, linestyle=":", label=f"{label}: value")
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_health(runs: List[Dict[str, object]], out_path: str, smooth: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    has_step_metrics = False
    for run in runs:
        update_df = run["update_df"]
        label = str(run["label"])
        x_u = update_df["update"].to_numpy(dtype=float)
        ent = update_df["entropy"].to_numpy(dtype=float)
        kl = update_df["kl_div"].to_numpy(dtype=float)
        xu_e, yu_e = _series_for_plot(x_u, ent, smooth)
        xu_k, yu_k = _series_for_plot(x_u, kl, smooth)
        axes[0, 0].plot(xu_e, yu_e, linewidth=2, label=label)
        axes[0, 1].plot(xu_k, yu_k, linewidth=2, label=label)

    axes[0, 0].set_title("Entropy")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].set_title("KL Divergence")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    for run in runs:
        step_df = run["step_df"]
        if step_df is None:
            continue
        has_step_metrics = True
        label = str(run["label"])
        x_s = step_df["env_step"].to_numpy(dtype=float)
        mask_d = step_df["action_mask_density"].to_numpy(dtype=float)
        invalid = step_df["invalid_count"].to_numpy(dtype=float)
        xs_m, ys_m = _series_for_plot(x_s, mask_d, smooth)
        xs_i, ys_i = _series_for_plot(x_s, invalid, smooth)
        axes[1, 0].plot(xs_m, ys_m, linewidth=2, label=label)
        axes[1, 1].plot(xs_i, ys_i, linewidth=2, label=label)

    if has_step_metrics:
        axes[1, 0].set_title("Action Mask Density")
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].legend()

        axes[1, 1].set_title("Invalid Action Count")
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].legend()
    else:
        axes[1, 0].text(0.5, 0.5, "step_metrics.csv not found", ha="center", va="center")
        axes[1, 0].set_axis_off()
        axes[1, 1].set_axis_off()

    for ax in axes.flat:
        if ax.has_data():
            ax.set_xlabel("Step/Update")

    fig.suptitle("Training Health Metrics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _load_run(log_dir: str, label: str) -> Dict[str, object]:
    episode_path = os.path.join(log_dir, "episode_metrics.csv")
    update_path = os.path.join(log_dir, "update_metrics.csv")
    step_path = os.path.join(log_dir, "step_metrics.csv")
    train_path = os.path.join(log_dir, "training_log.csv")

    episode_df = _read_csv(episode_path)
    if episode_df is None:
        fallback_df = _read_csv(train_path)
        if fallback_df is not None and {"episode", "success_rate", "reward"}.issubset(fallback_df.columns):
            episode_df = fallback_df.rename(columns={"reward": "avg_reward"})
    update_df = _read_csv(update_path)
    step_df = _read_csv(step_path)

    if episode_df is None:
        raise FileNotFoundError(f"No episode metrics found in {log_dir}")
    if update_df is None:
        raise FileNotFoundError(f"No update metrics found in {log_dir}")
    if "avg_reward" not in episode_df.columns:
        raise ValueError(f"avg_reward column not found in episode metrics: {log_dir}")

    return {
        "log_dir": log_dir,
        "label": label,
        "episode_df": episode_df,
        "update_df": update_df,
        "step_df": step_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MAPPO logs from one or multiple log folders.")
    parser.add_argument("--log-dir", type=str, default=None, help="Path to one MAPPO log directory.")
    parser.add_argument(
        "--log-dirs",
        type=str,
        nargs="+",
        default=None,
        help="Paths to multiple MAPPO log directories for comparison.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Optional labels for --log-dirs (same count).",
    )
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for figures.")
    parser.add_argument("--smooth", type=int, default=15, help="Smoothing window size.")
    parser.add_argument("--show-raw", action="store_true", help="Overlay raw curves in comparison plots.")
    args = parser.parse_args()

    log_dirs: List[str] = []
    if args.log_dirs:
        log_dirs.extend(args.log_dirs)
    if args.log_dir:
        log_dirs.append(args.log_dir)
    if not log_dirs:
        raise ValueError("Provide --log-dir or --log-dirs.")

    log_dirs = [os.path.abspath(p) for p in log_dirs]
    labels = args.labels if args.labels is not None else [os.path.basename(p.rstrip(os.sep)) for p in log_dirs]
    if len(labels) != len(log_dirs):
        raise ValueError("When --labels is provided, it must match the number of log directories.")

    default_base = log_dirs[0] if len(log_dirs) == 1 else os.path.commonpath(log_dirs)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(default_base, "plots_compare")
    os.makedirs(out_dir, exist_ok=True)
    runs = [_load_run(ld, lb) for ld, lb in zip(log_dirs, labels)]
    show_raw = bool(args.show_raw) if len(runs) > 1 else True

    p1 = os.path.join(out_dir, "1_success_rate_vs_episode.png")
    p2 = os.path.join(out_dir, "2_avg_reward_vs_episode.png")
    p3 = os.path.join(out_dir, "3_loss_curves.png")
    p4 = os.path.join(out_dir, "4_health_metrics.png")

    plot_success_rate(runs, p1, args.smooth, show_raw=show_raw)
    plot_avg_reward(runs, p2, args.smooth, show_raw=show_raw)
    plot_losses(runs, p3, args.smooth)
    plot_health(runs, p4, args.smooth)

    print(f"[OK] Saved: {p1}")
    print(f"[OK] Saved: {p2}")
    print(f"[OK] Saved: {p3}")
    print(f"[OK] Saved: {p4}")


if __name__ == "__main__":
    main()
