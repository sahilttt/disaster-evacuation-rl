"""
Reusable Matplotlib plotting utilities for disaster-evacuation RL experiments.

All functions accept pre-computed data (lists / dicts) and save figures to
disk.  They do **not** call ``plt.show()`` so they work headless.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (works without a display)
import matplotlib.pyplot as plt
import numpy as np


# ── Style defaults ────────────────────────────────────────────────────
_COLOURS = {
    "q_learning": "#2196F3",
    "dqn_static": "#4CAF50",
    "dqn_dynamic": "#FF9800",
}

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


# ── Smoothing helper ─────────────────────────────────────────────────

def _rolling_mean(values: List[float], window: int = 50) -> np.ndarray:
    """Compute a centred rolling mean with edge padding."""
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        window = max(1, len(arr))
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


# =====================================================================
# 1.  Training reward vs episode
# =====================================================================

def plot_training_reward(
    history: List[Dict[str, Any]],
    title: str = "Training Reward per Episode",
    save_path: str = "training_reward.png",
    colour: str = _COLOURS["dqn_dynamic"],
    smooth_window: int = 50,
) -> str:
    """Line plot of per-episode total reward with smoothed overlay.

    Args:
        history: Per-episode dicts with ``total_reward``.
        title: Figure title.
        save_path: Destination file.
        colour: Line colour.
        smooth_window: Rolling-average window.

    Returns:
        Absolute path to the saved figure.
    """
    rewards = [r["total_reward"] for r in history]
    episodes = list(range(1, len(rewards) + 1))
    smoothed = _rolling_mean(rewards, smooth_window)

    fig, ax = plt.subplots()
    ax.plot(episodes, rewards, alpha=0.25, color=colour, linewidth=0.6)
    ax.plot(episodes, smoothed, color=colour, linewidth=2,
            label=f"Rolling avg ({smooth_window} eps)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    _ensure_dir(save_path)
    fig.savefig(save_path)
    plt.close(fig)
    return os.path.abspath(save_path)


# =====================================================================
# 2.  Training success rate vs episode
# =====================================================================

def plot_training_success_rate(
    history: List[Dict[str, Any]],
    title: str = "Training Success Rate (rolling)",
    save_path: str = "training_success.png",
    colour: str = _COLOURS["dqn_dynamic"],
    smooth_window: int = 50,
) -> str:
    """Rolling success-rate curve.

    Args:
        history: Per-episode dicts with ``success`` (bool).
        title: Figure title.
        save_path: Destination file.
        colour: Line colour.
        smooth_window: Rolling-average window.

    Returns:
        Absolute path to the saved figure.
    """
    successes = [float(r["success"]) for r in history]
    episodes = list(range(1, len(successes) + 1))
    smoothed = _rolling_mean(successes, smooth_window) * 100  # percent

    fig, ax = plt.subplots()
    ax.plot(episodes, smoothed, color=colour, linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(-5, 105)
    ax.set_title(title)
    fig.tight_layout()

    _ensure_dir(save_path)
    fig.savefig(save_path)
    plt.close(fig)
    return os.path.abspath(save_path)


# =====================================================================
# 3.  Q-Learning vs DQN evaluation comparison (grouped bar chart)
# =====================================================================

def plot_eval_comparison(
    results: Dict[str, Dict[str, float]],
    title: str = "Q-Learning vs DQN — Evaluation",
    save_path: str = "eval_comparison.png",
) -> str:
    """Grouped bar chart comparing evaluation metrics across agents.

    Args:
        results: Mapping ``agent_name -> {metric: value}``.  Expected
            metrics: ``success_rate``, ``avg_reward``, ``avg_steps``,
            ``avg_path_length``.
        title: Figure title.
        save_path: Destination file.

    Returns:
        Absolute path to the saved figure.
    """
    agents = list(results.keys())
    metrics = ["success_rate", "avg_reward", "avg_steps", "avg_path_length"]
    labels = ["Success Rate (%)", "Avg Reward", "Avg Steps", "Avg Path Len"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5))

    colours = [_COLOURS.get(a, "#9E9E9E") for a in agents]
    display_names = [a.replace("_", " ").title() for a in agents]

    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[idx]
        values = []
        for a in agents:
            v = results[a].get(metric, 0.0)
            if metric == "success_rate":
                v *= 100  # convert to percent
            values.append(v)

        bars = ax.bar(display_names, values, color=colours, width=0.5)
        ax.set_ylabel(label)
        ax.set_title(label)

        # Value labels on bars.
        for bar, val in zip(bars, values):
            fmt = f"{val:.1f}" if abs(val) < 1000 else f"{val:.0f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt, ha="center", va="bottom", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    _ensure_dir(save_path)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(save_path)


# =====================================================================
# 4.  Static vs Dynamic DQN comparison
# =====================================================================

def plot_static_vs_dynamic(
    static_history: List[Dict[str, Any]],
    dynamic_history: List[Dict[str, Any]],
    static_eval: Dict[str, float],
    dynamic_eval: Dict[str, float],
    save_path: str = "static_vs_dynamic.png",
) -> str:
    """Side-by-side comparison of static and dynamic DQN.

    Top row: training reward + success curves.
    Bottom row: evaluation bar charts.

    Args:
        static_history: Static DQN per-episode training data.
        dynamic_history: Dynamic DQN per-episode training data.
        static_eval: Static DQN evaluation metrics.
        dynamic_eval: Dynamic DQN evaluation metrics.
        save_path: Destination file.

    Returns:
        Absolute path to the saved figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sw = 50  # smooth window

    # ── Row 0, Col 0: Training reward ─────────────────────────────────
    ax = axes[0, 0]
    for hist, label, clr in [
        (static_history, "Static DQN", _COLOURS["dqn_static"]),
        (dynamic_history, "Dynamic DQN", _COLOURS["dqn_dynamic"]),
    ]:
        rewards = [r["total_reward"] for r in hist]
        episodes = list(range(1, len(rewards) + 1))
        smoothed = _rolling_mean(rewards, sw)
        ax.plot(episodes, smoothed, label=label, color=clr, linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward (smoothed)")
    ax.set_title("Training Reward")
    ax.legend()

    # ── Row 0, Col 1: Training success rate ───────────────────────────
    ax = axes[0, 1]
    for hist, label, clr in [
        (static_history, "Static DQN", _COLOURS["dqn_static"]),
        (dynamic_history, "Dynamic DQN", _COLOURS["dqn_dynamic"]),
    ]:
        succ = [float(r["success"]) for r in hist]
        episodes = list(range(1, len(succ) + 1))
        smoothed = _rolling_mean(succ, sw) * 100
        ax.plot(episodes, smoothed, label=label, color=clr, linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("Training Success Rate")
    ax.legend()

    # ── Row 1, Col 0: Eval success rate ───────────────────────────────
    ax = axes[1, 0]
    names = ["Static DQN", "Dynamic DQN"]
    sr_vals = [static_eval["success_rate"] * 100,
               dynamic_eval["success_rate"] * 100]
    bars = ax.bar(names, sr_vals,
                  color=[_COLOURS["dqn_static"], _COLOURS["dqn_dynamic"]],
                  width=0.45)
    for bar, val in zip(bars, sr_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Evaluation Success Rate")

    # ── Row 1, Col 1: Eval avg reward & path length ───────────────────
    ax = axes[1, 1]
    metrics = ["avg_reward", "avg_steps", "avg_path_length"]
    x = np.arange(len(metrics))
    width = 0.3
    for i, (evl, label, clr) in enumerate([
        (static_eval, "Static DQN", _COLOURS["dqn_static"]),
        (dynamic_eval, "Dynamic DQN", _COLOURS["dqn_dynamic"]),
    ]):
        vals = [evl[m] for m in metrics]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=clr)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(), f"{val:.1f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Avg Reward", "Avg Steps", "Avg Path Len"])
    ax.set_title("Evaluation Metrics")
    ax.legend()

    fig.suptitle("Static vs Dynamic DQN", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    _ensure_dir(save_path)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(save_path)
