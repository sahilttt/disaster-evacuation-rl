"""
Experiment analysis — generate all plots and a summary results file.

This script re-runs the static comparison and dynamic experiments so that
episode-level histories are available, then produces four visualisation
artefacts and a consolidated JSON + CSV summary.

Usage:
    python -m experiments.analyze_results
    python experiments/analyze_results.py
"""

import csv
import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is on the path when run directly.
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.helpers import load_history, save_json, load_json

# Re-use the existing experiment runners (they persist histories).
from experiments.compare_agents import main as run_static_comparison
from experiments.train_dynamic import main as run_dynamic_experiment

from visualization.plot_results import (
    plot_training_reward,
    plot_training_success_rate,
    plot_eval_comparison,
    plot_static_vs_dynamic,
)


# ── Paths ─────────────────────────────────────────────────────────────
_RESULTS = Path(__file__).resolve().parent / "results"
_PLOTS = _RESULTS / "plots"


def _history_path(name: str) -> Path:
    return _RESULTS / f"{name}_history.json"


def _needs_run(*paths: Path) -> bool:
    """Return True if any of the expected files are missing."""
    return any(not p.exists() for p in paths)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 60)
    print("  Experiment Analysis & Visualization")
    print("=" * 60)

    # ── 1.  Ensure data exists ────────────────────────────────────────
    static_paths = (
        _history_path("q_learning"),
        _history_path("dqn_static"),
        _RESULTS / "static_eval_results.json",
    )
    if _needs_run(*static_paths):
        print("\n  [1/2] Running static comparison experiment ...")
        run_static_comparison()
    else:
        print("\n  [1/2] Static data found -- loading from disk.")

    dynamic_paths = (
        _history_path("dqn_dynamic"),
        _RESULTS / "dynamic_eval_results.json",
    )
    if _needs_run(*dynamic_paths):
        print("  [2/2] Running dynamic DQN experiment ...")
        run_dynamic_experiment()
    else:
        print("  [2/2] Dynamic data found -- loading from disk.")

    # ── 2.  Load data ─────────────────────────────────────────────────
    q_history = load_history(str(_history_path("q_learning")))
    dqn_static_history = load_history(str(_history_path("dqn_static")))
    dqn_dynamic_history = load_history(str(_history_path("dqn_dynamic")))

    static_eval = load_json(str(_RESULTS / "static_eval_results.json"))
    dynamic_eval = load_json(str(_RESULTS / "dynamic_eval_results.json"))

    q_eval = static_eval["q_learning"]
    dqn_s_eval = static_eval["dqn_static"]
    dqn_d_eval = dynamic_eval  # already flat dict

    # ── 3.  Generate plots ────────────────────────────────────────────
    _PLOTS.mkdir(parents=True, exist_ok=True)
    plots_dir = str(_PLOTS)

    print("\n  Generating plots ...")

    p1 = plot_training_reward(
        dqn_dynamic_history,
        title="Dynamic DQN — Training Reward per Episode",
        save_path=f"{plots_dir}/dqn_dynamic_training_reward.png",
    )
    print(f"    [OK] {Path(p1).name}")

    p2 = plot_training_success_rate(
        dqn_dynamic_history,
        title="Dynamic DQN — Training Success Rate",
        save_path=f"{plots_dir}/dqn_dynamic_training_success.png",
    )
    print(f"    [OK] {Path(p2).name}")

    p3 = plot_eval_comparison(
        {"q_learning": q_eval, "dqn_static": dqn_s_eval},
        title="Q-Learning vs DQN (Static) — Evaluation",
        save_path=f"{plots_dir}/q_vs_dqn_eval_comparison.png",
    )
    print(f"    [OK] {Path(p3).name}")

    p4 = plot_static_vs_dynamic(
        dqn_static_history,
        dqn_dynamic_history,
        dqn_s_eval,
        dqn_d_eval,
        save_path=f"{plots_dir}/static_vs_dynamic_dqn.png",
    )
    print(f"    [OK] {Path(p4).name}")

    # ── 4.  Consolidated summary JSON ─────────────────────────────────
    def _last_n_sr(hist, n=100):
        tail = hist[-n:]
        return sum(1 for r in tail if r["success"]) / len(tail) if tail else 0.0

    summary = {
        "generated_at": datetime.now().isoformat(),
        "q_learning": {
            "training_episodes": len(q_history),
            "training_success_rate": sum(1 for r in q_history if r["success"]) / len(q_history),
            "last_100_train_sr": _last_n_sr(q_history),
            "evaluation": q_eval,
        },
        "dqn_static": {
            "training_episodes": len(dqn_static_history),
            "training_success_rate": sum(1 for r in dqn_static_history if r["success"]) / len(dqn_static_history),
            "last_100_train_sr": _last_n_sr(dqn_static_history),
            "evaluation": dqn_s_eval,
        },
        "dqn_dynamic": {
            "training_episodes": len(dqn_dynamic_history),
            "training_success_rate": sum(1 for r in dqn_dynamic_history if r["success"]) / len(dqn_dynamic_history),
            "last_100_train_sr": _last_n_sr(dqn_dynamic_history),
            "evaluation": dqn_d_eval,
        },
    }

    summary_json = str(_RESULTS / "experiment_summary.json")
    save_json(summary, summary_json)
    print(f"\n    [OK] {Path(summary_json).name}")

    # ── 5.  CSV results table ─────────────────────────────────────────
    csv_path = str(_RESULTS / "experiment_summary.csv")
    _write_csv(summary, csv_path)
    print(f"    [OK] {Path(csv_path).name}")

    # ── 6.  Console summary ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    fmt = "  {:<24} {:>14} {:>14} {:>14}"
    print(fmt.format("Metric", "Q-Learning", "DQN Static", "DQN Dynamic"))
    print(fmt.format("-" * 24, "-" * 14, "-" * 14, "-" * 14))
    print(fmt.format("Train episodes",
                     str(len(q_history)),
                     str(len(dqn_static_history)),
                     str(len(dqn_dynamic_history))))
    print(fmt.format("Train SR (overall)",
                     f"{summary['q_learning']['training_success_rate']:.2%}",
                     f"{summary['dqn_static']['training_success_rate']:.2%}",
                     f"{summary['dqn_dynamic']['training_success_rate']:.2%}"))
    print(fmt.format("Train SR (last 100)",
                     f"{summary['q_learning']['last_100_train_sr']:.2%}",
                     f"{summary['dqn_static']['last_100_train_sr']:.2%}",
                     f"{summary['dqn_dynamic']['last_100_train_sr']:.2%}"))
    print(fmt.format("Eval success rate",
                     f"{q_eval['success_rate']:.2%}",
                     f"{dqn_s_eval['success_rate']:.2%}",
                     f"{dqn_d_eval['success_rate']:.2%}"))
    print(fmt.format("Eval avg reward",
                     f"{q_eval['avg_reward']:.2f}",
                     f"{dqn_s_eval['avg_reward']:.2f}",
                     f"{dqn_d_eval['avg_reward']:.2f}"))
    print(fmt.format("Eval avg steps",
                     f"{q_eval['avg_steps']:.2f}",
                     f"{dqn_s_eval['avg_steps']:.2f}",
                     f"{dqn_d_eval['avg_steps']:.2f}"))
    print(fmt.format("Eval avg path len",
                     f"{q_eval['avg_path_length']:.2f}",
                     f"{dqn_s_eval['avg_path_length']:.2f}",
                     f"{dqn_d_eval['avg_path_length']:.2f}"))
    print("=" * 60)

    print(f"\n  Plots saved to: {_PLOTS}")
    print(f"  Results saved to: {_RESULTS}")
    print()


# ── CSV writer ────────────────────────────────────────────────────────

def _write_csv(summary: dict, path: str) -> None:
    """Write a flat CSV table from the summary dict."""
    rows = []
    for agent_key in ("q_learning", "dqn_static", "dqn_dynamic"):
        data = summary[agent_key]
        evl = data["evaluation"]
        row = {
            "agent": agent_key,
            "training_episodes": data["training_episodes"],
            "training_success_rate": f"{data['training_success_rate']:.4f}",
            "last_100_train_sr": f"{data['last_100_train_sr']:.4f}",
            "eval_success_rate": f"{evl['success_rate']:.4f}",
            "eval_avg_reward": f"{evl['avg_reward']:.2f}",
            "eval_avg_steps": f"{evl['avg_steps']:.2f}",
            "eval_avg_path_length": f"{evl['avg_path_length']:.2f}",
        }
        # Dynamic-specific fields.
        if "disaster_terminations" in evl:
            row["disaster_terminations"] = evl["disaster_terminations"]
            row["disaster_encounter_rate"] = f"{evl['disaster_encounter_rate']:.4f}"
        rows.append(row)

    fieldnames = list(rows[-1].keys())  # dynamic row has the superset
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
