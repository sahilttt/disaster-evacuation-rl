"""
Experiment: Train and evaluate a DQN agent on a dynamic-disaster environment.

The environment periodically reshuffles fire and flood cells during each
episode, forcing the agent to generalise rather than memorise a single
hazard layout.

Usage:
    python -m experiments.train_dynamic
    python experiments/train_dynamic.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path when run directly.
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import random
import numpy as np
import torch

from environment.disaster_env import DisasterEnv
from agents.dqn_agent import DQNAgent
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from utils.helpers import save_history, save_json


# ── Reproducibility ──────────────────────────────────────────────────
SEED = 42


def seed_everything(seed: int) -> None:
    """Pin all sources of randomness."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Helpers ──────────────────────────────────────────────────────────

def _last_n_success_rate(history: list, n: int = 100) -> float:
    """Success rate over the last *n* training episodes."""
    tail = history[-n:]
    if not tail:
        return 0.0
    return sum(1 for r in tail if r["success"]) / len(tail)


def evaluate_dynamic(
    env: DisasterEnv,
    agent: DQNAgent,
    num_episodes: int = 100,
) -> dict:
    """Evaluate the agent on the dynamic env, also tracking disasters.

    Returns the standard Evaluator metrics plus:
        - disaster_terminations: episodes ending in fire or flood.
        - disaster_encounter_rate: fraction of episodes lost to hazards.
    """
    rewards = []
    steps_list = []
    successes = 0
    success_steps = []
    disaster_terminations = 0

    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        info = {}

        while not done:
            action = agent.choose_action(state, training=False)
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

        reason = info.get("reason", "")
        if reason == "goal_reached":
            successes += 1
            success_steps.append(steps)
        elif reason in ("fire", "flood"):
            disaster_terminations += 1

        rewards.append(total_reward)
        steps_list.append(steps)

    n = num_episodes
    return {
        "success_rate": successes / n if n > 0 else 0.0,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_steps": float(np.mean(steps_list)) if steps_list else 0.0,
        "avg_path_length": float(np.mean(success_steps)) if success_steps else 0.0,
        "disaster_terminations": disaster_terminations,
        "disaster_encounter_rate": disaster_terminations / n if n > 0 else 0.0,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main() -> dict:
    """Run the full train-then-evaluate pipeline. Returns eval metrics."""
    seed_everything(SEED)

    # ── Environment ---------------------------------------------------
    env = DisasterEnv(
        grid_size=5,
        max_steps=50,
        start_position=(0, 0),
        goal_position=(4, 4),
        obstacles=[(2, 2), (1, 3)],
        fire_cells=[(3, 1)],
        flood_cells=[(1, 1)],
        dynamic_disasters=True,
        disaster_change_interval=10,
        seed=SEED,
    )

    # ── Hyper-parameters ─────────────────────────────────────────────
    TRAIN_EPISODES = 1000
    EVAL_EPISODES = 100
    MODEL_PATH = str(Path(__file__).resolve().parent / "results" / "dqn_dynamic.pth")

    # ── Agent ─────────────────────────────────────────────────────────
    dqn_agent = DQNAgent(
        state_shape=env.get_state_shape(),
        learning_rate=0.001,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        batch_size=32,
        memory_size=10_000,
        hidden_size=128,
        target_update_freq=10,
    )

    # ── Training ──────────────────────────────────────────────────────
    print("=" * 60)
    print("  DQN Training on Dynamic Disaster Environment")
    print("=" * 60)

    trainer = Trainer(env, dqn_agent, num_episodes=TRAIN_EPISODES, seed=SEED)
    history = trainer.train()

    overall_sr = trainer.get_success_rate()
    last100_sr = _last_n_success_rate(history, 100)

    print(f"  Episodes          : {TRAIN_EPISODES}")
    print(f"  Final epsilon     : {dqn_agent.epsilon:.4f}")
    print(f"  Overall train SR  : {overall_sr:.2%}")
    print(f"  Last-100 train SR : {last100_sr:.2%}")

    # Quick disaster stats from training.
    fire_deaths = sum(
        1 for r, h in zip(history, history)
        if not h["success"] and h["steps"] < env.max_steps
    )

    # ── Save model ────────────────────────────────────────────────────
    dqn_agent.save(MODEL_PATH)
    print(f"\n  Model saved to: {MODEL_PATH}")

    # ── Evaluation ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Evaluation (greedy, no exploration)")
    print("=" * 60)

    # Use a fresh env with a different seed for evaluation.
    eval_env = DisasterEnv(
        grid_size=5,
        max_steps=50,
        start_position=(0, 0),
        goal_position=(4, 4),
        obstacles=[(2, 2), (1, 3)],
        fire_cells=[(3, 1)],
        flood_cells=[(1, 1)],
        dynamic_disasters=True,
        disaster_change_interval=10,
        seed=SEED + 1000,
    )

    results = evaluate_dynamic(eval_env, dqn_agent, num_episodes=EVAL_EPISODES)

    print(f"  Success rate          : {results['success_rate']:.2%}")
    print(f"  Avg reward            : {results['avg_reward']:.2f}")
    print(f"  Avg steps             : {results['avg_steps']:.2f}")
    print(f"  Avg path length       : {results['avg_path_length']:.2f}")
    print(f"  Disaster terminations : {results['disaster_terminations']}")
    print(f"  Disaster encounter %  : {results['disaster_encounter_rate']:.2%}")
    print("=" * 60)

    # ── Persist for analysis ──────────────────────────────────────────
    _results_dir = str(Path(__file__).resolve().parent / "results")
    save_history(history, f"{_results_dir}/dqn_dynamic_history.json")
    save_json(results, f"{_results_dir}/dynamic_eval_results.json")

    return results


if __name__ == "__main__":
    main()
