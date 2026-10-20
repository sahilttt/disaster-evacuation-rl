"""
Quick experiment: train Q-Learning and DQN on a small grid, evaluate, compare.

Usage:
    python -m experiments.compare_agents
    python experiments/compare_agents.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path when run directly.
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from environment.disaster_env import DisasterEnv
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from utils.helpers import save_history, save_json


def main() -> None:
    # ----- Environment ------------------------------------------------
    env = DisasterEnv(
        grid_size=5,
        max_steps=50,
        start_position=(0, 0),
        goal_position=(4, 4),
        obstacles=[(2, 2), (1, 3)],
        fire_cells=[(3, 1)],
        flood_cells=[(1, 1)],
        seed=42,
    )

    Q_TRAIN = 300
    DQN_TRAIN = 800
    EVAL_EPISODES = 50

    # ----- Q-Learning -------------------------------------------------
    print("=" * 55)
    print("  Training Q-Learning agent")
    print("=" * 55)
    q_agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
    )
    q_trainer = Trainer(env, q_agent, num_episodes=Q_TRAIN, seed=1)
    q_trainer.train()
    print(f"  Episodes    : {Q_TRAIN}")
    print(f"  Final eps   : {q_agent.epsilon:.4f}")
    print(f"  Train success rate: {q_trainer.get_success_rate():.2%}")

    q_results = Evaluator(env, q_agent, num_episodes=EVAL_EPISODES).evaluate()
    print(f"\n  [Eval] Success rate   : {q_results['success_rate']:.2%}")
    print(f"  [Eval] Avg reward     : {q_results['avg_reward']:.2f}")
    print(f"  [Eval] Avg steps      : {q_results['avg_steps']:.2f}")
    print(f"  [Eval] Avg path length: {q_results['avg_path_length']:.2f}")

    # ----- DQN --------------------------------------------------------
    print("\n" + "=" * 55)
    print("  Training DQN agent")
    print("=" * 55)
    dqn_agent = DQNAgent(
        state_shape=env.get_state_shape(),
        learning_rate=0.001,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        batch_size=32,
        memory_size=5000,
        hidden_size=64,
        target_update_freq=10,
    )
    dqn_trainer = Trainer(env, dqn_agent, num_episodes=DQN_TRAIN, seed=1)
    dqn_trainer.train()
    print(f"  Episodes    : {DQN_TRAIN}")
    print(f"  Final eps   : {dqn_agent.epsilon:.4f}")
    print(f"  Train success rate: {dqn_trainer.get_success_rate():.2%}")

    # Show last-100 training success rate
    last_100 = dqn_trainer.history[-100:]
    last_100_sr = sum(1 for r in last_100 if r["success"]) / len(last_100)
    print(f"  Last-100 train SR : {last_100_sr:.2%}")

    dqn_results = Evaluator(env, dqn_agent, num_episodes=EVAL_EPISODES).evaluate()
    print(f"\n  [Eval] Success rate   : {dqn_results['success_rate']:.2%}")
    print(f"  [Eval] Avg reward     : {dqn_results['avg_reward']:.2f}")
    print(f"  [Eval] Avg steps      : {dqn_results['avg_steps']:.2f}")
    print(f"  [Eval] Avg path length: {dqn_results['avg_path_length']:.2f}")

    # ----- Comparison -------------------------------------------------
    print("\n" + "=" * 55)
    print("  Comparison")
    print("=" * 55)
    print(f"  {'Metric':<22} {'Q-Learning':>12} {'DQN':>12}")
    print(f"  {'-'*22} {'-'*12} {'-'*12}")
    print(f"  {'Success rate':<22} {q_results['success_rate']:>11.2%} {dqn_results['success_rate']:>11.2%}")
    print(f"  {'Avg reward':<22} {q_results['avg_reward']:>12.2f} {dqn_results['avg_reward']:>12.2f}")
    print(f"  {'Avg steps':<22} {q_results['avg_steps']:>12.2f} {dqn_results['avg_steps']:>12.2f}")
    print(f"  {'Avg path length':<22} {q_results['avg_path_length']:>12.2f} {dqn_results['avg_path_length']:>12.2f}")
    print("=" * 55)

    # ----- Persist results for analysis -------------------------------
    _results_dir = str(Path(__file__).resolve().parent / "results")
    save_history(q_trainer.history, f"{_results_dir}/q_learning_history.json")
    save_history(dqn_trainer.history, f"{_results_dir}/dqn_static_history.json")
    save_json({"q_learning": q_results, "dqn_static": dqn_results},
              f"{_results_dir}/static_eval_results.json")


if __name__ == "__main__":
    main()
