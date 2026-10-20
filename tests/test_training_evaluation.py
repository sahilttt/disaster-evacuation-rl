"""
Tests for Trainer and Evaluator.

Covers:
  - Trainer with Q-Learning agent
  - Trainer with DQN agent
  - Training history structure
  - Evaluator metrics
"""

from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from environment.disaster_env import DisasterEnv
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent


# =====================================================================
# Helpers
# =====================================================================

def _small_env() -> DisasterEnv:
    return DisasterEnv(
        grid_size=5,
        max_steps=30,
        start_position=(0, 0),
        goal_position=(4, 4),
    )


# =====================================================================
# Trainer — Q-Learning
# =====================================================================

def test_trainer_q_learning_returns_history() -> None:
    """Trainer returns a list of per-episode dicts for Q-Learning."""
    env = _small_env()
    agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.9, min_epsilon=0.1)
    trainer = Trainer(env, agent, num_episodes=5, seed=42)
    history = trainer.train()
    assert len(history) == 5
    for rec in history:
        assert "episode" in rec
        assert "total_reward" in rec
        assert "steps" in rec
        assert "success" in rec
        assert isinstance(rec["steps"], int)
        assert rec["steps"] > 0


def test_trainer_q_learning_decays_epsilon() -> None:
    """Epsilon is lower after training."""
    agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.9, min_epsilon=0.01)
    trainer = Trainer(_small_env(), agent, num_episodes=5)
    trainer.train()
    assert agent.epsilon < 1.0


def test_trainer_q_learning_get_rewards() -> None:
    """get_rewards() returns a list of floats."""
    agent = QLearningAgent(epsilon=0.5)
    trainer = Trainer(_small_env(), agent, num_episodes=3)
    trainer.train()
    rewards = trainer.get_rewards()
    assert len(rewards) == 3
    assert all(isinstance(r, float) for r in rewards)


def test_trainer_q_learning_populates_q_table() -> None:
    """Q-table has entries after training."""
    agent = QLearningAgent()
    trainer = Trainer(_small_env(), agent, num_episodes=10)
    trainer.train()
    assert len(agent.q_table) > 0


# =====================================================================
# Trainer — DQN
# =====================================================================

def test_trainer_dqn_returns_history() -> None:
    """Trainer returns a list with loss field for DQN."""
    env = _small_env()
    agent = DQNAgent(
        state_shape=env.get_state_shape(),
        batch_size=4, memory_size=200,
        epsilon=1.0, epsilon_decay=0.9, min_epsilon=0.1,
    )
    trainer = Trainer(env, agent, num_episodes=5)
    history = trainer.train()
    assert len(history) == 5
    for rec in history:
        assert "loss" in rec


def test_trainer_dqn_decays_epsilon() -> None:
    """Epsilon is lower after DQN training."""
    env = _small_env()
    agent = DQNAgent(
        state_shape=env.get_state_shape(),
        batch_size=4, memory_size=200,
        epsilon=1.0, epsilon_decay=0.9,
    )
    trainer = Trainer(env, agent, num_episodes=5)
    trainer.train()
    assert agent.epsilon < 1.0


# =====================================================================
# Trainer — misc
# =====================================================================

def test_trainer_stores_history_attribute() -> None:
    """Trainer.history matches the return value."""
    agent = QLearningAgent()
    trainer = Trainer(_small_env(), agent, num_episodes=3)
    history = trainer.train()
    assert trainer.history is history


def test_trainer_success_rate() -> None:
    """get_success_rate returns a float between 0 and 1."""
    agent = QLearningAgent()
    trainer = Trainer(_small_env(), agent, num_episodes=5)
    trainer.train()
    rate = trainer.get_success_rate()
    assert 0.0 <= rate <= 1.0


def test_trainer_seed_reproducibility() -> None:
    """Two trainers with the same seed produce the same reward sequence."""
    def run(seed: int):
        env = _small_env()
        agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.9)
        trainer = Trainer(env, agent, num_episodes=5, seed=seed)
        trainer.train()
        return trainer.get_rewards()

    a = run(123)
    b = run(123)
    assert a == b


# =====================================================================
# Evaluator — Q-Learning
# =====================================================================

def test_evaluator_returns_metrics() -> None:
    """Evaluator returns a dict with the required metric keys."""
    env = _small_env()
    agent = QLearningAgent(epsilon=0.5, epsilon_decay=0.9)
    Trainer(env, agent, num_episodes=30).train()

    evaluator = Evaluator(env, agent, num_episodes=10)
    results = evaluator.evaluate()

    assert "success_rate" in results
    assert "avg_reward" in results
    assert "avg_steps" in results
    assert "avg_path_length" in results
    assert 0.0 <= results["success_rate"] <= 1.0
    assert results["avg_steps"] > 0


def test_evaluator_greedy_mode() -> None:
    """Evaluator runs without exploration (training=False)."""
    env = _small_env()
    agent = QLearningAgent(epsilon=1.0)  # high epsilon, but eval should ignore it
    Trainer(env, agent, num_episodes=20).train()
    agent.epsilon = 1.0  # reset to high — evaluator must still be greedy

    results = Evaluator(env, agent, num_episodes=5).evaluate()
    # Evaluator calls choose_action(..., training=False) which ignores epsilon
    assert isinstance(results["avg_reward"], float)


# =====================================================================
# Evaluator — DQN
# =====================================================================

def test_evaluator_dqn() -> None:
    """Evaluator works with a DQN agent."""
    env = _small_env()
    agent = DQNAgent(
        state_shape=env.get_state_shape(),
        batch_size=4, memory_size=200,
        epsilon=1.0, epsilon_decay=0.8, min_epsilon=0.05,
    )
    Trainer(env, agent, num_episodes=10).train()

    results = Evaluator(env, agent, num_episodes=5).evaluate()
    assert "success_rate" in results
    assert "avg_reward" in results


# =====================================================================
# Evaluator — edge cases
# =====================================================================

def test_evaluator_untrained_agent() -> None:
    """Evaluator works on a completely untrained agent (returns valid dict)."""
    env = _small_env()
    agent = QLearningAgent()
    results = Evaluator(env, agent, num_episodes=3).evaluate()
    assert isinstance(results["success_rate"], float)
    assert isinstance(results["avg_steps"], float)


# =====================================================================
# Dynamic DQN — training & evaluation smoke test
# =====================================================================

def test_dynamic_dqn_training() -> None:
    """DQN trains on a dynamic-disaster env and the history is recorded."""
    env = DisasterEnv(
        grid_size=5,
        max_steps=30,
        start_position=(0, 0),
        goal_position=(4, 4),
        fire_cells=[(3, 1)],
        flood_cells=[(1, 1)],
        dynamic_disasters=True,
        disaster_change_interval=5,
        seed=42,
    )
    agent = DQNAgent(
        state_shape=env.get_state_shape(),
        batch_size=4,
        memory_size=200,
        epsilon=1.0,
        epsilon_decay=0.9,
        min_epsilon=0.1,
    )
    trainer = Trainer(env, agent, num_episodes=10, seed=42)
    history = trainer.train()

    assert len(history) == 10
    for rec in history:
        assert "episode" in rec
        assert "total_reward" in rec
        assert "steps" in rec
        assert "success" in rec
        assert "loss" in rec


def test_dynamic_dqn_evaluation() -> None:
    """DQN can be evaluated on a dynamic-disaster env."""
    env = DisasterEnv(
        grid_size=5,
        max_steps=30,
        start_position=(0, 0),
        goal_position=(4, 4),
        fire_cells=[(3, 1)],
        dynamic_disasters=True,
        disaster_change_interval=5,
        seed=42,
    )
    agent = DQNAgent(
        state_shape=env.get_state_shape(),
        batch_size=4,
        memory_size=200,
        epsilon=1.0,
        epsilon_decay=0.9,
        min_epsilon=0.1,
    )
    Trainer(env, agent, num_episodes=10).train()

    results = Evaluator(env, agent, num_episodes=5).evaluate()
    assert "success_rate" in results
    assert "avg_reward" in results
    assert "avg_steps" in results
    assert "avg_path_length" in results
    assert 0.0 <= results["success_rate"] <= 1.0

