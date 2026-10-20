"""
Tests for the QLearningAgent.

Covers:
  1. Agent initialisation
  2. Action selection (epsilon-greedy & greedy)
  3. Q-value update
  4. Epsilon decay
  5. Short training run
  6. Save / load
"""

import os
import json
import numpy as np

from agents.q_learning_agent import QLearningAgent
from environment.disaster_env import DisasterEnv
from config.config import NUM_ACTIONS


# =====================================================================
# Helpers
# =====================================================================

def _make_small_env() -> DisasterEnv:
    """Create a small 5x5 env with fixed positions and no hazards."""
    return DisasterEnv(
        grid_size=5,
        max_steps=50,
        start_position=(0, 0),
        goal_position=(4, 4),
    )


# =====================================================================
# 1. Initialisation
# =====================================================================

def test_default_init() -> None:
    """Agent initialises with default hyperparameters."""
    agent = QLearningAgent()
    assert agent.num_actions == NUM_ACTIONS
    assert agent.lr == 0.1
    assert agent.gamma == 0.99
    assert agent.epsilon == 1.0
    assert agent.min_epsilon == 0.01
    assert len(agent.q_table) == 0


def test_custom_init() -> None:
    """Agent respects explicit keyword arguments."""
    agent = QLearningAgent(
        learning_rate=0.5,
        discount_factor=0.9,
        epsilon=0.8,
        epsilon_decay=0.99,
        min_epsilon=0.05,
    )
    assert agent.lr == 0.5
    assert agent.gamma == 0.9
    assert agent.epsilon == 0.8
    assert agent.epsilon_decay == 0.99
    assert agent.min_epsilon == 0.05


def test_backward_compat_state_shape() -> None:
    """Passing state_shape (legacy) does not raise."""
    agent = QLearningAgent(state_shape=(10, 10))
    assert agent.num_actions == NUM_ACTIONS


# =====================================================================
# 2. Action selection
# =====================================================================

def test_choose_action_returns_valid_int() -> None:
    """choose_action returns an int in [0, num_actions)."""
    env = _make_small_env()
    state = env.reset()
    agent = QLearningAgent(epsilon=0.5)
    for _ in range(20):
        action = agent.choose_action(state, training=True)
        assert 0 <= action < NUM_ACTIONS


def test_greedy_action_uses_q_table() -> None:
    """With epsilon=0, agent always picks the best Q-value action."""
    env = _make_small_env()
    state = env.reset()
    agent = QLearningAgent(epsilon=0.0)

    # Manually set Q-values so action 2 is best at (0,0).
    agent.q_table[(0, 0)] = np.array([1.0, 2.0, 10.0, 3.0])
    action = agent.choose_action(state, training=True)
    assert action == 2


def test_get_best_action() -> None:
    """get_best_action always returns the argmax action."""
    env = _make_small_env()
    state = env.reset()
    agent = QLearningAgent()
    agent.q_table[(0, 0)] = np.array([0.0, 5.0, 0.0, 0.0])
    assert agent.get_best_action(state) == 1


def test_eval_mode_is_greedy() -> None:
    """training=False produces greedy action even with high epsilon."""
    env = _make_small_env()
    state = env.reset()
    agent = QLearningAgent(epsilon=1.0)  # would always explore if training
    agent.q_table[(0, 0)] = np.array([0.0, 0.0, 0.0, 99.0])

    # Run many times — should always be action 3 in eval mode.
    actions = [agent.choose_action(state, training=False) for _ in range(50)]
    assert all(a == 3 for a in actions)


# =====================================================================
# 3. Q-value update
# =====================================================================

def test_update_changes_q_value() -> None:
    """A single update changes the Q-value for (state, action)."""
    env = _make_small_env()
    state = env.reset()
    agent = QLearningAgent(learning_rate=0.1, discount_factor=0.99)

    old_q = agent.q_table[(0, 0)][1]  # action DOWN, starts at 0
    next_state, reward, done, _ = env.step(1)  # DOWN
    agent.update(state, 1, reward, next_state, done)
    new_q = agent.q_table[(0, 0)][1]
    assert new_q != old_q


def test_update_q_learning_formula() -> None:
    """Verify the update follows Q(s,a) += lr * [r + gamma*max(Q(s')) - Q(s,a)]."""
    env = _make_small_env()
    state = env.reset()
    agent = QLearningAgent(learning_rate=0.5, discount_factor=0.9)

    # Set known Q-values.
    agent.q_table[(0, 0)] = np.array([1.0, 2.0, 3.0, 4.0])
    agent.q_table[(1, 0)] = np.array([10.0, 0.0, 0.0, 0.0])

    next_state, reward, done, _ = env.step(1)  # DOWN -> (1,0)
    # reward = step_penalty = -1.0

    # Expected: Q(s,a) += 0.5 * [-1.0 + 0.9 * 10.0 - 2.0]
    #         = 2.0 + 0.5 * (8.0 - 2.0) = 2.0 + 0.5 * 6.0 = 5.0
    expected = 2.0 + 0.5 * (reward + 0.9 * 10.0 - 2.0)
    agent.update(state, 1, reward, next_state, done)
    assert abs(agent.q_table[(0, 0)][1] - expected) < 1e-9


def test_update_terminal_state() -> None:
    """When done=True, the target is just the reward (no future value)."""
    env = DisasterEnv(
        grid_size=5, start_position=(0, 0), goal_position=(0, 1), max_steps=50,
    )
    state = env.reset()
    agent = QLearningAgent(learning_rate=1.0, discount_factor=0.99)

    next_state, reward, done, _ = env.step(3)  # RIGHT -> goal
    assert done is True

    agent.update(state, 3, reward, next_state, done)
    # lr=1.0, done=True => Q(s,a) = 0 + 1.0*(reward + 0 - 0) = reward
    assert abs(agent.q_table[(0, 0)][3] - reward) < 1e-9


# =====================================================================
# 4. Epsilon decay
# =====================================================================

def test_epsilon_decay_once() -> None:
    """One decay step multiplies epsilon by decay factor."""
    agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.01)
    agent.decay_epsilon()
    assert abs(agent.epsilon - 0.99) < 1e-9


def test_epsilon_decay_clamps_at_min() -> None:
    """Epsilon never goes below min_epsilon."""
    agent = QLearningAgent(epsilon=0.02, epsilon_decay=0.5, min_epsilon=0.01)
    agent.decay_epsilon()  # 0.02 * 0.5 = 0.01
    assert agent.epsilon >= agent.min_epsilon
    agent.decay_epsilon()  # should stay at 0.01
    assert abs(agent.epsilon - 0.01) < 1e-9


def test_epsilon_decay_over_many_episodes() -> None:
    """Epsilon decreases monotonically and hits the floor."""
    agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.9, min_epsilon=0.1)
    prev = agent.epsilon
    for _ in range(200):
        agent.decay_epsilon()
        assert agent.epsilon <= prev
        prev = agent.epsilon
    assert abs(agent.epsilon - 0.1) < 1e-9


# =====================================================================
# 5. Short training run
# =====================================================================

def test_train_returns_history() -> None:
    """train() returns a list of per-episode metric dicts."""
    env = _make_small_env()
    agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.9, min_epsilon=0.01)
    history = agent.train(env, episodes=10)

    assert len(history) == 10
    for record in history:
        assert "episode" in record
        assert "total_reward" in record
        assert "steps" in record
        assert "success" in record
        assert isinstance(record["steps"], int)
        assert record["steps"] > 0


def test_train_populates_q_table() -> None:
    """After training, the Q-table has entries."""
    env = _make_small_env()
    agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.9, min_epsilon=0.1)
    agent.train(env, episodes=20)
    assert len(agent.q_table) > 0


def test_train_epsilon_decays() -> None:
    """Epsilon is lower after training than before."""
    agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.95, min_epsilon=0.01)
    env = _make_small_env()
    agent.train(env, episodes=10)
    assert agent.epsilon < 1.0


def test_train_stores_history_on_agent() -> None:
    """training_history attribute matches the return value."""
    env = _make_small_env()
    agent = QLearningAgent()
    history = agent.train(env, episodes=5)
    assert agent.training_history is history


# =====================================================================
# 6. Save / load
# =====================================================================

def test_save_creates_file(tmp_path: object) -> None:
    """save() creates a JSON file at the given path."""
    path = os.path.join(str(tmp_path), "q_table.json")
    agent = QLearningAgent(learning_rate=0.2)
    agent.q_table[(0, 0)] = np.array([1.0, 2.0, 3.0, 4.0])
    agent.save(path)
    assert os.path.isfile(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "q_table" in data
    assert "0,0" in data["q_table"]


def test_load_restores_agent(tmp_path: object) -> None:
    """load() restores Q-table and hyperparameters exactly."""
    path = os.path.join(str(tmp_path), "q_table.json")

    # Save agent A.
    agent_a = QLearningAgent(
        learning_rate=0.3, discount_factor=0.95, epsilon=0.5,
        epsilon_decay=0.9, min_epsilon=0.02,
    )
    agent_a.q_table[(2, 3)] = np.array([1.1, 2.2, 3.3, 4.4])
    agent_a.q_table[(0, 0)] = np.array([5.0, 6.0, 7.0, 8.0])
    agent_a.save(path)

    # Load into agent B.
    agent_b = QLearningAgent()
    agent_b.load(path)

    assert agent_b.lr == 0.3
    assert agent_b.gamma == 0.95
    assert agent_b.epsilon == 0.5
    assert agent_b.epsilon_decay == 0.9
    assert agent_b.min_epsilon == 0.02
    assert np.allclose(agent_b.q_table[(2, 3)], [1.1, 2.2, 3.3, 4.4])
    assert np.allclose(agent_b.q_table[(0, 0)], [5.0, 6.0, 7.0, 8.0])


def test_save_load_round_trip_after_training(tmp_path: object) -> None:
    """Save after training, load, and verify Q-values match."""
    path = os.path.join(str(tmp_path), "trained.json")
    env = _make_small_env()

    agent = QLearningAgent(epsilon=0.5, epsilon_decay=0.95, min_epsilon=0.05)
    agent.train(env, episodes=20)
    agent.save(path)

    loaded = QLearningAgent()
    loaded.load(path)

    for key, q_vals in agent.q_table.items():
        assert key in loaded.q_table
        assert np.allclose(loaded.q_table[key], q_vals)
