"""
Tests for DQN network and agent.

Covers:
  1. Network output shape (4 action values)
  2. Action selection
  3. Replay buffer stores/samples
  4. Training step updates the network
  5. Target network update
  6. Epsilon decay
  7. Model save/load
"""

import os
import numpy as np
import torch

from models.dqn_network import DQNNetwork
from agents.dqn_agent import DQNAgent, ReplayBuffer
from environment.disaster_env import DisasterEnv
from config.config import NUM_ACTIONS


# =====================================================================
# Helpers
# =====================================================================

def _make_small_env() -> DisasterEnv:
    return DisasterEnv(
        grid_size=5,
        max_steps=50,
        start_position=(0, 0),
        goal_position=(4, 4),
    )


def _make_agent(grid_size: int = 5, **kwargs) -> DQNAgent:
    return DQNAgent(state_shape=(grid_size, grid_size), **kwargs)


# =====================================================================
# 1. Network output shape
# =====================================================================

def test_network_output_shape() -> None:
    """Network produces (batch, 4) Q-values."""
    net = DQNNetwork(input_size=25, hidden_size=32)
    x = torch.randn(8, 25)
    out = net(x)
    assert out.shape == (8, NUM_ACTIONS)


def test_network_single_input() -> None:
    """Forward pass works with a single sample."""
    net = DQNNetwork(input_size=25, hidden_size=16)
    x = torch.randn(1, 25)
    out = net(x)
    assert out.shape == (1, NUM_ACTIONS)


def test_network_gradient_flows() -> None:
    """Backward pass produces gradients on the parameters."""
    net = DQNNetwork(input_size=25, hidden_size=16)
    x = torch.randn(4, 25)
    out = net(x)
    loss = out.sum()
    loss.backward()
    for p in net.parameters():
        assert p.grad is not None


# =====================================================================
# 2. Action selection
# =====================================================================

def test_choose_action_returns_valid_int() -> None:
    """choose_action returns an int in [0, num_actions)."""
    env = _make_small_env()
    state = env.reset()
    agent = _make_agent(epsilon=0.5)
    for _ in range(20):
        a = agent.choose_action(state, training=True)
        assert 0 <= a < NUM_ACTIONS


def test_greedy_action_deterministic() -> None:
    """With epsilon=0, action is deterministic for a given state."""
    env = _make_small_env()
    state = env.reset()
    agent = _make_agent(epsilon=0.0)
    actions = [agent.choose_action(state, training=True) for _ in range(20)]
    assert len(set(actions)) == 1  # always the same


def test_eval_mode_greedy() -> None:
    """training=False uses greedy even with high epsilon."""
    env = _make_small_env()
    state = env.reset()
    agent = _make_agent(epsilon=1.0)
    actions = [agent.choose_action(state, training=False) for _ in range(30)]
    assert len(set(actions)) == 1  # always greedy


def test_get_best_action() -> None:
    """get_best_action returns a valid action."""
    env = _make_small_env()
    state = env.reset()
    agent = _make_agent()
    a = agent.get_best_action(state)
    assert 0 <= a < NUM_ACTIONS


# =====================================================================
# 3. Replay buffer
# =====================================================================

def test_buffer_push_and_len() -> None:
    """Buffer length grows with pushes."""
    buf = ReplayBuffer(max_size=100)
    assert len(buf) == 0
    for i in range(10):
        buf.push(np.zeros(25, dtype=np.float32), 0, 1.0,
                 np.zeros(25, dtype=np.float32), False)
    assert len(buf) == 10


def test_buffer_max_size() -> None:
    """Buffer does not exceed max_size."""
    buf = ReplayBuffer(max_size=5)
    for i in range(20):
        buf.push(np.zeros(25, dtype=np.float32), 0, 1.0,
                 np.zeros(25, dtype=np.float32), False)
    assert len(buf) == 5


def test_buffer_sample_shape() -> None:
    """Sampled batch has correct tensor shapes."""
    buf = ReplayBuffer(max_size=100)
    for _ in range(20):
        buf.push(np.ones(25, dtype=np.float32), 1, -1.0,
                 np.zeros(25, dtype=np.float32), True)
    batch = buf.sample(8)
    assert batch["states"].shape == (8, 25)
    assert batch["actions"].shape == (8,)
    assert batch["rewards"].shape == (8,)
    assert batch["next_states"].shape == (8, 25)
    assert batch["dones"].shape == (8,)


def test_store_transition_via_agent() -> None:
    """store_transition adds to the agent's replay buffer."""
    env = _make_small_env()
    state = env.reset()
    agent = _make_agent()
    next_state, reward, done, _ = env.step(1)
    agent.store_transition(state, 1, reward, next_state, done)
    assert len(agent.replay_buffer) == 1


# =====================================================================
# 4. Training step
# =====================================================================

def test_train_step_returns_none_when_buffer_small() -> None:
    """train_step returns None if buffer has fewer samples than batch_size."""
    agent = _make_agent(batch_size=32)
    assert agent.train_step() is None


def test_train_step_returns_loss() -> None:
    """train_step returns a float loss after enough transitions."""
    env = _make_small_env()
    agent = _make_agent(batch_size=4, memory_size=100)

    state = env.reset()
    for _ in range(10):
        action = agent.choose_action(state, training=True)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()

    loss = agent.train_step()
    assert loss is not None
    assert isinstance(loss, float)


def test_train_step_changes_weights() -> None:
    """A training step modifies the online network parameters."""
    env = _make_small_env()
    agent = _make_agent(batch_size=4, memory_size=100)

    state = env.reset()
    for _ in range(10):
        action = agent.choose_action(state, training=True)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()

    # Snapshot a parameter before training.
    old_param = list(agent.online_net.parameters())[0].data.clone()
    agent.train_step()
    new_param = list(agent.online_net.parameters())[0].data
    assert not torch.equal(old_param, new_param)


# =====================================================================
# 5. Target network update
# =====================================================================

def test_target_network_initially_matches() -> None:
    """After construction, target and online networks have the same weights."""
    agent = _make_agent()
    for p_online, p_target in zip(
        agent.online_net.parameters(), agent.target_net.parameters()
    ):
        assert torch.equal(p_online.data, p_target.data)


def test_update_target_network_syncs() -> None:
    """update_target_network copies online weights to target."""
    env = _make_small_env()
    agent = _make_agent(batch_size=4, memory_size=100)

    # Fill buffer & do a training step to change online weights.
    state = env.reset()
    for _ in range(10):
        action = agent.choose_action(state, training=True)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()
    agent.train_step()

    # Online and target should now differ.
    first_online = list(agent.online_net.parameters())[0].data
    first_target = list(agent.target_net.parameters())[0].data
    assert not torch.equal(first_online, first_target)

    # Sync and verify.
    agent.update_target_network()
    for p_online, p_target in zip(
        agent.online_net.parameters(), agent.target_net.parameters()
    ):
        assert torch.equal(p_online.data, p_target.data)


# =====================================================================
# 6. Epsilon decay
# =====================================================================

def test_epsilon_decay_once() -> None:
    """One decay step multiplies epsilon by decay factor."""
    agent = _make_agent(epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.01)
    agent.decay_epsilon()
    assert abs(agent.epsilon - 0.99) < 1e-9


def test_epsilon_decay_clamps_at_min() -> None:
    """Epsilon never drops below min_epsilon."""
    agent = _make_agent(epsilon=0.02, epsilon_decay=0.5, min_epsilon=0.01)
    agent.decay_epsilon()
    assert agent.epsilon >= agent.min_epsilon
    agent.decay_epsilon()
    assert abs(agent.epsilon - 0.01) < 1e-9


# =====================================================================
# 7. Save / load
# =====================================================================

def test_save_creates_file(tmp_path: object) -> None:
    """save() creates a checkpoint file."""
    path = os.path.join(str(tmp_path), "model.pt")
    agent = _make_agent()
    agent.save(path)
    assert os.path.isfile(path)


def test_load_restores_weights(tmp_path: object) -> None:
    """load() restores network weights and hyperparameters."""
    path = os.path.join(str(tmp_path), "model.pt")

    agent_a = _make_agent(epsilon=0.42, learning_rate=0.005, discount_factor=0.95)
    agent_a.save(path)

    agent_b = _make_agent()
    agent_b.load(path)

    assert abs(agent_b.epsilon - 0.42) < 1e-9
    assert abs(agent_b.lr - 0.005) < 1e-9
    assert abs(agent_b.gamma - 0.95) < 1e-9

    # Check that network weights match.
    for p_a, p_b in zip(
        agent_a.online_net.parameters(), agent_b.online_net.parameters()
    ):
        assert torch.equal(p_a.data, p_b.data)

    for p_a, p_b in zip(
        agent_a.target_net.parameters(), agent_b.target_net.parameters()
    ):
        assert torch.equal(p_a.data, p_b.data)


def test_save_load_after_training(tmp_path: object) -> None:
    """Save after a short training, load into a fresh agent, and verify."""
    path = os.path.join(str(tmp_path), "trained.pt")
    env = _make_small_env()

    agent = _make_agent(batch_size=4, memory_size=200, epsilon=0.5)
    agent.train(env, episodes=5)
    agent.save(path)

    loaded = _make_agent()
    loaded.load(path)

    # Greedy actions should match on a fresh state.
    state = env.reset()
    assert agent.get_best_action(state) == loaded.get_best_action(state)
