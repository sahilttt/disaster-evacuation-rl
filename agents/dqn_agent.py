"""
DQN Agent.

Deep Q-Network agent with experience replay and a target network,
designed to work with the DisasterEnv grid-world.
"""

import os
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config.config import DQN_CONFIG, NUM_ACTIONS, CELL_AGENT
from models.dqn_network import DQNNetwork


# -----------------------------------------------------------------
# Experience Replay Buffer
# -----------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size circular buffer that stores transitions for DQN.

    Each transition is a tuple:
        (state, action, reward, next_state, done)

    Attributes:
        buffer: Internal deque holding transitions.
        max_size: Maximum number of stored transitions.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random mini-batch and return it as tensors.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dict with keys ``states``, ``actions``, ``rewards``,
            ``next_states``, ``dones`` — each a torch Tensor.
        """
        batch = random.sample(list(self.buffer), batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return {
            "states": torch.FloatTensor(np.array(states)),
            "actions": torch.LongTensor(actions),
            "rewards": torch.FloatTensor(rewards),
            "next_states": torch.FloatTensor(np.array(next_states)),
            "dones": torch.FloatTensor(dones),
        }

    def __len__(self) -> int:
        return len(self.buffer)


# -----------------------------------------------------------------
# DQN Agent
# -----------------------------------------------------------------

class DQNAgent:
    """Deep Q-Network agent for the disaster grid environment.

    Uses an online network for action selection and learning, and a
    separate target network for stable Q-value targets.

    Attributes:
        num_actions: Size of the action space (4).
        lr: Optimiser learning rate.
        gamma: Discount factor.
        epsilon: Current exploration rate.
        min_epsilon: Floor for epsilon.
        epsilon_decay: Multiplicative decay per episode.
        batch_size: Mini-batch size for replay sampling.
        target_update_freq: Episodes between target-network syncs.
        online_net: The network being trained.
        target_net: Frozen copy used for computing targets.
        optimizer: Adam optimiser for the online network.
        replay_buffer: Experience replay storage.
        training_history: Per-episode metrics recorded during train().
    """

    def __init__(
        self,
        state_shape: Tuple[int, ...],
        num_actions: int = NUM_ACTIONS,
        learning_rate: float | None = None,
        discount_factor: float | None = None,
        epsilon: float | None = None,
        epsilon_decay: float | None = None,
        min_epsilon: float | None = None,
        batch_size: int | None = None,
        memory_size: int | None = None,
        hidden_size: int | None = None,
        target_update_freq: int | None = None,
        config: dict | None = None,
    ) -> None:
        """Initialise the DQN agent.

        Explicit keyword arguments override values from ``config``.

        Args:
            state_shape: Shape of the environment state, e.g. ``(10, 10)``.
            num_actions: Number of possible actions.
            learning_rate: Adam learning rate.
            discount_factor: Gamma.
            epsilon: Initial exploration rate.
            epsilon_decay: Multiplicative decay per episode.
            min_epsilon: Epsilon floor.
            batch_size: Replay mini-batch size.
            memory_size: Replay buffer capacity.
            hidden_size: Hidden-layer width for the Q-network.
            target_update_freq: Episodes between target-net updates.
            config: Dict of hyperparams (falls back to ``DQN_CONFIG``).
        """
        cfg = config or DQN_CONFIG

        self.state_shape = state_shape
        self.num_actions = num_actions
        self.lr: float = learning_rate if learning_rate is not None else cfg.get("learning_rate", 0.001)
        self.gamma: float = discount_factor if discount_factor is not None else cfg.get("discount_factor", 0.99)
        self.epsilon: float = epsilon if epsilon is not None else cfg.get("epsilon_start", 1.0)
        self.epsilon_decay: float = epsilon_decay if epsilon_decay is not None else cfg.get("epsilon_decay", 0.995)
        self.min_epsilon: float = min_epsilon if min_epsilon is not None else cfg.get("epsilon_end", 0.01)
        self.batch_size: int = batch_size if batch_size is not None else cfg.get("batch_size", 64)
        self.target_update_freq: int = target_update_freq if target_update_freq is not None else cfg.get("target_update_freq", 10)

        _memory = memory_size if memory_size is not None else cfg.get("memory_size", 10_000)
        _hidden = hidden_size if hidden_size is not None else cfg.get("hidden_size", 128)

        # Flatten the state for the network input.
        input_size = 1
        for dim in state_shape:
            input_size *= dim

        # Networks.
        self.online_net = DQNNetwork(input_size, _hidden)
        self.target_net = DQNNetwork(input_size, _hidden)
        self.update_target_network()          # sync weights at start
        self.target_net.eval()                # target net is never trained

        # Optimiser & loss.
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer.
        self.replay_buffer = ReplayBuffer(max_size=_memory)

        # Training history.
        self.training_history: List[Dict[str, float | int | bool]] = []

    # ------------------------------------------------------------------
    # State conversion
    # ------------------------------------------------------------------

    def _state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        """Flatten and normalise a grid state to [0, 1].

        Args:
            state: 2-D NumPy state array.

        Returns:
            Float tensor of shape ``(1, input_size)``.
        """
        flat = state.flatten().astype(np.float32) / 6.0  # max cell value
        return torch.FloatTensor(flat).unsqueeze(0)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy (training) or greedy (eval) action selection.

        Args:
            state: Current environment state.
            training: If True, explore with probability epsilon.

        Returns:
            Action index (0-3).
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        return self.get_best_action(state)

    def get_best_action(self, state: np.ndarray) -> int:
        """Return the greedy action (highest Q-value) for a state.

        Args:
            state: Current environment state.

        Returns:
            Action index with the highest Q-value.
        """
        with torch.no_grad():
            q_values = self.online_net(self._state_to_tensor(state))
        return int(q_values.argmax(dim=1).item())

    # Backward-compatible alias.
    def select_action(self, state: np.ndarray) -> int:
        """Alias for ``choose_action(state, training=True)``."""
        return self.choose_action(state, training=True)

    # ------------------------------------------------------------------
    # Experience replay
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a normalised transition in the replay buffer."""
        self.replay_buffer.push(
            state.flatten().astype(np.float32) / 6.0,
            action,
            reward,
            next_state.flatten().astype(np.float32) / 6.0,
            done,
        )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self) -> float | None:
        """Sample a mini-batch from replay and perform one gradient step.

        Returns:
            The loss value, or ``None`` if the buffer is too small.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states = batch["states"]          # (B, input_size)
        actions = batch["actions"]        # (B,)
        rewards = batch["rewards"]        # (B,)
        next_states = batch["next_states"]  # (B, input_size)
        dones = batch["dones"]            # (B,)  0.0 or 1.0

        # Current Q-values for the chosen actions.
        q_values = self.online_net(states)                       # (B, 4)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # Target Q-values from the frozen target network.
        with torch.no_grad():
            next_q = self.target_net(next_states)                # (B, 4)
            max_next_q = next_q.max(dim=1).values                # (B,)
            targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = self.loss_fn(q_selected, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return float(loss.item())

    # ------------------------------------------------------------------
    # Target network
    # ------------------------------------------------------------------

    def update_target_network(self) -> None:
        """Copy online network weights to the target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        """Decay exploration rate (called once per episode)."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, env: object, episodes: int = 500) -> List[Dict[str, float | int | bool]]:
        """Train the agent on the environment.

        Args:
            env: A DisasterEnv (or compatible) instance.
            episodes: Number of training episodes.

        Returns:
            Per-episode metrics: ``episode``, ``total_reward``,
            ``steps``, ``success``, ``loss``.
        """
        self.training_history = []

        for ep in range(1, episodes + 1):
            state = env.reset()  # type: ignore[union-attr]
            total_reward = 0.0
            total_loss = 0.0
            loss_count = 0
            done = False
            steps = 0

            while not done:
                action = self.choose_action(state, training=True)
                next_state, reward, done, info = env.step(action)  # type: ignore[union-attr]

                self.store_transition(state, action, reward, next_state, done)
                loss = self.train_step()

                if loss is not None:
                    total_loss += loss
                    loss_count += 1

                state = next_state
                total_reward += reward
                steps += 1

            self.decay_epsilon()

            # Sync target network periodically.
            if ep % self.target_update_freq == 0:
                self.update_target_network()

            success = info.get("reason") == "goal_reached"  # type: ignore[possibly-undefined]
            record: Dict[str, float | int | bool] = {
                "episode": ep,
                "total_reward": total_reward,
                "steps": steps,
                "success": success,
                "loss": total_loss / loss_count if loss_count > 0 else 0.0,
            }
            self.training_history.append(record)

        return self.training_history

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and hyperparameters.

        Args:
            path: File path for the checkpoint (``.pt``).
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "epsilon_decay": self.epsilon_decay,
                "min_epsilon": self.min_epsilon,
                "lr": self.lr,
                "gamma": self.gamma,
                "state_shape": self.state_shape,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model weights and hyperparameters.

        Args:
            path: Checkpoint file path.
        """
        checkpoint = torch.load(path, weights_only=False)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.epsilon_decay = checkpoint["epsilon_decay"]
        self.min_epsilon = checkpoint["min_epsilon"]
        self.lr = checkpoint["lr"]
        self.gamma = checkpoint["gamma"]
