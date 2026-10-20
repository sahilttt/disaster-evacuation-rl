"""
Q-Learning Agent.

Tabular Q-Learning agent that learns optimal evacuation routes
on the DisasterEnv grid-world.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.config import Q_LEARNING_CONFIG, NUM_ACTIONS


class QLearningAgent:
    """Tabular Q-Learning agent for the disaster grid environment.

    Uses agent position (row, col) as the state key and maintains
    a Q-table mapping each state to an array of Q-values (one per action).

    Attributes:
        num_actions: Number of possible actions (4).
        lr: Learning rate (alpha).
        gamma: Discount factor.
        epsilon: Current exploration rate.
        min_epsilon: Minimum exploration rate.
        epsilon_decay: Multiplicative decay applied after each episode.
        q_table: Mapping from (row, col) to Q-values array.
        training_history: Per-episode metrics recorded during train().
    """

    def __init__(
        self,
        num_actions: int = NUM_ACTIONS,
        learning_rate: float | None = None,
        discount_factor: float | None = None,
        epsilon: float | None = None,
        epsilon_decay: float | None = None,
        min_epsilon: float | None = None,
        config: dict | None = None,
        # Legacy kwarg kept so existing code that passes state_shape still works.
        state_shape: Tuple[int, ...] | None = None,
    ) -> None:
        """Initialise the Q-Learning agent.

        Explicit keyword arguments override values from ``config``.

        Args:
            num_actions: Size of the action space.
            learning_rate: Step-size parameter (alpha).
            discount_factor: Discount factor (gamma).
            epsilon: Initial exploration rate.
            epsilon_decay: Multiplicative decay per episode.
            min_epsilon: Floor for epsilon.
            config: Dict of hyperparameters (falls back to Q_LEARNING_CONFIG).
            state_shape: Ignored — kept for backward compatibility.
        """
        cfg = config or Q_LEARNING_CONFIG

        self.num_actions: int = num_actions
        self.lr: float = learning_rate if learning_rate is not None else cfg.get("learning_rate", 0.1)
        self.gamma: float = discount_factor if discount_factor is not None else cfg.get("discount_factor", 0.99)
        self.epsilon: float = epsilon if epsilon is not None else cfg.get("epsilon_start", 1.0)
        self.epsilon_decay: float = epsilon_decay if epsilon_decay is not None else cfg.get("epsilon_decay", 0.995)
        self.min_epsilon: float = min_epsilon if min_epsilon is not None else cfg.get("epsilon_end", 0.01)

        # Q-table: state (row, col) -> np.array of shape (num_actions,)
        self.q_table: Dict[Tuple[int, int], np.ndarray] = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=np.float64)
        )

        # Filled by train().
        self.training_history: List[Dict[str, float | int | bool]] = []

    # ------------------------------------------------------------------
    # State helper
    # ------------------------------------------------------------------

    @staticmethod
    def _state_key(state: np.ndarray) -> Tuple[int, int]:
        """Extract the agent position from a state grid.

        The environment marks the agent cell with CELL_AGENT (6).
        We find that cell and return its (row, col).

        Args:
            state: 2-D NumPy state array from the environment.

        Returns:
            (row, col) tuple used as the Q-table key.
        """
        from config.config import CELL_AGENT
        positions = np.argwhere(state == CELL_AGENT)
        if len(positions) == 0:
            raise ValueError("Agent marker not found in state array.")
        return (int(positions[0][0]), int(positions[0][1]))

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """Choose an action using epsilon-greedy (training) or greedy (eval).

        Args:
            state: Current environment state (2-D grid).
            training: If True, explore with probability epsilon.

        Returns:
            Selected action index (0-3).
        """
        if training and np.random.random() < self.epsilon:
            return int(np.random.randint(self.num_actions))
        return self.get_best_action(state)

    def get_best_action(self, state: np.ndarray) -> int:
        """Return the greedy (highest Q-value) action for a state.

        Args:
            state: Current environment state.

        Returns:
            Action index with the highest Q-value.
        """
        key = self._state_key(state)
        return int(np.argmax(self.q_table[key]))

    # Backward-compatible alias.
    def select_action(self, state: np.ndarray) -> int:
        """Alias for ``choose_action(state, training=True)``."""
        return self.choose_action(state, training=True)

    # ------------------------------------------------------------------
    # Q-value update
    # ------------------------------------------------------------------

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Update Q-value for a single (s, a, r, s', done) transition.

        Uses the standard Q-Learning update rule:
            Q(s,a) += lr * [r + gamma * max_a' Q(s',a') - Q(s,a)]

        Args:
            state: State before the action.
            action: Action taken.
            reward: Reward received.
            next_state: State after the action.
            done: Whether the episode ended.
        """
        key = self._state_key(state)
        next_key = self._state_key(next_state)

        best_next = 0.0 if done else float(np.max(self.q_table[next_key]))
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[key][action]
        self.q_table[key][action] += self.lr * td_error

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        """Decay exploration rate (called once per episode)."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, env: object, episodes: int = 1000) -> List[Dict[str, float | int | bool]]:
        """Train the agent on the environment for a number of episodes.

        Args:
            env: A DisasterEnv (or compatible) instance with reset() / step().
            episodes: Number of training episodes.

        Returns:
            List of per-episode metrics dicts with keys:
            ``episode``, ``total_reward``, ``steps``, ``success``.
        """
        self.training_history = []

        for ep in range(1, episodes + 1):
            state = env.reset()  # type: ignore[union-attr]
            total_reward = 0.0
            done = False
            steps = 0

            while not done:
                action = self.choose_action(state, training=True)
                next_state, reward, done, info = env.step(action)  # type: ignore[union-attr]
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1

            self.decay_epsilon()

            success = info.get("reason") == "goal_reached"  # type: ignore[possibly-undefined]
            record = {
                "episode": ep,
                "total_reward": total_reward,
                "steps": steps,
                "success": success,
            }
            self.training_history.append(record)

        return self.training_history

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save Q-table and hyperparameters to a JSON file.

        Args:
            path: File path to write (creates parent dirs if needed).
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Convert Q-table: tuple keys -> string keys, arrays -> lists.
        serialisable_table = {
            f"{r},{c}": q_values.tolist()
            for (r, c), q_values in self.q_table.items()
        }
        data = {
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "num_actions": self.num_actions,
            "q_table": serialisable_table,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load Q-table and hyperparameters from a JSON file.

        Args:
            path: File path to read.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.lr = data["lr"]
        self.gamma = data["gamma"]
        self.epsilon = data["epsilon"]
        self.epsilon_decay = data["epsilon_decay"]
        self.min_epsilon = data["min_epsilon"]
        self.num_actions = data["num_actions"]

        self.q_table = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=np.float64)
        )
        for key_str, q_values in data["q_table"].items():
            r, c = map(int, key_str.split(","))
            self.q_table[(r, c)] = np.array(q_values, dtype=np.float64)
