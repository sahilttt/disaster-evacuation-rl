"""
Trainer — unified training utility for Q-Learning and DQN agents.

Works with any agent that exposes:
    - choose_action(state, training=True) -> int
    - update(state, action, reward, next_state, done)   [Q-Learning]
    - store_transition(...) + train_step()               [DQN]
    - decay_epsilon()
"""

from typing import Any, Dict, List, Optional

import numpy as np

from environment.disaster_env import DisasterEnv


class Trainer:
    """Runs training episodes for a given agent and environment.

    Detects the agent type automatically:
      - If the agent has ``store_transition`` it is treated as DQN.
      - Otherwise it is treated as Q-Learning.

    Attributes:
        env: The grid-world environment.
        agent: An RL agent instance.
        num_episodes: Total training episodes.
        history: Per-episode metrics recorded during training.
    """

    def __init__(
        self,
        env: DisasterEnv,
        agent: Any,
        num_episodes: int = 1000,
        seed: Optional[int] = None,
    ) -> None:
        """Initialise the trainer.

        Args:
            env: Environment instance.
            agent: Agent instance (QLearningAgent or DQNAgent).
            num_episodes: Number of episodes to train.
            seed: Random seed for NumPy (reproducibility).
        """
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.history: List[Dict[str, Any]] = []

        if seed is not None:
            np.random.seed(seed)

        # Detect agent type.
        self._is_dqn: bool = hasattr(agent, "store_transition")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> List[Dict[str, Any]]:
        """Run the full training loop.

        Returns:
            List of per-episode metric dicts with keys:
            ``episode``, ``total_reward``, ``steps``, ``success``,
            and ``loss`` (DQN only).
        """
        self.history = []

        for ep in range(1, self.num_episodes + 1):
            record = self._run_episode(ep)
            self.history.append(record)

            # Sync DQN target network periodically.
            if self._is_dqn:
                freq = getattr(self.agent, "target_update_freq", 10)
                if ep % freq == 0:
                    self.agent.update_target_network()

        return self.history

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_episode(self, episode_num: int) -> Dict[str, Any]:
        """Run a single training episode.

        Args:
            episode_num: 1-based episode index.

        Returns:
            Dict with episode metrics.
        """
        state = self.env.reset()
        total_reward = 0.0
        total_loss = 0.0
        loss_count = 0
        done = False
        steps = 0
        info: Dict[str, Any] = {}

        while not done:
            action = self.agent.choose_action(state, training=True)
            next_state, reward, done, info = self.env.step(action)

            if self._is_dqn:
                self.agent.store_transition(state, action, reward, next_state, done)
                loss = self.agent.train_step()
                if loss is not None:
                    total_loss += loss
                    loss_count += 1
            else:
                self.agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

        self.agent.decay_epsilon()

        record: Dict[str, Any] = {
            "episode": episode_num,
            "total_reward": total_reward,
            "steps": steps,
            "success": info.get("reason") == "goal_reached",
        }
        if self._is_dqn:
            record["loss"] = total_loss / loss_count if loss_count > 0 else 0.0

        return record

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_rewards(self) -> List[float]:
        """Return the list of total rewards from the training history."""
        return [r["total_reward"] for r in self.history]

    def get_success_rate(self) -> float:
        """Return the fraction of episodes that reached the goal."""
        if not self.history:
            return 0.0
        return sum(1 for r in self.history if r["success"]) / len(self.history)
