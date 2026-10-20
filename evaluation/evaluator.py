"""
Evaluator — runs a trained agent without exploration and reports metrics.
"""

from typing import Any, Dict, List

import numpy as np

from environment.disaster_env import DisasterEnv


class Evaluator:
    """Evaluates a trained agent on the grid environment.

    The agent is run in greedy mode (``training=False``) so there is
    no exploration noise.

    Attributes:
        env: The grid-world environment.
        agent: A trained RL agent with ``choose_action(state, training)``.
        num_episodes: Number of evaluation episodes to run.
    """

    def __init__(
        self,
        env: DisasterEnv,
        agent: Any,
        num_episodes: int = 100,
    ) -> None:
        """Initialise the evaluator.

        Args:
            env: Environment instance.
            agent: Trained agent instance.
            num_episodes: Number of episodes for evaluation.
        """
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes

    def evaluate(self) -> Dict[str, float]:
        """Run evaluation episodes and compute aggregate metrics.

        Returns:
            Dictionary with:
                - ``success_rate``: fraction of episodes reaching the goal.
                - ``avg_reward``: mean total reward per episode.
                - ``avg_steps``: mean steps per episode.
                - ``avg_path_length``: mean steps in *successful* episodes
                  (0.0 if no successes).
        """
        rewards: List[float] = []
        steps_list: List[int] = []
        successes: int = 0
        success_steps: List[int] = []

        for _ in range(self.num_episodes):
            ep_reward, ep_steps, ep_success = self._run_episode()
            rewards.append(ep_reward)
            steps_list.append(ep_steps)
            if ep_success:
                successes += 1
                success_steps.append(ep_steps)

        n = self.num_episodes
        return {
            "success_rate": successes / n if n > 0 else 0.0,
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
            "avg_steps": float(np.mean(steps_list)) if steps_list else 0.0,
            "avg_path_length": float(np.mean(success_steps)) if success_steps else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_episode(self) -> tuple[float, int, bool]:
        """Run a single greedy evaluation episode.

        Returns:
            (total_reward, steps, success)
        """
        state = self.env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        info: Dict[str, Any] = {}

        while not done:
            action = self.agent.choose_action(state, training=False)
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            steps += 1

        success = info.get("reason") == "goal_reached"
        return total_reward, steps, success
