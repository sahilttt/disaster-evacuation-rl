"""
Grid Visualizer — placeholder.

Will render the grid environment and training curves in Phase 7.
"""

from typing import List, Optional
import numpy as np

from environment.disaster_env import DisasterGridEnv


class GridVisualizer:
    """Renders the disaster grid and training statistics.

    Attributes:
        env: The grid-world environment instance.
    """

    def __init__(self, env: DisasterGridEnv) -> None:
        """Initialise the visualizer.

        Args:
            env: Environment instance to visualise.
        """
        self.env = env

    def render_grid(self) -> None:
        """Display the current grid state using Matplotlib."""
        # TODO: Implement grid rendering in Phase 7.
        raise NotImplementedError("Grid rendering not yet implemented.")

    def plot_training_rewards(self, rewards: List[float]) -> None:
        """Plot episode rewards over time.

        Args:
            rewards: List of total rewards per episode.
        """
        # TODO: Implement reward plotting in Phase 7.
        raise NotImplementedError("Reward plotting not yet implemented.")
