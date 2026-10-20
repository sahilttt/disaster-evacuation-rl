"""
Disaster Grid-World Environment.

A grid-based environment where an agent must navigate from a start
position to a safe evacuation zone, avoiding hazards like fire,
flood, and obstacles.
"""

from typing import Tuple, Optional, List, Dict, Any
import numpy as np

from config.config import (
    GRID_CONFIG,
    CELL_EMPTY,
    CELL_OBSTACLE,
    CELL_FIRE,
    CELL_FLOOD,
    CELL_SAFE_ZONE,
    CELL_AGENT,
    NUM_ACTIONS,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    REWARD_CONFIG,
)

# Cell-type constants re-exported for convenience.
EMPTY = CELL_EMPTY
OBSTACLE = CELL_OBSTACLE
FIRE = CELL_FIRE
FLOOD = CELL_FLOOD
START = CELL_AGENT      # Agent / start marker on the grid
GOAL = CELL_SAFE_ZONE   # Evacuation target marker

# Action constants re-exported.
UP = ACTION_UP
DOWN = ACTION_DOWN
LEFT = ACTION_LEFT
RIGHT = ACTION_RIGHT

# Direction deltas: (row_delta, col_delta) for each action.
_ACTION_DELTAS: Dict[int, Tuple[int, int]] = {
    UP:    (-1,  0),
    DOWN:  ( 1,  0),
    LEFT:  ( 0, -1),
    RIGHT: ( 0,  1),
}


class DisasterEnv:
    """Grid-world environment for disaster evacuation training.

    The agent must navigate from a random start cell to a random goal
    cell on an N x N grid while avoiding obstacles, fire, and flood zones.

    Attributes:
        grid_size: Side length of the square grid.
        max_steps: Maximum steps allowed per episode.
        grid: 2-D NumPy array representing cell types.
        agent_position: Current (row, col) of the agent.
        start_position: Start cell for the current episode.
        goal_position: Goal cell for the current episode.
        current_step: Steps taken in the current episode.
        done: Whether the episode has ended.
    """

    def __init__(
        self,
        grid_size: int = 10,
        max_steps: int = 100,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        fire_cells: Optional[List[Tuple[int, int]]] = None,
        flood_cells: Optional[List[Tuple[int, int]]] = None,
        start_position: Optional[Tuple[int, int]] = None,
        goal_position: Optional[Tuple[int, int]] = None,
        rewards: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
        dynamic_disasters: bool = False,
        disaster_change_interval: int = 10,
    ) -> None:
        """Initialise the environment.

        Args:
            grid_size: Side length of the N x N grid (default 10).
            max_steps: Maximum steps before forced termination.
            obstacles: List of (row, col) cells that are impassable.
            fire_cells: List of (row, col) cells with fire hazard.
            flood_cells: List of (row, col) cells with flood hazard.
            start_position: Fixed start cell, or ``None`` for random.
            goal_position: Fixed goal cell, or ``None`` for random.
            rewards: Reward overrides (keys match ``REWARD_CONFIG``).
            seed: Random seed for reproducibility.
            dynamic_disasters: When True, fire and flood cells may
                change positions at regular intervals during an episode.
            disaster_change_interval: Number of steps between dynamic
                disaster updates (only used when ``dynamic_disasters``
                is True).
        """
        self.grid_size: int = grid_size
        self.max_steps: int = max_steps

        # Hazard placements (stored for re-use across resets).
        self._obstacles = obstacles or []
        self._fire_cells = fire_cells or []
        self._flood_cells = flood_cells or []

        # Fixed or random start / goal.
        self._fixed_start = start_position
        self._fixed_goal = goal_position

        # Rewards.
        self._rewards = {**REWARD_CONFIG, **(rewards or {})}

        # RNG for reproducible randomness.
        self._rng = np.random.default_rng(seed)

        # Dynamic disaster configuration.
        self.dynamic_disasters: bool = dynamic_disasters
        self.disaster_change_interval: int = disaster_change_interval
        # Separate RNG stream for dynamic updates so that enabling
        # dynamic mode does not disturb the main RNG sequence used
        # for start/goal placement.
        self._dynamic_rng = np.random.default_rng(
            self._rng.integers(0, 2**31) if seed is not None else None
        )

        # State variables (initialised on reset).
        self.grid: np.ndarray = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.agent_position: Tuple[int, int] = (0, 0)
        self.start_position: Tuple[int, int] = (0, 0)
        self.goal_position: Tuple[int, int] = (grid_size - 1, grid_size - 1)
        self.current_step: int = 0
        self.done: bool = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the grid and place the agent at the start.

        Returns:
            The initial state as a NumPy array.
        """
        self.grid = np.full((self.grid_size, self.grid_size), EMPTY, dtype=np.int32)

        # Place hazards.
        for r, c in self._obstacles:
            self.grid[r, c] = OBSTACLE
        for r, c in self._fire_cells:
            self.grid[r, c] = FIRE
        for r, c in self._flood_cells:
            self.grid[r, c] = FLOOD

        # Determine start & goal (random if not fixed).
        self.start_position = self._fixed_start or self._random_empty_cell()
        self.grid[self.start_position] = EMPTY  # keep start walkable

        self.goal_position = self._fixed_goal or self._random_empty_cell(
            exclude={self.start_position}
        )
        self.grid[self.goal_position] = GOAL

        # Place agent.
        self.agent_position = self.start_position
        self.current_step = 0
        self.done = False

        # Re-seed the dynamic-disaster RNG so each episode is
        # reproducible when a seed was supplied to __init__.
        self._dynamic_step_counter: int = 0
        self._dynamic_rng = np.random.default_rng(
            self._rng.integers(0, 2**31) if self.dynamic_disasters else None
        )

        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one action and return the transition.

        Args:
            action: Integer action — 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT.

        Returns:
            ``(next_state, reward, done, info)`` tuple.
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before step().")
        if action not in _ACTION_DELTAS:
            raise ValueError(f"Invalid action {action}. Must be 0-3.")

        self.current_step += 1

        # --- Dynamic disaster update (before processing action) ---
        disaster_update_details = self._maybe_update_disasters()

        dr, dc = _ACTION_DELTAS[action]
        new_row = self.agent_position[0] + dr
        new_col = self.agent_position[1] + dc

        # --- Boundary check ---
        if not self._in_bounds(new_row, new_col):
            reward = self._rewards["step_penalty"]
            info = self._build_info("boundary")
            info.update(disaster_update_details)
            if self.current_step >= self.max_steps:
                self.done = True
                info["termination"] = "max_steps"
            return self.get_state(), reward, self.done, info

        target_cell = self.grid[new_row, new_col]

        # --- Obstacle: agent stays in place ---
        if target_cell == OBSTACLE:
            reward = self._rewards["hit_obstacle"]
            info = self._build_info("obstacle")
            info.update(disaster_update_details)
            if self.current_step >= self.max_steps:
                self.done = True
                info["termination"] = "max_steps"
            return self.get_state(), reward, self.done, info

        # --- Move the agent ---
        self.agent_position = (new_row, new_col)

        # --- Goal reached ---
        if target_cell == GOAL:
            self.done = True
            reward = self._rewards["reach_safe_zone"]
            info = self._build_info("goal_reached")
            info.update(disaster_update_details)
            return self.get_state(), reward, self.done, info

        # --- Fire ---
        if target_cell == FIRE:
            self.done = True
            reward = self._rewards["enter_fire"]
            info = self._build_info("fire")
            info.update(disaster_update_details)
            return self.get_state(), reward, self.done, info

        # --- Flood ---
        if target_cell == FLOOD:
            self.done = True
            reward = self._rewards["enter_flood"]
            info = self._build_info("flood")
            info.update(disaster_update_details)
            return self.get_state(), reward, self.done, info

        # --- Normal step ---
        reward = self._rewards["step_penalty"]
        info = self._build_info("step")
        info.update(disaster_update_details)

        if self.current_step >= self.max_steps:
            self.done = True
            info["termination"] = "max_steps"

        return self.get_state(), reward, self.done, info

    def get_state(self) -> np.ndarray:
        """Return the current state as a 2-D grid with the agent marked.

        Returns:
            Copy of the grid with the agent's cell set to ``START``.
        """
        state = self.grid.copy()
        state[self.agent_position] = START
        return state

    def render(self) -> str:
        """Render the grid as a human-readable string.

        Returns:
            Multi-line string representation of the grid.
        """
        symbols = {
            EMPTY: ".",
            OBSTACLE: "#",
            FIRE: "F",
            FLOOD: "W",
            GOAL: "G",
            START: "A",
        }
        state = self.get_state()
        lines: List[str] = []
        for row in range(self.grid_size):
            cells = [symbols.get(int(state[row, col]), "?") for col in range(self.grid_size)]
            lines.append(" ".join(cells))
        rendered = "\n".join(lines)
        print(rendered)
        return rendered

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_action_space_size(self) -> int:
        """Return the number of possible actions (4)."""
        return NUM_ACTIONS

    def get_state_shape(self) -> Tuple[int, int]:
        """Return the shape of the state array."""
        return (self.grid_size, self.grid_size)

    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """Check whether a (row, col) is inside the grid."""
        r, c = position
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _in_bounds(self, row: int, col: int) -> bool:
        """Return True if (row, col) is within the grid."""
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size

    def _random_empty_cell(
        self, exclude: Optional[set] = None
    ) -> Tuple[int, int]:
        """Pick a random cell that is currently EMPTY.

        Args:
            exclude: Set of (row, col) to skip even if empty.

        Returns:
            A (row, col) tuple on an empty cell.
        """
        exclude = exclude or set()
        empty_cells = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if self.grid[r, c] == EMPTY and (r, c) not in exclude
        ]
        if not empty_cells:
            raise RuntimeError("No empty cells available for placement.")
        idx = self._rng.integers(len(empty_cells))
        return empty_cells[idx]

    def _maybe_update_disasters(self) -> Dict[str, Any]:
        """Check whether a dynamic disaster update is due and apply it.

        Returns:
            A dict with ``disaster_updated`` (bool) and, when an update
            occurred, ``new_fire_cells`` and ``new_flood_cells`` lists.
        """
        if not self.dynamic_disasters:
            return {"disaster_updated": False}

        self._dynamic_step_counter += 1
        if self._dynamic_step_counter < self.disaster_change_interval:
            return {"disaster_updated": False}

        # Time to update.
        self._dynamic_step_counter = 0
        return self._update_disasters()

    def _update_disasters(self) -> Dict[str, Any]:
        """Re-place fire and flood cells randomly on the grid.

        Obstacles remain fixed.  START (agent position) and GOAL are
        guaranteed safe.

        Returns:
            Info dict fragment with the new hazard positions.
        """
        # Protected cells: start, goal, agent position, and obstacles.
        protected = {
            self.start_position,
            self.goal_position,
            self.agent_position,
        }
        for r, c in self._obstacles:
            protected.add((r, c))

        # Clear existing fire & flood cells.
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] in (FIRE, FLOOD):
                    self.grid[r, c] = EMPTY

        # Determine candidate cells.
        candidates = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if self.grid[r, c] == EMPTY and (r, c) not in protected
        ]
        self._dynamic_rng.shuffle(candidates)  # type: ignore[arg-type]

        n_fire = len(self._fire_cells)
        n_flood = len(self._flood_cells)
        total_needed = n_fire + n_flood

        # Place as many as possible (grid may be small).
        placed = candidates[: min(total_needed, len(candidates))]
        new_fire: List[Tuple[int, int]] = []
        new_flood: List[Tuple[int, int]] = []

        for i, cell in enumerate(placed):
            if i < n_fire:
                self.grid[cell] = FIRE
                new_fire.append(cell)
            else:
                self.grid[cell] = FLOOD
                new_flood.append(cell)

        # Ensure GOAL marker survives (defensive).
        self.grid[self.goal_position] = GOAL

        return {
            "disaster_updated": True,
            "new_fire_cells": new_fire,
            "new_flood_cells": new_flood,
        }

    def _build_info(self, reason: str) -> Dict[str, Any]:
        """Build the info dict returned by ``step()``.

        Args:
            reason: Short description of what happened this step.

        Returns:
            Dictionary with position, cell type, step count, and reason.
        """
        r, c = self.agent_position
        return {
            "position": self.agent_position,
            "cell_type": int(self.grid[r, c]),
            "step": self.current_step,
            "termination": reason if self.done else None,
            "reason": reason,
        }

    def __repr__(self) -> str:
        return (
            f"DisasterEnv(grid_size={self.grid_size}, "
            f"start={self.start_position}, goal={self.goal_position})"
        )


# Backward-compatible alias so existing imports keep working.
DisasterGridEnv = DisasterEnv
