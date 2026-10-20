"""
Service layer -- manages the simulation state and trained model.

This module is imported by the FastAPI routes and holds a singleton
``SimulationService`` that owns the environment and agent.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from environment.disaster_env import DisasterEnv
from agents.dqn_agent import DQNAgent
from utils.helpers import load_json


# ── Paths ─────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH = str(_PROJECT_ROOT / "experiments" / "results" / "dqn_dynamic.pth")
_SUMMARY_PATH = str(_PROJECT_ROOT / "experiments" / "results" / "experiment_summary.json")


# =====================================================================
# Simulation service
# =====================================================================

class SimulationService:
    """Manages environment + trained agent for the API.

    Attributes:
        env: The current ``DisasterEnv`` instance (or None).
        agent: The loaded ``DQNAgent`` (or None).
        total_reward: Accumulated reward for the current episode.
    """

    def __init__(self) -> None:
        self.env: Optional[DisasterEnv] = None
        self.agent: Optional[DQNAgent] = None
        self.total_reward: float = 0.0
        self._model_loaded: bool = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, model_path: str = _MODEL_PATH) -> bool:
        """Load the trained DQN model from disk.

        Returns:
            True if the model was loaded successfully.
        """
        if not os.path.exists(model_path):
            return False

        # The training used grid_size=5 -> state_shape=(5,5).
        self.agent = DQNAgent(
            state_shape=(5, 5),
            hidden_size=128,
            epsilon=0.0,       # greedy -- no exploration
            min_epsilon=0.0,
        )
        self.agent.load(model_path)
        self.agent.epsilon = 0.0  # ensure greedy after load
        self._model_loaded = True
        return True

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def environment_ready(self) -> bool:
        return self.env is not None and not self.env.done

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        grid_size: int = 5,
        max_steps: int = 50,
        start_position: Tuple[int, int] = (0, 0),
        goal_position: Tuple[int, int] = (4, 4),
        obstacles: Optional[List[Tuple[int, int]]] = None,
        fire_cells: Optional[List[Tuple[int, int]]] = None,
        flood_cells: Optional[List[Tuple[int, int]]] = None,
        dynamic_disasters: bool = True,
        disaster_change_interval: int = 10,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Create a fresh environment and return the initial state."""
        self.env = DisasterEnv(
            grid_size=grid_size,
            max_steps=max_steps,
            start_position=start_position,
            goal_position=goal_position,
            obstacles=obstacles,
            fire_cells=fire_cells,
            flood_cells=flood_cells,
            dynamic_disasters=dynamic_disasters,
            disaster_change_interval=disaster_change_interval,
            seed=seed,
        )
        self.total_reward = 0.0
        return self.env.reset()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: Optional[int] = None) -> Dict[str, Any]:
        """Execute one step.  If *action* is ``None``, ask the agent.

        Returns:
            Dict with keys: grid, agent_position, reward, total_reward,
            done, current_step, action_taken, info.

        Raises:
            RuntimeError: If the environment has not been reset.
        """
        if self.env is None:
            raise RuntimeError("Environment not initialised. Call reset first.")

        # Pick action.
        if action is not None:
            act = action
        elif self.agent is not None:
            state = self.env.get_state()
            act = self.agent.choose_action(state, training=False)
        else:
            raise RuntimeError("No action provided and no model loaded.")

        next_state, reward, done, info = self.env.step(act)
        self.total_reward += reward

        return {
            "grid": self.env.grid.tolist(),
            "agent_position": list(self.env.agent_position),
            "reward": float(reward),
            "total_reward": float(self.total_reward),
            "done": done,
            "current_step": self.env.current_step,
            "action_taken": act,
            "info": _sanitise_info(info),
        }

    # ------------------------------------------------------------------
    # Full episode run
    # ------------------------------------------------------------------

    def run_episode(
        self,
        grid_size: int = 5,
        max_steps: int = 50,
        start_position: Tuple[int, int] = (0, 0),
        goal_position: Tuple[int, int] = (4, 4),
        obstacles: Optional[List[Tuple[int, int]]] = None,
        fire_cells: Optional[List[Tuple[int, int]]] = None,
        flood_cells: Optional[List[Tuple[int, int]]] = None,
        dynamic_disasters: bool = True,
        disaster_change_interval: int = 10,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Reset + run a full episode with the trained agent.

        Returns:
            Dict with success, total_reward, total_steps,
            termination_reason, path, steps, final_grid.
        """
        if self.agent is None:
            raise RuntimeError("Model not loaded. Cannot run episode.")

        self.reset(
            grid_size=grid_size,
            max_steps=max_steps,
            start_position=start_position,
            goal_position=goal_position,
            obstacles=obstacles,
            fire_cells=fire_cells,
            flood_cells=flood_cells,
            dynamic_disasters=dynamic_disasters,
            disaster_change_interval=disaster_change_interval,
            seed=seed,
        )

        path: List[List[int]] = [list(self.env.agent_position)]
        step_records: List[Dict[str, Any]] = []
        done = False
        info: Dict[str, Any] = {}

        while not done:
            result = self.step(action=None)
            done = result["done"]
            info = result["info"]
            path.append(result["agent_position"])
            step_records.append({
                "step": result["current_step"],
                "action": result["action_taken"],
                "agent_position": result["agent_position"],
                "reward": result["reward"],
                "done": done,
                "info": info,
            })

        return {
            "success": info.get("reason") == "goal_reached",
            "total_reward": float(self.total_reward),
            "total_steps": self.env.current_step,
            "termination_reason": info.get("reason"),
            "path": path,
            "steps": step_records,
            "final_grid": self.env.grid.tolist(),
        }

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    @staticmethod
    def get_experiment_results() -> Dict[str, Any]:
        """Load the saved experiment summary JSON."""
        if not os.path.exists(_SUMMARY_PATH):
            return {}
        return load_json(_SUMMARY_PATH)


# ── Helpers ───────────────────────────────────────────────────────────

def _sanitise_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """Make an info dict JSON-serialisable."""
    out: Dict[str, Any] = {}
    for k, v in info.items():
        if isinstance(v, tuple):
            out[k] = list(v)
        elif isinstance(v, (np.integer, np.int64, np.int32)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


# ── Module-level singleton ────────────────────────────────────────────
sim_service = SimulationService()
