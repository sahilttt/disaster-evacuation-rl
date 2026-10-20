"""
Pydantic schemas for the FastAPI backend.

All request bodies and response models are defined here.
"""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


# =====================================================================
# Shared / Common
# =====================================================================

class HealthResponse(BaseModel):
    """GET /api/health response."""
    status: str = "ok"
    model_loaded: bool
    environment_ready: bool


# =====================================================================
# Results
# =====================================================================

class EvalMetrics(BaseModel):
    """Evaluation metrics for a single agent."""
    success_rate: float
    avg_reward: float
    avg_steps: float
    avg_path_length: float
    disaster_terminations: Optional[int] = None
    disaster_encounter_rate: Optional[float] = None


class AgentSummary(BaseModel):
    """Training + evaluation summary for a single agent."""
    training_episodes: int
    training_success_rate: float
    last_100_train_sr: float
    evaluation: EvalMetrics


class ResultsResponse(BaseModel):
    """GET /api/results response."""
    generated_at: Optional[str] = None
    q_learning: Optional[AgentSummary] = None
    dqn_static: Optional[AgentSummary] = None
    dqn_dynamic: Optional[AgentSummary] = None


# =====================================================================
# Simulation — Reset
# =====================================================================

class SimResetRequest(BaseModel):
    """POST /api/simulation/reset request body."""
    grid_size: int = Field(default=5, ge=3, le=20)
    max_steps: int = Field(default=50, ge=10, le=500)
    start_position: List[int] = Field(default=[0, 0])
    goal_position: List[int] = Field(default=[4, 4])
    obstacles: List[List[int]] = Field(default=[[2, 2], [1, 3]])
    fire_cells: List[List[int]] = Field(default=[[3, 1]])
    flood_cells: List[List[int]] = Field(default=[[1, 1]])
    dynamic_disasters: bool = True
    disaster_change_interval: int = Field(default=10, ge=1)
    seed: Optional[int] = None
    mode: Optional[str] = None
    generate_new: bool = False


class SimResetResponse(BaseModel):
    """POST /api/simulation/reset response."""
    grid: List[List[int]]
    grid_size: int
    agent_position: List[int]
    start_position: List[int]
    goal_position: List[int]
    done: bool
    current_step: int
    dynamic_disasters: bool


# =====================================================================
# Simulation — Step
# =====================================================================

class SimStepRequest(BaseModel):
    """POST /api/simulation/step request body.

    If ``action`` is null the trained DQN agent selects the action.
    """
    action: Optional[int] = Field(default=None, ge=0, le=3)


class SimStepResponse(BaseModel):
    """POST /api/simulation/step response."""
    grid: List[List[int]]
    agent_position: List[int]
    reward: float
    total_reward: float
    done: bool
    current_step: int
    action_taken: int
    info: Dict[str, Any]


# =====================================================================
# Simulation — Run (full episode)
# =====================================================================

class SimRunRequest(BaseModel):
    """POST /api/simulation/run request body."""
    grid_size: int = Field(default=5, ge=3, le=20)
    max_steps: int = Field(default=50, ge=10, le=500)
    start_position: List[int] = Field(default=[0, 0])
    goal_position: List[int] = Field(default=[4, 4])
    obstacles: List[List[int]] = Field(default=[[2, 2], [1, 3]])
    fire_cells: List[List[int]] = Field(default=[[3, 1]])
    flood_cells: List[List[int]] = Field(default=[[1, 1]])
    dynamic_disasters: bool = True
    disaster_change_interval: int = Field(default=10, ge=1)
    seed: Optional[int] = None
    mode: Optional[str] = None
    generate_new: bool = False


class RunStepRecord(BaseModel):
    """One step within a full-episode run."""
    step: int
    action: int
    agent_position: List[int]
    reward: float
    done: bool
    info: Dict[str, Any]


class SimRunResponse(BaseModel):
    """POST /api/simulation/run response."""
    success: bool
    total_reward: float
    total_steps: int
    termination_reason: Optional[str]
    path: List[List[int]]
    steps: List[RunStepRecord]
    final_grid: List[List[int]]
