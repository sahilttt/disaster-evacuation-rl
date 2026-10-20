"""
FastAPI application -- disaster evacuation RL backend.

Endpoints
---------
GET  /api/health            Backend status
GET  /api/results           Saved experiment metrics
POST /api/simulation/reset  Create / reset a simulation
POST /api/simulation/step   Execute one step
POST /api/simulation/run    Run a complete evacuation episode

Single-port architecture
------------------------
In production the React/Vite build is served from ``frontend/dist``
as static files.  ``/api/*`` routes always take priority.  No
separate frontend server is needed.

Usage:
    uvicorn api.main:app --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.schemas import (
    HealthResponse,
    ResultsResponse,
    SimResetRequest,
    SimResetResponse,
    SimStepRequest,
    SimStepResponse,
    SimRunRequest,
    SimRunResponse,
    RunStepRecord,
)
from api.services import sim_service


# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(application: FastAPI):
    """Load the trained DQN model and initialise a default environment."""
    sim_service.load_model()
    sim_service.reset()          # default 5×5 dynamic env → environment_ready=True
    yield


# ── App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Disaster Evacuation RL API",
    version="1.0.0",
    description="Backend for the disaster evacuation reinforcement-learning project.",
    lifespan=_lifespan,
)

# CORS -- allow the future React frontend (and any origin during dev).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# Endpoints
# =====================================================================

@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return backend health / readiness."""
    return HealthResponse(
        status="ok",
        model_loaded=sim_service.model_loaded,
        environment_ready=sim_service.environment_ready,
    )


@app.get("/api/results", response_model=ResultsResponse)
def results() -> ResultsResponse:
    """Return saved experiment metrics."""
    data = sim_service.get_experiment_results()
    if not data:
        raise HTTPException(status_code=404, detail="No experiment results found.")
    return ResultsResponse(**data)


@app.post("/api/simulation/reset", response_model=SimResetResponse)
def simulation_reset(body: SimResetRequest) -> SimResetResponse:
    """Create or reset a simulation environment."""
    dyn_dis = body.dynamic_disasters
    if body.mode == "dynamic":
        dyn_dis = True
    elif body.mode == "static":
        dyn_dis = False

    obstacles = [tuple(o) for o in body.obstacles]
    fire_cells = [tuple(f) for f in body.fire_cells]
    flood_cells = [tuple(f) for f in body.flood_cells]

    if body.generate_new:
        import random
        start = tuple(body.start_position)
        goal = tuple(body.goal_position)
        all_cells = [(r, c) for r in range(body.grid_size) for c in range(body.grid_size) if (r, c) not in (start, goal)]
        random.shuffle(all_cells)
        # Assuming we need exactly the same number of hazards as defaults, or let's use 2 obs, 1 fire, 1 flood
        obstacles = [all_cells.pop() for _ in range(2)] if len(all_cells) >= 2 else []
        fire_cells = [all_cells.pop() for _ in range(1)] if len(all_cells) >= 1 else []
        flood_cells = [all_cells.pop() for _ in range(1)] if len(all_cells) >= 1 else []

    state = sim_service.reset(
        grid_size=body.grid_size,
        max_steps=body.max_steps,
        start_position=tuple(body.start_position),
        goal_position=tuple(body.goal_position),
        obstacles=obstacles,
        fire_cells=fire_cells,
        flood_cells=flood_cells,
        dynamic_disasters=dyn_dis,
        disaster_change_interval=body.disaster_change_interval,
        seed=body.seed,
    )
    env = sim_service.env
    return SimResetResponse(
        grid=env.grid.tolist(),
        grid_size=env.grid_size,
        agent_position=list(env.agent_position),
        start_position=list(env.start_position),
        goal_position=list(env.goal_position),
        done=env.done,
        current_step=env.current_step,
        dynamic_disasters=env.dynamic_disasters,
    )


@app.post("/api/simulation/step", response_model=SimStepResponse)
def simulation_step(body: SimStepRequest) -> SimStepResponse:
    """Execute one environment step."""
    if sim_service.env is None:
        raise HTTPException(status_code=400, detail="No active simulation. Call /reset first.")
    if sim_service.env.done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset first.")

    try:
        result = sim_service.step(action=body.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return SimStepResponse(**result)


@app.post("/api/simulation/run", response_model=SimRunResponse)
def simulation_run(body: SimRunRequest) -> SimRunResponse:
    """Run a complete evacuation episode with the trained agent."""
    if not sim_service.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    dyn_dis = body.dynamic_disasters
    if body.mode == "dynamic":
        dyn_dis = True
    elif body.mode == "static":
        dyn_dis = False

    obstacles = [tuple(o) for o in body.obstacles]
    fire_cells = [tuple(f) for f in body.fire_cells]
    flood_cells = [tuple(f) for f in body.flood_cells]

    if body.generate_new:
        import random
        start = tuple(body.start_position)
        goal = tuple(body.goal_position)
        all_cells = [(r, c) for r in range(body.grid_size) for c in range(body.grid_size) if (r, c) not in (start, goal)]
        random.shuffle(all_cells)
        obstacles = [all_cells.pop() for _ in range(2)] if len(all_cells) >= 2 else []
        fire_cells = [all_cells.pop() for _ in range(1)] if len(all_cells) >= 1 else []
        flood_cells = [all_cells.pop() for _ in range(1)] if len(all_cells) >= 1 else []

    try:
        result = sim_service.run_episode(
            grid_size=body.grid_size,
            max_steps=body.max_steps,
            start_position=tuple(body.start_position),
            goal_position=tuple(body.goal_position),
            obstacles=obstacles,
            fire_cells=fire_cells,
            flood_cells=flood_cells,
            dynamic_disasters=dyn_dis,
            disaster_change_interval=body.disaster_change_interval,
            seed=body.seed,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return SimRunResponse(
        success=result["success"],
        total_reward=result["total_reward"],
        total_steps=result["total_steps"],
        termination_reason=result["termination_reason"],
        path=result["path"],
        steps=[RunStepRecord(**s) for s in result["steps"]],
        final_grid=result["final_grid"],
    )


# =====================================================================
# Static file serving — React production build  (single-port mode)
# =====================================================================
# The React/Vite build is expected at ``frontend/dist``.  If that
# directory exists we:
#   1. Mount ``/assets`` as a StaticFiles sub-app so JS/CSS get proper
#      MIME types and cache headers.
#   2. Add a catch-all GET that serves ``index.html`` for client-side
#      routing (SPA fallback).
# All ``/api/*`` routes are registered above and always take priority.

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if _FRONTEND_DIR.is_dir():
    # Static assets (JS, CSS, images) — must be mounted before the
    # catch-all route so Starlette serves them directly.
    _ASSETS_DIR = _FRONTEND_DIR / "assets"
    if _ASSETS_DIR.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=str(_ASSETS_DIR)),
            name="static-assets",
        )

    _INDEX_HTML = _FRONTEND_DIR / "index.html"

    @app.get("/{full_path:path}")
    async def _spa_fallback(full_path: str) -> FileResponse:
        """Serve static files from the build, or index.html for SPA routes."""
        # Resolve and guard against path traversal.
        requested = (_FRONTEND_DIR / full_path).resolve()
        if (
            requested.is_file()
            and str(requested).startswith(str(_FRONTEND_DIR))
        ):
            return FileResponse(requested)
        return FileResponse(_INDEX_HTML)
