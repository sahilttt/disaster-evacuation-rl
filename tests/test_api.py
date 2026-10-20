"""
Tests for the FastAPI backend.

Uses FastAPI's TestClient (backed by httpx) for synchronous testing.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app, sim_service


@pytest.fixture(autouse=True)
def _load_model():
    """Ensure the model is loaded before every test."""
    sim_service.load_model()
    yield


client = TestClient(app)


# =====================================================================
# Health
# =====================================================================

def test_health_ok() -> None:
    """GET /api/health returns 200 with status ok."""
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "environment_ready" in data


def test_health_model_loaded() -> None:
    """Health shows model_loaded=True after startup."""
    r = client.get("/api/health")
    assert r.json()["model_loaded"] is True


# =====================================================================
# Results
# =====================================================================

def test_results_returns_summary() -> None:
    """GET /api/results returns the experiment summary."""
    r = client.get("/api/results")
    assert r.status_code == 200
    data = r.json()
    assert "q_learning" in data
    assert "dqn_static" in data
    assert "dqn_dynamic" in data


def test_results_contains_eval_metrics() -> None:
    """Each agent block has evaluation metrics."""
    r = client.get("/api/results")
    data = r.json()
    for key in ("q_learning", "dqn_static", "dqn_dynamic"):
        block = data[key]
        assert "training_episodes" in block
        assert "evaluation" in block
        evl = block["evaluation"]
        assert "success_rate" in evl
        assert "avg_reward" in evl


# =====================================================================
# Simulation — Reset
# =====================================================================

def test_reset_default() -> None:
    """POST /api/simulation/reset with defaults returns valid grid."""
    r = client.post("/api/simulation/reset", json={})
    assert r.status_code == 200
    data = r.json()
    assert data["grid_size"] == 5
    assert data["done"] is False
    assert data["current_step"] == 0
    assert data["agent_position"] == [0, 0]
    assert data["goal_position"] == [4, 4]
    assert len(data["grid"]) == 5
    assert len(data["grid"][0]) == 5


def test_reset_custom_params() -> None:
    """Reset with custom grid size works."""
    r = client.post("/api/simulation/reset", json={
        "grid_size": 4,
        "max_steps": 20,
        "start_position": [0, 0],
        "goal_position": [3, 3],
        "obstacles": [[1, 1]],
        "fire_cells": [],
        "flood_cells": [],
        "dynamic_disasters": False,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["grid_size"] == 4
    assert data["dynamic_disasters"] is False


def test_reset_with_seed_reproducible() -> None:
    """Same seed produces the same grid."""
    r1 = client.post("/api/simulation/reset", json={"seed": 42})
    r2 = client.post("/api/simulation/reset", json={"seed": 42})
    assert r1.json()["grid"] == r2.json()["grid"]


# =====================================================================
# Simulation — Step
# =====================================================================

def test_step_without_reset_fails() -> None:
    """Step without an active simulation returns 400."""
    sim_service.env = None
    r = client.post("/api/simulation/step", json={})
    assert r.status_code == 400


def test_step_with_agent_action() -> None:
    """Step with no action lets the agent decide."""
    client.post("/api/simulation/reset", json={})
    r = client.post("/api/simulation/step", json={})
    assert r.status_code == 200
    data = r.json()
    assert "agent_position" in data
    assert "reward" in data
    assert "action_taken" in data
    assert isinstance(data["action_taken"], int)
    assert 0 <= data["action_taken"] <= 3
    assert data["current_step"] == 1


def test_step_with_manual_action() -> None:
    """Step with an explicit action uses that action."""
    client.post("/api/simulation/reset", json={})
    r = client.post("/api/simulation/step", json={"action": 1})  # DOWN
    assert r.status_code == 200
    data = r.json()
    assert data["action_taken"] == 1
    assert data["agent_position"] == [1, 0]  # moved down from (0,0)


def test_step_returns_done_info() -> None:
    """After enough steps the episode terminates."""
    client.post("/api/simulation/reset", json={"max_steps": 10})
    done = False
    for _ in range(15):
        r = client.post("/api/simulation/step", json={})
        if r.status_code != 200:
            break
        if r.json()["done"]:
            done = True
            break
    assert done


def test_step_after_done_fails() -> None:
    """Stepping after done returns 400."""
    # Run a tiny episode that ends quickly.
    client.post("/api/simulation/reset", json={
        "grid_size": 3,
        "start_position": [0, 0],
        "goal_position": [0, 1],
        "obstacles": [],
        "fire_cells": [],
        "flood_cells": [],
    })
    # Move right to reach goal.
    client.post("/api/simulation/step", json={"action": 3})
    # Should fail now.
    r = client.post("/api/simulation/step", json={"action": 3})
    assert r.status_code == 400


# =====================================================================
# Simulation — Run
# =====================================================================

def test_run_default_episode() -> None:
    """POST /api/simulation/run returns a full episode."""
    r = client.post("/api/simulation/run", json={})
    assert r.status_code == 200
    data = r.json()
    assert "success" in data
    assert "total_reward" in data
    assert "total_steps" in data
    assert "path" in data
    assert "steps" in data
    assert "final_grid" in data
    assert isinstance(data["steps"], list)
    assert len(data["steps"]) > 0


def test_run_produces_success() -> None:
    """The trained agent should reach the goal on the default grid."""
    r = client.post("/api/simulation/run", json={"seed": 42})
    data = r.json()
    assert data["success"] is True
    assert data["termination_reason"] == "goal_reached"


def test_run_path_is_consistent() -> None:
    """The path starts at start and ends at final agent position."""
    r = client.post("/api/simulation/run", json={"seed": 99})
    data = r.json()
    path = data["path"]
    assert path[0] == [0, 0]  # start
    assert len(path) == data["total_steps"] + 1  # start + each step


def test_run_seeded_reproducibility() -> None:
    """Two runs with the same seed produce identical results."""
    r1 = client.post("/api/simulation/run", json={"seed": 77})
    r2 = client.post("/api/simulation/run", json={"seed": 77})
    assert r1.json()["path"] == r2.json()["path"]
    assert r1.json()["total_reward"] == r2.json()["total_reward"]


def test_run_static_mode() -> None:
    """Running in static mode also works."""
    r = client.post("/api/simulation/run", json={
        "dynamic_disasters": False,
        "seed": 42,
    })
    assert r.status_code == 200
    assert r.json()["success"] is True


# =====================================================================
# Integration — Single-port frontend serving
# =====================================================================

def test_root_serves_html() -> None:
    """GET / returns the React index.html (text/html)."""
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert b"<div id=\"root\">" in r.content


def test_spa_fallback_returns_index() -> None:
    """Non-API, non-file paths return index.html (SPA routing)."""
    r = client.get("/some/client/route")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert b"<div id=\"root\">" in r.content


def test_assets_served() -> None:
    """Static JS/CSS assets from /assets/ are served correctly."""
    import os
    assets_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "frontend", "dist", "assets",
    )
    if not os.path.isdir(assets_dir):
        pytest.skip("frontend/dist/assets not built")
    files = os.listdir(assets_dir)
    js_files = [f for f in files if f.endswith(".js")]
    css_files = [f for f in files if f.endswith(".css")]
    if js_files:
        r = client.get(f"/assets/{js_files[0]}")
        assert r.status_code == 200
        assert "javascript" in r.headers.get("content-type", "")
    if css_files:
        r = client.get(f"/assets/{css_files[0]}")
        assert r.status_code == 200
        assert "css" in r.headers.get("content-type", "")


def test_api_routes_take_priority() -> None:
    """API routes still return JSON, not the SPA index."""
    r = client.get("/api/health")
    assert r.status_code == 200
    assert "application/json" in r.headers.get("content-type", "")
    assert r.json()["status"] == "ok"


# =====================================================================
# Integration — Full simulation lifecycle
# =====================================================================

def test_full_lifecycle() -> None:
    """End-to-end: health → reset → step(s) → run → results."""
    # 1. Health
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True

    # 2. Results
    r = client.get("/api/results")
    assert r.status_code == 200
    assert "dqn_dynamic" in r.json()

    # 3. Reset
    r = client.post("/api/simulation/reset", json={"seed": 123})
    assert r.status_code == 200
    reset_data = r.json()
    assert reset_data["done"] is False
    assert reset_data["current_step"] == 0

    # 4. Step (agent chooses)
    r = client.post("/api/simulation/step", json={})
    assert r.status_code == 200
    step_data = r.json()
    assert step_data["current_step"] == 1
    assert 0 <= step_data["action_taken"] <= 3

    # 5. Run a full episode (separate from the reset/step session)
    r = client.post("/api/simulation/run", json={"seed": 42})
    assert r.status_code == 200
    run_data = r.json()
    assert run_data["success"] is True
    assert run_data["total_steps"] > 0
    assert len(run_data["path"]) == run_data["total_steps"] + 1

