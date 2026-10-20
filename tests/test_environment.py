"""
Tests for the DisasterEnv grid-world environment.

Covers:
  1. reset
  2. valid actions
  3. boundary handling
  4. obstacle handling
  5. goal termination
  6. disaster (fire / flood) termination
"""

import numpy as np
from environment.disaster_env import (
    DisasterEnv,
    DisasterGridEnv,
    EMPTY,
    OBSTACLE,
    FIRE,
    FLOOD,
    GOAL,
    START,
    UP,
    DOWN,
    LEFT,
    RIGHT,
)
from config.config import NUM_ACTIONS, REWARD_CONFIG


# =====================================================================
# 1. Reset
# =====================================================================

def test_reset_returns_ndarray() -> None:
    """reset() returns a numpy array of the correct shape."""
    env = DisasterEnv(grid_size=8, start_position=(0, 0), goal_position=(7, 7))
    state = env.reset()
    assert isinstance(state, np.ndarray)
    assert state.shape == (8, 8)


def test_reset_clears_done() -> None:
    """After reset, done is False and step count is 0."""
    env = DisasterEnv(grid_size=5, max_steps=3,
                      start_position=(0, 0), goal_position=(4, 4))
    env.reset()
    # Exhaust the episode.
    for _ in range(3):
        env.step(DOWN)
    assert env.done is True
    # Reset should clear it.
    env.reset()
    assert env.done is False
    assert env.current_step == 0


def test_reset_places_goal() -> None:
    """After reset, the goal cell is marked on the grid."""
    env = DisasterEnv(grid_size=5, start_position=(0, 0), goal_position=(4, 4))
    env.reset()
    assert env.grid[4, 4] == GOAL


def test_reset_agent_at_start() -> None:
    """After reset, the agent is at the start position."""
    env = DisasterEnv(grid_size=5, start_position=(1, 2), goal_position=(4, 4))
    env.reset()
    assert env.agent_position == (1, 2)


def test_reset_random_positions() -> None:
    """When no fixed positions given, start and goal are randomly chosen."""
    env = DisasterEnv(grid_size=5, seed=42)
    env.reset()
    assert env.is_valid_position(env.start_position)
    assert env.is_valid_position(env.goal_position)
    assert env.start_position != env.goal_position


# =====================================================================
# 2. Valid actions
# =====================================================================

def test_step_returns_tuple() -> None:
    """step() returns (state, reward, done, info)."""
    env = DisasterEnv(grid_size=5, start_position=(2, 2), goal_position=(4, 4))
    env.reset()
    result = env.step(DOWN)
    assert len(result) == 4
    state, reward, done, info = result
    assert isinstance(state, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_move_down() -> None:
    """Moving DOWN increases the row by 1."""
    env = DisasterEnv(grid_size=5, start_position=(1, 1), goal_position=(4, 4))
    env.reset()
    env.step(DOWN)
    assert env.agent_position == (2, 1)


def test_move_up() -> None:
    """Moving UP decreases the row by 1."""
    env = DisasterEnv(grid_size=5, start_position=(2, 2), goal_position=(4, 4))
    env.reset()
    env.step(UP)
    assert env.agent_position == (1, 2)


def test_move_left() -> None:
    """Moving LEFT decreases the column by 1."""
    env = DisasterEnv(grid_size=5, start_position=(2, 2), goal_position=(4, 4))
    env.reset()
    env.step(LEFT)
    assert env.agent_position == (2, 1)


def test_move_right() -> None:
    """Moving RIGHT increases the column by 1."""
    env = DisasterEnv(grid_size=5, start_position=(2, 2), goal_position=(4, 4))
    env.reset()
    env.step(RIGHT)
    assert env.agent_position == (2, 3)


def test_invalid_action_raises() -> None:
    """An invalid action integer raises ValueError."""
    env = DisasterEnv(grid_size=5, start_position=(0, 0), goal_position=(4, 4))
    env.reset()
    try:
        env.step(99)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_step_after_done_raises() -> None:
    """Calling step() after the episode ends raises RuntimeError."""
    env = DisasterEnv(grid_size=5, start_position=(0, 0), goal_position=(0, 1))
    env.reset()
    env.step(RIGHT)  # reaches goal -> done
    assert env.done is True
    try:
        env.step(RIGHT)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass


# =====================================================================
# 3. Boundary handling
# =====================================================================

def test_boundary_top() -> None:
    """Agent cannot move above row 0."""
    env = DisasterEnv(grid_size=5, start_position=(0, 2), goal_position=(4, 4))
    env.reset()
    env.step(UP)
    assert env.agent_position == (0, 2)  # stayed


def test_boundary_left() -> None:
    """Agent cannot move left of column 0."""
    env = DisasterEnv(grid_size=5, start_position=(2, 0), goal_position=(4, 4))
    env.reset()
    env.step(LEFT)
    assert env.agent_position == (2, 0)


def test_boundary_bottom() -> None:
    """Agent cannot move below the last row."""
    env = DisasterEnv(grid_size=5, start_position=(4, 2), goal_position=(0, 0))
    env.reset()
    env.step(DOWN)
    assert env.agent_position == (4, 2)


def test_boundary_right() -> None:
    """Agent cannot move right of the last column."""
    env = DisasterEnv(grid_size=5, start_position=(2, 4), goal_position=(0, 0))
    env.reset()
    env.step(RIGHT)
    assert env.agent_position == (2, 4)


def test_boundary_gives_step_penalty() -> None:
    """Hitting the boundary gives the step penalty, not a special reward."""
    env = DisasterEnv(grid_size=5, start_position=(0, 0), goal_position=(4, 4))
    env.reset()
    _, reward, _, info = env.step(UP)
    assert reward == REWARD_CONFIG["step_penalty"]
    assert info["reason"] == "boundary"


# =====================================================================
# 4. Obstacle handling
# =====================================================================

def test_obstacle_blocks_movement() -> None:
    """Agent cannot walk into an obstacle cell."""
    env = DisasterEnv(
        grid_size=5,
        start_position=(2, 2),
        goal_position=(4, 4),
        obstacles=[(2, 3)],
    )
    env.reset()
    env.step(RIGHT)  # (2,3) is obstacle
    assert env.agent_position == (2, 2)  # stayed


def test_obstacle_gives_penalty() -> None:
    """Walking into an obstacle gives the hit_obstacle penalty."""
    env = DisasterEnv(
        grid_size=5,
        start_position=(2, 2),
        goal_position=(4, 4),
        obstacles=[(2, 3)],
    )
    env.reset()
    _, reward, done, info = env.step(RIGHT)
    assert reward == REWARD_CONFIG["hit_obstacle"]
    assert done is False
    assert info["reason"] == "obstacle"


# =====================================================================
# 5. Goal termination
# =====================================================================

def test_goal_ends_episode() -> None:
    """Reaching the goal sets done=True."""
    env = DisasterEnv(grid_size=5, start_position=(0, 0), goal_position=(0, 1))
    env.reset()
    _, _, done, info = env.step(RIGHT)
    assert done is True
    assert info["reason"] == "goal_reached"


def test_goal_gives_positive_reward() -> None:
    """Reaching the goal gives a large positive reward."""
    env = DisasterEnv(grid_size=5, start_position=(0, 0), goal_position=(0, 1))
    env.reset()
    _, reward, _, _ = env.step(RIGHT)
    assert reward == REWARD_CONFIG["reach_safe_zone"]
    assert reward > 0


# =====================================================================
# 6. Disaster termination (fire & flood)
# =====================================================================

def test_fire_ends_episode() -> None:
    """Entering a fire cell ends the episode with a negative reward."""
    env = DisasterEnv(
        grid_size=5,
        start_position=(1, 1),
        goal_position=(4, 4),
        fire_cells=[(1, 2)],
    )
    env.reset()
    _, reward, done, info = env.step(RIGHT)
    assert done is True
    assert reward == REWARD_CONFIG["enter_fire"]
    assert reward < 0
    assert info["reason"] == "fire"


def test_flood_ends_episode() -> None:
    """Entering a flood cell ends the episode with a negative reward."""
    env = DisasterEnv(
        grid_size=5,
        start_position=(1, 1),
        goal_position=(4, 4),
        flood_cells=[(1, 2)],
    )
    env.reset()
    _, reward, done, info = env.step(RIGHT)
    assert done is True
    assert reward == REWARD_CONFIG["enter_flood"]
    assert reward < 0
    assert info["reason"] == "flood"


# =====================================================================
# Additional: max steps, info, render, helpers, backward compat
# =====================================================================

def test_max_steps_ends_episode() -> None:
    """Episode terminates when max_steps is reached."""
    env = DisasterEnv(
        grid_size=5, max_steps=3,
        start_position=(0, 0), goal_position=(4, 4),
    )
    env.reset()
    for _ in range(3):
        _, _, done, info = env.step(DOWN)
    assert done is True
    assert info.get("termination") == "max_steps"


def test_step_penalty_on_normal_move() -> None:
    """A normal step gives a small negative reward."""
    env = DisasterEnv(grid_size=5, start_position=(2, 2), goal_position=(4, 4))
    env.reset()
    _, reward, _, _ = env.step(DOWN)
    assert reward == REWARD_CONFIG["step_penalty"]
    assert reward < 0


def test_info_contains_position() -> None:
    """info dict contains the agent's position after a step."""
    env = DisasterEnv(grid_size=5, start_position=(2, 2), goal_position=(4, 4))
    env.reset()
    _, _, _, info = env.step(DOWN)
    assert info["position"] == (3, 2)


def test_get_state_marks_agent() -> None:
    """get_state() returns a grid with the agent cell marked as START."""
    env = DisasterEnv(grid_size=5, start_position=(0, 0), goal_position=(4, 4))
    env.reset()
    state = env.get_state()
    assert state[0, 0] == START


def test_render_returns_string() -> None:
    """render() returns a multi-line string representation."""
    env = DisasterEnv(grid_size=3, start_position=(0, 0), goal_position=(2, 2))
    env.reset()
    output = env.render()
    assert isinstance(output, str)
    assert "A" in output  # agent
    assert "G" in output  # goal


def test_action_space_size() -> None:
    """Action space size is 4."""
    env = DisasterEnv()
    assert env.get_action_space_size() == NUM_ACTIONS


def test_state_shape() -> None:
    """State shape matches the grid dimensions."""
    env = DisasterEnv(grid_size=7)
    assert env.get_state_shape() == (7, 7)


def test_is_valid_position() -> None:
    """Position validation works for in-bounds and out-of-bounds."""
    env = DisasterEnv(grid_size=5)
    assert env.is_valid_position((0, 0)) is True
    assert env.is_valid_position((4, 4)) is True
    assert env.is_valid_position((-1, 0)) is False
    assert env.is_valid_position((0, 5)) is False


def test_repr() -> None:
    """__repr__ returns a readable string."""
    env = DisasterEnv(grid_size=5, start_position=(0, 0), goal_position=(4, 4))
    env.reset()
    r = repr(env)
    assert "DisasterEnv" in r
    assert "grid_size=5" in r


def test_backward_compat_alias() -> None:
    """DisasterGridEnv is an alias for DisasterEnv."""
    assert DisasterGridEnv is DisasterEnv


def test_hazards_preserved_across_resets() -> None:
    """Obstacles, fire, and flood are placed on every reset."""
    env = DisasterEnv(
        grid_size=5,
        start_position=(0, 0),
        goal_position=(4, 4),
        obstacles=[(1, 1)],
        fire_cells=[(2, 2)],
        flood_cells=[(3, 3)],
    )
    for _ in range(3):
        env.reset()
        assert env.grid[1, 1] == OBSTACLE
        assert env.grid[2, 2] == FIRE
        assert env.grid[3, 3] == FLOOD


# =====================================================================
# 7. Dynamic disasters
# =====================================================================

def test_dynamic_mode_enabled() -> None:
    """When dynamic_disasters=True, disasters update at the interval."""
    env = DisasterEnv(
        grid_size=5,
        max_steps=50,
        start_position=(0, 0),
        goal_position=(4, 4),
        fire_cells=[(2, 2)],
        flood_cells=[(3, 3)],
        dynamic_disasters=True,
        disaster_change_interval=5,
        seed=42,
    )
    env.reset()
    # Initial fire/flood positions.
    initial_fire = set(zip(*np.where(env.grid == FIRE)))
    initial_flood = set(zip(*np.where(env.grid == FLOOD)))

    # Step 5 times to trigger the first update.
    info = None
    for _ in range(5):
        _, _, _, info = env.step(DOWN)

    # The 5th step should trigger a disaster update.
    assert info["disaster_updated"] is True
    assert "new_fire_cells" in info
    assert "new_flood_cells" in info

    # After update, grid should have fire and flood (possibly at new positions).
    updated_fire = set(zip(*np.where(env.grid == FIRE)))
    updated_flood = set(zip(*np.where(env.grid == FLOOD)))
    assert len(updated_fire) == 1  # same count as original
    assert len(updated_flood) == 1


def test_dynamic_interval_timing() -> None:
    """Disaster updates happen exactly at multiples of the interval."""
    env = DisasterEnv(
        grid_size=5,
        max_steps=50,
        start_position=(0, 0),
        goal_position=(4, 4),
        fire_cells=[(2, 2)],
        dynamic_disasters=True,
        disaster_change_interval=3,
        seed=7,
    )
    env.reset()

    update_steps = []
    for i in range(12):
        _, _, _, info = env.step(DOWN if i < 4 else UP)
        if info.get("disaster_updated"):
            update_steps.append(info["step"])

    # Updates at steps 3, 6, 9, 12.
    assert update_steps == [3, 6, 9, 12]


def test_dynamic_no_update_before_interval() -> None:
    """No disaster update before the interval is reached."""
    env = DisasterEnv(
        grid_size=5,
        max_steps=50,
        start_position=(0, 0),
        goal_position=(4, 4),
        fire_cells=[(2, 2)],
        dynamic_disasters=True,
        disaster_change_interval=5,
        seed=42,
    )
    env.reset()
    for i in range(4):
        _, _, _, info = env.step(DOWN if i < 4 else UP)
        assert info["disaster_updated"] is False


def test_dynamic_start_goal_safety() -> None:
    """START and GOAL cells are never placed with fire or flood."""
    env = DisasterEnv(
        grid_size=5,
        max_steps=100,
        start_position=(0, 0),
        goal_position=(4, 4),
        fire_cells=[(1, 1), (1, 2), (2, 1)],
        flood_cells=[(3, 1), (3, 2)],
        dynamic_disasters=True,
        disaster_change_interval=3,
        seed=123,
    )
    env.reset()
    for i in range(30):
        _, _, done, info = env.step(DOWN if i % 2 == 0 else UP)
        if done:
            env.reset()
        # After every step, verify start and goal are safe.
        assert env.grid[0, 0] != FIRE
        assert env.grid[0, 0] != FLOOD
        assert env.grid[4, 4] == GOAL  # goal is always marked as GOAL


def test_dynamic_seeded_reproducibility() -> None:
    """Two environments with the same seed produce identical dynamic updates."""
    kwargs = dict(
        grid_size=5,
        max_steps=50,
        start_position=(0, 0),
        goal_position=(4, 4),
        fire_cells=[(2, 2)],
        flood_cells=[(3, 3)],
        dynamic_disasters=True,
        disaster_change_interval=3,
        seed=99,
    )
    env1 = DisasterEnv(**kwargs)
    env2 = DisasterEnv(**kwargs)

    env1.reset()
    env2.reset()

    actions = [DOWN, DOWN, DOWN, UP, UP, UP, DOWN, DOWN, DOWN]
    for action in actions:
        s1, r1, d1, i1 = env1.step(action)
        s2, r2, d2, i2 = env2.step(action)
        np.testing.assert_array_equal(s1, s2)
        assert r1 == r2
        assert d1 == d2
        assert i1 == i2
        if d1:
            env1.reset()
            env2.reset()


def test_dynamic_reproducibility_across_resets() -> None:
    """Dynamic updates are reproducible even across multiple resets."""
    kwargs = dict(
        grid_size=5,
        max_steps=50,
        start_position=(0, 0),
        goal_position=(4, 4),
        fire_cells=[(2, 2)],
        dynamic_disasters=True,
        disaster_change_interval=2,
        seed=77,
    )
    env1 = DisasterEnv(**kwargs)
    env2 = DisasterEnv(**kwargs)

    for episode in range(3):
        env1.reset()
        env2.reset()
        for step in range(6):
            action = DOWN if step % 2 == 0 else UP
            s1, r1, d1, i1 = env1.step(action)
            s2, r2, d2, i2 = env2.step(action)
            np.testing.assert_array_equal(s1, s2)
            assert i1 == i2
            if d1:
                break


def test_static_mode_backward_compat() -> None:
    """Static mode (default) is completely unchanged — no disaster_updated=True ever."""
    env = DisasterEnv(
        grid_size=5,
        max_steps=50,
        start_position=(0, 0),
        goal_position=(4, 4),
        fire_cells=[(2, 2)],
        flood_cells=[(3, 3)],
        seed=42,
    )
    env.reset()
    # Fire and flood at original positions.
    assert env.grid[2, 2] == FIRE
    assert env.grid[3, 3] == FLOOD

    for i in range(10):
        _, _, done, info = env.step(DOWN if i % 2 == 0 else UP)
        assert info["disaster_updated"] is False
        if done:
            break

    # Fire/flood remain at original positions.
    assert env.grid[2, 2] == FIRE
    assert env.grid[3, 3] == FLOOD


def test_static_mode_no_grid_change() -> None:
    """In static mode, the grid doesn't change during an episode (except agent)."""
    env = DisasterEnv(
        grid_size=5,
        max_steps=20,
        start_position=(0, 0),
        goal_position=(4, 4),
        obstacles=[(1, 1)],
        fire_cells=[(2, 2)],
        flood_cells=[(3, 3)],
    )
    env.reset()
    grid_snapshot = env.grid.copy()

    for i in range(5):
        env.step(RIGHT)
        # Grid itself (not get_state which includes agent) should be unchanged.
        np.testing.assert_array_equal(env.grid, grid_snapshot)


def test_dynamic_default_is_false() -> None:
    """dynamic_disasters defaults to False."""
    env = DisasterEnv(grid_size=5)
    assert env.dynamic_disasters is False


def test_dynamic_info_flag_present() -> None:
    """The disaster_updated key is always present in info."""
    env = DisasterEnv(
        grid_size=5,
        start_position=(0, 0),
        goal_position=(4, 4),
    )
    env.reset()
    _, _, _, info = env.step(DOWN)
    assert "disaster_updated" in info


def test_dynamic_info_details_on_update() -> None:
    """When a disaster update occurs, info contains new cell positions."""
    env = DisasterEnv(
        grid_size=5,
        max_steps=50,
        start_position=(0, 0),
        goal_position=(4, 4),
        fire_cells=[(2, 2)],
        flood_cells=[(3, 3)],
        dynamic_disasters=True,
        disaster_change_interval=1,
        seed=42,
    )
    env.reset()
    _, _, _, info = env.step(DOWN)
    assert info["disaster_updated"] is True
    assert isinstance(info["new_fire_cells"], list)
    assert isinstance(info["new_flood_cells"], list)
    assert len(info["new_fire_cells"]) == 1
    assert len(info["new_flood_cells"]) == 1


def test_dynamic_agent_position_protected() -> None:
    """The agent's current cell is never turned into fire or flood."""
    env = DisasterEnv(
        grid_size=5,
        max_steps=100,
        start_position=(2, 2),
        goal_position=(4, 4),
        fire_cells=[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
                    (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],
        dynamic_disasters=True,
        disaster_change_interval=1,
        seed=42,
    )
    env.reset()
    for i in range(20):
        _, _, done, _ = env.step(DOWN if i % 2 == 0 else UP)
        if done:
            env.reset()
        # Agent cell should not be fire or flood.
        r, c = env.agent_position
        assert env.grid[r, c] != FIRE or (r, c) == env.agent_position  # covered by get_state
        # More directly: after dynamic update, agent position is safe.
        assert env.grid[env.agent_position] in (EMPTY, GOAL, START)


def test_dynamic_obstacle_count_preserved() -> None:
    """Obstacles stay fixed and their count is unchanged in dynamic mode."""
    obstacles = [(1, 1), (1, 3), (3, 1)]
    env = DisasterEnv(
        grid_size=5,
        max_steps=50,
        start_position=(0, 0),
        goal_position=(4, 4),
        obstacles=obstacles,
        fire_cells=[(2, 2)],
        dynamic_disasters=True,
        disaster_change_interval=2,
        seed=42,
    )
    env.reset()
    for i in range(10):
        _, _, done, _ = env.step(DOWN if i % 2 == 0 else UP)
        if done:
            env.reset()
        for r, c in obstacles:
            assert env.grid[r, c] == OBSTACLE

