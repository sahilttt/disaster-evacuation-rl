"""
Disaster Evacuation RL — Entry Point.

This script will serve as the main entry point for training,
evaluation, and demonstration once the pipeline is implemented.

Currently it validates that the project structure loads correctly.
"""

from environment.disaster_env import DisasterGridEnv
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from config.config import GRID_CONFIG, NUM_ACTIONS


def main() -> None:
    """Validate the project foundation by creating core objects."""
    print("=" * 60)
    print("  Disaster Evacuation RL — Project Foundation Check")
    print("=" * 60)

    # 1. Create environment
    env = DisasterGridEnv(
        grid_size=10,
        start_position=(0, 0),
        goal_position=(9, 9),
        obstacles=[(3, 3), (3, 4)],
        fire_cells=[(5, 5)],
        flood_cells=[(7, 2)],
    )
    state = env.reset()
    print(f"\n[OK] Environment created: {env}")
    print(f"     Grid size   : {env.grid_size}x{env.grid_size}")
    print(f"     Start       : {env.start_position}")
    print(f"     Goal        : {env.goal_position}")
    print(f"     Actions     : {env.get_action_space_size()}")
    print(f"     State shape : {env.get_state_shape()}")

    # 2. Render the grid
    print("\n--- Grid ---")
    env.render()

    # 3. Take a dummy step
    next_state, reward, done, info = env.step(1)  # DOWN
    print(f"\n[OK] Step taken.  Reward: {reward}, Done: {done}, Info: {info}")

    # 4. Create agents (constructors only)
    q_agent = QLearningAgent(state_shape=env.get_state_shape())
    dqn_agent = DQNAgent(state_shape=env.get_state_shape())
    print(f"\n[OK] Q-Learning agent created (epsilon={q_agent.epsilon})")
    print(f"[OK] DQN agent created        (epsilon={dqn_agent.epsilon})")

    # 5. Summary
    print("\n" + "=" * 60)
    print("  All modules imported and objects created successfully.")
    print("  Foundation is ready for Phase 2 implementation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
