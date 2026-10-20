"""
Central configuration for the disaster evacuation RL project.
All hyperparameters and grid settings are defined here.
"""

# --- Grid Environment Settings ---
GRID_CONFIG = {
    "grid_size": 10,           # N x N grid
    "max_steps": 100,          # Maximum steps per episode
    "start_position": (0, 0),  # Agent starting cell
    "safe_zone": (9, 9),       # Evacuation target cell
}

# --- Cell Types ---
# These constants identify what occupies each grid cell.
CELL_EMPTY = 0
CELL_OBSTACLE = 1
CELL_FIRE = 2
CELL_FLOOD = 3
CELL_BLOCKED_ROAD = 4
CELL_SAFE_ZONE = 5
CELL_AGENT = 6

# --- Actions ---
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
NUM_ACTIONS = 4

# --- Reward Settings (planned) ---
REWARD_CONFIG = {
    "reach_safe_zone": 100.0,
    "hit_obstacle": -10.0,
    "enter_fire": -50.0,
    "enter_flood": -30.0,
    "blocked_road": -10.0,
    "step_penalty": -1.0,
}

# --- Q-Learning Hyperparameters (planned) ---
Q_LEARNING_CONFIG = {
    "learning_rate": 0.1,
    "discount_factor": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "num_episodes": 1000,
}

# --- DQN Hyperparameters (planned) ---
DQN_CONFIG = {
    "learning_rate": 0.001,
    "discount_factor": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "batch_size": 64,
    "memory_size": 10000,
    "target_update_freq": 10,
    "num_episodes": 500,
    "hidden_size": 128,
}
