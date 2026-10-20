# AI-Based Disaster Evacuation Route Planning Using Deep Reinforcement Learning

## Project Objective

This project builds a grid-world simulation where a reinforcement-learning agent
learns optimal evacuation routes during disasters (fire, flood, blocked roads).
The agent is trained using **Q-Learning** and **Deep Q-Networks (DQN)** to navigate
from a start position to a safe evacuation zone while avoiding hazards.

The final deliverable is a web application that visualises the trained agent's
decision-making in real time.

## Planned ML / RL Pipeline

```
Grid-World Environment
        │
        ▼
State Representation (agent position, hazard map)
        │
        ▼
Agent selects action (UP / DOWN / LEFT / RIGHT)
        │
        ▼
Environment returns (next_state, reward, done)
        │
        ▼
Agent updates policy (Q-table or DQN)
        │
        ▼
Repeat until convergence
```

**Reward design (planned):**
- Reaching the safe zone → large positive reward
- Stepping into fire / flood → large negative reward
- Hitting an obstacle / boundary → negative reward
- Each step → small negative reward (encourages shorter paths)

## Planned Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project structure & environment skeleton | ✅ Done |
| 2 | Grid-world environment implementation | 🔲 Planned |
| 3 | Q-Learning agent | 🔲 Planned |
| 4 | DQN agent with PyTorch | 🔲 Planned |
| 5 | Training pipeline & experiments | 🔲 Planned |
| 6 | Dynamic disasters & advanced scenarios | 🔲 Planned |
| 7 | Evaluation & visualisation | 🔲 Planned |
| 8 | FastAPI backend | 🔲 Planned |
| 9 | React + TypeScript frontend | 🔲 Planned |

## Current Implementation Status

**Phase 1 — Foundation (current)**

- [x] Modular project structure
- [x] Configuration module with grid and training defaults
- [x] Environment class skeleton (`DisasterGridEnv`)
- [x] Agent placeholders (`QLearningAgent`, `DQNAgent`)
- [x] DQN network placeholder
- [x] Trainer, evaluator, and visualiser skeletons
- [x] Utility helpers
- [x] Basic test suite
- [x] Requirements file
- [x] `.gitignore`

No RL logic, training loops, or web components are implemented yet.

## Quick Start

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run the entry point (placeholder)
python main.py

# Run tests
python -m pytest tests/ -v
```

## Project Structure

```
disaster-evacuation-rl/
├── config/           # Hyperparameters & grid settings
├── environment/      # Grid-world environment
├── agents/           # RL agents (Q-Learning, DQN)
├── models/           # Neural network architectures
├── training/         # Training loop orchestration
├── evaluation/       # Policy evaluation & metrics
├── visualization/    # Grid rendering & plots
├── utils/            # Shared helper functions
├── experiments/      # Saved results & logs
├── tests/            # Unit tests
└── main.py           # Entry point
```

## Tech Stack

- **Python 3** — core language
- **PyTorch** — DQN neural network
- **NumPy** — numerical operations
- **Matplotlib** — visualisation
- **Pandas** — experiment logging
- **FastAPI** — backend API *(later)*
- **React + TypeScript + Tailwind** — frontend *(later)*
