"""
DQN Neural Network.

A small feed-forward network that maps a flattened grid state
to Q-values for each action.
"""

import torch
import torch.nn as nn

from config.config import DQN_CONFIG, NUM_ACTIONS


class DQNNetwork(nn.Module):
    """Feed-forward Q-network for the DQN agent.

    Architecture:
        Linear(input_size, hidden) -> ReLU
        Linear(hidden, hidden)     -> ReLU
        Linear(hidden, num_actions)

    Attributes:
        net: The sequential network.
    """

    def __init__(self, input_size: int, hidden_size: int | None = None) -> None:
        """Initialise the network.

        Args:
            input_size: Number of input features (flattened state).
            hidden_size: Neurons in each hidden layer.
        """
        super().__init__()
        h = hidden_size or DQN_CONFIG.get("hidden_size", 128)
        self.num_actions = NUM_ACTIONS

        self.net = nn.Sequential(
            nn.Linear(input_size, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, self.num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape ``(batch, input_size)``.

        Returns:
            Q-values tensor of shape ``(batch, num_actions)``.
        """
        return self.net(x)
