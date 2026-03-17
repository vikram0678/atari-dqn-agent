"""
Q-Network: CNN architecture for DQN agent.
Input: (batch, 4, 84, 84) stacked grayscale frames
Output: (batch, n_actions) Q-values
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, n_actions: int):
        super(QNetwork, self).__init__()

        # Convolutional layers (DeepMind DQN architecture)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # (4,84,84) -> (32,20,20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> (64,9,9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> (64,7,7)
            nn.ReLU(),
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 4, 84, 84), values in [0, 1]
        Returns:
            Q-values of shape (batch, n_actions)
        """
        return self.fc(self.conv(x))
