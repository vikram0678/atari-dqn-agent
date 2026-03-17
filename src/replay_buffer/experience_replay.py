"""
Experience Replay Buffer.
Stores (state, action, reward, next_state, done) transitions
and supports random mini-batch sampling.
"""

import random
import numpy as np
from collections import deque
from typing import Tuple


class ReplayBuffer:
    """
    Circular experience replay buffer using deque.

    Args:
        capacity: Maximum number of transitions to store.
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        Randomly sample a mini-batch of transitions.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            as numpy arrays ready for PyTorch.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough transitions to start training."""
        return len(self.buffer) >= min_size
