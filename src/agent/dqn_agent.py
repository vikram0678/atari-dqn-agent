"""
DQN Agent — ties together Q-network, target network,
replay buffer, and epsilon-greedy exploration.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from src.agent.q_network import QNetwork
from src.agent.exploration import EpsilonGreedy
from src.replay_buffer.experience_replay import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        n_actions: int,
        config: dict,
        device: torch.device,
    ):
        self.n_actions = n_actions
        self.config = config
        self.device = device
        self.steps_done = 0

        # Networks
        self.q_net = QNetwork(n_actions).to(device)
        self.target_net = QNetwork(n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.q_net.parameters(),
            lr=config["learning_rate"]
        )

        # Loss — Huber loss for stability
        self.loss_fn = nn.SmoothL1Loss()

        # Replay buffer
        self.buffer = ReplayBuffer(config["replay_buffer_size"])

        # Exploration
        self.explorer = EpsilonGreedy(
            epsilon_start=config["epsilon_start"],
            epsilon_min=config["epsilon_min"],
            epsilon_decay=config["epsilon_decay"],
            decay_type=config.get("decay_type", "exponential"),
            decay_steps=config.get("decay_steps", 100000),
        )

    @property
    def epsilon(self) -> float:
        return self.explorer.get_epsilon()

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.argmax().item()

    def select_action_greedy(self, state: np.ndarray) -> int:
        """Pure greedy action (for evaluation/inference)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.argmax().item()

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Return Q-values for logging average Q."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.cpu().numpy()[0]

    def train_step(self) -> Optional[float]:
        """Sample from buffer and perform one gradient update."""
        if len(self.buffer) < self.config["min_replay_size"]:
            return None

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.config["batch_size"])

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values for taken actions
        current_q = self.q_net(states_t).gather(
            1, actions_t.unsqueeze(1)
        ).squeeze(1)

        # TD target using target network (Bellman equation)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + self.config["gamma"] * next_q * (1 - dones_t)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.steps_done += 1

        # Periodically sync target network
        if self.steps_done % self.config["target_update"] == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.explorer.decay()

    def save(self, path: str):
        """Save checkpoint."""
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps_done,
        }, path)

    def load(self, path: str):
        """Load checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.explorer.epsilon = ckpt["epsilon"]
        self.steps_done = ckpt["steps"]
        print(f"Checkpoint loaded from {path}")
        print(f"  Steps: {self.steps_done} | Epsilon: {self.epsilon:.4f}")
