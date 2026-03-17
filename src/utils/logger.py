"""
Training logger using TensorBoard.
Logs: reward, loss, epsilon, avg Q-value, episode duration.
"""

import os
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logging to: {log_dir}")

    def log_episode(
        self,
        episode: int,
        reward: float,
        avg_reward: float,
        epsilon: float,
        loss: float,
        avg_q: float,
        duration: float,
    ):
        self.writer.add_scalar("Episode/Reward",          reward,      episode)
        self.writer.add_scalar("Episode/AvgReward_100",   avg_reward,  episode)
        self.writer.add_scalar("Episode/Epsilon",         epsilon,     episode)
        self.writer.add_scalar("Episode/Loss",            loss,        episode)
        self.writer.add_scalar("Episode/AvgQValue",       avg_q,       episode)
        self.writer.add_scalar("Episode/Duration_sec",    duration,    episode)

    def log_step(self, step: int, loss: float):
        self.writer.add_scalar("Step/Loss", loss, step)

    def close(self):
        self.writer.close()
