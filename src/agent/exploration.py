"""
Epsilon-Greedy Exploration Strategy with linear and exponential decay.
"""


class EpsilonGreedy:
    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        decay_type: str = "exponential",  # "exponential" or "linear"
        decay_steps: int = 100000,
    ):
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.decay_type = decay_type
        self.decay_steps = decay_steps
        self.steps = 0

    def decay(self):
        """Decay epsilon after each episode."""
        self.steps += 1
        if self.decay_type == "exponential":
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay
            )
        elif self.decay_type == "linear":
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon_start - (self.epsilon_start - self.epsilon_min)
                * (self.steps / self.decay_steps)
            )

    def get_epsilon(self) -> float:
        return self.epsilon
