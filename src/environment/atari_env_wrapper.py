"""
Atari environment wrapper:
- Frame skipping (skip N frames, action repeated)
- Max pooling over last 2 frames (removes flickering)
- Reward clipping to [-1, 1]
- Frame preprocessing + stacking
Compatible with both local and Kaggle/Docker environments.
"""

import numpy as np
import gymnasium as gym

try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass  # Already registered or not needed

from src.environment.preprocessing import FrameStack, preprocess_frame


class AtariEnv:
    """
    Wrapped Atari environment with:
      - frame skip
      - max pooling over last 2 raw frames
      - reward clipping
      - grayscale + resize preprocessing
      - 4-frame stacking
    """

    def __init__(
        self,
        game: str = "ALE/Pong-v5",
        frame_skip: int = 4,
        frame_size: int = 84,
        stack_frames: int = 4,
        render_mode: str = None,
    ):
        self.env = gym.make(game, frameskip=1, render_mode=render_mode)
        self.frame_skip = frame_skip
        self.frame_size = frame_size
        self.frame_stack = FrameStack(k=stack_frames, frame_size=frame_size)
        self.n_actions = self.env.action_space.n
        self.game = game

    def reset(self) -> np.ndarray:
        """Reset environment and return initial stacked state."""
        obs, _ = self.env.reset()
        state = self.frame_stack.reset(obs)
        return state

    def step(self, action: int):
        """
        Take action for frame_skip steps.
        Returns: (state, clipped_reward, done)
        """
        total_reward = 0.0
        frames = []

        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            frames.append(obs)
            if terminated or truncated:
                break

        # Max pooling over last 2 frames to remove flickering
        if len(frames) >= 2:
            max_frame = np.maximum(frames[-1], frames[-2])
        else:
            max_frame = frames[-1]

        # Clip reward to [-1, 1]
        total_reward = float(np.clip(total_reward, -1.0, 1.0))

        state = self.frame_stack.step(max_frame)
        done = terminated or truncated
        return state, total_reward, done

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()
