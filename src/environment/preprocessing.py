"""
Frame preprocessing: RGB -> Grayscale -> Resize 84x84 -> Stack 4 frames.
"""

import numpy as np
import cv2
from collections import deque


def preprocess_frame(frame: np.ndarray, frame_size: int = 84) -> np.ndarray:
    """
    Convert a raw RGB Atari frame to a grayscale 84x84 float32 array.

    Args:
        frame: Raw RGB frame of shape (H, W, 3)
        frame_size: Target size (default 84)
    Returns:
        Grayscale frame of shape (84, 84), values in [0.0, 1.0]
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (frame_size, frame_size),
                         interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


class FrameStack:
    """
    Maintains a rolling window of k preprocessed frames.
    Returns stacked state of shape (k, 84, 84).
    """

    def __init__(self, k: int = 4, frame_size: int = 84):
        self.k = k
        self.frame_size = frame_size
        self.frames = deque(maxlen=k)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        """Initialize stack by repeating first frame k times."""
        processed = preprocess_frame(frame, self.frame_size)
        for _ in range(self.k):
            self.frames.append(processed)
        return self._get_state()

    def step(self, frame: np.ndarray) -> np.ndarray:
        """Push new frame and return updated stack."""
        processed = preprocess_frame(frame, self.frame_size)
        self.frames.append(processed)
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Return stacked frames as (k, H, W) array."""
        return np.array(self.frames, dtype=np.float32)
