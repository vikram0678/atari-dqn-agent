"""Tests for frame preprocessing and FrameStack."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.environment.preprocessing import preprocess_frame, FrameStack


def random_rgb_frame(h=210, w=160):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def test_preprocess_output_shape():
    frame = random_rgb_frame()
    out   = preprocess_frame(frame)
    assert out.shape == (84, 84)


def test_preprocess_output_range():
    frame = random_rgb_frame()
    out   = preprocess_frame(frame)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_preprocess_dtype():
    frame = random_rgb_frame()
    out   = preprocess_frame(frame)
    assert out.dtype == np.float32


def test_framestack_reset_shape():
    fs    = FrameStack(k=4, frame_size=84)
    frame = random_rgb_frame()
    state = fs.reset(frame)
    assert state.shape == (4, 84, 84)


def test_framestack_step_shape():
    fs    = FrameStack(k=4, frame_size=84)
    frame = random_rgb_frame()
    fs.reset(frame)
    state = fs.step(random_rgb_frame())
    assert state.shape == (4, 84, 84)


def test_framestack_rolling():
    """After 4 steps, oldest frame should be replaced."""
    fs    = FrameStack(k=4, frame_size=84)
    frame = random_rgb_frame()
    state0 = fs.reset(frame)

    for _ in range(4):
        state = fs.step(random_rgb_frame())

    # State should change after new frames pushed
    assert not np.array_equal(state0, state)


def test_framestack_dtype():
    fs    = FrameStack(k=4, frame_size=84)
    state = fs.reset(random_rgb_frame())
    assert state.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
