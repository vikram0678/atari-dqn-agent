"""Tests for ExperienceReplayBuffer."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.replay_buffer.experience_replay import ReplayBuffer


def make_transition(seed=0):
    np.random.seed(seed)
    state      = np.random.rand(4, 84, 84).astype(np.float32)
    action     = np.random.randint(0, 6)
    reward     = float(np.random.uniform(-1, 1))
    next_state = np.random.rand(4, 84, 84).astype(np.float32)
    done       = bool(np.random.randint(0, 2))
    return state, action, reward, next_state, done


def test_push_and_len():
    buf = ReplayBuffer(capacity=100)
    assert len(buf) == 0
    buf.push(*make_transition(0))
    assert len(buf) == 1
    buf.push(*make_transition(1))
    assert len(buf) == 2


def test_capacity_overflow():
    buf = ReplayBuffer(capacity=5)
    for i in range(10):
        buf.push(*make_transition(i))
    assert len(buf) == 5  # deque maxlen caps it


def test_sample_shapes():
    buf = ReplayBuffer(capacity=1000)
    for i in range(100):
        buf.push(*make_transition(i))

    states, actions, rewards, next_states, dones = buf.sample(32)
    assert states.shape      == (32, 4, 84, 84)
    assert actions.shape     == (32,)
    assert rewards.shape     == (32,)
    assert next_states.shape == (32, 4, 84, 84)
    assert dones.shape       == (32,)


def test_sample_dtypes():
    buf = ReplayBuffer(capacity=1000)
    for i in range(100):
        buf.push(*make_transition(i))

    states, actions, rewards, next_states, dones = buf.sample(16)
    assert states.dtype      == np.float32
    assert actions.dtype     == np.int64
    assert rewards.dtype     == np.float32
    assert next_states.dtype == np.float32
    assert dones.dtype       == np.float32


def test_is_ready():
    buf = ReplayBuffer(capacity=1000)
    assert not buf.is_ready(10)
    for i in range(10):
        buf.push(*make_transition(i))
    assert buf.is_ready(10)
    assert not buf.is_ready(11)


def test_sample_raises_when_not_enough():
    buf = ReplayBuffer(capacity=100)
    buf.push(*make_transition(0))
    with pytest.raises(ValueError):
        buf.sample(32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
