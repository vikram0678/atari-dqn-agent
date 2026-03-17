"""Tests for DQNAgent, QNetwork, and EpsilonGreedy."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from src.agent.q_network import QNetwork
from src.agent.exploration import EpsilonGreedy
from src.agent.dqn_agent import DQNAgent


DEVICE = torch.device("cpu")

BASE_CONFIG = {
    "learning_rate"      : 0.0001,
    "gamma"              : 0.99,
    "epsilon_start"      : 1.0,
    "epsilon_min"        : 0.05,
    "epsilon_decay"      : 0.995,
    "decay_type"         : "exponential",
    "decay_steps"        : 100000,
    "replay_buffer_size" : 1000,
    "min_replay_size"    : 32,
    "batch_size"         : 32,
    "target_update"      : 100,
}


# ── QNetwork tests ────────────────────────────────────────────
def test_qnetwork_output_shape():
    net = QNetwork(n_actions=6)
    x   = torch.zeros(1, 4, 84, 84)
    out = net(x)
    assert out.shape == (1, 6)


def test_qnetwork_batch():
    net = QNetwork(n_actions=6)
    x   = torch.zeros(32, 4, 84, 84)
    out = net(x)
    assert out.shape == (32, 6)


def test_qnetwork_different_actions():
    for n in [4, 6, 9, 18]:
        net = QNetwork(n_actions=n)
        x   = torch.zeros(1, 4, 84, 84)
        out = net(x)
        assert out.shape == (1, n)


# ── EpsilonGreedy tests ───────────────────────────────────────
def test_epsilon_starts_at_one():
    eg = EpsilonGreedy(epsilon_start=1.0)
    assert eg.get_epsilon() == 1.0


def test_epsilon_decays():
    eg = EpsilonGreedy(epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.5)
    eg.decay()
    assert eg.get_epsilon() < 1.0


def test_epsilon_minimum():
    eg = EpsilonGreedy(epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.1)
    for _ in range(100):
        eg.decay()
    assert eg.get_epsilon() >= 0.05


# ── DQNAgent tests ────────────────────────────────────────────
def make_agent(n_actions=6):
    return DQNAgent(n_actions=n_actions, config=BASE_CONFIG, device=DEVICE)


def test_agent_action_range():
    agent = make_agent(n_actions=6)
    state = np.random.rand(4, 84, 84).astype(np.float32)
    for _ in range(20):
        action = agent.select_action(state)
        assert 0 <= action < 6


def test_agent_greedy_action():
    agent = make_agent(n_actions=6)
    state = np.random.rand(4, 84, 84).astype(np.float32)
    action = agent.select_action_greedy(state)
    assert 0 <= action < 6


def test_agent_train_step_returns_none_when_buffer_small():
    agent = make_agent()
    result = agent.train_step()
    assert result is None


def test_agent_train_step_returns_loss_when_ready():
    agent = make_agent()
    # Fill buffer past min_replay_size
    for i in range(40):
        s  = np.random.rand(4, 84, 84).astype(np.float32)
        ns = np.random.rand(4, 84, 84).astype(np.float32)
        agent.buffer.push(s, i % 6, 0.0, ns, False)
    loss = agent.train_step()
    assert loss is not None
    assert loss >= 0.0


def test_agent_save_load(tmp_path):
    agent = make_agent()
    path  = str(tmp_path / "test_model.pth")
    agent.save(path)
    assert os.path.exists(path)

    agent2 = make_agent()
    agent2.load(path)
    # Both should have same epsilon
    assert abs(agent.epsilon - agent2.epsilon) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
