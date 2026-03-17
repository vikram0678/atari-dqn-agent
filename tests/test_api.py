"""Tests for FastAPI inference endpoints."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Patch model_loader so API starts without a real model file
from src.api import model_loader as ml_module
from src.api.model_loader import ModelLoader

# Use a mock loader that's always "loaded"
class MockModelLoader:
    is_loaded = True

    def predict(self, state):
        arr = np.array(state)
        # Return argmax of first row just to return something
        return int(0), [0.1, 0.2, 0.3, 0.1, 0.1, 0.2]

ml_module.model_loader = MockModelLoader()

from src.api.main import app

client = TestClient(app)


# ── Health endpoint ───────────────────────────────────────────
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_root():
    r = client.get("/")
    assert r.status_code == 200


# ── Predict endpoint ──────────────────────────────────────────
def valid_state():
    """Return a valid (4, 84, 84) list."""
    return np.zeros((4, 84, 84), dtype=np.float32).tolist()


def test_predict_valid():
    r = client.post("/predict", json={"state": valid_state()})
    assert r.status_code == 200
    data = r.json()
    assert "action" in data
    assert isinstance(data["action"], int)


def test_predict_wrong_frames():
    """Only 3 frames instead of 4 — should fail validation."""
    bad_state = np.zeros((3, 84, 84), dtype=np.float32).tolist()
    r = client.post("/predict", json={"state": bad_state})
    assert r.status_code == 422  # Pydantic validation error


def test_predict_wrong_size():
    """64x64 instead of 84x84 — should fail validation."""
    bad_state = np.zeros((4, 64, 64), dtype=np.float32).tolist()
    r = client.post("/predict", json={"state": bad_state})
    assert r.status_code == 422


def test_predict_empty_body():
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_action_in_range():
    r = client.post("/predict", json={"state": valid_state()})
    assert r.status_code == 200
    action = r.json()["action"]
    assert 0 <= action <= 17  # Max Atari actions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
