"""
Model checkpointing utilities.
"""

import os
import torch


def save_checkpoint(agent, path: str):
    """Save agent checkpoint to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    agent.save(path)
    print(f"  [Checkpoint] Saved → {path}")


def load_checkpoint(agent, path: str) -> bool:
    """
    Load agent checkpoint from disk.
    Returns True if successful, False if file not found.
    """
    if not os.path.exists(path):
        print(f"  [Checkpoint] Not found: {path}")
        return False
    agent.load(path)
    return True


def get_latest_checkpoint(models_dir: str) -> str:
    """Find the latest checkpoint file in models directory."""
    latest = os.path.join(models_dir, "latest_model.pth")
    if os.path.exists(latest):
        return latest

    # Fallback: find highest episode checkpoint
    checkpoints = [
        f for f in os.listdir(models_dir)
        if f.startswith("model_ep") and f.endswith(".pth")
    ]
    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: int(x.replace("model_ep", "").replace(".pth", "")))
    return os.path.join(models_dir, checkpoints[-1])
