"""
Central configuration loader.
Reads hyperparameters.yaml and merges with environment variables.
"""

import os
import yaml
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "src" / "config"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR   = ROOT_DIR / "logs"
VIDEOS_DIR = ROOT_DIR / "gameplay_videos"

# Create dirs if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


def load_config(yaml_path: str = None) -> dict:
    """
    Load hyperparameters from YAML file.
    Environment variables override YAML values.
    """
    if yaml_path is None:
        yaml_path = CONFIG_DIR / "hyperparameters.yaml"

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Allow env var overrides
    config["game"]           = os.getenv("ATARI_GAME_ID",  config["game"])
    config["episodes"]       = int(os.getenv("EPISODES",   config["episodes"]))
    config["learning_rate"]  = float(os.getenv("LR",       config["learning_rate"]))

    # Model path
    config["model_path"] = os.getenv(
        "MODEL_PATH",
        str(MODELS_DIR / "latest_model.pth")
    )
    config["best_model_path"] = str(MODELS_DIR / "best_model.pth")
    config["models_dir"]      = str(MODELS_DIR)
    config["logs_dir"]        = str(LOGS_DIR)
    config["videos_dir"]      = str(VIDEOS_DIR)

    return config


# Default config instance
DEFAULT_CONFIG = load_config()
