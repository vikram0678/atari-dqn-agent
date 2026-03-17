"""
Play script — records agent gameplay as MP4 video.
Usage:
    python scripts/play.py
    python scripts/play.py --game ALE/Pong-v5 --model models/latest_model.pth --output gameplay_videos/demo.mp4
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch

try:
    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)
except Exception:
    pass

from src.config.config import load_config
from src.agent.dqn_agent import DQNAgent
from src.utils.checkpointing import load_checkpoint
from src.utils.video import record_gameplay


def parse_args():
    parser = argparse.ArgumentParser(description="Record agent gameplay")
    parser.add_argument("--game",   type=str, default=None)
    parser.add_argument("--model",  type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)

    if args.game:
        config["game"] = args.game
    if args.model:
        config["model_path"] = args.model

    output_path = args.output or os.path.join(
        config["videos_dir"], "demo.mp4"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"  Game   : {config['game']}")
    print(f"  Model  : {config['model_path']}")
    print(f"  Output : {output_path}")
    print("=" * 60)

    # Need n_actions — make a quick env to get it
    import gymnasium as gym
    _env = gym.make(config["game"], frameskip=1, render_mode=None)
    n_actions = _env.action_space.n
    _env.close()

    agent = DQNAgent(
        n_actions=n_actions,
        config=config,
        device=device,
    )

    loaded = load_checkpoint(agent, config["model_path"])
    if not loaded:
        loaded = load_checkpoint(agent, config["best_model_path"])
    if not loaded:
        print("ERROR: No model checkpoint found.")
        sys.exit(1)

    record_gameplay(
        agent=agent,
        game=config["game"],
        output_path=output_path,
        max_steps=config["max_steps_per_episode"],
        frame_size=config["frame_size"],
        stack_frames=config["stack_frames"],
    )

    print(f"\nVideo saved: {output_path}")


if __name__ == "__main__":
    main()
