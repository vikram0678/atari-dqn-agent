"""
Evaluation script — runs N episodes and prints average reward.
Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --game ALE/Pong-v5 --model models/latest_model.pth --episodes 100
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch

try:
    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)
except Exception:
    pass

from src.config.config import load_config
from src.environment.atari_env_wrapper import AtariEnv
from src.agent.dqn_agent import DQNAgent
from src.utils.checkpointing import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DQN agent")
    parser.add_argument("--game",     type=str, default=None)
    parser.add_argument("--model",    type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--config",   type=str, default=None)
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)

    if args.game:
        config["game"] = args.game
    if args.model:
        config["model_path"] = args.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"  Evaluating : {config['game']}")
    print(f"  Model      : {config['model_path']}")
    print(f"  Episodes   : {args.episodes}")
    print(f"  Device     : {device}")
    print("=" * 60)

    env = AtariEnv(
        game=config["game"],
        frame_skip=config["frame_skip"],
        frame_size=config["frame_size"],
        stack_frames=config["stack_frames"],
    )

    agent = DQNAgent(
        n_actions=env.n_actions,
        config=config,
        device=device,
    )

    loaded = load_checkpoint(agent, config["model_path"])
    if not loaded:
        # Try best model
        loaded = load_checkpoint(agent, config["best_model_path"])
    if not loaded:
        print("ERROR: No model checkpoint found.")
        sys.exit(1)

    # Pure greedy during evaluation
    agent.explorer.epsilon = 0.05  # small epsilon for eval

    rewards = []

    for ep in range(1, args.episodes + 1):
        state        = env.reset()
        total_reward = 0.0

        for _ in range(config["max_steps_per_episode"]):
            action = agent.select_action_greedy(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                break

        rewards.append(total_reward)

        if ep % 10 == 0:
            print(f"  Episode {ep:3d}/{args.episodes} | "
                  f"Reward: {total_reward:6.1f} | "
                  f"Avg so far: {np.mean(rewards):.2f}")

    env.close()

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)

    print("\n" + "=" * 60)
    print(f"  EVALUATION RESULTS ({args.episodes} episodes)")
    print(f"  Average Reward : {avg_reward:.2f}")
    print(f"  Std Dev        : {std_reward:.2f}")
    print(f"  Min Reward     : {min_reward:.1f}")
    print(f"  Max Reward     : {max_reward:.1f}")
    print("=" * 60)

    # Parsable line for automated evaluation
    print(f"AVERAGE_REWARD={avg_reward:.2f}")

    return avg_reward


if __name__ == "__main__":
    main()
