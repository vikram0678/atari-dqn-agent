"""
Training script for DQN agent on Atari games.
Usage:
    python scripts/train.py
    python scripts/train.py --game ALE/Pong-v5 --episodes 1000
    python scripts/train.py --resume models/latest_model.pth
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
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
from src.utils.logger import TrainingLogger
from src.utils.checkpointing import save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN on Atari")
    parser.add_argument("--game",     type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--resume",   type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--config",   type=str, default=None,
                        help="Path to custom hyperparameters.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.game:
        config["game"] = args.game
    if args.episodes:
        config["episodes"] = args.episodes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"  Game    : {config['game']}")
    print(f"  Device  : {device}")
    print(f"  Episodes: {config['episodes']}")
    print("=" * 60)

    # Environment
    env = AtariEnv(
        game=config["game"],
        frame_skip=config["frame_skip"],
        frame_size=config["frame_size"],
        stack_frames=config["stack_frames"],
    )

    # Agent
    agent = DQNAgent(
        n_actions=env.n_actions,
        config=config,
        device=device,
    )

    # Resume from checkpoint
    if args.resume:
        load_checkpoint(agent, args.resume)

    # Logger
    log_dir = os.path.join(config["logs_dir"], "tensorboard")
    logger = TrainingLogger(log_dir=log_dir)

    # Tracking
    episode_rewards  = []
    best_avg_reward  = -float("inf")
    start_training   = time.time()

    print("\nStarting training...\n")

    for episode in range(1, config["episodes"] + 1):
        state        = env.reset()
        total_reward = 0.0
        losses       = []
        q_values_log = []
        ep_start     = time.time()

        for step in range(config["max_steps_per_episode"]):
            action = agent.select_action(state)

            # Log Q-values occasionally
            if step % 100 == 0:
                q_vals = agent.get_q_values(state)
                q_values_log.append(np.max(q_vals))

            next_state, reward, done = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        avg_reward = np.mean(episode_rewards[-config["early_stop_window"]:])
        avg_loss   = np.mean(losses) if losses else 0.0
        avg_q      = np.mean(q_values_log) if q_values_log else 0.0
        duration   = time.time() - ep_start

        # Log to TensorBoard
        logger.log_episode(
            episode=episode,
            reward=total_reward,
            avg_reward=avg_reward,
            epsilon=agent.epsilon,
            loss=avg_loss,
            avg_q=avg_q,
            duration=duration,
        )

        # Console log every 10 episodes
        if episode % 10 == 0:
            elapsed = (time.time() - start_training) / 60
            print(
                f"Ep {episode:5d} | "
                f"Reward: {total_reward:6.1f} | "
                f"Avg(100): {avg_reward:6.2f} | "
                f"ε: {agent.epsilon:.3f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Q: {avg_q:.3f} | "
                f"Time: {elapsed:.1f}m"
            )

        # Save periodic checkpoint
        if episode % config["save_every"] == 0:
            ckpt_path = os.path.join(
                config["models_dir"], f"model_ep{episode}.pth"
            )
            save_checkpoint(agent, ckpt_path)

        # Save best model
        if avg_reward > best_avg_reward and episode >= 50:
            best_avg_reward = avg_reward
            save_checkpoint(agent, config["best_model_path"])
            print(f"  ★ New best avg reward: {best_avg_reward:.2f}")

        # Save latest always
        save_checkpoint(agent, config["model_path"])

        # Early stopping
        if (avg_reward >= config["early_stop_reward"]
                and episode >= config["early_stop_window"]):
            print(f"\n🎉 Solved at episode {episode}!")
            print(f"   Average reward: {avg_reward:.2f}")
            break

    logger.close()
    env.close()

    total_time = (time.time() - start_training) / 60
    print("\n" + "=" * 60)
    print(f"  Training complete in {total_time:.1f} minutes")
    print(f"  Best average reward : {best_avg_reward:.2f}")
    print(f"  Model saved to      : {config['model_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
