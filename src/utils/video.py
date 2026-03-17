"""
Gameplay video recorder.
Renders agent gameplay and saves as MP4.
"""

import os
import cv2
import numpy as np
import gymnasium as gym

try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass


def record_gameplay(
    agent,
    game: str,
    output_path: str,
    max_steps: int = 10000,
    fps: int = 30,
    frame_size: int = 84,
    stack_frames: int = 4,
):
    """
    Record a full episode of agent gameplay as MP4.

    Args:
        agent: Trained DQNAgent
        game: Atari game ID
        output_path: Path to save .mp4 file
        max_steps: Max steps per episode
        fps: Video FPS
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Use rgb_array render mode for recording
    env = gym.make(game, frameskip=1, render_mode="rgb_array")

    from src.environment.preprocessing import FrameStack
    frame_stack = FrameStack(k=stack_frames, frame_size=frame_size)

    obs, _ = env.reset()
    state = frame_stack.reset(obs)

    # Setup video writer
    raw_frame = env.render()
    h, w = raw_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total_reward = 0.0
    step = 0

    while step < max_steps:
        # Render and write frame
        raw_frame = env.render()
        bgr_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)

        # Agent picks action greedily
        action = agent.select_action_greedy(state)

        # Step environment (manual frame skip)
        total_reward_step = 0.0
        frames = []
        done = False

        for _ in range(4):  # frame skip
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward_step += reward
            frames.append(obs)
            if terminated or truncated:
                done = True
                break

        max_frame = np.maximum(frames[-1], frames[-2]) if len(frames) >= 2 else frames[-1]
        state = frame_stack.step(max_frame)
        total_reward += total_reward_step
        step += 1

        if done:
            break

    out.release()
    env.close()

    print(f"Video saved to: {output_path}")
    print(f"Total reward  : {total_reward:.1f}")
    print(f"Total steps   : {step}")
    return total_reward
