"""
Visualize a trained PPO policy on a gridworld environment.

Usage:
    uv run python scripts/visualize.py --model runs/test/ppo_policy --env boat_race
    uv run python scripts/visualize.py --model runs/test/ppo_policy --env distributional_shift --save-gif out.gif
    uv run python scripts/visualize.py --model runs/test/ppo_policy --env boat_race --episodes 3 --delay 0.15
"""

import argparse
import os
import sys
import time

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.gridworld_wrapper import LLMRewardGridworld, boat_race_fitness, lava_fitness

FITNESS_FNS = {
    "boat_race": boat_race_fitness,
    "distributional_shift": lava_fitness,
}


def _ansi_frame(env_inner, step, reward, fitness, ep_fitness):
    """Return a string to print for the current step."""
    ansi = env_inner.render(mode="ansi")
    lines = [
        f"  step={step:3d}  reward={reward:+.3f}  step_fitness={fitness:+.3f}  ep_fitness={ep_fitness:+.3f}",
        "",
        ansi,
    ]
    return "\n".join(lines)


def run_episode_ansi(model, env, delay, ep_num):
    """Run one episode with ansi terminal rendering."""
    obs, _ = env.reset()
    done = False
    step = 0
    ep_fitness = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        fitness = info.get("fitness", 0.0)
        ep_fitness += fitness
        step += 1

        os.system("clear")
        print(f"Episode {ep_num}  (press Ctrl-C to quit)")
        print(_ansi_frame(env.env, step, reward, fitness, ep_fitness))
        time.sleep(delay)

    print(f"\n  Episode {ep_num} done — total fitness: {ep_fitness:+.3f}  steps: {step}")
    time.sleep(1.0)
    return ep_fitness, step


def run_episode_rgb(model, env, ep_num):
    """Run one episode collecting RGB frames."""
    obs, _ = env.reset()
    done = False
    frames = []
    ep_fitness = 0.0
    step = 0

    # Capture initial frame
    rgb = env.env._rgb
    if rgb is not None:
        frames.append(rgb.copy())

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        ep_fitness += info.get("fitness", 0.0)
        step += 1

        rgb = env.env._rgb
        if rgb is not None:
            frames.append(rgb.copy())

    print(f"  Episode {ep_num} — fitness: {ep_fitness:+.3f}  steps: {step}")
    return frames, ep_fitness


def save_gif(all_frames, path, fps=5):
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not installed — run: uv add Pillow")
        return

    images = []
    for rgb in all_frames:
        # rgb shape is (H, W, 3) or (3, H, W); normalise to uint8 (H, W, 3)
        if rgb.ndim == 3 and rgb.shape[0] == 3:
            rgb = np.transpose(rgb, (1, 2, 0))
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        images.append(Image.fromarray(rgb))

    if images:
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        print(f"GIF saved to {path}  ({len(images)} frames)")
    else:
        print("No RGB frames captured — env may not expose rgb_array.")


def main():
    parser = argparse.ArgumentParser(description="Visualize a trained PPO policy on a gridworld.")
    parser.add_argument("--model", required=True, help="Path to saved model (.zip may be omitted)")
    parser.add_argument(
        "--env",
        required=True,
        choices=list(FITNESS_FNS.keys()),
        help="Environment name",
    )
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes to run")
    parser.add_argument("--delay", type=float, default=0.1, help="Seconds between frames (ansi mode)")
    parser.add_argument(
        "--save-gif",
        metavar="PATH",
        default=None,
        help="If set, collect RGB frames and save to this .gif path instead of ansi display",
    )
    args = parser.parse_args()

    fitness_fn = FITNESS_FNS[args.env]
    env = LLMRewardGridworld(env_name=args.env, max_iterations=100)
    env.set_fitness_fn(fitness_fn)

    model = PPO.load(args.model, env=env)
    print(f"Loaded model from {args.model}")
    print(f"Env: {args.env}  |  episodes: {args.episodes}")

    if args.save_gif:
        all_frames = []
        for ep in range(1, args.episodes + 1):
            frames, _ = run_episode_rgb(model, env, ep)
            all_frames.extend(frames)
        save_gif(all_frames, args.save_gif)
    else:
        total_fitness = 0.0
        for ep in range(1, args.episodes + 1):
            ep_fitness, _ = run_episode_ansi(model, env, args.delay, ep)
            total_fitness += ep_fitness
        print(f"\nAverage fitness over {args.episodes} episodes: {total_fitness / args.episodes:+.3f}")


if __name__ == "__main__":
    main()
