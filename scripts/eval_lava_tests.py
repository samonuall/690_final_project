"""
Evaluate a trained PPO policy on both distributional shift test environments.

Usage:
    uv run python scripts/eval_lava_tests.py --model runs/lava_manhatten_distance/test/ppo_policy
    uv run python scripts/eval_lava_tests.py --model runs/lava_manhatten_distance/test/ppo_policy --episodes 10 --seed 123
"""

import argparse
import os
import sys

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.gridworld_wrapper import LLMRewardGridworld
from src.envs.distributional_shift_adapter import ensure_distributional_shift_test_env_registered


def env_reward_as_total(_obs, _action, _next_obs, info):
    return {"total": float(info.get("env_reward", 0.0))}


def run_episode(model, env, seed=None):
    if seed is None:
        obs, _ = env.reset()
    else:
        obs, _ = env.reset(seed=seed)
    done = False
    steps = 0
    total_env_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        total_env_reward += float(info.get("env_reward", 0.0))
        steps += 1

    return total_env_reward, steps


def evaluate_level(model_path, level_choice, episodes, seed, max_steps):
    ensure_distributional_shift_test_env_registered()
    env = LLMRewardGridworld(
        env_name="distributional_shift_test",
        max_iterations=max_steps,
        env_kwargs={"is_testing": True, "level_choice": level_choice},
    )
    env.set_reward_fn(env_reward_as_total)

    model = PPO.load(model_path, env=env)

    returns = []
    steps = []
    for ep in range(episodes):
        ep_seed = None if seed is None else seed + ep
        ep_return, ep_steps = run_episode(model, env, seed=ep_seed)
        returns.append(ep_return)
        steps.append(ep_steps)

    returns = np.array(returns, dtype=np.float32)
    steps = np.array(steps, dtype=np.int32)
    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "mean_steps": float(steps.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO policy on lava test environments.")
    parser.add_argument("--model", required=True, help="Path to saved model (.zip may be omitted)")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per test level")
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    args = parser.parse_args()

    print(f"Loaded model from {args.model}")
    print(f"Episodes per test level: {args.episodes}")

    for level_choice, label in [(1, "lava_test_upper"), (2, "lava_test_lower")]:
        stats = evaluate_level(
            args.model,
            level_choice,
            args.episodes,
            args.seed,
            args.max_steps,
        )
        print(
            f"{label}: mean_return={stats['mean_return']:+.3f} "
            f"std_return={stats['std_return']:.3f} mean_steps={stats['mean_steps']:.1f}"
        )


if __name__ == "__main__":
    main()
