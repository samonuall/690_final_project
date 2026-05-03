"""
Evaluate a trained PPO policy on the train environment and both distributional
shift test environments, then print a formatted summary table.

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

from src.envs.gridworld_wrapper import LLMRewardGridworld, lava_fitness
from src.envs.distributional_shift_adapter import ensure_distributional_shift_test_env_registered


def _fitness_as_total(obs, action, next_obs, info):
    return {"total": lava_fitness(obs, action, next_obs, info)}


def run_episode(model, env, seed=None):
    obs, _ = env.reset(seed=seed) if seed is not None else env.reset()
    done = False
    steps = 0
    total_fitness = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        total_fitness += float(info.get("fitness", 0.0))
        steps += 1
    return total_fitness, steps


def evaluate_env(model_path, env, episodes, seed):
    model = PPO.load(model_path, env=env)
    returns, steps = [], []
    for ep in range(episodes):
        ep_seed = None if seed is None else seed + ep
        r, s = run_episode(model, env, seed=ep_seed)
        returns.append(r)
        steps.append(s)
    returns = np.array(returns, dtype=np.float32)
    steps = np.array(steps, dtype=np.int32)
    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "mean_steps": float(steps.mean()),
    }


def make_train_env(max_steps):
    env = LLMRewardGridworld(env_name="distributional_shift", max_iterations=max_steps)
    env.set_reward_fn(_fitness_as_total)
    env.set_fitness_fn(lava_fitness)
    return env


def make_test_env(level_choice, max_steps):
    ensure_distributional_shift_test_env_registered()
    env = LLMRewardGridworld(
        env_name="distributional_shift_test",
        max_iterations=max_steps,
        env_kwargs={"is_testing": True, "level_choice": level_choice},
    )
    env.set_reward_fn(_fitness_as_total)
    env.set_fitness_fn(lava_fitness)
    return env


def print_results(model_path, episodes, results):
    col_w = 20
    val_w = 14

    header_label = "Environment"
    headers = ["Mean Return", "Std Return", "Min Return", "Max Return", "Mean Steps"]

    sep = "─" * (col_w + val_w * len(headers))
    print()
    print(f"  Model   : {model_path}")
    print(f"  Episodes: {episodes} per environment")
    print()
    print(f"  {'─' * (col_w + val_w * len(headers))}")
    print(f"  {header_label:<{col_w}}" + "".join(f"{h:>{val_w}}" for h in headers))
    print(f"  {'─' * (col_w + val_w * len(headers))}")
    for label, stats in results:
        row = (
            f"  {label:<{col_w}}"
            f"{stats['mean_return']:>{val_w}.3f}"
            f"{stats['std_return']:>{val_w}.3f}"
            f"{stats['min_return']:>{val_w}.3f}"
            f"{stats['max_return']:>{val_w}.3f}"
            f"{stats['mean_steps']:>{val_w}.1f}"
        )
        print(row)
    print(f"  {'─' * (col_w + val_w * len(headers))}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO policy on lava environments.")
    parser.add_argument("--model", required=True, help="Path to saved model (.zip may be omitted)")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per environment")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    args = parser.parse_args()

    envs = [
        ("train",           make_train_env(args.max_steps)),
        ("lava_test_upper", make_test_env(1, args.max_steps)),
        ("lava_test_lower", make_test_env(2, args.max_steps)),
    ]

    results = []
    for label, env in envs:
        stats = evaluate_env(args.model, env, args.episodes, args.seed)
        results.append((label, stats))

    print_results(args.model, args.episodes, results)


if __name__ == "__main__":
    main()
