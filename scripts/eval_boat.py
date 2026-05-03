"""
Evaluate a trained PPO policy on the boat race environment.

Usage:
    uv run python scripts/eval_boat.py --model runs/boat_race_test/test/ppo_policy
    uv run python scripts/eval_boat.py --model runs/boat_race_test/test/ppo_policy --episodes 10 --seed 42
"""

import argparse
import os
import sys

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.gridworld_wrapper import LLMRewardGridworld, boat_race_fitness


def _fitness_as_total(obs, action, next_obs, info):
    return {"total": boat_race_fitness(obs, action, next_obs, info)}


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


def evaluate(model_path, episodes, seed, max_steps):
    env = LLMRewardGridworld(env_name="boat_race", max_iterations=max_steps)
    env.set_reward_fn(_fitness_as_total)
    env.set_fitness_fn(boat_race_fitness)

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


def print_results(model_path, episodes, stats):
    col_w = 20
    val_w = 14
    headers = ["Mean Return", "Std Return", "Min Return", "Max Return", "Mean Steps"]
    sep = "─" * (col_w + val_w * len(headers))

    print()
    print(f"  Model   : {model_path}")
    print(f"  Episodes: {episodes}")
    print()
    print(f"  {sep}")
    print(f"  {'Environment':<{col_w}}" + "".join(f"{h:>{val_w}}" for h in headers))
    print(f"  {sep}")
    print(
        f"  {'boat_race':<{col_w}}"
        f"{stats['mean_return']:>{val_w}.3f}"
        f"{stats['std_return']:>{val_w}.3f}"
        f"{stats['min_return']:>{val_w}.3f}"
        f"{stats['max_return']:>{val_w}.3f}"
        f"{stats['mean_steps']:>{val_w}.1f}"
    )
    print(f"  {sep}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO policy on the boat race environment.")
    parser.add_argument("--model", required=True, help="Path to saved model (.zip may be omitted)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    args = parser.parse_args()

    stats = evaluate(args.model, args.episodes, args.seed, args.max_steps)
    print_results(args.model, args.episodes, stats)


if __name__ == "__main__":
    main()
