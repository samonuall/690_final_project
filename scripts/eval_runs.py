"""
Evaluate all runs in a parent directory produced by run_experiments.sh.

For each run_*/iter_*/ folder, picks the iteration with the highest peak mean
fitness (from its training_log.json), loads that iteration's model, and runs
fresh evaluation episodes with the true fitness function.

Prints per-run results and aggregate stats (mean ± std across run-means).

Usage:
    uv run python scripts/eval_runs.py \
        --runs-dir runs/boat_race_llm_one_shot_runs \
        --config  configs/boat_race_llm_one_shot.yaml \
        --episodes 50
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs.gridworld_wrapper import LLMRewardGridworld, FITNESS_FNS


def _fitness_as_total(fitness_fn):
    return lambda obs, action, next_obs, info: {"total": fitness_fn(obs, action, next_obs, info)}


def find_best_iter(run_dir: Path) -> tuple[Path, float]:
    """Return (best_iter_dir, peak_mean_fitness) using training_log.json checkpoints."""
    iter_dirs = sorted(
        run_dir.glob("iter_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not iter_dirs:
        raise FileNotFoundError(f"No iter_* dirs in {run_dir}")

    best_iter, best_fitness = None, -float("inf")
    for d in iter_dirs:
        log_path = d / "training_log.json"
        if not log_path.exists():
            continue
        with open(log_path) as f:
            log = json.load(f)
        checkpoint_means = [
            np.mean([ep["fitness"] for ep in entry["episodes"]])
            for entry in log.get("entries", [])
            if entry.get("episodes")
        ]
        if not checkpoint_means:
            continue
        peak = max(checkpoint_means)
        if peak > best_fitness:
            best_fitness, best_iter = peak, d

    if best_iter is None:
        raise FileNotFoundError(f"No valid training_log.json found in {run_dir}")
    return best_iter, best_fitness


def run_episode(model, env, seed=None) -> tuple[float, int]:
    obs, _ = env.reset(seed=seed) if seed is not None else env.reset()
    done, total_fitness, steps = False, 0.0, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        total_fitness += float(info.get("fitness", 0.0))
        steps += 1
    return total_fitness, steps


def evaluate_model(model_path: Path, env_cfg: dict, fitness_fn, n_episodes: int, seed: int) -> dict:
    env = LLMRewardGridworld(
        env_name=env_cfg["env_name"],
        max_iterations=env_cfg.get("max_iterations", 100),
    )
    env.set_reward_fn(_fitness_as_total(fitness_fn))
    env.set_fitness_fn(fitness_fn)
    model = PPO.load(str(model_path), env=env)

    returns, lengths = [], []
    for ep in range(n_episodes):
        r, s = run_episode(model, env, seed=seed + ep)
        returns.append(r)
        lengths.append(s)

    arr = np.array(returns, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean_steps": float(np.mean(lengths)),
    }


def _model_path(iter_dir: Path) -> Path:
    """Prefer best_model (saved at peak reward) over final ppo_policy."""
    if (iter_dir / "best_model.zip").exists():
        return iter_dir / "best_model"
    return iter_dir / "ppo_policy"


def _sep(width=62):
    return "─" * width


def main():
    parser = argparse.ArgumentParser(description="Aggregate eval across multi-run experiment dirs.")
    parser.add_argument("--runs-dir", required=True, help="Parent dir containing run_1/, run_2/, ...")
    parser.add_argument("--config",   required=True, help="YAML config used for these runs")
    parser.add_argument("--episodes", type=int, default=100,  help="Eval episodes per run")
    parser.add_argument("--seed",     type=int, default=42,  help="Base seed for eval episodes")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["env"]
    fitness_fn = FITNESS_FNS[env_cfg["fitness_fn"]]

    run_dirs = sorted(runs_dir.glob("run_*"), key=lambda p: int(p.name.split("_")[1]))
    if not run_dirs:
        print(f"No run_* directories found in {runs_dir}")
        sys.exit(1)

    col = 10
    val = 12
    headers = ["Mean", "Std", "Min", "Max", "Steps"]

    print()
    print(f"  Runs dir : {runs_dir}")
    print(f"  Config   : {args.config}  ({env_cfg['env_name']}, fitness={env_cfg['fitness_fn']})")
    print(f"  Episodes : {args.episodes} per run  |  seed base: {args.seed}")
    print()
    print(f"  {_sep()}")
    print(f"  {'Run':<{col}} {'Best iter':<12}" + "".join(f"{h:>{val}}" for h in headers))
    print(f"  {_sep()}")

    run_means: list[float] = []

    for run_dir in run_dirs:
        try:
            best_iter_dir, train_peak = find_best_iter(run_dir)
        except FileNotFoundError as e:
            print(f"  {'[SKIP]':<{col}} {e}")
            continue

        model_path = _model_path(best_iter_dir)
        try:
            stats = evaluate_model(model_path, env_cfg, fitness_fn, args.episodes, args.seed)
        except Exception as e:
            print(f"  {run_dir.name:<{col}} {best_iter_dir.name:<12}  [ERROR] {e}")
            continue

        run_means.append(stats["mean"])
        print(
            f"  {run_dir.name:<{col}} {best_iter_dir.name:<12}"
            f"{stats['mean']:>{val}.3f}"
            f"{stats['std']:>{val}.3f}"
            f"{stats['min']:>{val}.3f}"
            f"{stats['max']:>{val}.3f}"
            f"{stats['mean_steps']:>{val}.1f}"
        )

    print(f"  {_sep()}")

    if not run_means:
        print("\n  No runs successfully evaluated.")
        sys.exit(1)

    arr = np.array(run_means)
    print()
    print(f"  AGGREGATE  ({len(run_means)} / {len(run_dirs)} runs)")
    print(f"  {'Mean of run-means':<28} {arr.mean():>10.3f}")
    print(f"  {'Std  of run-means':<28} {arr.std():>10.3f}")
    print(f"  {'Min run-mean':<28} {arr.min():>10.3f}")
    print(f"  {'Max run-mean':<28} {arr.max():>10.3f}")
    print()


if __name__ == "__main__":
    main()
