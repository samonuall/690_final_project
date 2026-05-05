"""
Evaluate all runs in a parent directory produced by run_experiments.sh (lava env).

For each run_*/iter_*/ folder, picks the iteration with the highest peak mean
fitness on the TRAIN environment, loads that iteration's model, and evaluates
on all three lava environments (train, lava_test_upper, lava_test_lower).

Prints per-run results and aggregate stats (mean ± std across run-means).

Usage:
    uv run python scripts/eval_lava_runs.py \
        --runs-dir runs/lava_llm_one_shot_runs
    uv run python scripts/eval_lava_runs.py \
        --runs-dir runs/lava_llm_one_shot_runs --episodes 50 --seed 123
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.gridworld_wrapper import LLMRewardGridworld, lava_fitness
from src.envs.distributional_shift_adapter import ensure_distributional_shift_test_env_registered


def _fitness_as_total(obs, action, next_obs, info):
    return {"total": lava_fitness(obs, action, next_obs, info)}


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


def _model_path(iter_dir: Path) -> Path:
    if (iter_dir / "best_model.zip").exists():
        return iter_dir / "best_model"
    return iter_dir / "ppo_policy"


def make_train_env(max_steps: int) -> LLMRewardGridworld:
    env = LLMRewardGridworld(env_name="distributional_shift", max_iterations=max_steps)
    env.set_reward_fn(_fitness_as_total)
    env.set_fitness_fn(lava_fitness)
    return env


def make_test_env(level_choice: int, max_steps: int) -> LLMRewardGridworld:
    env = LLMRewardGridworld(
        env_name="distributional_shift_test",
        max_iterations=max_steps,
        env_kwargs={"is_testing": True, "level_choice": level_choice},
    )
    env.set_reward_fn(_fitness_as_total)
    env.set_fitness_fn(lava_fitness)
    return env


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


def evaluate_env(model, env, n_episodes: int, seed: int) -> dict:
    model.set_env(env)
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


ENV_LABELS = ["train", "lava_test_upper", "lava_test_lower"]


def _sep(width: int) -> str:
    return "─" * width


def print_results(runs_dir: Path, episodes: int, seed: int, rows: list[tuple]) -> None:
    run_col = 10
    iter_col = 12
    val = 8
    n_envs = len(ENV_LABELS)
    env_w = val * 3 + 2  # mean + std + steps columns per env

    total_w = run_col + iter_col + n_envs * (env_w + 2)

    print()
    print(f"  Runs dir : {runs_dir}")
    print(f"  Episodes : {episodes} per environment  |  seed base: {seed}")
    print()

    # Header row 1 — env names
    header1 = f"  {'Run':<{run_col}} {'Best iter':<{iter_col}}"
    for label in ENV_LABELS:
        header1 += f"  {label:^{env_w}}"
    # Header row 2 — mean / std / steps
    header2 = f"  {'':<{run_col}} {'':<{iter_col}}"
    for _ in ENV_LABELS:
        header2 += f"  {'mean':>{val}} {'std':>{val}} {'steps':<{val}}"

    print(f"  {_sep(total_w)}")
    print(header1)
    print(header2)
    print(f"  {_sep(total_w)}")

    run_means = {label: [] for label in ENV_LABELS}
    run_steps = {label: [] for label in ENV_LABELS}

    for run_name, best_iter_name, stats_by_env, error in rows:
        if error:
            print(f"  {run_name:<{run_col}} {best_iter_name:<{iter_col}}  [ERROR] {error}")
            continue
        line = f"  {run_name:<{run_col}} {best_iter_name:<{iter_col}}"
        for label in ENV_LABELS:
            s = stats_by_env[label]
            line += f"  {s['mean']:>{val}.2f} {s['std']:>{val}.2f} {s['mean_steps']:<{val}.1f}"
            run_means[label].append(s["mean"])
            run_steps[label].append(s["mean_steps"])
        print(line)

    print(f"  {_sep(total_w)}")

    valid = len(run_means[ENV_LABELS[0]])
    total = len(rows)
    print()
    print(f"  AGGREGATE  ({valid} / {total} runs)")
    for label in ENV_LABELS:
        if not run_means[label]:
            continue
        arr = np.array(run_means[label])
        avg_steps = np.mean(run_steps[label])
        print(f"  {label:<20}  mean={arr.mean():>8.3f}  std={arr.std():>7.3f}"
              f"  min={arr.min():>8.3f}  max={arr.max():>8.3f}  avg_steps={avg_steps:>6.1f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Aggregate lava eval across multi-run experiment dirs.")
    parser.add_argument(
        "--runs-dir",
        required=True,
        help=(
            "Either (a) parent dir containing run_1/, run_2/, ... or "
            "(b) a single run dir containing iter_0/, iter_1/, ... directly"
        ),
    )
    parser.add_argument("--episodes", type=int, default=100, help="Eval episodes per environment per run")
    parser.add_argument("--seed",     type=int, default=42,  help="Base seed for eval episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    args = parser.parse_args()

    ensure_distributional_shift_test_env_registered()

    runs_dir = Path(args.runs_dir)

    run_dirs = sorted(runs_dir.glob("run_*"), key=lambda p: int(p.name.split("_")[1]))
    if not run_dirs:
        # Support a single-run layout where iter_* are directly under runs_dir.
        iter_dirs = list(runs_dir.glob("iter_*"))
        if iter_dirs:
            run_dirs = [runs_dir]
        else:
            print(f"No run_* directories found in {runs_dir}")
            print(f"No iter_* directories found in {runs_dir}")
            sys.exit(1)

    rows = []
    for run_dir in run_dirs:
        try:
            best_iter_dir, _ = find_best_iter(run_dir)
        except FileNotFoundError as e:
            rows.append((run_dir.name, "—", {}, str(e)))
            continue

        model_path = _model_path(best_iter_dir)
        train_env = make_train_env(args.max_steps)
        try:
            model = PPO.load(str(model_path), env=train_env)
        except Exception as e:
            rows.append((run_dir.name, best_iter_dir.name, {}, f"load failed: {e}"))
            continue

        envs = [
            ("train",           train_env),
            ("lava_test_upper", make_test_env(1, args.max_steps)),
            ("lava_test_lower", make_test_env(2, args.max_steps)),
        ]

        stats_by_env = {}
        error = None
        for label, env in envs:
            try:
                stats_by_env[label] = evaluate_env(model, env, args.episodes, args.seed)
            except Exception as e:
                error = f"{label}: {e}"
                break

        rows.append((run_dir.name, best_iter_dir.name, stats_by_env, error))

    print_results(runs_dir, args.episodes, args.seed, rows)


if __name__ == "__main__":
    main()
