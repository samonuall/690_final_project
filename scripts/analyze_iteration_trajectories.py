"""
Iteration-trajectory analysis for LLM-designed reward functions.

For each loop run (3 iterations), evaluates the iter's best_model on the relevant
environment(s) and reports mean fitness per iteration. This shows how the policy's
ground-truth performance changes as the LLM revises its reward function across
iterations.

  - Boat race: mean fitness across N eval episodes on the training env, per iter.
  - Lava: mean fitness on train + lava_test_upper + lava_test_lower, per iter
    (one line per env; tests reward-function generalization under distribution shift).
  - Plus per-run fitness/reward over training-step overlays across iterations.

Usage:
    uv run python scripts/analyze_iteration_trajectories.py
    uv run python scripts/analyze_iteration_trajectories.py --episodes 50
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.gridworld_wrapper import LLMRewardGridworld, lava_fitness, boat_race_fitness
from src.envs.distributional_shift_adapter import ensure_distributional_shift_test_env_registered

logging.basicConfig(level=logging.WARNING)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "iteration_traces"
FIG_DIR = ANALYSIS_DIR / "figures"


# ---------------------------------------------------------------------------
# Run registry
# ---------------------------------------------------------------------------

@dataclass
class RunGroup:
    label: str
    env_kind: str         # "boat_race" or "lava"
    model: str            # "claude" or "qwen"
    run_dirs: list[Path]


def discover_runs() -> list[RunGroup]:
    r = PROJECT_ROOT / "runs"
    return [
        RunGroup(
            label="boat_race_claude",
            env_kind="boat_race",
            model="claude",
            run_dirs=[
                r / "boat_race_llm_one_shot_3iters_runs/run_1",
                r / "boat_race_llm_one_shot_3iters_runs/run_2",
            ],
        ),
        RunGroup(
            label="boat_race_qwen",
            env_kind="boat_race",
            model="qwen",
            run_dirs=[
                # run_2 has only iter_0 — drop it
                r / "boat_race_llm_one_shot_qwen3_3_iters_runs/run_1",
            ],
        ),
        RunGroup(
            label="lava_claude",
            env_kind="lava",
            model="claude",
            run_dirs=[
                r / "lava_llm_3_iters_runs/run_1",
                r / "lava_llm_3_iters_runs/run_2",
            ],
        ),
        RunGroup(
            label="lava_qwen",
            env_kind="lava",
            model="qwen",
            run_dirs=[
                r / "lava_llm_3_iters_qwen",
                r / "lava_llm_3_iters_qwen_runs/run_2",
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iter_dirs(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("iter_*"), key=lambda p: int(p.name.split("_")[1]))


def model_path(iter_dir: Path) -> Path | None:
    if (iter_dir / "best_model.zip").exists():
        return iter_dir / "best_model"
    if (iter_dir / "ppo_policy.zip").exists():
        return iter_dir / "ppo_policy"
    return None


def _fitness_only(fitness_fn: Callable):
    """Wrap a fitness fn into the dict form expected by LLMRewardGridworld."""
    return lambda obs, action, next_obs, info: {"total": fitness_fn(obs, action, next_obs, info)}


def make_boat_race_env(max_steps: int) -> LLMRewardGridworld:
    env = LLMRewardGridworld(env_name="boat_race", max_iterations=max_steps)
    env.set_reward_fn(_fitness_only(boat_race_fitness))
    env.set_fitness_fn(boat_race_fitness)
    return env


def make_lava_env(role: str, max_steps: int) -> LLMRewardGridworld:
    """role ∈ {'train', 'test_upper', 'test_lower'}."""
    if role == "train":
        env = LLMRewardGridworld(env_name="distributional_shift", max_iterations=max_steps)
    else:
        # eval_lava_runs.py uses level_choice 1 = upper, 2 = lower
        level = 1 if role == "test_upper" else 2
        env = LLMRewardGridworld(
            env_name="distributional_shift_test",
            max_iterations=max_steps,
            env_kwargs={"is_testing": True, "level_choice": level},
        )
    env.set_reward_fn(_fitness_only(lava_fitness))
    env.set_fitness_fn(lava_fitness)
    return env


def evaluate_policy(model, env, n_episodes: int, base_seed: int) -> dict:
    """Run n_episodes episodes; return mean/std/sem of cumulative fitness."""
    model.set_env(env)
    fitnesses, lengths = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        done, total_f, steps = False, 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total_f += float(info.get("fitness", 0.0))
            steps += 1
        fitnesses.append(total_f)
        lengths.append(steps)
    arr = np.asarray(fitnesses, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "sem": float(arr.std() / max(np.sqrt(len(arr)), 1)),
        "mean_steps": float(np.mean(lengths)),
        "n_episodes": int(len(arr)),
    }


# ---------------------------------------------------------------------------
# Per-run analysis
# ---------------------------------------------------------------------------

def analyze_boat_race_run(run_dir: Path, n_episodes: int, base_seed: int, max_steps: int) -> list[dict]:
    out = []
    for it_dir in iter_dirs(run_dir):
        it = int(it_dir.name.split("_")[1])
        mp = model_path(it_dir)
        if mp is None:
            print(f"  [skip] {it_dir.name}: no model")
            continue
        env = make_boat_race_env(max_steps)
        try:
            model = PPO.load(
                str(mp),
                env=env,
                custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0,
                                "clip_range": lambda _: 0.0},
            )
        except Exception as e:
            print(f"  [skip] {it_dir.name}: model load failed: {e}")
            continue
        stats = evaluate_policy(model, env, n_episodes, base_seed)
        out.append({"iter": it, "scope": "train", **stats})
    return out


def analyze_lava_run(run_dir: Path, n_episodes: int, base_seed: int, max_steps: int) -> list[dict]:
    out = []
    for it_dir in iter_dirs(run_dir):
        it = int(it_dir.name.split("_")[1])
        mp = model_path(it_dir)
        if mp is None:
            print(f"  [skip] {it_dir.name}: no model")
            continue
        for role in ("train", "test_upper", "test_lower"):
            env = make_lava_env(role, max_steps)
            try:
                model = PPO.load(
                    str(mp),
                    env=env,
                    custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0,
                                    "clip_range": lambda _: 0.0},
                )
            except Exception as e:
                print(f"  [skip] {it_dir.name} {role}: model load failed: {e}")
                continue
            stats = evaluate_policy(model, env, n_episodes, base_seed)
            out.append({"iter": it, "scope": role, **stats})
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_overlay(group: RunGroup) -> None:
    """For each run in the group, overlay fitness/reward over training-steps across iters."""
    n_runs = len(group.run_dirs)
    fig, axes = plt.subplots(n_runs, 2, figsize=(12, 3.2 * max(1, n_runs)), squeeze=False)
    for row, run_dir in enumerate(group.run_dirs):
        ax_f, ax_r = axes[row, 0], axes[row, 1]
        for it_dir in iter_dirs(run_dir):
            log_path = it_dir / "training_log.json"
            if not log_path.exists():
                continue
            with open(log_path) as f:
                log = json.load(f)
            steps = [e["step"] for e in log["entries"]]
            mean_f = [np.mean([ep["fitness"] for ep in e["episodes"]]) for e in log["entries"]]
            mean_r = [np.mean([ep["reward_components"].get("total", 0.0) for ep in e["episodes"]])
                      for e in log["entries"]]
            ax_f.plot(steps, mean_f, label=it_dir.name)
            ax_r.plot(steps, mean_r, label=it_dir.name)
        ax_f.set_title(f"{run_dir.name}: fitness")
        ax_r.set_title(f"{run_dir.name}: reward total")
        ax_f.set_xlabel("training step")
        ax_r.set_xlabel("training step")
        ax_f.legend(fontsize=7)
        ax_r.legend(fontsize=7)
    fig.suptitle(f"{group.label}: fitness and reward per iteration during training")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{group.label}_training_overlay.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_boat_race_fitness(group: RunGroup, per_run_records: list[list[dict]]) -> None:
    """One line per run + a mean line. x=iter, y=mean fitness."""
    fig, ax = plt.subplots(figsize=(6, 4))
    iters_all = sorted({r["iter"] for recs in per_run_records for r in recs})
    for run_dir, records in zip(group.run_dirs, per_run_records):
        recs = sorted(records, key=lambda r: r["iter"])
        if not recs:
            continue
        xs = [r["iter"] for r in recs]
        ys = [r["mean"] for r in recs]
        yerrs = [r["sem"] for r in recs]
        ax.errorbar(xs, ys, yerr=yerrs, marker="o", capsize=3, label=run_dir.name)
    means = []
    for it in iters_all:
        vals = [r["mean"] for recs in per_run_records for r in recs if r["iter"] == it]
        means.append(np.mean(vals) if vals else float("nan"))
    if any(not np.isnan(m) for m in means):
        ax.plot(iters_all, means, color="black", linestyle="--", linewidth=2,
                marker="s", label="mean")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xticks(iters_all)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean episode fitness (eval)")
    ax.set_title(f"Boat Race — {group.model}: policy fitness across iterations")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{group.label}_fitness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lava_fitness(group: RunGroup, per_run_records: list[list[dict]]) -> None:
    """One subplot per env (train/test_upper/test_lower). Within each, one line per run + mean."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    role_order = ["train", "test_upper", "test_lower"]
    iters_all = sorted({r["iter"] for recs in per_run_records for r in recs})
    for ax, role in zip(axes, role_order):
        for run_dir, records in zip(group.run_dirs, per_run_records):
            recs = sorted([r for r in records if r["scope"] == role], key=lambda r: r["iter"])
            if not recs:
                continue
            xs = [r["iter"] for r in recs]
            ys = [r["mean"] for r in recs]
            yerrs = [r["sem"] for r in recs]
            ax.errorbar(xs, ys, yerr=yerrs, marker="o", capsize=3, label=run_dir.name)
        means = []
        for it in iters_all:
            vals = [r["mean"] for recs in per_run_records for r in recs
                    if r["iter"] == it and r["scope"] == role]
            means.append(np.mean(vals) if vals else float("nan"))
        if any(not np.isnan(m) for m in means):
            ax.plot(iters_all, means, color="black", linestyle="--", linewidth=2,
                    marker="s", label="mean")
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_xticks(iters_all)
        ax.set_xlabel("Iteration")
        ax.set_title(role)
        if role == "train":
            ax.set_ylabel("Mean episode fitness (eval)")
        ax.legend(fontsize=7)
    fig.suptitle(f"Lava — {group.model}: policy fitness across iterations and envs")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{group.label}_fitness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV / orchestration
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=100)
    args = p.parse_args()

    ensure_distributional_shift_test_env_registered()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for group in discover_runs():
        print(f"\n=== {group.label} ===")
        per_run_records: list[list[dict]] = []
        for run_dir in group.run_dirs:
            print(f"  → {run_dir}")
            if not run_dir.exists():
                print(f"    (missing dir — skipped)")
                per_run_records.append([])
                continue
            if group.env_kind == "boat_race":
                recs = analyze_boat_race_run(run_dir, args.episodes, args.seed, args.max_steps)
            else:
                recs = analyze_lava_run(run_dir, args.episodes, args.seed, args.max_steps)
            per_run_records.append(recs)
            for r in recs:
                row = {"group": group.label, "model": group.model, "env": group.env_kind,
                       "run": run_dir.name, **r}
                all_rows.append(row)
                print(f"    iter={r['iter']:>2} scope={r['scope']:<11} "
                      f"mean_fitness={r['mean']:>8.3f}  sem={r['sem']:>6.3f}  "
                      f"steps={r['mean_steps']:>5.1f}")

        plot_training_overlay(group)
        if group.env_kind == "boat_race":
            plot_boat_race_fitness(group, per_run_records)
        else:
            plot_lava_fitness(group, per_run_records)

    write_csv(all_rows, ANALYSIS_DIR / "iteration_fitness.csv")
    print(f"\nWrote CSV: {ANALYSIS_DIR / 'iteration_fitness.csv'}")
    print(f"Wrote figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
