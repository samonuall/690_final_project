"""
Analyze the prompt-ablation runs in runs/ (boat_race_simple_*, lava_distshift_*).

For each condition (env x model x {one-shot, loop}), evaluates every iter's
policy on the relevant env(s) and produces:
  - per-iteration fitness CSV
  - per-condition iteration-trajectory plot (loop only)
  - per-condition training overlay (loop only)
  - per-env summary plot across all 4 conditions
  - per-condition summary CSV (best-iter mean across runs)

Usage:
    uv run python scripts/analyze_prompt_ablation.py
    uv run python scripts/analyze_prompt_ablation.py --episodes 50
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

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.distributional_shift_adapter import ensure_distributional_shift_test_env_registered
from scripts.analyze_iteration_trajectories import (
    iter_dirs,
    model_path,
    make_boat_race_env,
    make_lava_env,
    evaluate_policy,
)

logging.basicConfig(level=logging.WARNING)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "prompt_ablation"
FIG_DIR = ANALYSIS_DIR / "figures"


# ---------------------------------------------------------------------------
# Condition registry
# ---------------------------------------------------------------------------

@dataclass
class Condition:
    label: str           # filename-safe label
    env_kind: str        # "boat_race" or "lava"
    model: str           # "claude" or "qwen"
    variant: str         # "one_shot" or "loop"
    runs_dir: Path       # parent dir containing run_1/, run_2/, ...

    @property
    def is_loop(self) -> bool:
        return self.variant == "loop"

    def run_dirs(self) -> list[Path]:
        if not self.runs_dir.exists():
            return []
        return sorted([p for p in self.runs_dir.iterdir()
                       if p.is_dir() and p.name.startswith("run_")])


def conditions() -> list[Condition]:
    r = PROJECT_ROOT / "runs"
    return [
        Condition("boat_simple_claude_oneshot", "boat_race", "claude", "one_shot",
                  r / "boat_race_simple_one_shot_runs"),
        Condition("boat_simple_claude_loop", "boat_race", "claude", "loop",
                  r / "boat_race_simple_3_iters_runs"),
        Condition("boat_simple_qwen_oneshot", "boat_race", "qwen", "one_shot",
                  r / "boat_race_simple_one_shot_qwen_runs"),
        Condition("boat_simple_qwen_loop", "boat_race", "qwen", "loop",
                  r / "boat_race_simple_3_iters_qwen_runs"),
        Condition("lava_distshift_claude_oneshot", "lava", "claude", "one_shot",
                  r / "lava_distshift_one_shot_runs"),
        Condition("lava_distshift_claude_loop", "lava", "claude", "loop",
                  r / "lava_distshift_3_iters_runs"),
        Condition("lava_distshift_qwen_oneshot", "lava", "qwen", "one_shot",
                  r / "lava_distshift_one_shot_qwen_runs"),
        Condition("lava_distshift_qwen_loop", "lava", "qwen", "loop",
                  r / "lava_distshift_3_iters_qwen_runs"),
    ]


# ---------------------------------------------------------------------------
# Per-condition evaluation
# ---------------------------------------------------------------------------

def eval_run_boat(run_dir: Path, n_episodes: int, base_seed: int, max_steps: int) -> list[dict]:
    out = []
    for it_dir in iter_dirs(run_dir):
        it = int(it_dir.name.split("_")[1])
        mp = model_path(it_dir)
        if mp is None:
            print(f"    [skip] {it_dir.name}: no model")
            continue
        env = make_boat_race_env(max_steps)
        try:
            model = PPO.load(str(mp), env=env, custom_objects={
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
            })
        except Exception as e:
            print(f"    [skip] {it_dir.name}: load failed: {e}")
            continue
        stats = evaluate_policy(model, env, n_episodes, base_seed)
        out.append({"iter": it, "scope": "train", **stats})
    return out


def eval_run_lava(run_dir: Path, n_episodes: int, base_seed: int, max_steps: int) -> list[dict]:
    out = []
    for it_dir in iter_dirs(run_dir):
        it = int(it_dir.name.split("_")[1])
        mp = model_path(it_dir)
        if mp is None:
            print(f"    [skip] {it_dir.name}: no model")
            continue
        for role in ("train", "test_upper", "test_lower"):
            env = make_lava_env(role, max_steps)
            try:
                model = PPO.load(str(mp), env=env, custom_objects={
                    "learning_rate": 0.0,
                    "lr_schedule": lambda _: 0.0,
                    "clip_range": lambda _: 0.0,
                })
            except Exception as e:
                print(f"    [skip] {it_dir.name} {role}: load failed: {e}")
                continue
            stats = evaluate_policy(model, env, n_episodes, base_seed)
            out.append({"iter": it, "scope": role, **stats})
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_iter_trajectory(cond: Condition, per_run_records: list[list[dict]]) -> None:
    """Iteration trajectory for loop conditions. Boat = 1 panel; lava = 3 panels."""
    if not cond.is_loop:
        return
    if cond.env_kind == "boat_race":
        roles = ["train"]
        fig, axes = plt.subplots(1, 1, figsize=(5, 4), squeeze=False)
    else:
        roles = ["train", "test_upper", "test_lower"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True, squeeze=False)

    iters_all = sorted({r["iter"] for recs in per_run_records for r in recs})

    for ax, role in zip(axes[0], roles):
        for run_dir, recs in zip(cond.run_dirs(), per_run_records):
            recs_role = sorted([r for r in recs if r["scope"] == role], key=lambda r: r["iter"])
            if not recs_role:
                continue
            xs = [r["iter"] for r in recs_role]
            ys = [r["mean"] for r in recs_role]
            yerrs = [r["sem"] for r in recs_role]
            ax.errorbar(xs, ys, yerr=yerrs, marker="o", capsize=3, label=run_dir.name)

        # Mean across runs
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
        ax.set_title(role if cond.env_kind == "lava" else "")
        ax.legend(fontsize=7)

    axes[0, 0].set_ylabel("Mean episode fitness (eval)")
    fig.suptitle(f"{cond.label}: fitness across iterations")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{cond.label}_iter_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_overlay(cond: Condition) -> None:
    """For each run in the loop condition, overlay fitness/reward over training steps."""
    if not cond.is_loop:
        return
    runs = cond.run_dirs()
    if not runs:
        return
    fig, axes = plt.subplots(len(runs), 2, figsize=(12, 3.2 * max(1, len(runs))), squeeze=False)
    for row, run_dir in enumerate(runs):
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
    fig.suptitle(f"{cond.label}: per-iter fitness and reward during training")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{cond.label}_training_overlay.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def best_iter_per_run(records: list[dict]) -> dict | None:
    """Pick the iter with highest train-fitness mean; return that iter's records by scope."""
    train_recs = [r for r in records if r["scope"] == "train"]
    if not train_recs:
        return None
    best = max(train_recs, key=lambda r: r["mean"])
    best_iter = best["iter"]
    return {r["scope"]: r for r in records if r["iter"] == best_iter}


def plot_env_summary(env_kind: str, all_results: dict[str, list[list[dict]]]) -> None:
    """One figure per env. Bars = mean fitness across runs of best iter (boat = 1 group, lava = 3 groups)."""
    conds = [c for c in conditions() if c.env_kind == env_kind]
    if not conds:
        return

    if env_kind == "boat_race":
        roles = ["train"]
    else:
        roles = ["train", "test_upper", "test_lower"]

    n_roles = len(roles)
    width = 0.8 / max(1, len(conds))
    x = np.arange(n_roles)

    fig, ax = plt.subplots(figsize=(2 + 1.5 * n_roles + 0.3 * len(conds), 4.5))

    for i, cond in enumerate(conds):
        per_run = all_results.get(cond.label, [])
        means_per_role = []
        sems_per_role = []
        for role in roles:
            vals = []
            for recs in per_run:
                bi = best_iter_per_run(recs)
                if bi and role in bi:
                    vals.append(bi[role]["mean"])
            if vals:
                means_per_role.append(float(np.mean(vals)))
                sems_per_role.append(float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
            else:
                means_per_role.append(float("nan"))
                sems_per_role.append(0.0)

        offsets = x + (i - (len(conds) - 1) / 2) * width
        bars = ax.bar(offsets, means_per_role, width, yerr=sems_per_role, capsize=3,
                      label=cond.label.replace("boat_simple_", "").replace("lava_distshift_", ""))

    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(roles)
    ax.set_ylabel("Mean episode fitness (best iter, averaged over runs)")
    title = "Boat Race — 'simple' prompt ablation" if env_kind == "boat_race" \
        else "Lava — 'distshift' prompt ablation"
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"summary_{env_kind}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_iter_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary_csv(all_results: dict[str, list[list[dict]]], path: Path) -> None:
    """One row per (condition, scope): mean ± sem of best-iter mean across runs."""
    rows = []
    for cond in conditions():
        per_run = all_results.get(cond.label, [])
        if not per_run:
            continue
        roles = ["train"] if cond.env_kind == "boat_race" else ["train", "test_upper", "test_lower"]
        for role in roles:
            vals = []
            steps = []
            for recs in per_run:
                bi = best_iter_per_run(recs)
                if bi and role in bi:
                    vals.append(bi[role]["mean"])
                    steps.append(bi[role]["mean_steps"])
            if not vals:
                continue
            arr = np.array(vals)
            rows.append({
                "condition": cond.label,
                "env_kind": cond.env_kind,
                "model": cond.model,
                "variant": cond.variant,
                "scope": role,
                "n_runs": len(vals),
                "mean_fitness": float(arr.mean()),
                "std_fitness": float(arr.std()),
                "sem_fitness": float(arr.std() / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
                "mean_steps": float(np.mean(steps)),
            })

    if not rows:
        return
    keys = ["condition", "env_kind", "model", "variant", "scope", "n_runs",
            "mean_fitness", "std_fitness", "sem_fitness", "mean_steps"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=100)
    args = p.parse_args()

    ensure_distributional_shift_test_env_registered()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    iter_rows: list[dict] = []
    all_results: dict[str, list[list[dict]]] = {}

    for cond in conditions():
        print(f"\n=== {cond.label} ({cond.runs_dir}) ===")
        per_run_records: list[list[dict]] = []
        for run_dir in cond.run_dirs():
            print(f"  → {run_dir.name}")
            if cond.env_kind == "boat_race":
                recs = eval_run_boat(run_dir, args.episodes, args.seed, args.max_steps)
            else:
                recs = eval_run_lava(run_dir, args.episodes, args.seed, args.max_steps)
            per_run_records.append(recs)
            for r in recs:
                iter_rows.append({
                    "condition": cond.label,
                    "env_kind": cond.env_kind,
                    "model": cond.model,
                    "variant": cond.variant,
                    "run": run_dir.name,
                    **r,
                })
                print(f"    iter={r['iter']:>2} scope={r['scope']:<11} "
                      f"mean={r['mean']:>8.3f}  sem={r['sem']:>6.3f}  steps={r['mean_steps']:>5.1f}")
        all_results[cond.label] = per_run_records

        plot_iter_trajectory(cond, per_run_records)
        plot_training_overlay(cond)

    plot_env_summary("boat_race", all_results)
    plot_env_summary("lava", all_results)

    write_iter_csv(iter_rows, ANALYSIS_DIR / "iteration_fitness.csv")
    write_summary_csv(all_results, ANALYSIS_DIR / "summary_best_iter.csv")
    print(f"\nWrote: {ANALYSIS_DIR / 'iteration_fitness.csv'}")
    print(f"Wrote: {ANALYSIS_DIR / 'summary_best_iter.csv'}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
