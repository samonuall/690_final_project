"""
Run all ablation experiments in parallel.

For each of the 8 ablation configs (2 models × 2 ablations × 2 methods):
  - one-shot configs are run 3 times each
  - loop (3-iter) configs are run 2 times each
Total: 4*3 + 4*2 = 20 training runs.

Each run launches `uv run python main.py --config <cfg> --output-dir <runs/<base>_runs/run_N>`
in its own subprocess. Up to --max-workers (default 4) run concurrently.

Usage:
    uv run python scripts/run_ablations.py
    uv run python scripts/run_ablations.py --max-workers 2 --dry-run
    uv run python scripts/run_ablations.py --only boat_race_simple_one_shot
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (config_name, n_runs)
ABLATION_CONFIGS: list[tuple[str, int]] = [
    # Boat race — "keep it simple" ablation
    ("boat_race_simple_one_shot",        2),
    ("boat_race_simple_one_shot_qwen",   3),
    ("boat_race_simple_3_iters",         2),
    ("boat_race_simple_3_iters_qwen",    2),
    # Lava — "distribution-shift aware" ablation
    ("lava_distshift_one_shot",          3),
    ("lava_distshift_one_shot_qwen",     3),
    ("lava_distshift_3_iters",           2),
    ("lava_distshift_3_iters_qwen",      2),
]


@dataclass
class RunSpec:
    config_name: str
    config_path: Path
    run_idx: int
    output_dir: Path

    @property
    def label(self) -> str:
        return f"{self.config_name}/run_{self.run_idx}"


def expand_runs() -> list[RunSpec]:
    runs: list[RunSpec] = []
    for cfg_name, n_runs in ABLATION_CONFIGS:
        cfg_path = PROJECT_ROOT / "configs" / f"{cfg_name}.yaml"
        if not cfg_path.exists():
            sys.exit(f"Missing config: {cfg_path}")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        base = cfg["output_dir"]  # e.g. "runs/boat_race_simple_one_shot"
        parent = PROJECT_ROOT / f"{base}_runs"
        for i in range(1, n_runs + 1):
            runs.append(RunSpec(
                config_name=cfg_name,
                config_path=cfg_path,
                run_idx=i,
                output_dir=parent / f"run_{i}",
            ))
    return runs


def execute_run(spec: RunSpec) -> tuple[str, int, str]:
    """Run main.py in a subprocess; return (label, returncode, log_path)."""
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = spec.output_dir / "stdout.log"
    cmd = [
        "uv", "run", "python", "main.py",
        "--config", str(spec.config_path),
        "--output-dir", str(spec.output_dir),
    ]
    start = time.time()
    with open(stdout_log, "w") as out:
        out.write(f"$ {' '.join(shlex.quote(c) for c in cmd)}\n\n")
        out.flush()
        proc = subprocess.run(
            cmd, stdout=out, stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT), env=os.environ.copy(),
        )
    elapsed = time.time() - start
    return spec.label, proc.returncode, str(stdout_log), elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true",
                   help="List the runs that would be executed, don't actually run.")
    p.add_argument("--only", action="append", default=None,
                   help="Only run configs whose name contains this substring (repeatable).")
    args = p.parse_args()

    runs = expand_runs()
    if args.only:
        runs = [r for r in runs if any(o in r.config_name for o in args.only)]

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Total runs:   {len(runs)}")
    print(f"Workers:      {args.max_workers}")
    print()
    for r in runs:
        print(f"  - {r.label:<55} → {r.output_dir.relative_to(PROJECT_ROOT)}")
    print()

    if args.dry_run:
        print("[dry-run] not executing.")
        return

    completed: list[tuple[str, int, str, float]] = []
    failed: list[str] = []

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(execute_run, r): r for r in runs}
        for fut in as_completed(futures):
            spec = futures[fut]
            try:
                label, rc, log_path, elapsed = fut.result()
            except Exception as e:
                failed.append(spec.label)
                print(f"[FAIL] {spec.label}: exception {e!r}")
                continue
            completed.append((label, rc, log_path, elapsed))
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"[{status}] {label}  ({elapsed:.1f}s)  log={log_path}")
            if rc != 0:
                failed.append(label)

    total = time.time() - t0
    print()
    print(f"Done in {total:.1f}s. {len(completed) - len(failed)}/{len(completed)} succeeded.")
    if failed:
        print(f"Failed runs ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
