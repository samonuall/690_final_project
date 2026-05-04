#!/usr/bin/env bash
# Usage: ./scripts/run_experiments.sh <config> <n_runs>
# Example: ./scripts/run_experiments.sh configs/boat_race_llm_one_shot.yaml 5
#
# Runs the experiment N times, storing each run under:
#   <base_output_dir>_runs/run_1/
#   <base_output_dir>_runs/run_2/
#   ...

set -euo pipefail

CONFIG="${1:?Usage: $0 <config> <n_runs>}"
N_RUNS="${2:?Usage: $0 <config> <n_runs>}"

BASE_OUTPUT_DIR=$(uv run python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['output_dir'])")
PARENT_DIR="${BASE_OUTPUT_DIR}_runs"

echo "Config:     ${CONFIG}"
echo "Runs:       ${N_RUNS}"
echo "Parent dir: ${PARENT_DIR}"
echo ""

for i in $(seq 1 "${N_RUNS}"); do
    RUN_DIR="${PARENT_DIR}/run_${i}"
    echo "=== Starting run ${i} / ${N_RUNS} → ${RUN_DIR} ==="
    mkdir -p "${RUN_DIR}"
    uv run python main.py --config "${CONFIG}" --output-dir "${RUN_DIR}" \
        2>&1 | tee "${RUN_DIR}/stdout.log"
    echo "=== Run ${i} complete ==="
    echo ""
done

echo "All ${N_RUNS} runs complete. Results in: ${PARENT_DIR}"
