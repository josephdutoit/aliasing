#!/usr/bin/env bash
# Resources and log paths are supplied by launch_overnight.sh after user selection.
set -euo pipefail

: "${CH2V3_RUN_DIR:?CH2V3_RUN_DIR must be set by the launcher}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON="${CH2V3_PYTHON:-.venv/bin/python}"
exec "$PYTHON" experiments/chapter2_v3/run.py \
  --run-dir "$CH2V3_RUN_DIR" \
  --job-index "$SLURM_ARRAY_TASK_ID"
