#!/usr/bin/env bash
set -euo pipefail

: "${CH2V3_RUN_DIR:?CH2V3_RUN_DIR must be set by the launcher}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHON="${CH2V3_PYTHON:-.venv/bin/python}"
"$PYTHON" experiments/chapter2_v3/collect.py --run-dir "$CH2V3_RUN_DIR"
