#!/usr/bin/env bash
#SBATCH --job-name=learned-gad
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --output=learned-gad-%j.log

set -euo pipefail

# Slurm executes a spool copy: locate the checkout from the submission directory.
REPO_DIR="${GAD_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_DIR"
PYTHON="$REPO_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" || ! -f experiments/14_learned_aliasing_hypotheses.py ]]; then
  echo 'Submit from the repository root with .venv installed; see LEARNED_ALIASING_EXPERIMENTS.md' >&2
  exit 2
fi
"$PYTHON" -c 'import numpy, matplotlib; import aliasing.controlled_superposition'

OUTPUT_DIR="${HYP_RUN_DIR:-outputs/learned_aliasing/${SLURM_JOB_ID:-manual}}"
ARGS=(--profile "${HYP_PROFILE:-overnight}" --mode "${HYP_MODE:-all}"
      --task-basis "${HYP_TASK_BASIS:-orthogonal}" --workers "${SLURM_CPUS_PER_TASK:-1}"
      --output-dir "$OUTPUT_DIR")
if [[ "${HYP_RESUME:-0}" == 1 ]]; then ARGS+=(--resume); fi
if [[ "${HYP_SUMMARIZE_ONLY:-0}" == 1 ]]; then ARGS+=(--summarize-only); fi

# Extra sbatch script arguments override defaults, for example --steps 5000.
export PYTHONUNBUFFERED=1
"$PYTHON" -u experiments/14_learned_aliasing_hypotheses.py "${ARGS[@]}" "$@"
