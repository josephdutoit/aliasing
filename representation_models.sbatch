#!/usr/bin/env bash
#SBATCH --job-name=gad-models
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --output=gad-models-%j.log

set -euo pipefail
REPO_DIR="${GAD_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_DIR"
PYTHON="$REPO_DIR/.venv/bin/python"
OUTPUT_DIR="${GAD_MODEL_OUTPUT:-outputs/representation_models/${SLURM_JOB_ID:-manual}}"
if [[ ! -x "$PYTHON" || ! -f experiments/16_representation_model_benchmark.py ]]; then
  echo 'Submit from the repository checkout with .venv installed. See REPRESENTATION_MODELS.md.' >&2
  exit 2
fi
ARGS=(--output-dir "$OUTPUT_DIR" --workers "${SLURM_CPUS_PER_TASK:-1}")
if [[ "${GAD_MODEL_RESUME:-0}" == 1 ]]; then
  ARGS+=(--resume)
else
  ARGS+=(--profile "${GAD_MODEL_PROFILE:-overnight}")
fi
if [[ "${GAD_MODEL_SUMMARIZE_ONLY:-0}" == 1 ]]; then ARGS+=(--summarize-only); fi
export PYTHONUNBUFFERED=1
"$PYTHON" -u experiments/16_representation_model_benchmark.py "${ARGS[@]}" "$@"
