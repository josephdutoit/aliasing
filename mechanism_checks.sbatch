#!/usr/bin/env bash
#SBATCH --job-name=gad-mechanism
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --output=gad-mechanism-%j.log

set -euo pipefail
REPO_DIR="${GAD_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_DIR"
PYTHON="$REPO_DIR/.venv/bin/python"
SOURCE_DIR="${GAD_SOURCE_DIR:-outputs/learned_aliasing/14814813}"
OUTPUT_DIR="${GAD_MECHANISM_DIR:-outputs/gad_mechanisms/${SLURM_JOB_ID:-manual}}"
if [[ ! -x "$PYTHON" || ! -f experiments/15_gad_mechanism_checks.py || ! -f "$SOURCE_DIR/configuration.json" || ! -d "$SOURCE_DIR/shards" ]]; then
  echo 'Submit from the checkout with .venv installed and the full experiment-14 source shards present. See GAD_MECHANISM_CHECKS.md.' >&2
  exit 2
fi
ARGS=(--source-dir "$SOURCE_DIR" --output-dir "$OUTPUT_DIR"
      --profile "${GAD_MECHANISM_PROFILE:-overnight}" --workers "${SLURM_CPUS_PER_TASK:-1}")
if [[ "${GAD_MECHANISM_RESUME:-0}" == 1 ]]; then ARGS+=(--resume); fi
if [[ "${GAD_MECHANISM_SUMMARIZE_ONLY:-0}" == 1 ]]; then ARGS+=(--summarize-only); fi
export PYTHONUNBUFFERED=1
"$PYTHON" -u experiments/15_gad_mechanism_checks.py "${ARGS[@]}" "$@"
