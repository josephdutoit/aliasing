#!/usr/bin/env bash
#SBATCH --job-name=gad-survey
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=96G
#SBATCH --array=0-12%3
#SBATCH --output=gad-survey-%A_%a.log

set -euo pipefail
REPO_DIR="${GAD_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_DIR"
PYTHON="$REPO_DIR/.venv/bin/python"
OUTPUT_DIR="${SURVEY_OUTPUT:-outputs/learned_feature_survey/${SLURM_ARRAY_JOB_ID:-manual}}"
if [[ ! -x "$PYTHON" || ! -f experiments/17_learned_feature_survey.py ]]; then
  echo 'Submit from the checkout with .venv installed. See BROAD_FEATURE_SURVEY.md.' >&2
  exit 2
fi
ARGS=(--output-dir "$OUTPUT_DIR" --workers "${SLURM_CPUS_PER_TASK:-1}")
if [[ "${SURVEY_RESUME:-0}" == 1 ]]; then ARGS+=(--resume); fi
if [[ "${SURVEY_SUMMARIZE_ONLY:-0}" == 1 ]]; then ARGS+=(--summarize-only); fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then ARGS+=(--job-index "$SLURM_ARRAY_TASK_ID"); fi
export PYTHONUNBUFFERED=1
"$PYTHON" -u experiments/17_learned_feature_survey.py "${ARGS[@]}" "$@"
