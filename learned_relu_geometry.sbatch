#!/usr/bin/env bash
#SBATCH --job-name=learned-relu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=96G
#SBATCH --output=gad-relu-%j.log

# One job runs all studies. --array=0-6 gives one independent job per study.
set -euo pipefail
REPO_DIR="${GAD_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_DIR"
PYTHON="$REPO_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" || ! -f experiments/17_learned_relu_geometry.py ]]; then
  echo 'Submit from the checkout with .venv installed. See LEARNED_RELU_GEOMETRY.md.' >&2
  exit 2
fi
STUDIES=(phase finite ablations tasks blocks landscape higher_order)
OUTPUT_ROOT="${GAD_GEOMETRY_OUTPUT:-outputs/learned_relu_geometry/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}"
STUDY="${GAD_GEOMETRY_EXPERIMENTS:-all}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#STUDIES[@]} )); then
    echo 'Array indices must be in 0..6.' >&2
    exit 2
  fi
  STUDY="${STUDIES[$SLURM_ARRAY_TASK_ID]}"
  OUTPUT_ROOT="$OUTPUT_ROOT/$STUDY"
fi
ARGS=(--output-dir "$OUTPUT_ROOT" --workers "${SLURM_CPUS_PER_TASK:-1}")
if [[ "${GAD_GEOMETRY_RESUME:-0}" == 1 ]]; then
  ARGS+=(--resume)
else
  ARGS+=(--profile "${GAD_GEOMETRY_PROFILE:-overnight}" --experiments "$STUDY")
fi
if [[ "${GAD_GEOMETRY_SUMMARIZE_ONLY:-0}" == 1 ]]; then ARGS+=(--summarize-only); fi
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PYTHON" -u experiments/17_learned_relu_geometry.py "${ARGS[@]}" "$@"
