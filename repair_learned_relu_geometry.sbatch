#!/usr/bin/env bash
#SBATCH --job-name=relu-repair
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=96G
#SBATCH --output=relu-repair-%j.log

set -euo pipefail
REPO_DIR="${GAD_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_DIR"
INPUT_ROOT="${1:?Pass the ORIGINAL output root, e.g. outputs/learned_relu_geometry/14872283}"
shift
STUDIES=(phase finite ablations tasks blocks landscape higher_order)
INDEX="${SLURM_ARRAY_TASK_ID:?Submit as --array=0,2,4,5 for job 14872283}"
if (( INDEX < 0 || INDEX >= ${#STUDIES[@]} )); then
  echo 'Array indices must be in 0..6' >&2
  exit 2
fi
STUDY="${STUDIES[$INDEX]}"
OUTPUT_ROOT="${GAD_GEOMETRY_REPAIR_OUTPUT:-${INPUT_ROOT}_repaired}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
"$REPO_DIR/.venv/bin/python" -u experiments/repair_learned_relu_geometry.py \
  --input-dir "$INPUT_ROOT/$STUDY" --output-dir "$OUTPUT_ROOT/$STUDY" \
  --workers "${SLURM_CPUS_PER_TASK:-1}" "$@"
