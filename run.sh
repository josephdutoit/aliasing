#!/bin/bash
#SBATCH --job-name=aliasing
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=1024M
#SBATCH --output=/home/jcdutoit/aliasing/slurm_logs/%x-%j.out
#SBATCH --error=/home/jcdutoit/aliasing/slurm_logs/%x-%j.err

set -euo pipefail

REPO_DIR="/home/jcdutoit/aliasing"
PYTHON="${REPO_DIR}/.venv/bin/python3"
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-1}"

SCRIPT="$1"
OUTPUT_ROOT="$2"

if [[ "$SCRIPT" != /* ]]; then
    SCRIPT="${REPO_DIR}/${SCRIPT}"
fi

if [[ "$OUTPUT_ROOT" != /* ]]; then
    OUTPUT_ROOT="${REPO_DIR}/${OUTPUT_ROOT}"
fi

if [[ ! -f "$SCRIPT" ]]; then
    echo "Experiment script not found: $SCRIPT" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT/samplewise"
mkdir -p "$OUTPUT_ROOT/featurewise"

cd "$REPO_DIR"

# One thread per worker avoids CPU oversubscription.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

source "${REPO_DIR}/.venv/bin/activate"

echo "Running samplewise experiment"
"$PYTHON" "$SCRIPT" \
    --num-workers "$NUM_WORKERS" \
    --sweep samplewise \
    --output-dir "$OUTPUT_ROOT/samplewise"

echo "Running featurewise experiment"
"$PYTHON" "$SCRIPT" \
    --num-workers "$NUM_WORKERS" \
    --sweep featurewise \
    --output-dir "$OUTPUT_ROOT/featurewise"

echo "Completed both experiments."
