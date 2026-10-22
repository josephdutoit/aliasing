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

if [[ "$(basename "$SCRIPT")" == "04_superposition_aliasing.py" ]]; then
    mkdir -p "$OUTPUT_ROOT/superposition"
    cd "$REPO_DIR"
    export MPLBACKEND=Agg
    source "${REPO_DIR}/.venv/bin/activate"
    echo "Running controlled superposition-aliasing experiment"
    "$PYTHON" "$SCRIPT" --output-dir "$OUTPUT_ROOT/superposition"
    exit 0
fi

if [[ "$(basename "$SCRIPT")" == "05_static_dictionary_geometry.py" ]]; then
    OUT_DIR="$OUTPUT_ROOT/static_dictionary_geometry"
    mkdir -p "$OUT_DIR"
    cd "$REPO_DIR"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
    source "${REPO_DIR}/.venv/bin/activate"
    "$PYTHON" "$SCRIPT" \
        --dimension "${DIMENSION:-4}" \
        --feature-counts "${FEATURE_COUNTS:-4,5,8}" \
        --max-sparsities "${MAX_SPARSITIES:-1,2}" \
        --sample-sizes "${SAMPLE_SIZES:-8,16,32}" \
        --num-seeds "${NUM_SEEDS:-20}" \
        --output-dir "$OUT_DIR"
    exit 0
fi

if [[ "$(basename "$SCRIPT")" == "06_trained_sparse_feature_geometry.py" ]]; then
    OUT_DIR="$OUTPUT_ROOT/trained_sparse_feature_geometry"
    mkdir -p "$OUT_DIR"
    cd "$REPO_DIR"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
    source "${REPO_DIR}/.venv/bin/activate"
    "$PYTHON" "$SCRIPT" \
        --feature-counts "${FEATURE_COUNTS:-8,12}" \
        --bottlenecks "${BOTTLENECKS:-3,4}" \
        --sample-sizes "${SAMPLE_SIZES:-32,64,128}" \
        --num-seeds "${NUM_SEEDS:-10}" \
        --epochs "${EPOCHS:-800}" \
        --output-dir "$OUT_DIR"
    exit 0
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
    --output-dir "$OUTPUT_ROOT/samplewise"

echo "Running featurewise experiment"
"$PYTHON" "$SCRIPT" \
    --num-workers "$NUM_WORKERS" \
    --output-dir "$OUTPUT_ROOT/featurewise"

echo "Completed both experiments."

