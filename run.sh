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

if [[ "$(basename "$SCRIPT")" == "07_static_geometry_factorial.py" ]]; then
    OUT_DIR="$OUTPUT_ROOT/static_geometry_factorial"
    mkdir -p "$OUT_DIR"
    cd "$REPO_DIR"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
    source "${REPO_DIR}/.venv/bin/activate"
    "$PYTHON" "$SCRIPT" \
        --dimension "${DIMENSION:-6}" \
        --feature-counts "${FEATURE_COUNTS:-6,8,10}" \
        --sample-sizes "${SAMPLE_SIZES:-8,16,24,32,48,64,96,128}" \
        --joint-probabilities "${JOINT_PROBABILITIES:-0,0.05,0.10,0.15,0.20}" \
        --tasks "${TASKS:-random,sum,pair_contrast,max_coherence_contrast,decoder_null}" \
        --families "${FAMILIES:-random,structured}" \
        --num-seeds "${NUM_SEEDS:-50}" \
        --output-dir "$OUT_DIR"
    exit 0
fi

if [[ "$(basename "$SCRIPT")" == "08_trained_superposition_aliasing.py" ]]; then
    OUT_DIR="$OUTPUT_ROOT/trained_superposition_aliasing"
    mkdir -p "$OUT_DIR"
    cd "$REPO_DIR"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
    source "${REPO_DIR}/.venv/bin/activate"
    "$PYTHON" "$SCRIPT" \
        --feature-counts "${FEATURE_COUNTS:-8,10,12}" \
        --bottlenecks "${BOTTLENECKS:-2,3,4}" \
        --sample-sizes "${SAMPLE_SIZES:-16,32,64,128,256}" \
        --joint-probabilities "${JOINT_PROBABILITIES:-0,0.05,0.10,0.15,0.20}" \
        --tasks "${TASKS:-sum,pair_contrast,single_feature,random}" \
        --encoder-samples "${ENCODER_SAMPLES:-4096}" \
        --num-seeds "${NUM_SEEDS:-20}" \
        --num-readout-draws "${READOUT_DRAWS:-5}" \
        --epochs "${EPOCHS:-1500}" \
        --output-dir "$OUT_DIR"
    exit 0
fi

if [[ "$(basename "$SCRIPT")" == "09_double_descent_decomposition.py" ]]; then
    OUT_DIR="$OUTPUT_ROOT/double_descent_decomposition"
    mkdir -p "$OUT_DIR"
    cd "$REPO_DIR"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
    source "${REPO_DIR}/.venv/bin/activate"
    "$PYTHON" "$SCRIPT" \
        --widths "${WIDTHS:-8,16,24,32,40,48,56,64,80,96,128}" \
        --sample-sizes "${SAMPLE_SIZES:-8,16,24,32,40,48,56,64,80,96,128,192,256}" \
        --input-dim "${INPUT_DIM:-20}" \
        --covariances "${COVARIANCES:-isotropic,spiked,low_rank}" \
        --tasks "${TASKS:-first_pc,random}" \
        --num-seeds "${NUM_SEEDS:-100}" \
        --n-test "${N_TEST:-20000}" \
        --output-dir "$OUT_DIR"
    exit 0
fi

if [[ "$(basename "$SCRIPT")" == "10_near_collision_noise.py" ]]; then
    OUT_DIR="$OUTPUT_ROOT/near_collision_noise"
    mkdir -p "$OUT_DIR"
    cd "$REPO_DIR"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
    source "${REPO_DIR}/.venv/bin/activate"
    "$PYTHON" "$SCRIPT" \
        --separations "${SEPARATIONS:-0,0.01,0.03,0.1,0.3,1,2}" \
        --noise-levels "${NOISE_LEVELS:-0.01,0.03,0.1,0.3,1}" \
        --sample-sizes "${SAMPLE_SIZES:-4,8,16,32,64,128,256}" \
        --priors "${PRIORS:-0.2,0.5,0.8}" \
        --num-seeds "${NUM_SEEDS:-100}" \
        --n-test "${N_TEST:-20000}" \
        --output-dir "$OUT_DIR"
    exit 0
fi

if [[ "$(basename "$SCRIPT")" == "11_collision_mass_law.py" ]]; then
    OUT_DIR="$OUTPUT_ROOT/collision_mass_law"
    mkdir -p "$OUT_DIR"
    cd "$REPO_DIR"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
    source "${REPO_DIR}/.venv/bin/activate"
    "$PYTHON" "$SCRIPT" \
        --collision-masses "${COLLISION_MASSES:-0.1,0.25,0.5,0.75,0.9}" \
        --conditional-masses "${CONDITIONAL_MASSES:-0.1,0.25,0.5,0.75,0.9}" \
        --separations "${SEPARATIONS:-0,1e-8,1e-5,1e-3,0.01,0.1,0.5}" \
        --tasks "${TASKS:-contrast,common}" \
        --sample-sizes "${SAMPLE_SIZES:-4,8,16,32,64,128}" \
        --num-seeds "${NUM_SEEDS:-100}" \
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

