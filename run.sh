#!/usr/bin/env bash
#SBATCH --job-name=aliasing
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=1024M

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR" ]]; then
    # Slurm may execute a staged copy of this script from /var/spool. Resolve
    # repository files from the directory where the job was submitted instead.
    REPO_DIR="$(cd -- "$SLURM_SUBMIT_DIR" && pwd)"
else
    REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi

usage() {
    cat <<'EOF'
Run one experiment from the Aliasing repository:

  ./run.sh 01 [experiment options]
  ./run.sh 02 featurewise [experiment options]
  ./run.sh 02 samplewise [experiment options]
  ./run.sh 03 [experiment options]
  ./run.sh 04 [experiment options]
  ./run.sh 05 [experiment options]

Experiment aliases:
  01, linear       Fixed-feature linear baseline
  02, random       Frozen random-feature sweep (requires featurewise or samplewise)
  03, mlp          Feature-learning MLP
  04, gad            Finite-dimensional GAD geometry pilot
  05, geometry-risk  Replicated feature-geometry and excess-risk study

Examples:
  ./run.sh 01 --n-train 64 --n-full 128
  ./run.sh 02 featurewise --num-seeds 8 --num-workers 4
  ./run.sh 02 samplewise --sample-sizes 8,16,32,64 --num-seeds 8
  ./run.sh 03 --widths 16,32,64 --num-seeds 4 --steps 1000
  ./run.sh 04 --replicates 4 --seed 7
  ./run.sh 05 --replicates 100 --n-test 4096

The runner supplies a default output directory for each run. Pass
--output-dir PATH to choose another one. The experiment's own options are
passed through unchanged. Set PYTHON=/path/to/python to choose an interpreter;
otherwise .venv/bin/python3 is used when available, then python3.

On a Slurm cluster, submit the same commands with sbatch, for example:
  sbatch run.sh 02 featurewise --num-seeds 100
EOF
}

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" || "$1" == "help" ]]; then
    usage
    exit 0
fi

selector="$1"
shift
EXPERIMENT_ARGS=("$@")

case "$selector" in
    01|linear|linear-baseline)
        SCRIPT="experiments/01_linear_aliasing_double_descent.py"
        DEFAULT_OUTPUT="outputs/linear_baseline"
        SUPPORTS_WORKERS=0
        ;;
    02|random|random-features|random-feature)
        SCRIPT="experiments/02_random_feature_aliasing.py"
        SUPPORTS_WORKERS=1
        sweep=""
        if [[ "${1:-}" == "featurewise" || "${1:-}" == "samplewise" ]]; then
            sweep="$1"
            shift
            EXPERIMENT_ARGS=("--sweep" "$sweep" "$@")
        else
            for ((i = 0; i < ${#EXPERIMENT_ARGS[@]}; i++)); do
                if [[ "${EXPERIMENT_ARGS[$i]}" == "--sweep" ]]; then
                    sweep="${EXPERIMENT_ARGS[$((i + 1))]:-}"
                    break
                elif [[ "${EXPERIMENT_ARGS[$i]}" == --sweep=* ]]; then
                    sweep="${EXPERIMENT_ARGS[$i]#--sweep=}"
                    break
                fi
            done
        fi
        if [[ "$sweep" != "featurewise" && "$sweep" != "samplewise" ]]; then
            echo "Experiment 02 requires 'featurewise' or 'samplewise'." >&2
            usage >&2
            exit 2
        fi
        DEFAULT_OUTPUT="outputs/random_feature_${sweep}"
        ;;
    03|mlp|feature-learning|feature-learning-mlp)
        SCRIPT="experiments/03_feature_learning_mlp.py"
        DEFAULT_OUTPUT="outputs/feature_learning_mlp"
        SUPPORTS_WORKERS=1
        EXPERIMENT_ARGS=("$@")
        ;;
    04|gad|gad-pilot|geometry)
        SCRIPT="experiments/04_gad_geometry_pilot.py"
        DEFAULT_OUTPUT="outputs/gad_geometry_pilot"
        SUPPORTS_WORKERS=0
        EXPERIMENT_ARGS=("$@")
        ;;
    05|geometry-risk|feature-risk|gad-risk)
        SCRIPT="experiments/05_feature_geometry_risk.py"
        DEFAULT_OUTPUT="outputs/chapter2_geometry_risk"
        SUPPORTS_WORKERS=0
        EXPERIMENT_ARGS=("$@")
        ;;
    *)
        echo "Unknown experiment: $selector" >&2
        usage >&2
        exit 2
        ;;
esac

has_option() {
    local option="$1"
    shift
    local arg
    for arg in "$@"; do
        if [[ "$arg" == "$option" || "$arg" == "$option="* ]]; then
            return 0
        fi
    done
    return 1
}

if ! has_option --output-dir "${EXPERIMENT_ARGS[@]}"; then
    EXPERIMENT_ARGS+=(--output-dir "$DEFAULT_OUTPUT")
fi

# Respect the scheduler allocation for multiprocessing experiments. Local runs
# use each experiment's own worker default unless --num-workers is supplied.
if [[ "$SUPPORTS_WORKERS" -eq 1 && -n "${SLURM_CPUS_PER_TASK:-}" ]] \
    && ! has_option --num-workers "${EXPERIMENT_ARGS[@]}"; then
    EXPERIMENT_ARGS+=(--num-workers "$SLURM_CPUS_PER_TASK")
fi

cd "$REPO_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${UID}}"
mkdir -p "$MPLCONFIGDIR"

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "$REPO_DIR/.venv/bin/python3" ]]; then
    PYTHON_BIN="$REPO_DIR/.venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "Python was not found. Set PYTHON or create .venv/bin/python3." >&2
    exit 1
fi

if [[ ! -f "$SCRIPT" ]]; then
    echo "Experiment script not found: $REPO_DIR/$SCRIPT" >&2
    exit 1
fi

echo "Running $SCRIPT"
printf 'Command:'
printf ' %q' "$PYTHON_BIN" "$SCRIPT" "${EXPERIMENT_ARGS[@]}"
printf '\n'
exec "$PYTHON_BIN" "$SCRIPT" "${EXPERIMENT_ARGS[@]}"
