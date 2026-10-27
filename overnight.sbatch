#!/usr/bin/env bash
#SBATCH --job-name=gad-overnight
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --signal=B:TERM@300
#SBATCH --output=gad-overnight-%j.log

set -euo pipefail

# Submit from the repository root; SLURM copies this file to its spool directory.
REPO_DIR="${GAD_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_DIR"
if [[ ! -f experiments/overnight_suite.py ]]; then
  echo "Submit from the repository root (or set GAD_REPO_DIR): $REPO_DIR" >&2
  exit 2
fi

PYTHON="$REPO_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing .venv/bin/python: install dependencies using OVERNIGHT_RUN.md" >&2
  exit 2
fi

export GAD_MAX_WORKERS="${GAD_MAX_WORKERS:-12}"
export GAD_EPOCHS="${GAD_EPOCHS:-200}"
export GAD_TRIALS="${GAD_TRIALS:-3}"
export GAD_GRID_STRIDE="${GAD_GRID_STRIDE:-4}"
export GAD_GLOBAL_SAMPLES="${GAD_GLOBAL_SAMPLES:-2000}"
export PYTHONUNBUFFERED=1

"$PYTHON" -c 'import aliasing, matplotlib, numpy, scipy, torch, tqdm'
SUITE_ARGS=(--max-hours "${GAD_MAX_HOURS:-11.5}" --step-minutes "${GAD_STEP_MINUTES:-25}")
if [[ -n "${GAD_RESUME_RUN_DIR:-}" ]]; then
  SUITE_ARGS+=(--run-dir "$GAD_RESUME_RUN_DIR" --resume)
else
  SUITE_ARGS+=(--output-root "${GAD_OUTPUT_ROOT:-outputs/overnight}")
fi
"$PYTHON" -u experiments/overnight_suite.py "${SUITE_ARGS[@]}" &
SUITE_PID=$!
trap 'kill -TERM "$SUITE_PID" 2>/dev/null || true; wait "$SUITE_PID" || true' TERM INT
wait "$SUITE_PID"
