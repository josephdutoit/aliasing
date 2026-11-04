#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

profile=""
output_dir=""
resume_run=""
mode="dry-run"
collect_only=0
partition="${CH2V3_PARTITION:-}"
cpus="${CH2V3_CPUS:-}"
memory="${CH2V3_MEM:-}"
time_limit="${CH2V3_TIME:-}"
array_throttle="${CH2V3_ARRAY_THROTTLE:-}"
train_cpus="${CH2V3_TRAIN_CPUS:-}"
train_memory="${CH2V3_TRAIN_MEM:-}"
train_time="${CH2V3_TRAIN_TIME:-}"
python_bin="${CH2V3_PYTHON:-.venv/bin/python}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/chapter2_v3/launch_overnight.sh --profile smoke [--dry-run|--submit]
  bash scripts/chapter2_v3/launch_overnight.sh --profile overnight_discovery [--dry-run|--submit]
  bash scripts/chapter2_v3/launch_overnight.sh --profile confirmation [--dry-run|--submit]
  bash scripts/chapter2_v3/launch_overnight.sh --resume-run <run-id> [--dry-run|--submit]
  bash scripts/chapter2_v3/launch_overnight.sh --resume-run <run-id> --collect-only

The default is --dry-run. Before either dry-run or submit, set explicit Slurm
resources from the target cluster: CH2V3_PARTITION, CH2V3_CPUS, CH2V3_MEM,
CH2V3_TIME, and CH2V3_ARRAY_THROTTLE. Training resource variables may be set
separately with CH2V3_TRAIN_CPUS, CH2V3_TRAIN_MEM, CH2V3_TRAIN_TIME; if omitted,
the explicit CPU resource settings above are used. No GPU is requested.

Options:
  --profile NAME          Named immutable config profile.
  --output-dir PATH       Run directory (default under outputs/chapter2_v3).
  --resume-run RUN-ID     Resume an existing run under outputs/chapter2_v3 or an absolute path.
  --dry-run               Print exact sbatch commands; this is the default.
  --submit                Submit arrays and an afterany collector.
  --collect-only          Aggregate an existing run without submitting jobs.
  --partition NAME --cpus N --mem SIZE --time HH:MM:SS --array-throttle N
  --train-cpus N --train-mem SIZE --train-time HH:MM:SS
EOF
}

while (($#)); do
  case "$1" in
    --profile) profile="${2:?missing profile}"; shift 2 ;;
    --output-dir) output_dir="${2:?missing output directory}"; shift 2 ;;
    --resume-run) resume_run="${2:?missing run id}"; shift 2 ;;
    --dry-run) mode="dry-run"; shift ;;
    --submit) mode="submit"; shift ;;
    --collect-only) collect_only=1; shift ;;
    --partition) partition="${2:?missing partition}"; shift 2 ;;
    --cpus) cpus="${2:?missing CPUs}"; shift 2 ;;
    --mem) memory="${2:?missing memory}"; shift 2 ;;
    --time) time_limit="${2:?missing time limit}"; shift 2 ;;
    --array-throttle) array_throttle="${2:?missing array throttle}"; shift 2 ;;
    --train-cpus) train_cpus="${2:?missing training CPUs}"; shift 2 ;;
    --train-mem) train_memory="${2:?missing training memory}"; shift 2 ;;
    --train-time) train_time="${2:?missing training time limit}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -n "$resume_run" ]]; then
  if [[ "$resume_run" = /* ]]; then output_dir="$resume_run"; else output_dir="outputs/chapter2_v3/$resume_run"; fi
  [[ -n "$output_dir" ]] || { echo "--resume-run needs a path" >&2; exit 2; }
  [[ -f "$output_dir/plan.json" ]] || { echo "No plan at $output_dir/plan.json" >&2; exit 2; }
  profile="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["profile"])' "$output_dir/plan.json")"
elif [[ -z "$profile" ]]; then
  echo "--profile is required unless --resume-run is used" >&2
  usage >&2
  exit 2
fi

case "$profile" in smoke|overnight_discovery|confirmation) ;; *) echo "Unknown profile: $profile" >&2; exit 2 ;; esac
if [[ -z "$output_dir" ]]; then
  output_dir="outputs/chapter2_v3/${profile}"
fi
if [[ "$output_dir" != /* ]]; then output_dir="$ROOT/$output_dir"; fi
mkdir -p "$output_dir/logs"

if ((collect_only)); then
  exec "$python_bin" experiments/chapter2_v3/collect.py --run-dir "$output_dir"
fi

"$python_bin" experiments/chapter2_v3/plan.py --profile "$profile" --output-dir "$output_dir"

if [[ -z "$partition" || -z "$cpus" || -z "$memory" || -z "$time_limit" || -z "$array_throttle" ]]; then
  cat >&2 <<'EOF'
Resource settings are required even for a dry-run so the printed commands are exact.
Set CH2V3_PARTITION, CH2V3_CPUS, CH2V3_MEM, CH2V3_TIME, and CH2V3_ARRAY_THROTTLE
from observed cluster policy. The launcher does not guess resource limits.
EOF
  exit 2
fi
train_cpus="${train_cpus:-$cpus}"
train_memory="${train_memory:-$memory}"
train_time="${train_time:-$time_limit}"

indices_for() {
  local lanes="$1"
  "$python_bin" - "$output_dir/plan.json" "$output_dir" "$lanes" <<'PY'
import json, pathlib, sys
plan_path, run_dir, lanes_text = sys.argv[1:]
plan = json.loads(pathlib.Path(plan_path).read_text())
allowed = set(lanes_text.split(","))
indices = []
for job in plan["jobs"]:
    if job["lane"] not in allowed:
        continue
    status_path = pathlib.Path(run_dir) / "lanes" / job["lane"] / "shards" / job["job_id"] / "job_status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text()).get("status")
        if status in ("COMPLETED", "SKIPPED_DEPENDENCY"):
            continue
    indices.append(str(job["job_index"]))
print(",".join(indices))
PY
}

cpu_indices="$(indices_for A,B,D2,P)"
train_indices="$(indices_for C)"
d1_indices="$(indices_for D1)"
count_indices() { [[ -n "$1" ]] && awk -F, '{print NF}' <<<"$1" || echo 0; }
echo "Run directory: $output_dir"
echo "Profile: $profile; plan hash: $("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["plan_hash"])' "$output_dir/plan.json")"
echo "Planned indices: CPU=$(( $(count_indices "$cpu_indices") )) C-training=$(( $(count_indices "$train_indices") )) D1=$(( $(count_indices "$d1_indices") ))"
echo "Resources: partition=$partition CPU_ARRAY=${cpus}CPU/${memory}/${time_limit} throttle=$array_throttle; C_TRAIN=${train_cpus}CPU/${train_memory}/${train_time}; GPU=0"
echo "Threads per task: OMP=1 MKL=1 OpenBLAS=1 NumExpr=1"

if [[ "$mode" == "submit" ]]; then
  command -v sbatch >/dev/null || { echo "sbatch is unavailable" >&2; exit 2; }
  command -v squeue >/dev/null || { echo "squeue is unavailable; refusing duplicate-unsafe submission" >&2; exit 2; }
  job_name="ch2v3-$(basename "$output_dir")"
  active="$(squeue -h -n "$job_name" -o '%A' 2>/dev/null || true)"
  if [[ -n "$active" ]]; then
    echo "An active job with this immutable run name already exists: $active" >&2
    exit 2
  fi
  if [[ "$profile" == "smoke" ]]; then
    audit_status="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status","NOT_RUN"))' "$output_dir/AUDIT_PASS.json")"
    [[ "$audit_status" == "UNIT_TESTS_PASSED" || "$audit_status" == "PASSED" ]] || {
      echo "Smoke submit requires passing unit tests in AUDIT_PASS.json; got '$audit_status'" >&2; exit 2;
    }
  elif [[ "$profile" == "overnight_discovery" ]]; then
    smoke_status="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status","MISSING"))' "$ROOT/outputs/chapter2_v3/smoke/MANIFEST.json" 2>/dev/null || echo MISSING)"
    [[ "$smoke_status" == "COMPLETE" ]] || { echo "A completed smoke run is required before overnight_discovery" >&2; exit 2; }
    audit_status="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status","NOT_RUN"))' "$ROOT/outputs/chapter2_v3/smoke/AUDIT_PASS.json" 2>/dev/null || echo NOT_RUN)"
    [[ "$audit_status" == "PASSED" ]] || { echo "Smoke unit and bundle audit must say PASSED before discovery" >&2; exit 2; }
  elif [[ "$profile" == "confirmation" ]]; then
    discovery_status="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status","MISSING"))' "$ROOT/outputs/chapter2_v3/overnight_discovery/MANIFEST.json" 2>/dev/null || echo MISSING)"
    discovery_audit="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status","NOT_RUN"))' "$ROOT/outputs/chapter2_v3/overnight_discovery/AUDIT_PASS.json" 2>/dev/null || echo NOT_RUN)"
    [[ "$discovery_status" == "COMPLETE" && "$discovery_audit" == "PASSED" && "${CH2V3_CONFIRMATION_REVIEWED:-0}" == "1" ]] || {
      echo "Confirmation requires a completed discovery run, passing discovery audit, and CH2V3_CONFIRMATION_REVIEWED=1 after human review" >&2; exit 2;
    }
  fi
fi

render_command() { printf '  '; printf '%q ' "$@"; printf '\n'; }
submit_array() {
  local label="$1" script="$2" indices="$3" array_cpus="$4" array_mem="$5" array_time="$6" dependency="${7:-}"
  [[ -n "$indices" ]] || { echo "$label: no missing jobs"; return 0; }
  local -a cmd=(sbatch --parsable --job-name "$job_name" --partition "$partition"
    --cpus-per-task "$array_cpus" --mem "$array_mem" --time "$array_time"
    --array "${indices}%${array_throttle}" --signal B:USR1@120)
  [[ -n "$dependency" ]] && cmd+=(--dependency "$dependency")
  cmd+=(--output "$output_dir/logs/%x_%A_%a.out" --error "$output_dir/logs/%x_%A_%a.err"
    --export "ALL,CH2V3_RUN_DIR=$output_dir,CH2V3_PYTHON=$python_bin" "$script")
  if [[ "$mode" == "dry-run" ]]; then
    echo "$label ($(( $(count_indices "$indices") )) jobs):"
    render_command "${cmd[@]}"
    echo "  array indices: $indices"
  else
    local returned
    returned="$("${cmd[@]}")"
    local job_id="${returned%%;*}"
    echo "$label submitted: $job_id"
    printf '%s\n' "$job_id"
  fi
}

if [[ "$mode" == "dry-run" ]]; then job_name="ch2v3-$(basename "$output_dir")"; fi
cpu_id=""
train_id=""
d1_id=""
if [[ -n "$cpu_indices" ]]; then
  result="$(submit_array "CPU A/B/D2/P" "$ROOT/scripts/chapter2_v3/array_cpu.sbatch" "$cpu_indices" "$cpus" "$memory" "$time_limit")"
  echo "$result"
  [[ "$mode" == "submit" ]] && cpu_id="$(tail -n 1 <<<"$result")"
fi
if [[ -n "$train_indices" ]]; then
  result="$(submit_array "C training" "$ROOT/scripts/chapter2_v3/array_train.sbatch" "$train_indices" "$train_cpus" "$train_memory" "$train_time")"
  echo "$result"
  [[ "$mode" == "submit" ]] && train_id="$(tail -n 1 <<<"$result")"
fi
if [[ -n "$d1_indices" ]]; then
  dependency=""
  [[ -n "$train_id" ]] && dependency="afterany:$train_id"
  result="$(submit_array "D1 samplewise" "$ROOT/scripts/chapter2_v3/array_cpu.sbatch" "$d1_indices" "$cpus" "$memory" "$time_limit" "$dependency")"
  echo "$result"
  [[ "$mode" == "submit" ]] && d1_id="$(tail -n 1 <<<"$result")"
fi
if [[ "$mode" == "submit" ]]; then
  dependencies=()
  [[ -n "$cpu_id" ]] && dependencies+=("$cpu_id")
  [[ -n "$train_id" ]] && dependencies+=("$train_id")
  [[ -n "$d1_id" ]] && dependencies+=("$d1_id")
  if ((${#dependencies[@]})); then
    dependency="afterany:$(IFS=:; echo "${dependencies[*]}")"
    aggregate_id="$(sbatch --parsable --job-name "$job_name" --partition "$partition" \
      --cpus-per-task "$cpus" --mem "$memory" --time "$time_limit" \
      --dependency "$dependency" --output "$output_dir/logs/%x_%A.out" \
      --error "$output_dir/logs/%x_%A.err" \
      --export "ALL,CH2V3_RUN_DIR=$output_dir,CH2V3_PYTHON=$python_bin" \
      "$ROOT/scripts/chapter2_v3/aggregate.sbatch")"
    echo "Aggregate collector submitted: ${aggregate_id%%;*}; output: $output_dir/results_bundle.zip"
  else
    echo "No missing jobs; run --collect-only to refresh reports and the bundle."
  fi
else
  echo "Dry-run only. No Slurm jobs were submitted."
fi
