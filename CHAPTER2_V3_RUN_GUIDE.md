# Chapter 2 v3 run guide

The v3 code is isolated from the legacy GAD-v2 experiments. Install the project on the target cluster, create an immutable smoke plan, run the math and compatibility audit, and only then submit the smoke array. The assistant did not run these commands.

```bash
cd /home/jcdutoit/aliasing
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pip install -r requirements-neural.txt
.venv/bin/python experiments/chapter2_v3/plan.py --profile smoke --output-dir outputs/chapter2_v3/smoke
.venv/bin/python experiments/chapter2_v3/audit.py --run-dir outputs/chapter2_v3/smoke
```

Select a partition, CPU count, memory, wall time, and array throttle from the actual cluster policy and measured usage. The launcher requires explicit values even for a dry run and requests no GPU by default. For example, after replacing each angle-bracket value with a cluster-approved value:

```bash
export CH2V3_PARTITION='<observed-partition>'
export CH2V3_CPUS='<observed-cpus-per-task>'
export CH2V3_MEM='<observed-memory-limit>'
export CH2V3_TIME='<observed-walltime>'
export CH2V3_ARRAY_THROTTLE='<approved-concurrency>'
export CH2V3_TRAIN_CPUS="$CH2V3_CPUS"
export CH2V3_TRAIN_MEM="$CH2V3_MEM"
export CH2V3_TRAIN_TIME="$CH2V3_TIME"

bash scripts/chapter2_v3/launch_overnight.sh --profile smoke --dry-run
bash scripts/chapter2_v3/launch_overnight.sh --profile smoke --submit
squeue -u "$USER"
bash scripts/chapter2_v3/launch_overnight.sh --resume-run smoke --collect-only
.venv/bin/python experiments/chapter2_v3/audit.py --run-dir outputs/chapter2_v3/smoke --check-smoke
```

The default is `--dry-run`; it writes or reuses the immutable plan and prints the exact arrays, global job indices, dependencies, resource requests, output directory, and thread limits. `--submit` checks `squeue` for an active job with the same run name and refuses duplicate-unsafe submission if the command is unavailable. It requires `AUDIT_PASS.json` to say `PASSED`. A smoke submission runs A/B/D2/P in the CPU array, C in its independent training array, and D1 after C finishes; collection uses `afterany` so partial reports and a ZIP are still written after failures.

Lane D2 keeps the same dense or sparse raw teacher across independently trained widths, using paired downstream samples and noise. It uses separate nested-bank and width-matched random controls. Lane P, if an activation cache is supplied, reports its finite-reference oracle size, held-out residual-feature cross term, residual-moment uncertainty, and a smaller-reference-size sensitivity alongside estimated GAD components.

After smoke completes, inspect `MANIFEST.json`, `AUDIT_PASS.json`, `MORNING_README.md`, `DECISION_MEMO.md`, and `results_bundle.zip`. A discovery submission requires a `COMPLETE` smoke manifest. Then use:

```bash
bash scripts/chapter2_v3/launch_overnight.sh --profile overnight_discovery --dry-run
bash scripts/chapter2_v3/launch_overnight.sh --profile overnight_discovery --submit
squeue -u "$USER"
bash scripts/chapter2_v3/launch_overnight.sh --resume-run overnight_discovery --collect-only
.venv/bin/python experiments/chapter2_v3/audit.py --run-dir outputs/chapter2_v3/overnight_discovery --check-smoke
```

Confirmation uses fresh representation seeds and requires a complete discovery manifest plus an explicit review marker after inspecting those results:

```bash
export CH2V3_CONFIRMATION_REVIEWED=1
bash scripts/chapter2_v3/launch_overnight.sh --profile confirmation --dry-run
bash scripts/chapter2_v3/launch_overnight.sh --profile confirmation --submit
squeue -u "$USER"
bash scripts/chapter2_v3/launch_overnight.sh --resume-run confirmation --collect-only
```

Resume only an existing, unchanged run. The planner compares configuration and code hashes with the immutable `plan.json`; completed shards are skipped and missing or failed shards are selected for resubmission.

```bash
bash scripts/chapter2_v3/launch_overnight.sh --resume-run overnight_discovery --dry-run
bash scripts/chapter2_v3/launch_overnight.sh --resume-run overnight_discovery --submit
```

Each Slurm task writes an atomic attempt shard under `outputs/chapter2_v3/<run-id>/lanes/<lane>/shards/`. Training checkpoints are periodic and resumable. Run collection without launching jobs with `--collect-only`; it writes lane tables, seed-level paired contrasts, figure data, figures, reports, and `results_bundle.zip`. A partial collection still writes the bundle and exits with status 2. Inspect `MANIFEST.json` and `failed_jobs/` for missing tasks and error logs. The optional P lane is `SKIPPED_DEPENDENCY` when `activation_cache` is null; it performs no network downloads.

The environment used to prepare this repository had no available Slurm CLI, so no partition, resource limit, runtime, GPU, or scratch path is asserted here. Choose those values on the supercomputer before submitting.
