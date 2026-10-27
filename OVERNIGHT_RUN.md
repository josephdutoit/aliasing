# One-node overnight experiment run

The [suite](experiments/overnight_suite.py) discovers numbered `experiments/`
scripts (including 12 and 13), runs the ten registered `generalized_gad`
experiments, and attempts the three `ggad_old` GLM prototypes. Other unnumbered
one-off files are not included. Every experiment gets a distinct directory and
log. The runner continues after failures, caps each step at 25 minutes, and
keeps partial output when a step times out. The job has a 12-hour SLURM limit
and stops starting new steps after 11.5 hours, leaving room to write its report.

## Submit from the repository checkout on the compute cluster

```bash
cd /home/jcdutoit/aliasing
git pull --ff-only
test -x .venv/bin/python || python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install -r requirements-neural.txt scipy tqdm
.venv/bin/python -c 'import aliasing, matplotlib, numpy, scipy, torch, tqdm'
mkdir -p outputs/overnight
.venv/bin/python experiments/overnight_suite.py --dry-run
sbatch overnight.sbatch
```

The model probe in experiment 13 needs a corpus **and** the locally cached OPT
checkpoint, or a saved activation NPZ. If available, supply one before `sbatch`:

```bash
# Option A: six or more independent documents separated by blank lines.
sbatch --export=ALL,GAD_TEXT_FILE=/absolute/path/corpus.txt,GAD_MODEL_ID=/absolute/path/cached/opt-125m overnight.sbatch

# Option B: arrays in an NPZ (train/ref/test_X and train/ref/test_y).
sbatch --export=ALL,GAD_FEATURES_NPZ=/absolute/path/features.npz overnight.sbatch
```

For option A, install `transformers` into `.venv` in advance; the job uses
offline model loading. The default device is CPU because this job does not
request a GPU. If neither input is supplied, 13 is reported as `skipped` with
the reason; every other discovered experiment is attempted.

## Review in the morning

```bash
squeue -u "$USER"
ls -td outputs/overnight/* | head -1
# Substitute the newest suite path shown by the previous command:
cat outputs/overnight/<suite-id>/SUMMARY.md
```

`SUMMARY.md` lists completed, timed out, failed, skipped and not-run steps.
`GALLERY.md` embeds every generated PNG. `results.json` contains the exact
commands, script SHA256s, parameters, elapsed times and artifact paths. Each
step contains its own `stdout.log`; numbered scripts write their CSVs and plots
there too. Legacy `generalized_gad` scripts put a nested manifest and plots
inside the step directory. Nothing in `outputs/overnight` is committed by
default. Large outputs can stay on the cluster until you choose which figures
or summaries to upload.

Resume unfinished steps after a preemption or timeout, retaining old attempts:

```bash
cd /home/jcdutoit/aliasing
.venv/bin/python experiments/overnight_suite.py \
  --run-dir outputs/overnight/<suite-id> --resume --max-hours 11.5
```

The overnight profile is a **survey**, not the published/full sweeps. It runs
fewer seeds, epochs and grid points in numbered scripts. For the older scripts
it sets `GAD_MAX_WORKERS=12`, `GAD_EPOCHS=200`, `GAD_TRIALS=3`,
`GAD_GRID_STRIDE=4` and `GAD_GLOBAL_SAMPLES=2000`. Unset those variables when
repeating a promising finding at the original fidelity. The older plots are
still pseudoinverse diagnostics: their titles do not establish held-out aliasing
mitigation or double descent.

For a larger node allocation, adjust the `#SBATCH` limits and the per-step
`GAD_STEP_MINUTES` and `GAD_MAX_HOURS`. You can preview a subset with
`--only 04,12,embedded-complexity --dry-run`.
