# Broad survey of static learned features (experiment 17)

This survey extends experiment 16's plug-in models, sampler, float64 Adam,
canonical GAD-v2 evaluator, validation-selected ridge controls, independent
splits, saved checkpoints, and perturbation probes. It asks which geometric
changes explain trained representations' advantage over frozen random features.
No ordering of model insufficiency, aliasing, data insufficiency or total risk
is assumed. These are numerical hypotheses, not proved neural-network claims.

## Run on the supercomputer

From your repository checkout, after pulling this branch:

```bash
cd /home/jcdutoit/aliasing
git fetch origin
git switch chapter2-broad-feature-survey-20261101
git pull --ff-only
.venv/bin/python -m pip install -e .

# Optional: inspect the exact 13 blocks without training.
.venv/bin/python experiments/17_learned_feature_survey.py \
  --profile research --output-dir outputs/learned_feature_survey/preview --plan-only

# Submit all blocks as 13 array tasks, at most 3 concurrently.
sbatch learned_feature_survey.sbatch
```

The array requests 28 CPUs, 96 GB, and up to 24 hours PER task. These are
resource requests, not a measured duration. Adjust your account/partition,
concurrency, time, memory or CPUs to your allocation. Submit from the checkout:
the script resolves sources via SLURM_SUBMIT_DIR rather than the spool directory.
No GPU, PyTorch or new package dependency is required.

The submitted array ID is the common output directory:
`outputs/learned_feature_survey/ARRAY_ID/`. Each task owns one `jobs/JOB/`
directory and has a separate `gad-survey-ARRAY_ID_TASK.log`. The survey plan is
installed atomically; completed world artifacts are checksummed, and source
and scientific configuration must match on resume. A lock prevents simultaneous
writers of the same task.

Once EVERY array task has stopped, collect summaries and the review ZIP:

```bash
.venv/bin/python experiments/17_learned_feature_survey.py \
  --output-dir outputs/learned_feature_survey/ARRAY_ID --collect-only
```

Collection reads existing summaries and never starts training. It reports
PARTIAL and exits nonzero if a job/world is missing or failed. Upload
`results_bundle.zip`, plus any failed task logs. The ZIP contains the plan,
manifests, aggregate/paired tables, per-job seed summaries, reports, and figures.
It excludes raw shards and checkpoints. Keep the entire original directory on
the node: those checkpoints let us investigate mechanisms without retraining.
Collect again after any resumed tasks finish. Do not collect while tasks are
still updating their summaries.

## Resume failures or timeouts

For example, to retry only array indices 2 and 7 in the ORIGINAL output:

```bash
SURVEY_OUTPUT=outputs/learned_feature_survey/ARRAY_ID SURVEY_RESUME=1 \
  sbatch --array=2,7 learned_feature_survey.sbatch
```

Successful worlds are verified and skipped; failed/incomplete worlds are
retried. To rebuild a task's summaries from its available worlds, add
`SURVEY_SUMMARIZE_ONLY=1` with `SURVEY_RESUME=1`. Incomplete summaries remain
explicitly PARTIAL. Code or configuration changes require a new output directory.

## Smoke run

On an allocated CPU node, run the smaller ten-block end-to-end profile:

```bash
.venv/bin/python experiments/17_learned_feature_survey.py \
  --profile smoke --output-dir outputs/learned_feature_survey/smoke --workers 4
```

Or submit just the smoke baseline and task-subset blocks:

```bash
SURVEY_OUTPUT=outputs/learned_feature_survey/smoke_array \
  sbatch --array=1-3 learned_feature_survey.sbatch --profile smoke
```

That subset deliberately collects as PARTIAL. Smoke verifies execution and
identities, not scientific convergence. No full research survey is run by
installation or by --plan-only.

## Research design

All jobs use p=16 known latent factors, input dimension 32, and a standardized
full-rank linear task bank. Exactly k=1,2,4,8,12,16 factors are active with
uniform amplitudes on [0.5,1.5], scaled to E||x||²=16. Coactivation, means and
covariance change with k; these are not independently controlled factors.

Two architectures are evaluated at each width: the ordinary two-layer network
`V ReLU(Ux+b)+c`, and the supporting bottleneck model
`V ReLU(DEx+b)+c` with compression m=4. Each has a frozen random and trained
ReLU map, separately trained/random all-linear controls, gate removal, three
gate rotations and their activity-matched versions, and threshold offsets
-0.5, 0.5, 1, 2 per-unit training-preactivation standard deviations. Positive
offsets raise thresholds. Every intervention receives fresh readouts.

| Array index | Block | Difference from baseline |
| ---: | --- | --- |
| 0–4 | Widths 4,8,16,32,64 | Both architectures use the specified final width |
| 2 | Baseline width 16 | Also runs balanced/A/B training priorities on identical tasks |
| 5 | First-task training | Only task 0 enters representation-training prediction loss |
| 6 | Half-task training | Only the first eight tasks enter that loss |
| 7–8 | Representation data | 512 or 8192 examples instead of 2048 |
| 9 | Longer optimization budget | 7500 updates instead of 2500; final selected map only |
| 10–11 | Weight decay | 0 or 1e-3 instead of 1e-5 |
| 12 | Task basis | Identity queries instead of orthogonally mixed queries |

The full plan has 2160 seed/k/profile worlds across 13 jobs. Each uses 24
independent seeds; Adam batch size 128, learning rate .005; reference sizes
16,384 and 65,536; validation 2048; test 8192; three readout orders; eight
probe directions. Readout sizes include 3,4,5,8,9,15,16,17,18,31,32,33,
63,64,65,66,128,256,512,1024,2048,4096. Actual ranks, not width alone,
determine interpolation. Repeated orders use the same readout pool.

Task-subset training still evaluates ALL original tasks with independently
refitted heads; this tests both specialization and retained information.
Per-task GAD at n=128 and 4096 uses the same global reference head. A/B
priority comparisons keep the actual task fixed, including random negative
controls. Unweighted GAD describes the uniform all-task average; weighted
test risk describes the selected importance objective. These must not be
interchanged. The oracles use the evaluation weights and known population
information; they are privileged controls, not learned competitors.

Widths, other blocks and priorities share problem and initialization seed
rules. Changing width does NOT produce nested random hidden units. Changing
representation sample count does NOT produce nested datasets with the current
sampler. Those budget comparisons are paired by generated problem/seed only.
Representations see extra training labels compared with frozen random maps;
this is a learned-representation comparison, not an equal-total-label claim.
Models match data and updates, not parameter count or optimization difficulty.
All survey blocks share a separate unlabeled initialization-calibration split
(2048 examples in research, 128 in smoke), so random hidden maps stay identical
when only the labeled representation-data budget changes. This extra common
unlabeled exposure is distinct from the listed labeled training budget.
Different task bases also change standardized task coordinates, so that block
is a robustness check rather than an identical-target intervention.

## Outputs and interpretation

- `survey_paired.csv`: combined within-job learned-minus-random and intervention
  contrasts, including all six signed GAD cross terms and seed bootstrap CIs.
- `block_paired.csv`: seed-paired job-minus-baseline effects for width, task
  exposure and budgets; task-basis changes are explicitly flagged.
- Each job's `readouts_*`, `geometry_*`, `references_*`, `probes_*`, and
  `feature_recovery_*`: existing benchmark outputs with sample standard
  deviations, valid seed counts, risk/alias medians, maxima, and top-five shares.
- `task_readouts_*`: per-original-task GAD, training task weight, and all cross
  terms. `*_profile_paired.csv` compares B-minus-A on the same task/group.
- `alias_modes_*`: SVD-mode residual projections, training singular values,
  population visibility, diagonal energies, and the signed OFF-diagonal sum.
  Sum diagonal mode energies PLUS that signed sum to reconstruct total alias
  energy. Do not pair mode indices semantically between different maps.
- Geometry additionally reports overlap of local hidden responses to known
  latent-coordinate perturbations, using finite differences on 128 evaluation
  contexts. Perturbations may leave the support at zero coordinates: this is
  a local Jacobian diagnostic, not a collision, injectivity, or superposition
  certificate. No checkpoint or intervention is selected using this metric.

Ridge/native rows retain canonical MP aliasing plus their explicit estimator
correction and all six cross terms. Probe sensitivities are not automatically
GAD aliasing. Finite references remain estimated population projections; retain
their signed cross terms and inspect convergence of both predictions and alias
vectors. Positive energies do not form an additive risk partition.

Use the survey to select mechanisms, then confirm selected comparisons with
fresh seeds. All bootstrap intervals are pointwise exploratory intervals and
resample independent seeds after averaging draws/rotations/probes within seed.
No global-optimality, training-dynamics, spectral-universality, or guaranteed
superposition-to-aliasing claim follows from this survey.
