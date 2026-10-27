# Learned sparse compression: controlled numerical hypotheses

Experiment 14 tests the proposed mechanisms using the corrected GAD-v2
definitions. It does not assume that learning, ReLU, or sparsity must reduce
aliasing. Decreasing test error solely through model insufficiency is a possible
outcome, not confirmation of aliasing mitigation.

## What each experiment asks

| Hypothesis | Controlled comparison | Evidence needed / informative failure |
| --- | --- | --- |
| H1: learned nonlinear compression reduces harmful interference | Exact active count k varies at fixed possible-feature count, compression m, expansion h, training size and task bank. Compare fixed/learned linear and fixed/learned ReLU, plus a population-optimal rank-m linear oracle. | Compare test error AND all GAD components. Lower test error with unchanged or higher aliasing supports a representation benefit, not de-aliasing. Compare residual-normalized gain to distinguish smaller residuals from less amplification. |
| H2: gate orientation matters beyond weight spectra | Rotate the coordinates of the learned pre-ReLU expansion, refit every head, and retain the original weight singular spectrum and preactivation sample Gram. | Pure rotations have invariant checks. A linear counterfactual must retain its predictions and GAD components. An additional activity-matched rotation checks whether overall activation density explains the effect; its bias adjustment breaks the biased-Gram invariant, which is explicitly flagged. |
| H3: learned thresholds filter interference while retaining information | Freeze learned weights, shift gate biases in units of representation-training preactivation standard deviation, and refit heads. | Inspect the joint tradeoff between aliasing, model insufficiency, test error, rank, latent recovery and activation rate. A nearly constant map with low aliasing is not useful filtering. |
| H4: compression is task-selective | Train on the SAME inputs and standardized task bank with profiles A and B swapping which half of tasks receives greater loss weight. | Compare the SAME task group across A/B using paired seed differences. Fixed random maps should not change under this swap. Near-boundary classification and score error are recorded separately. |

## Data and architecture

There are p possible latent features and exactly k active features in each
example. Nonzero amplitudes are independently uniform on [0.5,1.5]. Their scale
makes the analytic expected input squared norm equal p for every k. A seeded
orthogonal embedding hides the coordinate features in an observed input space.
Changing k changes co-activation, input means/covariance and margin statistics;
those distributional changes are part of the test and are not claimed to be
independently fixed. Known covariance and feature means are saved.

The teacher is a fixed full-rank bank of p linear queries of the latent features.
Each query is centered and normalized to analytic population variance one for
each k. The default bank is a seeded orthogonal mixing; `--task-basis identity`
instead queries each feature directly. The score bank is invertible, so all
queries together require preserving information about all latent features.
This avoids a single-label shortcut and the previous identically-zero contrast
task. Analytic unit variance does not imply equal classification difficulty
across k: positive-class fractions, conditional margins and target moments are
reported. The teacher and embedding are shared across k within each seed.

Every fitted model has the same architecture and parameter dimensions:

    observed input -> linear compression (m) -> linear expansion (h)
                   -> identity or ReLU -> p task scores.

All variants share m and h. The compression dimension is m, not the expanded
GAD readout dimension. The latter is h+1 including an always-present intercept;
its measured rank can differ between identity and ReLU. We expand after the
bottleneck because a fixed linear output from m hidden coordinates would itself
limit all task predictions to a rank-m subspace, obscuring nonlinear sparse
recovery. The learned linear and learned ReLU models use matched Adam updates,
batch sequences, objective weights and L2 decay. Best checkpoints are selected
only by the representation-training objective. No test or reference samples
are used for representation training or threshold calibration.

The linear oracle uses known synthetic covariance/teacher information to find
the optimal weighted rank-m linear predictor. It is explicitly privileged; it
guards against mistaking a poorly optimized linear model for a nonlinear
advantage. Its population weighted-risk floor is in geometry.csv.
All finite-sample heads, including the oracle representation's heads, are
refitted on the same readout data; the oracle's test MSE is not the population
floor itself. Unweighted
all-task MSE and weighted test/model error are reported separately: compare the
weighted floor to the weighted objective, not a differently weighted metric.

Balanced training weights all tasks equally. A uses weights 1 and 0.25 on the
first and second halves; B swaps them. All tasks retain positive weight.
Encoder feature-direction norms/coherence and normalized per-feature latent
recovery are measured using the known generative features. The count of latent
features below normalized MSE 0.1 is a descriptive threshold, not an injectivity
certificate. Interpret task-bank results and feature recovery together before
claiming learned superposition. Compression alone is not such a certificate.
Encoder coherence also depends on the chosen bottleneck coordinates; it is not
an invariant proof of the mechanism. Hidden scale and reference conditioning
are saved alongside it.

## Correct GAD and evaluation splits

Representation training, readout fitting, reference projection and evaluation
use four independent sample sets. All methods/interventions use identical
readout indices and nested training sizes. The representation is frozen before
fitting each minimum-norm readout.

For a reference head beta and training/prediction matrices H_T,H_P, with
r_T = Y_T-H_T beta, r_P=Y_P-H_P beta, and B=H_T^+ H_T:

    prediction - target = H_P(B-I)beta + H_P H_T^+ r_T - r_P
                          data insuff.     aliasing       model insuff.

All three energies and signed cross terms are saved. Multi-output metrics are
averages over task scores; per-task rows are saved at the largest readout size.
The reference head is estimated on independent finite reference data. Its
model-insufficiency term is an estimated comparator, not an oracle population
projection. Increase `--n-reference` to check sensitivity before a strong claim.
No separate label noise is added; the residual is model/representation mismatch.

`alias_residual_gain` is alias RMS divided by the training residual RMS for
the specified task group. It is target-dependent and is not the full aliasing
operator norm. Negligible denominators produce blank values and a valid-seed
count rather than an artificial zero (cutoff: training-residual MSE <= 1e-20).
Classification is the sign error of a
continuous teacher score. Near/far regions use the TRUE standardized score,
not each model's predicted margin. Conditional GAD energies retain the global
reference head and all cross terms; they are not new conditional-optimal
projections. Zero-margin ties are excluded from classification counts.

## Run on the cluster

After merging this branch, from the repository checkout:

```bash
cd /home/jcdutoit/aliasing
git pull --ff-only
.venv/bin/python -m pip install -e .
sbatch learned_aliasing.sbatch
```

The new module needs only the existing NumPy and Matplotlib dependencies.
The SLURM job requests one node, 28 CPUs, 64 GB and 12 hours. It trains the
representations once per seed/k/task-profile combination and reuses them for
all frozen-map interventions. It writes `outputs/learned_aliasing/<job-id>/`.
This dedicated job runs the research profile. The older overnight suite may
discover experiment 14 with its default smoke configuration; use the dedicated
launcher for the intended study.

The research profile has p=16, observed dimension 32, compression m=4 and
expansion h=16; k=1,2,4,8,12,16; 24 independent seeds; balanced/A/B objectives;
2,500 Adam updates; 2,048 representation-training points; 4,096 reference and
4,096 test points; 3 readout draws; and 3 rotations per seed. Readout counts
4,8,16,17,24,32,64,128 cross both linear and expanded readout ranks. The
threshold offsets are -1,0,0.5,1,2,4 standard deviations; zero reproduces the
learned threshold. The interpolation point is determined by measured rank,
not asserted to be the compression dimension.

For a quick installation check on an allocated node:

```bash
.venv/bin/python experiments/14_learned_aliasing_hypotheses.py \
  --profile smoke --workers 2 --output-dir outputs/learned_aliasing/smoke
```

Additional controlled runs:

```bash
# Feature-query teacher instead of the orthogonal task bank:
sbatch --export=ALL,HYP_TASK_BASIS=identity learned_aliasing.sbatch

# Isolate one intervention family, or extend the training budget:
sbatch --export=ALL,HYP_MODE=rotation learned_aliasing.sbatch
sbatch learned_aliasing.sbatch --steps 5000 --num-seeds 32
```

Use a new directory for changed code or settings. Resume the exact same code
and configuration in a new compute allocation:

```bash
export HYP_RUN_DIR=outputs/learned_aliasing/<job-id>
export HYP_RESUME=1
sbatch learned_aliasing.sbatch
```

Completed configurations are atomic shards and are not retrained. If a job is
interrupted, use the same settings with `HYP_SUMMARIZE_ONLY=1` and `HYP_RESUME=1`
to summarize completed shards without training the remaining configurations.
The summary exit code remains nonzero when configurations are missing. Resume
requires an identical configuration/code hash; preserve overrides such as the
task basis or update count when resubmitting.

## Results to inspect and upload

- `RESULTS.md`: figure gallery, selected results, hypothesis reading guide.
- `sparsity_gad.png`: fixed-capacity active-count comparison and GAD terms.
- `residual_amplification.png`: residual size versus realized amplification.
- `rotation_intervention.png`: original/rotated/activity-matched gates and
  the linear invariant control after refitting every readout.
- `threshold_intervention.png`: useful filtering versus information loss.
- `task_priority_swap.png`: same task groups under swapped objectives.
- `decision_margin.png`: near/far classification and score-risk comparison.
- `paired_contrasts.csv`: candidate-minus-learned-ReLU seed differences;
  linear rotation controls instead use the gate-removed linear reference.
- `paired_profile_contrasts.csv`: B-minus-A differences for fixed task groups,
  including fixed random negative controls and the population linear oracle.
- `geometry.csv`, `training_history.csv`, `configuration.json`, `progress.json`,
  and `run_manifest.json`: geometry, optimization and provenance checks.
- `raw.csv`, `task_raw.csv`, `seed_means.csv`, `mean.csv`, `std.csv`, `ci95.csv`:
  individual measurements, independent-seed summaries and pointwise intervals.
- `shards/*/checkpoints.npz`: static maps and teacher/embedding/covariance arrays.
- `results_bundle.zip`: upload this for review. It omits raw/task-level CSVs and
  shards, retaining summaries, paired contrasts, geometry, history and figures.

Readout draws and rotations are averaged WITHIN each representation seed before
forming confidence intervals. Neither is counted as an independent seed.
Intervals are left blank when fewer than two independent seeds have a valid
metric or paired difference, rather than displaying a degenerate interval.
Intervals are pointwise, not simultaneous or corrected for choosing a crossover
after viewing the grid. Missing metric values have per-metric valid-seed counts.
Failed configurations are recorded rather than silently dropped; inspect the
manifest before interpreting a partial summary. The full study has not been
run merely because the smoke tests pass.

## Numerical controls

Tests cover analytic gradients, exact-k moments, task normalization, the
population linear floor, agreement with scalar GAD-v2 in both rank regimes,
conditional risk identities, gate rotation invariants, refitted linear
predictions, training-objective reduction, and seed rather than draw counting.
The suite asserts preactivation Gram and weight-spectrum preservation for pure
rotations, linear prediction invariance, and GAD reconstruction on each fit.
No theorem about MP spectra, HTSR, or an inevitable nonlinear advantage is
assumed. Null results and tradeoffs are legitimate outcomes of these tests.
