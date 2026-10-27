# Experiments 1–3 before the mathematical explanation

Experiment 15 resolves three gaps in experiment 14: population-reference and
readout stability, attribution to the encoder versus gated decoder, and the
difference between smaller task residuals and reduced amplification. It reuses
the original saved representations. It does not rerun or overwrite experiment
14. Training dynamics and boundary-specific classification are outside this run.

## Run on the existing node checkout

After merging the PR:

```bash
cd /home/jcdutoit/aliasing
git pull --ff-only
.venv/bin/python -m pip install -e .
sbatch mechanism_checks.sbatch
```

The launcher defaults to the full source run at
`outputs/learned_aliasing/14814813`, including its local `shards/` directory.
The uploaded review ZIP does not contain checkpoints and cannot replace that
directory. The original full output must remain on the node. No additional
dependency is required beyond NumPy and Matplotlib.

Results go to `outputs/gad_mechanisms/<job-id>/`. Upload the resulting
`results_bundle.zip` for review, or commit the summary CSVs and `RESULTS.md`.
The bundle contains the manifest, configuration, seed-level tables, pointwise
intervals, paired comparisons and four figures; checkpoint shards are excluded.

Use a different source/output or override the default research sizes:

```bash
GAD_SOURCE_DIR=outputs/learned_aliasing/14814813 sbatch mechanism_checks.sbatch
sbatch mechanism_checks.sbatch --active-counts 1,2,4,16 --num-seeds 24
sbatch mechanism_checks.sbatch --reference-sizes 4096,16384,65536,262144
```

The default research profile uses all 24 source seeds and all six active counts,
balanced task priorities, three readout draws, three rotations, eight probe
directions, 2,500 fresh decoder updates, 8,192 test examples, 2,048 validation
examples, reference sizes 4,096/16,384/65,536 and readout sizes
8/16/17/32/64/128/256/512/1,024/2,048/4,096. That is 144 source configurations.
All source and fresh maps use the same compression/expansion dimensions.
`--task-profiles balanced,A,B` adds task-priority source configurations.

For a quick check using the existing full source:

```bash
.venv/bin/python experiments/15_gad_mechanism_checks.py \
  --source-dir outputs/learned_aliasing/14814813 \
  --output-dir outputs/gad_mechanisms/smoke --profile smoke --workers 2
```

To create a completely independent local smoke source first:

```bash
.venv/bin/python experiments/14_learned_aliasing_hypotheses.py \
  --profile smoke --mode sparsity --task-profiles balanced \
  --output-dir /tmp/gad-mechanism-source --workers 2
.venv/bin/python experiments/15_gad_mechanism_checks.py \
  --source-dir /tmp/gad-mechanism-source \
  --output-dir /tmp/gad-mechanism-smoke --profile smoke --workers 2
```

## 1. Readout fitting and reference convergence

Original frozen maps include learned/fixed linear and ReLU representations and
the population linear oracle. Gate rotations and fresh decoder arms are also
evaluated. Each readout is refitted on independent data across the sample-size
grid. Ridge minimizes mean squared training loss plus a penalty on standardized
slopes. Its intercept is unpenalized. Training means/scales use only the head
training data; the positive penalty grid defaults to
`1e-6,1e-4,0.01,0.1,1,10`. Selection uses a separate validation split, never the
reference or test split. Readout and validation splits are shared across maps.

Native trained heads are compared with pseudoinverse heads fitted on exactly
the SAME representation-training examples. These fits are descriptive because
those examples also trained the representation. Independent 2,048-sample heads
provide a matched-sample-count comparison. Native heads have seen 2,048 examples,
so comparing them only with independent 128-sample heads is not a fair data-budget
comparison. Fresh and original native heads may have different total training
budgets; these are identified in the ablation design below.

The old sampler is not prefix stable: requesting a longer array changes its
amplitude RNG sequence. The implementation therefore recreates the original
readout/reference/test sample arrays at their ORIGINAL sizes, then appends
independent extension blocks. Readout draw zero preserves the old ordered
128-example readout. Later draws permute the larger readout pool. Reference
fits are nested, and their differences in predictions AND alias vectors are
reported against the largest reference fit at several readout sizes.
The largest fit remains an estimate, not exact population truth.

### Exact GAD for a changed estimator

For reference head beta, MP head beta_MP=H_T^+Y_T, and candidate head beta_c:

```text
prediction - target
 = H_P(H_T^+H_T-I) beta            [data insufficiency]
 + H_P H_T^+(Y_T-H_T beta)         [canonical MP aliasing]
 + (H_P beta-Y_P)                 [model insufficiency]
 + H_P(beta_c-beta_MP)            [explicit estimator correction].
```

All four energies and all six signed pairwise cross terms are saved. The alias
column is deliberately named `mp_aliasing`, including in ridge/native rows.
It is the MP BASELINE term, not the candidate estimator's residual leakage.
Ridge can cancel part of that baseline through the correction and cross terms.
Neither the correction energy nor the MP alias energy alone is ridge's error.
Pseudoinverse rows have zero correction and recover the canonical GAD-v2 terms.
Near singular readouts, ridge may cancel enormous MP baseline components.
Absolute, risk-relative and component-scale-normalized identity errors and a
cancellation ratio are all saved. Checks allow floating-point error relative
to the component scale; direct prediction MSE remains the primary risk value.

Inspect `references_*`, `readouts_*` and `readout_stability.png`. Convergence
supports using an accurate finite-reference comparator. Stabilization with
larger n or validation-selected ridge identifies readout instability; persistent
model insufficiency identifies a limitation of the frozen function space.

## 2. Encoder versus gated decoder

Three fresh arms train on the original representation-training split:

| Arm | Encoder | Decoder/head | Budget |
| --- | --- | --- | --- |
| `joint_fresh` | Original random initialization; trained | Fresh initialization; trained | 2,500 joint updates |
| `fixed_random_encoder` | Same random initialization; frozen | Exactly the same fresh initialization; trained | 2,500 decoder updates |
| `frozen_learned_encoder` | Original learned encoder; frozen | Same fresh decoder/head initialization; trained | Original encoder pretraining PLUS 2,500 decoder updates |

Joint-fresh and fixed-random arms have identical initial parameters, minibatch
sequences, objective weights and update counts; only encoder freezing differs.
All arms share initial decoder weights and task-head weights; the gate baseline
is calibrated separately to the same initial 30% activation rate for each
encoder. Frozen-encoder equality is checked exactly. Best checkpoints are
selected by the representation-training objective only.

Joint-fresh versus fixed-random is the primary matched-budget comparison. If
they perform similarly, a learned gated decoder may suffice for recovery. If
joint training consistently improves native AND stable independent readouts,
learning the compression adds value under this training setup. It does not
prove globally optimality of either optimizer. The pretrained-encoder arm is a
secondary diagnostic with extra pretraining, not a matched-total-compute claim.
Decoder/head checkpoints and objective histories are saved for inspection.

## 3. Controlled perturbation propagation

For each frozen map and readout sample size, define K=H_P H_T^+. Compare original
learned gates with the same orthogonal gate rotations as experiment 14, an
activity-matched rotation, gate removal, learned linear, fixed random ReLU, and
the fresh joint/fixed-random encoder arms. Pure rotations preserve weight
singular values and the biased preactivation Gram. Matching activity adjusts
bias, preserving only the weight spectrum. Linear rotation predictions must
remain invariant.

The measured probes are:

1. Exact isotropic unit-variance perturbation energy `||K||_F^2/n_test` and
   worst-case training-RMS-to-prediction-RMS gain
   `sqrt(n_train/n_test)*||K||_2`, computed with small matrix products.
2. Shared Gaussian target perturbations: identical directions across maps.
3. Shared nonlinear functions of latent squared amplitudes, with fixed random
   coefficients and common reference-based centering/scaling across maps.
4. The same nonlinear bank after subtracting EACH map's largest-reference
   projection. These probes approximate omitted-function residual directions;
   their directions differ across maps, which is explicitly part of this test.

Probe gain is `RMS(K delta)/RMS(delta)`; zero-energy probes are marked invalid
rather than assigned a zero gain. Exact-zero and linear-response controls check
that the measured prediction change equals K delta. Probes do not retrain the
representation. Gaussian corruption is fitted-noise sensitivity; shared
structured corruption is sensitivity to a specified contaminant; neither is
automatically called GAD aliasing. K is not the full omitted-feature operator
H_P H_T^+ U_T. Actual task GAD residuals are reported separately.

Lower exact operator sensitivity supports broad robustness to imposed training
perturbations. Lower task-residual gain without lower operator sensitivity
supports favorable residual alignment instead. Lower model insufficiency
without either reduction supports an approximation benefit. Mixed outcomes
are informative; no test is coded to require learned gates to win.

## Reproducibility, resumption and review

Source metadata, generator geometry, task weights and per-checkpoint SHA-256
hashes are checked. Code and source hashes are part of the resume identity.
Source checkpoints are never modified. Each world is written to an output
shard, then atomically marked complete. Failed worlds remain explicit and the
runner exits nonzero if any expected world is missing. BLAS threads are limited
before NumPy imports; process workers are capped to the SLURM allocation.

```bash
GAD_MECHANISM_DIR=outputs/gad_mechanisms/JOB_ID GAD_MECHANISM_RESUME=1 \
  sbatch mechanism_checks.sbatch
GAD_MECHANISM_DIR=outputs/gad_mechanisms/JOB_ID GAD_MECHANISM_RESUME=1 \
  GAD_MECHANISM_SUMMARIZE_ONLY=1 sbatch mechanism_checks.sbatch
.venv/bin/python experiments/plot_gad_mechanism_checks.py \
  --output-dir outputs/gad_mechanisms/JOB_ID
```

All draws, rotation IDs and probe directions are averaged WITHIN representation
seed before bootstrap intervals or paired comparisons. At the largest readout
size, permutations use the same full pool and do not add independent samples.
Independent seed count remains 24. `*_seed_means.csv` allows medians/outlier
inspection; mean intervals are pointwise exploratory intervals. Source paths
are local provenance. These runs establish neither MP edge calibration nor
boundary-specific classification claims.
