# Learned ReLU geometry: overnight experiments for proof discovery

These seven studies ask what JOINT TRAINING selects, then evaluate canonical
GAD on independent readout samples. They do not repeat a random-feature study
or assume that learned ReLUs suppress aliasing. All networks use NumPy CPU
float64 and the existing package dependencies. No GPU or SciPy is required.

## Run

From the repository checkout with the existing `.venv`:

```bash
# First: short execution/identity check of all seven families.
.venv/bin/python experiments/17_learned_relu_geometry.py \
  --profile smoke --workers 4 --output-dir outputs/relu_geometry_smoke

# Recommended with several nodes available: seven parallel jobs.
sbatch --array=0-6 learned_relu_geometry.sbatch

# Alternatively, all seven families in one allocation.
sbatch learned_relu_geometry.sbatch
```

The launcher requests 28 CPUs and 96 GB per job for 24 hours. These are
allocation limits, not measured completion-time guarantees. Override them
using your site's normal sbatch options. For example `--array=0-6%3` permits
three study jobs at once. The large profiles have NOT been run by merely
passing the smoke check.

The default overnight plan contains **1,672 worlds**, eight seeds and four
initializations per trained arm. Each world trains separate ReLU and linear
models; ablations additionally train fixed-encoder and fixed-threshold arms.
Every start is saved. `--list-plan` prints exact counts without running:

```bash
.venv/bin/python experiments/17_learned_relu_geometry.py \
  --profile overnight --list-plan

# A single study, useful for queueing or changing its budget independently.
GAD_GEOMETRY_EXPERIMENTS=phase sbatch learned_relu_geometry.sbatch

# Larger optimization/replicate budget, same scenario grid.
GAD_GEOMETRY_PROFILE=extended sbatch --array=0-6 learned_relu_geometry.sbatch

# Or explicitly override budgets.
sbatch learned_relu_geometry.sbatch --num-seeds 16 --restarts 12 --steps 8000

# Existing run.sh workflow is also supported.
sbatch run.sh experiments/17_learned_relu_geometry.py outputs/my_relu_run
```

Do not submit multiple processes into the same output directory. The array
launcher automatically gives each family a separate directory. Array outputs
are `outputs/learned_relu_geometry/ARRAY_JOB_ID/STUDY/`; a single allocation
uses `outputs/learned_relu_geometry/JOB_ID/`.

## Seven questions and default grids

| Array index / study | Controlled experiment | Proof-relevant output |
| --- | --- | --- |
| 0 / phase | Two features; one compression coordinate; 2 or 4 learned ReLUs; coactivation fractions 0,.02,.05,.1,.25,.5,.75,1; weight decay 0,1e-5,1e-3 | Encoder sign/angle, learned thresholds, competing local solutions, model energy and finite-sample aliasing |
| 1 / finite | Same pair problem; coactivation 0,.1,.5,1; 64,256,2048,8192 representation-training examples | Which geometry needs population coverage, empirical versus evaluation stationarity, comparison with quadrature-trained phase solutions |
| 2 / ablations | Joint learning versus fixed random encoder or fixed initial thresholds; gate removal, spectrum/Gram-preserving rotations and threshold shifts | Whether the benefit needs encoder learning, threshold learning or gate orientation; separate approximation from amplification |
| 3 / tasks | Coactivation 0,.25,1; marginals (.1,.1) or (.1,.3); task weights (1,1),(1,.1),(.1,1) | Task-selective learned geometry; paired objective swaps on identical inputs/initializations |
| 4 / blocks | 8 or 16 features in independent correlated pairs; compression p/4 or p/2; expansion p or 2p; marginal .1 or .3; coactivation 0,.25,1 | Whether the small learned mechanism survives interacting compressed feature groups |
| 5 / landscape | Coactivation 0,.25,1; 17 encoder angles in [0,pi); learn all decoder/head parameters at each fixed encoder | Conditional loss/aliasing profiles and competing decoder solutions to guide a candidate proof |
| 6 / higher_order | 8 or 16 features in independent four-bit parity blocks; eta=0,.25,.5,.75,1; block activation rho=.2,.6 | Whether higher-order coactivation changes LEARNED geometry while low-order activation laws are held fixed |

### Data and targets

For pair marginals a,b, q runs from max(0,a+b-1) to min(a,b). The reported
`coactivation` is the fraction along this feasible interval. The probabilities
of neither, first only, second only and both are respectively
`1-a-b+q, a-q, b-q, q`. Individual activation/amplitude laws stay fixed;
covariance DOES change. Active amplitudes are independent Uniform[.5,1.5].
Targets are all latent coordinates, centered and normalized by their analytic
individual variances. Balanced training means equal task weights. This is
a full feature-query bank, not an easily solved single aggregate label.

Parity blocks switch off with probability 1-rho. Otherwise choose a uniform
even-parity bit pattern with probability eta and a uniform odd-parity pattern
with probability 1-eta. All joint activation laws involving at most three
bits per block match across eta. Independent amplitudes preserve that matching.
This deliberately differs from exact-k sampling. Its purpose is a matched
higher-order test of learned geometry, not a claimed theorem about the old
generator. Neither pair nor parity experiments imply covariance is enough.

### Training and numerical integration

Main models are `x -> learned linear compression -> learned affine expansion
-> ReLU -> learned affine task bank`. The matched linear arm has identical
parameterization, data, optimizer streams, task objective and update budget.
Regularization is the existing factorwise L2 penalty on weight matrices;
biases are unpenalized. Parameter counts are the same within these pairs.

Two-feature population studies use deterministic Gauss-Legendre quadrature
on each support: order 20 for training, orders 40/80 for independent reference
projections, order 120 for evaluation. Piecewise ReLU kinks mean these are
NUMERICAL approximations, not exact integrals or global optima. Reference
tables check projection/alias-vector stability. To assess optimization's
quadrature error, rerun with `--training-order 40` in a NEW directory. For
evaluation/projection refinement use a JSON override, e.g.:

```json
{"training_order":40,"reference_orders":[80,160],"evaluation_order":240}
```

Pass that file with `--config-json path/to/refined.json`. Quadrature head
fits are probability weighted. Quadrature nodes are NEVER counted as iid
readout samples. Finite/large models use empirical minibatches and independent
reference fits on 16,384/65,536 observations; evaluation uses 16,384 observations.
The native head's GAD is reported only for empirical representation training.

All starts and checkpoints are selected by representation-training objective
only. No reference, validation or test loss selects representation checkpoints
or the winning start. `population_gradient_norm` includes the stated penalty
and is an evaluation diagnostic, not proof of stationarity/global optimality.
The fixed-angle landscape freezes only the encoder and learns its decoder;
it is a conditional multistart profile, not an exhaustive landscape search.

### Independent heads and GAD

Independent readout sizes are 3,4,5,8,9,15,16,17,18,31,32,33,34,64,128,256,512,2048,4096, with
eight orders of one shared pool. Native training, head fitting, reference
projection, ridge selection and prediction evaluation have separate roles.
Repeated orders use the same full set at the largest n; they do not add seeds.
Ridge penalties are validation-selected using another independent split.

For r_T=Y_T-H_T beta and r_P=Y_P-H_P beta:

```text
prediction - target
 = H_P(H_T^+H_T-I)beta     data insufficiency
 + H_P H_T^+ r_T           MP generalized aliasing
 - r_P                    signed model residual
 + H_P(beta_candidate-beta_MP)  estimator correction for ridge/native
```

The existing precision-fallback evaluator verifies the prediction and risk
identities; it does not change the estimator or singular-value threshold.
All component energies and all six signed cross terms are saved. The MP
alias column is still the MP baseline on ridge rows. Classification entries
are incidental sign diagnostics; this suite's target is reconstruction risk.

`residual_leverage_J` records E[mean_tasks(residual^2)*leverage] using the
evaluation feature span. Compare n*aliasing with J only at stable large n.
This standard asymptotic statistic is not itself the novelty claim. Near
singularities, inspect raw risk, rank/condition, medians, maxima, cancellation
and extended-precision flags. Undefined ratios remain NaN.

For pair support strata, labels 0,1,2,3 mean neither, first only, second only,
both. Larger-model strata use active count. Conditional terms retain the
global reference and include aggregate signed cross terms. Zero-probability
strata are absent, not assigned invented errors.

## Inspect and return results

Each completed family writes overview figures and `RESULTS.md`; pair studies
also show optimization basins, and phase studies show learned gate boundaries. All scenarios
remain in the CSVs; the sample-size figure labels a deterministic representative
scenario rather than selecting a winner by evaluation loss. Check paired
contrasts, seed means and starts instead of relying on a crowded phase plot.

- `geometry_*`: model error, encoder geometry, residual alignment and gate coverage.
- `readouts_*`: independent MP/ridge GAD, all cross terms and identity diagnostics.
  Also inspect `unseen_gate_pattern_mass` and `never_active_gate_count` for
  sampled coverage. These diagnostics are not renamed as GAD aliasing.
- `strata_*`: probability and conditional GAD by support/active count.
- `feature_errors_*`: per-feature model and MP errors, useful for task-priority
  swaps. `sample_size=0, estimator=reference` denotes the reference model
  energy; selected positive sample sizes retain all three energies and total
  signed cross term for draw zero.
- `references_*`: sensitivity to numerical reference resolution/sample size.
- `starts.csv`: every optimization objective and free-gradient norm.
- `gates.csv`, `parameters.csv`: unit-level geometry; do not average arbitrary gate labels.
- `shards/*/checkpoints.npz`: every trained restart and intervention, including native heads.
- `shards/*/history.csv`: training histories for every restart.
- `review_bundle.zip`: compact review package; return one ZIP for each array family.

Draws are averaged within representation seed before inference. Starts are
retained separately and never counted as independent seeds. Pointwise
bootstrap intervals are exploratory, not simultaneous or corrected for
selecting a crossover after seeing the plots. Task swaps and adjacent-q
comparisons can be paired by seed using the retained seed tables.

## Resume or replot

Each world completes atomically with checksummed artifacts. Failed worlds
remain explicit; the runner exits nonzero for missing/failed configurations.
Resume requires identical scientific configuration and Python source hashes.
It retries unfinished worlds and verifies completed CSV/checkpoint checksums.
Do not edit the checkout while a run is active.

```bash
# Existing single job/family output; no scientific flags need repeating.
.venv/bin/python experiments/17_learned_relu_geometry.py \
  --resume --workers 28 --output-dir outputs/learned_relu_geometry/JOB_ID/phase

# Summarize completed shards without training; PARTIAL reports exit nonzero.
.venv/bin/python experiments/17_learned_relu_geometry.py \
  --resume --summarize-only --output-dir outputs/learned_relu_geometry/JOB_ID/phase

# Plot-only reads summaries; no training.
.venv/bin/python experiments/plot_learned_relu_geometry.py \
  --output-dir outputs/learned_relu_geometry/JOB_ID/phase
```

For SLURM resume, explicitly set `GAD_GEOMETRY_OUTPUT` to the OLD directory
and `GAD_GEOMETRY_RESUME=1`. When resuming an array, set the root to the old
array root; the launcher appends the selected family name. Source snapshots,
configuration and invocation/environment details are saved with results.

## Scientific checks

```bash
.venv/bin/python -m pytest tests/test_learned_geometry.py \
  tests/test_representation_models.py tests/test_mechanism_checks.py
```

Tests check analytic moments, quadrature weighting, gradients and frozen
parameters, GAD signs and rank regimes, residual leverage, probability-weighted
stratum reconstruction, seed aggregation, plan construction and shard integrity.
The experiment's purpose is to find or reject proof candidates; passing tests
does not establish the scientific hypotheses.
