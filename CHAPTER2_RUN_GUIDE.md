> **GAD definition correction:** see [GAD_DEFINITIONS_AUDIT.md](GAD_DEFINITIONS_AUDIT.md). Historical `finite_support_gad` outputs reversed aliasing and data insufficiency. Old figures require relabeling before GAD interpretation.

# Chapter 2 experiment run guide

Run the small exact checks first. The new experiments save raw per-seed rows,
summary tables, confidence intervals, configuration files, and plots under the
requested output directory.

## Install and validate

```bash
python -m pip install -e ".[dev]"
python -m pip install -r requirements-neural.txt
pytest -q
```

Run the existing exact finite-support baseline:

```bash
python experiments/04_superposition_aliasing.py \
  --dimension 4 \
  --collision-mass 0.5 \
  --output-dir outputs/chapter2/04_pilot
```

## Experiment 05: dictionary geometry and finite-sample aliasing

Pilot:

```bash
python experiments/05_static_dictionary_geometry.py \
  --dimension 4 \
  --feature-counts 4,5,8 \
  --max-sparsities 1,2 \
  --sample-sizes 8,16 \
  --num-seeds 3 \
  --output-dir outputs/chapter2/05_pilot
```

Default study:

```bash
python experiments/05_static_dictionary_geometry.py \
  --output-dir outputs/chapter2/05_dictionary_geometry
```

The experiment compares orthogonal, random, and structured dictionaries. It
reports dictionary coherence, the support-enumerated minimum singular value,
exact representation Bayes risk, a finite-support GAD decomposition, and
held-out risk. Orthogonal configurations with more features than dimensions
and structured configurations that are not overcomplete are skipped.

## Experiment 06: trained sparse-feature representations

Pilot:

```bash
python experiments/06_trained_sparse_feature_geometry.py \
  --feature-counts 6 \
  --bottlenecks 2 \
  --sample-sizes 16,32 \
  --num-seeds 2 \
  --n-test 512 \
  --epochs 150 \
  --output-dir outputs/chapter2/06_pilot
```

Default study:

```bash
python experiments/06_trained_sparse_feature_geometry.py \
  --output-dir outputs/chapter2/06_trained_sparse_geometry
```

The autoencoder learns a bottleneck representation of known sparse binary
features. At the final checkpoint, the script measures feature-intervention
directions, reconstruction, exact codebook collisions, frozen linear-probe
risk and its finite-support decomposition, and pre/post-ReLU sample Gram
summaries. It does not use activation-pattern correlation as a superposition
metric. Exact codebook diagnostics are limited to at most 16 input features.

## Experiments 07–11: hypothesis tests

The new experiments focus on rival explanations and use repeated seeds. Run
each independently with `run.sh`:

```bash
sbatch run.sh experiments/07_static_geometry_factorial.py outputs/chapter2
sbatch run.sh experiments/08_trained_superposition_aliasing.py outputs/chapter2
sbatch run.sh experiments/09_double_descent_decomposition.py outputs/chapter2
sbatch run.sh experiments/10_near_collision_noise.py outputs/chapter2
sbatch run.sh experiments/11_collision_mass_law.py outputs/chapter2
```

Experiment 07 holds the marginal activity of a feature pair fixed while
changing their joint activation probability. It tests five tasks against
random and structured dictionaries and separates exact representation risk,
linear model insufficiency, finite-sample aliasing, and data/algorithm error.
Its exact binary codebook is limited to 16 features.

Experiment 08 trains each sparse autoencoder once, freezes it, then evaluates
four target tasks over multiple readout sample sizes. Exact collision metrics
enumerate all binary inputs and are therefore limited to 12 features. The
reported confidence intervals use the independently trained representation
seeds; repeated readout draws are first averaged within each seed.

Experiment 09 builds paired random-feature designs and evaluates the full
sample-size-by-width grid under several input covariances and task alignments.
It records rank, condition number, and all GAD terms at each point. Output
directories contain raw rows, means, standard deviations, pointwise 95%
bootstrap intervals, a config file, and a PNG.

Experiment 10 varies the separation between two target-distinct hidden states
and adds Gaussian observation noise. It compares the quadrature-computed Bayes
risk with finite-sample affine-readout risk and its GAD terms. This is a
controlled robustness model for approximate, rather than exact, collisions.

Experiment 11 is the sharp exact control: at zero separation, its contrast-task
Bayes risk must equal `collision_mass * conditional_positive_mass * (1 -
conditional_positive_mass) * target_gap**2`. The common-target task should have
zero Bayes risk even though the represented states collide. It is the fastest
check that the collision diagnostic measures target aliasing rather than
geometric overlap alone. The script supplies the analytic population-optimal
head to GAD and stops if the balanced exact-collision control has a nonzero
GAD data-insufficiency term. Finite-sample residual aliasing can still be nonzero. It writes a `run_manifest.json` with formula and decomposition
checks.

Experiments 09 and 10 now plot pointwise seed-bootstrap confidence bands.
Experiment 09 writes a separate double-descent figure for every covariance and
task pair. Experiment 10 separates Bayes uncertainty from the noisy observation
channel, population affine-model insufficiency, finite-sample readout risk, and
the GAD alias component; it also plots readout sample-size convergence.
Regenerate only that sample-size figure from existing summaries, without
rerunning experiment 10, with:

```bash
MPLBACKEND=Agg python experiments/plot_near_collision_summary.py \
  outputs/chapter2/near_collision_noise
```

## Results to preserve and upload

Each script writes `raw.csv`, `mean.csv`, `std.csv`, `ci95.csv`, `config.csv`,
and `run_manifest.json` alongside its PNG figures. Keep `raw.csv` in the
cluster's experiment archive; it can be very large and does not need to be
committed for an initial review. Commit the PNGs plus `mean.csv`, `ci95.csv`,
`config.csv`, and `run_manifest.json`. The aggregate tables and manifest are
enough to check the plotted seed means, pointwise intervals, run settings, and
decomposition identities. Keep `std.csv` with the archive if disk allows. For
experiments 09–11, inspect `max_risk_identity_error` in the manifest; for
experiment 11 also check `max_formula_error` and the per-row
`balanced_exact_data_control` field.

Experiment 11 treats only `separation == 0` as an exact collision. Small
positive separations remain distinct support points, even when they fall below
`--collision-tolerance`; the tolerance is recorded as a practical scale only.
The shared GAD routine now obtains the training row-space projector and
minimum-norm fit from one SVD rank decision, avoiding roundoff amplification
when the near-collision readout coefficients are large.

For a focused smoke run, set the relevant environment variables before
submission. For example:

```bash
export NUM_SEEDS=3 FEATURE_COUNTS=6,8 SAMPLE_SIZES=8,16,32
export JOINT_PROBABILITIES=0,0.1,0.2 TASKS=random,decoder_null FAMILIES=random,structured
sbatch --export=ALL run.sh experiments/07_static_geometry_factorial.py outputs/chapter2
```

Full default sweeps are intentionally substantial. Override `NUM_SEEDS`,
`FEATURE_COUNTS`, `SAMPLE_SIZES`, `JOINT_PROBABILITIES`, `TASKS`, `WIDTHS`,
`COVARIANCES`, `SEPARATIONS`, `NOISE_LEVELS`, `PRIORS`, `COLLISION_MASSES`,
`CONDITIONAL_MASSES`, `EPOCHS`, and `N_TEST` as needed for a pilot or larger
run.
The hypothesis-to-metric mapping and interpretation rules are in
`STATIC_GEOMETRY_HYPOTHESES.md`.

The current `run.sh` SLURM header requests a one-hour limit. The full 08 and 09
sweeps may need more wall time; pass your cluster's allowed limit with
`sbatch --time=...` or update the header before submitting those full runs.

## SLURM on the configured cluster

The repository's `run.sh` dispatches experiments 04–06. From the repository
directory, submit the dictionary pilot with:

```bash
sbatch run.sh experiments/05_static_dictionary_geometry.py outputs/chapter2
```

Submit the trained-network pilot with reduced settings:

```bash
export FEATURE_COUNTS=6 BOTTLENECKS=2 SAMPLE_SIZES=16,32 NUM_SEEDS=2 EPOCHS=150
sbatch --export=ALL run.sh experiments/06_trained_sparse_feature_geometry.py outputs/chapter2
unset FEATURE_COUNTS BOTTLENECKS SAMPLE_SIZES NUM_SEEDS EPOCHS
```

The full default studies use 20 seeds for experiment 05 and 10 seeds for
experiment 06. You can override `NUM_SEEDS`, `SAMPLE_SIZES`, `FEATURE_COUNTS`,
`MAX_SPARSITIES`, `BOTTLENECKS`, and `EPOCHS` through the SLURM environment.

## Interpretation

- Check `decomposition_reconstruction_error` and `risk_identity_error` before
  interpreting any GAD component.
- `representation_bayes_mse` measures target ambiguity after observing the
  represented state. It is distinct from finite-sample readout aliasing.
- `aliasing` is the population risk contribution of the population-optimal
  readout component in the training null space. Prediction components need
  not be orthogonal; the CSV includes their cross-terms.
- Covariance ranks and stable ranks describe sample Gram geometry. They are
  not latent feature counts and are not direct superposition measurements.
- Compare mean and confidence-interval files across independent seeds; retain
  raw results and configuration files for provenance.
