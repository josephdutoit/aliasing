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
