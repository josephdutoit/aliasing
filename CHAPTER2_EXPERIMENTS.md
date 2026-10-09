# Chapter 2: finite-dimensional GAD pilots

These small synthetic experiments are meant to help select the theory to
develop. They are diagnostics and illustrations; they do not establish the
analytic claims on their own.

## Run

```bash
python experiments/04_gad_geometry_pilot.py --output-dir outputs/gad_geometry_pilot
```

The output directory contains three CSV files and one figure:

- `exact_identity.csv` and the first panel of `gad_geometry_pilot.png` compare
  held-out squared risk for train-null perturbations with the quadratic GAD
  prediction. The script tests both the top-eigenvector direction and the
  average over random unit directions in the training nullspace.
- `nested_geometry.csv` records nullity, operator norm, and trace as nested
  samples grow under isotropic, power-law, and low-rank population covariances.
  It keeps the worst-case operator norm separate from average aliasing measured
  by the trace. No Marchenko–Pastur curve is used for this population-covariance
  operator.
- `fisher_weights.csv` tracks logistic Fisher weights as logits become more
  confident, then separately sets selected weights exactly to zero. Positive
  weights preserve the exact row space, though the weighted design can become
  poorly conditioned. The script reports numerical rank and singular values
  because floating-point rank thresholds can make near-zero directions appear
  to be null directions. For exact-zero weights and identity population
  covariance, the trace equals the nullity.

The seed, number of nested-geometry replicates, test sample size, and output
directory can be changed with `--seed`, `--replicates`, `--n-test`, and
`--output-dir`.
