# Chapter 2: feature geometry and train-null risk

`experiments/05_feature_geometry_risk.py` is the main numerical study for the
finite-dimensional GAD results. It repeats nested training designs across
isotropic, power-law, and low-rank population feature covariances. At each
sample size it measures the excess test risk of a top-eigenvector null
perturbation and the average risk over random null directions, then compares
those measurements with the GAD operator norm and trace predictions.

The summary reports means and 95% normal-approximation confidence intervals
across independent training samples. The replicate-level file includes the
training-null residual to verify that the sampled perturbations preserve the
training predictions up to floating-point error.

## Run

Local run:

```bash
./run.sh 05 --replicates 100 --n-test 4096 --output-dir outputs/chapter2_geometry_risk
```

Slurm run:

```bash
sbatch run.sh 05 --replicates 100 --n-test 4096 --output-dir outputs/chapter2_geometry_risk
```

The output directory contains `replicates.csv`, `summary.csv`, and
`feature_geometry_risk.png`. Change `--seed`, `--replicates`, `--n-test`,
`--n-directions`, `--p`, or `--sample-sizes` to vary the run.

`experiments/04_gad_geometry_pilot.py` and its existing outputs are retained as
the initial sanity check, including the exploratory Fisher-weight diagnostic.
That diagnostic illustrates that positive Fisher weights can worsen
conditioning without changing exact nullity; it is not used as evidence that
effective sample size determines the aliasing-space dimension.
