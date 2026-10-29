# Model benchmark precision fix and full rerun — 2026-10-28

**All 144/144 worlds completed with zero failures.** Every mean-table group contains 24 seeds.

## Original failure

Node job 14840288 completed 143/144 worlds. The single failure was seed 8, k=8, the wide MLP with its final gate removed, independent MP readout n=17, draw 2. It was reproduced from the exact generator and training configuration.

The readout design condition number was about 1.23e7 and the reference-head norm about 1.50e4. Computing `(H^+ H-I) beta_reference` in float64 amplified roundoff, producing a prediction-identity gap of 1.67e-7. Direct test MSE was only 3.68e-20. The assertion was a numerical failure in the decomposition evaluator.

## Fix and validation

- Successful float64 evaluations retain exactly the old metrics.
- Failed checks retry the same GAD contractions in extended precision. Inverse, selected head, singular-value cutoff, direct predictions and check tolerances stay unchanged.
- Failed retries still stop the world, with its architecture/readout context in the error.
- Experiments 14/15 and their evaluator are unchanged.
- All 27 scientific test functions passed, including legacy compatibility, GAD rank regimes, cancellation regression and unchanged successful metrics.

The rerun required exactly one extended-precision readout evaluation: the original failing case. Its direct MSE is 3.6827618e-20, and its prediction-identity gap is 1.5615717e-10.

## Rerun scope and provenance

This is a fresh full overnight-profile run in the Codex workspace using 8 workers, not a submission to the supercomputer. It uses 24 seeds, k=1,2,4,8,12,16, all three default architectures and all configured controls, references, readout sizes, draws and perturbations. The node output remains preserved.

NumPy: 2.3.5. Source contents were verified against commit `ac05f1b1ccbc1205965f6c2c427f6ffd7085069d`. The archive contains configuration, environment, source hashes, per-seed summaries, paired intervals and figures. Floating-point details can vary between NumPy/BLAS builds.

## Large independent readouts

Mean MP test MSE at n=4096; representations are frozen before fitting these heads.

| Active count k | Bottleneck ReLU | Wide MLP ReLU | Narrow MLP ReLU | Wide MLP linear |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0002148109 | 0.0003114172 | 0.722094 | 3.59537e-30 |
| 2 | 0.2758327 | 1.51763e-06 | 0.7203773 | 3.384001e-30 |
| 4 | 0.4570831 | 0.005347326 | 0.7308485 | 5.181781e-30 |
| 8 | 0.6367728 | 0.05029862 | 0.773668 | 1.299063e-29 |
| 12 | 0.709235 | 0.1842209 | 0.8309298 | 2.491266e-29 |
| 16 | 0.7521246 | 0.3714839 | 0.8998853 | 1.818319e-29 |

The wide linear model has sufficient capacity for these linear targets. Its near-zero large-readout error is a necessary control. The narrow MLP retains the affine-output dimension restriction. These comparisons do not establish a universal nonlinear advantage or uniform alias suppression.

## GAD terms at k=4, independent readout n=128

| Architecture / map | Direct test MSE | Model term | MP alias term |
| --- | ---: | ---: | ---: |
| bottleneck / learned_linear | 0.7486737 | 0.7205863 | 0.02808749 |
| bottleneck / learned_relu | 0.5246894 | 0.4554193 | 0.06968459 |
| mlp_wide / learned_relu | 0.006167582 | 0.005332755 | 0.0008695212 |
| mlp_narrow / learned_relu | 0.7580886 | 0.7300764 | 0.02798011 |

Signed cross terms and data insufficiency remain in the CSVs; the displayed energies are not an additive risk partition. Ridge/native rows retain the canonical MP alias baseline plus their estimator correction.

## Repeat on the node

After merging the fix, run from the checkout:

```bash
git pull
sbatch representation_models.sbatch
```

The changed evaluator has a new fingerprint, so use a new output directory. Strict resume of job 14840288 with changed source is intentionally rejected. Full instructions are in REPRESENTATION_MODELS.md.
