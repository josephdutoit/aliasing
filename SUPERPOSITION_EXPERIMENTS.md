# Chapter 2: superposition and GAD experiments

These studies test the frozen-representation squared-loss theory. Encoders are
fixed before fitting data are drawn. Semantic targets are `f_beta(s)=beta^T s`.
Only the readout is fitted. No encoder-learning or novelty claim is made.

The code needs the existing NumPy and Matplotlib dependencies; it does not need
PyTorch. From the repository root:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/test_superposition.py
```

The new theory checks also run without pytest:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_superposition.py' -v
```

## 06: exact binary calibration and interventions

```bash
bash run.sh 06
```

This is a deterministic four-state calculation, not a Monte Carlo estimate.
It compares one-dimensional antipodal sharing (`z=s1-s2`) against dropping
feature 2 (`z=s1`), and an uncompressed control. All heads include an intercept.
Sharing uses the readout `(1,ReLU(z),ReLU(-z))`; dropping has the same architecture
with the identically zero channel removed. The uncompressed control has twice
the hidden width. Sum, contrast, and single-feature targets are evaluated.

The default grid includes:

- Independent activations and the sum-task B/A threshold
  `p=(1-1/sqrt(3))/2`.
- Marginals held fixed while joint probability changes. In particular,
  `p=.1,q=.022` demonstrates smaller B and larger expected A for sharing.
- `p=.2,q=.04`: sharing lowers B+A but Euclidean minimum-norm head selection
  increases D enough to reverse the total-risk benefit.
- A task-angle sweep, showing that the same geometry can help one target and
  hurt another.
- Exact interpolation probabilities versus sample size, including
  `(1-rho)^n+(1-q)^n-(1-rho-q)^n` for the shared sum task.
- One targeted conflict addition versus one random addition, with all baseline
  and random-added samples enumerated. The targeted rule adds the counterpart
  of a seen `00` or `11`, or adds `11` after a singleton. Both strategies use
  exactly one extra example. The encoder stays frozen: this tests readout
  selection and exposure of incompatible constraints, not representation repair.

Files under `outputs/superposition_exact/`:

| File | Content |
|---|---|
| `exact.csv` | Exact expected B,A,D,R and analytic B/A checks |
| `tasks.csv` | Task-angle comparisons |
| `coverage.csv` | Exact feasibility probabilities; no rejection sampling |
| `interventions.csv` | Every intervention outcome and its probability |
| `intervention_summary.csv` | Probability-weighted risk and feasibility |
| `independent.png`, `correlation.png` | Capacity-matched tradeoff plots |
| `metadata.json` | Parameters and interpretation |

## 07: continuous amplitudes and general task matrices

A quick end-to-end check:

```bash
bash run.sh 07 --pairs 2 --marginals 0.1 \
  --cooccurrence-fractions 0,0.22 --sample-sizes 1,3,8 \
  --replicates 4 --validation-replicates 4 --n-directions 2 --n-test 256 \
  --output-dir outputs/superposition_smoke
```

The main study, locally or on Slurm:

```bash
bash run.sh 07 --pairs 4,8 --replicates 100 --validation-replicates 50
sbatch run.sh 07 --pairs 4,8 --replicates 100 --validation-replicates 50
```

For each pair, `P(11)=q`, `P(10)=P(01)=p-q`, and `P(00)=1-2p+q`.
Nonzero amplitudes are independently uniform on `[0,1]`. Different pairs are
independent. `--cooccurrence-fractions` specifies `q/p`; thus changing q holds
the semantic marginals fixed. The defaults use p=.1 and .3, and q/p=0,.1,.22,.6.
Training sizes cross the **readout** ranks, which differ from hidden width.

Population moments are exact polynomial quadrature on singleton intervals and
the two triangles of the joint-activation square. This is sufficient for all
products used by the specified piecewise-linear decoders. Cross-pair moments
use independence. B,A,D and population risk therefore do not depend on a finite
evaluation grid. A separate independent Monte Carlo sample checks actual
prediction risk; its uncertainty is recorded separately.

Controls:

| Encoder | Hidden width | Readout features |
|---|---:|---|
| `shared` | m | Intercept and both ReLU signs of each antipodal pair |
| `shared_scaled` | m | Same functions, channels multiplied by 10 and .1 |
| `drop` | m | Intercept and one retained semantic feature per pair |
| `linear` | m | Intercept and signed differences, without ReLU decoding |
| `positive` | m | Intercept and pair sums |
| `uncompressed` | 2m | Intercept and all semantic features |

The scaled control preserves the function class and B,A. It can change D for
Euclidean fitting. The positive-sharing control exactly realizes sum tasks;
the antipodal ReLU control exactly realizes contrast tasks. With q=0,
antipodal ReLU sharing realizes every linear semantic task.

Two head rules are available. `euclidean` is the minimum Euclidean-norm least
squares head, equivalent to unregularized squared-loss gradient descent from
zero. `population` minimizes the known population-G norm from zero. This is
a controlled metric comparison, not the population-best interpolant centered
at the unknown target optimum; its D need not vanish. Both are unregularized.

Every replicate uses common semantic samples across encoders and common nested
training prefixes across sample sizes. No ill-conditioned or incompatible run
is discarded, and no ridge term is silently introduced.

The general B,A,D matrices are obtained by fitting semantic basis targets.
For the shared-versus-drop risk difference, calibration replicates estimate a
matrix Delta C. Its lowest and highest eigenvectors select two new semantic
tasks. **New independently drawn training datasets** validate their risk
differences; validation data never enter task selection. All eigenvalues are
reported, whether or not both signs occur. Finite-replicate eigendirections are
predictions to test, not guaranteed signs of population expectation.

Files under `outputs/superposition_continuous/`:

| File | Content |
|---|---|
| `replicates.csv` | Every task/encoder/head/sample-size attempt |
| `summary.csv` | Means, SEs, risk quantiles and finite counts; A/D counts reveal conditioning |
| `paired.csv`, `paired_summary.csv` | Sharing-minus-dropping effects on the same draws |
| `population.csv` | Exact population B, task energy, rank, overlap bounds |
| `semantic_errors.csv` | Error for every semantic feature, including joint-activation error |
| `task_spectrum.csv` | Risk-difference eigenvalues from calibration |
| `task_vectors.csv` | Original and matrix-selected target coefficients |
| `task_validation.csv` | Predicted versus new-dataset risk differences |
| `risk_matrices.npz` | B, risk, projected-label A/D and risk-difference matrices |
| `continuous_tradeoffs.png`, `task_validation.png` | Tradeoff and prospective-validation plots |
| `metadata.json` | Configuration, sampling law and caveats |

The continuous driver checkpoints completed-condition raw CSVs during execution.
Final summary CSVs, matrices and figures are written at completion. Seed streams
for original teachers, calibration datasets, population checks and validation
datasets are separate.

Raw and summary rows also report loss conditional on any within-pair joint
activation (and its complement), observed co-occurrence coverage, and training
joint-activation rates. The per-feature CSV averages the semantic basis-task
risks across calibration replicates. Conditional test losses use the independent
Monte Carlo sample, so an absent event is NaN rather than a fabricated zero.

## Interpolation and numerical interpretation

Original label-interpolation GAD requires feasible interpolation. Every row
includes `interpolates`, the actual training residual, numerical training rank,
population rank, and nonzero extremes of the whitened training Gram spectrum.
On incompatible runs **A and D are NaN**, and means of them are conditional on
their finite rows; use the accompanying counts and feasibility rates. Total
risk means include **all** attempts. Paired A/D differences require both heads
to interpolate; paired risk differences include all draws.

`projected_A` and `projected_D` are explicitly separate diagnostics obtained by
projecting unattainable training labels onto the training design range. Together
with B and `signed_remainder` they reconstruct risk. The remainder is roundoff
for these converged least-squares heads, even on incompatible labels; that does
not make the original label-interpolation theorem applicable. Do not relabel
projected-label terms as ordinary A,D outside interpolation.

SVD uses a relative cutoff of 1e-12. Singular population coordinates are removed
for the population geometry while retaining the original coordinates for the
Euclidean head. Very small singular values can produce heavy-tailed risk;
reported Monte Carlo SEs are descriptive and do not establish existence of an
expected-risk moment. Increase replicates and inspect individual runs before
making claims near interpolation.

## What these experiments establish

Study 08 adds an exact sample-size preference-reversal theorem and finite-n
learning curves:

```bash
bash run.sh 08 --max-n 64
```

See [SUPERPOSITION_SAMPLE_THEOREM.md](SUPERPOSITION_SAMPLE_THEOREM.md) for the
parameter interval, proof, sufficient sample bound, and interpolation scope.
Outputs are `curves.csv`, `boundaries.csv`, `sample_boundary.png`, and
`metadata.json` under `outputs/superposition_sample_boundary/`.

The exact study verifies the binary formulas, task dependence, genuine B/A
tradeoffs, and algorithm-dependent D reversals. The continuous study tests
multi-pair extensions, decoder dependence, metric invariance, sample coverage,
population transfer and prospective task predictions.

They do not establish which representations neural-network training selects.
That requires a subsequent feature-learning experiment with encoders trained
on a separate dataset or an explicit treatment of label-dependent feature
learning. Arbitrary dictionaries and dependencies between different pairs are
also outside the exact-moment distribution used here.
