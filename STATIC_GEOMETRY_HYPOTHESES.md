# Static geometry and aliasing: numerical hypothesis map

This plan treats representation geometry, superposition, and aliasing as
related but distinct objects. The outcome is not assumed to be positive: each
experiment includes controls that can falsify the proposed link.

## Main hypotheses

| ID | Hypothesis | Evidence that would support it | Evidence against it |
|---|---|---|---|
| H1 | Exact collisions in a fixed representation create a target-dependent irreducible risk floor. | Exact code collisions have positive conditional target variance, and that Bayes risk matches the predicted floor across collision masses and task contrasts. | Collision mass rises while conditional target variance remains negligible, or the computed risk floor fails exact controls. |
| H2 | Geometry alone does not determine risk; task alignment and the data distribution determine how much of a poorly separated direction matters. | At fixed dictionary and marginal activity, co-occurrence/task changes move risk and GAD components; matched contrasts are strongest for aligned tasks. | Risk is essentially invariant to task and co-occurrence after matching representation geometry. |
| H3 | Representation aliasing, linear-model insufficiency, and finite-sample aliasing are empirically separable. | Decoder-null tasks can have zero representation Bayes risk but positive linear-probe insufficiency; reducing sample size changes GAD alias/data components without changing the representation floor. | The diagnostics collapse these mechanisms or cannot recover known exact controls. |
| H4 | Feature coherence or stable rank by itself is not a sufficient predictor of aliasing. | Different tasks/distributions at the same representation geometry have different risks, or similar geometry scores produce different alias components. | One geometry scalar predicts alias risk across held-out families, tasks, and activity laws. |
| H5 | In trained sparse autoencoders, superposition metrics relate to downstream aliasing only after specifying the task and readout sample size. | Feature-direction overlap/collision diagnostics predict task-specific aliasing out of sample, with different behavior for sum, contrast, and single-feature tasks. | Geometry metrics have no reproducible association, or their association vanishes after simple baselines such as rank and reconstruction error. |
| H6 | Double-descent spikes should be attributed to the GAD component that carries them, not inferred from the curve shape. | Replicated spikes coincide with a rise in alias energy or data/algorithm error and a measurable rank/conditioning transition. | Spikes disappear over seeds, are numerical artifacts, or occur without a corresponding GAD component change. |
| H7 | Distribution covariance and task alignment change which representation directions are practically relevant even when the encoder is fixed. | Isotropic, spiked, and low-rank input laws produce distinct task-conditioned aliasing under the same random-feature construction. | The effects vanish under matched paired simulations and adequate test sets. |
| H8 | Near-collisions create a noise- and task-dependent transition from recoverable codes to practical aliasing. | The exact noisy-channel Bayes floor changes with separation/noise ratio and target contrast, and finite-sample linear risk tracks it in the appropriate regime. | Risk is insensitive to separation/noise or tracks sample conditioning alone. |
| H9 | Exact collision risk follows the conditional-variance law and vanishes for target-invariant collisions. | Risk matches \(c\rho(1-\rho)\Delta^2\) across collision mass \(c\), conditional mass \(\rho\), and target gap \(\Delta\), but is zero for the common-target control. | The exact formula fails or risk does not depend on target disagreement. |

## New scripts

### 07 — `experiments/07_static_geometry_factorial.py`

Enumerates a finite binary codebook with one correlated feature pair and
compares random and structured dictionaries. The factorial varies sample size,
joint activation probability while holding marginals fixed, dictionary family,
and five task vectors. It reports exact representation Bayes risk, aliased
mass, linear model insufficiency, finite-sample aliasing, data/algorithm error,
coverage, dictionary conditioning, and task-to-decoder residual.

The `decoder_null` target is deliberately chosen outside the linear decoder's
row space. On a finite codebook the representation can still identify it
nonlinearly when there are no collisions. This is the control separating
representation ambiguity from linear readout insufficiency.

### 08 — `experiments/08_trained_superposition_aliasing.py`

Trains sparse autoencoders under the correlated activity law, freezes each
encoder, then varies readout sample size and task. It measures causal feature
intervention directions, exact represented-code collisions and target risk,
and the finite-support GAD decomposition. Readout draws are averaged within an
encoder seed before confidence intervals are formed, so repeated draws do not
masquerade as independent learned representations.

### 09 — `experiments/09_double_descent_decomposition.py`

Uses paired nested random-feature designs across width and sample size, with
isotropic, spiked, and low-rank input covariance and two teacher alignments.
Each risk curve point includes training rank/nullity and condition number plus
model, alias, and data/algorithm GAD terms. This directly tests whether a
double-descent-shaped risk spike is carried by aliasing, by estimation, or by
conditioning, and whether the pattern survives seed replication.

### 10 — `experiments/10_near_collision_noise.py`

Varies hidden-state separation, observation noise, class prior, and readout
sample size in an exactly solvable two-state channel. Gauss-Hermite quadrature
supplies the Bayes risk; Monte Carlo test draws supply affine-readout and GAD
terms. It tests whether approximate overlap acts like practical aliasing once
observation noise is present.

### 11 — `experiments/11_collision_mass_law.py`

Implements a three-state support with a pair of hidden states whose separation
is controlled exactly. For zero separation, its contrast-task Bayes risk must
equal \(c\rho(1-\rho)\Delta^2\); a common-target control has no collision
penalty. The sweep independently varies collision mass, pair imbalance,
separation, target task, and readout sample size. `formula_error` directly
checks the exact risk law.

## Reading the results

1. Validate decomposition reconstruction and risk-identity errors before
   interpreting component curves.
2. Keep exact representation Bayes risk distinct from population linear-model
   insufficiency and finite-sample aliasing.
3. Use paired seed-level contrasts for changes in co-occurrence, task, width,
   and sample size; report pointwise intervals and do not select the largest
   spike without correcting for the sweep.
4. Treat coherence, singular values, effective rank, and feature-direction
   overlap as candidate predictors. Compare their out-of-sample predictive
   value against task, activity, sample coverage, and simple rank baselines.
5. A successful numerical pattern motivates a theorem; it does not establish
   one. A null result should be retained with its configuration and raw rows.
