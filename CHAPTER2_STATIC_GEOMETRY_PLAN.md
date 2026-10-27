> **GAD definition correction:** see [GAD_DEFINITIONS_AUDIT.md](GAD_DEFINITIONS_AUDIT.md). Historical `finite_support_gad` outputs reversed aliasing and data insufficiency. Old figures require relabeling before GAD interpretation.

# Chapter 2: Static geometry, superposition, and aliasing

## Purpose

This plan develops numerical evidence for a chapter about **static geometry in trained neural networks and aliasing**. Superposition, finite-sample aliasing, and representation compression are related questions, but they are different quantities. The experiments must measure them separately and test their relationship.

The main question is:

> When does a compact or superposed representation make task-relevant distinctions invisible to the training data?

This is a research plan, not a claim that the proposed relationship must hold. Report null or mixed results clearly.

## Scope and repository context

Work from the existing implementations and definitions before adding new code:

- `MATHEMATICAL_RESULTS.md`: exact fixed-feature GAD decomposition, frozen-readout null-space result, and the finite-support construction.
- `experiments/01_linear_aliasing_double_descent.py`: fixed-feature baseline.
- `experiments/02_random_feature_aliasing.py`: frozen random ReLU features.
- `experiments/03_feature_learning_mlp.py`: trained MLP and frozen-head null-response diagnostics.
- `experiments/04_superposition_aliasing.py`: exact dictionary (D_r=[I_d\;v_r]) example.
- `NN_METRICS.md` and `NN_EXPERIMENTS.md`: metric meanings and existing commands.
- `tests/`: current unit tests.

Do not duplicate these experiments. Preserve their result files. Do not interpret `activation_pattern_correlation_max` as a superposition measure; it measures similarity of activation patterns. Chapter 2 analyzes trained checkpoints as static representations. Do not add a training-trajectory or grokking claim here; those belong to the dynamics chapter.

The NeurIPS 2022 paper “Feature Learning in L2-regularized DNNs: Attraction/Repulsion and Sparsity” motivates measuring hidden-layer sample Gram geometry and width plateaus. For a layer activation matrix $Z\in\mathbb R^{m\times n}$ with n examples, record
$$
K=Z^\top Z,\qquad K_\sigma=\operatorname{ReLU}(Z)^\top\operatorname{ReLU}(Z).
$$
These are **sample-geometry/compressibility diagnostics**. The paper's minimum atom count for a covariance pair is not the number of latent features represented in superposition, and its global-optimum width theorem is not verified merely by a finite-seed training sweep.

## Quantities to keep distinct

### 1. Task-relevant data insufficiency for fixed features

Use the repository's existing GAD conventions and exact decompositions. In the scalar noiseless linear case, with training design $Z_T\in\mathbb R^{n_T\times p}$, target coefficients $\beta$, population feature covariance $\Sigma$, and
$$
Q_T=I-Z_T^\dagger Z_T,
$$
the minimum-norm fitted coefficient is $\widehat\beta=Z_T^\dagger Z_T\beta$, and its population squared prediction error is
$$
A_T=\beta^\top Q_T\Sigma Q_T\beta.
$$
This is the task-relevant data-insufficiency contribution, not generalized aliasing. With noise, multiple outputs, or other GAD terms, use and numerically verify the formulas in `MATHEMATICAL_RESULTS.md`; do not silently replace the full decomposition with this special case.

Report direct held-out risk alongside each decomposed term. Include cross-terms when required by the exact identity.

### 2. Feature superposition and sparse decodability

In a controlled representation $h=Ds$, let $D\in\mathbb R^{m\times p}$ map p known latent features into m representation dimensions. Report:
- feature-to-dimension ratio $p/m$;
- normalized feature-direction coherence;
- support-conditioned minimum singular value
$$
  \kappa_{2k}(D)=
  \min_{\|v\|_2=1,\;\|v\|_0\le 2k}\|Dv\|_2;
$$
- exact collision probability on the specified code distribution;
- reconstruction or feature-recovery error.

For codes with at most k active features, exact sparse-code injectivity is equivalent to there being no nonzero $2k$-sparse vector in $\ker D$, i.e. $\kappa_{2k}(D)>0$. Compute this exactly by support enumeration only for small dimensions where feasible. For larger cases, label estimates or bounds honestly. Separate exact collisions from near-collisions caused by small nonzero singular values.

### 3. Sample-geometry compressibility

For each selected layer/checkpoint, report (K,K_\sigma) spectral summaries (including numerical ranks under documented tolerances) and the actual layer width. If estimating the paper's minimum atom count, validate the estimator on small cases and label it as an estimate or bound unless exact minimality is certified. Do not use this quantity as a superposition score.

## Prioritized implementation

### Phase 0 — establish the baseline

1. Inspect the existing scripts, mathematical results, and test coverage.
2. Run the unit tests and the existing finite-support experiment on a small configuration.
3. Add or extend tests for the exact identities and threshold already claimed in `MATHEMATICAL_RESULTS.md`. In float64, compare direct and decomposed risk on small deterministic and random rank-deficient examples.
4. Report any mismatch before adding new metrics or experiments.

### Phase 1 — broaden the exact controlled geometry

Extend the existing `04_superposition_aliasing.py` experiment or add a clearly named successor. Retain its $D_r$ case as a regression control and add a small, reproducible factorial design:

- dictionary families: orthogonal baseline, structured $D_r$, and seeded normalized random/coherent dictionaries;
- overcompleteness (p/m\in\{1, 1.5, 2\}) where dimensions permit;
- sparse support sizes (k\in\{1,2,3\}) for small (p);
- code distributions with and without feature co-occurrence;
- data coverage levels below, near, and above the relevant feature dimension.

Include four explicit controls:
1. no superposition and no aliasing;
2. superposition with sparse codes that remain distinguishable;
3. finite-sample aliasing without superposition (for example, an orthogonal feature direction absent from training);
4. superposition with a supported pair of distinct codes that collide.

For each case compute exact or support-enumerated collision statistics, $\kappa_{2k}$ when tractable, GAD terms for the downstream fixed-feature task, direct population/held-out risk, and feature recovery. Preserve the same dictionary and data-generation seed for paired comparisons across sample sizes. Keep the sweep small enough that exact support enumeration is practical.

### Phase 2 — test the relationship in trained networks

Add a small teacher/student or sparse-feature autoencoder setup with known latent features, a bottleneck, and a held-out code distribution. Choose the simplest model that can demonstrate feature sharing and evaluate it only at the final trained checkpoint for this chapter.

At that checkpoint:
1. Measure feature directions using the known latent features (e.g. controlled feature interventions or encoder sensitivities), not activation-pattern correlation alone.
2. Freeze the encoder and fit the same specified readout on training data. Compute the exact fixed-representation GAD/null-space quantities where their assumptions apply.
3. Compare train risk, held-out risk, feature recovery/interference, and GAD aliasing across the controlled sparsity, bottleneck, sample-size, and regularization settings.
4. Include a frozen/random-encoder baseline with the same task and data.
5. Record (K,K_\sigma) and width for the trained hidden layer as separate compressibility diagnostics.

Do not interpret a frozen-head response as total test risk. Do not infer superposition in a trained network unless the known-feature measurements support it. If the model cannot reliably identify ground-truth feature directions, report that limitation and keep claims to observable geometry.

### Phase 3 — width plateaus and double descent

Only after Phases 0–2 pass, run modest width and sample-size sweeps:
- samplewise: vary $n_T$ at fixed latent task and feature dimension;
- featurewise: vary $p$ or bottleneck width at fixed sample count and data-generation protocol;
- width: record train objective/test risk and covariance diagnostics across widths.

Report samplewise and featurewise double-descent curves only when the tested range actually spans the interpolation or peak region. A plateau from finite optimization is an *observed trained-width plateau*, not evidence that a global optimum theorem was reached. Pair seeds where the data-generation design allows it and state what randomness each confidence interval summarizes.

## Required outputs

For each phase, save:
- raw per-seed CSV with configuration and seed columns;
- summary mean, standard deviation, and confidence intervals across independent seeds;
- a machine-readable configuration/provenance file (dimensions, distribution, model, optimizer, tolerances, software versions, and seed);
- figures with uncertainty intervals and axes/units labeled;
- a short `RESULTS.md` stating what was validated, what was not, and any failed or unstable calculations.

Do not commit large raw outputs unless that is already the repository convention. Keep generated outputs in the configured output directory.

## Decision criteria

- Exact algebraic identities agree with direct computations to numerical precision on small float64 cases.
- The four controls behave as designed: superposition can occur without aliasing, and finite-sample aliasing can occur without superposition.
- For each controlled dictionary, the reported exact collision result agrees with support enumeration; near-alias results are reported separately.
- Any empirical claim linking superposition, covariance compressibility, GAD aliasing, and held-out risk is replicated across seeds and is not driven by one dictionary or one rank tolerance.
- If superposition or covariance compressibility fails to predict aliasing after the controls, report the dissociation. That is a useful result and should refine the theory rather than be hidden.

## First Codex task

Start with Phase 0, then implement and run a small Phase 1 pilot. Reuse current utilities and output conventions. Do not launch a large cluster sweep until the pilot passes the exact checks and the output schema is reviewed. Summarize the pilot and list the remaining Phase 1 work before moving on to Phase 2.
