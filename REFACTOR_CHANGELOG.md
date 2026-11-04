# Chapter 2 v3 refactor changelog

## Added

- `src/aliasing/intrinsic_gad.py`: separate float64 population-metric GAD-v3 oracle, strict intrinsic-kernel solver, interpolating and noninterpolating risk APIs, gauge transform, independent-residual and spectral diagnostics, and weighted min-norm readout.
- `src/aliasing/chapter2_v3.py`: stable independent RNG streams, exact-k Gaussian and Bernoulli-uniform input laws with analytic moments, a tied sparse reconstruction autoencoder, and frozen-representation diagnostics.
- `experiments/chapter2_v3/`: immutable planner, locked atomic job runner, periodic resumable training checkpoints, partial collector, seed-level summaries, paired contrasts, figures, reports, and a compact results bundle.
- `configs/chapter2_v3/`: named smoke, overnight-discovery, and fresh confirmation profiles.
- `scripts/chapter2_v3/`: dry-run-first Slurm orchestration, CPU arrays, independent representation-training arrays, and an `afterany` aggregate collector.
- v3-specific tests and a future-cluster audit command. Tests are intentionally not run during this implementation.

## Preserved interfaces and output provenance

Legacy `gad_definition_version=2` results, `finite_support_gad`, `src/aliasing/linear.py`, experiments 01–17, and their historical output schemas remain unchanged. The existing GAD-v2 `data_insufficiency`, `aliasing`, `model_insufficiency`, and fitted-noise quantities are prediction-space components whose squared norms can have cross terms. They are not renamed to v3 `A`, `D0`, or `D_e`. The historical label reversal remains documented in `GAD_DEFINITIONS_AUDIT.md`.

All v3 output is isolated under `outputs/chapter2_v3/`; prior result folders and paper figures are not rewritten. Planned jobs record unavailable scheduler resources as `NOT_QUERIED`. Missing activation caches produce `SKIPPED_DEPENDENCY`; no synthetic or real-network results are fabricated.

Lane D2 holds raw teachers fixed across widths (dense and sparse targets); its representation-relative in-span control remains in the exact-width and learned-representation lanes. Risk outputs distinguish the `theta_circle` oracle-nearest interpolant, the `theta_0` population-metric minimum-norm interpolant, and the Euclidean minimum-norm readout. Lane P retains its estimated-oracle cross-term and reference-size sensitivity diagnostics instead of treating finite-reference deviations as exact-math failures.

## Mathematical specification review

The displayed `theta_0` equation is consistent with `theta_circle`: their difference is the population-oracle component in the sample nullspace. The implementation follows the equations literally and reports `D0` as samplewise data insufficiency. Lane A records both the finite-mean Gaussian prediction for `A` and the conditional Gaussian prediction for `D0`; v3 tests include both definition-specific counterexamples. See `MATH_SPEC_REVIEW.md` for the derivation.

## Validation state for this commit

No experiment, test suite, Slurm command, or result collection was run. The future workflow must run the v3 and legacy test suite, complete the smoke profile, and inspect its generated bundle before discovery or confirmation submissions.
