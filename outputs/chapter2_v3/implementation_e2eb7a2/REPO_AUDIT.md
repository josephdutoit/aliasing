# Chapter 2 v3 repository audit

**Recorded before source edits.** Repository: `josephdutoit/aliasing`. Starting
commit: `e2eb7a2a6b4511c583292deadd9732787c0ed081`. Starting branch: `work`.
The tracked and untracked working tree was clean at inspection time.

## Existing mathematics and compatibility

- `MATHEMATICAL_RESULTS.md` and `src/aliasing/linear.py` define the existing
  modeled/unmodeled GAD-v2 prediction identity, including fitted noise and
  signed cross terms. It is not the population-metric v3 decomposition in the
  implementation brief.
- `src/aliasing/static_geometry.py::finite_support_gad` now uses a single SVD
  decision for its pseudoinverse and row-space projector and emits v2 names.
  `GAD_DEFINITIONS_AUDIT.md` records the historical name reversal and how to
  interpret old result files. New v3 measurements must remain separately named
  and versioned.
- `src/aliasing/precision_gad.py` is a fallback for experiment 16's existing
  float64 reference-head contractions; it does not implement the new oracle.
- `src/aliasing/controlled_superposition.py` provides deterministic sparse
  synthetic inputs, a NumPy Adam trainer, ReLU features, and v2 readout metrics.
  Its encoder/decoder/task-head architecture is not the tied-weight
  reconstruction model specified for v3 Lane C. `representation_models.py`
  provides reusable model interfaces and feature helpers, but its GAD outputs
  are also v2/reference-head based.

## Existing experiments, tests, and outputs

- Experiments 07–11 already cover finite factorial geometry, trained sparse
  features, random-feature double descent, noisy collisions, and collision
  mass controls. Experiments 14–17 add learned-model, mechanism, benchmark, and
  survey workflows with useful seed/shard/resume patterns. They do not provide
  the v3 population-oracle lanes or the requested v3 dispatcher/collector.
- Existing tests cover the legacy linear decomposition and v2 finite-support
  semantics. No v3 math tests or `experiments/chapter2_v3/` implementation
  existed at this commit.
- Prior files under `outputs/chapter2/`, `outputs/learned_aliasing/`,
  `outputs/gad_mechanisms/`, `outputs/representation_models/`, and
  `outputs/overnight/` are retained exploratory results. No checkpoint or
  activation arrays were found in the checkout or inspected result bundles;
  Lane P therefore needs an external activation cache and must skip cleanly
  when one is unavailable.

## Launch and compute environment

- Existing `run.sh`, `overnight.sbatch`, `learned_aliasing.sbatch`, and
  `learned_feature_survey.sbatch` establish repository-root and `.venv` usage,
  CPU thread caps, and resume conventions. The new launcher must not alter
  these historical entry points.
- `sbatch`, `squeue`, and `sinfo` were unavailable in this executor. No
  supercomputer partition, wall-time limit, memory limit, GPU availability, or
  scratch path can be established here. New submissions must require explicit
  user-provided resource settings and make no runtime or GPU claims.
- The request explicitly says not to run. No experiment, test suite, Slurm job,
  or result collection will be executed during implementation. Tests and
  validation commands will be added to the workflow for the user's cluster.

## Implementation boundary

Add an isolated v3 API and namespace, v3 configs, deterministic planning,
immutable shards, resumable lane execution, partial-run collection, reports,
and a bundle builder. Preserve the legacy APIs, experiment numbers, and prior
outputs. Treat all result reports as `NOT_RUN` until the user runs the jobs and
collects actual outputs.
