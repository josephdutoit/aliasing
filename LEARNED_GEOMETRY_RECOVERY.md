# Recover numerical failures from job 14872283

Reported totals: 4,848/5,016 worlds completed; 168 failed. These failures are
evaluator errors, not evidence for or against the learned-geometry hypotheses.
The exact node failure inputs/logs have not yet been examined in this patch.

## What was reproduced and fixed

A deterministic regression with nearly dependent modeled features and a
large reference head reproduces `extended GAD identity failed after precision
retry`. Longdouble still loses digits in large intermediate products that
cancel, even when the final error vectors are modest. On that regression the
old retry has a prediction gap around 4.8e-5; the new evaluation reconstructs
the same implemented predictor to around 1e-16.

Experiment 17 now tries the old evaluator first, retaining its accepted
metrics unchanged, then retries failed contractions with 80/160 decimal digits.
Decimal is part of Python's standard library; no dependency installation is
needed. Training, sampling, weights, inverse, SVD cutoff, reference head,
candidate head and direct float64 predictions are unchanged. Experiment 16's
precision evaluator is unchanged.

Canonical GAD terms retain their definitions. If the float64 implementation
itself has appreciable rounding, its effect is reported separately as
`arithmetic_roundoff`, together with its four signed cross terms. This vector
is computed independently from fitting/prediction/subtraction rounding, not
assigned as an unexplained identity gap. It is not generalized aliasing,
data insufficiency or a ridge correction.

`gad_arithmetic_reliable=0` flags rounding beyond the risk-scale tolerance.
Inspect `readouts_numerical_warnings.csv` before interpreting these rows as
statistical error. Accepted old rows have `arithmetic_roundoff_measured=0`;
their zero arithmetic fields are a schema convention at the previous identity
tolerance, not a claim of exact arithmetic. All original energies and cross
terms plus the extra arithmetic terms reconstruct the implemented risk.
Conditional/per-feature outputs use the same evaluated vectors.

## Retry only the four failed families

From the branch checkout, after pulling the fix:

```bash
git pull --ff-only
sbatch --array=0,2,4,5 repair_learned_relu_geometry.sbatch \
  outputs/learned_relu_geometry/14872283
```

This prepares NEW directories under
`outputs/learned_relu_geometry/14872283_repaired/STUDY/`. It verifies the
original source snapshots and all completed artifact checksums, reuses completed
shards using hard links on the same filesystem, and reruns unfinished worlds.
The old root, successful CSVs/checkpoints and saved configuration are not
modified. Across filesystems it copies instead of hard-linking.

Each repaired family should initially retain/retry:

| Family | Completed retained | Worlds retried |
| --- | ---: | ---: |
| phase | 1,022 | 130 |
| ablations | 167 | 25 |
| blocks | 1,142 | 10 |
| landscape | 1,221 | 3 |

The three already complete families need no rerun. Repeated recovery commands
resume the NEW outputs and verify provenance again. Do not run two recovery
jobs for the same destination simultaneously or edit linked completed shards.

For one family, directly:

```bash
.venv/bin/python experiments/repair_learned_relu_geometry.py \
  --input-dir outputs/learned_relu_geometry/14872283/phase \
  --output-dir outputs/learned_relu_geometry/14872283_repaired/phase \
  --workers 28
```

Add `--prepare-only` to validate provenance and link completed shards without
running anything. Ordinary `--resume` still rejects changed source/configuration;
the recovery command is a narrow, audited migration, not a general hash bypass.
Only the previous known evaluator/summary sources and the new precision helper
are accepted differences. Any training/generator/model source change is rejected.

`recovery.json` lists inherited/pending worlds and exact old/new hashes.
`provenance/` retains the original configuration, manifest, progress and sources.
The new reports/bundles include a `RECOVERY.md` explanation. Review original
clean-family bundles alongside repaired-family bundles, and retain the old
partial outputs for provenance. Do not interpret old partial-family means as
unbiased estimates: failed worlds may depend on learned geometry.

## Share compact results

Return `review_bundle.zip` from original `finite`, `tasks`, `higher_order`,
and from the four repaired families. The 8.6 GB of raw outputs/checkpoints
can stay on the node. For confirming the node-specific errors, include the
old `phase/progress.json` or `phase/run_manifest.json` too.
