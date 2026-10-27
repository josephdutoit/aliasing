# Review of job 14794630: corrected GAD interpretation

**The original interpretation reversed generalized aliasing and data insufficiency.**
See [GAD_DEFINITIONS_AUDIT.md](GAD_DEFINITIONS_AUDIT.md) for the authoritative
equations, affected experiments, repair and migration steps. The source CSVs
and figures still have the historical labels until migrated.

The suite attempted 26 steps: 22 completed, 02 and 08 failed, 06 timed out,
and 13 was skipped. The roughly 45-minute runtime includes the timeout.
The job's failed status does not invalidate its completed outputs.

- **09:** the interpolation spike is primarily generalized aliasing. At 16
  samples and width 16, mean test MSE is 5558.49, generalized aliasing is
  5557.94, model insufficiency is 0.5514, and data insufficiency is effectively
  zero. The earlier statement assigning the spike to data/algorithm error
  was a labeling error in the shared decomposition.
- **08:** the CSVs are usable after migration. For ten features, zero pair
  co-activation and 128 samples, increasing width from two to four lowers
  contrast-task test MSE from 0.2004 to 0.01282 and generalized aliasing from
  about 0.002989 to 0.000404. Most of the error is model insufficiency. This
  comparison alone does not establish a learned-superposition mechanism.
  At maximal pair co-activation the contrast target is identically zero.
- **12:** identical weight singular spectra still give different task
  recoverability. The contrast task's representation Bayes risk is 0.4 for
  the duplicate dictionary and zero for its rotated counterpart.
- **10 and 11:** observation-channel and collision Bayes-risk results survive.
  Their GAD columns require migration. The balanced exact-collision control
  has zero data insufficiency, not necessarily zero generalized aliasing.
- **01/02:** the original modeled/unmodeled linear decomposition uses the
  correct definitions. **03** records readout-null diagnostics, not a complete
  GAD decomposition. Older pseudoinverse plots are conditioning diagnostics;
  the GLM plots do not yet implement a verified KL/Bregman decomposition.

These are small-seed survey results. Near interpolation, mean risk can be
dominated by extreme draws. Matching a total-risk identity validates arithmetic;
definition-specific controls are also required to validate the named terms.

No retraining is needed to relabel saved helper CSVs. The audit contains
commands for copying them to a corrected directory and recovering figures.
