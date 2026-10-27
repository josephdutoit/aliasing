# Review of job 14794630

Source: `outputs/overnight/20261027T082310Z-14794630/`. The suite attempted
26 steps: 22 completed, 02 and 08 failed, 06 timed out, and 13 was skipped.
The roughly 45-minute runtime includes the 25-minute timeout; SLURM's failed
status reflects those unfinished steps, rather than a loss of the other results.
These are reduced survey settings, with small seed counts.

## Scientific findings

**12: weight spectra do not determine task recovery.** The duplicate and rotated
dictionaries have exactly matching singular spectra. At 128 readout samples,
the duplicate dictionary has representation Bayes risk 0.4 for the contrast
task, versus 0 for the rotated dictionary. Their test MSEs are approximately
0.404 and 0.103. For the common task, the duplicate dictionary's Bayes risk and
linear model insufficiency are essentially zero; the rotated dictionary has
zero Bayes risk but linear model insufficiency about 0.0855. This is a concrete
finite-support counterexample to inferring recoverability from weight spectra
alone. It does not disprove more conditional spectral theories or establish a
mechanism for HTSR. The saved maximum GAD risk-identity error is 9.94e-14.

**09: the interpolation peak is largely the data/algorithm component.** For
the isotropic, first-PC task with 16 training samples, mean test MSE at widths
8, 16, 24, 32, 40 is 3.10, 5558.49, 2.25, 1.15, 0.90. At width 16,
model insufficiency is about 0.55 and the data/algorithm component is 5557.94;
the training-null alias component is numerically zero. This distinction is
specific to the implemented GAD decomposition: a large interpolation error
does not imply a large training-null component. Near interpolation, the mean
over eight seeds can be dominated by ill-conditioned draws; inspect raw draws
and intervals before interpreting the height of the spike. The maximum saved
risk-identity error is 1.46e-10 (an absolute error).

**10 and 11: collisions, noise, and task identity determine irreducible loss.**
In 11, a collision mass 0.5 with balanced opposing targets yields Bayes risk
0.125, while a common target on the same representation yields zero. The exact
collision formula agrees to 1.39e-17 across the sweep. In 10, at balanced prior
and noise 0.1, separation 0, 0.03, 0.3, 1 gives exact Bayes risk approximately
0.25, 0.2445, 0.0493, 2.47e-7. At 128 samples the corresponding mean test MSEs
are 0.2533, 0.2512, 0.0776, 0.00989. Extra observations reduce estimation error;
they do not remove a fixed observation channel's information loss.

**08: the numerical run succeeded; plotting used the wrong column name.** All
raw and aggregate CSVs were saved. The plotting key should be `test_mse`, not
`probe_test_mse`. Across its 72 aggregate rows, the alias component is at most
2.78e-31. Readout sample counts are 16, 64, 128, while bottleneck widths are 2
and 4; this run therefore does not establish mitigation of training-null
aliasing. For ten possible features, zero pair co-activation, and 128 samples,
increasing width from 2 to 4 changes contrast-task test MSE from 0.2004 to
0.01282. Model insufficiency changes from 0.1974 to 0.01241. This supports a
task-dependent effect of learned representation capacity, without identifying
superposition or ReLU as the cause; matched baselines are still needed.

At joint activity 0.2 with marginal 0.2, the correlated features are always
equal. Their contrast target is identically zero. Its zero error is a trivial
target result, not evidence of successful feature separation. More generally,
changing co-activation changes the contrast target variance too. Compare
normalized risks and task variance before interpreting sparsity curves.

**07 and 05** provide descriptive static dictionary and co-activation comparisons.
The same zero-target caveat applies to 07 at maximal pair co-activation.
They should be interpreted with 12's explicit task comparison, rather than
as evidence that coherence or feature count alone determines risk.

**The older generalized_gad plots** remain pseudoinverse-norm diagnostics.
The new learned-bias figure often has a larger pseudoinverse norm than its
linear counterpart. It does not show a universal improvement from ReLU.
A norm reduction cannot distinguish beneficial suppression from discarded
singular directions without rank and task-error measurements. These runs do
not identify an MP bulk, demonstrate spectral shearing, or test the proposed
bias-to-bulk-edge scaling law.

**01 and the GLM runs** are calibration experiments for sample/feature sweeps.
**03** follows a trained MLP across checkpoints and is more relevant to the
dynamics chapter. Its paired-width CSV pools checkpoints: its 48 pairs are
not 48 independent seeds. Avoid interpreting those intervals as independent
seed-level evidence without grouping by seed or fixing a checkpoint.
**06 and 13** supply no new scientific result from this job.

## Repairs and next run

The branch fixes 08's plotting key, explicitly launches both sweeps for 02,
reduces the overnight 06 codebook sizes from 8/12 features to 8/10, and saves
06's raw rows after each fit. This reduces the cost of exact support diagnostics
and retains completed fits if it times out again. Original 06 defaults remain
available outside the overnight profile. The timeout's precise bottleneck was
not profiled, so completion within 25 minutes is not guaranteed.

Recover 08's figure from its existing CSV without retraining:

```bash
.venv/bin/python experiments/replot_trained_superposition.py \
  --results-dir outputs/overnight/20261027T082310Z-14794630/08
```

The recovery plot shows all co-activation settings and both widths, so the
trivial maximal-co-activation contrast is visible alongside other tasks.
It does not rewrite the historical job's failure status or ZIP archive.

Run only the unfinished computational experiments in another allocated job:

Set the experiment list in the shell so SLURM's export parser does not split it:

```bash
export GAD_ONLY=02,02-featurewise,06
export GAD_STEP_MINUTES=60
sbatch overnight.sbatch
```

The strongest next scientific test is a controlled active-count sweep with a
fixed feature universe, signal scale, task variance, sample count and width,
comparing trained linear, random ReLU and trained ReLU representations. Include
readout sample counts below and above measured rank. Record target error,
effective rank, lost target energy and actual GAD terms together.
