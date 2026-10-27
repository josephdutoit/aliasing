# GAD definitions audit

## Finding and source of truth

The exact identity in `MATHEMATICAL_RESULTS.md`, section 1, and its original
implementation `src/aliasing/linear.py` agree with the modeled/unmodeled block
definition in Transtrum et al., arXiv:2408.08294, equations 15 and 21.
The later `static_geometry.finite_support_gad` reversed the names of two terms.
The Chapter 2 plan repeated the nullspace terminology error. Checking only
reconstruction of total error did not detect this semantic reversal.

Let H_T and H_P be the modeled feature matrices, beta a fixed reference head,
and r_T=y_T-H_T beta, r_P=y_P-H_P beta the unmodeled target residuals. Write
B=H_T^+ H_T and fit beta_hat=H_T^+ y_T. The exact prediction-error identity is

    prediction - target
      = H_P(B-I)beta      [data insufficiency]
      + H_P H_T^+ r_T     [generalized aliasing]
      - r_P              [model insufficiency].

If noise is specified separately, its fitted contribution H_P H_T^+ epsilon
must also be reported separately, as in linear.py. Otherwise r_T includes
all target residuals and no attribution of them to noise versus misspecification
is justified. A caller-provided reference head must be specified and fixed
across the training-sample sweep; a fitted reference head is an estimated
comparator, not an oracle population optimum.

The squared component norms require all signed pairwise cross terms to recover
the squared prediction error. Coefficient-space energies and prediction risks
are distinct quantities. Neither a small pseudoinverse norm nor an exact risk
reconstruction establishes that the components have been named correctly.

## Schema correction

| Legacy finite_support_gad column | Correct meaning / version 2 column |
| --- | --- |
| aliasing | data_insufficiency |
| data_algorithm | aliasing |
| cross_model_alias | cross_model_data |
| cross_model_data | cross_model_alias |
| cross_alias_data | unchanged |
| alias_operator_rms | data_insufficiency_operator_frobenius |

The last quantity is a probability-weighted Frobenius norm of H_P(I-B), not
the generalized aliasing operator. The full label-independent aliasing operator
requires a specified unmodeled feature matrix: H_P H_T^+ U_T in prediction
space, or H_T^+ U_T in coefficient space. The helper's residual decomposition
computes a realized target-dependent aliasing vector without inventing U_T.
New outputs contain gad_definition_version=2. Historical results are preserved.

## Experiment audit

| Experiments | Assessment |
| --- | --- |
| 01 linear / 02 random features | The modeled/unmodeled partition and four error vectors in linear.py follow the stated identity, including fitted noise and cross terms. Prediction-space alias operator naming is documented. |
| 03 trained MLP | H_P(I-B) and the learned head's null response are readout-ambiguity diagnostics. They are not generalized aliasing. Current operator columns are renamed readout_null_operator_*; historical alias_operator_* columns from 03 retain their old numerical meaning. |
| 04 exact superposition | Collision probability and conditional target variance are valid representation diagnostics. They are not the GAD generalized-aliasing term. |
| 05--08 static / trained geometry | Their shared GAD terms were reversed. Collision and reconstruction metrics remain distinct. 08's maximum co-activation contrast target is identically zero. |
| 09 double descent | Its spike is primarily generalized aliasing after correcting the labels. |
| 10 noisy observation | The same reversal applies. Bayes risk of the observation channel remains a separate quantity. |
| 11 collision law | The Bayes-risk formula is unaffected. The assertion that balanced exact collisions must have zero aliasing was incorrect: the zero reference head implies zero data insufficiency, while finite-sample residual leakage may be nonzero. The assertion is corrected. |
| 12 matched spectra | The spectrum-matching and task-recoverability results survive; GAD component columns require the same relabeling. |
| 13 frozen OPT | The decomposition is relative to the independent reference head; it is not an oracle population decomposition. Corrected labels apply. No OPT result was produced in the uploaded run. |
| generalized_gad superposition variants | These mostly plot norm(pinv(H)), not a complete alias operator or realized GAD error. They are conditioning diagnostics until explicit residuals/unmodeled features and GAD components are added. |
| glm_gad_samplewise.py / glm_gad_featurewise.py | Euclidean coefficient proxies do not establish the KL/Bregman extension. See below. |
| ggad_old GLM prototypes | Unlike the two glm_gad scripts, these compute actual distribution divergences and optimize the constrained reference model. They need projection and equality checks; see below. |

## GLM issues requiring a separate implementation

The two glm_gad_* scripts call squared Euclidean coefficient distances Bregman divergences.
They generally are not the exponential-family KL divergence or held-out risk.
For Gaussian linear models with an identity population feature covariance,
coefficient distances can coincide with population prediction energies up to
scale; that does not justify their use for logistic, Poisson or exponential
responses. Setting omitted coefficients to zero does not generally compute the
KL-optimal restricted GLM. Euclidean nullspace augmentation is not a proof of a
Bregman projection or Pythagorean equality. Logistic separation can preclude a
finite MLE. Kish effective sample size is not the rank of the weighted design;
crossing n_eff=width is not an exact singularity criterion. Fisher conditioning
can be informative but does not repair these missing projection/divergence
definitions.

The ggad_old prototypes require a different assessment: logistic computes
Bernoulli KL, Poisson computes Poisson KL, and exponential computes the
Itakura-Saito divergence between exponential means. They optimize the model
over beta_hat + ker(X_train), using a finite prediction set to define the
divergence. This is substantially closer to the intended projection definition.
For unclipped logistic and Poisson natural parameters, that constraint is an
affine exponential-family set, so the appropriate projection can support a
Pythagorean equality. But the scripts do not record the equality residual,
constraint residual or projected-gradient convergence; clipping and finite
optimization can invalidate an asserted exact numerical equality. For the
exponential prototype, log-mean is not the natural parameter (-1/mean), so
affinity in coefficients does not automatically give the required exponential
flatness. These prototypes should be checked rather than discarded as coefficient
distance proxies. Their current setup also uses a realizable full model and does
not test a nonzero model-insufficiency projection.

The audit deliberately does not replace these projection choices or claim a
validated general GLM decomposition without the Chapter 1 assumptions.

## Corrected reading of the uploaded results

In 09, for 16 samples and width 16, mean test MSE was 5558.4933. Generalized
aliasing was 5557.9419, model insufficiency 0.55143, and data insufficiency
approximately 6.69e-31. This reverses the interpretation in the earlier review:
the double-descent experiment is consistent with a large GAD aliasing spike.

In 08, the old near-zero aliasing column was actually data insufficiency. For
10 possible features, zero pair co-activation, 128 samples and the contrast
task, widths 2 and 4 had test MSE 0.200402 and 0.012817, model insufficiency
0.197413 and 0.012413, and generalized aliasing 0.002989 and 0.000404.
Thus both model insufficiency and realized aliasing improved in this comparison.
It does not isolate a learned-superposition mechanism without matched linear
and random-ReLU controls, but it should not have been dismissed as zero aliasing.

## Recover historical results without training again

```bash
python experiments/relabel_gad_results.py \
  outputs/overnight/20261027T082310Z-14794630 \
  --output-dir outputs/gad_v2/20261027T082310Z-14794630
python experiments/replot_trained_superposition.py \
  --results-dir outputs/gad_v2/20261027T082310Z-14794630/08
python experiments/plot_near_collision_summary.py \
  outputs/gad_v2/20261027T082310Z-14794630/10
```

The migration copies CSVs, swaps the semantic column names and CI field names,
and records a migration manifest. It does not change numerical values or copy
incorrect old figures. It applies only to the legacy helper schema, not the
original linear/GLM metrics. Current plot helpers reject or lack the required
columns in legacy helper data, preventing silent interpretation of old labels.

## Verification

Definition-specific tests cover a full-rank design with nonzero generalized
aliasing and zero data insufficiency; an unobserved modeled direction with
nonzero data insufficiency and zero aliasing; and agreement term by term with
the original modeled/unmodeled linear implementation in both rank regimes.
These tests supplement the total-risk identity tests that previously passed
despite the reversed labels. Torch-dependent training is not revalidated by
those algebra tests.
