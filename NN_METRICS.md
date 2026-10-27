# Neural-network diagnostics

## Risk and hidden-feature geometry

- `train_mse`, `test_mse`: empirical training risk and held-out squared risk.
- `hidden_rank`, `hidden_singular_value_max`, `hidden_singular_value_min`, `hidden_condition_log10`: numerical rank and singular-value geometry of the training hidden-feature matrix. Rank uses the same scale-relative tolerance as the Moore-Penrose inverse.
- `activation_pattern_correlation_max`: largest absolute correlation between centered hidden-unit activation patterns across examples. This measures similarity of unit activity; it is **not** a direct measure of latent feature directions in superposition.
- `feature_effective_rank`: participation ratio of the singular spectrum of the test hidden-feature matrix.
- `ntk_drift`: relative Frobenius drift of the empirical tangent kernel from initialization.
- `feature_drift`, `feature_transport_train`, `feature_transport_test`: parameter and representation movement from initialization.

## Frozen-representation readout ambiguity

At a checkpoint, let \(Z_{\rm tr}\) and \(Z_{\rm te}\) be the current train
and test hidden-feature matrices, \(a\) the output head, and
\(Q=I-Z_{\rm tr}^\dagger Z_{\rm tr}\) the projector onto the training
readout nullspace. Then \(Z_{\rm tr}Qa=0\), while the test prediction
difference between \(a\) and \((I-Q)a\) is exactly \(Z_{\rm te}Qa\).

- `head_null_fraction`: \(\|Qa\|/\|a\|\).
- `test_null_response_rms`: RMS test prediction difference \(Z_{\rm te}Qa\) for the learned head.
- `test_null_response_relative`: that RMS divided by the RMS prediction from the learned head.
- `alias_operator_norm`, `initial_alias_operator_norm`, `alias_operator_drift`: unnormalized spectral norms for the current/initial test-visible train-null operators and their difference.
- `alias_operator_rms_norm`, `initial_alias_operator_rms_norm`, `alias_operator_rms_drift`: the corresponding norms divided by \(\sqrt{n_{\rm test}}\), making their scale comparable across test-set sizes.
- `head_minimum_norm_gap`: \(\|a-Z_{\rm tr}^\dagger y\|\). It equals the null-head norm only when the model interpolates; otherwise it also includes a row-space fit residual.
- `interpolates`: whether training MSE is below the configured threshold.

These are exact diagnostics conditional on a frozen hidden map. They do not
equal total test risk, and they do not establish that hidden units encode
features in superposition. The controlled dictionary experiment in
`experiments/04_superposition_aliasing.py` measures feature-direction overlap
and representation-level aliasing separately.

The complete definitions and proofs are in
[`MATHEMATICAL_RESULTS.md`](MATHEMATICAL_RESULTS.md).

Earlier committed result tables call `activation_pattern_correlation_max`
`feature_coherence_max`; those values should be interpreted as activation
pattern correlations, not as superposition measurements.
