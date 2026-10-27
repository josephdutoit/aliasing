# GAD mechanism checks: results

Completed 144/144 configurations; failures: 0.
Compression 4, expansion 16; profile balanced.

## Experiment 1: readout and reference stability

Independent readout n=128. Model terms use the largest independent reference fit.

| Active features | Map | Estimator | Test MSE | MP model term | MP alias term | Estimator correction energy |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | learned_linear | mp | 0.740336 | 0.717486 | 0.0226429 | 0 |
| 1 | learned_linear | ridge_validation | 0.739286 | 0.717486 | 0.0226429 | 0.00106507 |
| 1 | learned_relu | mp | 0.000382794 | 0.000320124 | 6.24229e-05 | 0 |
| 1 | learned_relu | ridge_validation | 0.000382827 | 0.000320124 | 6.24229e-05 | 4.71777e-09 |
| 1 | linear_oracle | mp | 0.745102 | 0.709089 | 0.0358612 | 0 |
| 1 | linear_oracle | ridge_validation | 0.741771 | 0.709089 | 0.0358612 | 0.00267439 |
| 2 | learned_linear | mp | 0.746944 | 0.71713 | 0.0299189 | 0 |
| 2 | learned_linear | ridge_validation | 0.745742 | 0.71713 | 0.0299189 | 0.000905465 |
| 2 | learned_relu | mp | 0.331321 | 0.285959 | 0.0454833 | 0 |
| 2 | learned_relu | ridge_validation | 0.328817 | 0.285959 | 0.0454833 | 0.00400859 |
| 2 | linear_oracle | mp | 0.73939 | 0.708327 | 0.0311194 | 0 |
| 2 | linear_oracle | ridge_validation | 0.737364 | 0.708327 | 0.0311194 | 0.00253242 |
| 4 | learned_linear | mp | 0.744773 | 0.71636 | 0.0284856 | 0 |
| 4 | learned_linear | ridge_validation | 0.743785 | 0.71636 | 0.0284856 | 0.000450609 |
| 4 | learned_relu | mp | 0.520973 | 0.449936 | 0.0711812 | 0 |
| 4 | learned_relu | ridge_validation | 0.511732 | 0.449936 | 0.0711812 | 0.0116353 |
| 4 | linear_oracle | mp | 0.736282 | 0.708379 | 0.0280854 | 0 |
| 4 | linear_oracle | ridge_validation | 0.734785 | 0.708379 | 0.0280854 | 0.00254286 |
| 8 | learned_linear | mp | 0.746693 | 0.719534 | 0.0270292 | 0 |
| 8 | learned_linear | ridge_validation | 0.745784 | 0.719534 | 0.0270292 | 0.000226014 |
| 8 | learned_relu | mp | 0.732287 | 0.632712 | 0.0999551 | 0 |
| 8 | learned_relu | ridge_validation | 0.696915 | 0.632712 | 0.0999551 | 0.040273 |
| 8 | linear_oracle | mp | 0.739079 | 0.711864 | 0.0270833 | 0 |
| 8 | linear_oracle | ridge_validation | 0.737726 | 0.711864 | 0.0270833 | 0.00222779 |
| 12 | learned_linear | mp | 0.751283 | 0.72334 | 0.0280888 | 0 |
| 12 | learned_linear | ridge_validation | 0.750296 | 0.72334 | 0.0280888 | 0.000218641 |
| 12 | learned_relu | mp | 0.840545 | 0.705889 | 0.135163 | 0 |
| 12 | learned_relu | ridge_validation | 0.761588 | 0.705889 | 0.135163 | 0.0811159 |
| 12 | linear_oracle | mp | 0.743273 | 0.716583 | 0.0268821 | 0 |
| 12 | linear_oracle | ridge_validation | 0.741905 | 0.716583 | 0.0268821 | 0.0022988 |
| 16 | learned_linear | mp | 0.779089 | 0.750297 | 0.0290921 | 0 |
| 16 | learned_linear | ridge_validation | 0.777962 | 0.750297 | 0.0290921 | 0.000612881 |
| 16 | learned_relu | mp | 21.8408 | 0.749232 | 21.0943 | 0 |
| 16 | learned_relu | ridge_validation | 0.795615 | 0.749232 | 21.0943 | 21.0773 |
| 16 | linear_oracle | mp | 0.778277 | 0.749584 | 0.0290719 | 0 |
| 16 | linear_oracle | ridge_validation | 0.776288 | 0.749584 | 0.0290719 | 0.00238025 |

The MP alias term remains the canonical MP baseline even for ridge rows. Ridge changes prediction through the estimator correction AND signed cross terms; do not interpret the baseline alias term as ridge leakage.

## Experiment 2: encoder/decoder identification

| Active features | Arm | Native test MSE | Independent MP test MSE at largest n |
| ---: | --- | ---: | ---: |
| 1 | joint_fresh | 0.000154366 | 0.000103635 |
| 1 | fixed_random_encoder | 0.0196263 | 0.0134972 |
| 1 | frozen_learned_encoder | 0.00912769 | 0.00790491 |
| 2 | joint_fresh | 0.295348 | 0.284359 |
| 2 | fixed_random_encoder | 0.484163 | 0.478781 |
| 2 | frozen_learned_encoder | 0.371024 | 0.357149 |
| 4 | joint_fresh | 0.465732 | 0.454399 |
| 4 | fixed_random_encoder | 0.662163 | 0.65739 |
| 4 | frozen_learned_encoder | 0.508628 | 0.495706 |
| 8 | joint_fresh | 0.638519 | 0.633365 |
| 8 | fixed_random_encoder | 0.733619 | 0.729145 |
| 8 | frozen_learned_encoder | 0.650624 | 0.643687 |
| 12 | joint_fresh | 0.712354 | 0.707899 |
| 12 | fixed_random_encoder | 0.739263 | 0.735876 |
| 12 | frozen_learned_encoder | 0.711524 | 0.707604 |
| 16 | joint_fresh | 0.752301 | 0.75139 |
| 16 | fixed_random_encoder | 0.754358 | 0.752583 |
| 16 | frozen_learned_encoder | 0.752482 | 0.751264 |

Joint-fresh versus fixed-random-encoder is the matched-start/matched-update comparison. The frozen-learned-encoder arm has additional pretraining and is a secondary diagnostic.

## Experiment 3: controlled perturbations

Shared Gaussian and quadratic probes impose the same perturbations across maps. Reference-residualized quadratic probes instead change residual direction with each map. Exact operator sensitivities refer to K=H_P H_T^+, not the full omitted-feature GAD operator.

See probes_paired.csv for paired seed contrasts in gain, isotropic prediction energy, and worst training-RMS gain. No automatic confirmation is inferred.

## Controls

```json
{
  "frozen_encoder_error": 0.0,
  "shared_decoder_initialization_error": 0.0,
  "zero_perturbation_error": 0.0,
  "perturbation_superposition_relative_error": 7.356205264222466e-13,
  "prediction_identity_relative_error": 7.092952739975338e-08,
  "risk_identity_relative_error": 9.090980253774329e-06,
  "risk_identity_scaled_error": 1.6848410339228569e-09,
  "risk_identity_absolute_error": 4.9284400894578084e-05,
  "identity_cancellation_ratio": 242467065428.62683,
  "rotation_gram_error": 8.102444855176067e-16,
  "rotation_spectrum_error": 6.662471122268034e-16,
  "linear_rotation_prediction_error": 1.839191160062522e-14
}
```

## Interpretation limits

- Reference convergence is empirical; the largest finite fit is not exact population truth.
- Representation-training fits reuse representation-training data; independent fits use separate samples.
- Original native heads and 128-sample probes have different data budgets; compare matching sizes as well.
- The fresh joint/frozen-random arms have matched updates and starts. The frozen learned encoder has extra pretraining.
- Perturbation directions, rotations, and readout draws are averaged within each representation seed.
- Intervals are pointwise exploratory intervals; unstable means should be read alongside seed_means tables.
- These checks do not establish Marchenko-Pastur calibration, a universal spectral filter, or boundary-specific protection.

## Figures

![readout_stability.png](readout_stability.png)

![reference_convergence.png](reference_convergence.png)

![encoder_decoder_ablation.png](encoder_decoder_ablation.png)

![controlled_perturbations.png](controlled_perturbations.png)
