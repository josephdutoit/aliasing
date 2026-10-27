# Learned aliasing hypotheses: run results

This is a hypothesis test, not automatic confirmation. Frozen-map GAD uses an independent finite reference head.
See configuration.json, geometry.csv, paired_contrasts.csv and paired_profile_contrasts.csv.

## Selected full-readout results

Task profile balanced; readout n=128. MSEs are averaged across all task scores, whose analytic variances are one.

| Active features | Method | Test MSE | Model insufficiency | Aliasing | Data insufficiency | Residual gain |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | learned_linear | 0.74261 | 0.71953 | 0.024598 | 1.1293e-31 | 0.18078 |
| 1 | learned_relu | 0.00044647 | 0.00032394 | 0.0001243 | 6.5082e-30 | 0.49606 |
| 1 | linear_oracle | 0.74503 | 0.71021 | 0.038409 | 1.4874e-31 | 0.22893 |
| 1 | random_linear | 0.77678 | 0.74014 | 0.039802 | 1.3176e-31 | 0.22747 |
| 1 | random_relu | 0.27312 | 0.22489 | 0.049802 | 1.6149e-29 | 0.45599 |
| 2 | learned_linear | 0.74813 | 0.71792 | 0.03089 | 1.0241e-31 | 0.20543 |
| 2 | learned_relu | 0.33361 | 0.28722 | 0.048451 | 8.5982e-30 | 0.40436 |
| 2 | linear_oracle | 0.74286 | 0.70884 | 0.034701 | 1.3659e-31 | 0.21883 |
| 2 | random_linear | 0.77129 | 0.73715 | 0.035299 | 1.6026e-31 | 0.21659 |
| 2 | random_relu | 0.71154 | 0.60611 | 0.10907 | 2.25e-30 | 0.41995 |
| 4 | learned_linear | 0.74833 | 0.71786 | 0.031961 | 1.0815e-31 | 0.21055 |
| 4 | learned_relu | 0.52011 | 0.45193 | 0.071159 | 1.5207e-29 | 0.39497 |
| 4 | linear_oracle | 0.73763 | 0.70928 | 0.029957 | 1.2869e-31 | 0.20577 |
| 4 | random_linear | 0.76798 | 0.73717 | 0.032427 | 1.909e-31 | 0.20724 |
| 4 | random_relu | 0.83491 | 0.71804 | 0.12268 | 8.771e-31 | 0.4109 |
| 8 | learned_linear | 0.74765 | 0.72069 | 0.027969 | 1.1863e-31 | 0.19634 |
| 8 | learned_relu | 0.73374 | 0.63536 | 0.1029 | 3.3144e-06 | 0.40048 |
| 8 | linear_oracle | 0.73969 | 0.7126 | 0.027815 | 1.5893e-31 | 0.19691 |
| 8 | random_linear | 0.76771 | 0.73903 | 0.029534 | 1.9368e-31 | 0.19915 |
| 8 | random_relu | 0.89068 | 0.76963 | 0.12674 | 6.4314e-31 | 0.40457 |
| 12 | learned_linear | 0.75219 | 0.72421 | 0.029968 | 1.0987e-31 | 0.20239 |
| 12 | learned_relu | 0.82763 | 0.70822 | 0.1253 | 9.7137e-07 | 0.41922 |
| 12 | linear_oracle | 0.74395 | 0.71742 | 0.028693 | 1.2759e-31 | 0.19926 |
| 12 | random_linear | 0.76953 | 0.74185 | 0.029351 | 2.7179e-31 | 0.19824 |
| 12 | random_relu | 0.88897 | 0.77211 | 0.12261 | 6.6503e-31 | 0.39791 |
| 16 | learned_linear | 0.77994 | 0.75147 | 0.029326 | 1.146e-31 | 0.19701 |
| 16 | learned_relu | 21.195 | 0.75207 | 20.468 | 1.9529e-05 | 2.6609 |
| 16 | linear_oracle | 0.77819 | 0.74996 | 0.029616 | 1.2238e-31 | 0.19854 |
| 16 | random_linear | 0.77882 | 0.75142 | 0.029114 | 7.5346e-31 | 0.19642 |
| 16 | random_relu | 0.90192 | 0.78557 | 0.12294 | 7.2732e-31 | 0.39503 |

## Reading the tests

- H1: compare learned ReLU with learned linear AND the population linear oracle, at the same compression and expansion dimensions.
- H2: compare learned gates with orthogonal rotations after refitting heads; the linear rotation control must be invariant. The rate-matched variant changes biases, so only its weight spectrum, not its biased preactivation Gram, is preserved.
- H3: inspect threshold tradeoffs in test risk, model insufficiency, aliasing, activation rate and rank. Zero aliasing from an all-off map is a failure to retain information.
- H4: inspect paired A/B task-priority contrasts for the SAME task group. A single easy task or a change in task variance is not evidence of many-feature recovery.
- Near-boundary classification is measured separately from squared-risk GAD. Conditional component cross terms are retained.
- Gain is the realized target-residual RMS ratio; it is not the full generalized-aliasing operator norm. Blank gains have negligible denominators.
- No Marchenko-Pastur, spectral-filter, or universal superiority claim follows from these measurements.

## Figures

![sparsity_gad.png](sparsity_gad.png)

![residual_amplification.png](residual_amplification.png)

![rotation_intervention.png](rotation_intervention.png)

![threshold_intervention.png](threshold_intervention.png)

![task_priority_swap.png](task_priority_swap.png)

![decision_margin.png](decision_margin.png)
