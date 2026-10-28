# Representation model benchmark

Completed 143/144 worlds; failures: 1.
Status: PARTIAL: conclusions require checking missing worlds.

Features p=16; input dimension=32; representation-training examples=2048; updates=2500.

## Model specifications

```json
[
  {
    "name": "bottleneck",
    "family": "bottleneck",
    "compression": 4,
    "width": 16
  },
  {
    "name": "mlp_wide",
    "family": "mlp",
    "widths": [
      16
    ]
  },
  {
    "name": "mlp_narrow",
    "family": "mlp",
    "widths": [
      4
    ]
  }
]
```

## Independent readouts

Readout n=128, profile=balanced. Means and medians are over seed-level averages of repeated draws.

| k | Architecture | Map | Head | Mean test MSE | Median test MSE | Model term | MP alias term |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | bottleneck | learned_linear | mp | 0.744774 | 0.741771 | 0.718029 | 0.0268276 |
| 1 | bottleneck | learned_linear | ridge_validation | 0.742343 | 0.740267 | 0.718029 | 0.0268276 |
| 1 | bottleneck | learned_relu | mp | 0.00297755 | 1.43841e-05 | 0.000213807 | 0.00276441 |
| 1 | bottleneck | learned_relu | ridge_validation | 0.00402943 | 1.44425e-05 | 0.000213807 | 0.00276441 |
| 1 | mlp_narrow | learned_linear | mp | 0.745099 | 0.742741 | 0.71841 | 0.0266553 |
| 1 | mlp_narrow | learned_linear | ridge_validation | 0.742756 | 0.740975 | 0.71841 | 0.0266553 |
| 1 | mlp_narrow | learned_relu | mp | 0.744959 | 0.744451 | 0.721511 | 0.0235515 |
| 1 | mlp_narrow | learned_relu | ridge_validation | 0.743897 | 0.743661 | 0.721511 | 0.0235515 |
| 1 | mlp_wide | learned_linear | mp | 0.00100729 | 3.56702e-30 | 4.13838e-30 | 4.07323e-30 |
| 1 | mlp_wide | learned_linear | ridge_validation | 0.00112574 | 4.84973e-12 | 4.13838e-30 | 4.07323e-30 |
| 1 | mlp_wide | learned_relu | mp | 0.00362374 | 2.12962e-07 | 0.000310669 | 0.0033169 |
| 1 | mlp_wide | learned_relu | ridge_validation | 0.00209306 | 2.1294e-07 | 0.000310669 | 0.0033169 |
| 1 | oracle_rank16 | population_linear_oracle | mp | 0.000859766 | 3.01213e-30 | 2.95562e-30 | 2.89742e-30 |
| 1 | oracle_rank16 | population_linear_oracle | ridge_validation | 0.00104366 | 1.84265e-12 | 2.95562e-30 | 2.89742e-30 |
| 1 | oracle_rank4 | population_linear_oracle | mp | 0.744789 | 0.743126 | 0.710593 | 0.0341702 |
| 1 | oracle_rank4 | population_linear_oracle | ridge_validation | 0.741698 | 0.739497 | 0.710593 | 0.0341702 |
| 2 | bottleneck | learned_linear | mp | 0.749145 | 0.746934 | 0.71885 | 0.0305214 |
| 2 | bottleneck | learned_linear | ridge_validation | 0.747869 | 0.746247 | 0.71885 | 0.0305214 |
| 2 | bottleneck | learned_relu | mp | 0.320118 | 0.312233 | 0.274801 | 0.0453706 |
| 2 | bottleneck | learned_relu | ridge_validation | 0.313526 | 0.311001 | 0.274801 | 0.0453706 |
| 2 | mlp_narrow | learned_linear | mp | 0.74951 | 0.749205 | 0.719108 | 0.0306892 |
| 2 | mlp_narrow | learned_linear | ridge_validation | 0.747737 | 0.747336 | 0.719108 | 0.0306892 |
| 2 | mlp_narrow | learned_relu | mp | 0.750295 | 0.749117 | 0.71955 | 0.0309137 |
| 2 | mlp_narrow | learned_relu | ridge_validation | 0.748888 | 0.747858 | 0.71955 | 0.0309137 |
| 2 | mlp_wide | learned_linear | mp | 3.27357e-30 | 2.78882e-30 | 2.57654e-30 | 2.50383e-30 |
| 2 | mlp_wide | learned_linear | ridge_validation | 4.16723e-12 | 3.90627e-12 | 2.57654e-30 | 2.50383e-30 |
| 2 | mlp_wide | learned_relu | mp | 1.68408e-06 | 1.79566e-06 | 1.51172e-06 | 1.74891e-07 |
| 2 | mlp_wide | learned_relu | ridge_validation | 1.68316e-06 | 1.79577e-06 | 1.51172e-06 | 1.74891e-07 |
| 2 | oracle_rank16 | population_linear_oracle | mp | 2.82797e-30 | 1.96769e-30 | 2.09761e-30 | 2.03748e-30 |
| 2 | oracle_rank16 | population_linear_oracle | ridge_validation | 1.54115e-12 | 1.52102e-12 | 2.09761e-30 | 2.03748e-30 |
| 2 | oracle_rank4 | population_linear_oracle | mp | 0.742935 | 0.740776 | 0.711502 | 0.0316361 |
| 2 | oracle_rank4 | population_linear_oracle | ridge_validation | 0.740738 | 0.739031 | 0.711502 | 0.0316361 |
| 4 | bottleneck | learned_linear | mp | 0.748674 | 0.748236 | 0.720586 | 0.0280875 |
| 4 | bottleneck | learned_linear | ridge_validation | 0.747705 | 0.747231 | 0.720586 | 0.0280875 |
| 4 | bottleneck | learned_relu | mp | 0.524689 | 0.52296 | 0.455419 | 0.0696846 |
| 4 | bottleneck | learned_relu | ridge_validation | 0.518233 | 0.517468 | 0.455419 | 0.0696846 |
| 4 | mlp_narrow | learned_linear | mp | 0.749324 | 0.748747 | 0.720861 | 0.0284707 |
| 4 | mlp_narrow | learned_linear | ridge_validation | 0.7481 | 0.747585 | 0.720861 | 0.0284707 |
| 4 | mlp_narrow | learned_relu | mp | 0.758089 | 0.750494 | 0.730076 | 0.0279801 |
| 4 | mlp_narrow | learned_relu | ridge_validation | 0.757267 | 0.749803 | 0.730076 | 0.0279801 |
| 4 | mlp_wide | learned_linear | mp | 4.38365e-30 | 3.4592e-30 | 4.32314e-30 | 4.24233e-30 |
| 4 | mlp_wide | learned_linear | ridge_validation | 4.04335e-12 | 3.83604e-12 | 4.32314e-30 | 4.24233e-30 |
| 4 | mlp_wide | learned_relu | mp | 0.00616758 | 2.53436e-05 | 0.00533275 | 0.000869521 |
| 4 | mlp_wide | learned_relu | ridge_validation | 0.00616661 | 2.53378e-05 | 0.00533275 | 0.000869521 |
| 4 | oracle_rank16 | population_linear_oracle | mp | 3.52579e-30 | 3.60587e-30 | 2.7057e-30 | 2.63767e-30 |
| 4 | oracle_rank16 | population_linear_oracle | ridge_validation | 1.47048e-12 | 1.48093e-12 | 2.7057e-30 | 2.63767e-30 |
| 4 | oracle_rank4 | population_linear_oracle | mp | 0.740031 | 0.739671 | 0.712846 | 0.0272643 |
| 4 | oracle_rank4 | population_linear_oracle | ridge_validation | 0.738671 | 0.738384 | 0.712846 | 0.0272643 |
| 8 | bottleneck | learned_linear | mp | 0.748802 | 0.752205 | 0.721837 | 0.0270548 |
| 8 | bottleneck | learned_linear | ridge_validation | 0.747887 | 0.751256 | 0.721837 | 0.0270548 |
| 8 | bottleneck | learned_relu | mp | 0.757781 | 0.75937 | 0.634443 | 0.123577 |
| 8 | bottleneck | learned_relu | ridge_validation | 0.697 | 0.695724 | 0.634443 | 0.123577 |
| 8 | mlp_narrow | learned_linear | mp | 0.749294 | 0.752193 | 0.721864 | 0.0275635 |
| 8 | mlp_narrow | learned_linear | ridge_validation | 0.747958 | 0.750476 | 0.721864 | 0.0275635 |
| 8 | mlp_narrow | learned_relu | mp | 0.791782 | 0.803037 | 0.766461 | 0.0254942 |
| 8 | mlp_narrow | learned_relu | ridge_validation | 0.790928 | 0.80264 | 0.766461 | 0.0254942 |
| 8 | mlp_wide | learned_linear | mp | 6.79626e-30 | 4.80243e-30 | 1.52515e-29 | 1.51602e-29 |
| 8 | mlp_wide | learned_linear | ridge_validation | 4.2856e-12 | 4.1726e-12 | 1.52515e-29 | 1.51602e-29 |
| 8 | mlp_wide | learned_relu | mp | 0.0513907 | 0.0153625 | 0.0461111 | 0.00532427 |
| 8 | mlp_wide | learned_relu | ridge_validation | 0.0513481 | 0.0153615 | 0.0461111 | 0.00532427 |
| 8 | oracle_rank16 | population_linear_oracle | mp | 3.27056e-30 | 3.06342e-30 | 3.18849e-30 | 3.09557e-30 |
| 8 | oracle_rank16 | population_linear_oracle | ridge_validation | 1.43626e-12 | 1.43854e-12 | 3.18849e-30 | 3.09557e-30 |
| 8 | oracle_rank4 | population_linear_oracle | mp | 0.739822 | 0.741517 | 0.713574 | 0.0263845 |
| 8 | oracle_rank4 | population_linear_oracle | ridge_validation | 0.738536 | 0.739622 | 0.713574 | 0.0263845 |
| 12 | bottleneck | learned_linear | mp | 0.753803 | 0.754441 | 0.726516 | 0.027301 |
| 12 | bottleneck | learned_linear | ridge_validation | 0.752877 | 0.7535 | 0.726516 | 0.027301 |
| 12 | bottleneck | learned_relu | mp | 0.826273 | 0.822519 | 0.706823 | 0.119387 |
| 12 | bottleneck | learned_relu | ridge_validation | 0.761858 | 0.761856 | 0.706823 | 0.119387 |
| 12 | mlp_narrow | learned_linear | mp | 0.753331 | 0.754295 | 0.726501 | 0.0268754 |
| 12 | mlp_narrow | learned_linear | ridge_validation | 0.752103 | 0.752866 | 0.726501 | 0.0268754 |
| 12 | mlp_narrow | learned_relu | mp | 0.851464 | 0.823476 | 0.830373 | 0.0210373 |
| 12 | mlp_narrow | learned_relu | ridge_validation | 0.850613 | 0.822353 | 0.830373 | 0.0210373 |
| 12 | mlp_wide | learned_linear | mp | 8.42926e-30 | 8.44087e-30 | 2.99389e-29 | 2.97968e-29 |
| 12 | mlp_wide | learned_linear | ridge_validation | 4.21329e-12 | 4.17818e-12 | 2.99389e-29 | 2.97968e-29 |
| 12 | mlp_wide | learned_relu | mp | 0.201462 | 0.167621 | 0.183729 | 0.0177138 |
| 12 | mlp_wide | learned_relu | ridge_validation | 0.201107 | 0.167201 | 0.183729 | 0.0177138 |
| 12 | oracle_rank16 | population_linear_oracle | mp | 3.321e-30 | 3.00236e-30 | 2.57081e-30 | 2.42252e-30 |
| 12 | oracle_rank16 | population_linear_oracle | ridge_validation | 1.44285e-12 | 1.43052e-12 | 2.57081e-30 | 2.42252e-30 |
| 12 | oracle_rank4 | population_linear_oracle | mp | 0.745943 | 0.745922 | 0.71906 | 0.0268885 |
| 12 | oracle_rank4 | population_linear_oracle | ridge_validation | 0.74458 | 0.744604 | 0.71906 | 0.0268885 |
| 16 | bottleneck | learned_linear | mp | 0.780276 | 0.780384 | 0.750492 | 0.0297882 |
| 16 | bottleneck | learned_linear | ridge_validation | 0.779178 | 0.779278 | 0.750492 | 0.0297882 |
| 16 | bottleneck | learned_relu | mp | 10.6871 | 1.60408 | 0.749769 | 9.94016 |
| 16 | bottleneck | learned_relu | ridge_validation | 0.809158 | 0.795731 | 0.749769 | 9.94016 |
| 16 | mlp_narrow | learned_linear | mp | 0.780355 | 0.780062 | 0.751073 | 0.0293662 |
| 16 | mlp_narrow | learned_linear | ridge_validation | 0.778397 | 0.778614 | 0.751073 | 0.0293662 |
| 16 | mlp_narrow | learned_relu | mp | 0.91636 | 0.922687 | 0.899143 | 0.0174583 |
| 16 | mlp_narrow | learned_relu | ridge_validation | 0.915577 | 0.922072 | 0.899143 | 0.0174583 |
| 16 | mlp_wide | learned_linear | mp | 1.3418e-29 | 1.36312e-29 | 3.317e-29 | 3.28941e-29 |
| 16 | mlp_wide | learned_linear | ridge_validation | 3.25591e-12 | 3.15491e-12 | 3.317e-29 | 3.28941e-29 |
| 16 | mlp_wide | learned_relu | mp | 387.637 | 0.407931 | 0.370266 | 387.273 |
| 16 | mlp_wide | learned_relu | ridge_validation | 4.3682 | 0.40713 | 0.370266 | 387.273 |
| 16 | oracle_rank16 | population_linear_oracle | mp | 3.44475e-30 | 3.18862e-30 | 3.1083e-30 | 2.64203e-30 |
| 16 | oracle_rank16 | population_linear_oracle | ridge_validation | 1.46545e-12 | 1.45889e-12 | 3.1083e-30 | 2.64203e-30 |
| 16 | oracle_rank4 | population_linear_oracle | mp | 0.77949 | 0.779385 | 0.750937 | 0.028634 |
| 16 | oracle_rank4 | population_linear_oracle | ridge_validation | 0.777527 | 0.777473 | 0.750937 | 0.028634 |

MP aliasing remains the Moore–Penrose baseline for ridge and native heads. Their changed predictions include an explicit estimator correction and all signed cross terms.

## Representation quality

Largest independent reference fit; this estimates population projection quality.

| k | Architecture | Reference model MSE | Readout rank | Output-subspace lower bound | Features with normalized MSE < 0.1 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | bottleneck | 0.000213807 | 17 | 0 | 16 |
| 1 | mlp_narrow | 0.721511 | 5 | 0.710684 | 1.08333 |
| 1 | mlp_wide | 0.000310669 | 16.9583 | 0 | 16 |
| 2 | bottleneck | 0.274801 | 17 | 0 | 4.91667 |
| 2 | mlp_narrow | 0.71955 | 5 | 0.710942 | 0.0416667 |
| 2 | mlp_wide | 1.51172e-06 | 17 | 0 | 16 |
| 4 | bottleneck | 0.455419 | 16.9583 | 0 | 3.95833 |
| 4 | mlp_narrow | 0.730076 | 4.91667 | 0.711573 | 0 |
| 4 | mlp_wide | 0.00533275 | 16.7917 | 0 | 15.875 |
| 8 | bottleneck | 0.634443 | 16.9565 | 0 | 3.04348 |
| 8 | mlp_narrow | 0.766461 | 4.69565 | 0.713741 | 0 |
| 8 | mlp_wide | 0.0461111 | 16.2174 | 0 | 13.6522 |
| 12 | bottleneck | 0.706823 | 17 | 0 | 0.208333 |
| 12 | mlp_narrow | 0.830373 | 3.79167 | 0.718646 | 0 |
| 12 | mlp_wide | 0.183729 | 14 | 0 | 6.375 |
| 16 | bottleneck | 0.749769 | 16.5417 | 0 | 0 |
| 16 | mlp_narrow | 0.899143 | 2.75 | 0.75 | 0 |
| 16 | mlp_wide | 0.370266 | 11.625 | 0 | 1.33333 |

## Native heads and larger independent readouts

Native and representation-training MP heads use the same 2048 examples. The final column uses a separate n=4096 readout.

| k | Architecture | Native test MSE | MP on representation-training data | Independent MP test MSE |
| ---: | --- | ---: | ---: | ---: |
| 1 | bottleneck | 0.000283132 | 0.000215093 | 0.000214811 |
| 1 | mlp_wide | 0.000319235 | 0.000313 | 0.000311417 |
| 1 | mlp_narrow | 0.722566 | 0.722275 | 0.722094 |
| 2 | bottleneck | 0.286603 | 0.276706 | 0.275833 |
| 2 | mlp_wide | 1.71364e-06 | 1.53086e-06 | 1.51763e-06 |
| 2 | mlp_narrow | 0.72103 | 0.720431 | 0.720377 |
| 4 | bottleneck | 0.469454 | 0.458706 | 0.457083 |
| 4 | mlp_wide | 0.00543052 | 0.00536443 | 0.00534733 |
| 4 | mlp_narrow | 0.731582 | 0.730934 | 0.730848 |
| 8 | bottleneck | 0.643098 | 0.639144 | 0.636706 |
| 8 | mlp_wide | 0.0464532 | 0.0462338 | 0.0462377 |
| 8 | mlp_narrow | 0.767935 | 0.767349 | 0.767175 |
| 12 | bottleneck | 0.714111 | 0.712011 | 0.709232 |
| 12 | mlp_wide | 0.184621 | 0.184118 | 0.184221 |
| 12 | mlp_narrow | 0.831715 | 0.831355 | 0.83093 |
| 16 | bottleneck | 0.753523 | 0.754103 | 0.752125 |
| 16 | mlp_wide | 0.373076 | 0.3706 | 0.371484 |
| 16 | mlp_narrow | 0.899989 | 0.899662 | 0.899885 |

## Reading the evidence

- Compare learned ReLU with frozen random ReLU to test representation learning, and with learned linear at the same architecture to test nonlinear benefit.
- The unrestricted wide linear model can solve these linear target tasks. Its population floor is zero; beating a compressed linear model alone does not establish an advantage over this control.
- A narrow final hidden layer has an output-subspace floor even with nonlinear features. The bottleneck model expands before its final readout.
- Use readouts_paired.csv and probes_paired.csv for within-seed contrasts. Intervals are pointwise and exploratory; inspect seed tables, maxima and top-five shares when means are unstable.
- Compare independent readouts at matching sample sizes. Native heads reuse representation-training data and have an explicit estimator correction.
- Reference convergence is empirical. The finite reference model term is an estimate, and its cross terms need not vanish.
- Sensitivities measure the specified MP readout operator or perturbation. Lower model error does not imply uniformly lower aliasing or noise sensitivity.
- Pure gate rotations preserve preactivation geometry. Activity-matched rotations additionally change biases; inspect activity_match_error for ties.
- A deeper custom MLP removes only its final gate in the no_final_gate intervention. The separately trained learned_linear arm removes every gate.
- These tests do not establish a Marchenko–Pastur filter, a heavy-tail mechanism, or preferential decision-boundary protection.

## Controls

```json
{
  "prediction_identity_relative_error": 1.9013195038414647e-08,
  "risk_identity_relative_error": 2.646484980162178e-06,
  "risk_identity_scaled_error": 8.850968569174e-10,
  "risk_identity_absolute_error": 0.0003007054328918457,
  "identity_cancellation_ratio": 164842376773.16373,
  "zero_perturbation_error": 0.0,
  "perturbation_response_relative_error": 5.277944747916763e-12,
  "rotation_gram_error": 8.462830660426787e-16,
  "rotation_spectrum_error": 7.527711111420343e-16,
  "linear_rotation_prediction_error": 7.473108378155114e-15,
  "activity_match_error": 0.0001220703125
}
```

## Figures

![architecture_comparison.png](architecture_comparison.png)

![readout_and_gad.png](readout_and_gad.png)

![gate_interventions.png](gate_interventions.png)

![reference_convergence.png](reference_convergence.png)

