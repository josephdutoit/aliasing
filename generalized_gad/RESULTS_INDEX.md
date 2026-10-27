# Uploaded figure index

These are historical figures already present in `generalized_gad/`.
See [README.md](README.md) for experimental design, limitations, and reproduction.
Regenerate this index after new runs with
`python generalized_gad/experiment_cli.py index` from the repository root.

## Embedded and sparse representation experiments

| Figure | Question explored |
| --- | --- |
| [Embedded complexity sweep](embedded_complexity_double_descent.png) | Change possible latent features at fixed width and sample count; chart shows pseudoinverse norm. |
| [Embedded mechanics](superposition_mechanics.png) | Compare latent feature Gram and active neuron Gram, plus pseudoinverse norm. |
| [Capacity by latent feature heatmap](aliasing_2d_phase_diagram.png) | Joint sweep of latent feature count and bottleneck width. |
| [Frozen random projection](superposition_aliasing_mitigation.png) | Compare linear and ReLU transforms with random weights. |
| [Learned superposition with bias](superposition_aliasing_bias_only.png) | Compare random and globally trained weights with learned bias. |
| [Noisy evaluation variant](corrected_superposition_aliasing.png) | Binary sparse training data, Gaussian noise added to evaluation data. |
| [No-bias variant](clean_superposition_aliasing.png) | Remove encoder bias. |
| [Regularization comparison](regularization_comparison_aliasing.png) | Compare autoencoder penalties. |

Additional uploaded historical figures include
[superposition_no_bias.png](superposition_no_bias.png),
[learned_superposition_aliasing.png](learned_superposition_aliasing.png).
Their specific code revisions cannot be inferred from PNG filenames alone.

## GLM GAD experiments

| Distribution | Sample sweep | Feature sweep |
| --- | --- | --- |
| Gaussian | [samplewise](gaussian_samplewise.png) | [featurewise](gaussian_featurewise.png) |
| Logistic | [samplewise](logistic_samplewise.png) | [featurewise](logistic_featurewise.png) |
| Poisson | [samplewise](poisson_samplewise.png) | [featurewise](poisson_featurewise.png) |
| Exponential | [samplewise](exponential_samplewise.png) | [featurewise](exponential_featurewise.png) |

Older prototypes and their plots live under [`ggad_old/`](ggad_old/).
