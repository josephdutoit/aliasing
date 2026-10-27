"""Names and entry points for the existing generalized GAD experiments."""

EXPERIMENTS = {
    "glm-samplewise": ("glm_gad_samplewise.py", "GLM: vary sample count"),
    "glm-featurewise": ("glm_gad_featurewise.py", "GLM: vary modeled feature count"),
    "frozen-superposition": ("frozen_superposition.py", "Fixed random projection: linear versus ReLU"),
    "learned-superposition": ("superposition.py", "Globally trained sparse autoencoder with bias"),
    "gaussian-superposition": ("superposition_gaussian.py", "Learned superposition with Gaussian background"),
    "no-bias-superposition": ("superposition_no_bias.py", "Learned superposition without encoder bias"),
    "embedded-complexity": ("embedded_dd.py", "Sparse latent dimension sweep, fixed sample count"),
    "embedded-mechanics": ("embedded_superposition.py", "Latent weight Gram versus active representation Gram"),
    "embedded-regularization": ("superposition_full.py", "Embedded autoencoder regularization comparison"),
    "embedded-heatmap": ("aliasing_heatmap.py", "Capacity by latent dimension pseudoinverse map"),
}
