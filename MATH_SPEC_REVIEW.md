# GAD-v3 `theta_0` equation review

The equations in the implementation brief are consistent. In particular, `theta_0` and `theta_circle` are not the same: `theta_circle` retains the part of the population oracle that lies in the sample nullspace.

Let

```text
P = Sigma^-1 H.T K^-1 H
theta_0 = Sigma^-1 H.T K^-1 y
```

Then `P` is the population-metric projection onto the row space identified by `H`, and

```text
theta_circle = theta_0 + (I-P) theta_star
D0 = 0.5 ||(I-P) theta_star||_Sigma^2
```

Thus `D0` measures samplewise data insufficiency. It can be nonzero when `A=0` (noiseless labels with oracle signal in an unobserved feature direction), while residual aliasing `A` can be nonzero with `D0=0` (an oracle already in the sampled feature row space plus label noise). These are covered by explicit v3 tests.

For a Gaussian isotropic design in a `d`-dimensional readout feature space, `E[D0 | W] = 0.5 (1-n/d) ||theta_star||_Sigma^2`. The implementation records this analytic prediction in Lane A and compares it with observed D0, alongside the finite-mean prediction for A when `d > n+1`.

The risk columns distinguish three fitted readouts. `R_oracle_nearest_interpolant = M + A` is the risk of `theta_circle`, the interpolant closest to the population oracle in the `Sigma` metric. `R_metric_min_norm = M + A + D0` is the risk of `theta_0`, which minimizes `theta.T @ Sigma @ theta` among interpolants. `R_euclidean_min_norm` is the direct population risk of `theta_e`, equal to `M + A + D_e` when the population oracle is exact.

The signed identity remains
`D_e-D0 = -<theta_circle-theta_0, theta_e-theta_0>_Sigma + 0.5||theta_e-theta_0||_Sigma^2`.
The `population_metric_v3` tag identifies this API, which stays separate from all legacy GAD-v2 definitions.
