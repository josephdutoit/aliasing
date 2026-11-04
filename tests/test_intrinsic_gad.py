import numpy as np
import pytest

from aliasing.intrinsic_gad import (
    GAD_DEFINITION,
    SingularDesignError,
    gauge_transform,
    independent_residual_control,
    interpolating_population_gad,
    intrinsic_kernel,
    noninterpolating_risk,
    population_linear_oracle,
    weighted_min_norm_readout,
)
from aliasing.static_geometry import finite_support_gad


def test_population_oracle_uses_exact_affine_raw_moments():
    W = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    mean = np.array([0.3, 0.2, -0.1])
    covariance = np.array([[1.0, 0.1, 0.0], [0.1, 0.8, 0.05], [0.0, 0.05, 0.9]])
    second = covariance + np.outer(mean, mean)
    target = np.array([0.5, -0.3, 0.2])
    result = population_linear_oracle(W, second, target, mean_x=mean, target_intercept=0.7)
    sigma = result["Sigma"]
    cross = result["feature_target_moment"]
    assert result["gad_definition"] == GAD_DEFINITION
    assert np.allclose(sigma @ result["theta_star"], cross, atol=1e-12, rtol=1e-12)
    residual = result["target_second_moment"] - float(cross @ result["theta_star"])
    assert np.isclose(result["M"], 0.5 * residual, atol=1e-12, rtol=1e-12)
    assert result["is_oracle_exact"] is True


def test_interpolating_population_identity_and_signed_readout_identity():
    rng = np.random.default_rng(29)
    sigma = np.array([[1.5, 0.2, 0.1, 0.0], [0.2, 1.1, 0.0, 0.1],
                      [0.1, 0.0, 0.9, 0.05], [0.0, 0.1, 0.05, 1.2]])
    H = rng.normal(size=(3, 4))
    theta_star = rng.normal(size=4)
    y = rng.normal(size=3)
    result = interpolating_population_gad(H, y, sigma, theta_star, 0.17)
    assert np.linalg.norm(H @ result["theta_circle"] - y) < 1e-10
    assert np.allclose(sigma @ (result["theta_circle"] - theta_star),
                       H.T @ np.linalg.solve(result["K"], y - H @ theta_star),
                       atol=1e-11, rtol=1e-11)
    assert np.allclose(H @ result["theta_e"], y, atol=1e-11, rtol=1e-11)
    assert np.allclose(result["theta_e"], H.T @ np.linalg.solve(H @ H.T, y),
                       atol=1e-11, rtol=1e-11)
    assert np.isclose(result["S_e"], result["D_e"] - result["D0"], atol=1e-11, rtol=1e-11)
    assert np.isclose(result["S_e"], result["S_e_identity"], atol=1e-11, rtol=1e-11)
    assert result["rel_decomposition_error"] < 1e-10
    assert result["diagnostics"]["status"] == "OK"


def test_v3_aliasing_and_data_insufficiency_counterexamples():
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    sigma = np.eye(3)

    # Label noise leaks into a sampled direction, but the oracle has no
    # component in the sample nullspace: A > 0 and D0 = 0.
    realized_alias = interpolating_population_gad(
        H, np.array([1.0, 0.5]), sigma, np.array([1.0, 0.0, 0.0]), 0.0
    )
    assert realized_alias["A"] > 0.0
    assert realized_alias["D0"] < 1e-20

    # The labels exactly match the oracle, while its third feature is unseen:
    # A = 0 and D0 > 0.
    data_limited = interpolating_population_gad(
        H, np.array([1.0, 0.0]), sigma, np.array([1.0, 0.0, 2.0]), 0.0
    )
    assert data_limited["A"] < 1e-20
    assert np.isclose(data_limited["D0"], 2.0, atol=1e-12, rtol=1e-12)

    projector = np.linalg.solve(sigma, H.T) @ np.linalg.solve(
        realized_alias["K"], H
    )
    assert np.allclose(realized_alias["theta_circle"] - realized_alias["theta_0"],
                       (np.eye(3) - projector) @ np.array([1.0, 0.0, 0.0]))


def test_intrinsic_kernel_uses_solve_and_rejects_singular_design():
    sigma = np.diag([1.0, 2.0, 4.0])
    H = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]])
    K, diagnostics = intrinsic_kernel(H, sigma)
    assert np.allclose(K, H @ np.linalg.solve(sigma, H.T))
    assert diagnostics["rank_H"] == 2
    assert diagnostics["solver_path"] == "raw_design_svd_rank_cholesky_whitened_kernel"
    with pytest.raises(SingularDesignError):
        intrinsic_kernel(np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), sigma)


def test_noninterpolating_risk_has_no_interpolating_components():
    H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([1.0, 2.0, 2.5])
    theta_star = np.array([0.25, -0.1])
    theta_hat = np.linalg.lstsq(H, y, rcond=None)[0]
    result = noninterpolating_risk(H, y, np.eye(2), theta_star, theta_hat, 0.2)
    assert result["R_direct"] >= result["M"]
    assert result["is_interpolating"] is False
    for name in ("A", "D0", "D_e", "S_e"):
        assert name not in result


def test_gauge_transform_preserves_kernel_aliasing_and_population_risk():
    rng = np.random.default_rng(31)
    Sigma = np.array([[1.5, 0.2, 0.0], [0.2, 0.8, 0.1], [0.0, 0.1, 1.1]])
    H = rng.normal(size=(2, 3))
    features = rng.normal(size=(9, 3))
    theta_star = rng.normal(size=3)
    y = rng.normal(size=2)
    G = np.array([[1.7, 0.2, 0.0], [0.0, 0.8, -0.1], [0.1, 0.0, 1.2]])
    original = interpolating_population_gad(H, y, Sigma, theta_star, 0.4)
    transformed = gauge_transform(G, features, H, Sigma, theta_star, A=original["A"])
    changed = interpolating_population_gad(
        transformed["H"], y, transformed["Sigma"], transformed["theta_star"], 0.4
    )
    assert np.allclose(original["K"], changed["K"], atol=1e-10, rtol=1e-10)
    for metric in ("A", "D0", "M"):
        assert np.isclose(original[metric], changed[metric], atol=1e-10, rtol=1e-10)
    assert np.isclose(original["M"] + original["A"], changed["M"] + changed["A"],
                      atol=1e-10, rtol=1e-10)
    assert np.allclose(transformed["features"], features @ G.T)


def test_independent_residual_draws_match_trace_inverse_expectation():
    K = np.diag([0.7, 1.3, 2.2])
    pool = np.array([-1.0, 1.0])
    draws = independent_residual_control(K, pool, np.random.default_rng(43), draws=3000)["A_ind_draws"]
    expected = 0.5 * np.mean(pool**2) * np.trace(np.linalg.solve(K, np.eye(3)))
    standard_error = float(np.std(draws, ddof=1) / np.sqrt(draws.size))
    assert abs(float(np.mean(draws)) - expected) < 5.0 * standard_error


def test_gaussian_wishart_mean_away_from_singularity():
    rng = np.random.default_rng(47)
    dimension, n, noise_sigma, trials = 10, 3, 0.4, 800
    values = []
    for _ in range(trials):
        H = rng.normal(size=(n, dimension))
        y = rng.normal(scale=noise_sigma, size=n)
        result = interpolating_population_gad(H, y, np.eye(dimension), np.zeros(dimension), 0.0)
        values.append(result["A"])
    expected = 0.5 * noise_sigma**2 * n / (dimension - n - 1)
    standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    assert abs(float(np.mean(values)) - expected) < 5.0 * standard_error + 1e-3


def test_gaussian_data_insufficiency_expectation():
    rng = np.random.default_rng(48)
    dimension, n, trials = 9, 3, 600
    theta_star = rng.normal(size=dimension)
    values = []
    for _ in range(trials):
        H = rng.normal(size=(n, dimension))
        result = interpolating_population_gad(H, H @ theta_star, np.eye(dimension), theta_star, 0.0)
        values.append(result["D0"])
    expected = 0.5 * (1.0 - n / dimension) * float(theta_star @ theta_star)
    standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    assert abs(float(np.mean(values)) - expected) < 5.0 * standard_error + 1e-3


def test_definition_specific_counterexamples_cover_both_directions():
    # GAD-v2 meanings are intentionally tested in their historical API.
    h_alias = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    target_alias = np.array([1.0, 1.0, -1.0])
    alias = finite_support_gad(h_alias[:2], target_alias[:2], h_alias, target_alias,
                               np.ones(3) / 3)
    assert alias["aliasing"] > 1.0
    assert alias["data_insufficiency"] < 1e-20

    h_data = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target_data = np.array([1.0, -2.0])
    support = np.eye(3)
    target_support = support @ np.array([1.0, -2.0, 3.0])
    data = finite_support_gad(h_data, target_data, support, target_support, np.ones(3) / 3)
    assert data["data_insufficiency"] > 1.0
    assert data["aliasing"] < 1e-20


def test_weighted_minimum_norm_readout_interpolates():
    H = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    y = np.array([2.0, -1.0])
    Q = np.diag([1.0, 2.0, 4.0])
    theta = weighted_min_norm_readout(H, y, Q)
    assert np.allclose(H @ theta, y, atol=1e-11, rtol=1e-11)
