"""Population-metric GAD-v3 for frozen linear readouts.

This module is intentionally separate from the repository's GAD-v2
reference-head decomposition. Risks use the half-squared-error convention in
the Chapter 2 v3 brief. Each design's numerical rank is decided once from its
float64 SVD; singular interpolating designs fail explicitly.
"""
from __future__ import annotations

from typing import Any

import numpy as np


GAD_DEFINITION = "population_metric_v3"


class SingularDesignError(np.linalg.LinAlgError):
    """Raised when a requested exact interpolation solve is not full row rank."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def _matrix(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite values")
    return result


def _vector(value: Any, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite values")
    return result


def _symmetric_spd(value: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    matrix = _matrix(value, name)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    scale = max(1.0, float(np.max(np.abs(matrix))))
    if not np.allclose(matrix, matrix.T, rtol=1e-11, atol=1e-12 * scale):
        raise ValueError(f"{name} must be symmetric")
    matrix = 0.5 * (matrix + matrix.T)
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    return matrix, factor


def _metric_squared(vector: np.ndarray, metric: np.ndarray) -> float:
    return float(vector @ metric @ vector)


def _solve_spd(matrix: np.ndarray, factor: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a symmetric positive-definite system through its Cholesky factor."""
    return np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))


def population_linear_oracle(
    W: Any,
    C_x: Any,
    target: Any,
    *,
    intercept: bool = True,
    mean_x: Any | None = None,
    target_intercept: float = 0.0,
) -> dict[str, Any]:
    """Compute the exact affine population oracle for ``m(x) = W x``.

    ``W`` has shape ``(features, input_dimension)`` and ``C_x`` is the raw
    *second moment* E[xx^T], not the centered covariance. ``mean_x`` defaults
    to zero. The target is ``target @ x + target_intercept``. When requested,
    the intercept is prepended to the feature vector and its moments are
    computed from the supplied raw moments.
    """
    weight = _matrix(W, "W")
    second = _matrix(C_x, "C_x")
    if second.shape[0] != second.shape[1] or second.shape[0] != weight.shape[1]:
        raise ValueError("C_x must be square and match W's input dimension")
    if not np.allclose(second, second.T, rtol=1e-11, atol=1e-12):
        raise ValueError("C_x must be symmetric")
    second = 0.5 * (second + second.T)
    mu = np.zeros(weight.shape[1], dtype=np.float64) if mean_x is None else _vector(
        mean_x, weight.shape[1], "mean_x"
    )
    beta = _vector(target, weight.shape[1], "target")
    intercept_value = float(target_intercept)
    if not np.isfinite(intercept_value):
        raise ValueError("target_intercept must be finite")

    feature_second = weight @ second @ weight.T
    feature_target = weight @ (second @ beta + intercept_value * mu)
    target_mean = float(beta @ mu + intercept_value)
    target_second = float(beta @ second @ beta + 2.0 * intercept_value * (beta @ mu)
                          + intercept_value**2)
    if intercept:
        sigma = np.empty((weight.shape[0] + 1, weight.shape[0] + 1), dtype=np.float64)
        sigma[0, 0] = 1.0
        sigma[0, 1:] = (weight @ mu)
        sigma[1:, 0] = sigma[0, 1:]
        sigma[1:, 1:] = feature_second
        cross = np.concatenate(([target_mean], feature_target))
    else:
        sigma = feature_second
        cross = feature_target

    sigma, factor = _symmetric_spd(sigma, "feature second moment Sigma")
    theta_star = _solve_spd(sigma, factor, cross)
    model_error_twice = target_second - 2.0 * float(theta_star @ cross) + _metric_squared(theta_star, sigma)
    error_scale = max(1.0, abs(target_second), abs(float(theta_star @ cross)))
    if model_error_twice < -1e-10 * error_scale:
        raise FloatingPointError("population oracle produced a negative squared model error")
    model_error_twice = max(0.0, model_error_twice)
    return {
        "Sigma": sigma,
        "theta_star": theta_star,
        "M": 0.5 * model_error_twice,
        "target_second_moment": target_second,
        "feature_target_moment": cross,
        "is_oracle_exact": True,
        "oracle_reference_size": "NA",
        "gad_definition": GAD_DEFINITION,
    }


def intrinsic_kernel(
    H: Any,
    Sigma: Any,
    rank_policy: str = "strict",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``K = H Sigma^-1 H.T`` and diagnostics using Cholesky solves.

    The strict policy requires full row rank. Singular designs are returned as
    a structured ``SingularDesignError``; no jitter, truncation, or implicit
    pseudoinverse is applied.
    """
    design = _matrix(H, "H")
    sigma, factor = _symmetric_spd(Sigma, "Sigma")
    if design.shape[1] != sigma.shape[0]:
        raise ValueError("H's feature dimension must match Sigma")
    if rank_policy != "strict":
        raise ValueError("rank_policy must be 'strict'; sensitivity solvers are separate analyses")

    # Use one rank decision on the raw design, matching the ordinary least-
    # squares path above the interpolation boundary. Since Sigma is SPD, this
    # is also the exact rank of H Sigma^-1/2. The whitened SVD is retained only
    # for the population-metric condition estimate, not a second rank test.
    _, raw_singular_values, _ = np.linalg.svd(design, full_matrices=False)
    largest_raw = float(raw_singular_values[0]) if raw_singular_values.size else 0.0
    tolerance = max(design.shape, default=0) * np.finfo(np.float64).eps * largest_raw
    rank = int(np.count_nonzero(raw_singular_values > tolerance))
    whitened = np.linalg.solve(factor, design.T).T
    _, singular_values, _ = np.linalg.svd(whitened, full_matrices=False)
    kernel = whitened @ whitened.T
    kernel = 0.5 * (kernel + kernel.T)
    condition = (float(singular_values[0] / singular_values[rank - 1])
                 if rank and singular_values.size >= rank else float("inf"))
    diagnostics = {
        "rank_H": rank,
        "n_rows": int(design.shape[0]),
        "d_features": int(design.shape[1]),
        "rank_tolerance": float(tolerance),
        "solver_path": "raw_design_svd_rank_cholesky_whitened_kernel",
        "rank_basis": "unwhitened_H",
        "condition_H_whitened": condition,
        "condition_K": condition**2 if np.isfinite(condition) else float("inf"),
        "interpolation_residual": "NA",
        "finite": bool(np.isfinite(kernel).all()),
    }
    if rank < design.shape[0]:
        diagnostics["status"] = "SINGULAR_DESIGN"
        raise SingularDesignError(
            f"design rank {rank} is below row count {design.shape[0]}",
            diagnostics,
        )
    diagnostics["status"] = "OK"
    return kernel, diagnostics


def interpolating_population_gad(
    H: Any,
    y: Any,
    Sigma: Any,
    theta_star: Any,
    oracle_risk: float,
) -> dict[str, Any]:
    """Evaluate the exact population-metric decomposition for a feasible fit.

    The caller must provide a full-row-rank design and positive-definite
    population second moment. The returned ``A``, ``D0``, ``D_e`` and signed
    ``S_e`` use the v3 definitions and the half-squared-risk convention.
    """
    design = _matrix(H, "H")
    target = _vector(y, design.shape[0], "y")
    sigma, sigma_factor = _symmetric_spd(Sigma, "Sigma")
    reference = _vector(theta_star, design.shape[1], "theta_star")
    model_risk = float(oracle_risk)
    if not np.isfinite(model_risk) or model_risk < 0:
        raise ValueError("oracle_risk must be finite and nonnegative")
    kernel, diagnostics = intrinsic_kernel(design, sigma)

    residual = target - design @ reference
    kernel_factor = np.linalg.cholesky(kernel)
    alpha = _solve_spd(kernel, kernel_factor, residual)
    sigma_inv_h_t = _solve_spd(sigma, sigma_factor, design.T)
    theta_circle = reference + sigma_inv_h_t @ alpha
    theta_0 = sigma_inv_h_t @ _solve_spd(kernel, kernel_factor, target)
    euclidean_gram = design @ design.T
    euclidean_factor = np.linalg.cholesky(0.5 * (euclidean_gram + euclidean_gram.T))
    theta_e = design.T @ _solve_spd(euclidean_gram, euclidean_factor, target)

    A = 0.5 * float(residual @ alpha)
    z = theta_circle - theta_0
    delta = theta_e - theta_0
    D0 = 0.5 * _metric_squared(z, sigma)
    D_e = 0.5 * _metric_squared(theta_e - theta_circle, sigma)
    signed_S = D_e - D0
    signed_identity = -float(z @ sigma @ delta) + 0.5 * _metric_squared(delta, sigma)
    direct = model_risk + 0.5 * _metric_squared(theta_e - reference, sigma)
    decomposition = model_risk + A + D_e
    interpolation_residual = float(np.linalg.norm(design @ theta_e - target))
    relative_identity = abs(direct - decomposition) / max(1.0, abs(direct), abs(decomposition))
    diagnostics = diagnostics | {
        "interpolation_residual": interpolation_residual,
        "condition_HH_T": float(np.linalg.cond(euclidean_gram)),
        "finite": bool(np.isfinite([A, D0, D_e, signed_S, direct]).all()),
        "gad_definition": GAD_DEFINITION,
    }
    return {
        "M": model_risk,
        "A": A,
        "D0": D0,
        "D_e": D_e,
        "S_e": signed_S,
        "S_e_identity": signed_identity,
        "signed_identity_error": abs(signed_S - signed_identity),
        "R_direct": direct,
        "R_decomposition": decomposition,
        "rel_decomposition_error": relative_identity,
        "residual_second_moment": float(np.mean(residual**2)),
        "theta_circle": theta_circle,
        "theta_0": theta_0,
        "theta_e": theta_e,
        "r": residual,
        "K": kernel,
        "diagnostics": diagnostics,
        "is_interpolating": True,
        "gad_definition": GAD_DEFINITION,
    }


def noninterpolating_risk(
    H: Any,
    y: Any,
    Sigma: Any,
    theta_star: Any,
    theta_hat: Any,
    oracle_risk: float,
) -> dict[str, Any]:
    """Return direct population risk when the interpolating split is inapplicable.

    This function intentionally does not return ``A``, ``D0``, ``D_e`` or
    ``S_e``. The general risk identity remains valid above the interpolation
    threshold; the three-term interpolating decomposition does not.
    """
    design = _matrix(H, "H")
    target = _vector(y, design.shape[0], "y")
    sigma, _ = _symmetric_spd(Sigma, "Sigma")
    reference = _vector(theta_star, design.shape[1], "theta_star")
    estimate = _vector(theta_hat, design.shape[1], "theta_hat")
    model_risk = float(oracle_risk)
    if not np.isfinite(model_risk) or model_risk < 0:
        raise ValueError("oracle_risk must be finite and nonnegative")
    excess = 0.5 * _metric_squared(estimate - reference, sigma)
    residual = design @ estimate - target
    return {
        "M": model_risk,
        "risk_excess": excess,
        "R_direct": model_risk + excess,
        "training_residual_mse": float(np.mean(residual**2)),
        "interpolation_residual": float(np.linalg.norm(residual)),
        "is_interpolating": False,
        "interpolating_components_applicable": False,
        "gad_definition": GAD_DEFINITION,
    }


def gauge_transform(
    G: Any,
    features: Any,
    H: Any,
    Sigma: Any,
    theta_star: Any,
    A: float | None = None,
) -> dict[str, Any]:
    """Apply an invertible feature-coordinate change ``m' = G m`` exactly."""
    transform = _matrix(G, "G")
    if transform.shape[0] != transform.shape[1]:
        raise ValueError("G must be square")
    sample_features = _matrix(features, "features")
    design = _matrix(H, "H")
    sigma = _matrix(Sigma, "Sigma")
    reference = _vector(theta_star, transform.shape[0], "theta_star")
    if sample_features.shape[1] != transform.shape[0] or design.shape[1] != transform.shape[0]:
        raise ValueError("feature and design dimensions must match G")
    if sigma.shape != transform.shape:
        raise ValueError("Sigma shape must match G")
    transformed_reference = np.linalg.solve(transform.T, reference)
    transformed_sigma = transform @ sigma @ transform.T
    return {
        "features": sample_features @ transform.T,
        "H": design @ transform.T,
        "Sigma": transformed_sigma,
        "theta_star": transformed_reference,
        "A": A,
        "gad_definition": GAD_DEFINITION,
    }


def independent_residual_control(
    K: Any,
    residual_pool: Any,
    rng: np.random.Generator,
    draws: int = 1,
) -> dict[str, Any]:
    """Resample fresh residuals independently of a fixed sample kernel."""
    kernel, factor = _symmetric_spd(K, "K")
    pool = np.asarray(residual_pool, dtype=np.float64)
    if pool.ndim != 1 or pool.size < 1 or not np.isfinite(pool).all():
        raise ValueError("residual_pool must be a nonempty finite vector")
    if not isinstance(draws, int) or isinstance(draws, bool) or draws < 1:
        raise ValueError("draws must be a positive integer")
    n = kernel.shape[0]
    values = []
    for _ in range(draws):
        independent = rng.choice(pool, size=n, replace=True)
        alpha = _solve_spd(kernel, factor, independent)
        values.append(0.5 * float(independent @ alpha))
    return {
        "A_ind": float(np.mean(values)),
        "A_ind_draws": np.asarray(values, dtype=np.float64),
        "independent_residual_second_moment": float(np.mean(pool**2)),
    }


def spectral_dependence_diagnostics(
    K: Any,
    residual: Any,
    residual_population: Any,
) -> dict[str, float]:
    """Split realized intrinsic aliasing into spectrum and dependence terms."""
    kernel, factor = _symmetric_spd(K, "K")
    r = _vector(residual, kernel.shape[0], "residual")
    population = np.asarray(residual_population, dtype=np.float64)
    if population.ndim != 1 or population.size < 1 or not np.isfinite(population).all():
        raise ValueError("residual_population must be a nonempty finite vector")
    alpha = _solve_spd(kernel, factor, r)
    A = 0.5 * float(r @ alpha)
    inverse_columns = _solve_spd(kernel, factor, np.eye(kernel.shape[0]))
    trace_inverse = float(np.trace(inverse_columns))
    residual_second = float(np.mean(population**2))
    A_spec = 0.5 * residual_second * trace_inverse
    A_dep = A - A_spec
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    coefficients = eigenvectors.T @ r
    contributions = 0.5 * coefficients**2 / eigenvalues
    bottom_count = max(1, int(np.ceil(0.1 * kernel.shape[0])))
    bottom_share = float(np.sum(contributions[:bottom_count]) / A) if A > 0 else 0.0
    return {
        "A": A,
        "A_spec": A_spec,
        "A_dep": A_dep,
        "trace_K_inv": trace_inverse,
        "residual_second_moment": residual_second,
        "min_eig_K": float(eigenvalues[0]),
        "cond_K": float(eigenvalues[-1] / eigenvalues[0]),
        "bottom_eigenmode_share_A": bottom_share,
    }


def weighted_min_norm_readout(H: Any, y: Any, Q: Any) -> np.ndarray:
    """Minimum ``theta.T @ Q @ theta`` interpolant via positive-definite solves."""
    design = _matrix(H, "H")
    target = _vector(y, design.shape[0], "y")
    metric, metric_factor = _symmetric_spd(Q, "Q")
    if metric.shape[0] != design.shape[1]:
        raise ValueError("Q must match H's feature dimension")
    q_inv_h_t = _solve_spd(metric, metric_factor, design.T)
    gram = design @ q_inv_h_t
    gram, gram_factor = _symmetric_spd(gram, "H Q^-1 H.T")
    return q_inv_h_t @ _solve_spd(gram, gram_factor, target)
