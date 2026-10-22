"""Small exact helpers for static representation-geometry experiments."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np


def make_dictionary(
    dimension: int,
    feature_count: int,
    family: str,
    seed: int,
    spread: int = 2,
) -> np.ndarray:
    """Build normalized feature directions for controlled comparisons."""
    if dimension < 1 or feature_count < 1:
        raise ValueError("dimension and feature_count must be positive")
    rng = np.random.default_rng(seed)
    if family == "orthogonal":
        if feature_count > dimension:
            raise ValueError("orthogonal family requires feature_count <= dimension")
        return np.eye(dimension, feature_count)
    if family == "random":
        dictionary = rng.normal(size=(dimension, feature_count))
    elif family == "structured":
        if feature_count <= dimension:
            raise ValueError("structured family requires an overcomplete dictionary")
        if not 1 <= spread <= dimension:
            raise ValueError("spread must be between one and dimension")
        columns = [np.eye(dimension)[:, j] for j in range(dimension)]
        support_sets = list(combinations(range(dimension), spread))
        for index in range(feature_count - dimension):
            direction = np.zeros(dimension)
            coords = support_sets[index % len(support_sets)]
            direction[list(coords)] = 1.0 / np.sqrt(spread)
            columns.append(direction)
        dictionary = np.column_stack(columns)
    else:
        raise ValueError("family must be one of: orthogonal, random, structured")
    norms = np.linalg.norm(dictionary, axis=0)
    return dictionary / np.maximum(norms, np.finfo(float).eps)


def sparse_binary_support(
    feature_count: int,
    max_sparsity: int,
    *,
    allow_cooccurrence: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate binary codes with at most max_sparsity active features.

    Probabilities are a Bernoulli product law conditioned on the support
    constraint. Setting allow_cooccurrence=False restricts the support to zero
    and one active feature, providing a useful data-support control.
    """
    if feature_count < 1 or max_sparsity < 1:
        raise ValueError("feature_count and max_sparsity must be positive")
    cap = min(max_sparsity, feature_count)
    if not allow_cooccurrence:
        cap = 1
    codes: list[np.ndarray] = []
    weights: list[float] = []
    q = min(max_sparsity / feature_count, 0.5)
    for size in range(cap + 1):
        for active in combinations(range(feature_count), size):
            code = np.zeros(feature_count)
            code[list(active)] = 1.0
            codes.append(code)
            weights.append(q**size * (1.0 - q) ** (feature_count - size))
    probabilities = np.asarray(weights, dtype=float)
    probabilities /= probabilities.sum()
    return np.stack(codes), probabilities


def restricted_min_singular_value(
    dictionary: np.ndarray,
    max_support: int,
) -> float:
    """Return min singular value over all column subsets of size <= max_support.

    This is the exact support-enumerated value for the supplied finite matrix.
    It is computationally intended for small feature counts.
    """
    if dictionary.ndim != 2 or max_support < 1:
        raise ValueError("dictionary must be a matrix and max_support positive")
    p = dictionary.shape[1]
    minimum = float("inf")
    for size in range(1, min(max_support, p) + 1):
        for support in combinations(range(p), size):
            if size > dictionary.shape[0]:
                value = 0.0
            else:
                singular_values = np.linalg.svd(dictionary[:, support], compute_uv=False)
                value = float(singular_values[-1]) if singular_values.size else 0.0
            minimum = min(minimum, value)
    return 0.0 if not np.isfinite(minimum) else minimum


def finite_support_gad(
    hidden_train: np.ndarray,
    target_train: np.ndarray,
    hidden_support: np.ndarray,
    target_support: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Decompose finite-support prediction error for a frozen linear readout.

    The population-optimal linear readout defines the model projection. Its
    component in the training null space is the alias component omitted by the
    minimum-norm training fit. The remaining row-space coefficient shift is
    reported separately. Prediction components need not be orthogonal, so all
    pairwise cross terms are included.
    """
    if hidden_train.ndim != 2 or hidden_support.ndim != 2:
        raise ValueError("hidden feature inputs must be matrices")
    if hidden_train.shape[1] != hidden_support.shape[1]:
        raise ValueError("train and support hidden dimensions must agree")
    if target_train.shape != (hidden_train.shape[0],):
        raise ValueError("target_train shape does not match hidden_train")
    if target_support.shape != (hidden_support.shape[0],):
        raise ValueError("target_support shape does not match hidden_support")
    if probabilities.shape != (hidden_support.shape[0],):
        raise ValueError("probabilities shape does not match hidden_support")
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("probabilities must be nonnegative and sum to one")

    pinv_train = np.linalg.pinv(hidden_train)
    fitted_head = pinv_train @ target_train
    projector = pinv_train @ hidden_train
    projector = 0.5 * (projector + projector.T)
    null_projector = np.eye(hidden_train.shape[1]) - projector

    sqrt_prob = np.sqrt(probabilities)
    population_head = np.linalg.lstsq(
        hidden_support * sqrt_prob[:, None],
        target_support * sqrt_prob,
        rcond=None,
    )[0]
    residual_model = hidden_support @ population_head - target_support
    residual_alias = -(hidden_support @ (null_projector @ population_head))
    residual_data = hidden_support @ (projector @ (fitted_head - population_head))
    residual_total = residual_model + residual_alias + residual_data
    direct = hidden_support @ fitted_head - target_support
    weighted_mse = lambda values: float(np.dot(probabilities, values**2))
    cross = {
        "cross_model_alias": float(2.0 * np.dot(probabilities, residual_model * residual_alias)),
        "cross_model_data": float(2.0 * np.dot(probabilities, residual_model * residual_data)),
        "cross_alias_data": float(2.0 * np.dot(probabilities, residual_alias * residual_data)),
    }
    return {
        "train_mse": float(np.mean((hidden_train @ fitted_head - target_train) ** 2)),
        "test_mse": weighted_mse(direct),
        "model_insufficiency": weighted_mse(residual_model),
        "aliasing": weighted_mse(residual_alias),
        "data_algorithm": weighted_mse(residual_data),
        "decomposition_reconstruction_error": float(np.max(np.abs(residual_total - direct))),
        "risk_identity_error": float(
            abs(
                weighted_mse(direct)
                - weighted_mse(residual_model)
                - weighted_mse(residual_alias)
                - weighted_mse(residual_data)
                - sum(cross.values())
            )
        ),
        "alias_operator_rms": float(
            np.sqrt(np.sum(probabilities[:, None] * (hidden_support @ null_projector) ** 2))
        ),
        **cross,
    }


def covariance_pair_metrics(preactivations: np.ndarray) -> dict[str, float]:
    """Summarize pre/post-ReLU sample Gram geometry for a frozen layer."""
    if preactivations.ndim != 2:
        raise ValueError("preactivations must have shape (samples, neurons)")
    post = np.maximum(preactivations, 0.0)
    k = preactivations @ preactivations.T
    k_sigma = post @ post.T

    def summarize(matrix: np.ndarray, prefix: str) -> dict[str, float]:
        values = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        values = np.maximum(values, 0.0)
        largest = float(values[-1]) if values.size else 0.0
        tolerance = max(matrix.shape) * np.finfo(float).eps * max(largest, 1.0)
        positive = values[values > tolerance]
        stable_rank = float(values.sum() / largest) if largest > 0 else 0.0
        return {
            f"{prefix}_rank": float(positive.size),
            f"{prefix}_stable_rank": stable_rank,
            f"{prefix}_trace": float(values.sum()),
        }

    return {**summarize(k, "pre_cov"), **summarize(k_sigma, "post_cov")}


def feature_direction_metrics(directions: np.ndarray) -> dict[str, float]:
    """Measure overlap among known-feature directions in a hidden space."""
    if directions.ndim != 2:
        raise ValueError("directions must be a matrix with features in columns")
    norms = np.linalg.norm(directions, axis=0)
    valid = norms > np.finfo(float).eps
    if valid.sum() < 2:
        coherence = 0.0
    else:
        normalized = directions[:, valid] / norms[valid]
        gram = np.abs(normalized.T @ normalized)
        np.fill_diagonal(gram, 0.0)
        coherence = float(gram.max())
    singular = np.linalg.svd(directions, compute_uv=False)
    squared = singular**2
    total = float(squared.sum())
    effective_rank = total**2 / float(np.sum(squared**2)) if total else 0.0
    return {
        "feature_direction_coherence": coherence,
        "feature_direction_effective_rank": float(effective_rank),
        "feature_count": float(directions.shape[1]),
        "hidden_dimension": float(directions.shape[0]),
        "feature_pressure": float(directions.shape[1] / directions.shape[0]),
    }
