"""Random ReLU feature maps used by the neural aliasing baseline."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def relu_random_features(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Evaluate a fixed random ReLU feature map."""
    return np.maximum(x @ weights.T + bias[None, :], 0.0)


def make_random_feature_problem(
    rng: np.random.Generator,
    n_train: int,
    n_test: int,
    input_dim: int = 20,
    full_features: int = 128,
    noise_std: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a teacher signal represented in a random ReLU feature basis.

    Returns x_train, x_test, z_train, z_test, y_train, theta_full.
    The last two arrays are returned separately despite the type annotation
    above for backwards-compatible unpacking at call sites.
    """
    if n_train < 1 or n_test < 1:
        raise ValueError("sample counts must be positive")
    if input_dim < 1 or full_features < 1:
        raise ValueError("input_dim and full_features must be positive")

    x_train = rng.normal(size=(n_train, input_dim))
    x_test = rng.normal(size=(n_test, input_dim))
    weights = rng.normal(size=(full_features, input_dim)) / np.sqrt(input_dim)
    bias = rng.normal(scale=0.1, size=full_features)

    z_train = relu_random_features(x_train, weights, bias)
    z_test = relu_random_features(x_test, weights, bias)

    theta_full = rng.normal(size=full_features)
    theta_full /= np.sqrt(np.arange(1, full_features + 1))
    theta_full /= np.linalg.norm(theta_full)

    y_train = z_train @ theta_full
    y_train = y_train + noise_std * rng.normal(size=n_train)

    return x_train, x_test, z_train, z_test, y_train, theta_full
