"""Fixed-feature linear baseline for generalized aliasing experiments."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def _squared_mean(vector: np.ndarray) -> float:
    return float(np.mean(np.asarray(vector, dtype=float) ** 2))


def _operator_norm(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    return float(np.linalg.norm(matrix, ord=2))


def evaluate_width(
    x_train: np.ndarray,
    x_test: np.ndarray,
    theta_full: np.ndarray,
    y_train: np.ndarray,
    d: int,
) -> Dict[str, float]:
    """Evaluate one modeled-feature width.

    The learner uses only the first d columns and fits the minimum-norm
    solution with a Moore-Penrose pseudoinverse. The prediction error is
    reconstructed exactly as data insufficiency + aliasing + model
    insufficiency + fitted noise. Their squared norms need not add because
    the components need not be orthogonal; the residual cross-term is reported.
    """
    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError("x_train and x_test must be matrices")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("train and test matrices need the same number of columns")
    if x_train.shape[1] != theta_full.shape[0]:
        raise ValueError("theta_full must match the number of feature columns")
    if y_train.shape[0] != x_train.shape[0]:
        raise ValueError("y_train must have one value per training example")
    if not 0 <= d <= x_train.shape[1]:
        raise ValueError("d must be between zero and the number of features")

    x_tm = x_train[:, :d]
    x_tu = x_train[:, d:]
    x_pm = x_test[:, :d]
    x_pu = x_test[:, d:]
    theta_m = theta_full[:d]
    theta_u = theta_full[d:]

    design_pinv = np.linalg.pinv(x_tm)
    fitted_theta_m = design_pinv @ y_train
    prediction = x_pm @ fitted_theta_m
    truth = x_test @ theta_full
    prediction_error = prediction - truth

    b = design_pinv @ x_tm
    alias_operator = x_pm @ design_pinv @ x_tu

    data_vector = x_pm @ ((b - np.eye(d)) @ theta_m)
    alias_vector = alias_operator @ theta_u
    model_vector = -(x_pu @ theta_u)
    train_noise = y_train - x_train @ theta_full
    noise_vector = x_pm @ (design_pinv @ train_noise)

    reconstructed_error = data_vector + alias_vector + model_vector + noise_vector
    cross_term = _squared_mean(prediction_error) - sum(
        _squared_mean(component)
        for component in (data_vector, alias_vector, model_vector, noise_vector)
    )

    return {
        "d": float(d),
        "test_mse": _squared_mean(prediction_error),
        "alias_operator_norm": _operator_norm(alias_operator),
        "model_insufficiency": _squared_mean(model_vector),
        "data_insufficiency": _squared_mean(data_vector),
        "aliasing": _squared_mean(alias_vector),
        "fitted_noise": _squared_mean(noise_vector),
        "component_cross_term": float(cross_term),
        "reconstruction_error": float(
            np.max(np.abs(reconstructed_error - prediction_error))
        ),
    }


def run_sweep(
    x_train: np.ndarray,
    x_test: np.ndarray,
    theta_full: np.ndarray,
    y_train: np.ndarray,
    widths: Iterable[int],
) -> List[Dict[str, float]]:
    """Evaluate a sequence of modeled feature counts."""
    return [
        evaluate_width(x_train, x_test, theta_full, y_train, int(d))
        for d in widths
    ]
