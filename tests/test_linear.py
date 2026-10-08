import numpy as np

from aliasing.linear import evaluate_width


def make_problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    x_train = rng.normal(size=(8, 12))
    x_test = rng.normal(size=(32, 12))
    theta = rng.normal(size=12)
    y_train = x_train @ theta + 0.1 * rng.normal(size=8)
    return x_train, x_test, theta, y_train


def test_prediction_error_reconstruction_is_exact():
    x_train, x_test, theta, y_train = make_problem()
    result = evaluate_width(x_train, x_test, theta, y_train, d=5)
    assert result["reconstruction_error"] < 1e-10


def test_aliasing_is_zero_when_no_unmodeled_features_remain():
    x_train, x_test, theta, y_train = make_problem()
    result = evaluate_width(x_train, x_test, theta, y_train, d=x_train.shape[1])
    assert result["alias_operator_norm"] == 0.0
    assert result["aliasing"] == 0.0


def test_capacity_sweep_has_an_interpolation_point():
    x_train, x_test, theta, y_train = make_problem()
    result = evaluate_width(x_train, x_test, theta, y_train, d=x_train.shape[0])
    assert np.isfinite(result["test_mse"])
    assert result["reconstruction_error"] < 1e-10
