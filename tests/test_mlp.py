import pytest


torch = pytest.importorskip("torch")

from aliasing.mlp import TwoLayerMLP, hidden_aliasing_metrics, make_teacher_problem


def test_mlp_metrics_have_expected_geometry():
    x_train, x_test, y_train, y_test = make_teacher_problem(
        seed=0,
        n_train=8,
        n_test=12,
        input_dim=3,
        teacher_width=4,
        noise_std=0.0,
    )
    model = TwoLayerMLP(input_dim=3, hidden_width=5)
    metrics = hidden_aliasing_metrics(
        model,
        x_train,
        x_test,
        y_train,
        y_test,
        model.fc1.weight.detach().clone(),
        model.fc1.bias.detach().clone(),
        model.fc2.weight.detach().clone(),
    )
    assert metrics["hidden_rank"] <= 5
    assert metrics["ntk_drift"] == 0.0
    assert metrics["feature_drift"] == 0.0
    assert metrics["test_mse"] >= 0.0
