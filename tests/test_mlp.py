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
    assert "activation_pattern_correlation_max" in metrics
    assert "feature_coherence_max" not in metrics


def test_null_head_component_is_train_invisible_and_test_effect_is_exact():
    x_train, x_test, y_train, y_test = make_teacher_problem(
        seed=7,
        n_train=5,
        n_test=9,
        input_dim=2,
        teacher_width=3,
        noise_std=0.0,
    )
    model = TwoLayerMLP(input_dim=2, hidden_width=8)
    initial_weight = model.fc1.weight.detach().clone()
    initial_bias = model.fc1.bias.detach().clone()
    initial_head = model.fc2.weight.detach().clone()
    metrics = hidden_aliasing_metrics(
        model, x_train, x_test, y_train, y_test,
        initial_weight, initial_bias, initial_head,
    )
    with torch.no_grad():
        z_train = model.hidden(x_train)
        z_test = model.hidden(x_test)
        head = model.fc2.weight.reshape(-1)
        projector = torch.linalg.pinv(z_train) @ z_train
        null_projector = torch.eye(8, dtype=z_train.dtype) - projector
        null_head = null_projector @ head
        train_effect = z_train @ null_head
        test_effect = z_test @ null_head
    assert torch.linalg.norm(train_effect).item() < 1e-10
    assert metrics["test_null_response_rms"] == pytest.approx(
        torch.sqrt(torch.mean(test_effect**2)).item(), abs=1e-10
    )
    assert metrics["test_null_response_relative"] == pytest.approx(
        metrics["test_null_response_rms"]
        / max(metrics["test_prediction_rms"], torch.finfo(torch.float64).eps),
        abs=1e-10,
    )

