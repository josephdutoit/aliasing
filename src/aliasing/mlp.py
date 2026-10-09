"""Small feature-learning MLPs and teacher-data generation."""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from torch import nn


torch.set_default_dtype(torch.float64)


class TwoLayerMLP(nn.Module):
    """A scalar-output ReLU network with an explicit hidden representation."""

    def __init__(self, input_dim: int, hidden_width: int) -> None:
        super().__init__()
        if input_dim < 1 or hidden_width < 1:
            raise ValueError("input_dim and hidden_width must be positive")
        self.fc1 = nn.Linear(input_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, 1, bias=False)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.hidden(x)).squeeze(-1)


def seed_everything(seed: int) -> None:
    """Seed all RNGs used by the experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_teacher_problem(
    seed: int,
    n_train: int,
    n_test: int,
    input_dim: int = 20,
    teacher_width: int = 64,
    noise_std: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create data shared by all student widths for one seed."""
    if n_train < 1 or n_test < 1:
        raise ValueError("sample counts must be positive")
    if input_dim < 1 or teacher_width < 1:
        raise ValueError("input_dim and teacher_width must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be nonnegative")

    rng = np.random.default_rng(seed)
    x_train = torch.from_numpy(rng.normal(size=(n_train, input_dim)))
    x_test = torch.from_numpy(rng.normal(size=(n_test, input_dim)))

    # The teacher is deterministic conditional on the data seed and is not
    # retrained for each student width.
    seed_everything(seed + 1_000_003)
    teacher = TwoLayerMLP(input_dim, teacher_width)
    teacher.eval()
    with torch.no_grad():
        y_train = teacher(x_train)
        y_test = teacher(x_test)
    if noise_std:
        y_train = y_train + noise_std * torch.from_numpy(rng.normal(size=n_train))
    return x_train, x_test, y_train, y_test


def _relative_norm(value: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.linalg.norm(reference).item()
    numerator = torch.linalg.norm(value).item()
    return numerator / max(denominator, np.finfo(float).eps)


def hidden_aliasing_metrics(
    model: TwoLayerMLP,
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    initial_fc1_weight: torch.Tensor,
    initial_fc1_bias: torch.Tensor,
    initial_fc2_weight: torch.Tensor,
    interpolation_threshold: float = 1e-10,
) -> dict[str, float]:
    """Measure interpolation, hidden-feature aliasing, and feature drift.

    For the current hidden matrix Z_train, the least-norm output head is
    Z_train^+ y_train.  The residual head component in ker(Z_train) is the
    exact finite-sample aliasing direction.  Its test effect is nonzero when
    the hidden features have changed outside the sampled data geometry.
    """
    model.eval()
    with torch.no_grad():
        z_train = model.hidden(x_train)
        z_test = model.hidden(x_test)
        pred_train = model(x_train)
        pred_test = model(x_test)

        train_mse = torch.mean((pred_train - y_train) ** 2).item()
        test_mse = torch.mean((pred_test - y_test) ** 2).item()

        z_pinv = torch.linalg.pinv(z_train)
        projector = z_pinv @ z_train
        projector = 0.5 * (projector + projector.T)
        identity = torch.eye(z_train.shape[1], dtype=z_train.dtype)
        null_projector = identity - projector

        # The initialization representation is the reference geometry for
        # measuring feature transport and movement of train-null directions.
        initial_z_train = torch.relu(
            x_train @ initial_fc1_weight.T + initial_fc1_bias[None, :]
        )
        initial_z_test = torch.relu(
            x_test @ initial_fc1_weight.T + initial_fc1_bias[None, :]
        )
        initial_z_pinv = torch.linalg.pinv(initial_z_train)
        initial_projector = initial_z_pinv @ initial_z_train
        initial_projector = 0.5 * (initial_projector + initial_projector.T)
        initial_null_projector = identity - initial_projector

        alias_operator = z_test @ null_projector
        initial_alias_operator = initial_z_test @ initial_null_projector

        head = model.fc2.weight.detach().reshape(-1)
        minimum_norm_head = z_pinv @ y_train
        null_head = null_projector @ head
        test_null_response = z_test @ null_head

        # For a two-layer ReLU network, the per-example parameter gradient is
        # available in closed form.  This makes NTK drift cheap enough to log
        # at every checkpoint without retaining an autograd graph.
        active_train = (model.fc1(x_train) > 0).to(z_train.dtype)
        active_initial = (
            x_train @ initial_fc1_weight.T + initial_fc1_bias[None, :] > 0
        ).to(z_train.dtype)
        current_jacobian = _output_jacobian_features(x_train, active_train, head)
        initial_jacobian = _output_jacobian_features(
            x_train, active_initial, initial_fc2_weight.reshape(-1)
        )
        current_ntk = current_jacobian @ current_jacobian.T
        initial_ntk = initial_jacobian @ initial_jacobian.T

        singular_values = torch.linalg.svdvals(z_train)
        leading = singular_values[0].item() if singular_values.numel() else 0.0
        tolerance = max(z_train.shape) * torch.finfo(z_train.dtype).eps * max(leading, 1.0)
        positive = singular_values[singular_values > tolerance]
        rank = int(positive.numel())
        condition_log10 = float("nan")
        if positive.numel() > 1:
            condition_log10 = float(torch.log10(positive[0] / positive[-1]).item())

        feature_coherence = _feature_coherence(z_test)
        ntk_drift = _relative_norm(current_ntk - initial_ntk, initial_ntk)

        train_feature_norm = torch.linalg.norm(z_train).item()
        test_feature_norm = torch.linalg.norm(z_test).item()
        feature_transport_train = _relative_norm(
            z_train - initial_z_train, initial_z_train
        )
        feature_transport_test = _relative_norm(
            z_test - initial_z_test, initial_z_test
        )
        null_projector_drift = torch.linalg.matrix_norm(
            null_projector - initial_null_projector, ord=2
        ).item()
        alias_operator_norm = torch.linalg.matrix_norm(alias_operator, ord=2).item()
        initial_alias_operator_norm = torch.linalg.matrix_norm(
            initial_alias_operator, ord=2
        ).item()
        alias_operator_drift = torch.linalg.matrix_norm(
            alias_operator - initial_alias_operator, ord=2
        ).item()
        metrics = {
            "train_mse": train_mse,
            "test_mse": test_mse,
            "interpolates": float(train_mse <= interpolation_threshold),
            "hidden_rank": float(rank),
            "hidden_condition_log10": condition_log10,
            "hidden_singular_value_max": float(singular_values[0].item())
            if singular_values.numel()
            else 0.0,
            "hidden_singular_value_min": float(positive[-1].item()) if positive.numel() else 0.0,
            "feature_coherence_max": feature_coherence,
            "feature_effective_rank": _effective_rank(z_test),
            "ntk_drift": ntk_drift,
            "feature_transport_train": feature_transport_train,
            "feature_transport_test": feature_transport_test,
            "null_projector_drift": null_projector_drift,
            "alias_operator_norm": alias_operator_norm,
            "initial_alias_operator_norm": initial_alias_operator_norm,
            "alias_operator_drift": alias_operator_drift,
            "head_null_fraction": _relative_norm(null_head, head),
            "head_minimum_norm_gap": torch.linalg.norm(head - minimum_norm_head).item(),
            "test_null_response_rms": torch.sqrt(torch.mean(test_null_response**2)).item(),
            "test_null_response_relative": torch.linalg.norm(test_null_response).item()
            / max(test_feature_norm * max(torch.linalg.norm(null_head).item(), 1.0), np.finfo(float).eps),
            "feature_drift": _relative_norm(
                model.fc1.weight.detach() - initial_fc1_weight,
                initial_fc1_weight,
            ),
            "train_feature_norm": train_feature_norm,
            "test_feature_norm": test_feature_norm,
        }
        return metrics


def _output_jacobian_features(
    x: torch.Tensor,
    active: torch.Tensor,
    head: torch.Tensor,
) -> torch.Tensor:
    """Return per-example gradients flattened in a fixed parameter order."""
    hidden_weight = active[:, :, None] * x[:, None, :]
    hidden_weight = hidden_weight * head[None, :, None]
    hidden_bias = active * head[None, :]
    return torch.cat((active, hidden_weight.reshape(x.shape[0], -1), hidden_bias), dim=1)


def _feature_coherence(features: torch.Tensor) -> float:
    centered = features - features.mean(dim=0, keepdim=True)
    norms = torch.linalg.norm(centered, dim=0)
    valid = norms > torch.finfo(features.dtype).eps
    if int(valid.sum().item()) < 2:
        return 0.0
    normalized = centered[:, valid] / norms[valid]
    correlation = torch.abs(normalized.T @ normalized)
    correlation.fill_diagonal_(0.0)
    return float(correlation.max().item())


def _effective_rank(features: torch.Tensor) -> float:
    singular_values = torch.linalg.svdvals(features)
    squared = singular_values.square()
    total = squared.sum().item()
    if total <= np.finfo(float).eps:
        return 0.0
    return float((total * total / squared.square().sum()).item())
