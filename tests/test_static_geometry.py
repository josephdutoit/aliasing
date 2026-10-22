import numpy as np

from aliasing.static_geometry import (
    covariance_pair_metrics,
    feature_direction_metrics,
    finite_support_gad,
    make_dictionary,
    restricted_min_singular_value,
    sparse_binary_support,
)


def test_structured_dictionary_sparse_threshold_is_detected_by_support_singular_values():
    dimension, spread = 4, 3
    dictionary = make_dictionary(dimension, dimension + 1, "structured", seed=0, spread=spread)
    assert restricted_min_singular_value(dictionary, 2) > 1e-10
    assert restricted_min_singular_value(dictionary, 2 * spread) < 1e-10


def test_sparse_support_distribution_is_normalized_and_obeys_cooccurrence_control():
    codes, probabilities = sparse_binary_support(5, 3, allow_cooccurrence=False)
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(codes.sum(axis=1) <= 1)
    mixed_codes, mixed_probabilities = sparse_binary_support(5, 3, allow_cooccurrence=True)
    assert np.isclose(mixed_probabilities.sum(), 1.0)
    assert np.any(mixed_codes.sum(axis=1) > 1)


def test_finite_support_gad_reconstructs_direct_population_risk():
    hidden_support = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    target_support = np.array([0.0, 1.0, -0.5, 0.5])
    probabilities = np.array([0.4, 0.2, 0.2, 0.2])
    train_indices = np.array([0, 1, 3])
    metrics = finite_support_gad(
        hidden_support[train_indices],
        target_support[train_indices],
        hidden_support,
        target_support,
        probabilities,
    )
    assert metrics["decomposition_reconstruction_error"] < 1e-12
    assert metrics["risk_identity_error"] < 1e-12


def test_finite_support_gad_accepts_precomputed_population_head():
    hidden = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]])
    target = np.array([0.0, 1.0, -1.0])
    probabilities = np.array([0.2, 0.4, 0.4])
    train = np.array([0, 1])
    weighted_head = np.linalg.lstsq(
        hidden * np.sqrt(probabilities[:, None]),
        target * np.sqrt(probabilities),
        rcond=None,
    )[0]
    reference = finite_support_gad(hidden[train], target[train], hidden, target, probabilities)
    optimized = finite_support_gad(
        hidden[train], target[train], hidden, target, probabilities,
        population_head=weighted_head,
    )
    assert np.isclose(reference["test_mse"], optimized["test_mse"])
    assert np.isclose(reference["aliasing"], optimized["aliasing"])
    assert optimized["decomposition_reconstruction_error"] < 1e-12


def test_covariance_and_feature_metrics_are_finite():
    pre = np.array([[1.0, -1.0], [0.5, 2.0], [-2.0, 0.0]])
    covariance = covariance_pair_metrics(pre)
    direction = feature_direction_metrics(np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]))
    assert covariance["pre_cov_rank"] >= 1
    assert covariance["post_cov_rank"] >= 1
    assert 0.0 <= direction["feature_direction_coherence"] <= 1.0
    assert direction["feature_pressure"] == 1.5
