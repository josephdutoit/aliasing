import numpy as np
from aliasing.linear import evaluate_width

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


def test_near_collision_svd_projector_does_not_create_spurious_data_insufficiency():
    separation = 1e-8
    hidden = np.array([[1.0, 0.0], [1.0, separation], [0.0, 0.0]])
    target = np.array([0.5, -0.5, 0.0])
    probabilities = np.array([0.25, 0.25, 0.5])
    population_head = np.array([0.5, -1.0 / separation])
    metrics = finite_support_gad(
        hidden, target, hidden, target, probabilities,
        population_head=population_head,
    )
    assert metrics["data_insufficiency"] < 1e-20
    assert metrics["decomposition_reconstruction_error"] < 1e-8
    assert metrics["risk_identity_error"] < 1e-8


def test_covariance_and_feature_metrics_are_finite():
    pre = np.array([[1.0, -1.0], [0.5, 2.0], [-2.0, 0.0]])
    covariance = covariance_pair_metrics(pre)
    direction = feature_direction_metrics(np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]))
    assert covariance["pre_cov_rank"] >= 1
    assert covariance["post_cov_rank"] >= 1
    assert 0.0 <= direction["feature_direction_coherence"] <= 1.0
    assert direction["feature_pressure"] == 1.5


def test_full_rank_design_can_have_gad_aliasing_without_data_insufficiency():
    h = np.array([[1., 0.], [0., 1.], [1., 1.]])
    target = np.array([1., 1., -1.])  # orthogonal to the modeled columns
    g = finite_support_gad(h[:2], target[:2], h, target, np.ones(3)/3)
    assert np.isclose(g["aliasing"], 2.)
    assert g["data_insufficiency"] < 1e-25
    assert np.isclose(g["test_mse"], 3.)


def test_unobserved_modeled_direction_is_data_insufficiency_not_aliasing():
    h = np.eye(2)
    y = np.array([1., 2.])
    g = finite_support_gad(h[:1], y[:1], h, y, np.ones(2)/2)
    assert g["aliasing"] < 1e-25
    assert np.isclose(g["data_insufficiency"], 2.)


def test_shared_decomposition_matches_original_gad_with_explicit_unmodeled_features():
    rng = np.random.default_rng(802)
    for n in (2, 5):
        zt, zp = rng.normal(size=(n, 5)), rng.normal(size=(9, 5))
        theta = rng.normal(size=5)
        truth_t, truth_p = zt @ theta, zp @ theta
        original = evaluate_width(zt, zp, theta, truth_t, 3)
        current = finite_support_gad(zt[:, :3], truth_t, zp[:, :3], truth_p,
                                     np.ones(9)/9, population_head=theta[:3])
        for metric in ("model_insufficiency", "aliasing", "data_insufficiency", "test_mse"):
            assert np.isclose(current[metric], original[metric], atol=1e-11)
        assert current["risk_identity_error"] < 1e-11
