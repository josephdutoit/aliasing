import numpy as np

from aliasing.superposition import (
    controlled_code_distribution,
    dictionary_geometry,
    representation_bayes_mse,
    structured_overcomplete_dictionary,
)


def test_dictionary_geometry_and_collision_threshold():
    dimension = 6
    collision_mass = 0.6
    for spread in range(1, dimension + 1):
        dictionary = structured_overcomplete_dictionary(dimension, spread)
        geometry = dictionary_geometry(dictionary)
        assert geometry["feature_count"] == dimension + 1
        assert geometry["feature_pressure"] == (dimension + 1) / dimension
        assert np.isclose(geometry["mutual_coherence"], 1.0 / np.sqrt(spread))
        assert np.isclose(geometry["last_feature_participation_ratio"], spread)

        for max_sparsity in range(1, dimension + 1):
            representations, targets, probabilities = controlled_code_distribution(
                dimension, spread, max_sparsity, collision_mass
            )
            risk, aliased_mass = representation_bayes_mse(
                representations, targets, probabilities
            )
            if max_sparsity >= spread:
                assert np.isclose(risk, collision_mass / 4.0)
                assert np.isclose(aliased_mass, collision_mass)
            else:
                assert np.isclose(risk, 0.0)
                assert np.isclose(aliased_mass, 0.0)


def test_bayes_risk_uses_conditional_target_variance():
    representations = np.array([[0.0], [0.0], [1.0]])
    targets = np.array([0.0, 1.0, 2.0])
    probabilities = np.array([0.2, 0.3, 0.5])
    risk, aliased_mass = representation_bayes_mse(
        representations, targets, probabilities
    )
    assert np.isclose(risk, 0.2 * 0.3 / (0.2 + 0.3))
    assert np.isclose(aliased_mass, 0.5)
