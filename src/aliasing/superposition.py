"""Controlled overcomplete feature dictionaries and representation aliasing."""

from __future__ import annotations

import numpy as np


def structured_overcomplete_dictionary(dimension: int, spread: int) -> np.ndarray:
    """Return D=[I_d, v] where v spreads one feature over ``spread`` axes.

    The dictionary has d+1 unit-norm feature directions in dimension d.  The
    final direction is the equal-weight combination of the first ``spread``
    coordinate directions, so its participation ratio is exactly spread.
    """
    if dimension < 2:
        raise ValueError("dimension must be at least two")
    if not 1 <= spread <= dimension:
        raise ValueError("spread must lie between one and dimension")
    direction = np.zeros(dimension, dtype=float)
    direction[:spread] = 1.0 / np.sqrt(spread)
    return np.column_stack((np.eye(dimension), direction))


def dictionary_geometry(dictionary: np.ndarray) -> dict[str, float]:
    """Compute scale-invariant coherence and last-direction spread."""
    if dictionary.ndim != 2 or dictionary.shape[1] < 2:
        raise ValueError("dictionary must be a matrix with at least two columns")
    norms = np.linalg.norm(dictionary, axis=0)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("dictionary columns must be nonzero")
    normalized = dictionary / norms
    gram = normalized.T @ normalized
    np.fill_diagonal(gram, 0.0)
    direction = normalized[:, -1]
    participation = 1.0 / np.sum(direction**4)
    return {
        "ambient_dimension": float(dictionary.shape[0]),
        "feature_count": float(dictionary.shape[1]),
        "feature_pressure": float(dictionary.shape[1] / dictionary.shape[0]),
        "mutual_coherence": float(np.max(np.abs(gram))),
        "last_feature_participation_ratio": float(participation),
    }


def representation_bayes_mse(
    representations: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    atol: float = 1e-12,
) -> tuple[float, float]:
    """Return Bayes squared risk and probability mass in aliased code groups.

    This finite-support calculation groups latent states that produce the
    same represented vector.  Within each group, the optimal squared-loss
    predictor is the conditional target mean, so the returned risk is exactly
    E[Var(Y | H)] for the specified discrete distribution.
    """
    if representations.ndim != 2 or targets.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("representations must be a matrix and targets/probabilities vectors")
    if representations.shape[0] != targets.size or targets.size != probabilities.size:
        raise ValueError("one target and probability are required per represented state")
    if probabilities.size == 0 or np.any(probabilities < 0.0):
        raise ValueError("probabilities must be nonnegative and nonempty")
    total_probability = float(probabilities.sum())
    if not np.isclose(total_probability, 1.0):
        raise ValueError("probabilities must sum to one")

    groups: list[list[int]] = []
    for i, representation in enumerate(representations):
        if probabilities[i] == 0.0:
            continue
        for group in groups:
            if np.allclose(representation, representations[group[0]], atol=atol, rtol=0.0):
                group.append(i)
                break
        else:
            groups.append([i])

    risk = 0.0
    aliased_mass = 0.0
    for group in groups:
        mass = float(probabilities[group].sum())
        if len(group) > 1:
            aliased_mass += mass
        conditional_mean = float(np.dot(probabilities[group], targets[group]) / mass)
        risk += float(np.dot(probabilities[group], (targets[group] - conditional_mean) ** 2))
    return risk, aliased_mass


def controlled_code_distribution(
    dimension: int,
    spread: int,
    max_sparsity: int,
    collision_mass: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make a sparse nonnegative distribution with a known collision threshold.

    The extra feature's one-hot code and its ``spread``-sparse coordinate
    decomposition are both in the support exactly when max_sparsity >= spread.
    Their represented states coincide, while the target selects the extra
    feature.  Remaining probability is assigned to the zero code.
    """
    if not 0 <= collision_mass <= 1:
        raise ValueError("collision_mass must lie in [0, 1]")
    dictionary = structured_overcomplete_dictionary(dimension, spread)
    extra_feature = np.zeros(dimension + 1, dtype=float)
    extra_feature[-1] = 1.0
    zero = np.zeros(dimension + 1, dtype=float)

    if max_sparsity >= spread:
        coordinate_decomposition = np.zeros(dimension + 1, dtype=float)
        coordinate_decomposition[:spread] = 1.0 / np.sqrt(spread)
        codes = np.stack((extra_feature, coordinate_decomposition, zero))
        targets = np.array([1.0, 0.0, 0.0])
        probabilities = np.array(
            [collision_mass / 2.0, collision_mass / 2.0, 1.0 - collision_mass]
        )
    else:
        codes = np.stack((extra_feature, zero))
        targets = np.array([1.0, 0.0])
        probabilities = np.array([collision_mass, 1.0 - collision_mass])
    representations = codes @ dictionary.T
    return representations, targets, probabilities
