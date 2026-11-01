"""Static geometry diagnostics; neither overlap nor probe gain defines GAD."""
from __future__ import annotations

import numpy as np

from .representation_benchmark import TABLE_GROUPS


EXTRA_GROUPS = {
    'task_readouts': TABLE_GROUPS['readouts'] + ('feature',),
    'alias_modes': ('active_count', 'task_profile', 'architecture', 'method', 'sample_size', 'mode'),
}


def alias_modes(h, hp, residual):
    """SVD diagnosis of canonical aliasing, including total mode interference.

    Diagonal mode energies need not sum to alias energy: population-visible
    modes are generally not orthogonal. The signed off-diagonal sum is saved.
    Singular mode indices have no semantic correspondence between maps.
    """
    u, s, vh = np.linalg.svd(h, full_matrices=False)
    rank = int(np.sum(s > max(h.shape)*np.finfo(float).eps*s[0])) if len(s) else 0
    if not rank:
        return []
    projected = u[:, :rank].T @ residual
    visible = hp @ vh[:rank].T
    coefficients = projected/s[:rank, None]
    alias = visible @ coefficients
    diagonal = np.mean(visible**2, axis=0)*np.mean(coefficients**2, axis=1)
    total = float(np.mean(alias**2))
    offdiag = total-float(diagonal.sum())
    return [dict(singular_value=float(s[j]),
        training_residual_projection_mse=float(np.mean(projected[j]**2)),
        population_visibility_mse=float(np.mean(visible[:, j]**2)),
        mode_diagonal_alias_energy=float(diagonal[j]),
        total_alias_energy=total, summed_mode_diagonal_energy=float(diagonal.sum()),
        signed_mode_offdiagonal_sum=offdiag) for j in range(rank)]


def latent_geometry(model, params, spec, x, contexts=128):
    """Finite-difference latent Jacobians on a fixed subset of evaluation inputs.

    These infinitesimal amplitude interventions may leave the support at zero
    coordinates. They diagnose local overlap, not sparse-code injectivity,
    collisions, or a certified count of represented features.
    """
    x = x[:contexts]
    epsilon = 1e-5
    base = model.features(x, params, intercept=False)
    jac = np.stack([(model.features(x+epsilon*direction, params, intercept=False)-base)/epsilon
                    for direction in spec['embedding']], axis=1)
    gram = np.einsum('cih,cjh->ij', jac, jac)/len(x)
    norms = np.sqrt(np.maximum(np.diag(gram), 0.))
    valid = norms > 1e-12
    normalized = gram/np.maximum(np.outer(norms, norms), 1e-24)
    np.fill_diagonal(normalized, 0.)
    normalized[~valid, :] = normalized[:, ~valid] = 0.
    return dict(latent_jacobian_overlap=float(np.max(np.abs(normalized))),
        latent_jacobian_norm_min=float(norms.min()),
        latent_jacobian_norm_max=float(norms.max()),
        latent_jacobian_nonzero_directions=int(valid.sum()),
        latent_jacobian_contexts=len(x), latent_jacobian_step=epsilon)
