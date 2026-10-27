"""Readout, decoder and controlled-perturbation checks for frozen GAD maps.

The canonical alias term always uses the Moore--Penrose estimator. Other
heads add an explicit estimator correction; its cross terms are not dropped.
Perturbation sensitivities are separately named, not relabeled as GAD aliasing.
"""
from __future__ import annotations

import math
import numpy as np

from .controlled_superposition import loss_grad, sample


def extend_sample(spec, original_count, count, seed, extension_seed):
    """Preserve old samples exactly (the original sampler is not prefix stable)."""
    original = sample(spec, original_count, seed)
    if count <= original_count:
        return tuple(value[:count] for value in original)
    extra = sample(spec, count-original_count, extension_seed)
    return tuple(np.concatenate((a, b)) for a, b in zip(original, extra))


def pinverse(h):
    u, s, vh = np.linalg.svd(h, full_matrices=False)
    tolerance = max(h.shape)*np.finfo(float).eps*(s[0] if len(s) else 0.)
    rank = int(np.sum(s > tolerance))
    inverse = (vh[:rank].T/s[:rank]) @ u[:, :rank].T
    condition = float(s[0]/s[rank-1]) if rank else float('nan')
    return inverse, rank, condition


def ridge_head(h, y, penalty):
    """Mean squared loss plus penalty*||standardized slopes||^2; free intercept.

    All centering and scaling is estimated on the head-training split only.
    Returns coefficients in the ORIGINAL feature coordinates, including bias.
    """
    if penalty <= 0 or not np.isfinite(penalty):
        raise ValueError('ridge penalty must be finite and positive')
    if not np.allclose(h[:, 0], 1.):
        raise ValueError('first design column must be the intercept')
    x = h[:, 1:]
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    z = (x-center)/scale
    y_mean = y.mean(axis=0)
    slopes = np.linalg.solve(z.T @ z/len(z)+penalty*np.eye(z.shape[1]),
                             z.T @ (y-y_mean)/len(z)) / scale[:, None]
    return np.vstack((y_mean-center @ slopes, slopes))


def select_ridge(h, y, h_validation, y_validation, penalties, weights):
    trials = []
    for penalty in penalties:
        head = ridge_head(h, y, penalty)
        loss = float(np.mean((h_validation @ head-y_validation)**2 @ weights)/weights.sum())
        trials.append((loss, penalty, head))
    loss, penalty, head = min(trials, key=lambda item: (item[0], item[1]))
    return head, float(penalty), loss


def gad_metrics(h, y, ht, yt, reference, candidate=None, inverse=None, weights=None):
    """Canonical three terms plus H_P(beta_candidate-beta_MP) for other heads.

    The saved aliasing is the MP baseline term, INCLUDING for ridge/native
    rows. Estimator correction and all six signed cross terms account for the
    changed predictor. It must not be interpreted as ridge's residual leakage.
    """
    inverse = pinverse(h)[0] if inverse is None else inverse
    fitted = inverse @ y
    candidate = fitted if candidate is None else candidate
    remainder = y-h @ reference
    vectors = dict(model_insufficiency=ht @ reference-yt,
                   data_insufficiency=ht @ ((inverse @ h-np.eye(h.shape[1])) @ reference),
                   mp_aliasing=ht @ (inverse @ remainder),
                   estimator_correction=ht @ (candidate-fitted))
    prediction = ht @ candidate
    error = prediction-yt
    reconstructed = sum(vectors.values())
    identity = float(np.max(np.abs(error-reconstructed))/max(1., np.max(np.abs(error))))
    result = {name: float(np.mean(value**2)) for name, value in vectors.items()}
    names = list(vectors)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            result['cross_'+left+'__'+right] = float(2*np.mean(vectors[left]*vectors[right]))
    result.update(test_mse=float(np.mean(error**2)),
                  residual_train_mse=float(np.mean(remainder**2)),
                  train_mse=float(np.mean((h @ candidate-y)**2)),
                  classification_error=float(np.mean((prediction >= 0) != (yt >= 0))),
                  prediction_identity_relative_error=identity)
    denominator = result['residual_train_mse']
    result['mp_residual_gain'] = float(np.sqrt(result['mp_aliasing']/denominator)) if denominator > 1e-20 else float('nan')
    terms = [value for key, value in result.items() if key in vectors or key.startswith('cross_')]
    component_sum = math.fsum(terms)
    absolute_error = abs(result['test_mse']-component_sum)
    component_scale = max(1., math.fsum(abs(value) for value in terms))
    prediction_scale = max(1., float(np.max(sum(np.abs(value) for value in vectors.values()))))
    result['risk_identity_absolute_error'] = absolute_error
    result['risk_identity_relative_error'] = absolute_error/max(1., result['test_mse'])
    result['risk_identity_scaled_error'] = absolute_error/component_scale
    result['identity_cancellation_ratio'] = component_scale/max(1., result['test_mse'])
    # Ridge can cancel huge MP baseline terms. Preserve the absolute/risk-
    # relative error, but allow the floating-point precision of those terms.
    prediction_absolute_error = float(np.max(np.abs(error-reconstructed)))
    if (prediction_absolute_error > 1e-7*max(1.,np.max(np.abs(error)))+1e-12*prediction_scale or
            absolute_error > 1e-7*max(1.,result['test_mse'])+1e-12*component_scale):
        raise AssertionError('extended GAD identity failed')
    weights = np.ones(yt.shape[1]) if weights is None else weights
    result['weighted_test_mse'] = float(np.mean(error**2 @ weights)/weights.sum())
    return result


def train_decoder(initial, x, y, weights, steps, batch, lr, decay, seed, frozen_encoder):
    """Matched Adam runs: joint and fixed-encoder variants differ only by freeze."""
    params = {key: value.copy() for key, value in initial.items()}
    keys = [key for key in params if key != 'enc' or not frozen_encoder]
    first = {key: np.zeros_like(params[key]) for key in keys}
    second = {key: np.zeros_like(params[key]) for key in keys}
    rng = np.random.default_rng(seed)
    best, best_loss, history = None, float('inf'), []
    for step in range(steps+1):
        if step % 50 == 0 or step == steps:
            loss, _ = loss_grad(params, x, y, weights, True, decay)
            if not np.isfinite(loss):
                raise FloatingPointError('nonfinite decoder objective')
            if loss < best_loss:
                best_loss, best = loss, {key: value.copy() for key, value in params.items()}
            history.append(dict(step=step, objective=loss, best_objective=best_loss))
        if step == steps:
            break
        indices = rng.integers(len(x), size=min(batch, len(x)))
        _, gradients = loss_grad(params, x[indices], y[indices], weights, True, decay)
        for key in keys:
            first[key] = .9*first[key]+.1*gradients[key]
            second[key] = .999*second[key]+.001*gradients[key]**2
            params[key] -= lr*(first[key]/(1-.9**(step+1)))/(np.sqrt(second[key]/(1-.999**(step+1)))+1e-8)
    if frozen_encoder and not np.array_equal(initial['enc'], best['enc']):
        raise AssertionError('frozen encoder changed')
    return best, history


def sensitivity(h, ht, inverse=None):
    """Exact isotropic-noise energy and worst RMS gain of K=H_P H_T^+.

    These are READOUT sensitivities, not the GAD omitted-feature operator A.
    Small matrix products avoid materializing a test-by-training matrix.
    """
    inverse = pinverse(h)[0] if inverse is None else inverse
    covariance = ht.T @ ht/len(ht)
    inverse_covariance = inverse @ inverse.T
    energy = max(0., float(np.sum(covariance*inverse_covariance.T)))
    eig, vec = np.linalg.eigh(covariance)
    root = (vec*np.sqrt(np.maximum(eig, 0.))) @ vec.T
    largest = max(0., float(np.linalg.eigvalsh(root @ inverse_covariance @ root)[-1]))
    return dict(isotropic_prediction_energy=energy,
                readout_operator_rms_spectral=float(np.sqrt(largest)),
                worst_training_rms_gain=float(np.sqrt(len(h)*largest)))


def normalized_probe_gain(inverse, ht, perturbation):
    energy = float(np.mean(perturbation**2))
    if energy <= 1e-20:
        return dict(prediction_energy=float('nan'), training_energy=energy,
                    rms_gain=float('nan'), valid=0)
    propagated = ht @ (inverse @ perturbation)
    prediction_energy = float(np.mean(propagated**2))
    return dict(prediction_energy=prediction_energy, training_energy=energy,
                rms_gain=float(np.sqrt(prediction_energy/energy)), valid=1)
