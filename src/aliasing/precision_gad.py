"""Precision fallback for experiment-16 GAD contractions.

Preserve the legacy float64 results whenever their identity checks pass.
If those checks fail, reevaluate the SAME coefficients and terms in extended
precision. Never truncate additional singular values or change the estimator.
The authoritative prediction and test risk still use the original float64 head.
"""
from __future__ import annotations

import math
import numpy as np

from .mechanism_checks import gad_metrics as float64_gad_metrics, pinverse


def gad_metrics(h, y, ht, yt, reference, candidate=None, inverse=None, weights=None):
    inverse = pinverse(h)[0] if inverse is None else inverse
    fitted = inverse @ y
    candidate = fitted if candidate is None else candidate
    try:
        result = float64_gad_metrics(h, y, ht, yt, reference, candidate, inverse, weights)
        result['gad_extended_precision'] = 0
        return result
    except AssertionError as original:
        if str(original) != 'extended GAD identity failed':
            raise
        if np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps:
            raise ArithmeticError('GAD identity needs extended precision, unavailable on this platform') from original

    # The inverse and selected head are unchanged. Higher precision prevents
    # (H^+ H-I) beta* from acquiring a large spurious value when beta* has
    # large cancelling coefficients, even if predictions are near perfect.
    H, Y, P, Z, beta, inv, fit, head = [np.asarray(value, dtype=np.longdouble)
        for value in (h, y, ht, yt, reference, inverse, fitted, candidate)]
    remainder = Y-H @ beta
    vectors = dict(model_insufficiency=P @ beta-Z,
                   data_insufficiency=P @ ((inv @ H-np.eye(H.shape[1], dtype=np.longdouble)) @ beta),
                   mp_aliasing=P @ (inv @ remainder),
                   estimator_correction=P @ (head-fit))
    # Direct predictions are exactly those of the original selected estimator.
    prediction = ht @ candidate
    error = prediction-yt
    reconstructed = sum(vectors.values())
    prediction_gap = float(np.max(np.abs(np.asarray(error, dtype=np.longdouble)-reconstructed)))
    error_scale = max(1., float(np.max(np.abs(error))))
    result = {name: float(np.mean(value**2)) for name, value in vectors.items()}
    names = list(vectors)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            result['cross_'+left+'__'+right] = float(2*np.mean(vectors[left]*vectors[right]))
    result.update(test_mse=float(np.mean(error**2)), residual_train_mse=float(np.mean(remainder**2)),
        train_mse=float(np.mean((h @ candidate-y)**2)),
        classification_error=float(np.mean((prediction >= 0) != (yt >= 0))),
        prediction_identity_relative_error=prediction_gap/error_scale, gad_extended_precision=1)
    denominator = result['residual_train_mse']
    result['mp_residual_gain'] = float(np.sqrt(result['mp_aliasing']/denominator)) if denominator > 1e-20 else float('nan')
    terms = [value for key, value in result.items() if key in vectors or key.startswith('cross_')]
    if not np.isfinite(terms).all() or not np.isfinite(result['test_mse']):
        raise FloatingPointError('nonfinite GAD energies after extended-precision evaluation')
    # Evaluate the identity of the SERIALIZED float64 energies too: extended
    # precision must not conceal cancellation in the published CSV terms.
    absolute_error = abs(result['test_mse']-math.fsum(terms))
    component_scale = max(1., math.fsum(abs(value) for value in terms))
    prediction_scale = max(1., float(np.max(sum(np.abs(value) for value in vectors.values()))))
    result.update(risk_identity_absolute_error=absolute_error,
        risk_identity_relative_error=absolute_error/max(1., result['test_mse']),
        risk_identity_scaled_error=absolute_error/component_scale,
        identity_cancellation_ratio=component_scale/max(1., result['test_mse']))
    # Identical tolerances to the legacy checks; improve arithmetic, not gates.
    if (prediction_gap > 1e-7*error_scale+1e-12*prediction_scale or
            absolute_error > 1e-7*max(1., result['test_mse'])+1e-12*component_scale):
        raise AssertionError('extended GAD identity failed after precision retry: '
            f'prediction_gap={prediction_gap:.6g}, risk_gap={absolute_error:.6g}, '
            f'prediction_scale={prediction_scale:.6g}, component_scale={component_scale:.6g}')
    weights = np.ones(yt.shape[1]) if weights is None else weights
    result['weighted_test_mse'] = float(np.mean(error**2 @ weights)/weights.sum())
    return result
