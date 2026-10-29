"""Experiment-17 recovery: high-precision contractions, unchanged estimator.

Try the previous evaluator first. Decimal is needed when longdouble cannot
resolve products of a badly conditioned inverse and large reference head.
Canonical GAD terms are evaluated independently. Float64 fit/prediction
rounding is a SEPARATE arithmetic term, never assigned to aliasing or data
insufficiency. The experiment-16 evaluator remains unchanged.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
import math
import numpy as np

from .mechanism_checks import pinverse
from .precision_gad import gad_metrics as previous_metrics


def _vectors(h, y, ht, yt, reference, candidate, inverse, fitted, dtype):
    H, Y, P, Z, beta, inv, fit, head = [np.asarray(a, dtype=dtype)
        for a in (h, y, ht, yt, reference, inverse, fitted, candidate)]
    return dict(model_insufficiency=P @ beta-Z,
                data_insufficiency=P @ ((inv @ H-np.eye(H.shape[1], dtype=dtype)) @ beta),
                mp_aliasing=P @ (inv @ (Y-H @ beta)),
                estimator_correction=P @ (head-fit))


def _decimal(array):
    array = np.asarray(array, dtype=float)
    if not np.isfinite(array).all():
        raise FloatingPointError('nonfinite input to high-precision GAD')
    return np.array([Decimal.from_float(float(x)) for x in array.flat], dtype=object).reshape(array.shape)


def _as_float(array):
    return np.asarray(array, dtype=float)


def decimal_components(h, y, ht, yt, reference, candidate, inverse, fitted, digits=80):
    """Return independently evaluated GAD vectors plus implementation rounding.

The arithmetic term is computed from fit rounding, prediction rounding and
subtraction rounding. It is NOT defined as an unexplained identity residual.
"""
    prediction = ht @ candidate
    error = prediction-yt
    with localcontext() as context:
        context.prec = digits
        H, Y, P, Z, beta, inv, fit, head = [_decimal(a)
            for a in (h, y, ht, yt, reference, inverse, fitted, candidate)]
        exact_fit = inv @ Y
        coefficients = dict(
            data_insufficiency=((inv @ H)-np.eye(H.shape[1], dtype=int)) @ beta,
            mp_aliasing=inv @ (Y-H @ beta),
            estimator_correction=head-fit)
        vectors = dict(model_insufficiency=P @ beta-Z)
        vectors.update({name: P @ value for name, value in coefficients.items()})
        exact_prediction = P @ head
        # Three independently identifiable operations in the original float64
        # estimator. Their sum closes the implemented prediction identity.
        fit_roundoff = P @ (fit-exact_fit)
        prediction_roundoff = _decimal(prediction)-exact_prediction
        subtraction_roundoff = _decimal(error)-(_decimal(prediction)-Z)
        vectors['arithmetic_roundoff'] = fit_roundoff+prediction_roundoff+subtraction_roundoff
        gap = _as_float(sum(vectors.values())-_decimal(error))
        scale = max(1., float(np.max(sum(np.abs(_as_float(v)) for v in vectors.values()))))
        if np.max(np.abs(gap)) > 1e-7*max(1., float(np.max(np.abs(error))))+1e-12*scale:
            raise ArithmeticError(f'Decimal GAD contraction failed at {digits} digits')
        return {name: _as_float(value) for name, value in vectors.items()}


def _serialize(vectors, h, y, ht, yt, reference, candidate, weights, digits):
    error = ht @ candidate-yt
    result = {name: float(np.mean(value.astype(np.longdouble)**2)) for name, value in vectors.items()}
    names = list(vectors)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            result['cross_'+left+'__'+right] = float(2*np.mean(vectors[left].astype(np.longdouble)*vectors[right]))
    prediction_gap = float(np.max(np.abs(np.asarray(error, dtype=np.longdouble)-sum(v.astype(np.longdouble) for v in vectors.values()))))
    terms = [value for key, value in result.items() if key in vectors or key.startswith('cross_')]
    risk = float(np.mean(error**2))
    absolute = abs(risk-math.fsum(terms))
    component_scale = max(1., math.fsum(abs(value) for value in terms))
    prediction_scale = max(1., float(np.max(sum(np.abs(v) for v in vectors.values()))))
    if (not np.isfinite(terms).all() or not np.isfinite(risk) or
            prediction_gap > 1e-7*max(1., float(np.max(np.abs(error))))+1e-12*prediction_scale or
            absolute > 1e-7*max(1., risk)+1e-12*component_scale):
        raise ArithmeticError('serialized high-precision GAD identity failed')
    # Residual energy is derived from the same high-precision reference.
    with localcontext() as context:
        context.prec = digits
        residual = _as_float(_decimal(y)-_decimal(h) @ _decimal(reference))
    residual_energy = float(np.mean(residual**2))
    result.update(test_mse=risk, train_mse=float(np.mean((h @ candidate-y)**2)),
        residual_train_mse=residual_energy,
        classification_error=float(np.mean(((ht @ candidate) >= 0) != (yt >= 0))),
        prediction_identity_relative_error=prediction_gap/max(1., float(np.max(np.abs(error)))),
        risk_identity_absolute_error=absolute, risk_identity_relative_error=absolute/max(1., risk),
        risk_identity_scaled_error=absolute/component_scale,
        identity_cancellation_ratio=component_scale/max(1., risk),
        gad_extended_precision=1, gad_decimal_digits=digits,
        arithmetic_roundoff_measured=1,
        gad_arithmetic_reliable=int(np.sqrt(result['arithmetic_roundoff']) <= 1e-7*max(1., np.sqrt(risk))),
        arithmetic_roundoff_to_risk=result['arithmetic_roundoff']/max(risk, 1e-300),
        mp_residual_gain=float(np.sqrt(result['mp_aliasing']/residual_energy)) if residual_energy > 1e-20 else float('nan'))
    weights = np.ones(yt.shape[1]) if weights is None else weights
    result['weighted_test_mse'] = float(np.mean(error**2 @ weights)/weights.sum())
    return result


def gad_metrics(h, y, ht, yt, reference, candidate=None, inverse=None, weights=None, return_vectors=False):
    inverse = pinverse(h)[0] if inverse is None else inverse
    fitted = inverse @ y
    candidate = fitted if candidate is None else candidate
    try:
        result = previous_metrics(h, y, ht, yt, reference, candidate, inverse, weights)
    except (AssertionError, ArithmeticError) as error:
        if 'GAD' not in str(error) and 'precision' not in str(error):
            raise
        for digits in (80, 160):
            try:
                vectors = decimal_components(h, y, ht, yt, reference, candidate, inverse, fitted, digits)
                result = _serialize(vectors, h, y, ht, yt, reference, candidate, weights, digits)
                break
            except ArithmeticError:
                if digits == 160:
                    raise
    else:
        # Preserve every previously accepted metric bit-for-bit. These fields
        # declare the earlier tolerance passed; no high-precision correction
        # was measured on this path.
        result.update(gad_decimal_digits=0, gad_arithmetic_reliable=1)
        result.update(arithmetic_roundoff=0., arithmetic_roundoff_measured=0, arithmetic_roundoff_to_risk=0.)
        for name in ('model_insufficiency', 'data_insufficiency', 'mp_aliasing', 'estimator_correction'):
            result['cross_'+name+'__arithmetic_roundoff'] = 0.
        vectors = None
        if return_vectors:
            dtype = np.longdouble if result['gad_extended_precision'] else float
            vectors = {k: np.asarray(v, dtype=float) for k, v in
                _vectors(h, y, ht, yt, reference, candidate, inverse, fitted, dtype).items()}
    return (result, vectors) if return_vectors else result
