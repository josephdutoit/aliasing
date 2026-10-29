"""Architecture-independent sparse-task training and frozen-map GAD checks."""
from __future__ import annotations

import importlib
import time

import numpy as np

from .controlled_superposition import importance, linear_oracle, problem, sample
from .mechanism_checks import normalized_probe_gain, pinverse, select_ridge, sensitivity
from .precision_gad import gad_metrics
from .representation_models import DenseRepresentation, make_model, train_model


TABLE_GROUPS = {
    'readouts': ('active_count', 'task_profile', 'architecture', 'method', 'sample_size', 'fit_split', 'estimator'),
    'geometry': ('active_count', 'task_profile', 'architecture', 'method'),
    'references': ('active_count', 'task_profile', 'architecture', 'method', 'sample_size', 'reference_size'),
    'probes': ('active_count', 'task_profile', 'architecture', 'method', 'sample_size', 'probe_kind'),
    'feature_recovery': ('active_count', 'task_profile', 'architecture', 'method', 'sample_size', 'estimator', 'feature'),
    'history': ('active_count', 'task_profile', 'architecture', 'method', 'step'),
}


def latent_errors(prediction, target, spec):
    error = ((prediction-target)*spec['scales']) @ spec['teacher'].T
    return np.mean(error**2, axis=0)/np.diag(spec['covariance'])


def output_floor(spec, dimension):
    eigenvalues = np.linalg.eigvalsh(spec['target_covariance'])
    return float(np.maximum(eigenvalues[:max(0, len(eigenvalues)-dimension)], 0.).sum()/len(eigenvalues))


def _variants(config, spec, xr, yr, weights, seed, tables, metadata, states, controls):
    for architecture in config['models']:
        # Architecture list ordering must not affect initialization or data.
        import hashlib
        salt = int.from_bytes(hashlib.sha256(architecture['name'].encode()).digest()[:4], 'little')
        initialization_seed = seed+7001+salt
        optimization_seed = seed+7003
        learned_relu = None
        for activation in ('linear', 'relu'):
            model = make_model(architecture, activation)
            initial = model.initialize(xr, spec['p'], initialization_seed)
            trained, history = train_model(model, initial, xr, yr, weights, config['steps'],
                config['batch_size'], config['learning_rate'], config['weight_decay'], optimization_seed)
            tag = metadata | dict(architecture=architecture['name'], method='learned_'+activation)
            tables['history'].extend(tag | row for row in history)
            for key, value in trained.items():
                states[f'{architecture["name"]}__{activation}__{key}'] = value
            yield architecture['name'], 'random_'+activation, 0, model, initial, False
            yield architecture['name'], 'learned_'+activation, 0, model, trained, True
            if activation == 'relu':
                learned_relu = model, trained
        model, trained = learned_relu
        yield architecture['name'], 'no_final_gate', 0, model.linearized(), trained, False
        desired = float(np.mean(model.preactivation(xr, trained) > 0))
        for rid in range(1, config['rotations']+1):
            q = np.linalg.qr(np.random.default_rng(seed+1009*rid+salt).normal(size=(model.width, model.width)))[0]
            rotated = model.rotate(trained, q)
            a, b = model.preactivation(xr[:64], trained), model.preactivation(xr[:64], rotated)
            gram_error = float(np.linalg.norm(a @ a.T-b @ b.T)/max(np.linalg.norm(a @ a.T), 1e-15))
            wa = np.linalg.svd(model.preactivation_weight(trained), compute_uv=False)
            wb = np.linalg.svd(model.preactivation_weight(rotated), compute_uv=False)
            spectrum_error = float(np.linalg.norm(wa-wb)/max(np.linalg.norm(wa), 1e-15))
            linear = model.linearized()
            pa = linear.features(xr[:64], trained) @ linear.native_head(trained)
            pb = linear.features(xr[:64], rotated) @ linear.native_head(rotated)
            linear_error = float(np.max(np.abs(pa-pb))/max(1., np.max(np.abs(pa))))
            for key, value in dict(rotation_gram_error=gram_error, rotation_spectrum_error=spectrum_error,
                                   linear_rotation_prediction_error=linear_error).items():
                controls[key] = max(controls.get(key, 0.), value)
            if max(gram_error, spectrum_error, linear_error) > 1e-9:
                raise AssertionError('rotation geometry/compensated linear predictor control failed')
            yield architecture['name'], 'rotated_relu', rid, model, rotated, False
            if hasattr(model, 'shift_final_bias'):
                shift = np.quantile(model.preactivation(xr, rotated), 1-desired)
                matched = model.shift_final_bias(rotated, shift)
                error = abs(float(np.mean(model.preactivation(xr, matched) > 0))-desired)
                controls['activity_match_error'] = max(controls.get('activity_match_error', 0.), error)
                # Quantile ties can make exact matching impossible; record it.
                yield architecture['name'], 'rotated_relu_rate_matched', rid, model, matched, False
    for rank in sorted(set([min(config['compression'], spec['p']), spec['p']])):
        old, _ = linear_oracle(spec, weights, rank, rank)
        model = DenseRepresentation((rank, rank), ('linear', 'linear'), (False, True), True)
        params = dict(w0=old['enc'], w1=old['dec'], b1=old['bias'])
        yield 'oracle_rank'+str(rank), 'population_linear_oracle', 0, model, params, False


def run_world(config, seed_index, active_count, task_profile):
    for module in config['model_modules']:
        importlib.import_module(module)
    started = time.monotonic()
    seed = config['seed']+seed_index*104729
    data_seed = seed+10000019*active_count
    spec = problem(seed, config['features'], config['input_dim'], active_count, config['task_basis'])
    xr, yr, _ = sample(spec, config['n_representation'], data_seed+1)
    xp, yp, lp = sample(spec, max(config['sample_sizes']), data_seed+2)
    xf, yf, lf = sample(spec, max(config['reference_sizes']), data_seed+3)
    xt, yt, _ = sample(spec, config['n_test'], data_seed+4)
    xv, yv, _ = sample(spec, config['n_validation'], data_seed+5)
    weights = importance(task_profile, spec['p'], config['secondary_weight'])
    meta = dict(seed=seed_index, active_count=active_count, task_profile=task_profile)
    tables = {name: [] for name in TABLE_GROUPS}
    states = dict(embedding=spec['embedding'], teacher=spec['teacher'], target_scales=spec['scales'],
                  latent_mean=spec['mean'], importance=weights)
    controls = {}
    orders = [np.arange(len(xp))]+[np.random.default_rng(data_seed+9001+draw).permutation(len(xp))
                                  for draw in range(1, config['draws'])]
    directions = np.random.default_rng(seed+8001).normal(size=(spec['p'], config['probe_repeats']))/np.sqrt(spec['p'])
    psi_p, psi_f = lp**2 @ directions, lf**2 @ directions
    center, scale = psi_f.mean(axis=0), np.maximum(psi_f.std(axis=0), 1e-12)
    psi_p, psi_f = (psi_p-center)/scale, (psi_f-center)/scale
    for architecture, method, rid, model, params, native in _variants(
            config, spec, xr, yr, weights, seed, tables, meta, states, controls):
        tag = meta | dict(architecture=architecture, method=method, rotation_id=rid)
        hp, hf, ht, hv, hr = [model.features(x, params) for x in (xp, xf, xt, xv, xr)]
        if any(h.shape != (len(x), model.width+1) or not np.allclose(h[:, 0], 1.) or not np.isfinite(h).all()
               for h, x in zip((hp, hf, ht, hv, hr), (xp, xf, xt, xv, xr))):
            raise ValueError('model features must be finite [intercept, hidden] matrices')
        references = {n: np.linalg.lstsq(hf[:n], yf[:n], rcond=None)[0] for n in config['reference_sizes']}
        reference = references[max(references)]
        _, rank, condition = pinverse(hf)
        pre = model.preactivation(xt, params)
        reference_prediction = ht @ reference
        errors = latent_errors(reference_prediction, yt, spec)
        tables['geometry'].append(tag | dict(hidden_width=model.width, nominal_readout_dimension=model.width+1,
            parameter_count=sum(value.size for value in params.values()), reference_rank=rank,
            reference_condition=condition, reference_test_mse=float(np.mean((reference_prediction-yt)**2)),
            reference_fit_mse=float(np.mean((hf @ reference-yf)**2)),
            output_subspace_lower_bound=output_floor(spec, model.width),
            positive_preactivation_fraction=float(np.mean(pre > 0)),
            latent_features_below_0p1=int(np.sum(errors < .1)), latent_worst_normalized_mse=float(errors.max()),
            latent_mean_normalized_mse=float(errors.mean()), input_second_moment=float(np.mean(np.sum(xt**2, axis=1)))))
        quadratic_reference = np.linalg.lstsq(hf, psi_f, rcond=None)[0]
        structured = psi_p-hp @ quadratic_reference

        def record(h, y, n, split, draw, estimator, candidate=None, penalty=float('nan'), validation_mse=float('nan'), inverse=None):
            if inverse is None:
                inverse, training_rank, training_condition = pinverse(h)
            else:
                _, training_rank, training_condition = pinverse(h)
            try:
                metrics = gad_metrics(h, y, ht, yt, reference, candidate, inverse, weights)
            except (AssertionError, ArithmeticError) as exc:
                raise ArithmeticError(f'{tag}, n={n}, split={split}, draw={draw}, estimator={estimator}: {exc}') from exc
            controls['extended_precision_evaluations'] = controls.get('extended_precision_evaluations', 0)+metrics['gad_extended_precision']
            prediction = ht @ (inverse @ y if candidate is None else candidate)
            feature_errors = latent_errors(prediction, yt, spec)
            metrics.update(latent_mean_normalized_mse=float(feature_errors.mean()),
                latent_worst_normalized_mse=float(feature_errors.max()),
                latent_features_below_0p1=int(np.sum(feature_errors < .1)))
            for key in ('prediction_identity_relative_error', 'risk_identity_relative_error', 'risk_identity_scaled_error',
                        'risk_identity_absolute_error', 'identity_cancellation_ratio'):
                controls[key] = max(controls.get(key, 0.), metrics[key])
            tables['readouts'].append(tag | dict(sample_size=n, fit_split=split, readout_draw=draw,
                estimator=estimator, penalty=penalty, validation_mse=validation_mse,
                training_rank=training_rank, training_condition=training_condition, **metrics))
            if split == 'independent' and n in config['detail_sample_sizes']:
                tables['feature_recovery'].extend(tag | dict(sample_size=n, readout_draw=draw, estimator=estimator,
                    feature=i, normalized_mse=float(value)) for i, value in enumerate(feature_errors))
            return inverse

        if native:
            record(hr, yr, len(hr), 'representation_training', 0, 'native', model.native_head(params))
            record(hr, yr, len(hr), 'representation_training', 0, 'mp')
        for draw, order in enumerate(orders):
            for n in config['sample_sizes']:
                idx = order[:n]
                h, y = hp[idx], yp[idx]
                inverse = record(h, y, n, 'independent', draw, 'mp')
                ridge, penalty, validation_mse = select_ridge(h, y, hv, yv, config['penalties'], weights)
                record(h, y, n, 'independent', draw, 'ridge_validation', ridge, penalty, validation_mse, inverse)
                if draw == 0 and n in config['detail_sample_sizes']:
                    largest_alias = ht @ (inverse @ (y-h @ reference))
                    for reference_size, head in references.items():
                        metrics = gad_metrics(h, y, ht, yt, head, inverse=inverse, weights=weights)
                        tables['references'].append(tag | dict(sample_size=n, reference_size=reference_size,
                            reference_prediction_difference_mse=float(np.mean((ht @ head-reference_prediction)**2)),
                            alias_vector_difference_mse=float(np.mean((ht @ (inverse @ (y-h @ head))-largest_alias)**2)), **metrics))
                # Sensitivities apply to the MP readout, independently of ridge.
                tables['probes'].append(tag | dict(sample_size=n, readout_draw=draw,
                    probe_kind='exact_readout_operator', probe_id=0, **sensitivity(h, ht, inverse)))
                noise = np.random.default_rng(data_seed+10007+draw*1009+n).normal(size=(n, config['probe_repeats']))
                for kind, perturbation in (('shared_gaussian', noise), ('shared_quadratic', psi_p[idx]),
                                            ('reference_residualized_quadratic', structured[idx])):
                    for repeat in range(config['probe_repeats']):
                        tables['probes'].append(tag | dict(sample_size=n, readout_draw=draw, probe_kind=kind,
                            probe_id=repeat, **normalized_probe_gain(inverse, ht, perturbation[:, repeat:repeat+1])))
                if draw == 0 and n == max(config['sample_sizes']):
                    delta = noise[:, :1]
                    fitted = inverse @ y
                    zero = ht @ (inverse @ (y+np.zeros_like(y)))-ht @ fitted
                    observed = ht @ (inverse @ (y+delta))-ht @ fitted
                    expected = ht @ (inverse @ delta)
                    control = float(np.max(np.abs(observed-expected))/max(1., np.max(np.abs(expected))))
                    controls['zero_perturbation_error'] = max(controls.get('zero_perturbation_error', 0.), float(np.max(np.abs(zero))))
                    controls['perturbation_response_relative_error'] = max(controls.get('perturbation_response_relative_error', 0.), control)
                    if control > 1e-7:
                        raise AssertionError('controlled readout perturbation response failed')
    return dict(tables=tables, states=states, controls=controls, seconds=time.monotonic()-started)
