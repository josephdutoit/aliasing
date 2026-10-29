"""Learned ReLU geometry worlds, interventions and canonical GAD evaluation.

Numerical quadrature is an estimated population geometry. No search is called
a global optimum. Geometry is learned before independent heads are fitted.
"""
from __future__ import annotations

import time
import numpy as np

from .learned_geometry_data import moments, pair_quadrature, sample, support_labels
from .learned_geometry_training import initialize, loss_grad, train
from .mechanism_checks import pinverse, select_ridge
from .learned_geometry_precision import gad_metrics
from .representation_models import make_model


TABLES = ('geometry', 'readouts', 'starts', 'history', 'gates', 'strata', 'references', 'parameters', 'feature_errors')


def weighted_reference(h, y, mass):
    root = np.sqrt(mass)
    return pinverse(h*root[:, None])[0] @ (y*root[:, None])


def evaluation_data(spec, config, seed):
    if spec['features'] == 2 and spec['law'] == 'pair':
        references = [pair_quadrature(spec, order) for order in config['reference_orders']]
        test = pair_quadrature(spec, config['evaluation_order'])
    else:
        x, y = sample(spec, max(config['reference_sizes']), seed+31)
        references = [(x[:n], y[:n], np.full(n, 1/n)) for n in config['reference_sizes']]
        x, y = sample(spec, config['n_test'], seed+37)
        test = x, y, np.full(len(x), 1/len(x))
    return references, test


def diagnostics(model, params, x, y, mass, reference, weights, config):
    h = model.features(x, params)
    residual = y-h @ reference
    root = np.sqrt(mass)
    _, singular, vh = np.linalg.svd(h*root[:, None], full_matrices=False)
    tolerance = max(h.shape)*np.finfo(float).eps*singular[0]
    rank = int(np.sum(singular > tolerance))
    coordinates = h @ (vh[:rank].T/singular[:rank])
    leverage = np.sum(coordinates**2, axis=1)
    residual_energy = np.mean(residual**2, axis=1)
    model_energy = float(mass @ residual_energy)
    J = float(mass @ (residual_energy*leverage))
    order = np.argsort(leverage)[::-1]
    cumulative = np.cumsum(mass[order])
    # Fractional inclusion gives exactly 10% probability mass, including atoms.
    fractions = np.clip((.1-(cumulative-mass[order]))/mass[order], 0., 1.)
    top_residual = float(np.sum(mass[order]*fractions*residual_energy[order]))
    pre = model.preactivation(x, params)
    gate = (pre > 0) if model.activations[-1] == 'relu' else np.ones_like(pre, dtype=bool)
    patterns, inverse = np.unique(np.packbits(gate, axis=1), axis=0, return_inverse=True)
    pattern_mass = np.bincount(inverse, weights=mass)
    _, gradient = loss_grad(model, params, x, y, weights, config['decay'], mass)
    encoder = params['w0']
    gram = encoder @ encoder.T
    norms = np.sqrt(np.maximum(np.diag(gram), 0.))
    cosine = np.divide(gram, np.outer(norms, norms), out=np.zeros_like(gram), where=np.outer(norms, norms) > 1e-24)
    np.fill_diagonal(cosine, 0.)
    angle = float(np.mod(np.arctan2(encoder[1, 0], encoder[0, 0]), np.pi)) if encoder.shape == (2, 1) else float('nan')
    # z -> -z is a global factorization symmetry. Keep relative feature signs
    # and feature identities: folding theta and pi-theta would erase the
    # same-sign versus opposite-sign geometry this study is meant to measure.
    angle_canonical = angle
    output = dict(model_insufficiency=model_energy, residual_leverage_J=J,
        residual_leverage_ratio=J/(rank*model_energy) if model_energy > 1e-20 else float('nan'),
        residual_top_leverage_decile_share=top_residual/model_energy if model_energy > 1e-20 else float('nan'),
        readout_rank=rank, reference_condition=float(singular[0]/singular[rank-1]),
        reference_normal_equation_norm=float(np.linalg.norm(h.T @ (mass[:, None]*residual))),
        population_gradient_norm=float(np.sqrt(sum(np.sum(v**2) for v in gradient.values()))),
        population_encoder_gradient_norm=float(np.linalg.norm(gradient['w0'])),
        population_gate_gradient_norm=float(np.sqrt(np.sum(gradient['w1']**2)+np.sum(gradient['b1']**2))),
        encoder_angle=angle, encoder_angle_canonical=angle_canonical,
        encoder_first_pair_cosine=float(cosine[0, 1]),
        encoder_max_abs_cosine=float(np.abs(cosine).max()),
        encoder_smallest_singular=float(np.linalg.svd(encoder, compute_uv=False)[-1]),
        active_gate_fraction=float(np.mean(mass @ gate)), gate_pattern_count=len(patterns),
        gate_pattern_entropy=float(-np.sum(pattern_mass*np.log(np.maximum(pattern_mass, 1e-300)))),
        leverage_mean=float(mass @ leverage), leverage_max=float(leverage.max()),
        weighted_model_insufficiency=float(mass @ (residual**2 @ weights)/weights.sum()))
    return output, h, residual, leverage, gate


def interventions(model, params, x_train, mass, seed, config):
    """Every intervened representation gets a fresh independent head."""
    yield 'learned_relu', model, params, {}
    yield 'gate_removed', model.linearized(), params, {}
    pre = model.preactivation(x_train, params)
    probability = np.full(len(x_train), 1/len(x_train)) if mass is None else mass
    center = probability @ pre
    scale = np.sqrt(probability @ (pre-center)**2)
    for offset in config['threshold_offsets']:
        shifted = {key: value.copy() for key, value in params.items()}
        shifted['b1'] += offset*scale
        yield 'threshold_'+str(offset), model, shifted, dict(threshold_offset=offset)
    for rid in range(config['rotations']):
        q = np.linalg.qr(np.random.default_rng(seed+101+rid).normal(size=(model.width, model.width)))[0]
        rotated = model.rotate(params, q)
        a, b = model.preactivation(x_train[:64], params), model.preactivation(x_train[:64], rotated)
        gram_error = float(np.linalg.norm(a @ a.T-b @ b.T)/max(np.linalg.norm(a @ a.T), 1e-20))
        linear = model.linearized()
        linear_error = float(np.max(np.abs(linear.features(x_train[:64], params) @ linear.native_head(params)-
            linear.features(x_train[:64], rotated) @ linear.native_head(rotated))))
        if gram_error > 1e-9 or linear_error > 1e-8:
            raise AssertionError('rotation control failed')
        yield 'rotated_'+str(rid), model, rotated, dict(rotation_id=rid, rotation_gram_error=gram_error,
                                                       linear_rotation_error=linear_error)


def run_world(config, world):
    started = time.monotonic()
    spec = world['distribution']
    seed = config['seed']+world['seed']*104729
    # Common data/initialization streams across matched interventions and tasks.
    training = world['training']
    if training == 'quadrature':
        xr, yr, mass = pair_quadrature(spec, config['training_order'])
    else:
        xr, yr = sample(spec, world['n_representation'], seed+11)
        mass = None
    references, (xt, yt, test_mass) = evaluation_data(spec, config, seed)
    xp, yp = sample(spec, max(config['sample_sizes']), seed+41)
    xv, yv = sample(spec, config['n_validation'], seed+43)
    permutations = [np.arange(len(xp))]+[np.random.default_rng(seed+47+i).permutation(len(xp)) for i in range(config['draws']-1)]
    weights = np.array(world.get('task_weights', [1.]*spec['features']))
    architecture = dict(family='bottleneck', compression=world['compression'], width=world['width'])
    tables, states = {name: [] for name in TABLES}, {}
    metadata = {key: world[key] for key in ('experiment', 'scenario', 'seed', 'training', 'compression', 'width', 'n_representation')}
    metadata.update(features=spec['features'], coactivation=spec.get('coactivation', -1.),
                    eta=spec.get('eta', -1.), rho=spec.get('rho', -1.), decay=world['decay'],
                    marginal_first=spec.get('marginals', [-1., -1.])[0],
                    marginal_second=spec.get('marginals', [-1., -1.])[1],
                    task_weight_first=float(weights[0]), task_weight_last=float(weights[-1]),
                    fixed_angle=world.get('fixed_angle', -1.))
    local_config = config | dict(decay=world['decay'])
    best_relu = None

    def evaluate(method, model, params, extra=None, detailed=True, native=False):
        tag = metadata | dict(method=method) | (extra or {})
        ref_heads = [weighted_reference(model.features(x, params), y, w) for x, y, w in references]
        reference = ref_heads[-1]
        geometry, ht, residual, leverage, gate = diagnostics(model, params, xt, yt, test_mass, reference, weights, local_config)
        tables['geometry'].append(tag | geometry)
        for feature in range(spec['features']):
            tables['feature_errors'].append(tag | dict(sample_size=0, estimator='reference', target_feature=feature,
                model_insufficiency=float(test_mass @ residual[:, feature]**2)))
        key = method.replace('.', '_').replace('-', 'minus')
        for name, value in params.items():
            states[key+'__'+name] = value
        # Raw parameter rows permit symmetry-aware inspection and reconstruction.
        for name, value in params.items():
            if name in ('head', 'out_bias'):
                continue
            for index in np.ndindex(value.shape):
                tables['parameters'].append(tag | dict(parameter=name, index='_'.join(map(str, index)), value=float(value[index])))
        pre = model.preactivation(xt, params)
        effective = model.preactivation_weight(params)
        for j in range(model.width):
            norm = np.linalg.norm(effective[:, j])
            row = tag | dict(gate=j, active_probability=float(test_mass @ gate[:, j]),
                effective_normal_norm=float(norm), signed_offset=float(-params['b1'][j]/norm) if norm > 1e-14 else float('nan'),
                near_boundary_mass=float(test_mass @ (np.abs(pre[:, j]) < .02*max(np.sqrt(test_mass @ pre[:, j]**2), 1e-14))))
            row.update({'normal_'+str(i): float(v/norm) if norm > 1e-14 else float('nan') for i, v in enumerate(effective[:, j])})
            tables['gates'].append(row)
        root = np.sqrt(len(xt)*test_mass)
        he, ye = ht*root[:, None], yt*root[:, None]
        hp, hv = model.features(xp, params), model.features(xv, params)
        pattern_bits = np.packbits(gate, axis=1)
        pattern_keys, pattern_index = np.unique(pattern_bits, axis=0, return_inverse=True)
        pattern_probability = np.bincount(pattern_index, weights=test_mass)
        pattern_bytes = [row.tobytes() for row in pattern_keys]
        pool_gate = model.preactivation(xp, params) > 0 if model.activations[-1] == 'relu' else np.ones((len(xp), model.width), dtype=bool)
        sizes = config['sample_sizes'] if detailed else [min(config['sample_sizes'], key=lambda n: abs(n-128))]
        for draw, order in enumerate(permutations if detailed else permutations[:1]):
            for n in sizes:
                ix = order[:n]
                h, y = hp[ix], yp[ix]
                inverse, rank, condition = pinverse(h)
                observed = {row.tobytes() for row in np.packbits(pool_gate[ix], axis=1)}
                missing_mass = float(sum(probability for key, probability in zip(pattern_bytes, pattern_probability) if key not in observed))
                coverage = dict(unseen_gate_pattern_mass=missing_mass,
                                never_active_gate_count=int(np.sum(~pool_gate[ix].any(axis=0))))
                candidates = [('mp', None, float('nan'))]
                if detailed:
                    candidate, penalty, _ = select_ridge(h, y, hv, yv, config['penalties'], weights)
                    candidates.append(('ridge_validation', candidate, penalty))
                for estimator, candidate, penalty in candidates:
                    need_vectors = estimator == 'mp' and draw == 0 and n in (sizes[0], min(sizes, key=lambda n: abs(n-128)), sizes[-1])
                    evaluated = gad_metrics(h, y, he, ye, reference, candidate=candidate, inverse=inverse, weights=weights, return_vectors=need_vectors)
                    values = evaluated[0] if need_vectors else evaluated
                    if need_vectors:
                        component_vectors = evaluated[1]
                        mp_vectors = {name: vector/root[:, None] for name, vector in component_vectors.items()}
                    values.update(n_times_aliasing=n*values['mp_aliasing'],
                        aliasing_to_J_ratio=n*values['mp_aliasing']/geometry['residual_leverage_J'] if geometry['residual_leverage_J'] > 1e-20 else float('nan'))
                    tables['readouts'].append(tag | dict(sample_size=n, draw=draw, estimator=estimator,
                        penalty=penalty, training_rank=rank, training_condition=condition,
                        fit_split='independent', **coverage, **values))
                if draw == 0 and n in (sizes[0], min(sizes, key=lambda n: abs(n-128)), sizes[-1]):
                    alias = mp_vectors['mp_aliasing']
                    data = mp_vectors['data_insufficiency']
                    model_residual = mp_vectors['model_insufficiency']
                    arithmetic = mp_vectors.get('arithmetic_roundoff', np.zeros_like(alias))
                    # Match the same weighted float64 predictor used for the
                    # authoritative global metrics, including its rounding.
                    error = (he @ (inverse @ y)-ye)/root[:, None]
                    for feature in range(spec['features']):
                        vectors = [data[:, feature], alias[:, feature], model_residual[:, feature], arithmetic[:, feature]]
                        energies = [float(test_mass @ v**2) for v in vectors]
                        tables['feature_errors'].append(tag | dict(sample_size=n, estimator='mp', target_feature=feature,
                            test_mse=float(test_mass @ error[:, feature]**2), data_insufficiency=energies[0],
                            mp_aliasing=energies[1], model_insufficiency=energies[2],
                            arithmetic_roundoff=energies[3],
                            cross_total=float(sum(2*(test_mass @ (a*b)) for i, a in enumerate(vectors) for b in vectors[i+1:]))))
                    labels = support_labels(xt)
                    for label in np.unique(labels):
                        selected = labels == label
                        probability = test_mass[selected].sum()
                        conditional_mass = test_mass[selected]/probability
                        tables['strata'].append(tag | dict(sample_size=n, stratum=int(label), probability=float(probability),
                            model_insufficiency=float(conditional_mass @ np.mean(model_residual[selected]**2, axis=1)),
                            mp_aliasing=float(conditional_mass @ np.mean(alias[selected]**2, axis=1)),
                            data_insufficiency=float(conditional_mass @ np.mean(data[selected]**2, axis=1)),
                            arithmetic_roundoff=float(conditional_mass @ np.mean(arithmetic[selected]**2, axis=1)),
                            test_mse=float(conditional_mass @ np.mean(error[selected]**2, axis=1)),
                            cross_total=float(sum(2*(conditional_mass @ np.mean(a[selected]*b[selected], axis=1))
                                for i, a in enumerate([data, alias, model_residual, arithmetic])
                                for b in [data, alias, model_residual, arithmetic][i+1:])),
                            leverage=float(conditional_mass @ leverage[selected])))
                    if detailed:
                        base_alias = alias
                        for level, head in zip(config['reference_orders'] if spec['features'] == 2 else config['reference_sizes'], ref_heads):
                            tables['references'].append(tag | dict(sample_size=n, reference_level=level,
                                prediction_difference_mse=float(test_mass @ np.mean((ht @ (head-reference))**2, axis=1)),
                                alias_vector_difference_mse=float(test_mass @ np.mean((ht @ (inverse @ (y-h @ head))-base_alias)**2, axis=1))))
        if native:
            hr = model.features(xr, params)
            # Quadrature objectives have a WEIGHTED native-fit baseline; do not
            # pretend quadrature nodes are iid readout samples.
            if mass is None:
                values = gad_metrics(hr, yr, he, ye, reference, candidate=model.native_head(params), weights=weights)
                tables['readouts'].append(tag | dict(sample_size=len(xr), draw=0, estimator='native',
                    fit_split='representation_training', **values))
        return geometry

    modes = ['joint']
    if world['experiment'] == 'ablations':
        modes += ['fixed_random_encoder', 'fixed_thresholds']
    for activation in ('linear', 'relu'):
        model = make_model(architecture, activation)
        for mode in modes if activation == 'relu' else ['joint']:
            best, best_loss = None, float('inf')
            for start in range(config['restarts']):
                initial = initialize(model, xr, mass, spec['features'], seed+1009*start+101,
                                     world.get('fixed_angle'))
                frozen = []
                if mode == 'fixed_random_encoder' or 'fixed_angle' in world:
                    frozen.append('w0')
                if mode == 'fixed_thresholds':
                    frozen.append('b1')
                params, history = train(model, initial, xr, yr, mass, weights, local_config,
                                        seed+211, frozen)
                value, gradients = loss_grad(model, params, xr, yr, weights, world['decay'], mass)
                start_tag = metadata | dict(activation=activation, mode=mode, start=start)
                tables['starts'].append(start_tag | dict(training_objective=value,
                    free_gradient_norm=float(np.sqrt(sum(np.sum(g**2) for k, g in gradients.items() if k not in frozen)))))
                tables['history'].extend(start_tag | row for row in history)
                states[f'{activation}_{mode}_start{start}__initial_w0'] = initial['w0']
                for name, array in params.items():
                    states[f'{activation}_{mode}_start{start}__'+name] = array
                if value < best_loss:
                    best_loss, best = value, params
                # Preserve every joint restart's geometry; never count starts
                # as independent seeds in summaries.
                if mode == 'joint' and activation == 'relu' and config['evaluate_starts']:
                    evaluate('restart_'+str(start), model, params, dict(start=start), detailed=False)
            if activation == 'linear':
                evaluate('learned_linear', model, best, dict(best_training_objective=best_loss), native=True)
            elif mode != 'joint':
                evaluate(mode, model, best, dict(best_training_objective=best_loss), native=True)
            else:
                best_relu = model, best, best_loss
    model, params, value = best_relu
    variants = interventions(model, params, xr, mass, seed, config) if world['experiment'] == 'ablations' else [('learned_relu', model, params, {})]
    for method, variant, state, extra in variants:
        evaluate(method, variant, state, extra | dict(best_training_objective=value), native=method == 'learned_relu')
    # Exact population linear baselines for every task; an arbitrary PCA basis
    # within a degenerate eigenspace is NOT called the unique optimum.
    mean, covariance = moments(spec)
    standardized = covariance/np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    eigenvalues = np.linalg.eigvalsh(standardized*np.sqrt(np.outer(weights, weights)))
    floor = float(np.maximum(eigenvalues[:max(0, len(eigenvalues)-world['compression'])], 0).sum()/weights.sum())
    for row in tables['geometry']:
        row['weighted_rank_m_linear_floor'] = floor
    states['latent_mean'], states['latent_covariance'], states['task_weights'] = mean, covariance, weights
    return dict(tables=tables, states=states, seconds=time.monotonic()-started)
