"""Scientific controls for learned geometry, independent of optional torch."""
import importlib.util
import itertools
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from aliasing.benchmark_results import csv_write, read_rows
from aliasing.learned_geometry import diagnostics, run_world, weighted_reference
from aliasing.learned_geometry_data import moments, pair_probabilities, pair_quadrature, sample
from aliasing.learned_geometry_plan import defaults, scenarios, validate, worlds
from aliasing.learned_geometry_results import seed_means, summarize
from aliasing.learned_geometry_training import initialize, loss_grad, train
from aliasing.mechanism_checks import pinverse
from aliasing.precision_gad import gad_metrics
from aliasing.representation_models import make_model


def runner():
    path = Path(__file__).resolve().parents[1]/'experiments/17_learned_relu_geometry.py'
    spec = importlib.util.spec_from_file_location('learned_geometry_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pair_quadrature_preserves_exact_known_moments_and_normalized_tasks():
    for marginals, fraction in itertools.product([(.1, .1), (.1, .3), (.8, .7)], [0., .23, 1.]):
        spec = dict(law='pair', features=2, marginals=marginals, coactivation=fraction)
        x, y, mass = pair_quadrature(spec, 5)
        mean, covariance = moments(spec)
        assert np.isclose(mass.sum(), 1.) and np.all(mass > 0)
        assert np.allclose(mass @ x, mean, atol=1e-14)
        assert np.allclose((x.T*mass) @ x-np.outer(mean, mean), covariance, atol=1e-14)
        assert np.allclose(mass @ y, 0., atol=1e-14)
        assert np.allclose(mass @ y**2, 1., atol=1e-14)
        assert np.all(pair_probabilities(marginals, fraction) >= -1e-15)


def test_sampler_matches_pair_and_parity_moments_and_higher_order_control():
    for law in ('pair', 'parity'):
        spec = dict(law=law, features=4, marginals=(.1, .3), coactivation=.7, rho=.6, eta=.2)
        x, y = sample(spec, 200000, 37)
        mean, covariance = moments(spec)
        assert np.max(abs(x.mean(0)-mean)) < .006
        assert np.max(abs(np.cov(x, rowvar=False, ddof=0)-covariance)) < .006
        assert np.max(abs(y.var(0)-1)) < .025
    patterns = np.array(list(itertools.product([0, 1], repeat=4)))
    probabilities = []
    for eta in (0., 1.):
        weights = .6*np.where(patterns.sum(1) % 2 == 0, eta/8, (1-eta)/8)
        weights[0] += .4
        probabilities.append(weights)
    for size in (1, 2, 3):
        for indices in itertools.combinations(range(4), size):
            event = np.all(patterns[:, indices] == 1, axis=1)
            assert np.isclose(probabilities[0] @ event, probabilities[1] @ event)
    assert probabilities[0][-1] == 0 and probabilities[1][-1] > 0


def test_probability_weighted_gradients_match_finite_differences_and_uniform_legacy():
    rng = np.random.default_rng(91)
    x, y = rng.normal(size=(11, 2)), rng.normal(size=(11, 2))
    mass = rng.uniform(.1, 2, len(x)); mass /= mass.sum()
    weights = np.array([1., .2])
    for activation in ('linear', 'relu'):
        model = make_model(dict(family='bottleneck', compression=1, width=3), activation)
        params = model.initialize(x, 2, 78)
        params['b1'] += .23456
        value, gradient = loss_grad(model, params, x, y, weights, .013)
        old, old_gradient = model.loss_grad(params, x, y, weights, .013)
        assert np.isclose(value, old)
        assert all(np.allclose(gradient[k], old_gradient[k]) for k in gradient)
        _, gradient = loss_grad(model, params, x, y, weights, .013, mass)
        for name, array in params.items():
            for index in np.ndindex(array.shape):
                before = array[index]
                array[index] = before+1e-6
                plus = loss_grad(model, params, x, y, weights, .013, mass)[0]
                array[index] = before-1e-6
                minus = loss_grad(model, params, x, y, weights, .013, mass)[0]
                array[index] = before
                assert np.isclose((plus-minus)/2e-6, gradient[name][index], atol=2e-7, rtol=2e-5), name


def test_weighted_gad_preserves_canonical_terms_in_both_rank_regimes():
    rng = np.random.default_rng(123)
    ht = rng.normal(size=(13, 5)); yt = rng.normal(size=(13, 2))
    mass = rng.uniform(.1, 1., 13); mass /= mass.sum()
    reference = rng.normal(size=(5, 2))
    root = np.sqrt(len(ht)*mass)
    for n in (2, 9):
        h, y = rng.normal(size=(n, 5)), rng.normal(size=(n, 2))
        inverse = pinverse(h)[0]
        values = gad_metrics(h, y, ht*root[:, None], yt*root[:, None], reference)
        expected_alias = ht @ (inverse @ (y-h @ reference))
        expected_data = ht @ ((inverse @ h-np.eye(5)) @ reference)
        assert np.isclose(values['mp_aliasing'], mass @ np.mean(expected_alias**2, axis=1))
        assert np.isclose(values['data_insufficiency'], mass @ np.mean(expected_data**2, axis=1))
        assert np.isclose(values['model_insufficiency'], mass @ np.mean((ht @ reference-yt)**2, axis=1))
        assert values['risk_identity_scaled_error'] < 1e-12
        if n > 5:
            assert values['data_insufficiency'] < 1e-25


def test_training_freezes_parameters_and_preserves_best_training_checkpoint():
    config = defaults('smoke') | dict(steps=120, checkpoint_every=20, decay=0.)
    spec = dict(law='pair', features=2, marginals=[.3, .3], coactivation=.2)
    x, y, mass = pair_quadrature(spec, 8)
    model = make_model(dict(family='bottleneck', compression=1, width=4))
    params = initialize(model, x, mass, 2, 6, .3)
    final, history = train(model, params, x, y, mass, np.ones(2), config, 7, ['w0', 'b1'])
    assert np.array_equal(final['w0'], params['w0']) and np.array_equal(final['b1'], params['b1'])
    value = loss_grad(model, final, x, y, np.ones(2), 0., mass)[0]
    assert np.isclose(value, min(row['objective'] for row in history))
    assert value < history[0]['objective']


def test_residual_leverage_and_reference_orthogonality_use_probability_measure():
    config = defaults('smoke') | dict(decay=0.)
    spec = dict(law='pair', features=2, marginals=[.2, .3], coactivation=.7)
    x, y, mass = pair_quadrature(spec, 12)
    model = make_model(dict(family='bottleneck', compression=1, width=2))
    params = initialize(model, x, mass, 2, 99)
    h = model.features(x, params)
    reference = weighted_reference(h, y, mass)
    result, _, residual, leverage, _ = diagnostics(model, params, x, y, mass, reference, np.ones(2), config)
    gram = (h.T*mass) @ h
    direct = np.sum((h @ np.linalg.pinv(gram))*h, axis=1)
    assert np.allclose(leverage, direct, atol=1e-10)
    assert np.isclose(mass @ leverage, result['readout_rank'])
    assert result['reference_normal_equation_norm'] < 1e-12
    assert np.isclose(result['residual_leverage_J'], mass @ (np.mean(residual**2, axis=1)*direct))
    # Retain opposite-sign encoder geometry; do not fold it onto same-sign.
    params['w0'] = np.array([[-1.], [1.]])
    result = diagnostics(model, params, x, y, mass, weighted_reference(model.features(x, params), y, mass), np.ones(2), config)[0]
    assert result['encoder_angle_canonical'] > np.pi/2
    assert result['encoder_first_pair_cosine'] < 0


def test_world_support_strata_reconstruct_global_gad_and_starts_are_retained():
    config = defaults('smoke') | dict(num_seeds=2, steps=8)
    world = worlds(config)[0]
    result = run_world(config, world)
    readouts = [row for row in result['tables']['readouts'] if row['estimator'] == 'mp' and row['draw'] == 0 and row['method'] == 'learned_relu']
    for row in readouts:
        strata = [r for r in result['tables']['strata'] if r['method'] == row['method'] and r['sample_size'] == row['sample_size']]
        if not strata:
            continue  # Detailed conditional rows are recorded at selected sizes.
        assert np.isclose(sum(r['probability'] for r in strata), 1.)
        for metric in ('test_mse', 'mp_aliasing', 'model_insufficiency', 'data_insufficiency'):
            assert np.isclose(sum(r['probability']*r[metric] for r in strata), row[metric])
            features = [r for r in result['tables']['feature_errors'] if r['method'] == row['method'] and r['sample_size'] == row['sample_size'] and r['estimator'] == 'mp']
            assert np.isclose(np.mean([r[metric] for r in features]), row[metric])
        for r in strata:
            assert np.isclose(r['test_mse'], r['model_insufficiency']+r['data_insufficiency']+r['mp_aliasing']+r['cross_total'])
    assert len(result['tables']['starts']) == config['restarts']*2
    assert all(f'relu_joint_start{s}__w0' in result['states'] for s in range(config['restarts']))


def test_readout_draws_are_averaged_within_seed_and_paired_by_seed():
    config = defaults('smoke')
    result = run_world(config | dict(steps=2), worlds(config)[0])
    rows = []
    base = result['tables']['readouts'][0]
    for seed in (0, 1):
        for draw in range(5):
            for method, offset in [('learned_relu', 0), ('learned_linear', 3)]:
                rows.append(base | dict(seed=seed, draw=draw, method=method, test_mse=seed+draw+offset))
    collapsed, _ = seed_means(rows, 'readouts')
    assert len(collapsed) == 4
    assert next(r['test_mse'] for r in collapsed if r['seed'] == 0 and r['method'] == 'learned_relu') == 2.
    with TemporaryDirectory() as tmp:
        output = Path(tmp); shard = output/'shard'; shard.mkdir()
        csv_write(shard/'readouts.csv', rows)
        for table in ('geometry', 'strata', 'references', 'starts', 'gates', 'parameters', 'feature_errors'):
            csv_write(shard/(table+'.csv'), [])
        summarize([shard], output, 100)
        means = list(read_rows(output/'readouts_mean.csv'))
        assert all(r['num_seeds'] == 2 for r in means)
        paired = [r for r in read_rows(output/'readouts_paired.csv') if r['metric'] == 'test_mse']
        assert len(paired) == 1 and paired[0]['candidate_minus_reference'] == -3
        assert paired[0]['num_pairs'] == 2


def test_plans_have_unique_worlds_and_expected_controls():
    config = validate(defaults('overnight'))
    plan = worlds(config)
    assert len(plan) == 1672
    assert len({(w['scenario'], w['seed']) for w in plan}) == len(plan)
    assert all('fixed_angle' in s for s in scenarios(config) if s['experiment'] == 'landscape')
    assert all('fixed_angle' not in s for s in scenarios(config) if s['experiment'] != 'landscape')
    assert {w['experiment'] for w in plan} == set(config['experiments'])


def test_completed_shard_checksum_detects_changed_checkpoints():
    module = runner()
    config = defaults('smoke') | dict(steps=2)
    world = worlds(config)[0]
    with TemporaryDirectory() as tmp:
        directory = Path(tmp)/'world'
        assert not module.complete(directory)
        module.save(directory, run_world(config, world), world)
        assert module.complete(directory)
        with (directory/'checkpoints.npz').open('ab') as handle:
            handle.write(b'corrupted')
        try:
            module.complete(directory)
        except ValueError:
            pass
        else:
            raise AssertionError('corrupt checkpoint was accepted')


def test_omnibus_runner_points_to_the_dedicated_learned_geometry_launcher():
    path = Path(__file__).resolve().parents[1]/'experiments/overnight_suite.py'
    spec = importlib.util.spec_from_file_location('geometry_omnibus', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command, reason = module.command(path.parents[1], 'python', '17', path.with_name('17_learned_relu_geometry.py'), None, Path('output'), 28)
    assert not command and 'learned_relu_geometry.sbatch' in reason
