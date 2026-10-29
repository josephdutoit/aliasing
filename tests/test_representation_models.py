"""Architecture, gradient, GAD and aggregation controls for experiment 16."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from aliasing.benchmark_results import csv_write, read_rows, summarize
from aliasing.controlled_superposition import features, initialize, loss_grad, problem, sample, train
from aliasing.mechanism_checks import gad_metrics, pinverse
from aliasing.representation_benchmark import TABLE_GROUPS, output_floor, run_world
from aliasing.representation_models import default_models, make_model, register_model, train_model
from aliasing.precision_gad import gad_metrics as precision_gad_metrics


def runner():
    path = Path(__file__).resolve().parents[1]/'experiments/16_representation_model_benchmark.py'
    spec = importlib.util.spec_from_file_location('representation_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bottleneck_adapter_matches_legacy_features_gradients_and_training():
    spec = problem(14, 4, 6, 1)
    x, y, _ = sample(spec, 100, 19)
    weights = np.array([1., .5, .25, .7])
    mapping = dict(w0='enc', w1='dec', b1='bias', head='head', out_bias='out_bias')
    for activation in ('linear', 'relu'):
        model = make_model(dict(family='bottleneck', compression=2, width=6), activation)
        initial = model.initialize(x, 4, 17)
        old = initialize(x, 2, 6, 4, 17, activation == 'relu')
        for key, old_key in mapping.items():
            assert np.array_equal(initial[key], old[old_key])
        assert np.array_equal(model.features(x, initial), features(x, old, activation == 'relu'))
        value, grad = model.loss_grad(initial, x, y, weights, .013)
        expected, expected_grad = loss_grad(old, x, y, weights, activation == 'relu', .013)
        assert np.isclose(value, expected, atol=1e-14)
        for key, old_key in mapping.items():
            assert np.allclose(grad[key], expected_grad[old_key], atol=1e-14)
        final, _ = train_model(model, initial, x, y, weights, 100, 64, .005, 1e-5, 23)
        legacy, _ = train(old, x, y, weights, activation == 'relu', 100, 64, .005, 1e-5, 23)
        for key, old_key in mapping.items():
            assert np.allclose(final[key], legacy[old_key], atol=1e-12), key


def test_all_registered_dense_architectures_have_correct_gradients():
    rng = np.random.default_rng(13)
    x, y = rng.normal(size=(9, 4)), rng.normal(size=(9, 3))
    weights = np.array([1., .4, .7])
    specs = default_models(2, 5)+[dict(family='mlp', widths=[3, 5])]
    for architecture in specs:
        for activation in ('linear', 'relu'):
            model = make_model(architecture, activation)
            params = model.initialize(x, 3, 33)
            for key in params:
                if key.startswith('b'):
                    params[key] += .12345
            _, gradients = model.loss_grad(params, x, y, weights, .013)
            for name, values in params.items():
                for index in np.ndindex(values.shape):
                    before, epsilon = values[index], 1e-6
                    values[index] = before+epsilon
                    plus = model.loss_grad(params, x, y, weights, .013)[0]
                    values[index] = before-epsilon
                    minus = model.loss_grad(params, x, y, weights, .013)[0]
                    values[index] = before
                    assert np.isclose((plus-minus)/(2*epsilon), gradients[name][index], atol=1e-7, rtol=1e-5), (architecture, activation, name, index)


def test_rotation_preserves_pregeometry_and_linearized_predictions_for_deeper_models():
    x = np.random.default_rng(43).normal(size=(100, 6))
    specs = default_models(2, 5)+[dict(family='mlp', widths=[3, 5])]
    for architecture in specs:
        model = make_model(architecture)
        params = model.initialize(x, 4, 3)
        q = np.linalg.qr(np.random.default_rng(6).normal(size=(model.width, model.width)))[0]
        rotated = model.rotate(params, q)
        a, b = model.preactivation(x, params), model.preactivation(x, rotated)
        assert np.allclose(a @ a.T, b @ b.T, atol=1e-10)
        assert np.allclose(np.linalg.svd(model.preactivation_weight(params), compute_uv=False),
                           np.linalg.svd(model.preactivation_weight(rotated), compute_uv=False))
        linear = model.linearized()
        assert np.allclose(linear.features(x, params) @ linear.native_head(params),
                           linear.features(x, rotated) @ linear.native_head(rotated), atol=1e-12)


def test_hidden_width_bound_and_full_rank_linear_control():
    spec = problem(43, 6, 8, 2)
    x, y, _ = sample(spec, 1000, 14)
    wide = make_model(dict(family='mlp', widths=[6]), 'linear')
    params = wide.initialize(x, 6, 19)
    h = wide.features(x, params)
    assert np.linalg.matrix_rank(h) == 7
    assert np.mean((h @ np.linalg.lstsq(h, y, rcond=None)[0]-y)**2) < 1e-20
    assert output_floor(spec, 2) > .3
    assert output_floor(spec, 6) == 0.
    narrow = make_model(dict(family='mlp', widths=[2]))
    hn = narrow.features(x, narrow.initialize(x, 6, 19))
    assert np.linalg.matrix_rank(hn) <= 3


def test_exact_gad_works_for_every_model_in_both_rank_regimes():
    spec = problem(12, 4, 6, 2)
    x, y, _ = sample(spec, 160, 30)
    for architecture in default_models(2, 6):
        model = make_model(architecture)
        h = model.features(x, model.initialize(x[:30], 4, 14))
        ref = np.linalg.lstsq(h[50:100], y[50:100], rcond=None)[0]
        for n in (2, 30):
            inverse = pinverse(h[:n])[0]
            result = gad_metrics(h[:n], y[:n], h[100:], y[100:], ref, inverse=inverse)
            assert result['risk_identity_relative_error'] < 1e-10
            assert result['prediction_identity_relative_error'] < 1e-10
            assert result['estimator_correction'] == 0.


def test_seed_summaries_and_pairing_average_repeats_within_seed():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        shards = []
        for seed in (0, 1):
            shard = root/str(seed)
            shard.mkdir()
            shards.append(shard)
            for table in TABLE_GROUPS:
                rows = []
                for rotation in (1, 2):
                    for draw in (0, 1):
                        for probe in (0, 1, 2):
                            meta = dict(seed=seed, active_count=1, task_profile='balanced', architecture='mlp_wide',
                                sample_size=16, fit_split='independent', estimator='mp', reference_size=512,
                                probe_kind='shared_gaussian', feature=0, step=50,
                                rotation_id=rotation, readout_draw=draw, probe_id=probe)
                            for method, offset in (('learned_relu', 0), ('random_relu', 2)):
                                rows.append(meta | dict(method=method, test_mse=seed+rotation+offset, rms_gain=seed+rotation+offset))
                csv_write(shard/(table+'.csv'), rows)
        summarize(shards, root, 100)
        means = [row for row in read_rows(root/'probes_mean.csv') if row['method'] == 'learned_relu']
        assert len(means) == 1 and means[0]['num_seeds'] == 2
        assert means[0]['rms_gain'] == 2.
        paired = [row for row in read_rows(root/'readouts_paired.csv') if row['reference_method'] == 'random_relu' and row['metric'] == 'test_mse']
        assert len(paired) == 1 and paired[0]['num_pairs'] == 2
        assert paired[0]['candidate_minus_reference'] == -2.


def test_plugin_registration_uses_the_same_evaluator():
    name = 'test_custom_dense'
    from aliasing.representation_models import FACTORIES
    if name not in FACTORIES:
        register_model(name, lambda spec, activation: make_model(dict(family='mlp', widths=spec['sizes']), activation))
    model = make_model(dict(family=name, sizes=[3]))
    assert model.width == 3


def test_omnibus_suite_directs_model_runs_to_the_dedicated_batch_script():
    path = Path(__file__).resolve().parents[1]/'experiments/overnight_suite.py'
    spec = importlib.util.spec_from_file_location('model_suite', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    script = path.with_name('16_representation_model_benchmark.py')
    command, reason = module.command(path.parents[1], 'python', '16', script, None, Path('output'), 28)
    assert not command and 'representation_models.sbatch' in reason


def test_precision_retry_preserves_existing_successful_gad_results_exactly():
    rng = np.random.default_rng(918)
    for n in (3, 30):
        h, ht = rng.normal(size=(n, 7)), rng.normal(size=(60, 7))
        y, yt, reference = rng.normal(size=(n, 4)), rng.normal(size=(60, 4)), rng.normal(size=(7, 4))
        for candidate in (None, rng.normal(size=(7, 4))):
            original = gad_metrics(h, y, ht, yt, reference, candidate)
            checked = precision_gad_metrics(h, y, ht, yt, reference, candidate)
            assert checked['gad_extended_precision'] == 0
            for key, value in original.items():
                assert checked[key] == value or (np.isnan(value) and np.isnan(checked[key])), key


def test_precision_retry_recovers_cancelling_gad_terms_without_changing_predictions():
    rng = np.random.default_rng(1)
    u = np.linalg.qr(rng.normal(size=(17, 17)))[0]
    v = np.linalg.qr(rng.normal(size=(17, 17)))[0]
    h = (u*np.geomspace(1., 1e-5, 17)) @ v.T
    ht = rng.normal(size=(64, 17))
    reference = rng.normal(size=(17, 3))*1e4
    y, yt = h @ reference, ht @ reference
    inverse = pinverse(h)[0]
    try:
        gad_metrics(h, y, ht, yt, reference, inverse=inverse)
    except AssertionError:
        pass
    else:
        raise AssertionError('the regression case must reproduce float64 cancellation')
    result = precision_gad_metrics(h, y, ht, yt, reference, inverse=inverse)
    assert result['gad_extended_precision'] == 1
    assert result['test_mse'] == float(np.mean((ht @ (inverse @ y)-yt)**2))
    assert result['prediction_identity_relative_error'] <= 1e-7
    assert result['risk_identity_absolute_error'] <= 1e-7
    assert result['estimator_correction'] == 0.


def test_small_world_produces_all_scientific_tables_and_controls():
    module = runner()
    config = module.defaults('smoke')
    config.update(steps=2, rotations=1, draws=1, probe_repeats=1, sample_sizes=[3, 8],
                  detail_sample_sizes=[8], reference_sizes=[48, 96], n_test=32, n_validation=24,
                  n_representation=32, models=default_models(2, 4))
    result = run_world(config, 0, 1, 'balanced')
    assert all(result['tables'][table] for table in TABLE_GROUPS)
    architectures = {row['architecture'] for row in result['tables']['readouts']}
    assert {'bottleneck', 'mlp_wide', 'mlp_narrow', 'oracle_rank2', 'oracle_rank4'} <= architectures
    assert result['controls']['rotation_gram_error'] < 1e-12
    assert result['controls']['linear_rotation_prediction_error'] < 1e-12
    assert result['controls']['risk_identity_scaled_error'] < 1e-9
    for table in ('readouts', 'references'):
        assert all('mp_aliasing' in row and 'estimator_correction' in row for row in result['tables'][table])
    with TemporaryDirectory() as tmp:
        shard = Path(tmp)/'shard'
        module.save_world(shard, result, 0, 1, 'balanced')
        assert module.shard_complete(shard)
        with (shard/'geometry.csv').open('a') as handle:
            handle.write('changed\n')
        try:
            module.shard_complete(shard)
        except ValueError:
            pass
        else:
            raise AssertionError('corrupt completed shard was accepted')
