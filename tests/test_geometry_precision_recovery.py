"""Cancellation regressions and conservative old-run recovery controls."""
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from aliasing.learned_geometry import run_world
from aliasing.learned_geometry_plan import defaults, worlds
from aliasing.learned_geometry_precision import gad_metrics, decimal_components
from aliasing.mechanism_checks import pinverse
from aliasing.precision_gad import gad_metrics as previous_metrics


def case(last_epsilon=1e-8):
    rng = np.random.default_rng(91)
    for epsilon in [1e-7, last_epsilon]:
        x, xp = rng.normal(size=(30, 2)), rng.normal(size=(60, 2))
        h = np.column_stack([np.ones(30), x, x[:, 0]+epsilon*rng.normal(size=30)])
        ht = np.column_stack([np.ones(60), xp, xp[:, 0]+epsilon*rng.normal(size=60)])
        beta = np.zeros((4, 2)); beta[1] = 1/epsilon; beta[3] = -1/epsilon
        y, yt = rng.normal(size=(30, 2)), rng.normal(size=(60, 2))
    return h, y, ht, yt, beta


def test_decimal_repairs_a_failure_of_the_previous_longdouble_retry():
    arrays = case()
    try:
        previous_metrics(*arrays)
    except AssertionError as error:
        assert 'failed after precision retry' in str(error)
    else:
        raise AssertionError('regression did not reproduce the old failure')
    inverse = pinverse(arrays[0])[0]
    expected_head = inverse @ arrays[1]
    result, vectors = gad_metrics(*arrays, return_vectors=True)
    assert result['gad_decimal_digits'] == 80
    assert result['test_mse'] == float(np.mean((arrays[2] @ expected_head-arrays[3])**2))
    assert result['risk_identity_scaled_error'] < 1e-12
    assert result['prediction_identity_relative_error'] < 1e-12
    assert result['gad_arithmetic_reliable'] == 1
    assert np.max(abs(sum(vectors.values())-(arrays[2] @ expected_head-arrays[3]))) < 1e-12
    assert result['estimator_correction'] == 0.


def test_larger_estimator_roundoff_is_reported_separately_not_called_aliasing():
    arrays = case(1e-12)
    result, vectors = gad_metrics(*arrays, return_vectors=True)
    assert result['gad_decimal_digits'] >= 80
    assert result['gad_arithmetic_reliable'] == 0
    assert result['arithmetic_roundoff'] > 1e-14
    # Ordinary canonical components alone cannot reproduce the implemented
    # float64 predictor here; the independent arithmetic term explains why.
    error = arrays[2] @ (pinverse(arrays[0])[0] @ arrays[1])-arrays[3]
    canonical = sum(v for name, v in vectors.items() if name != 'arithmetic_roundoff')
    assert np.max(abs(error-canonical)) > 1e-7
    assert np.allclose(error-canonical, vectors['arithmetic_roundoff'], atol=1e-12)
    assert result['risk_identity_scaled_error'] < 1e-12


def test_accepted_legacy_metrics_and_candidate_predictions_are_preserved():
    rng = np.random.default_rng(193)
    for n in (3, 20):
        h, ht = rng.normal(size=(n, 5)), rng.normal(size=(31, 5))
        y, yt, beta = rng.normal(size=(n, 2)), rng.normal(size=(31, 2)), rng.normal(size=(5, 2))
        for candidate in (None, rng.normal(size=(5, 2))):
            old = previous_metrics(h, y, ht, yt, beta, candidate=candidate)
            new = gad_metrics(h, y, ht, yt, beta, candidate=candidate)
            for key, value in old.items():
                assert new[key] == value or (np.isnan(new[key]) and np.isnan(value)), key
            assert new['arithmetic_roundoff_measured'] == 0
            assert new['gad_decimal_digits'] == 0


def test_decimal_vectors_match_known_canonical_values_in_both_rank_regimes():
    rng = np.random.default_rng(74)
    for n in (2, 12):
        h, ht = rng.normal(size=(n, 4)), rng.normal(size=(17, 4))
        y, yt, beta = rng.normal(size=(n, 2)), rng.normal(size=(17, 2)), rng.normal(size=(4, 2))
        inverse = pinverse(h)[0]; fit = inverse @ y
        candidate = fit+rng.normal(size=fit.shape)
        vectors = decimal_components(h, y, ht, yt, beta, candidate, inverse, fit)
        assert np.allclose(vectors['model_insufficiency'], ht @ beta-yt)
        assert np.allclose(vectors['data_insufficiency'], ht @ ((inverse @ h-np.eye(4)) @ beta), atol=1e-12)
        assert np.allclose(vectors['mp_aliasing'], ht @ (inverse @ (y-h @ beta)), atol=1e-12)
        assert np.allclose(vectors['estimator_correction'], ht @ (candidate-fit), atol=1e-12)
        assert np.max(abs(vectors['arithmetic_roundoff'])) < 1e-12


def test_decimal_retry_keeps_an_explicit_candidate_head_and_its_correction():
    arrays = case()
    h, y, ht, yt, beta = arrays
    inverse = pinverse(h)[0]
    fit = inverse @ y
    candidate = fit+np.array([[.1, -.2], [.2, .3], [-.4, .2], [.1, .1]])
    result, vectors = gad_metrics(*arrays, candidate=candidate, inverse=inverse, return_vectors=True)
    assert result['gad_decimal_digits'] >= 80
    assert result['test_mse'] == float(np.mean((ht @ candidate-yt)**2))
    assert np.allclose(vectors['estimator_correction'], ht @ (candidate-fit), atol=1e-12)


def recovery_module():
    path = Path(__file__).resolve().parents[1]/'experiments/repair_learned_relu_geometry.py'
    spec = importlib.util.spec_from_file_location('precision_recovery', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_old_run(module, source):
    runner = module.runner()
    _, hashes = runner.fingerprint()
    previous = dict(hashes)
    for name in module.ADDITIONS:
        previous.pop(name)
    replacements = {}
    for name in module.PREVIOUS_REPLACEMENTS:
        content = '# synthetic old evaluation source: '+name+'\n'
        previous[name] = hashlib.sha256(content.encode()).hexdigest()
        replacements[name] = content
    module.PREVIOUS_REPLACEMENTS = {name: previous[name] for name in replacements}
    source.mkdir()
    for name in previous:
        target = source/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(replacements[name] if name in replacements else (module.ROOT/name).read_text())
    config = defaults('smoke') | dict(steps=2, experiments=['phase'])
    record = dict(config=config, source_sha256=previous,
        code_sha256=hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest(), gad_definition_version=2)
    (source/'configuration.json').write_text(json.dumps(record))
    world = worlds(config)[0]
    name = f'{world["experiment"]}_{world["scenario"]}_seed{world["seed"]}'
    runner.save(source/'shards'/name, run_world(config, world), world)
    return name


def test_recovery_preserves_completed_shards_and_retries_only_missing_worlds():
    module = recovery_module()
    with TemporaryDirectory() as tmp:
        source, target = Path(tmp)/'old', Path(tmp)/'recovered'
        name = make_old_run(module, source)
        before = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in source.rglob('*') if path.is_file()}
        record = module.prepare(source, target)
        assert record['inherited_count'] == 1 and record['pending_count'] == 1
        assert module.runner().complete(target/'shards'/name)
        assert (source/'shards'/name/'checkpoints.npz').read_bytes() == (target/'shards'/name/'checkpoints.npz').read_bytes()
        assert before == {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source.rglob('*') if path.is_file()}
        assert module.prepare(source, target) == record
        assert json.loads((source/'configuration.json').read_text())['config'] == json.loads((target/'configuration.json').read_text())['config']


def test_recovery_rejects_unapproved_training_source_changes_and_bad_snapshots():
    module = recovery_module()
    with TemporaryDirectory() as tmp:
        source, target = Path(tmp)/'old', Path(tmp)/'recovered'
        make_old_run(module, source)
        snapshot = source/'source/src/aliasing/learned_geometry_training.py'
        snapshot.write_text(snapshot.read_text()+'# changed\n')
        try:
            module.prepare(source, target)
        except ValueError as error:
            assert 'snapshot' in str(error)
        else:
            raise AssertionError('changed original training snapshot accepted')
        assert not target.exists()
        # Even an internally consistent altered source manifest may not
        # migrate a different optimizer into a recovery of the old run.
        record_path = source/'configuration.json'
        record = json.loads(record_path.read_text())
        record['source_sha256']['src/aliasing/learned_geometry_training.py'] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        record['code_sha256'] = hashlib.sha256(json.dumps(record['source_sha256'], sort_keys=True).encode()).hexdigest()
        record_path.write_text(json.dumps(record))
        try:
            module.prepare(source, target)
        except ValueError as error:
            assert 'unapproved source change' in str(error)
        else:
            raise AssertionError('different training algorithm accepted by migration')
