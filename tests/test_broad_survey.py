"""Definition and mechanism checks for the broad static-feature survey."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from aliasing.benchmark_results import csv_write, read_rows, summarize
from aliasing.broad_survey import build_jobs, profile_pairs
from aliasing.representation_benchmark import TABLE_GROUPS, run_world
from aliasing.survey_diagnostics import EXTRA_GROUPS, alias_modes


def load_runner():
    path = Path(__file__).resolve().parents[1]/'experiments/17_learned_feature_survey.py'
    spec = importlib.util.spec_from_file_location('broad_survey_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_modes_reconstruct_canonical_aliasing_with_population_cross_terms():
    rng = np.random.default_rng(30)
    for n, width in ((3, 7), (12, 4)):
        h, hp, residual = rng.normal(size=(n, width)), rng.normal(size=(25, width)), rng.normal(size=(n, 3))
        rows = alias_modes(h, hp, residual)
        actual = np.mean((hp @ np.linalg.pinv(h) @ residual)**2)
        assert np.isclose(sum(r['mode_diagonal_alias_energy'] for r in rows)+rows[0]['signed_mode_offdiagonal_sum'], actual)
        assert abs(rows[0]['signed_mode_offdiagonal_sum']) > 1e-5
        assert all(np.isclose(row['total_alias_energy'], actual) for row in rows)
        zero = alias_modes(h, hp, np.zeros_like(residual))
        assert all(row['total_alias_energy'] == 0. for row in zero)


def test_plan_has_expected_blocks_and_shared_baseline_controls():
    runner = load_runner()
    base = runner.legacy_runner().defaults('overnight')
    jobs = build_jobs(base)
    assert len(jobs) == 13
    assert jobs[2]['job_id'] == 'width_16'
    assert sum(j['config']['num_seeds']*len(j['config']['active_counts'])*len(j['config']['task_profiles']) for j in jobs) == 2160
    for job in jobs:
        c = job['config']
        assert c['seed'] == base['seed']
        assert c['reference_sizes'] == base['reference_sizes']
        assert c['models'][0]['family'] == 'mlp' and len(c['models'][0]['widths']) == 1
    assert jobs[5]['config']['training_tasks'] == 'first'
    assert jobs[6]['config']['training_tasks'] == 'first_half'
    assert jobs[2]['config']['task_profiles'] == ['balanced', 'A', 'B']


def test_subset_training_preserves_random_baseline_and_task_gad_averages():
    runner = load_runner()
    config = build_jobs(runner.legacy_runner().defaults('smoke'), True)[1]['config']
    config.update(steps=3, draws=1, rotations=0, threshold_offsets=[.5],
        sample_sizes=[3, 16], detail_sample_sizes=[16], reference_sizes=[48, 96],
        n_test=32, n_validation=24, n_representation=32, probe_repeats=1,
        models=[dict(name='standard', family='mlp', widths=[4])])
    full = run_world(config, 0, 1, 'balanced')
    subset = run_world(config | dict(training_tasks='first'), 0, 1, 'balanced')
    larger = run_world(config | dict(n_representation=64), 0, 1, 'balanced')
    keys = ('architecture', 'method', 'sample_size', 'fit_split', 'estimator')
    def random_rows(result):
        return {tuple(r[k] for k in keys): r for r in result['tables']['readouts'] if r['method'] == 'random_relu'}
    for key, r in random_rows(full).items():
        other = random_rows(subset)[key]
        for metric in ('test_mse', 'model_insufficiency', 'mp_aliasing', 'data_insufficiency'):
            assert r[metric] == other[metric]
            assert r[metric] == random_rows(larger)[key][metric]
    task_rows = subset['tables']['task_readouts']
    assert all(row['training_task_weight'] == (1. if row['feature'] == 0 else 0.) for row in task_rows)
    for row in subset['tables']['readouts']:
        if row['fit_split'] != 'independent' or row['sample_size'] != 16:
            continue
        tasks = [r for r in task_rows if all(r[k] == row[k] for k in keys)]
        assert len(tasks) == 4
        for metric in ('test_mse', 'model_insufficiency', 'mp_aliasing', 'data_insufficiency',
                       'cross_data_insufficiency__mp_aliasing'):
            assert np.isclose(np.mean([t[metric] for t in tasks]), row[metric], atol=1e-10)


def test_threshold_and_signed_cross_term_contrasts_are_paired_by_seed():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        shards = []
        for seed in (0, 1):
            shard = root/str(seed)
            shard.mkdir()
            shards.append(shard)
            rows = []
            for profile in ('A', 'B'):
                for method, value in (('learned_relu', -2.), ('threshold_0.5', -1.)):
                    rows.append(dict(seed=seed, active_count=1, task_profile=profile,
                        architecture='standard', method=method, sample_size=16,
                        fit_split='independent', estimator='mp', feature=0,
                        test_mse=2.+value/10,
                        cross_data_insufficiency__mp_aliasing=value+(profile == 'B')))
            csv_write(shard/'task_readouts.csv', rows)
        groups = {'task_readouts': EXTRA_GROUPS['task_readouts']}
        summarize(shards, root, 40, groups)
        paired = list(read_rows(root/'task_readouts_paired.csv'))
        signed = [r for r in paired if r['metric'] == 'cross_data_insufficiency__mp_aliasing']
        assert len(signed) == 2 and all(r['candidate_minus_reference'] == 1. for r in signed)
        # Profile-pair helper expects both readout tables, with the same schema.
        csv_write(root/'readouts_seed_means.csv', list(read_rows(root/'task_readouts_seed_means.csv')))
        profile_pairs(root, TABLE_GROUPS | EXTRA_GROUPS, 40)
        contrasts = list(read_rows(root/'task_readouts_profile_paired.csv'))
        signed = [r for r in contrasts if r['metric'] == 'cross_data_insufficiency__mp_aliasing']
        assert len(signed) == 2 and all(r['candidate_minus_reference'] == 1. for r in signed)


def test_shared_plan_is_immutable_and_job_lock_rejects_duplicate_writer():
    import fcntl
    from types import SimpleNamespace
    runner = load_runner()
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        runner.install_plan(root, {'example': 1})
        runner.install_plan(root, {'example': 1})
        try:
            runner.install_plan(root, {'example': 2})
        except ValueError:
            pass
        else:
            raise AssertionError('changed shared plan was accepted')
        directory = root/'jobs'/'example'
        directory.mkdir(parents=True)
        with (directory/'run.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                runner.execute_job(root, {'job_id': 'example'}, {}, SimpleNamespace(resume=True), None)
            except ValueError as exc:
                assert 'already running' in str(exc)
            else:
                raise AssertionError('duplicate job writer was accepted')
