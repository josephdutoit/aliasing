"""Resolve reference/readout, encoder/decoder and residual-amplification gaps.

Reuses experiment-14 checkpoints. See GAD_MECHANISM_CHECKS.md for definitions,
budgets, independent splits, controls and interpretation criteria.
"""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from aliasing.controlled_superposition import (
    features, geometry, importance, initialize, preactivation, problem, rotate,
    rotation_checks, sample,
)
from aliasing.mechanism_checks import (
    extend_sample, gad_metrics, normalized_probe_gain, pinverse, select_ridge,
    sensitivity, train_decoder,
)


TABLES = ('readouts', 'references', 'probes', 'geometry', 'history')
GROUPS = {
    'readouts': ('active_count', 'task_profile', 'method', 'sample_size', 'fit_split', 'estimator'),
    'references': ('active_count', 'task_profile', 'method', 'sample_size', 'reference_size'),
    'probes': ('active_count', 'task_profile', 'method', 'sample_size', 'probe_kind'),
    'geometry': ('active_count', 'task_profile', 'method'),
    'history': ('active_count', 'task_profile', 'method', 'step'),
}


def json_write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def csv_write(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def read_rows(path):
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            for key, value in row.items():
                try:
                    row[key] = float(value) if value else float('nan')
                except ValueError:
                    pass
            yield row


def state_params(state, name):
    prefix = name+'_'
    result = {key[len(prefix):]: value.copy() for key, value in state.items() if key.startswith(prefix)}
    if not {'enc', 'dec', 'bias'} <= result.keys():
        raise ValueError('checkpoint missing '+name)
    return result


def run_world(source_dir, original, config, seed_index, k, profile, checkpoint_hash):
    started = time.monotonic()
    directory = Path(source_dir)/'shards'/f'seed{seed_index}_k{k}_{profile}'
    checkpoint = directory/'checkpoints.npz'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
        raise ValueError('source checkpoint changed after submission')
    with np.load(checkpoint, allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    p, m, h = original['features'], original['compression'], original['expansion']
    seed = original['seed']+seed_index*104729
    data_seed = seed+10000019*k
    spec = problem(seed, p, original['input_dim'], k, original['task_basis'])
    for key, expected in (('embedding', spec['embedding']), ('teacher', spec['teacher']),
                          ('target_scales', spec['scales']), ('latent_mean', spec['mean'])):
        if key not in state or not np.allclose(state[key], expected, atol=1e-10, rtol=1e-10):
            raise ValueError('checkpoint/data generator mismatch: '+key)
    xr, yr, _ = sample(spec, original['n_representation'], data_seed+1)
    xp, yp, lp = extend_sample(spec, max(original['sample_sizes']), max(config['sample_sizes']), data_seed+2, data_seed+101)
    xf, yf, lf = extend_sample(spec, original['n_reference'], max(config['reference_sizes']), data_seed+3, data_seed+102)
    xt, yt, _ = extend_sample(spec, original['n_test'], config['n_test'], data_seed+4, data_seed+103)
    xv, yv, _ = sample(spec, config['n_validation'], data_seed+104)
    weights = importance(profile, p, original['secondary_weight'])
    if not np.array_equal(state['importance'], weights):
        raise ValueError('checkpoint task priority mismatch')
    meta = dict(seed=seed_index, active_count=k, task_profile=profile)
    controls = dict(frozen_encoder_error=0., shared_decoder_initialization_error=0.,
                    zero_perturbation_error=0., perturbation_superposition_relative_error=0.,
                    prediction_identity_relative_error=0., risk_identity_relative_error=0.,
                    risk_identity_scaled_error=0., risk_identity_absolute_error=0., identity_cancellation_ratio=0.,
                    rotation_gram_error=0., rotation_spectrum_error=0., linear_rotation_prediction_error=0.)
    tables = {name: [] for name in TABLES}
    variants = []
    for name in ('random_linear', 'learned_linear', 'random_relu', 'learned_relu', 'linear_oracle'):
        variants.append((name, 0, state_params(state, name), name.endswith('relu')))
    learned = state_params(state, 'learned_relu')
    variants.append(('no_gate', 0, learned, False))
    desired_rate = float(np.mean(preactivation(xr, learned) > 0))
    for rid in range(1, config['rotations']+1):
        q = np.linalg.qr(np.random.default_rng(seed+rid*1009).normal(size=(h, h)))[0]
        rotated = rotate(learned, q)
        for key, value in rotation_checks(xt, learned, rotated).items():
            controls[key] = max(controls[key], value)
        variants.append(('rotated_relu', rid, rotated, True))
        matched = {key: value.copy() for key, value in rotated.items()}
        matched['bias'] -= np.quantile(preactivation(xr, rotated), 1-desired_rate)
        if abs(np.mean(preactivation(xr, matched) > 0)-desired_rate) > 2/(len(xr)*h):
            raise AssertionError('activity matching failed')
        variants.append(('rotated_relu_rate_matched', rid, matched, True))
        # Strong control on all original readout examples, evaluated linearly.
        ha, hb = features(xp, learned, False), features(xp, rotated, False)
        pa = features(xt, learned, False) @ (pinverse(ha)[0] @ yp)
        pb = features(xt, rotated, False) @ (pinverse(hb)[0] @ yp)
        error = float(np.max(np.abs(pa-pb))/max(1., np.max(np.abs(pa))))
        controls['linear_rotation_prediction_error'] = max(controls['linear_rotation_prediction_error'], error)
        if error > 1e-8:
            raise AssertionError('linear rotation control failed')
    # Matched fresh optimization: joint_fresh and fixed_random_encoder start
    # from exactly the same parameters. The pretrained-encoder arm has an
    # explicitly larger total training budget and is a secondary comparison.
    initial = initialize(xr, m, h, p, seed+7001, True)
    initial['enc'] = state_params(state, 'random_relu')['enc']
    initial['bias'] = -np.quantile(xr @ initial['enc'] @ initial['dec'], .7, axis=0)
    trained_states = {}
    for name, frozen, pretrained in (('joint_fresh', False, False),
                                      ('fixed_random_encoder', True, False),
                                      ('frozen_learned_encoder', True, True)):
        start = {key: value.copy() for key, value in initial.items()}
        if pretrained:
            start['enc'] = learned['enc'].copy()
            start['bias'] = -np.quantile(xr @ start['enc'] @ start['dec'], .7, axis=0)
        for key in ('dec', 'head', 'out_bias'):
            controls['shared_decoder_initialization_error'] = max(controls['shared_decoder_initialization_error'],
                float(np.max(np.abs(start[key]-initial[key]))))
        params, history = train_decoder(start, xr, yr, weights, config['decoder_steps'], original['batch_size'],
            original['learning_rate'], original['weight_decay'], seed+7003, frozen)
        if frozen:
            controls['frozen_encoder_error'] = max(controls['frozen_encoder_error'],
                                                  float(np.max(np.abs(params['enc']-start['enc']))))
        tables['history'].extend(meta | dict(method=name, **row) for row in history)
        variants.append((name, 0, params, True))
        trained_states.update({name+'_'+key: value for key, value in params.items()})
    # Shared nonlinear function bank, with coefficients fixed across maps.
    rng = np.random.default_rng(seed+8001)
    quadratic_directions = rng.normal(size=(p, config['probe_repeats']))/np.sqrt(p)
    psi_p, psi_f = lp**2 @ quadratic_directions, lf**2 @ quadratic_directions
    psi_center, psi_scale = psi_f.mean(axis=0), psi_f.std(axis=0)
    psi_scale = np.maximum(psi_scale, 1e-12)
    psi_p, psi_f = (psi_p-psi_center)/psi_scale, (psi_f-psi_center)/psi_scale
    orders = [np.arange(len(xp))] + [np.random.default_rng(data_seed+9001+draw).permutation(len(xp))
                                   for draw in range(1, config['draws'])]
    for name, rid, params, relu in variants:
        tag = meta | dict(method=name, rotation_id=rid)
        hp, hf, ht, hv, hr = [features(x, params, relu) for x in (xp, xf, xt, xv, xr)]
        references = {n: np.linalg.lstsq(hf[:n], yf[:n], rcond=None)[0] for n in config['reference_sizes']}
        reference = references[max(references)]
        geom = geometry(spec, params, xt, relu)
        _, reference_rank, reference_condition = pinverse(hf)
        tables['geometry'].append(tag | geom | dict(reference_rank=reference_rank,
            reference_condition=reference_condition, reference_test_mse=float(np.mean((ht @ reference-yt)**2)),
            reference_fit_mse=float(np.mean((hf @ reference-yf)**2)),
            decoder_steps=config['decoder_steps'] if name in trained_states_names() else 0,
            encoder_pretraining_steps=original['steps'] if name == 'frozen_learned_encoder' else 0))
        residual_reference = np.linalg.lstsq(hf, psi_f, rcond=None)[0]
        structured_residual = psi_p-hp @ residual_reference

        def record(htrain, ytrain, n, fit_split, draw, estimator, candidate=None, penalty=float('nan'), validation_mse=float('nan')):
            inverse, rank, condition = pinverse(htrain)
            metrics = gad_metrics(htrain, ytrain, ht, yt, reference, candidate, inverse, weights)
            evaluated_head = inverse @ ytrain if candidate is None else candidate
            latent_error = ((ht @ evaluated_head-yt)*spec['scales']) @ spec['teacher'].T
            feature_errors = np.mean(latent_error**2,axis=0)/np.diag(spec['covariance'])
            metrics.update(latent_mean_normalized_mse=float(feature_errors.mean()),
                           latent_worst_normalized_mse=float(feature_errors.max()),
                           latent_features_below_0p1=int(np.sum(feature_errors<.1)))
            for key in ('prediction_identity_relative_error', 'risk_identity_relative_error',
                        'risk_identity_scaled_error','risk_identity_absolute_error','identity_cancellation_ratio'):
                controls[key] = max(controls[key], metrics[key])
            tables['readouts'].append(tag | dict(sample_size=n, fit_split=fit_split, readout_draw=draw,
                estimator=estimator, penalty=penalty, validation_mse=validation_mse,
                training_rank=rank, training_condition=condition, **metrics))
            return inverse

        # Native trained heads and MP heads on the SAME representation-training
        # examples distinguish head optimization from data-size differences.
        if name in ('learned_relu', 'learned_linear', *trained_states_names()):
            native = np.vstack((params['out_bias'], params['head']))
            record(hr, yr, len(hr), 'representation_training', 0, 'native', native)
            record(hr, yr, len(hr), 'representation_training', 0, 'mp')
        for draw, order in enumerate(orders):
            for n in config['sample_sizes']:
                idx = order[:n]
                htrain, ytrain = hp[idx], yp[idx]
                inverse = record(htrain, ytrain, n, 'independent', draw, 'mp')
                ridge, penalty, validation_mse = select_ridge(htrain, ytrain, hv, yv, config['penalties'], weights)
                record(htrain, ytrain, n, 'independent', draw, 'ridge_validation', ridge, penalty, validation_mse)
                if draw == 0 and n in config['reference_readout_sizes']:
                    largest_prediction = ht @ reference
                    largest_alias = ht @ (inverse @ (ytrain-htrain @ reference))
                    for reference_size, head in references.items():
                        metrics = gad_metrics(htrain, ytrain, ht, yt, head, inverse=inverse, weights=weights)
                        tables['references'].append(tag | dict(sample_size=n, reference_size=reference_size,
                            reference_prediction_difference_mse=float(np.mean((ht @ head-largest_prediction)**2)),
                            alias_vector_difference_mse=float(np.mean((ht @ (inverse @ (ytrain-htrain @ head))-largest_alias)**2)),
                            **metrics))
                if name not in ('learned_relu', 'learned_linear', 'random_relu', 'rotated_relu',
                                 'rotated_relu_rate_matched', 'no_gate', 'joint_fresh', 'fixed_random_encoder'):
                    continue
                exact = sensitivity(htrain, ht, inverse)
                tables['probes'].append(tag | dict(sample_size=n, readout_draw=draw, probe_kind='exact_readout_operator',
                    probe_id=0, rms_gain=float('nan'), valid=1, **exact))
                noise = np.random.default_rng(data_seed+10007+draw*1009+n).normal(size=(n, config['probe_repeats']))
                for kind, perturbations in (('shared_gaussian', noise), ('shared_quadratic', psi_p[idx]),
                                             ('reference_residualized_quadratic', structured_residual[idx])):
                    for repeat in range(config['probe_repeats']):
                        delta = perturbations[:, repeat:repeat+1]
                        tables['probes'].append(tag | dict(sample_size=n, readout_draw=draw, probe_kind=kind,
                            probe_id=repeat, **normalized_probe_gain(inverse, ht, delta)))
                if draw == 0 and n == max(config['sample_sizes']):
                    fitted = inverse @ ytrain
                    zero = ht @ (inverse @ (ytrain+np.zeros_like(ytrain)))-ht @ fitted
                    controls['zero_perturbation_error'] = max(controls['zero_perturbation_error'], float(np.max(np.abs(zero))))
                    delta = noise[:, :1]
                    observed = ht @ (inverse @ (ytrain+delta))-ht @ fitted
                    expected = ht @ (inverse @ delta)
                    error = float(np.max(np.abs(observed-expected))/max(1., np.max(np.abs(expected))))
                    controls['perturbation_superposition_relative_error'] = max(controls['perturbation_superposition_relative_error'], error)
                    if error > 1e-7:
                        raise AssertionError('controlled perturbation response failed')
    return dict(tables=tables, states=trained_states, controls=controls, seconds=time.monotonic()-started)


def trained_states_names():
    return ('joint_fresh', 'fixed_random_encoder', 'frozen_learned_encoder')


def summarize(completed, output, bootstrap):
    """Average all repeated draws, rotations and probe directions WITHIN seed."""
    rng = np.random.default_rng(15027)
    for table in TABLES:
        if table == 'history':
            csv_write(output/'training_history.csv', (row for d in completed for row in read_rows(d/'history.csv')))
            continue
        groups = GROUPS[table]
        collapsed = {}
        for directory in completed:
            for row in read_rows(directory/(table+'.csv')):
                key = tuple(row[name] for name in groups)+(row['seed'],)
                sums, counts = collapsed.setdefault(key, ({}, {}))
                for name, value in row.items():
                    if name in (*groups, 'seed', 'rotation_id', 'readout_draw', 'probe_id') or not isinstance(value, (int, float)):
                        continue
                    if np.isfinite(value):
                        sums[name] = sums.get(name, 0.)+value
                        counts[name] = counts.get(name, 0)+1
        seed_rows = [dict(zip((*groups, 'seed'), key)) | {name: value/counts[name] for name, value in sums.items()}
                     for key, (sums, counts) in collapsed.items()]
        csv_write(output/(table+'_seed_means.csv'), seed_rows)
        group_rows = {}
        for row in seed_rows:
            group_rows.setdefault(tuple(row[name] for name in groups), []).append(row)
        means, cis = [], []
        for key, rows in sorted(group_rows.items()):
            mean = dict(zip(groups, key)) | dict(num_seeds=len(rows))
            ci = mean.copy()
            metrics = sorted(set().union(*(row.keys() for row in rows))-set((*groups, 'seed')))
            for metric in metrics:
                values = np.array([row.get(metric, np.nan) for row in rows]); values = values[np.isfinite(values)]
                mean[metric] = float(values.mean()) if len(values) else float('nan')
                boots = rng.choice(values, (bootstrap, len(values))).mean(axis=1) if len(values)>1 else np.array([np.nan])
                ci[metric+'_ci95_low'], ci[metric+'_ci95_high'] = map(float, np.quantile(boots, [.025, .975]))
                ci[metric+'_num_seeds'] = len(values)
            means.append(mean); cis.append(ci)
        csv_write(output/(table+'_mean.csv'), means); csv_write(output/(table+'_ci95.csv'), cis)
        paired = []
        lookup = {tuple(row[name] for name in groups)+(row['seed'],): row for row in seed_rows}
        metrics = ('test_mse', 'model_insufficiency', 'mp_aliasing', 'mp_residual_gain') if table != 'probes' else (
            'rms_gain', 'isotropic_prediction_energy', 'worst_training_rms_gain')
        for key, rows in sorted(group_rows.items()):
            tag = dict(zip(groups, key))
            reference = tag.copy()
            reference['method'] = 'fixed_random_encoder' if tag['method'] in ('joint_fresh', 'frozen_learned_encoder') else 'learned_relu'
            if table == 'readouts' and tag['estimator'] == 'ridge_validation':
                reference['method'], reference['estimator'] = tag['method'], 'mp'
            elif tag['method'] == reference['method']:
                continue
            for metric in metrics:
                values = []
                for row in rows:
                    other = lookup.get(tuple(reference[name] for name in groups)+(row['seed'],))
                    a, b = row.get(metric, np.nan), other.get(metric, np.nan) if other else np.nan
                    if np.isfinite(a) and np.isfinite(b):
                        values.append(a-b)
                if not values:
                    continue
                boots = rng.choice(values, (bootstrap, len(values))).mean(axis=1) if len(values)>1 else np.array([np.nan])
                paired.append(tag | dict(reference_method=reference['method'], reference_estimator=reference.get('estimator', ''),
                    metric=metric, num_pairs=len(values), candidate_minus_reference=float(np.mean(values)),
                    ci95_low=float(np.quantile(boots, .025)), ci95_high=float(np.quantile(boots, .975))))
        csv_write(output/(table+'_paired.csv'), paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--profile', choices=('smoke', 'overnight'), default='smoke')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    for name in ('sample-sizes', 'reference-sizes', 'active-counts', 'task-profiles', 'penalties'):
        parser.add_argument('--'+name)
    for name in ('num-seeds', 'decoder-steps', 'n-test', 'n-validation', 'draws', 'rotations', 'probe-repeats', 'bootstrap'):
        parser.add_argument('--'+name, type=int)
    args = parser.parse_args()
    source = args.source_dir.resolve(); output = args.output_dir.resolve()
    source_configuration = json.loads((source/'configuration.json').read_text())
    if source_configuration.get('gad_definition_version') != 2:
        parser.error('source must use GAD definition version 2')
    original = source_configuration['config']; smoke = args.profile == 'smoke'
    source_code_hash = hashlib.sha256(Path(__file__).with_name('14_learned_aliasing_hypotheses.py').read_bytes()+
        Path(sys.modules['aliasing.controlled_superposition'].__file__).read_bytes()).hexdigest()
    if source_code_hash != source_configuration.get('code_sha256'):
        parser.error('source generator/training code differs from the code recorded by experiment 14')
    h = original['expansion']
    config = dict(active_counts=original['active_counts'][:2] if smoke else original['active_counts'],
        task_profiles=['balanced'], num_seeds=min(2, original['num_seeds']) if smoke else original['num_seeds'],
        sample_sizes=sorted(set([4, h, h+1, 2*h+2, 32])) if smoke else sorted(set([8,h,h+1,32,64,128,256,512,1024,2048,4096])),
        reference_sizes=[max(128, h+2), max(512, h+3)] if smoke else [4096,16384,65536],
        decoder_steps=120 if smoke else original['steps'], n_test=256 if smoke else max(8192, original['n_test']),
        n_validation=128 if smoke else 2048, draws=2 if smoke else 3, rotations=1 if smoke else 3,
        probe_repeats=2 if smoke else 8, bootstrap=200 if smoke else 1000,
        penalties=[1e-6,1e-4,.01,.1,1.,10.])
    for key in config:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value.split(',') if key == 'task_profiles' else (
                [float(item) if key == 'penalties' else int(item) for item in value.split(',')]
                if isinstance(config[key], list) else value)
    for key in ('sample_sizes', 'reference_sizes', 'active_counts', 'task_profiles', 'penalties'):
        if not config[key] or len(config[key]) != len(set(config[key])):
            parser.error('empty or duplicate '+key)
    if any(k not in original['active_counts'] for k in config['active_counts']) or any(p not in original['task_profiles'] for p in config['task_profiles']):
        parser.error('requested worlds must exist in source configuration')
    if not 2 <= config['num_seeds'] <= original['num_seeds'] or min(config['decoder_steps'],config['draws'],config['rotations'],config['probe_repeats'],config['n_validation'],config['n_test']) < 1 or config['bootstrap'] < 20:
        parser.error('invalid counts; need >=2 available seeds and >=20 bootstrap replicates')
    if min(config['sample_sizes']) < 1 or min(config['reference_sizes']) <= h+1 or any(not np.isfinite(x) or x <= 0 for x in config['penalties']):
        parser.error('invalid sample/reference sizes or penalties')
    config['sample_sizes'].sort(); config['reference_sizes'].sort()
    config['reference_readout_sizes'] = [n for n in config['sample_sizes'] if n in (128,512,2048,max(config['sample_sizes']))]
    if not config['reference_readout_sizes']:
        config['reference_readout_sizes'] = [max(config['sample_sizes'])]
    if output == source or source in output.parents or output in source.parents:
        parser.error('output must be separate from source')
    if args.summarize_only and not args.resume:
        parser.error('--summarize-only requires --resume')
    code_files = (Path(__file__), Path(sys.modules['aliasing.mechanism_checks'].__file__),
                  Path(sys.modules['aliasing.controlled_superposition'].__file__),
                  Path(__file__).with_name('plot_gad_mechanism_checks.py'))
    digest = hashlib.sha256(b''.join(path.read_bytes() for path in code_files)).hexdigest()
    worlds = list(itertools.product(range(config['num_seeds']), config['active_counts'], config['task_profiles']))
    checkpoint_hashes = {}
    for seed, k, profile in worlds:
        directory = source/'shards'/f'seed{seed}_k{k}_{profile}'
        if not (directory/'complete.json').exists():
            parser.error('source shard is incomplete: '+str(directory))
        checkpoint_hashes[directory.name] = hashlib.sha256((directory/'checkpoints.npz').read_bytes()).hexdigest()
    expected = dict(config=config, source_dir=str(source), source_configuration=source_configuration,
        source_checkpoint_sha256=checkpoint_hashes, code_sha256=digest, gad_definition_version=2,
        estimator_extension='canonical MP baseline plus explicit head correction and all cross terms')
    if output.exists() and not args.resume:
        parser.error('output exists; use a new directory or --resume')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'configuration.json').exists() and json.loads((output/'configuration.json').read_text()) != expected:
        parser.error('resume requires identical configuration, code and source checkpoints')
    json_write(output/'configuration.json', expected)
    completed, pending, failures = [], [], []
    for seed, k, profile in worlds:
        directory = output/'shards'/f'seed{seed}_k{k}_{profile}'
        if (directory/'complete.json').exists():
            completed.append(directory)
        else:
            pending.append((seed,k,profile,directory))
    if args.summarize_only:
        pending = []
    workers = max(1,min(args.workers,int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))))
    print(f'{len(pending)} pending, {len(completed)} complete; {workers} workers', flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_world, str(source), original, config, seed,k,profile,checkpoint_hashes[d.name]):
                   (seed,k,profile,d) for seed,k,profile,d in pending}
        for future in as_completed(futures):
            seed,k,profile,directory = futures.pop(future)
            try:
                result = future.result(); directory.mkdir(parents=True, exist_ok=True)
                for table in TABLES:
                    csv_write(directory/(table+'.csv'), result['tables'][table])
                np.savez_compressed(directory/'decoder_checkpoints.npz', **result['states'])
                json_write(directory/'complete.json', dict(seed=seed,active_count=k,task_profile=profile,
                    seconds=result['seconds'],controls=result['controls']))
                completed.append(directory)
                print(f'Complete seed={seed}, k={k}, profile={profile}: {result["seconds"]:.1f}s', flush=True)
            except Exception as exc:
                failures.append(dict(seed=seed,active_count=k,task_profile=profile,error=repr(exc)))
                print(f'FAILED seed={seed}, k={k}: {exc!r}', flush=True)
            json_write(output/'progress.json', dict(completed=len(completed),expected=len(worlds),failures=failures))
    json_write(output/'progress.json',dict(completed=len(completed),expected=len(worlds),failures=failures))
    maxima = {}
    for directory in completed:
        for key,value in json.loads((directory/'complete.json').read_text())['controls'].items():
            maxima[key] = max(maxima.get(key,0.),value)
    revision = subprocess.run(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1], capture_output=True,text=True)
    json_write(output/'run_manifest.json', dict(completed=len(completed),expected=len(worlds),failures=failures,
        code_sha256=digest,control_maxima=maxima,gad_definition_version=2,
        git_revision=revision.stdout.strip() if revision.returncode == 0 else 'unknown',
        numpy_version=np.__version__,matplotlib_version=importlib.metadata.version('matplotlib'),python_version=sys.version))
    if completed:
        summarize(sorted(completed), output, config['bootstrap'])
        from plot_gad_mechanism_checks import plot_all
        plot_all(output)
    print('Results: '+str(output), flush=True)
    return int(bool(failures) or len(completed)!=len(worlds))


if __name__ == '__main__':
    raise SystemExit(main())
