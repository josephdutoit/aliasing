"""Compare plug-in representations on sparse tasks with canonical frozen-map GAD.

Run --profile smoke locally, or submit representation_models.sbatch overnight.
No experiment-14/15 checkpoints are required. See REPRESENTATION_MODELS.md.
"""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib
import importlib.metadata
import inspect
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np

from aliasing.benchmark_results import csv_write, json_write, summarize
from aliasing.representation_benchmark import TABLE_GROUPS, run_world
from aliasing.representation_models import FACTORIES, default_models, make_model


def defaults(profile):
    smoke = profile == 'smoke'
    return dict(profile=profile, seed=1729, features=4 if smoke else 16, input_dim=6 if smoke else 32,
        compression=2 if smoke else 4, width=6 if smoke else 16, task_basis='orthogonal',
        task_profiles=['balanced'], secondary_weight=.25, num_seeds=2 if smoke else 24,
        active_counts=[1, 2, 4] if smoke else [1, 2, 4, 8, 12, 16],
        sample_sizes=[3, 4, 6, 7, 16, 32, 128] if smoke else [3, 4, 5, 8, 15, 16, 17, 18, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        reference_sizes=[256, 1024] if smoke else [16384, 65536],
        n_representation=128 if smoke else 2048, n_validation=128 if smoke else 2048,
        n_test=256 if smoke else 8192, steps=80 if smoke else 2500, batch_size=128,
        learning_rate=.005, weight_decay=1e-5, draws=2 if smoke else 3, rotations=1 if smoke else 3,
        probe_repeats=2 if smoke else 8, bootstrap=100 if smoke else 1000,
        penalties=[1e-6, 1e-4, .01, .1, 1., 10.], model_modules=[])


def configure(args, saved=None):
    config = dict(saved) if saved is not None else defaults(args.profile or 'smoke')
    if saved is not None and args.profile is not None and args.profile != saved['profile']:
        raise ValueError('resume profile differs from saved configuration')
    lists = ('active_counts', 'sample_sizes', 'reference_sizes', 'task_profiles', 'penalties')
    for key in list(config):
        value = getattr(args, key, None)
        if value is not None:
            if key in lists:
                value = value.split(',')
                if key != 'task_profiles':
                    value = [float(x) if key == 'penalties' else int(x) for x in value]
            config[key] = value
    if args.model_module is not None:
        config['model_modules'] = args.model_module
    if args.models_file is not None:
        models = json.loads(args.models_file.read_text())
        config['models'] = models['models'] if isinstance(models, dict) else models
    elif saved is None:
        config['models'] = default_models(config['compression'], config['width'])
    if args.models is not None:
        requested = args.models.split(',')
        lookup = {m['name']: m for m in config['models']}
        if len(set(requested)) != len(requested) or any(name not in lookup for name in requested):
            raise ValueError('--models must select distinct names from the model configuration')
        config['models'] = [lookup[name] for name in requested]
    for key in lists:
        values = config[key]
        if not values or len(values) != len(set(values)):
            raise ValueError('empty or duplicate '+key)
    for key in ('sample_sizes', 'reference_sizes', 'active_counts'):
        config[key] = sorted(config[key])
    if config['features'] < 2 or config['input_dim'] < config['features']:
        raise ValueError('require input_dim >= features >= 2')
    positive = ('num_seeds', 'compression', 'width', 'n_representation', 'n_validation', 'n_test',
                'steps', 'batch_size', 'draws', 'probe_repeats')
    if any(config[key] < 1 for key in positive) or config['rotations'] < 0 or config['num_seeds'] < 2 or config['bootstrap'] < 20:
        raise ValueError('invalid counts; require >=2 seeds, >=20 bootstrap replicates and positive budgets')
    if config['seed'] < 0 or min(config['sample_sizes']) < 1 or not all(1 <= k <= config['features'] for k in config['active_counts']):
        raise ValueError('invalid seeds, readout sizes or active counts')
    if any(p not in ('balanced', 'A', 'B') for p in config['task_profiles']) or config['task_basis'] not in ('orthogonal', 'identity'):
        raise ValueError('unknown task profile or basis')
    if (not 0 < config['secondary_weight'] <= 1 or not np.isfinite(config['learning_rate']) or config['learning_rate'] <= 0 or
            not np.isfinite(config['weight_decay']) or config['weight_decay'] < 0 or
            any(not np.isfinite(p) or p <= 0 for p in config['penalties'])):
        raise ValueError('invalid importance weights, learning rate, decay or ridge penalties')
    if not isinstance(config['models'], list) or not config['models']:
        raise ValueError('models must be a nonempty JSON list')
    names = [model['name'] for model in config['models']]
    if len(names) != len(set(names)) or any(not name.replace('_', '').isalnum() or name.startswith('oracle_') for name in names):
        raise ValueError('model names must be unique alphanumeric/underscore names; oracle_ is reserved')
    for module in config['model_modules']:
        importlib.import_module(module)
    widths = [make_model(model).width for model in config['models']]
    if min(config['reference_sizes']) <= max(max(widths)+1, config['features']+1):
        raise ValueError('reference sizes must exceed every readout dimension, including the full-rank oracle')
    config['detail_sample_sizes'] = sorted(set([min(config['sample_sizes'], key=lambda n: abs(n-128)), max(config['sample_sizes'])]))
    return config


def code_fingerprint(config):
    files = [Path(__file__), Path(__file__).with_name('plot_representation_model_benchmark.py')]
    files += [ROOT/'src'/'aliasing'/(name+'.py') for name in (
        'controlled_superposition', 'mechanism_checks', 'precision_gad', 'representation_models', 'representation_benchmark', 'benchmark_results')]
    for module_name in config['model_modules']:
        module = importlib.import_module(module_name)
        source = Path(inspect.getfile(module)).resolve()
        files.extend(sorted(source.parent.rglob('*.py')) if source.name == '__init__.py' else [source])
    for spec in config['models']:
        files.append(Path(inspect.getfile(FACTORIES[spec['family']])).resolve())
        files.append(Path(inspect.getfile(type(make_model(spec)))).resolve())
    hashes = {}
    for path in sorted(set(path.resolve() for path in files)):
        name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return digest, hashes


def shard_complete(directory):
    path = directory/'complete.json'
    if not path.exists():
        return False
    recorded = json.loads(path.read_text())
    for name, expected in recorded['artifacts'].items():
        artifact = directory/name
        if not artifact.exists() or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
            raise ValueError('completed shard artifact missing or changed: '+str(artifact))
    return True


def save_world(directory, result, seed, k, profile):
    directory.mkdir(parents=True, exist_ok=True)
    for table in TABLE_GROUPS:
        csv_write(directory/(table+'.csv'), result['tables'][table])
    checkpoint = directory/'checkpoints.npz'
    temporary = directory/'checkpoints.npz.tmp'
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **result['states'])
    temporary.replace(checkpoint)
    artifacts = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in [checkpoint, *(directory/(name+'.csv') for name in TABLE_GROUPS)]}
    json_write(directory/'complete.json', dict(seed=seed, active_count=k, task_profile=profile,
        seconds=result['seconds'], controls=result['controls'], artifacts=artifacts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--profile', choices=('smoke', 'overnight'))
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--models-file', type=Path)
    parser.add_argument('--models', help='comma-separated names from defaults or --models-file')
    parser.add_argument('--model-module', action='append', help='import a module that registers custom model families')
    for key in ('active-counts', 'sample-sizes', 'reference-sizes', 'task-profiles', 'penalties', 'task-basis'):
        parser.add_argument('--'+key)
    for key in ('seed', 'features', 'input-dim', 'compression', 'width', 'num-seeds', 'n-representation',
                'n-validation', 'n-test', 'steps', 'batch-size', 'draws', 'rotations', 'probe-repeats', 'bootstrap'):
        parser.add_argument('--'+key, type=int)
    for key in ('learning-rate', 'weight-decay', 'secondary-weight'):
        parser.add_argument('--'+key, type=float)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.summarize_only and not args.resume:
        parser.error('--summarize-only requires --resume')
    if output.exists() and not args.resume:
        parser.error('output exists; select a new directory or use --resume')
    if args.resume and not (output/'configuration.json').exists():
        parser.error('--resume requires an existing configuration.json')
    saved = json.loads((output/'configuration.json').read_text()) if args.resume else None
    try:
        config = configure(args, saved['config'] if saved else None)
        digest, hashes = code_fingerprint(config)
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        parser.error(str(exc))
    expected = dict(config=config, code_sha256=digest, source_sha256=hashes, gad_definition_version=2,
        estimator_extension='canonical MP baseline plus explicit head correction and all six signed cross terms')
    if saved is not None and saved != expected:
        parser.error('resume requires identical scientific configuration and source code')
    output.mkdir(parents=True, exist_ok=True)
    json_write(output/'configuration.json', expected)
    worlds = list(itertools.product(range(config['num_seeds']), config['active_counts'], config['task_profiles']))
    completed, pending, failures = [], [], []
    try:
        for seed, k, profile in worlds:
            directory = output/'shards'/f'seed{seed}_k{k}_{profile}'
            if shard_complete(directory):
                completed.append(directory)
            else:
                pending.append((seed, k, profile, directory))
    except ValueError as exc:
        parser.error(str(exc))
    if args.summarize_only:
        pending = []
    workers = min(args.workers, int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1)))
    print(f'{len(pending)} pending, {len(completed)} complete; {workers} workers', flush=True)
    started = time.monotonic()
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_world, config, seed, k, profile): (seed, k, profile, directory)
                       for seed, k, profile, directory in pending}
            for future in as_completed(futures):
                seed, k, profile, directory = futures.pop(future)
                try:
                    result = future.result()
                    save_world(directory, result, seed, k, profile)
                    completed.append(directory)
                    print(f'Complete seed={seed}, k={k}, profile={profile}: {result["seconds"]:.1f}s', flush=True)
                except Exception as exc:
                    failure = dict(seed=seed, active_count=k, task_profile=profile, error=repr(exc), traceback=traceback.format_exc())
                    failures.append(failure)
                    print(f'FAILED seed={seed}, k={k}: {exc!r}', flush=True)
                json_write(output/'progress.json', dict(completed=len(completed), expected=len(worlds), failures=failures))
    json_write(output/'progress.json', dict(completed=len(completed), expected=len(worlds), failures=failures))
    controls = {}
    for directory in completed:
        for key, value in json.loads((directory/'complete.json').read_text())['controls'].items():
            controls[key] = max(controls.get(key, 0.), value)
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    json_write(output/'run_manifest.json', dict(completed=len(completed), expected=len(worlds), failures=failures,
        control_maxima=controls, code_sha256=digest, gad_definition_version=2,
        git_revision=revision.stdout.strip() if revision.returncode == 0 else 'unknown',
        numpy_version=np.__version__, matplotlib_version=importlib.metadata.version('matplotlib'),
        python_version=sys.version, workers=workers, invocation=sys.argv,
        invocation_seconds=time.monotonic()-started, complete=(len(completed) == len(worlds) and not failures)))
    if completed:
        print('Writing seed summaries, paired intervals, figures and review bundle', flush=True)
        summarize(sorted(completed), output, config['bootstrap'])
        from plot_representation_model_benchmark import plot_all
        plot_all(output)
    print('Results: '+str(output), flush=True)
    return int(bool(failures) or len(completed) != len(worlds))


if __name__ == '__main__':
    raise SystemExit(main())
