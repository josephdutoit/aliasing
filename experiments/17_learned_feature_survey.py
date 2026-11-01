"""Broad static learned-feature survey using experiment 16's models and GAD.

Use --plan-only to inspect all blocks, --job-index for a SLURM array, and
--collect-only after the array finishes. See BROAD_FEATURE_SURVEY.md.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np

from aliasing.benchmark_results import json_write, summarize
from aliasing.broad_survey import build_jobs, collect, fingerprint, profile_pairs
from aliasing.representation_benchmark import TABLE_GROUPS, run_world
from aliasing.survey_diagnostics import EXTRA_GROUPS

GROUPS = TABLE_GROUPS | EXTRA_GROUPS


def legacy_runner():
    spec = importlib.util.spec_from_file_location('survey_benchmark_runner', Path(__file__).with_name('16_representation_model_benchmark.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan_for(args, previous=None):
    if previous and args.profile and args.profile != previous['profile']:
        raise ValueError('existing survey profile differs; select a new output directory')
    profile = args.profile or (previous['profile'] if previous else 'research')
    runner = legacy_runner()
    smoke = profile == 'smoke'
    base = previous['base_config'].copy() if previous else runner.defaults('smoke' if smoke else 'overnight')
    if not previous:
        base.update(width=4 if smoke else 16, num_seeds=2 if smoke else 24,
            active_counts=[1, 4] if smoke else [1, 2, 4, 8, 12, 16],
            reference_sizes=[128, 256] if smoke else [16384, 65536],
            sample_sizes=[2, 3, 4, 5, 6, 8, 16, 32] if smoke else
                [3, 4, 5, 8, 9, 15, 16, 17, 18, 31, 32, 33, 63, 64, 65, 66, 128, 256, 512, 1024, 2048, 4096],
            n_test=128 if smoke else 8192, n_validation=64 if smoke else 2048,
            steps=40 if smoke else 2500, probe_repeats=1 if smoke else 8,
            draws=1 if smoke else 3, rotations=1 if smoke else 3)
    for key in ('seed', 'num_seeds', 'bootstrap'):
        if getattr(args, key) is not None:
            base[key] = getattr(args, key)
    for key in ('reference_sizes', 'sample_sizes', 'active_counts'):
        value = getattr(args, key)
        if value is not None:
            values = [int(x) for x in value.split(',')]
            if not values or min(values) < 1 or len(values) != len(set(values)):
                raise ValueError('invalid '+key)
            base[key] = sorted(values)
    if base['num_seeds'] < 2 or base['bootstrap'] < 20 or base['seed'] < 0:
        raise ValueError('require >=2 independent seeds, >=20 bootstrap replicates and nonnegative seed')
    if not all(1 <= k <= base['features'] for k in base['active_counts']):
        raise ValueError('active counts must be between 1 and features')
    jobs = build_jobs(base, smoke)
    digest, hashes = fingerprint(ROOT)
    return dict(profile=profile, base_config=base, jobs=jobs, code_sha256=digest,
                source_sha256=hashes, gad_definition_version=2)


def install_plan(output, plan):
    """Atomically install a shared immutable plan, safe for concurrent array jobs."""
    output.mkdir(parents=True, exist_ok=True)
    target = output/'survey_plan.json'
    # PIDs can coincide on different cluster nodes; use a random unique name.
    with tempfile.NamedTemporaryFile(mode='w', dir=output, prefix='plan.', suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(plan, sort_keys=True, indent=2)+'\n')
    try:
        try:
            os.link(temporary, target)
        except FileExistsError:
            if json.loads(target.read_text()) != plan:
                raise ValueError('output contains a different scientific plan or source fingerprint')
    finally:
        temporary.unlink(missing_ok=True)


def execute_job(output, job, plan, args, runner):
    directory = output/'jobs'/job['job_id']
    if directory.exists() and not args.resume:
        raise ValueError(f'{directory} exists; use --resume with identical code/configuration')
    directory.mkdir(parents=True, exist_ok=True)
    # Prevent accidental concurrent writers of the SAME job. Different array
    # indices have disjoint directories. Locks auto-release on process exit.
    import fcntl
    lock = (directory/'run.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise ValueError('another process is already running '+job['job_id']) from exc
    try:
        return _execute_locked(directory, job, plan, args, runner)
    finally:
        lock.close()


def _execute_locked(directory, job, plan, args, runner):
    config = job['config']
    expected = dict(config=config, code_sha256=plan['code_sha256'], source_sha256=plan['source_sha256'],
                    job_id=job['job_id'], study=job['study'], gad_definition_version=2)
    saved = directory/'configuration.json'
    if saved.exists() and json.loads(saved.read_text()) != expected:
        raise ValueError('resume requires identical job configuration and source code')
    json_write(saved, expected)
    worlds = list(itertools.product(range(config['num_seeds']), config['active_counts'], config['task_profiles']))
    done, pending, failures = [], [], []
    for seed, k, profile in worlds:
        shard = directory/'shards'/f'seed{seed}_k{k}_{profile}'
        if runner.shard_complete(shard):
            done.append(shard)
        else:
            pending.append((seed, k, profile, shard))
    workers = min(args.workers, int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1)))
    started = time.monotonic()
    print(f"{job['job_id']}: {len(done)}/{len(worlds)} complete, {len(pending)} pending; {workers} workers", flush=True)
    if pending and not args.summarize_only:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_world, config, seed, k, profile): (seed, k, profile, shard)
                       for seed, k, profile, shard in pending}
            for future in as_completed(futures):
                seed, k, profile, shard = futures[future]
                try:
                    result = future.result()
                    runner.save_world(shard, result, seed, k, profile, GROUPS)
                    done.append(shard)
                    print(f"{job['job_id']}: seed={seed}, k={k}, {profile}: {result['seconds']:.1f}s", flush=True)
                except Exception as exc:
                    failures.append(dict(seed=seed, active_count=k, task_profile=profile,
                        error=repr(exc), traceback=traceback.format_exc()))
                    print(f"FAILED {job['job_id']}: seed={seed}, k={k}: {exc!r}", flush=True)
                json_write(directory/'progress.json', dict(completed=len(done), expected=len(worlds), failures=failures))
    controls = {}
    for shard in done:
        for key, value in json.loads((shard/'complete.json').read_text())['controls'].items():
            controls[key] = max(controls.get(key, 0.), value)
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    complete = len(done) == len(worlds) and not failures
    manifest = dict(completed=len(done), expected=len(worlds), failures=failures, complete=complete,
        summaries_complete=False,
        control_maxima=controls, code_sha256=plan['code_sha256'], gad_definition_version=2,
        git_revision=revision.stdout.strip() if revision.returncode == 0 else 'unknown',
        numpy_version=np.__version__, python_version=sys.version, workers=workers,
        invocation=sys.argv, invocation_seconds=time.monotonic()-started)
    json_write(directory/'run_manifest.json', manifest)
    if done:
        summarize(sorted(done), directory, config['bootstrap'], GROUPS)
        profile_pairs(directory, GROUPS, config['bootstrap'])
        from plot_representation_model_benchmark import plot_all
        plot_all(directory)
        manifest['summaries_complete'] = True
        json_write(directory/'run_manifest.json', manifest)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--profile', choices=('smoke', 'research'))
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--job-index', type=int)
    for flag in ('resume', 'plan-only', 'collect-only', 'summarize-only'):
        parser.add_argument('--'+flag, action='store_true')
    for flag in ('num-seeds', 'seed', 'bootstrap'):
        parser.add_argument('--'+flag, type=int)
    for flag in ('reference-sizes', 'sample-sizes', 'active-counts'):
        parser.add_argument('--'+flag)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if sum((args.plan_only, args.collect_only, args.summarize_only)) > 1:
        parser.error('select only one of plan/collect/summarize-only')
    if args.summarize_only and not args.resume:
        parser.error('--summarize-only requires --resume')
    output = args.output_dir.resolve()
    try:
        saved = output/'survey_plan.json'
        previous = json.loads(saved.read_text()) if saved.exists() else None
        if (args.collect_only or args.resume) and previous is None:
            raise ValueError('collect/resume requires an existing survey_plan.json')
        if args.collect_only:
            if args.job_index is not None:
                raise ValueError('--collect-only covers the entire survey; omit --job-index')
            return int(not collect(output, previous))
        plan = plan_for(args, previous)
        install_plan(output, plan)
        if args.job_index is not None and not 0 <= args.job_index < len(plan['jobs']):
            raise ValueError('--job-index outside the planned array range')
        for i, job in enumerate(plan['jobs']):
            c = job['config']
            worlds = c['num_seeds']*len(c['active_counts'])*len(c['task_profiles'])
            print(f"{i}: {job['job_id']} ({worlds} worlds, width={c['width']}, tasks={c['training_tasks']})", flush=True)
        if args.plan_only:
            return 0
        selected = plan['jobs'] if args.job_index is None else [plan['jobs'][args.job_index]]
        results = [execute_job(output, job, plan, args, legacy_runner()) for job in selected]
        if args.job_index is None:
            collect(output, plan)
        return int(not all(results))
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
