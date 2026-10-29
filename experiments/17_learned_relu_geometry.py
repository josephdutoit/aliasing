"""Learned ReLU geometry reconnaissance. See LEARNED_RELU_GEOMETRY.md."""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
from aliasing.benchmark_results import csv_write, json_write
from aliasing.learned_geometry import TABLES, run_world
from aliasing.learned_geometry_plan import EXPERIMENTS, defaults, validate, worlds
from aliasing.learned_geometry_results import summarize


def fingerprint():
    paths = [Path(__file__), ROOT/'experiments/plot_learned_relu_geometry.py']
    # Include transitive local imports, not just the new evaluator.
    paths += sorted((ROOT/'src/aliasing').glob('*.py'))
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return digest, hashes


def complete(directory):
    marker = directory/'complete.json'
    if not marker.exists():
        return False
    manifest = json.loads(marker.read_text())
    for name, digest in manifest['artifacts'].items():
        path = directory/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('completed shard artifact missing or changed: '+str(path))
    return True


def save(directory, result, world):
    directory.mkdir(parents=True, exist_ok=True)
    for table in TABLES:
        csv_write(directory/(table+'.csv'), result['tables'][table])
    temporary = directory/'checkpoints.npz.tmp'
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **result['states'])
    temporary.replace(directory/'checkpoints.npz')
    paths = [directory/'checkpoints.npz', *(directory/(table+'.csv') for table in TABLES)]
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    json_write(directory/'complete.json', dict(world=world, seconds=result['seconds'], artifacts=hashes))


def write_review(output, config, completed, expected, failures):
    status = 'COMPLETE' if len(completed) == expected and not failures else 'PARTIAL'
    extra_figures = '\n\n'.join(f'![{name}]({name}.png)' for name in ('gate_boundaries', 'optimization_basins') if (output/(name+'.png')).exists())
    content = f'''# Learned ReLU geometry reconnaissance — {status}

Completed {len(completed)}/{expected} worlds. Current invocation failures: {len(failures)}.

Read `geometry_mean.csv`, `geometry_paired.csv`, `readouts_paired.csv`, then
`geometry_seed_means.csv` and `readouts_seed_means.csv`. Read starts/gates and
shard checkpoints to inspect competing learned geometries. Never count starts
or head draws as independent representation seeds.

## Questions

| Study | What to look for |
| --- | --- |
| phase | Learned encoder signs/angles and thresholds as coactivation changes; decay dependence; competing starts |
| finite | Whether finite representation-training data change the selected geometry or just its accuracy |
| ablations | Joint versus fixed encoder/fixed thresholds; refitted gate removal, rotations and bias shifts |
| tasks | Whether task importance changes which features/regions the learned gates preserve |
| blocks | Whether the pair mechanism persists in larger jointly learned compressed networks |
| landscape | Conditional encoder-angle profiles with optimized decoders; compare multiple starts, not a global certificate |
| higher_order | Learned geometry at matched activation statistics through order three within parity blocks |

## Interpretation

All maps are trained before independent heads are fitted. `restart_*` maps
are trained local solutions, not frozen random features. Best starts and
checkpoints are selected ONLY using the representation-training objective.
Population-gradient diagnostics are measured afterward and do not select maps.
Fixed-angle profiles freeze the encoder as a controlled conditional search;
other main arms learn it jointly. No run certifies global optimality.

Pair marginals are fixed while covariance changes with coactivation. Parity
blocks preserve low-order laws, but have variable active count and are not
the old exact-k generator. Gate angles require permutation/scaling/sign care;
raw parameters and restart-level geometry are retained.

Pair population integrals use numerical quadrature, NOT an exact population
oracle. Compare reference levels and rerun at higher training/evaluation orders
before a theorem claim. Larger models use independent finite reference fits.

Canonical GAD: data insufficiency H_P(H_T^+H_T-I)beta; MP aliasing H_PH_T^+r_T;
signed model residual H_P beta-Y_P. Ridge/native predictions add an explicit
estimator correction. All six signed cross terms are retained. The alias
column on ridge rows remains the MP baseline. Zero residual denominators and
undefined diagnostics are blank/NaN rather than artificial zeroes.

`residual_leverage_J` is E[mean_tasks(r^2)*leverage] on the evaluation geometry.
It is an established large-sample diagnostic, not a new theorem or a guarantee
near rank changes. Check n*aliasing only at adequately sampled, stable heads.
Conditional strata use the SAME global reference. Probability-weighted strata
energies and their cross terms reconstruct the corresponding global MP risk.

Seed bootstrap intervals are pointwise exploratory intervals. Inspect medians,
maxima and top-one shares when aliasing means are unstable. These tests are
designed to reject mechanisms as well as support them.

## Figures

![geometry](learned_geometry.png)

![readouts](finite_sample_gad.png)

![alignment](residual_alignment.png)

{extra_figures}

## Files to return

Return `review_bundle.zip`. It contains summaries, seed-level means, paired
contrasts, starts, configuration, provenance and plots. Full raw readouts,
histories, gate/parameter tables and all restart checkpoints remain in shards
on the compute node. This run used {config['num_seeds']} seeds and
{config['restarts']} starts per trained arm.
'''
    (output/'RESULTS.md').write_text(content)
    with zipfile.ZipFile(output/'review_bundle.zip.tmp', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.suffix in ('.csv', '.json', '.md', '.png') and path.name not in ('gates.csv', 'parameters.csv'):
                archive.write(path, path.name)
        for path in sorted((output/'source').rglob('*.py')):
            archive.write(path, str(path.relative_to(output)))
    (output/'review_bundle.zip.tmp').replace(output/'review_bundle.zip')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--profile', choices=('smoke', 'overnight', 'extended'))
    parser.add_argument('--experiments', help='comma-separated study names or all')
    parser.add_argument('--config-json', type=Path, help='override scientific defaults with a JSON object')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--list-plan', action='store_true')
    for option in ('num-seeds', 'restarts', 'steps', 'draws', 'training-order', 'evaluation-order', 'n-test'):
        parser.add_argument('--'+option, type=int)
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.summarize_only and not args.resume):
        parser.error('positive workers required; summarize-only requires resume')
    output = args.output_dir.resolve() if args.output_dir else None
    if not args.list_plan and output is None:
        parser.error('--output-dir required')
    saved = None
    if args.resume:
        if output is None or not (output/'configuration.json').exists():
            parser.error('resume requires existing configuration.json')
        saved = json.loads((output/'configuration.json').read_text())
    config = dict(saved['config']) if saved else defaults(args.profile or 'smoke')
    if args.profile and args.profile != config['profile']:
        parser.error('profile differs from saved configuration')
    if args.config_json:
        overrides = json.loads(args.config_json.read_text())
        if not isinstance(overrides, dict) or set(overrides)-set(config):
            parser.error('config JSON must be an object containing recognized configuration keys')
        config.update(overrides)
    if args.experiments:
        config['experiments'] = list(EXPERIMENTS) if args.experiments == 'all' else args.experiments.split(',')
    for key in ('num_seeds', 'restarts', 'steps', 'draws', 'training_order', 'evaluation_order', 'n_test'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    try:
        validate(config)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    planned = worlds(config)
    if args.list_plan:
        print(json.dumps(dict(worlds=len(planned), worlds_by_experiment=dict(Counter(w['experiment'] for w in planned)),
                              config=config), indent=2))
        return 0
    digest, hashes = fingerprint()
    expected = dict(config=config, source_sha256=hashes, code_sha256=digest, gad_definition_version=2)
    if saved is not None and expected != saved:
        parser.error('resume requires identical scientific configuration and source hashes')
    if output.exists() and not args.resume:
        parser.error('output exists; choose a new output directory or use resume')
    output.mkdir(parents=True, exist_ok=True)
    json_write(output/'configuration.json', expected)
    json_write(output/'worlds.json', planned)
    for name in hashes:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    pending, completed, failures = [], [], []
    for world in planned:
        directory = output/'shards'/f'{world["experiment"]}_{world["scenario"]}_seed{world["seed"]}'
        if complete(directory):
            completed.append(directory)
        else:
            pending.append((world, directory))
    workers = min(args.workers, int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1)))
    print(f'{len(pending)} pending, {len(completed)} complete; {workers} workers', flush=True)
    start = time.monotonic()
    if pending and not args.summarize_only:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_world, config, world): (world, directory) for world, directory in pending}
            for future in as_completed(futures):
                world, directory = futures[future]
                try:
                    result = future.result()
                    save(directory, result, world)
                    completed.append(directory)
                    print(f'Complete {directory.name}: {result["seconds"]:.1f}s', flush=True)
                except Exception as error:
                    failures.append(dict(world=world, error=repr(error), traceback=traceback.format_exc()))
                    print(f'FAILED {directory.name}: {error!r}', flush=True)
                json_write(output/'progress.json', dict(completed=len(completed), expected=len(planned), failures=failures))
    json_write(output/'progress.json', dict(completed=len(completed), expected=len(planned), failures=failures))
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    json_write(output/'run_manifest.json', dict(completed=len(completed), expected=len(planned), failures=failures,
        complete=(len(completed) == len(planned) and not failures), numpy_version=np.__version__,
        python_version=sys.version, platform=platform.platform(), workers=workers, invocation=sys.argv,
        git_revision=revision.stdout.strip() if revision.returncode == 0 else 'connector-checkout',
        source_sha256=hashes, invocation_seconds=time.monotonic()-start))
    if completed:
        print('Writing seed summaries, paired contrasts and review bundle', flush=True)
        summarize(sorted(completed), output, config['bootstrap'])
        path = ROOT/'experiments/plot_learned_relu_geometry.py'
        module_spec = importlib.util.spec_from_file_location('geometry_plots', path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        module.plot(output)
    write_review(output, config, completed, len(planned), failures)
    print('Results: '+str(output), flush=True)
    return 0 if len(completed) == len(planned) and not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
