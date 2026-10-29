"""Recover experiment-17 failures into a NEW directory, preserving old runs.

Only the documented numerical evaluator and summary-schema changes are
accepted. Scientific config and all training/generator/model sources must
match. Completed shards are checksummed and hard-linked (copied across disks).
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from aliasing.benchmark_results import json_write
from aliasing.learned_geometry_plan import validate, worlds

PREVIOUS_REPLACEMENTS = {
    'src/aliasing/learned_geometry.py': '321d28e2b6b9e81485336e9723159b910a69db286eb2593655bd597791e9720c',
    'src/aliasing/learned_geometry_results.py': '301a8f8c8f8ae195bd30c896f3820856091e81c8992ce71119f4b8b8e2a7462e',
}
ADDITIONS = {'src/aliasing/learned_geometry_precision.py'}


def runner():
    path = ROOT/'experiments/17_learned_relu_geometry.py'
    spec = importlib.util.spec_from_file_location('geometry_recovery_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, target)
    return target


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('recovery output must be separate from, and not contain, the original output')
    original_path = source/'configuration.json'
    original = json.loads(original_path.read_text())
    config = validate(original['config'])
    module = runner()
    digest, current_hashes = module.fingerprint()
    previous_hashes = original['source_sha256']
    recorded_digest = hashlib.sha256(json.dumps(previous_hashes, sort_keys=True).encode()).hexdigest()
    if original.get('gad_definition_version') != 2 or original['code_sha256'] != recorded_digest:
        raise ValueError('original definition version or source-manifest digest is inconsistent')
    differences = {name: dict(previous=previous_hashes.get(name), current=current_hashes.get(name))
                   for name in set(previous_hashes) | set(current_hashes)
                   if previous_hashes.get(name) != current_hashes.get(name)}
    for name, change in differences.items():
        if name in ADDITIONS and change['previous'] is None and change['current'] is not None:
            continue
        if name in PREVIOUS_REPLACEMENTS and change['previous'] == PREVIOUS_REPLACEMENTS[name] and change['current'] is not None:
            continue
        raise ValueError('recovery rejects an unapproved source change: '+name)
    # Validate original frozen source snapshots too, including every generator
    # and training dependency. Metadata alone is not sufficient provenance.
    for name, expected_hash in previous_hashes.items():
        snapshot = source/'source'/name
        if not snapshot.is_file() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != expected_hash:
            raise ValueError('original source snapshot missing or changed: '+name)
    expected = dict(config=config, source_sha256=current_hashes, code_sha256=digest, gad_definition_version=2)
    source_digest = hashlib.sha256(original_path.read_bytes()).hexdigest()
    if destination.exists():
        recovery = json.loads((destination/'recovery.json').read_text())
        if (recovery['original_output'] != str(source) or recovery['original_configuration_sha256'] != source_digest or
                json.loads((destination/'configuration.json').read_text()) != expected):
            raise ValueError('existing recovery directory has different provenance/configuration')
        return recovery
    planned = worlds(config)
    inherited, pending = [], []
    for world in planned:
        name = f'{world["experiment"]}_{world["scenario"]}_seed{world["seed"]}'
        directory = source/'shards'/name
        if module.complete(directory):
            inherited.append(name)
        else:
            pending.append(name)
    temporary = destination.with_name(destination.name+'.preparing')
    if temporary.exists():
        raise ValueError('incomplete preparation exists; inspect before retry: '+str(temporary))
    temporary.mkdir(parents=True)
    try:
        (temporary/'shards').mkdir()
        for name in inherited:
            shutil.copytree(source/'shards'/name, temporary/'shards'/name, copy_function=link_or_copy)
        for name in current_hashes:
            target = temporary/'source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, target)
        provenance = temporary/'provenance'
        provenance.mkdir()
        shutil.copy2(original_path, provenance/'original_configuration.json')
        for name in ('progress.json', 'run_manifest.json'):
            if (source/name).is_file():
                shutil.copy2(source/name, provenance/('original_'+name))
        shutil.copytree(source/'source', provenance/'original_source', copy_function=link_or_copy)
        recovery = dict(original_output=str(source), original_configuration_sha256=source_digest,
            original_code_sha256=original['code_sha256'], current_code_sha256=digest,
            changed_sources=differences, inherited_worlds=inherited, pending_worlds=pending,
            inherited_count=len(inherited), pending_count=len(pending), expected_count=len(planned),
            repair_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            inheritance='Completed artifacts unchanged; hard-linked where supported. Original configurations and snapshots retained.')
        json_write(temporary/'configuration.json', expected)
        json_write(temporary/'recovery.json', recovery)
        json_write(temporary/'worlds.json', planned)
        (temporary/'RECOVERY.md').write_text(f'''# Numerical recovery

Original output: {source}

Reused {len(inherited)} checksummed completed worlds; {len(pending)} worlds
need retry. Scientific configuration and training/generator/model code are
unchanged. Original outputs are not modified. See recovery.json and provenance/.

The earlier evaluator is tried first. Its accepted values remain unchanged.
Failed contractions retry at 80, then 160 decimal digits with the SAME inverse,
singular-value cutoff, reference head, candidate head and float64 predictions.
Canonical GAD components retain their definitions. A separately computed
arithmetic_roundoff vector measures fit, prediction and subtraction rounding;
it is not called aliasing or data insufficiency. Its four extra signed cross
terms must be included when reconstructing risk.

arithmetic_roundoff_measured=0 means the earlier identity tolerance passed and
no separate arithmetic contribution was measured (represented as zero in the
schema). Legacy inherited rows are normalized to this convention only during
summary creation; their source CSVs remain byte-for-byte unchanged.

gad_arithmetic_reliable=0 identifies rows where implementation rounding is
larger than the original risk-scale tolerance. Inspect readouts_numerical_warnings.csv
before interpreting such rows as statistical aliasing. Means of this flag
represent fractions, not a new proof of numerical accuracy.

Conditional and per-feature rows use the same evaluated component vectors;
cross_total includes actual pairwise cross products, not an unexplained gap.
Old accepted rows and repaired rows have explicit source provenance. The
recovery does not certify that old tiny component energies were exact.
''')
        temporary.replace(destination)
    except BaseException:
        # Never remove the original. Leave interrupted preparations visible.
        raise
    return recovery


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True, help='one old family directory, e.g. .../14872283/phase')
    parser.add_argument('--output-dir', type=Path, required=True, help='NEW family directory')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('workers must be positive')
    try:
        recovery = prepare(args.input_dir, args.output_dir)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(f'Recovery: {recovery["inherited_count"]} completed worlds retained; {recovery["pending_count"]} originally unfinished', flush=True)
    if args.prepare_only:
        return 0
    return subprocess.run([sys.executable, str(ROOT/'experiments/17_learned_relu_geometry.py'),
        '--resume', '--output-dir', str(args.output_dir), '--workers', str(args.workers)], cwd=ROOT).returncode


if __name__ == '__main__':
    raise SystemExit(main())
