"""A blocked survey around experiment 16, with shared seeds across contrasts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .benchmark_results import CI_METRICS, csv_write, json_write, read_rows
from .representation_benchmark import TABLE_GROUPS


def build_jobs(base, smoke=False):
    """Change one factor per block; baseline also contains paired A/B priorities."""
    widths = [2, 4] if smoke else [4, 8, 16, 32, 64]
    anchor = 4 if smoke else 16
    blocks = [('width_'+str(w), 'width', dict(width=w)) for w in widths]
    blocks += [
        ('task_first', 'task', dict(training_tasks='first')),
        ('task_half', 'task', dict(training_tasks='first_half')),
        ('data_small', 'representation_data', dict(n_representation=64 if smoke else 512)),
        ('data_large', 'representation_data', dict(n_representation=256 if smoke else 8192)),
        ('optimization_long', 'optimization', dict(steps=120 if smoke else 7500)),
        ('decay_zero', 'regularization', dict(weight_decay=0.)),
        ('decay_large', 'regularization', dict(weight_decay=1e-3)),
        ('identity_tasks', 'task_basis', dict(task_basis='identity')),
    ]
    jobs = []
    for job_id, study, changes in blocks:
        config = base | dict(width=anchor, training_tasks='all', survey_diagnostics=True,
                            n_initialization=base['n_representation'],
                            threshold_offsets=[-.5, .5, 1., 2.]) | changes
        # Stable architecture names share data/init seeds across other factors.
        config['models'] = [dict(name='standard', family='mlp', widths=[config['width']]),
            dict(name='bottleneck', family='bottleneck', compression=config['compression'], width=config['width'])]
        config['task_profiles'] = ['balanced', 'A', 'B'] if job_id == 'width_'+str(anchor) else ['balanced']
        config['detail_sample_sizes'] = sorted(set([min(config['sample_sizes'], key=lambda n: abs(n-128)),
                                                   max(config['sample_sizes'])]))
        if min(config['reference_sizes']) <= max(config['width'], config['features'])+1:
            raise ValueError('references must exceed every nominal readout dimension')
        jobs.append(dict(job_id=job_id, study=study, config=config))
    return jobs


def fingerprint(root):
    paths = sorted((root/'src'/'aliasing').glob('*.py')) + [root/'experiments'/name for name in (
        '16_representation_model_benchmark.py', '17_learned_feature_survey.py', 'plot_representation_model_benchmark.py')]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes


def profile_pairs(output, groups, bootstrap):
    """Compare the SAME evaluated tasks under B-minus-A training priorities."""
    rng = np.random.default_rng(70123)
    for table in ('readouts', 'task_readouts'):
        keys = tuple(k for k in groups[table] if k != 'task_profile')
        rows = list(read_rows(output/(table+'_seed_means.csv')))
        lookup = {tuple(r[k] for k in (*keys, 'seed')): r for r in rows if r['task_profile'] == 'A'}
        collected = {}
        for r in rows:
            if r['task_profile'] != 'B':
                continue
            other = lookup.get(tuple(r[k] for k in (*keys, 'seed')))
            if other is None:
                continue
            key = tuple(r[k] for k in keys)
            for metric in CI_METRICS:
                a, b = r.get(metric, np.nan), other.get(metric, np.nan)
                if np.isfinite(a) and np.isfinite(b):
                    collected.setdefault((key, metric), []).append(a-b)
        paired = []
        for (key, metric), values in sorted(collected.items()):
            boots = rng.choice(values, (bootstrap, len(values))).mean(axis=1) if len(values) > 1 else [np.nan]
            lo, hi = np.quantile(boots, [.025, .975])
            paired.append(dict(zip(keys, key)) | dict(contrast='B_minus_A', metric=metric,
                num_pairs=len(values), candidate_minus_reference=float(np.mean(values)),
                ci95_low=float(lo), ci95_high=float(hi)))
        csv_write(output/(table+'_profile_paired.csv'), paired)


def collect(output, plan):
    """Review existing seed summaries only; never launch training during collection."""
    manifests, missing, all_pairs = [], [], []
    for job in plan['jobs']:
        path = output/'jobs'/job['job_id']
        manifest = path/'run_manifest.json'
        if not manifest.exists():
            missing.append(job['job_id'])
            continue
        record = json.loads(manifest.read_text())
        manifests.append(dict(job_id=job['job_id'], study=job['study'], **record))
        for table in ('readouts', 'task_readouts'):
            source = path/(table+'_paired.csv')
            if source.exists():
                all_pairs.extend(dict(job_id=job['job_id'], study=job['study'], table=table, **r)
                                 for r in read_rows(source))
    csv_write(output/'survey_paired.csv', all_pairs)
    block_pairs(output, plan)
    overview_plot(output, plan)
    complete = not missing and all(m['complete'] and m.get('summaries_complete', False) for m in manifests)
    json_write(output/'survey_manifest.json', dict(complete=complete, missing_jobs=missing,
        jobs=manifests, code_sha256=plan['code_sha256'], gad_definition_version=2))
    lines = ['# Learned-feature broad survey', '',
        'Status: '+('COMPLETE' if complete else 'PARTIAL; inspect missing jobs and failed worlds.'), '',
        '## Matched comparisons', '',
        'Each job compares learned and frozen random features from identical initial hidden parameters. '
        'Independent readout orders, reference, validation and test samples are shared. '
        'The representation learner also sees its separately reported training split. '
        'Comparisons match architecture and updates, not total labeled-data exposure or parameter counts across architectures.', '',
        '| Job | Study | Width | Trained tasks | Representation examples | Updates | Completed worlds |',
        '| --- | --- | ---: | --- | ---: | ---: | ---: |']
    lookup = {m['job_id']: m for m in manifests}
    for job in plan['jobs']:
        c, m = job['config'], lookup.get(job['job_id'])
        status = f"{m['completed']}/{m['expected']}" if m else 'missing'
        lines.append(f"| {job['job_id']} | {job['study']} | {c['width']} | {c['training_tasks']} | {c['n_representation']} | {c['steps']} | {status} |")
    lines += ['', '## Reading the outputs', '',
        '- Start with survey_paired.csv: within-job candidate-minus-reference effects and pointwise seed bootstrap intervals. '
        'Use independent MP rows for canonical GAD; ridge rows retain MP aliasing plus an estimator correction.',
        '- block_paired.csv compares each job against the width-16 balanced/all-task baseline on shared problem seeds. '
        'The task-basis block changes target definitions and is explicitly flagged. '
        'Width, task exposure, data budget and optimization contrasts are hypotheses, not claims of improvement.',
        '- Every job retains raw per-world CSVs/checkpoints, seed means, sample standard deviations, medians, maxima, '
        'top-five shares for risk/aliasing, reference convergence and four figures in jobs/JOB/.',
        '- task_readouts_* decomposes each original task with the same global reference head. '
        'Single/half-task training still evaluates every task; training_task_weight records exposure. '
        'readouts_profile_paired.csv and task_readouts_profile_paired.csv compare B-minus-A priorities. '
        'Frozen random MP controls should be invariant to that priority swap; weighted risk/ridge selection need not be.',
        '- alias_modes_* records residual projection, singular value and population visibility. '
        'Diagonal mode energies are NOT an additive partition: signed_mode_offdiagonal_sum retains their cross terms. '
        'Mode numbers cannot be matched semantically across maps.',
        '- Latent Jacobian overlap is a local diagnostic. It does not certify superposition, injectivity or collisions. '
        'Check latent recovery, task GAD and gate interventions together.',
        '- Threshold units are per-unit representation-training preactivation standard deviations. '
        'Positive offsets raise thresholds. Rotations preserve preactivation Gram/weight spectra; '
        'activity matching additionally changes biases. No post-ReLU spectrum invariance is asserted.',
        '- The largest reference remains a finite estimate. Retain all signed risk cross terms; '
        'never report positive energies alone as an additive partition.',
        '- k changes coactivation AND means/covariance. Widths use shared problem seeds, not nested hidden units. '
        'Across representation-data budgets the current sampler is not prefix-stable, so those datasets are not nested.',
        '- All intervals are exploratory and pointwise. Confirm selected mechanisms with fresh seeds before theorem claims.',
        '- Stable-output native heads have seen representation-training labels; compare their data budgets explicitly. '
        'The wide all-linear control already expresses the synthetic linear teacher when its rank is sufficient.', '',
        '## Review figures', '',
        '![Width survey](survey_overview.png)', '']
    for job in plan['jobs']:
        path = output/'jobs'/job['job_id']/'readout_and_gad.png'
        if path.exists():
            lines += [f"### {job['job_id']}", '', f'![Readouts and GAD]({path.relative_to(output)})', '']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    temporary = output/'results_bundle.zip.tmp'
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
        paths = list(output.glob('*.json')) + list(output.glob('*.csv')) + list(output.glob('*.png')) + [output/'RESULTS.md']
        for job in plan['jobs']:
            directory = output/'jobs'/job['job_id']
            paths += [p for p in directory.glob('*') if p.is_file() and p.suffix in ('.csv', '.json', '.md', '.png')]
        for path in sorted(paths):
            archive.write(path, path.relative_to(output))
    temporary.replace(output/'results_bundle.zip')
    return complete


def block_pairs(output, plan):
    """Seed-paired block effects; changes of task basis are explicitly flagged."""
    anchor_width = 4 if plan['profile'] == 'smoke' else 16
    anchor = output/'jobs'/('width_'+str(anchor_width))/'readouts_seed_means.csv'
    if not anchor.exists():
        csv_write(output/'block_paired.csv', [])
        return
    keys = tuple(k for k in TABLE_GROUPS['readouts'] if k != 'task_profile')
    lookup = {tuple(r[k] for k in (*keys, 'seed')): r for r in read_rows(anchor)
              if r['task_profile'] == 'balanced' and r['fit_split'] == 'independent'}
    rng = np.random.default_rng(9131)
    output_rows = []
    for job in plan['jobs']:
        if job['job_id'] == 'width_'+str(anchor_width):
            continue
        path = output/'jobs'/job['job_id']/'readouts_seed_means.csv'
        if not path.exists():
            continue
        differences = {}
        for row in read_rows(path):
            if row['task_profile'] != 'balanced' or row['fit_split'] != 'independent':
                continue
            other = lookup.get(tuple(row[k] for k in (*keys, 'seed')))
            if other is None:
                continue
            key = tuple(row[k] for k in keys)
            for metric in CI_METRICS:
                a, b = row.get(metric, np.nan), other.get(metric, np.nan)
                if np.isfinite(a) and np.isfinite(b):
                    differences.setdefault((key, metric), []).append(a-b)
        for (key, metric), values in sorted(differences.items()):
            boots = rng.choice(values, (job['config']['bootstrap'], len(values))).mean(axis=1) if len(values) > 1 else [np.nan]
            lo, hi = np.quantile(boots, [.025, .975])
            output_rows.append(dict(zip(keys, key)) | dict(job_id=job['job_id'], study=job['study'],
                reference_job='width_'+str(anchor_width), same_task_definition=job['study'] != 'task_basis',
                metric=metric, num_pairs=len(values), candidate_minus_reference=float(np.mean(values)),
                ci95_low=float(lo), ci95_high=float(hi)))
    csv_write(output/'block_paired.csv', output_rows)


def overview_plot(output, plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = []
    anchor = plan['base_config']['compression']
    active = min(plan['base_config']['active_counts'], key=lambda k: abs(k-anchor))
    n = min(plan['base_config']['sample_sizes'], key=lambda n: abs(n-128))
    for job in plan['jobs']:
        if job['study'] != 'width':
            continue
        source = output/'jobs'/job['job_id']/'readouts_mean.csv'
        if not source.exists():
            continue
        selected += [r | dict(width=job['config']['width']) for r in read_rows(source)
            if r['active_count'] == active and r['task_profile'] == 'balanced' and r['sample_size'] == n
            and r['fit_split'] == 'independent' and r['estimator'] == 'mp'
            and r['method'] in ('random_relu', 'learned_relu')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, metric in zip(axes.flat, ('test_mse', 'model_insufficiency', 'mp_aliasing', 'data_insufficiency')):
        for architecture, method in sorted({(r['architecture'], r['method']) for r in selected}):
            rows = sorted([r for r in selected if (r['architecture'], r['method']) == (architecture, method)], key=lambda r: r['width'])
            ax.plot([r['width'] for r in rows], [r[metric] for r in rows], marker='o', label=architecture+' / '+method)
        ax.set(xlabel='hidden width', ylabel=metric, xscale='log')
        ax.set_yscale('symlog', linthresh=1e-6)
        ax.grid(alpha=.2)
        if selected:
            ax.legend(fontsize=7)
    fig.suptitle(f'Exploratory seed means: k={active}, n={n}. Signed cross terms and intervals are in tables.')
    fig.savefig(output/'survey_overview.png', dpi=150)
    plt.close(fig)
