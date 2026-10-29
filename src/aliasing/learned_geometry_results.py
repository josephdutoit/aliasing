"""Seed-level inference; starts and readout draws are not independent seeds."""
from __future__ import annotations

import csv
from collections import defaultdict
import numpy as np

from .benchmark_results import csv_write, read_rows


TAGS = ('experiment', 'scenario', 'seed', 'training', 'compression', 'width', 'n_representation',
        'features', 'coactivation', 'eta', 'rho', 'decay', 'marginal_first', 'marginal_second',
        'task_weight_first', 'task_weight_last', 'fixed_angle', 'method')
METRICS = ('test_mse', 'weighted_test_mse', 'model_insufficiency', 'mp_aliasing', 'data_insufficiency',
           'estimator_correction', 'n_times_aliasing', 'aliasing_to_J_ratio', 'residual_leverage_J',
           'residual_leverage_ratio', 'residual_top_leverage_decile_share', 'encoder_angle_canonical',
           'population_gradient_norm', 'reference_normal_equation_norm')
ARITHMETIC_CROSS = tuple('cross_'+name+'__arithmetic_roundoff' for name in
                        ('model_insufficiency', 'data_insufficiency', 'mp_aliasing', 'estimator_correction'))


def seed_means(rows, table):
    groups = list(TAGS)
    if table == 'readouts':
        groups += ['sample_size', 'estimator', 'fit_split']
    elif table == 'strata':
        groups += ['sample_size', 'stratum']
    elif table == 'references':
        groups += ['sample_size', 'reference_level']
    elif table == 'feature_errors':
        groups += ['sample_size', 'estimator', 'target_feature']
    sums, counts = {}, {}
    for row in rows:
        if table in ('strata', 'feature_errors') and 'test_mse' in row:
            row = dict(row)
            row.setdefault('arithmetic_roundoff', 0.)
        if table == 'readouts':
            # Old accepted rows had no separately measured arithmetic term.
            # A zero here means omitted at the previous identity tolerance;
            # the explicit measured flag distinguishes it from Decimal data.
            row = dict(row)
            for name in ('arithmetic_roundoff', 'arithmetic_roundoff_to_risk', 'arithmetic_roundoff_measured', 'gad_decimal_digits', *ARITHMETIC_CROSS):
                row.setdefault(name, 0.)
            row.setdefault('gad_arithmetic_reliable', 1.)
        key = tuple(row.get(name, '') for name in groups)
        a, b = sums.setdefault(key, {}), counts.setdefault(key, {})
        for name, value in row.items():
            if name in (*groups, 'draw', 'start', 'rotation_id') or not isinstance(value, (int, float)) or not np.isfinite(value):
                continue
            a[name] = a.get(name, 0.)+value
            b[name] = b.get(name, 0)+1
    return [dict(zip(groups, key)) | {name: value/counts[key][name] for name, value in metrics.items()}
            for key, metrics in sums.items()], groups


def summarize(shards, output, bootstrap):
    rng = np.random.default_rng(29107)
    for table in ('geometry', 'readouts', 'strata', 'references', 'feature_errors'):
        rows, groups = seed_means((row for shard in shards for row in read_rows(shard/(table+'.csv'))), table)
        csv_write(output/(table+'_seed_means.csv'), rows)
        group_names = [name for name in groups if name != 'seed']
        grouped = defaultdict(list)
        for row in rows:
            grouped[tuple(row[name] for name in group_names)].append(row)
        means, intervals = [], []
        for key, members in sorted(grouped.items(), key=lambda item: str(item[0])):
            result = dict(zip(group_names, key)) | dict(num_seeds=len(members))
            ci = result.copy()
            names = set().union(*(row.keys() for row in members))-set(groups)
            for name in sorted(names):
                values = np.array([row.get(name, np.nan) for row in members])
                values = values[np.isfinite(values)]
                result[name] = float(values.mean()) if len(values) else float('nan')
                result[name+'_num_seeds'] = len(values)
                if name in METRICS:
                    result[name+'_median'] = float(np.median(values)) if len(values) else float('nan')
                    result[name+'_max'] = float(values.max()) if len(values) else float('nan')
                    result[name+'_std'] = float(values.std(ddof=1)) if len(values) > 1 else float('nan')
                    if name in ('test_mse', 'mp_aliasing') and len(values) and values.sum() > 0:
                        result[name+'_top1_share'] = float(values.max()/values.sum())
                    samples = rng.choice(values, (bootstrap, len(values))).mean(1) if len(values) > 1 else np.array([np.nan])
                    ci[name+'_low'], ci[name+'_high'] = map(float, np.quantile(samples, [.025, .975]))
            means.append(result)
            intervals.append(ci)
        csv_write(output/(table+'_mean.csv'), means)
        csv_write(output/(table+'_ci95.csv'), intervals)
        lookup = {tuple(row[name] for name in groups): row for row in rows}
        contrasts = []
        for key, members in grouped.items():
            tag = dict(zip(group_names, key))
            method = tag['method']
            baseline = 'learned_linear' if method == 'learned_relu' else 'learned_relu'
            if method.startswith('restart_') or method == 'learned_linear':
                continue
            for metric in METRICS:
                differences = []
                for row in members:
                    reference = row | dict(method=baseline)
                    other = lookup.get(tuple(reference[name] for name in groups))
                    a, b = row.get(metric, np.nan), other.get(metric, np.nan) if other else np.nan
                    if np.isfinite(a) and np.isfinite(b):
                        differences.append(a-b)
                if not differences:
                    continue
                boot = rng.choice(differences, (bootstrap, len(differences))).mean(1) if len(differences) > 1 else np.array([np.nan])
                contrasts.append(tag | dict(reference_method=baseline, metric=metric, num_pairs=len(differences),
                    candidate_minus_reference=float(np.mean(differences)),
                    ci95_low=float(np.quantile(boot, .025)), ci95_high=float(np.quantile(boot, .975))))
        csv_write(output/(table+'_paired.csv'), contrasts)
    # Keep optimization starts and gate parameters visible without averaging
    # gate labels across independently permuted units or treating starts as seeds.
    for table in ('starts', 'gates', 'parameters'):
        csv_write(output/(table+'.csv'), (row for shard in shards for row in read_rows(shard/(table+'.csv'))))
    csv_write(output/'readouts_numerical_warnings.csv', (row for shard in shards for row in read_rows(shard/'readouts.csv')
        if row.get('gad_arithmetic_reliable', 1.) == 0))
