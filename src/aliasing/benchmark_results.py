"""Atomic experiment files, seed-level summaries and paired comparisons."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .representation_benchmark import TABLE_GROUPS


CI_METRICS = ('test_mse', 'weighted_test_mse', 'model_insufficiency', 'mp_aliasing', 'data_insufficiency',
              'estimator_correction', 'mp_residual_gain', 'reference_test_mse', 'latent_mean_normalized_mse',
              'latent_worst_normalized_mse', 'latent_features_below_0p1', 'normalized_mse',
              'isotropic_prediction_energy', 'worst_training_rms_gain', 'rms_gain',
              'reference_prediction_difference_mse', 'alias_vector_difference_mse')


def json_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def csv_write(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ['seed']
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_rows(path):
    with Path(path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            for key, value in row.items():
                try:
                    row[key] = float(value) if value else float('nan')
                except ValueError:
                    pass
            yield row


def summarize(shards, output, bootstrap):
    """A seed is the resampling unit, never a draw, rotation or probe direction."""
    output = Path(output)
    rng = np.random.default_rng(16027)
    for table, groups in TABLE_GROUPS.items():
        collapsed = {}
        for shard in shards:
            for row in read_rows(shard/(table+'.csv')):
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
                values = np.array([row.get(metric, np.nan) for row in rows])
                values = values[np.isfinite(values)]
                mean[metric] = float(values.mean()) if len(values) else float('nan')
                mean[metric+'_num_seeds'] = len(values)
                if metric in ('test_mse', 'mp_aliasing') and len(values):
                    mean[metric+'_median'] = float(np.median(values))
                    mean[metric+'_max'] = float(values.max())
                    total = values.sum()
                    mean[metric+'_top5_share'] = float(np.sort(values)[-5:].sum()/total) if total > 0 else float('nan')
                if metric in CI_METRICS:
                    boots = rng.choice(values, (bootstrap, len(values))).mean(axis=1) if len(values) > 1 else np.array([np.nan])
                    ci[metric+'_ci95_low'], ci[metric+'_ci95_high'] = map(float, np.quantile(boots, [.025, .975]))
                    ci[metric+'_num_seeds'] = len(values)
            means.append(mean)
            cis.append(ci)
        csv_write(output/(table+'_mean.csv'), means)
        csv_write(output/(table+'_ci95.csv'), cis)
        if table in ('history', 'feature_recovery'):
            continue
        lookup = {tuple(row[name] for name in groups)+(row['seed'],): row for row in seed_rows}
        paired = []
        for key, rows in sorted(group_rows.items()):
            tag = dict(zip(groups, key))
            references = []
            if tag['method'] == 'learned_relu':
                references += [tag | dict(method='random_relu'), tag | dict(method='learned_linear')]
                if tag['architecture'] != 'bottleneck':
                    references.append(tag | dict(architecture='bottleneck'))
            elif tag['method'] in ('rotated_relu', 'rotated_relu_rate_matched', 'no_final_gate'):
                references.append(tag | dict(method='learned_relu'))
            if table == 'readouts' and tag['estimator'] == 'ridge_validation':
                references.append(tag | dict(estimator='mp'))
            for reference in references:
                for metric in CI_METRICS:
                    differences = []
                    for row in rows:
                        other = lookup.get(tuple(reference[name] for name in groups)+(row['seed'],))
                        a, b = row.get(metric, np.nan), other.get(metric, np.nan) if other else np.nan
                        if np.isfinite(a) and np.isfinite(b):
                            differences.append(a-b)
                    if not differences:
                        continue
                    boots = rng.choice(differences, (bootstrap, len(differences))).mean(axis=1) if len(differences) > 1 else np.array([np.nan])
                    paired.append(tag | dict(reference_architecture=reference['architecture'], reference_method=reference['method'],
                        reference_estimator=reference.get('estimator', ''), metric=metric, num_pairs=len(differences),
                        candidate_minus_reference=float(np.mean(differences)),
                        ci95_low=float(np.quantile(boots, .025)), ci95_high=float(np.quantile(boots, .975))))
        csv_write(output/(table+'_paired.csv'), paired)
