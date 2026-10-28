"""Regenerate experiment-16 figures, a descriptive report and compact ZIP."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with Path(path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value) if value else float('nan')
            except ValueError:
                pass
    return rows


def plot_all(output):
    output = Path(output)
    config = json.loads((output/'configuration.json').read_text())['config']
    manifest = json.loads((output/'run_manifest.json').read_text())
    tables = {name: read(output/(name+'_mean.csv')) for name in ('readouts', 'geometry', 'references', 'probes')}
    cis = read(output/'readouts_ci95.csv')
    keys = ('active_count', 'task_profile', 'architecture', 'method', 'sample_size', 'fit_split', 'estimator')
    intervals = {tuple(row[key] for key in keys): row for row in cis}
    profile = config['task_profiles'][0]
    selected_n = min(config['sample_sizes'], key=lambda n: abs(n-128))
    nmax = max(config['sample_sizes'])
    active = min(config['active_counts'], key=lambda k: abs(k-config['compression']))
    architectures = [spec['name'] for spec in config['models']]
    figures = []

    def finish(fig, filename, title):
        fig.suptitle(title, fontsize=11)
        fig.savefig(output/filename, dpi=150)
        plt.close(fig)
        figures.append(filename)

    def curves(ax, selected, metric, x, groups, ci=False):
        for values in sorted({tuple(row[key] for key in groups) for row in selected}):
            rows = sorted([row for row in selected if tuple(row[key] for key in groups) == values], key=lambda row: row[x])
            if len(rows) != len({row[x] for row in rows}):
                raise ValueError('ambiguous figure slice')
            xs, ys = [row[x] for row in rows], [row.get(metric, np.nan) for row in rows]
            line, = ax.plot(xs, ys, marker='o', markersize=3, label=' / '.join(map(str, values)))
            if ci:
                bounds = [intervals[tuple(row[key] for key in keys)] for row in rows]
                ax.fill_between(xs, [row.get(metric+'_ci95_low', np.nan) for row in bounds],
                                [row.get(metric+'_ci95_high', np.nan) for row in bounds], alpha=.1, color=line.get_color())
        ax.set_xlabel(x.replace('_', ' '))
        ax.set_ylabel(metric.replace('_', ' '))
        if x in ('sample_size', 'reference_size'):
            ax.set_xscale('log')
        ax.set_yscale('symlog', linthresh=1e-5)
        ax.set_ylim(bottom=0.)
        ax.grid(alpha=.2)

    independent = [row for row in tables['readouts'] if row['fit_split'] == 'independent' and row['task_profile'] == profile]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, estimator in zip(axes, ('mp', 'ridge_validation')):
        selected = [row for row in independent if row['sample_size'] == selected_n and row['estimator'] == estimator and
                    row['method'] in ('learned_relu', 'learned_linear', 'population_linear_oracle')]
        curves(ax, selected, 'test_mse', 'active_count', ('architecture', 'method'), True)
        ax.set_title(estimator+' readouts')
    axes[0].legend(fontsize=6)
    finish(fig, 'architecture_comparison.png', f'Input complexity at independent readout n={selected_n}; pointwise seed bootstrap intervals')

    fig, axes = plt.subplots(len(architectures), 2, figsize=(13, 4*len(architectures)), squeeze=False, constrained_layout=True)
    for i, architecture in enumerate(architectures):
        selected = [row for row in independent if row['architecture'] == architecture and row['active_count'] == active and
                    row['method'] in ('learned_relu', 'random_relu', 'learned_linear')]
        curves(axes[i, 0], selected, 'test_mse', 'sample_size', ('method', 'estimator'), True)
        axes[i, 0].set_title(architecture+': prediction error')
        mp = [row for row in selected if row['estimator'] == 'mp']
        for metric in ('model_insufficiency', 'mp_aliasing', 'data_insufficiency'):
            rows = [row for row in mp if row['method'] == 'learned_relu']
            axes[i, 1].plot([row['sample_size'] for row in rows], [row[metric] for row in rows], marker='o', markersize=3, label=metric)
        axes[i, 1].set(xscale='log', yscale='symlog', xlabel='sample size', title=architecture+': MP term energies')
        axes[i, 1].set_yscale('symlog', linthresh=1e-5)
        axes[i, 1].set_ylim(bottom=0.)
        for ax in axes[i]:
            ax.grid(alpha=.2)
            ax.legend(fontsize=6)
    finish(fig, 'readout_and_gad.png', f'Frozen maps at k={active}; term energies are not an additive risk partition (cross terms retained in tables)')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    selected = [row for row in independent if row['active_count'] == active and row['estimator'] == 'mp' and
                row['method'] in ('learned_relu', 'rotated_relu', 'rotated_relu_rate_matched', 'no_final_gate')]
    selected = [row for row in selected if row['sample_size'] == selected_n]
    for architecture in architectures:
        rows = [row for row in selected if row['architecture'] == architecture]
        methods = [row['method'] for row in rows]
        axes[0].plot(methods, [row['test_mse'] for row in rows], marker='o', label=architecture)
    probes = [row for row in tables['probes'] if row['active_count'] == active and row['task_profile'] == profile and
              row['sample_size'] == selected_n and row['probe_kind'] == 'exact_readout_operator' and
              row['method'] in ('learned_relu', 'rotated_relu', 'rotated_relu_rate_matched', 'no_final_gate')]
    for architecture in architectures:
        rows = [row for row in probes if row['architecture'] == architecture]
        axes[1].plot([row['method'] for row in rows], [row['isotropic_prediction_energy'] for row in rows], marker='o', label=architecture)
    axes[0].set_title('Task error after independently refitting heads')
    axes[1].set_title('Exact unit-variance isotropic perturbation energy')
    for ax in axes:
        ax.set_yscale('symlog', linthresh=1e-5)
        ax.set_ylim(bottom=0.)
        ax.tick_params(axis='x', labelrotation=25)
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    finish(fig, 'gate_interventions.png', f'Gate interventions at k={active}, n={selected_n}; pure rotations preserve preactivation Gram matrices')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    references = [row for row in tables['references'] if row['task_profile'] == profile and row['active_count'] == active and
                  row['sample_size'] == nmax and row['method'] == 'learned_relu']
    for ax, metric in zip(axes, ('reference_prediction_difference_mse', 'alias_vector_difference_mse')):
        curves(ax, references, metric, 'reference_size', ('architecture',))
    axes[0].legend(fontsize=7)
    finish(fig, 'reference_convergence.png', f'Nested independent reference fits at k={active}; largest reference is an estimated comparator')

    lines = ['# Representation model benchmark', '',
        f'Completed {manifest["completed"]}/{manifest["expected"]} worlds; failures: {len(manifest["failures"])}.',
        'Status: '+('COMPLETE' if manifest['complete'] else 'PARTIAL: conclusions require checking missing worlds.'), '',
        f'Features p={config["features"]}; input dimension={config["input_dim"]}; representation-training examples={config["n_representation"]}; updates={config["steps"]}.', '',
        '## Model specifications', '', '```json', json.dumps(config['models'], indent=2), '```', '',
        '## Independent readouts', '',
        f'Readout n={selected_n}, profile={profile}. Means and medians are over seed-level averages of repeated draws.', '',
        '| k | Architecture | Map | Head | Mean test MSE | Median test MSE | Model term | MP alias term |',
        '| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |']
    for row in sorted(independent, key=lambda r: (r['active_count'], r['architecture'], r['method'], r['estimator'])):
        if row['sample_size'] == selected_n and row['method'] in ('learned_relu', 'learned_linear', 'population_linear_oracle'):
            lines.append(f"| {row['active_count']:g} | {row['architecture']} | {row['method']} | {row['estimator']} | "+
                ' | '.join(f"{row.get(key, np.nan):.6g}" for key in ('test_mse', 'test_mse_median', 'model_insufficiency', 'mp_aliasing'))+' |')
    lines += ['', 'MP aliasing remains the Moore–Penrose baseline for ridge and native heads. Their changed predictions include an explicit estimator correction and all signed cross terms.', '',
        '## Representation quality', '', 'Largest independent reference fit; this estimates population projection quality.', '',
        '| k | Architecture | Reference model MSE | Readout rank | Output-subspace lower bound | Features with normalized MSE < 0.1 |',
        '| ---: | --- | ---: | ---: | ---: | ---: |']
    for row in sorted(tables['geometry'], key=lambda r: (r['active_count'], r['architecture'])):
        if row['task_profile'] == profile and row['method'] == 'learned_relu':
            lines.append(f"| {row['active_count']:g} | {row['architecture']} | "+' | '.join(
                f"{row[key]:.6g}" for key in ('reference_test_mse', 'reference_rank', 'output_subspace_lower_bound', 'latent_features_below_0p1'))+' |')
    lines += ['', '## Native heads and larger independent readouts', '',
        f'Native and representation-training MP heads use the same {config["n_representation"]} examples. The final column uses a separate n={nmax} readout.', '',
        '| k | Architecture | Native test MSE | MP on representation-training data | Independent MP test MSE |',
        '| ---: | --- | ---: | ---: | ---: |']
    for k in config['active_counts']:
        for architecture in architectures:
            rows = [r for r in tables['readouts'] if r['active_count'] == k and r['task_profile'] == profile and
                    r['architecture'] == architecture and r['method'] == 'learned_relu']
            native = next((r for r in rows if r['fit_split'] == 'representation_training' and r['estimator'] == 'native'), None)
            fitted = next((r for r in rows if r['fit_split'] == 'representation_training' and r['estimator'] == 'mp'), None)
            independent_row = next((r for r in rows if r['fit_split'] == 'independent' and r['estimator'] == 'mp' and r['sample_size'] == nmax), None)
            if native and fitted and independent_row:
                lines.append(f"| {k} | {architecture} | {native['test_mse']:.6g} | {fitted['test_mse']:.6g} | {independent_row['test_mse']:.6g} |")
    lines += ['', '## Reading the evidence', '',
        '- Compare learned ReLU with frozen random ReLU to test representation learning, and with learned linear at the same architecture to test nonlinear benefit.',
        '- The unrestricted wide linear model can solve these linear target tasks. Its population floor is zero; beating a compressed linear model alone does not establish an advantage over this control.',
        '- A narrow final hidden layer has an output-subspace floor even with nonlinear features. The bottleneck model expands before its final readout.',
        '- Use readouts_paired.csv and probes_paired.csv for within-seed contrasts. Intervals are pointwise and exploratory; inspect seed tables, maxima and top-five shares when means are unstable.',
        '- Compare independent readouts at matching sample sizes. Native heads reuse representation-training data and have an explicit estimator correction.',
        '- Reference convergence is empirical. The finite reference model term is an estimate, and its cross terms need not vanish.',
        '- Sensitivities measure the specified MP readout operator or perturbation. Lower model error does not imply uniformly lower aliasing or noise sensitivity.',
        '- Pure gate rotations preserve preactivation geometry. Activity-matched rotations additionally change biases; inspect activity_match_error for ties.',
        '- A deeper custom MLP removes only its final gate in the no_final_gate intervention. The separately trained learned_linear arm removes every gate.',
        '- These tests do not establish a Marchenko–Pastur filter, a heavy-tail mechanism, or preferential decision-boundary protection.', '',
        '## Controls', '', '```json', json.dumps(manifest['control_maxima'], indent=2), '```', '', '## Figures', '']
    for filename in figures:
        lines += [f'![{filename}]({filename})', '']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    bundle = output/'results_bundle.zip'
    temporary = output/'results_bundle.zip.tmp'
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.suffix in ('.csv', '.json', '.md', '.png'):
                archive.write(path, path.name)
    temporary.replace(bundle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    plot_all(parser.parse_args().output_dir)
