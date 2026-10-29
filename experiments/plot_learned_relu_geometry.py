"""Replot the new suite without training; reads only compact summary tables."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from aliasing.benchmark_results import read_rows


def plot(output):
    output = Path(output)
    geometry = list(read_rows(output/'geometry_mean.csv'))
    readouts = list(read_rows(output/'readouts_mean.csv'))
    if not geometry:
        return
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    experiments = sorted(set(row['experiment'] for row in geometry))
    fig, axes = plt.subplots(len(experiments), 3, figsize=(15, 3.1*len(experiments)), squeeze=False)
    for i, experiment in enumerate(experiments):
        rows = [r for r in geometry if r['experiment'] == experiment and r['method'] in ('learned_relu', 'learned_linear')]
        if experiment == 'landscape':
            xkey, xlabel = 'fixed_angle', 'Fixed encoder angle (radians); decoder learned'
        elif experiment == 'finite':
            xkey, xlabel = 'n_representation', 'Representation-training examples'
        elif experiment == 'higher_order':
            xkey, xlabel = 'eta', 'Higher-order coactivation mixture'
        else:
            xkey, xlabel = 'coactivation', 'Pair coactivation fraction'
        # Keep each architectural/objective/distribution setting separate.
        keys = ['method', 'width', 'compression', 'features', 'decay', 'marginal_first', 'marginal_second',
                'task_weight_first', 'task_weight_last', 'rho']
        if xkey != 'coactivation':
            keys.append('coactivation')
        grouped = defaultdict(list)
        for row in rows:
            grouped[tuple(row.get(k, '') for k in keys)].append(row)
        for key, values in grouped.items():
            values.sort(key=lambda r: r[xkey])
            label = f'{key[0]}, p={key[3]:g}, m={key[2]:g}, w={key[1]:g}, decay={key[4]:g}'
            if experiment == 'tasks':
                label += f', weights={key[7]:g}:{key[8]:g}, pi={key[5]:g}:{key[6]:g}'
            if experiment in ('finite', 'landscape'):
                label += f', q={values[0]["coactivation"]:g}'
            if experiment == 'higher_order':
                label += f', rho={values[0]["rho"]:g}'
            for ax, metric in zip(axes[i], ['model_insufficiency', 'encoder_angle_canonical', 'residual_leverage_J']):
                y = [row.get(metric, np.nan) for row in values]
                if np.isfinite(y).any():
                    ax.plot([row[xkey] for row in values], y, '.-', alpha=.65, label=label)
                ax.set(xlabel=xlabel, ylabel=metric.replace('_', ' '), title=experiment)
                ax.grid(alpha=.2)
                if xkey == 'n_representation':
                    ax.set_xscale('log')
        # Full labels remain in CSV; a huge phase grid is clearer without a
        # hundred-line legend obscuring the measurements.
        if len(grouped) <= 8:
            axes[i, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output/'learned_geometry.png', dpi=150)
    plt.close(fig)
    # Per-study representative scenarios are EXPLICITLY labeled; no minima or
    # winning runs are selected using evaluation risk.
    fig, axes = plt.subplots(len(experiments), 3, figsize=(15, 3.1*len(experiments)), squeeze=False)
    for i, experiment in enumerate(experiments):
        candidates = sorted(set(r['scenario'] for r in readouts if r['experiment'] == experiment))
        if not candidates:
            continue
        selected = candidates[len(candidates)//2]
        rows = [r for r in readouts if r['experiment'] == experiment and r['scenario'] == selected and
                r.get('fit_split') == 'independent' and r.get('estimator') == 'mp' and not r['method'].startswith('restart_')]
        for method in sorted(set(r['method'] for r in rows)):
            values = sorted([r for r in rows if r['method'] == method], key=lambda r: r['sample_size'])
            for ax, metric in zip(axes[i], ['test_mse', 'mp_aliasing', 'n_times_aliasing']):
                ax.plot([r['sample_size'] for r in values], [r.get(metric, np.nan) for r in values], '.-', label=method)
                ax.set(xscale='log', yscale='symlog', xlabel='Independent readout samples', ylabel=metric.replace('_', ' '),
                       title=f'{experiment}: scenario {selected}')
                ax.grid(alpha=.2)
        axes[i, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output/'finite_sample_gad.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method, marker in [('learned_relu', 'o'), ('learned_linear', 'x')]:
        rows = [r for r in geometry if r['method'] == method]
        axes[0].scatter([r['model_insufficiency'] for r in rows], [r['residual_leverage_J'] for r in rows], s=15, alpha=.4, marker=marker, label=method)
        axes[1].scatter([r['residual_leverage_ratio'] for r in rows], [r['residual_top_leverage_decile_share'] for r in rows], s=15, alpha=.4, marker=marker, label=method)
    axes[0].set(xlabel='Population/reference model energy', ylabel='Residual–leverage J', xscale='symlog', yscale='symlog')
    axes[1].set(xlabel='J / (rank × model energy)', ylabel='Residual energy share in top leverage decile')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output/'residual_alignment.png', dpi=150)
    plt.close(fig)
    # Show actual learned ReLU boundaries in the physical two-feature domain.
    gates = list(read_rows(output/'gates.csv'))
    phase = [r for r in geometry if r['experiment'] == 'phase' and r['method'] == 'learned_relu' and r['width'] == 2 and np.isclose(r['decay'], 1e-5)]
    if phase:
        phase.sort(key=lambda r: r['coactivation'])
        selected = [phase[i] for i in sorted(set([0, len(phase)//2, len(phase)-1]))]
        fig, axes = plt.subplots(1, len(selected), figsize=(5*len(selected), 4.5), squeeze=False)
        for ax, row in zip(axes[0], selected):
            a, b = row['marginal_first'], row['marginal_second']
            lower, upper = max(0., a+b-1), min(a, b)
            joint = lower+row['coactivation']*(upper-lower)
            ax.plot([.5, 1.5, 1.5, .5, .5], [.5, .5, 1.5, 1.5, .5], color='.7',
                    linestyle='-' if joint > 0 else ':', label=f'Both active: P={joint:.3f}')
            ax.plot([.5, 1.5], [0, 0], lw=4 if a-joint > 0 else 1, color='.7', linestyle='-' if a-joint > 0 else ':')
            ax.plot([0, 0], [.5, 1.5], lw=4 if b-joint > 0 else 1, color='.7', linestyle='-' if b-joint > 0 else ':')
            ax.scatter([0], [0], color='.5', s=30)
            units = [g for g in gates if g['scenario'] == row['scenario'] and g['method'] == 'learned_relu' and g['seed'] == 0]
            coordinate = np.linspace(-.1, 1.6, 200)
            for unit in units:
                a, b, offset = unit['normal_0'], unit['normal_1'], unit['signed_offset']
                if not np.isfinite([a, b, offset]).all():
                    continue
                label = f'Gate {unit["gate"]:g}, active P={unit["active_probability"]:.2f}'
                if abs(b) > 1e-10:
                    ax.plot(coordinate, (offset-a*coordinate)/b, label=label)
                elif abs(a) > 1e-10:
                    ax.axvline(offset/a, label=label)
            ax.set(xlim=(-.1, 1.6), ylim=(-.1, 1.6), aspect='equal', xlabel='Latent feature 1', ylabel='Latent feature 2',
                   title=f'Learned boundaries, seed 0; q fraction={row["coactivation"]:g}')
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output/'gate_boundaries.png', dpi=150)
        plt.close(fig)
    seeds = list(read_rows(output/'geometry_seed_means.csv'))
    restarts = [r for r in seeds if r['method'].startswith('restart_') and r['features'] == 2]
    if restarts:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        color = [r['coactivation'] for r in restarts]
        scatter = axes[0].scatter([r['encoder_angle_canonical'] for r in restarts], [r['model_insufficiency'] for r in restarts], c=color, s=12, alpha=.5)
        axes[1].scatter([r['model_insufficiency'] for r in restarts], [r['residual_leverage_J'] for r in restarts], c=color, s=12, alpha=.5)
        axes[0].set(xlabel='Encoder angle modulo global sign', ylabel='Model energy', title='All trained starts; not independent replicates')
        axes[1].set(xlabel='Model energy', ylabel='Residual–leverage J')
        fig.colorbar(scatter, ax=axes.ravel().tolist(), label='Pair coactivation fraction')
        fig.subplots_adjust(right=.83, wspace=.3)
        fig.savefig(output/'optimization_basins.png', dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    plot(parser.parse_args().output_dir)
