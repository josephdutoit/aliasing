"""Regenerate experiment-15 figures and a compact review bundle; no training."""
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


LABELS = {'learned_relu': 'Original learned ReLU', 'learned_linear': 'Learned linear',
    'linear_oracle': 'Linear oracle', 'random_relu': 'Fixed random ReLU',
    'joint_fresh': 'Joint training, fresh start', 'fixed_random_encoder': 'Fixed random encoder, learned decoder',
    'frozen_learned_encoder': 'Pretrained encoder, fresh learned decoder',
    'rotated_relu': 'Rotated gates', 'rotated_relu_rate_matched': 'Rotated gates, matched activity',
    'no_gate': 'Gate removed'}


def read(path):
    with path.open() as handle:
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
    configuration = json.loads((output/'configuration.json').read_text())
    config = configuration['config']; original = configuration['source_configuration']['config']
    manifest = json.loads((output/'run_manifest.json').read_text())
    profile = config['task_profiles'][0]
    active = min(config['active_counts'], key=lambda x: abs(x-original['compression']))
    n_selected = min(config['sample_sizes'], key=lambda x: abs(x-128))
    n_max = max(config['sample_sizes'])
    tables = {name: read(output/(name+'_mean.csv')) for name in ('readouts','references','probes')}
    cis = {name: read(output/(name+'_ci95.csv')) for name in tables}
    keys = {
        'readouts': ('active_count','task_profile','method','sample_size','fit_split','estimator'),
        'references': ('active_count','task_profile','method','sample_size','reference_size'),
        'probes': ('active_count','task_profile','method','sample_size','probe_kind'),
    }
    lookups = {name: {tuple(row[key] for key in keys[name]): row for row in rows} for name,rows in cis.items()}
    figures = []

    def curves(ax, table, selected, metric, x, groups, title):
        for values in sorted({tuple(row[key] for key in groups) for row in selected}):
            rows = sorted([row for row in selected if tuple(row[key] for key in groups) == values], key=lambda row: row[x])
            if len({row[x] for row in rows}) != len(rows):
                raise ValueError('ambiguous figure slice')
            label = ', '.join(LABELS.get(str(value), str(value)) for value in values)
            line, = ax.plot([row[x] for row in rows], [row.get(metric, np.nan) for row in rows], marker='o', label=label)
            intervals = [lookups[table][tuple(row[key] for key in keys[table])] for row in rows]
            ax.fill_between([row[x] for row in rows], [row.get(metric+'_ci95_low', np.nan) for row in intervals],
                            [row.get(metric+'_ci95_high', np.nan) for row in intervals], color=line.get_color(), alpha=.13)
        ax.set(title=title, xlabel=x.replace('_',' '))
        values = [row.get(metric,np.nan) for row in selected]
        values = [value for value in values if np.isfinite(value)]
        if values and min(values)>0:
            ax.set_yscale('log' if max(values)/min(values)>50 else 'linear')
        else:
            ax.set_yscale('symlog',linthresh=1e-9)
        if x in ('sample_size','reference_size'):
            ax.set_xscale('log')
        ax.grid(alpha=.25)

    def finish(fig, filename, title):
        fig.suptitle(title+'\nPointwise 95% bootstrap intervals over seeds; repeated draws/rotations/probes averaged within seed')
        fig.savefig(output/filename, dpi=150); plt.close(fig); figures.append(filename)

    rows = tables['readouts']
    selected = [r for r in rows if r['task_profile']==profile and r['active_count']==active and r['fit_split']=='independent'
                and r['method'] in ('learned_relu','learned_linear','linear_oracle')]
    fig,axes = plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    curves(axes[0], 'readouts', selected, 'test_mse','sample_size',('method','estimator'),'Held-out error: MP versus validation-selected ridge')
    mp = [r for r in selected if r['estimator']=='mp']
    curves(axes[1], 'readouts', mp, 'mp_aliasing','sample_size',('method',),'Canonical GAD aliasing of MP heads')
    for r in rows:
        if r['task_profile']==profile and r['active_count']==active and r['fit_split']=='representation_training' and r['estimator']=='native' and r['method'] in ('learned_relu','learned_linear'):
            axes[0].scatter([r['sample_size']],[r['test_mse']],marker='*',s=130,label=LABELS[r['method']]+', native head')
    for ax in axes: ax.legend(fontsize=6)
    finish(fig,'readout_stability.png',f'Frozen maps; active count {active}, profile {profile}. Native heads saw representation-training data')

    selected = [r for r in tables['references'] if r['task_profile']==profile and r['method']=='learned_relu'
                and r['sample_size']==max(config['reference_readout_sizes'])]
    fig,axes = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    curves(axes[0],'references',selected,'reference_prediction_difference_mse','reference_size',('active_count',),'Reference prediction difference from largest reference')
    curves(axes[1],'references',selected,'alias_vector_difference_mse','reference_size',('active_count',),'GAD alias-vector difference from largest reference')
    axes[0].legend(title='Active features',fontsize=7)
    finish(fig,'reference_convergence.png','Nested independent reference fits; largest reference is an estimated comparator, not population truth')

    methods = ('joint_fresh','fixed_random_encoder','frozen_learned_encoder')
    fig,axes = plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    selected = [r for r in rows if r['task_profile']==profile and r['method'] in methods and r['sample_size']==n_max and r['fit_split']=='independent' and r['estimator']=='mp']
    curves(axes[0],'readouts',selected,'test_mse','active_count',('method',),f'Independent MP readout, n={n_max}')
    selected = [r for r in rows if r['task_profile']==profile and r['method'] in methods and r['fit_split']=='representation_training' and r['estimator']=='native']
    curves(axes[1],'readouts',selected,'test_mse','active_count',('method',),'Native trained head: test MSE')
    axes[0].legend(fontsize=6)
    finish(fig,'encoder_decoder_ablation.png','Joint and fixed-random arms have identical starts/budgets; pretrained-encoder arm has extra pretraining')

    fig,axes = plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for ax,kind,metric,title in zip(axes.flat,
        ('exact_readout_operator','shared_gaussian','shared_quadratic','reference_residualized_quadratic'),
        ('isotropic_prediction_energy','rms_gain','rms_gain','rms_gain'),
        ('Exact unit-variance isotropic perturbation energy','Shared Gaussian perturbation: realized RMS gain',
         'Shared nonlinear-function perturbation: RMS gain','Reference-residualized nonlinear functions: RMS gain')):
        selected = [r for r in tables['probes'] if r['task_profile']==profile and r['active_count']==active and r['probe_kind']==kind
                    and r['method'] in ('learned_relu','rotated_relu','rotated_relu_rate_matched','no_gate')]
        curves(ax,'probes',selected,metric,'sample_size',('method',),title)
    axes[0,0].legend(fontsize=6)
    finish(fig,'controlled_perturbations.png',f'Fixed maps, active count {active}; readout sensitivity and specified residual probes are separate measurements')

    lines = ['# GAD mechanism checks: results','',
        f'Completed {manifest["completed"]}/{manifest["expected"]} configurations; failures: {len(manifest["failures"])}.',
        f'Compression {original["compression"]}, expansion {original["expansion"]}; profile {profile}.', '',
        '## Experiment 1: readout and reference stability','',
        f'Independent readout n={n_selected}. Model terms use the largest independent reference fit.', '',
        '| Active features | Map | Estimator | Test MSE | MP model term | MP alias term | Estimator correction energy |',
        '| ---: | --- | --- | ---: | ---: | ---: | ---: |']
    for r in sorted(rows,key=lambda r:(r['active_count'],r['method'],r['estimator'],r['sample_size'])):
        if r['task_profile']==profile and r['sample_size']==n_selected and r['fit_split']=='independent' and r['method'] in ('learned_relu','learned_linear','linear_oracle'):
            lines.append(f"| {r['active_count']:g} | {r['method']} | {r['estimator']} | "+' | '.join(f"{r[x]:.6g}" for x in ('test_mse','model_insufficiency','mp_aliasing','estimator_correction'))+' |')
    lines += ['', 'The MP alias term remains the canonical MP baseline even for ridge rows. Ridge changes prediction through the estimator correction AND signed cross terms; do not interpret the baseline alias term as ridge leakage.', '',
        '## Experiment 2: encoder/decoder identification','',
        '| Active features | Arm | Native test MSE | Independent MP test MSE at largest n |','| ---: | --- | ---: | ---: |']
    for k in config['active_counts']:
        for method in methods:
            native = next((r for r in rows if r['active_count']==k and r['task_profile']==profile and r['method']==method and r['fit_split']=='representation_training' and r['estimator']=='native'),None)
            fitted = next((r for r in rows if r['active_count']==k and r['task_profile']==profile and r['method']==method and r['fit_split']=='independent' and r['estimator']=='mp' and r['sample_size']==n_max),None)
            if native and fitted:
                lines.append(f"| {k} | {method} | {native['test_mse']:.6g} | {fitted['test_mse']:.6g} |")
    lines += ['', 'Joint-fresh versus fixed-random-encoder is the matched-start/matched-update comparison. The frozen-learned-encoder arm has additional pretraining and is a secondary diagnostic.', '',
        '## Experiment 3: controlled perturbations','',
        'Shared Gaussian and quadratic probes impose the same perturbations across maps. Reference-residualized quadratic probes instead change residual direction with each map. Exact operator sensitivities refer to K=H_P H_T^+, not the full omitted-feature GAD operator.', '',
        'See probes_paired.csv for paired seed contrasts in gain, isotropic prediction energy, and worst training-RMS gain. No automatic confirmation is inferred.', '',
        '## Controls','', '```json', json.dumps(manifest['control_maxima'],indent=2), '```', '',
        '## Interpretation limits','',
        '- Reference convergence is empirical; the largest finite fit is not exact population truth.',
        '- Representation-training fits reuse representation-training data; independent fits use separate samples.',
        '- Original native heads and 128-sample probes have different data budgets; compare matching sizes as well.',
        '- The fresh joint/frozen-random arms have matched updates and starts. The frozen learned encoder has extra pretraining.',
        '- Perturbation directions, rotations, and readout draws are averaged within each representation seed.',
        '- Intervals are pointwise exploratory intervals; unstable means should be read alongside seed_means tables.',
        '- These checks do not establish Marchenko-Pastur calibration, a universal spectral filter, or boundary-specific protection.', '', '## Figures','']
    for filename in figures: lines += [f'![{filename}]({filename})','']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    # Include reviewable seed tables and configuration; omit checkpoint shards.
    paths = [p for p in output.iterdir() if p.is_file() and p.suffix in ('.csv','.json','.md','.png')]
    with ZipFile(output/'results_bundle.zip','w',ZIP_DEFLATED) as archive:
        for path in sorted(paths): archive.write(path,path.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    plot_all(parser.parse_args().output_dir)
