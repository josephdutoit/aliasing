"""Regenerate hypothesis figures from experiment 14 summaries; no training."""
from pathlib import Path
import argparse
import csv
import json
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


KEYS=('active_count','task_profile','method','threshold_offset','sample_size','region','task_group')
LABELS={'random_linear':'Fixed linear','learned_linear':'Learned linear',
        'linear_oracle':'Population linear oracle','random_relu':'Fixed ReLU','learned_relu':'Learned ReLU',
        'rotated_relu':'Rotated ReLU','rotated_relu_rate_matched':'Rotated ReLU, matched activity',
        'no_gate':'Learned weights, gate removed','linear_rotation_control':'Rotated linear control'}


def read(path):
    with path.open() as handle:rows=list(csv.DictReader(handle))
    for row in rows:
        for k,v in row.items():
            try:row[k]=float(v) if v else float('nan')
            except ValueError:pass
    return rows


def plot_all(output):
    output=Path(output)
    rows=read(output/'mean.csv');cis=read(output/'ci95.csv')
    if not rows:raise ValueError('empty summaries')
    lookup={tuple(r[k] for k in KEYS):r for r in cis}
    config=json.loads((output/'configuration.json').read_text())['config']
    profile=config['task_profiles'][0]
    n=max(config['sample_sizes'])
    k=min(config['active_counts'],key=lambda k:abs(k-config['compression']))
    figures=[]

    def panel(ax,selected,metric,x,groups,title):
        for values in sorted({tuple(r[g] for g in groups) for r in selected}):
            curve=sorted((r for r in selected if tuple(r[g] for g in groups)==values),key=lambda r:r[x])
            assert len({r[x] for r in curve})==len(curve),'ambiguous plot slice'
            label=', '.join(LABELS.get(str(v),str(v)) for v in values) or 'mean'
            xx=[r[x] for r in curve];yy=[r[metric] for r in curve]
            line,=ax.plot(xx,yy,marker='o',label=label)
            lower=[lookup[tuple(r[q] for q in KEYS)][metric+'_ci95_low'] for r in curve]
            upper=[lookup[tuple(r[q] for q in KEYS)][metric+'_ci95_high'] for r in curve]
            ax.fill_between(xx,lower,upper,alpha=.13,color=line.get_color())
        values=[r[metric] for r in selected if np.isfinite(r[metric])]
        if values and max(abs(v) for v in values)<1e-12 and 'error' not in metric:
            ax.set_ylim(0,1e-12);title+=' (numerically zero)'
        elif metric not in ('classification_error','activation_rate'):
            ax.set_yscale('symlog',linthresh=1e-5)
        ax.set(title=title,xlabel=x.replace('_',' '));ax.grid(alpha=.25)

    def finish(fig,name,title):
        fig.suptitle(title+'\nPointwise CIs over representation seeds; draws/rotations averaged within seed')
        fig.savefig(output/name,dpi=150);plt.close(fig);figures.append(name)

    base=[r for r in rows if r['sample_size']==n and r['task_profile']==profile and r['region']=='all'
          and r['task_group']=='all' and r['method'] in ('random_linear','learned_linear','linear_oracle','random_relu','learned_relu')]
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,metric,title in zip(axes.flat,('test_mse','model_insufficiency','aliasing','data_insufficiency'),
                              ('Task-score test MSE','Model insufficiency','Generalized aliasing','Data insufficiency')):
        panel(ax,base,metric,'active_count',('method',),title)
    axes[0,0].legend(fontsize=7)
    finish(fig,'sparsity_gad.png',f'Fixed compression {config["compression"]}, expansion {config["expansion"]}; task profile {profile}, n={n}')

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,metric,title in zip(axes,('residual_train_mse','alias_residual_gain'),('Unmodeled training residual energy','Realized residual amplification (RMS ratio)')):
        panel(ax,base,metric,'active_count',('method',),title)
    axes[0].legend(fontsize=7)
    finish(fig,'residual_amplification.png',f'Distinguishing smaller residuals from less amplification; profile {profile}, n={n}')

    rotation=[r for r in rows if r['active_count']==k and r['task_profile']==profile and r['region']=='all' and r['task_group']=='all'
              and r['method'] in ('learned_relu','rotated_relu','rotated_relu_rate_matched','no_gate','linear_rotation_control')]
    if any(r['method']=='rotated_relu' for r in rotation):
        fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        for ax,metric,title in zip(axes.flat,('test_mse','model_insufficiency','aliasing','alias_residual_gain'),
                                  ('Test MSE','Model insufficiency','Generalized aliasing','Realized residual amplification')):
            panel(ax,rotation,metric,'sample_size',('method',),title)
        axes[0,0].legend(fontsize=6)
        finish(fig,'rotation_intervention.png',f'Gate-coordinate interventions at active count {k}, profile {profile}; heads refitted')

    threshold=[r for r in rows if r['active_count']==k and r['task_profile']==profile and r['sample_size']==n
               and r['region']=='all' and r['task_group']=='all' and r['method']=='threshold_relu']
    if threshold:
        fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
        for ax,metric,title in zip(axes.flat,('test_mse','model_insufficiency','aliasing','data_insufficiency','alias_residual_gain','activation_rate'),
                                  ('Test MSE','Model insufficiency','Generalized aliasing','Data insufficiency','Residual amplification','Activation rate')):
            panel(ax,threshold,metric,'threshold_offset',(),title)
            ax.axvline(0,color='gray',linestyle=':')
        finish(fig,'threshold_intervention.png',f'Bias shift in preactivation-standard-deviation units; active count {k}, n={n}; zero=learned setting')

    task=[r for r in rows if r['sample_size']==n and r['region']=='all' and r['method']=='learned_relu'
          and r['task_group'] in ('first_half','second_half') and r['task_profile'] in ('A','B')]
    if {r['task_profile'] for r in task}=={'A','B'}:
        fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        for i,group in enumerate(('first_half','second_half')):
            selected=[r for r in task if r['task_group']==group]
            for j,metric in enumerate(('test_mse','aliasing')):
                panel(axes[i,j],selected,metric,'active_count',('task_profile',),group+': '+metric)
        axes[0,0].legend(title='Representation training priority')
        finish(fig,'task_priority_swap.png','Same inputs and task definitions; A emphasizes first-half tasks, B emphasizes second-half tasks')

    boundary=[r for r in rows if r['sample_size']==n and r['task_profile']==profile and r['task_group']=='all'
              and r['method'] in ('learned_linear','learned_relu') and r['region'] in ('near_boundary','far_boundary')]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,metric,title in zip(axes,('classification_error','test_mse'),('Score-sign classification error','Task-score MSE')):
        panel(ax,boundary,metric,'active_count',('method','region'),title)
    axes[0].legend(fontsize=7)
    finish(fig,'decision_margin.png',f'Conditional on |true standardized score| <= {config["margin_band"]} or above; n={n}, profile {profile}')

    lines=['# Learned aliasing hypotheses: run results','',
           'This is a hypothesis test, not automatic confirmation. Frozen-map GAD uses an independent finite reference head.',
           'See configuration.json, geometry.csv, paired_contrasts.csv and paired_profile_contrasts.csv.','',
           '## Selected full-readout results','',
           f'Task profile {profile}; readout n={n}. MSEs are averaged across all task scores, whose analytic variances are one.','',
           '| Active features | Method | Test MSE | Model insufficiency | Aliasing | Data insufficiency | Residual gain |',
           '| ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in sorted(base,key=lambda r:(r['active_count'],r['method'])):
        lines.append(f"| {r['active_count']:g} | {r['method']} | "+' | '.join(f"{r[m]:.5g}" for m in ('test_mse','model_insufficiency','aliasing','data_insufficiency','alias_residual_gain'))+' |')
    lines += ['', '## Reading the tests','',
      '- H1: compare learned ReLU with learned linear AND the population linear oracle, at the same compression and expansion dimensions.',
      '- H2: compare learned gates with orthogonal rotations after refitting heads; the linear rotation control must be invariant. The rate-matched variant changes biases, so only its weight spectrum, not its biased preactivation Gram, is preserved.',
      '- H3: inspect threshold tradeoffs in test risk, model insufficiency, aliasing, activation rate and rank. Zero aliasing from an all-off map is a failure to retain information.',
      '- H4: inspect paired A/B task-priority contrasts for the SAME task group. A single easy task or a change in task variance is not evidence of many-feature recovery.',
      '- Near-boundary classification is measured separately from squared-risk GAD. Conditional component cross terms are retained.',
      '- Gain is the realized target-residual RMS ratio; it is not the full generalized-aliasing operator norm. Blank gains have negligible denominators.',
      '- No Marchenko-Pastur, spectral-filter, or universal superiority claim follows from these measurements.','',
      '## Figures','']
    for name in figures:lines += [f'![{name}]({name})','']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    with ZipFile(output/'results_bundle.zip','w',compression=ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name not in ('raw.csv','task_raw.csv','results_bundle.zip'):
                archive.write(path,path.name)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    plot_all(parser.parse_args().output_dir)
