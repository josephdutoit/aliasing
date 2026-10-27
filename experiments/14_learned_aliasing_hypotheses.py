"""Controlled hypotheses about learned sparse compression and GAD.

See LEARNED_ALIASING_EXPERIMENTS.md. Training uses only representation-training
data. Readout, reference and test sets are independent. All variants refit the
same minimum-norm task bank using identical readout samples. No MP assumptions.
"""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import importlib.metadata
import json
from pathlib import Path
import sys
import subprocess
import time

import numpy as np

from aliasing.controlled_superposition import (
    features, fit_gad, geometry, importance, initialize, linear_oracle,
    measure, preactivation, problem, rotate, rotation_checks, sample, train,
)


PROFILES = {
 'smoke': dict(features=6,input_dim=8,compression=2,expansion=6,active_counts=[1,3],
               num_seeds=2,steps=120,n_representation=128,n_reference=128,n_test=128,
               sample_sizes=[2,4,7,12],draws=2,rotations=1,bootstrap=200),
 'overnight': dict(features=16,input_dim=32,compression=4,expansion=16,active_counts=[1,2,4,8,12,16],
               num_seeds=24,steps=2500,n_representation=2048,n_reference=4096,n_test=4096,
               sample_sizes=[4,8,16,17,24,32,64,128],draws=3,rotations=3,bootstrap=1000),
}
GROUP_KEYS = ('active_count','task_profile','method','threshold_offset','sample_size','region','task_group')
METRICS = ('test_mse','model_insufficiency','aliasing','data_insufficiency',
 'cross_model_alias','cross_model_data','cross_alias_data','risk_identity_error',
 'residual_train_mse','alias_residual_gain','classification_error','target_second_moment',
 'weighted_test_mse','weighted_model_insufficiency','training_rank','training_nullity',
 'activation_rate','gate_flip_rate','train_mse','prediction_identity_relative_error',
 'linear_control_prediction_error','test_entries','classification_entries',
 'target_mean','target_entry_variance','positive_label_fraction',
 'global_latent_reconstruction_mse','global_latent_worst_feature_mse','global_latent_features_below_0p1',
 'threshold_zero_prediction_error','risk_identity_relative_error','training_condition_nonzero')


def csv_write(path, rows):
    rows = list(rows)
    if not rows: return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k:('' if isinstance(v,float) and not np.isfinite(v) else v) for k,v in row.items()})


def json_write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def world(config, seed_index, k, task_profile):
    start=time.monotonic()
    p,m,h=config['features'],config['compression'],config['expansion']
    seed=config['seed']+seed_index*104729
    spec=problem(seed,p,config['input_dim'],k,config['task_basis'])
    data_seed=seed+10000019*k
    xr,yr,_=sample(spec,config['n_representation'],data_seed+1)
    xp,yp,_=sample(spec,max(config['sample_sizes']),data_seed+2)
    xf,yf,_=sample(spec,config['n_reference'],data_seed+3)
    xt,yt,latent=sample(spec,config['n_test'],data_seed+4)
    weights=importance(task_profile,p,config['secondary_weight'])
    base={}
    history=[]
    for nonlin in (False,True):
        name='relu' if nonlin else 'linear'
        init=initialize(xr,m,h,p,seed+17,nonlin)
        base['random_'+name]=(init,nonlin)
        learned,trace=train(init,xr,yr,weights,nonlin,config['steps'],config['batch_size'],
                            config['learning_rate'],config['weight_decay'],seed+23)
        base['learned_'+name]=(learned,nonlin)
        history.extend(dict(method='learned_'+name,**row) for row in trace)
    oracle,floor=linear_oracle(spec,weights,m,h)
    base['linear_oracle']=(oracle,False)
    learned=base['learned_relu'][0]
    variants=[dict(method=name,rotation_id=0,threshold_offset=0.,params=params,relu=nonlin,
                   rotation_gram_error=0.,rotation_spectrum_error=0.,preactivation_geometry_preserved=1)
              for name,(params,nonlin) in base.items()]
    if config['mode'] in ('all','rotation','threshold'):
        variants.append(dict(method='no_gate',rotation_id=0,threshold_offset=0.,params=learned,relu=False,
                             rotation_gram_error=0.,rotation_spectrum_error=0.,preactivation_geometry_preserved=1))
    if config['mode'] in ('all','rotation'):
        desired_rate=float(np.mean(preactivation(xr,learned)>0))
        for rid in range(1,config['rotations']+1):
            q=np.linalg.qr(np.random.default_rng(seed+rid*1009).normal(size=(h,h)))[0]
            rotated=rotate(learned,q)
            checks=rotation_checks(xt,learned,rotated)
            for method,nonlin in (('rotated_relu',True),('linear_rotation_control',False)):
                variants.append(dict(method=method,rotation_id=rid,threshold_offset=0.,params=rotated,
                    relu=nonlin,**checks,preactivation_geometry_preserved=1))
            matched={key:value.copy() for key,value in rotated.items()}
            shift=float(np.quantile(preactivation(xr,rotated),1-desired_rate))
            matched['bias']-=shift
            actual=float(np.mean(preactivation(xr,matched)>0))
            if abs(actual-desired_rate)>2./(len(xr)*h):
                raise AssertionError('activation-rate matching failed')
            variants.append(dict(method='rotated_relu_rate_matched',rotation_id=rid,threshold_offset=0.,
                params=matched,relu=True,rotation_gram_error=float('nan'),
                rotation_spectrum_error=checks['rotation_spectrum_error'],preactivation_geometry_preserved=0))
    if config['mode'] in ('all','threshold'):
        scale=np.std(preactivation(xr,learned),axis=0)
        for offset in config['threshold_offsets']:
            params={key:value.copy() for key,value in learned.items()}
            params['bias']-=offset*scale
            variants.append(dict(method='threshold_relu',rotation_id=0,threshold_offset=offset,
                params=params,relu=True,rotation_gram_error=float('nan'),rotation_spectrum_error=0.,
                preactivation_geometry_preserved=int(offset==0)))
    common=dict(seed=seed_index,active_count=k,task_profile=task_profile,
                compression=m,expansion=h,possible_features=p,gad_definition_version=2)
    cache=[];geometries=[]
    gate_base=preactivation(xt,learned)>0
    for variant in variants:
        params,nonlin=variant['params'],variant['relu']
        hp,hf,ht=[features(x,params,nonlin) for x in (xp,xf,xt)]
        head=np.linalg.lstsq(hf,yf,rcond=None)[0]
        geom=geometry(spec,params,xt,nonlin)
        singular=np.linalg.svd(hf,compute_uv=False)
        positive=singular[singular>max(hf.shape)*np.finfo(float).eps*singular[0]]
        geom['reference_rank']=int(len(positive))
        geom['reference_condition_nonzero']=float(positive.max()/positive.min())
        geom['reference_projection_mse']=float(np.mean((hf @ head-yf)**2))
        geom['reference_head_frobenius']=float(np.linalg.norm(head))
        geom['calibration_train_activation_rate']=float(np.mean(preactivation(xr,params)>0)) if nonlin else float('nan')
        geom['gate_flip_rate']=float(np.mean((preactivation(xt,params)>0)!=gate_base)) if nonlin else float('nan')
        tag={key:variant[key] for key in ('method','rotation_id','threshold_offset')}
        geometries.append(common|tag|geom|{key:variant[key] for key in
            ('rotation_gram_error','rotation_spectrum_error','preactivation_geometry_preserved')}|
            dict(linear_weighted_population_floor=floor,task_variance_min=float(np.var(yt,axis=0).min()),
                 task_variance_max=float(np.var(yt,axis=0).max()),analytic_task_variance=1.))
        cache.append((variant,hp,ht,head,geom))
    rows=[];task_rows=[]
    task_groups={'all':np.arange(p),'first_half':np.arange(p//2),'second_half':np.arange(p//2,p)}
    for draw in range(config['draws']):
        order=np.random.default_rng(data_seed+draw*101+7).permutation(len(xp))
        for n in config['sample_sizes']:
            idx=order[:n]
            linear_prediction=None
            learned_prediction=None
            for variant,hp,ht,head,geom in cache:
                context=fit_gad(hp[idx],yp[idx],ht,yt,head)
                latent_error=((context['prediction']-yt)*spec['scales']) @ spec['teacher'].T
                feature_mse=np.mean(latent_error**2,axis=0)/np.diag(spec['covariance'])
                invariant_error=float('nan')
                zero_error=float('nan')
                if variant['method']=='learned_relu':learned_prediction=context['prediction']
                if variant['method']=='threshold_relu' and variant['threshold_offset']==0:
                    zero_error=float(np.max(np.abs(context['prediction']-learned_prediction)))
                    if zero_error>1e-12:raise AssertionError('zero threshold shift changed predictions')
                if variant['method']=='no_gate': linear_prediction=context['prediction']
                if variant['method']=='linear_rotation_control':
                    if linear_prediction is None: raise AssertionError('missing linear control')
                    invariant_error=float(np.max(np.abs(context['prediction']-linear_prediction)) /
                        max(1.,np.max(np.abs(linear_prediction))))
                    if invariant_error>1e-7: raise AssertionError('linear predictions changed under an orthogonal rotation')
                tag={key:variant[key] for key in ('method','rotation_id','threshold_offset')}
                metadata=common|tag|dict(readout_draw=draw,sample_size=n,
                    training_rank=context['training_rank'],training_nullity=context['training_nullity'],
                    training_condition_nonzero=context['training_condition_nonzero'],
                    train_mse=context['train_mse'],prediction_identity_relative_error=context['prediction_identity_relative_error'],
                    activation_rate=geom['activation_rate'],gate_flip_rate=geom['gate_flip_rate'],
                    linear_control_prediction_error=invariant_error,threshold_zero_prediction_error=zero_error,
                    global_latent_reconstruction_mse=float(feature_mse.mean()),
                    global_latent_worst_feature_mse=float(feature_mse.max()),
                    global_latent_features_below_0p1=int(np.sum(feature_mse<.1)))
                for group,tasks in task_groups.items():
                    regions=('all','near_boundary','far_boundary') if group=='all' else ('all',)
                    for region in regions:
                        metrics=measure(context,tasks,region,config['margin_band'],weights)
                        rows.append(metadata|dict(region=region,task_group=group)|metrics)
                if n==max(config['sample_sizes']):
                    for task in range(p):
                        task_rows.append(metadata|dict(task_id=task,importance=float(weights[task]))|
                            measure(context,np.array([task]),task_weights=weights))
    state={}
    for name,(params,_) in base.items():
        for key,value in params.items():state[name+'_'+key]=value
    state.update(embedding=spec['embedding'],teacher=spec['teacher'],target_scales=spec['scales'],
                 latent_mean=spec['mean'],latent_covariance=spec['covariance'],
                 target_covariance=spec['target_covariance'],importance=weights)
    return dict(rows=rows,task_rows=task_rows,geometry=geometries,history=[common|row for row in history],
                state=state,seconds=time.monotonic()-start)


def read_rows(path):
    with path.open(newline='') as f:yield from csv.DictReader(f)


def summarize(paths,output,bootstrap):
    # Collapse readout draws AND rotations within each representation seed.
    sums={};counts={}
    for path in paths:
        for row in read_rows(path):
            key=tuple(row[k] for k in GROUP_KEYS)+(row['seed'],)
            sums.setdefault(key,np.zeros(len(METRICS)));counts.setdefault(key,np.zeros(len(METRICS)))
            for j,name in enumerate(METRICS):
                value=float(row[name]) if row[name] else float('nan')
                if np.isfinite(value):sums[key][j]+=value;counts[key][j]+=1
    seed_rows=[];groups={}
    for key,total in sums.items():
        values=np.divide(total,counts[key],out=np.full(len(METRICS),np.nan),where=counts[key]>0)
        seed_rows.append(dict(zip((*GROUP_KEYS,'seed'),key))|dict(zip(METRICS,values)))
        groups.setdefault(key[:-1],[]).append(values)
    means=[];stds=[];cis=[]
    rng=np.random.default_rng(8401)
    for key,values in sorted(groups.items()):
        arr=np.asarray(values);meta=dict(zip(GROUP_KEYS,key))|dict(num_seeds=len(arr))
        mean=meta.copy();std=meta.copy();ci=meta.copy()
        # Reuse draws across metrics for this group; rows remain independent seeds.
        indexes=rng.integers(len(arr),size=(bootstrap,len(arr)))
        for j,metric in enumerate(METRICS):
            v=arr[:,j];valid=np.isfinite(v)
            mean[metric]=float(np.mean(v[valid])) if valid.any() else float('nan')
            std[metric]=float(np.std(v[valid],ddof=1)) if valid.sum()>1 else float('nan')
            if valid.sum()<2:boots=np.array([np.nan])
            elif valid.all():boots=np.mean(v[indexes],axis=1)
            elif valid.any():boots=np.mean(rng.choice(v[valid],size=(bootstrap,valid.sum())),axis=1)
            else:boots=np.array([np.nan])
            ci[metric+'_ci95_low'],ci[metric+'_ci95_high']=map(float,np.quantile(boots,[.025,.975]))
            ci[metric+'_num_seeds']=int(valid.sum())
        means.append(mean);stds.append(std);cis.append(ci)
    csv_write(output/'seed_means.csv',seed_rows)
    csv_write(output/'mean.csv',means);csv_write(output/'std.csv',stds);csv_write(output/'ci95.csv',cis)
    return seed_rows,means


def contrasts(seed_rows,output,bootstrap):
    table={tuple(r[k] for k in GROUP_KEYS)+(r['seed'],):r for r in seed_rows}
    all_groups=sorted({tuple(r[k] for k in GROUP_KEYS) for r in seed_rows})
    rng=np.random.default_rng(5023);rows=[];profile_rows=[]
    seeds=sorted({r['seed'] for r in seed_rows})
    for key in all_groups:
        row=dict(zip(GROUP_KEYS,key))
        method=row['method']
        if method=='learned_relu':continue
        ref=row.copy();ref['method']='no_gate' if method=='linear_rotation_control' else 'learned_relu'
        ref['threshold_offset']='0.0'
        refkey=tuple(ref[k] for k in GROUP_KEYS)
        for metric in ('test_mse','model_insufficiency','aliasing','alias_residual_gain','classification_error'):
            values=[]
            for seed in seeds:
                a,b=table.get(key+(seed,)),table.get(refkey+(seed,))
                if a and b and np.isfinite(a[metric]) and np.isfinite(b[metric]):values.append(a[metric]-b[metric])
            if not values:continue
            v=np.asarray(values);boots=np.mean(rng.choice(v,(bootstrap,len(v))),axis=1) if len(v)>1 else np.array([np.nan])
            rows.append(row|dict(reference_method=ref['method'],metric=metric,num_pairs=len(v),
                candidate_minus_reference=float(v.mean()),ci95_low=float(np.quantile(boots,.025)),
                ci95_high=float(np.quantile(boots,.975))))
    for key in all_groups:
        row=dict(zip(GROUP_KEYS,key))
        if row['task_profile']!='B' or row['method'] not in ('learned_relu','learned_linear','random_relu','random_linear','linear_oracle'):continue
        ref=row.copy();ref['task_profile']='A';refkey=tuple(ref[k] for k in GROUP_KEYS)
        for metric in ('test_mse','aliasing','model_insufficiency','classification_error'):
            values=[]
            for seed in seeds:
                a,b=table.get(key+(seed,)),table.get(refkey+(seed,))
                if a and b and np.isfinite(a[metric]) and np.isfinite(b[metric]):values.append(a[metric]-b[metric])
            if not values:continue
            v=np.asarray(values);boots=np.mean(rng.choice(v,(bootstrap,len(v))),axis=1) if len(v)>1 else np.array([np.nan])
            profile_rows.append(row|dict(metric=metric,num_pairs=len(v),profile_B_minus_A=float(v.mean()),
                ci95_low=float(np.quantile(boots,.025)),ci95_high=float(np.quantile(boots,.975))))
    csv_write(output/'paired_contrasts.csv',rows);csv_write(output/'paired_profile_contrasts.csv',profile_rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile',choices=PROFILES,default='smoke')
    ap.add_argument('--mode',choices=('all','sparsity','rotation','threshold','task'),default='all')
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--summarize-only',action='store_true')
    ap.add_argument('--seed',type=int,default=20261027)
    ap.add_argument('--task-basis',choices=('orthogonal','identity'),default='orthogonal')
    ap.add_argument('--task-profiles',default='balanced,A,B')
    ap.add_argument('--secondary-weight',type=float,default=.25)
    ap.add_argument('--threshold-offsets',default='-1,0,0.5,1,2,4')
    ap.add_argument('--margin-band',type=float,default=.25)
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--learning-rate',type=float,default=.005)
    ap.add_argument('--weight-decay',type=float,default=1e-5)
    for key in ('features','input-dim','compression','expansion','num-seeds','steps','n-representation','n-reference','n-test','draws','rotations','bootstrap'):
        ap.add_argument('--'+key,type=int,default=None)
    ap.add_argument('--active-counts',default=None)
    ap.add_argument('--sample-sizes',default=None)
    args=ap.parse_args();config=PROFILES[args.profile].copy()
    config.update({k:v for k,v in vars(args).items() if v is not None and k not in ('active_counts','sample_sizes','output_dir','workers','resume','profile','summarize_only')})
    for key in ('active_counts','sample_sizes'):
        if getattr(args,key) is not None:config[key]=[int(x) for x in getattr(args,key).split(',')]
    config['task_profiles']=args.task_profiles.split(',')
    config['threshold_offsets']=[float(x) for x in args.threshold_offsets.split(',')]
    p,m,h=config['features'],config['compression'],config['expansion']
    if not 2<=p or not 1<=m<p or h<m or config['input_dim']<p:
        ap.error('require 1 <= compression < features <= input-dim and expansion >= compression')
    if config['num_seeds']<2 or config['draws']<1 or config['steps']<1 or config['bootstrap']<20:
        ap.error('need >=2 seeds, positive steps/draws, >=20 bootstrap replicates')
    if any(not 1<=k<=p for k in config['active_counts']) or not config['sample_sizes'] or min(config['sample_sizes'])<1:
        ap.error('invalid active counts or sample sizes')
    if config['n_reference']<=h+1 or min(config['n_representation'],config['n_test'],config['batch_size'])<2:
        ap.error('reference size must exceed expansion+1; sample sizes must be >=2')
    if not 0<config['secondary_weight']<=1 or config['margin_band']<=0 or config['rotations']<1:
        ap.error('invalid importance, margin or rotations')
    if any(x not in ('A','B','balanced') for x in config['task_profiles']):ap.error('unknown task profile')
    for key in ('active_counts','sample_sizes','task_profiles','threshold_offsets'):
        if len(config[key])!=len(set(config[key])):ap.error('duplicate values in '+key)
    if config['learning_rate']<=0 or config['weight_decay']<0:ap.error('invalid optimizer settings')
    if args.summarize_only and not args.resume:ap.error('--summarize-only requires --resume')
    output=args.output_dir.resolve()
    if output.exists() and not args.resume:ap.error('output exists; choose a new directory or --resume')
    output.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256((Path(__file__).read_bytes()+Path(sys.modules['aliasing.controlled_superposition'].__file__).read_bytes())).hexdigest()
    config_file=output/'configuration.json'
    expected=dict(config=config,code_sha256=digest,gad_definition_version=2)
    if args.resume and config_file.exists() and json.loads(config_file.read_text())!=expected:
        ap.error('resume requires the same configuration AND code; select a new output directory')
    json_write(config_file,expected)
    worlds=list(itertools.product(range(config['num_seeds']),config['active_counts'],config['task_profiles']))
    pending=[];completed=[];failures=[]
    for seed,k,profile in worlds:
        directory=output/'shards'/f'seed{seed}_k{k}_{profile}'
        if (directory/'complete.json').exists():completed.append(directory)
        else:pending.append((seed,k,profile,directory))
    workers=max(1,min(args.workers,int(os.getenv('SLURM_CPUS_PER_TASK',str(os.cpu_count() or 1)))))
    if args.summarize_only:pending=[]
    print(f'{len(pending)} configurations pending, {len(completed)} complete; {workers} workers',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(world,config,s,k,p):(s,k,p,d) for s,k,p,d in pending}
        for future in as_completed(futures):
            seed,k,profile,directory=futures.pop(future)
            try:
                result=future.result();directory.mkdir(parents=True,exist_ok=True)
                for name in ('rows','task_rows','geometry','history'):csv_write(directory/(name+'.csv'),result[name])
                np.savez_compressed(directory/'checkpoints.npz',**result['state'])
                controls={}
                for key in ('risk_identity_error','risk_identity_relative_error','prediction_identity_relative_error',
                            'linear_control_prediction_error','threshold_zero_prediction_error'):
                    valid=[r[key] for r in result['rows'] if np.isfinite(r[key])]
                    controls[key]=max(valid) if valid else None
                for key in ('rotation_gram_error','rotation_spectrum_error'):
                    valid=[r[key] for r in result['geometry'] if np.isfinite(r[key])]
                    controls[key]=max(valid) if valid else None
                json_write(directory/'complete.json',dict(seed=seed,active_count=k,task_profile=profile,
                    seconds=result['seconds'],control_maxima=controls))
                completed.append(directory)
                print(f'Completed seed={seed}, k={k}, task={profile}: {result["seconds"]:.1f}s',flush=True)
            except Exception as exc:
                failures.append(dict(seed=seed,active_count=k,task_profile=profile,error=repr(exc)))
                print(f'FAILED seed={seed}, k={k}, task={profile}: {exc}',flush=True)
            json_write(output/'progress.json',dict(completed=len(completed),total=len(worlds),failures=failures))
    if completed:
        for source,name in (('rows','raw'),('task_rows','task_raw'),('geometry','geometry'),('history','training_history')):
            def records():
                for directory in sorted(completed):yield from read_rows(directory/(source+'.csv'))
            csv_write(output/(name+'.csv'),records())
        seed_rows,means=summarize([d/'rows.csv' for d in completed],output,config['bootstrap'])
        contrasts(seed_rows,output,config['bootstrap'])
    maxima={}
    for directory in completed:
        for key,value in json.loads((directory/'complete.json').read_text()).get('control_maxima',{}).items():
            if value is not None:maxima[key]=max(maxima.get(key,0.),value)
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],
                            capture_output=True,text=True,check=False)
    json_write(output/'run_manifest.json',expected|dict(completed=len(completed),expected=len(worlds),failures=failures,
        control_maxima=maxima,git_revision=revision.stdout.strip() if revision.returncode==0 else 'unknown',
        numpy_version=np.__version__,matplotlib_version=importlib.metadata.version('matplotlib'),python_version=sys.version,
        note='Finite-reference linear-head GAD. Draws and rotations collapsed within representation seed. Pointwise bootstrap CIs, not independent corpus intervals.'))
    if completed:
        from plot_learned_aliasing_hypotheses import plot_all
        plot_all(output)
    print(f'Results: {output}',flush=True)
    return int(bool(failures) or len(completed)!=len(worlds))


if __name__=='__main__':
    raise SystemExit(main())
