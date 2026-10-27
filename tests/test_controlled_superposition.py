"""Definition, normalization and intervention controls for experiment 14."""
import csv
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from aliasing.controlled_superposition import (
    features, fit_gad, importance, initialize, linear_oracle, loss_grad,
    measure, preactivation, problem, rotate, rotation_checks, sample, train,
)
from aliasing.static_geometry import finite_support_gad


def test_exact_activity_and_analytic_normalization():
    for k in (1,3,6):
        spec=problem(12,6,8,k)
        x,y,s=sample(spec,20000,42)
        assert np.all(np.count_nonzero(s,axis=1)==k)
        assert np.allclose(spec['embedding'] @ spec['embedding'].T,np.eye(6))
        assert abs(np.mean(np.sum(x*x,axis=1))-6)<.1
        assert np.allclose(np.diag(spec['target_covariance']),1.)
        assert np.allclose(np.var(y,axis=0),1.,atol=.08)
        assert np.allclose(np.mean(y,axis=0),0.,atol=.04)
        assert np.allclose(x @ spec['embedding'].T,s)


def test_analytic_training_gradient_matches_finite_differences():
    rng=np.random.default_rng(80)
    x,y=rng.normal(size=(9,4)),rng.normal(size=(9,3))
    weights=np.array([1.,.25,.7])
    for relu in (False,True):
        params=initialize(x,2,5,3,19,relu)
        params['bias']+=.12345  # keep the finite-difference probe away from gate ties
        _,gradient=loss_grad(params,x,y,weights,relu,.013)
        for name,value in params.items():
            for index in np.ndindex(value.shape):
                before=value[index];epsilon=1e-6
                value[index]=before+epsilon;plus=loss_grad(params,x,y,weights,relu,.013)[0]
                value[index]=before-epsilon;minus=loss_grad(params,x,y,weights,relu,.013)[0]
                value[index]=before
                assert np.isclose((plus-minus)/(2*epsilon),gradient[name][index],atol=1e-7,rtol=1e-5), (relu,name,index)


def test_training_reduces_the_representation_objective():
    spec=problem(14,4,6,1)
    x,y,_=sample(spec,128,20)
    for relu in (False,True):
        init=initialize(x,2,6,4,42,relu)
        before=loss_grad(init,x,y,np.ones(4),relu,1e-5)[0]
        final,_=train(init,x,y,np.ones(4),relu,100,64,.01,1e-5,50)
        after=loss_grad(final,x,y,np.ones(4),relu,1e-5)[0]
        assert after < .8*before


def test_vectorized_gad_matches_the_canonical_scalar_definition():
    rng=np.random.default_rng(23)
    href,yref=rng.normal(size=(40,5)),rng.normal(size=(40,3))
    ht,yt=rng.normal(size=(30,5)),rng.normal(size=(30,3))
    reference=np.linalg.lstsq(href,yref,rcond=None)[0]
    for n in (3,9):
        h,y=rng.normal(size=(n,5)),rng.normal(size=(n,3))
        context=fit_gad(h,y,ht,yt,reference)
        for task in range(3):
            expected=finite_support_gad(h,y[:,task],ht,yt[:,task],np.ones(30)/30,population_head=reference[:,task])
            actual=measure(context,np.array([task]))
            for key in ('test_mse','model_insufficiency','aliasing','data_insufficiency',
                        'cross_model_alias','cross_model_data','cross_alias_data'):
                assert np.isclose(actual[key],expected[key],atol=1e-10),(n,task,key)
        for region in ('all','near_boundary','far_boundary'):
            assert measure(context,np.arange(3),region,margin=.5)['risk_identity_error']<1e-10


def test_rotations_preserve_linear_geometry_and_refitted_predictions():
    spec=problem(42,4,6,2)
    x,y,_=sample(spec,100,18)
    params=initialize(x[:30],2,6,4,45,True)
    q=np.linalg.qr(np.random.default_rng(5).normal(size=(6,6)))[0]
    rotated=rotate(params,q)
    checks=rotation_checks(x,params,rotated)
    assert max(checks.values())<1e-12
    assert np.any((preactivation(x,params)>0)!=(preactivation(x,rotated)>0))
    contexts=[]
    for candidate in (params,rotated):
        h=features(x,candidate,False)
        ref=np.linalg.lstsq(h[30:60],y[30:60],rcond=None)[0]
        contexts.append(fit_gad(h[:3],y[:3],h[60:],y[60:],ref))
    assert np.allclose(contexts[0]['prediction'],contexts[1]['prediction'],atol=1e-9)
    for component in ('alias','model','data'):
        assert np.allclose(contexts[0][component],contexts[1][component],atol=1e-9)


def test_oracle_is_the_optimal_weighted_linear_rank_bound():
    spec=problem(9,6,8,2);weights=importance('A',6,.25)
    params,floor=linear_oracle(spec,weights,2,6)
    latent_map=spec['embedding'] @ params['enc'] @ params['dec']
    teacher=spec['teacher']/spec['scales']
    cov=spec['covariance']
    head=np.linalg.pinv(latent_map.T @ cov @ latent_map) @ latent_map.T @ cov @ teacher
    error_map=latent_map @ head-teacher
    exact=float(np.dot(weights,np.diag(error_map.T @ cov @ error_map))/weights.sum())
    assert np.isclose(exact,floor,atol=1e-10)
    assert 0<floor<1


def test_all_off_hidden_units_do_not_imply_good_task_recovery():
    rng=np.random.default_rng(3)
    h=np.column_stack([np.ones(100),np.zeros((100,5))])
    y=rng.normal(size=(100,4));y-=y.mean(axis=0)
    context=fit_gad(h[:20],y[:20],h,y,np.zeros((6,4)))
    metrics=measure(context,np.arange(4))
    assert metrics['data_insufficiency']<1e-20
    # The intercept still fits a training mean and can alias residual noise.
    assert metrics['model_insufficiency']>.5
    assert metrics['test_mse']>.5


def test_summary_counts_seeds_not_repeated_draws_or_rotations():
    path=Path(__file__).resolve().parents[1]/'experiments/14_learned_aliasing_hypotheses.py'
    spec=importlib.util.spec_from_file_location('hypothesis_runner',path)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    records=[]
    for seed,offset in ((0,0),(1,3)):
        for rotation in (1,2):
            for draw in (0,1):
                meta=dict(active_count=1,task_profile='A',method='rotated_relu',threshold_offset=0.,
                          sample_size=4,region='all',task_group='all',seed=seed)
                records.append(meta|{k:float(rotation*2-1+offset) for k in runner.METRICS})
    with TemporaryDirectory() as tmp:
        root=Path(tmp);source=root/'raw.csv';runner.csv_write(source,records)
        seed_rows,means=runner.summarize([source],root,100)
        assert len(seed_rows)==2
        assert means[0]['num_seeds']==2
        assert np.isclose(means[0]['test_mse'],3.5)
