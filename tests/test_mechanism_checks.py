"""Scientific controls for experiment 15, including non-MP estimators."""
import csv
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from aliasing.controlled_superposition import features, fit_gad, initialize, measure, problem, sample, train
from aliasing.mechanism_checks import (
    extend_sample, gad_metrics, normalized_probe_gain, pinverse, ridge_head,
    select_ridge, sensitivity, train_decoder,
)


def test_extending_samples_preserves_original_checkpoint_inputs_exactly():
    spec = problem(8,6,8,2)
    original = sample(spec,17,12)
    extended = extend_sample(spec,17,41,12,33)
    shortened = extend_sample(spec,17,9,12,33)
    for a,b,c in zip(original,extended,shortened):
        assert np.array_equal(a,b[:17])
        assert np.array_equal(a[:9],c)
    assert not np.array_equal(original[0], sample(spec,41,12)[0][:17])


def test_mp_and_extended_gad_preserve_canonical_terms_in_both_rank_regimes():
    rng = np.random.default_rng(43)
    ht = np.column_stack((np.ones(30),rng.normal(size=(30,6))))
    yt = rng.normal(size=(30,3)); reference = rng.normal(size=(7,3))
    for n in (4,15):
        h = np.column_stack((np.ones(n),rng.normal(size=(n,6)))); y=rng.normal(size=(n,3))
        expected = measure(fit_gad(h,y,ht,yt,reference),np.arange(3))
        actual = gad_metrics(h,y,ht,yt,reference)
        for key in ('test_mse','model_insufficiency','data_insufficiency'):
            assert np.isclose(actual[key],expected[key],atol=1e-10)
        assert np.isclose(actual['mp_aliasing'],expected['aliasing'],atol=1e-10)
        assert actual['estimator_correction']==0.
        ridge = ridge_head(h,y,.2)
        extended = gad_metrics(h,y,ht,yt,reference,ridge)
        assert extended['estimator_correction']>0
        assert extended['risk_identity_relative_error']<1e-10
        assert extended['prediction_identity_relative_error']<1e-10
        assert np.isclose(extended['mp_aliasing'],actual['mp_aliasing'])
        # An arbitrary native head has the same exact correction identity.
        native = gad_metrics(h,y,ht,yt,reference,rng.normal(size=(7,3)))
        assert native['risk_identity_relative_error']<1e-10


def test_reference_choice_changes_components_but_not_fitted_prediction_error():
    rng=np.random.default_rng(22)
    h=rng.normal(size=(12,5)); ht=rng.normal(size=(20,5))
    y=rng.normal(size=(12,2)); yt=rng.normal(size=(20,2))
    a=gad_metrics(h,y,ht,yt,np.zeros((5,2)))
    b=gad_metrics(h,y,ht,yt,rng.normal(size=(5,2)))
    assert a['test_mse']==b['test_mse']
    assert not np.isclose(a['mp_aliasing'],b['mp_aliasing'])
    assert max(a['risk_identity_relative_error'],b['risk_identity_relative_error'])<1e-10


def test_ridge_does_not_penalize_intercept_or_depend_on_feature_units():
    rng=np.random.default_rng(7)
    h=np.column_stack((np.ones(40),rng.normal(size=(40,4)),np.zeros(40)))
    y=rng.normal(size=(40,3))+3.
    original=ridge_head(h,y,.4)
    shifted=h.copy(); shifted[:,1:5]=shifted[:,1:5]*np.array([.01,100.,3.,.5])+2.
    changed=ridge_head(shifted,y,.4)
    assert np.allclose(h@original,shifted@changed,atol=1e-10)
    assert np.allclose((h@original).mean(axis=0),y.mean(axis=0))
    high=ridge_head(h,y,1e10)
    assert np.allclose(h@high,y.mean(axis=0),atol=1e-8)


def test_ridge_selection_uses_only_validation_loss():
    rng=np.random.default_rng(19)
    h=np.column_stack((np.ones(8),rng.normal(size=(8,10)))); y=rng.normal(size=(8,2))
    hv=np.column_stack((np.ones(100),rng.normal(size=(100,10)))); yv=np.ones((100,2))
    penalties=[1e-6,.01,1.,100.]; weights=np.array([1.,.25])
    head,penalty,loss=select_ridge(h,y,hv,yv,penalties,weights)
    losses=[float(np.mean((hv@ridge_head(h,y,a)-yv)**2@weights)/weights.sum()) for a in penalties]
    assert penalty==penalties[int(np.argmin(losses))]
    assert np.isclose(loss,min(losses))
    assert np.allclose(head,ridge_head(h,y,penalty))


def test_decoder_freeze_is_exact_and_joint_training_matches_original_adam():
    spec=problem(1,4,6,1); x,y,_=sample(spec,100,4)
    initial=initialize(x,2,6,4,19,True); weights=np.ones(4)
    frozen,_=train_decoder(initial,x,y,weights,100,64,.005,1e-5,30,True)
    joint,_=train_decoder(initial,x,y,weights,100,64,.005,1e-5,30,False)
    expected,_=train(initial,x,y,weights,True,100,64,.005,1e-5,30)
    assert np.array_equal(frozen['enc'],initial['enc'])
    assert not np.allclose(frozen['dec'],initial['dec'])
    assert not np.allclose(joint['enc'],initial['enc'])
    for key in joint:
        assert np.array_equal(joint[key],expected[key]),key


def test_exact_sensitivities_match_explicit_operators():
    rng=np.random.default_rng(72)
    for n in (3,14):
        h=rng.normal(size=(n,7)); ht=rng.normal(size=(25,7))
        inverse,_,_=pinverse(h); operator=ht@inverse
        exact=sensitivity(h,ht,inverse)
        assert np.isclose(exact['isotropic_prediction_energy'],np.linalg.norm(operator,'fro')**2/len(ht))
        assert np.isclose(exact['readout_operator_rms_spectral'],np.linalg.norm(operator,2)/np.sqrt(len(ht)))
        assert np.isclose(exact['worst_training_rms_gain'],np.linalg.norm(operator,2)*np.sqrt(n/len(ht)))
        delta=rng.normal(size=(n,1)); gain=normalized_probe_gain(inverse,ht,delta)
        assert np.isclose(gain['rms_gain'],np.sqrt(np.mean((operator@delta)**2)/np.mean(delta**2)))
        assert normalized_probe_gain(inverse,ht,np.zeros((n,1)))['valid']==0


def test_seed_bootstrap_does_not_count_rotations_draws_or_probe_repeats_as_seeds():
    path=Path(__file__).resolve().parents[1]/'experiments/15_gad_mechanism_checks.py'
    spec=importlib.util.spec_from_file_location('mechanism_runner',path)
    runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    with TemporaryDirectory() as tmp:
        root=Path(tmp); shards=[]
        for seed in (0,1):
            shard=root/f'seed{seed}'; shard.mkdir(); shards.append(shard)
            for table in runner.TABLES:
                records=[]
                for rotation in (1,2):
                    for draw in (0,1):
                        for repeat in (0,1,2):
                            row=dict(active_count=1,task_profile='balanced',method='learned_relu',sample_size=16,
                                fit_split='independent',estimator='mp',reference_size=128,probe_kind='shared_gaussian',
                                step=50,seed=seed,rotation_id=rotation,readout_draw=draw,probe_id=repeat,
                                test_mse=seed+rotation,rms_gain=seed+rotation)
                            records.append(row)
                if table=='readouts':
                    records += [row|dict(estimator='ridge_validation',test_mse=row['test_mse']/2) for row in records.copy()]
                runner.csv_write(shard/(table+'.csv'),records)
            # Tables are consumed independently with their own grouping keys.
        runner.summarize(shards,root,100)
        rows=list(runner.read_rows(root/'probes_mean.csv'))
        assert rows[0]['num_seeds']==2
        assert np.isclose(rows[0]['rms_gain'],2.)
        paired=list(runner.read_rows(root/'readouts_paired.csv'))
        ridge=[r for r in paired if r['estimator']=='ridge_validation' and r['metric']=='test_mse']
        assert len(ridge)==1 and ridge[0]['num_pairs']==2
        assert ridge[0]['candidate_minus_reference']==-1.
