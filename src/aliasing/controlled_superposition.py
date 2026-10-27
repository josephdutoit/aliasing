"""Matched sparse-compression experiments with the repository's GAD-v2 terms.

All training and evaluation use float64 NumPy. The network is
input -> linear compression -> linear expansion -> activation -> linear tasks.
Nonlinear expansion after compression is essential: a purely linear output
from an m-dimensional hidden vector cannot reconstruct a full-rank task bank.
"""
from __future__ import annotations

import numpy as np


def problem(seed: int, p: int, input_dim: int, k: int, basis: str = "orthogonal") -> dict:
    if not 1 <= k <= p or input_dim < p:
        raise ValueError("require 1 <= active_count <= features <= input_dim")
    rng = np.random.default_rng(seed)
    embedding = np.linalg.qr(rng.normal(size=(input_dim, p)))[0].T
    teacher = np.eye(p) if basis == "identity" else np.linalg.qr(rng.normal(size=(p, p)))[0]
    # Nonzeros are uniform [.5,1.5]: E[a]=1 and E[a^2]=13/12.
    second = 13. / 12.
    mean = np.full(p, np.sqrt(k / (p * second)))
    off = (k - 1.) / ((p - 1.) * second) if p > 1 else 0.
    covariance = np.full((p, p), off)
    np.fill_diagonal(covariance, 1.)
    covariance -= np.outer(mean, mean)
    scales = np.sqrt(np.diag(teacher.T @ covariance @ teacher))
    target_covariance = (teacher.T @ covariance @ teacher) / np.outer(scales, scales)
    return dict(p=p, input_dim=input_dim, k=k, embedding=embedding, teacher=teacher,
                mean=mean, scales=scales, covariance=covariance,
                target_covariance=target_covariance)


def sample(spec: dict, count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p, k = spec['p'], spec['k']
    support = np.argpartition(rng.random((count, p)), k-1, axis=1)[:, :k]
    latent = np.zeros((count, p))
    latent[np.arange(count)[:, None], support] = rng.uniform(.5, 1.5, (count, k))
    latent *= np.sqrt(p / (k * (13./12.)))
    x = latent @ spec['embedding']
    y = ((latent - spec['mean']) @ spec['teacher']) / spec['scales']
    return x, y, latent


def importance(profile: str, p: int, secondary: float) -> np.ndarray:
    w = np.ones(p)
    if profile == 'A':
        w[p//2:] = secondary
    elif profile == 'B':
        w[:p//2] = secondary
    elif profile != 'balanced':
        raise ValueError(profile)
    return w


def initialize(x: np.ndarray, compression: int, expansion: int, tasks: int,
               seed: int, relu: bool) -> dict:
    rng = np.random.default_rng(seed)
    enc = rng.normal(size=(x.shape[1], compression)) / np.sqrt(x.shape[1])
    dec = rng.normal(size=(compression, expansion)) / np.sqrt(compression)
    pre = x @ enc @ dec
    bias = -np.quantile(pre, .7, axis=0) if relu else np.zeros(expansion)
    return dict(enc=enc, dec=dec, bias=bias,
                head=rng.normal(size=(expansion, tasks))/np.sqrt(expansion),
                out_bias=np.zeros(tasks))


def preactivation(x: np.ndarray, params: dict) -> np.ndarray:
    return x @ params['enc'] @ params['dec'] + params['bias']


def features(x: np.ndarray, params: dict, relu: bool, intercept: bool = True) -> np.ndarray:
    pre = preactivation(x, params)
    h = np.maximum(pre, 0.) if relu else pre
    return np.column_stack((np.ones(len(x)), h)) if intercept else h


def loss_grad(params: dict, x: np.ndarray, y: np.ndarray, weights: np.ndarray,
              relu: bool, decay: float) -> tuple[float, dict]:
    z = x @ params['enc']
    pre = z @ params['dec'] + params['bias']
    h = np.maximum(pre, 0.) if relu else pre
    residual = h @ params['head'] + params['out_bias'] - y
    loss = float(np.sum(residual**2 * weights) / (len(x)*weights.sum()))
    g = 2*residual*weights / (len(x)*weights.sum())
    grads = dict(head=h.T @ g, out_bias=g.sum(axis=0))
    gp = g @ params['head'].T
    if relu:
        gp *= pre > 0
    grads.update(dec=z.T @ gp, bias=gp.sum(axis=0), enc=x.T @ (gp @ params['dec'].T))
    for key in ('enc','dec','head'):
        loss += .5*decay*float(np.sum(params[key]**2))
        grads[key] += decay*params[key]
    return loss, grads


def train(initial: dict, x: np.ndarray, y: np.ndarray, weights: np.ndarray,
          relu: bool, steps: int, batch: int, lr: float, decay: float,
          seed: int) -> tuple[dict, list[dict]]:
    params = {k:v.copy() for k,v in initial.items()}
    first = {k:np.zeros_like(v) for k,v in params.items()}
    second = {k:np.zeros_like(v) for k,v in params.items()}
    rng = np.random.default_rng(seed)
    best, best_loss = {k:v.copy() for k,v in params.items()}, float('inf')
    history = []
    for step in range(steps+1):
        if step % 50 == 0 or step == steps:
            value, _ = loss_grad(params, x, y, weights, relu, decay)
            if not np.isfinite(value):
                raise FloatingPointError('nonfinite representation training objective')
            if value < best_loss:
                best_loss, best = value, {k:v.copy() for k,v in params.items()}
            history.append(dict(step=step, objective=value, best_objective=best_loss))
        if step == steps:
            break
        idx = rng.integers(len(x), size=min(batch,len(x)))
        _, grads = loss_grad(params, x[idx], y[idx], weights, relu, decay)
        t = step+1
        for key, grad in grads.items():
            first[key] = .9*first[key] + .1*grad
            second[key] = .999*second[key] + .001*grad**2
            params[key] -= lr*(first[key]/(1-.9**t))/(np.sqrt(second[key]/(1-.999**t))+1e-8)
    return best, history


def linear_oracle(spec: dict, weights: np.ndarray, compression: int, expansion: int) -> tuple[dict, float]:
    """Optimal weighted rank-m linear population predictor, using known moments.

    This privileged benchmark is labeled as an oracle, not a learned baseline.
    Its features are computed from inputs using the known synthetic teacher.
    """
    root_w = np.sqrt(weights)
    cov = spec['target_covariance'] * np.outer(root_w,root_w)
    eig, vec = np.linalg.eigh(cov)
    top = vec[:, np.argsort(eig)[::-1][:compression]]
    mapping = (spec['embedding'].T @ spec['teacher']) / spec['scales']
    enc = (mapping * root_w) @ top
    offset = (-spec['mean'] @ spec['teacher'] / spec['scales'] * root_w) @ top
    dec = np.zeros((compression,expansion)); dec[:, :compression] = np.eye(compression)
    bias = np.zeros(expansion); bias[:compression] = offset
    floor = max(0., float(np.sort(eig)[:-compression].sum()/weights.sum()))
    return dict(enc=enc,dec=dec,bias=bias), floor


def rotate(params: dict, orthogonal: np.ndarray) -> dict:
    result = {k:v.copy() for k,v in params.items()}
    result['dec'] = params['dec'] @ orthogonal
    result['bias'] = params['bias'] @ orthogonal
    return result


def rotation_checks(x: np.ndarray, original: dict, rotated: dict) -> dict:
    a, b = preactivation(x[:64],original), preactivation(x[:64],rotated)
    ga, gb = a @ a.T, b @ b.T
    wa = np.linalg.svd(original['enc'] @ original['dec'], compute_uv=False)
    wb = np.linalg.svd(rotated['enc'] @ rotated['dec'], compute_uv=False)
    gram = float(np.linalg.norm(ga-gb)/max(np.linalg.norm(ga),1e-15))
    spectrum = float(np.linalg.norm(wa-wb)/max(np.linalg.norm(wa),1e-15))
    if max(gram,spectrum)>1e-10:
        raise AssertionError('orthogonal intervention failed to preserve preactivation geometry')
    return dict(rotation_gram_error=gram, rotation_spectrum_error=spectrum)


def geometry(spec: dict, params: dict, test_x: np.ndarray, relu: bool) -> dict:
    directions = (spec['embedding'] @ params['enc']).T
    norms = np.linalg.norm(directions,axis=0)
    normalized = directions / np.maximum(norms,1e-15)
    gram = normalized.T @ normalized
    np.fill_diagonal(gram,0.)
    pre = preactivation(test_x,params)
    hidden = features(test_x,params,relu)
    return dict(encoder_coherence=float(np.max(np.abs(gram))),
                encoder_feature_norm_min=float(norms.min()), encoder_feature_norm_max=float(norms.max()),
                encoder_participation_rank=float(np.linalg.norm(directions,'fro')**4 /
                    max(np.linalg.norm(directions @ directions.T,'fro')**2,1e-30)),
                activation_rate=float(np.mean(pre>0)) if relu else float('nan'),
                preactivation_rms=float(np.sqrt(np.mean(pre**2))),
                hidden_rms=float(np.sqrt(np.mean(hidden[:,1:]**2))),
                hidden_test_rank=float(np.linalg.matrix_rank(hidden)),
                input_second_moment=float(np.mean(np.sum(test_x**2,axis=1))))


def fit_gad(h_train: np.ndarray, y_train: np.ndarray, h_test: np.ndarray,
            y_test: np.ndarray, reference_head: np.ndarray) -> dict:
    """Vectorized counterpart of static_geometry.finite_support_gad, version 2."""
    u, singular, vh = np.linalg.svd(h_train,full_matrices=False)
    tol = max(h_train.shape)*np.finfo(float).eps*(singular[0] if len(singular) else 0.)
    rank = int(np.sum(singular>tol))
    basis = vh[:rank]
    pinv = (basis.T/singular[:rank]) @ u[:, :rank].T if rank else np.zeros(h_train.T.shape)
    projector = basis.T @ basis
    null = np.eye(h_train.shape[1])-projector
    fitted = pinv @ y_train
    remainder = y_train-h_train @ reference_head
    model = h_test @ reference_head-y_test
    alias = h_test @ (pinv @ remainder)
    data = -h_test @ (null @ reference_head)
    prediction = h_test @ fitted
    error = prediction-y_test
    relative = float(np.max(np.abs(error-model-alias-data))/max(1.,np.max(np.abs(error))))
    if relative>1e-7:
        raise AssertionError(f'GAD prediction identity failed: {relative}')
    return dict(model=model,alias=alias,data=data,error=error,prediction=prediction,target=y_test,
                remainder=remainder,training_rank=rank,
                training_nullity=h_train.shape[1]-rank,
                training_condition_nonzero=float(singular[0]/singular[rank-1]) if rank else float('nan'),
                train_mse=float(np.mean((h_train @ fitted-y_train)**2)),
                prediction_identity_relative_error=relative)


def measure(context: dict, tasks: np.ndarray, region: str = 'all', margin: float = .25,
            task_weights: np.ndarray | None = None) -> dict:
    target = context['target'][:,tasks]
    mask = np.ones_like(target,dtype=bool)
    if region == 'near_boundary': mask = np.abs(target)<=margin
    if region == 'far_boundary': mask = np.abs(target)>margin
    count = int(mask.sum())
    keys = ('test_mse','model_insufficiency','aliasing','data_insufficiency',
            'cross_model_alias','cross_model_data','cross_alias_data','risk_identity_error',
            'residual_train_mse','alias_residual_gain','classification_error','target_second_moment',
            'weighted_test_mse','weighted_model_insufficiency','target_mean',
            'target_entry_variance','positive_label_fraction','risk_identity_relative_error')
    if not count:
        return {k:float('nan') for k in keys} | dict(test_entries=0, classification_entries=0)
    mean = lambda values: float(np.sum(values*mask)/count)
    a,m,d,e = [context[k][:,tasks] for k in ('alias','model','data','error')]
    result = dict(test_mse=mean(e**2), model_insufficiency=mean(m**2),
                  aliasing=mean(a**2), data_insufficiency=mean(d**2),
                  cross_model_alias=mean(2*m*a), cross_model_data=mean(2*m*d), cross_alias_data=mean(2*a*d),
                  residual_train_mse=float(np.mean(context['remainder'][:,tasks]**2)),
                  target_second_moment=mean(target**2), test_entries=count)
    result['risk_identity_error'] = abs(result['test_mse']-sum(result[k] for k in
        ('model_insufficiency','aliasing','data_insufficiency','cross_model_alias','cross_model_data','cross_alias_data')))
    result['risk_identity_relative_error'] = result['risk_identity_error']/max(1.,result['test_mse'])
    if result['risk_identity_error']>1e-7*max(1.,result['test_mse']):
        raise AssertionError('GAD squared-risk identity failed')
    denominator = result['residual_train_mse']
    result['alias_residual_gain'] = float(np.sqrt(result['aliasing']/denominator)) if denominator>1e-20 else float('nan')
    valid = mask & (np.abs(target)>1e-12)
    result['classification_entries'] = int(valid.sum())
    result['classification_error'] = float(np.mean((context['prediction'][:,tasks][valid]>=0)!=(target[valid]>=0))) if valid.any() else float('nan')
    result['target_mean'] = mean(target)
    result['target_entry_variance'] = mean((target-result['target_mean'])**2)
    result['positive_label_fraction'] = float(np.mean(target[valid]>=0)) if valid.any() else float('nan')
    weights = np.ones(len(tasks)) if task_weights is None else task_weights[tasks]
    weighted_mask = mask * weights
    result['weighted_test_mse'] = float(np.sum(e**2*weighted_mask)/weighted_mask.sum())
    result['weighted_model_insufficiency'] = float(np.sum(m**2*weighted_mask)/weighted_mask.sum())
    return result
