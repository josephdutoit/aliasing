"""Joint representation training, with empirical or weighted quadrature loss."""
from __future__ import annotations

import numpy as np


def loss_grad(model, params, x, y, task_weights, decay, mass=None):
    mass = np.full(len(x), 1/len(x)) if mass is None else np.asarray(mass)
    hidden, pres = model._forward(x, params)
    residual = hidden[-1] @ params['head']+params['out_bias']-y
    loss = float(np.sum(mass[:, None]*residual**2*task_weights)/task_weights.sum())
    g = 2*mass[:, None]*residual*task_weights/task_weights.sum()
    gradients = dict(head=hidden[-1].T @ g, out_bias=g.sum(0))
    g = g @ params['head'].T
    for i in reversed(range(len(model.widths))):
        if model.activations[i] == 'relu':
            g = g*(pres[i] > 0)
        gradients[f'w{i}'] = hidden[i].T @ g
        if model.biases[i]:
            gradients[f'b{i}'] = g.sum(0)
        if i:
            g = g @ params[f'w{i}'].T
    for key in params:
        if key == 'head' or key.startswith('w'):
            loss += .5*decay*float(np.sum(params[key]**2))
            gradients[key] += decay*params[key]
    return loss, gradients


def train(model, initial, x, y, mass, weights, config, seed, frozen=()):
    """Select checkpoints using ONLY the representation-training objective."""
    params = {key: value.copy() for key, value in initial.items()}
    keys = [key for key in params if key not in frozen]
    first = {key: np.zeros_like(params[key]) for key in keys}
    second = {key: np.zeros_like(params[key]) for key in keys}
    rng = np.random.default_rng(seed)
    best, best_loss, history = None, float('inf'), []
    for step in range(config['steps']+1):
        if step % config['checkpoint_every'] == 0 or step == config['steps']:
            value, gradient = loss_grad(model, params, x, y, weights, config['decay'], mass)
            if not np.isfinite(value) or any(not np.isfinite(g).all() for g in gradient.values()):
                raise FloatingPointError('nonfinite representation objective/gradient')
            if value < best_loss:
                best_loss, best = value, {key: value.copy() for key, value in params.items()}
            history.append(dict(step=step, objective=value, best_objective=best_loss,
                free_gradient_norm=float(np.sqrt(sum(np.sum(gradient[k]**2) for k in keys)))))
        if step == config['steps']:
            break
        if mass is not None:
            _, gradient = loss_grad(model, params, x, y, weights, config['decay'], mass)
        else:
            ix = rng.integers(len(x), size=min(config['batch_size'], len(x)))
            _, gradient = loss_grad(model, params, x[ix], y[ix], weights, config['decay'])
        for key in keys:
            grad = gradient[key]
            first[key] = .9*first[key]+.1*grad
            second[key] = .999*second[key]+.001*grad**2
            params[key] -= config['learning_rate']*(first[key]/(1-.9**(step+1)))/(np.sqrt(second[key]/(1-.999**(step+1)))+1e-8)
    for key in frozen:
        if not np.array_equal(best[key], initial[key]):
            raise AssertionError('frozen parameter changed: '+key)
    return best, history


def initialize(model, x, mass, tasks, seed, fixed_angle=None):
    params = model.initialize(x, tasks, seed)
    if fixed_angle is not None:
        params['w0'] = np.array([[np.cos(fixed_angle)], [np.sin(fixed_angle)]])
    # Weighted quantile calibration depends on TRAINING inputs only.
    pre = x @ params['w0'] @ params['w1']
    if model.activations[-1] == 'relu':
        probability = np.full(len(x), 1/len(x)) if mass is None else mass
        for j in range(model.width):
            order = np.argsort(pre[:, j], kind='stable')
            index = min(np.searchsorted(np.cumsum(probability[order]), .7), len(order)-1)
            params['b1'][j] = -pre[order[index], j]
    return params
