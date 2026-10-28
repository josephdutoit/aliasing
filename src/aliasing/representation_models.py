"""Small NumPy representations behind a common, extensible model interface.

The final head is affine. GAD is evaluated on [1, hidden(x)] after freezing
the representation. A custom family registers a factory with register_model;
the experiment runner never branches on a family name.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Protocol

import numpy as np

from .controlled_superposition import initialize as legacy_initialize


class RepresentationModel(Protocol):
    width: int

    def initialize(self, x, tasks, seed): ...
    def features(self, x, params, intercept=True): ...
    def preactivation(self, x, params): ...
    def loss_grad(self, params, x, y, weights, decay): ...
    def native_head(self, params): ...
    def rotate(self, params, orthogonal): ...
    def linearized(self): ...
    def preactivation_weight(self, params): ...


FACTORIES: dict[str, Callable] = {}


def register_model(name, factory):
    if name in FACTORIES:
        raise ValueError('model family already registered: '+name)
    FACTORIES[name] = factory


def make_model(spec, activation='relu') -> RepresentationModel:
    if activation not in ('relu', 'linear'):
        raise ValueError('activation must be relu or linear')
    try:
        factory = FACTORIES[spec['family']]
    except KeyError as error:
        raise ValueError('unknown model family: '+str(spec.get('family'))) from error
    return factory(spec, activation)


@dataclass(frozen=True)
class DenseRepresentation:
    widths: tuple[int, ...]
    activations: tuple[str, ...]
    biases: tuple[bool, ...]
    legacy_bottleneck: bool = False

    def __post_init__(self):
        if (not self.widths or any(not isinstance(w, int) or isinstance(w, bool) or w < 1 for w in self.widths) or
                len(self.widths) != len(self.activations) or len(self.widths) != len(self.biases) or
                any(a not in ('relu', 'linear') for a in self.activations)):
            raise ValueError('invalid layer widths, activations or biases')

    @property
    def width(self):
        return self.widths[-1]

    def initialize(self, x, tasks, seed):
        if self.legacy_bottleneck:
            old = legacy_initialize(x, *self.widths, tasks, seed, self.activations[-1] == 'relu')
            return dict(w0=old['enc'], w1=old['dec'], b1=old['bias'],
                        head=old['head'], out_bias=old['out_bias'])
        rng = np.random.default_rng(seed)
        params = {}
        h = x
        for i, (width, activation, bias) in enumerate(zip(self.widths, self.activations, self.biases)):
            params[f'w{i}'] = rng.normal(size=(h.shape[1], width))/np.sqrt(h.shape[1])
            pre = h @ params[f'w{i}']
            if bias:
                params[f'b{i}'] = -np.quantile(pre, .7, axis=0) if activation == 'relu' else np.zeros(width)
                pre = pre+params[f'b{i}']
            h = np.maximum(pre, 0.) if activation == 'relu' else pre
        params['head'] = rng.normal(size=(self.width, tasks))/np.sqrt(self.width)
        params['out_bias'] = np.zeros(tasks)
        return params

    def _forward(self, x, params):
        hidden, pres = [x], []
        for i, activation in enumerate(self.activations):
            pre = hidden[-1] @ params[f'w{i}']
            if self.biases[i]:
                pre = pre+params[f'b{i}']
            pres.append(pre)
            hidden.append(np.maximum(pre, 0.) if activation == 'relu' else pre)
        return hidden, pres

    def preactivation(self, x, params):
        return self._forward(x, params)[1][-1]

    def features(self, x, params, intercept=True):
        h = self._forward(x, params)[0][-1]
        return np.column_stack((np.ones(len(x)), h)) if intercept else h

    def native_head(self, params):
        return np.vstack((params['out_bias'], params['head']))

    def loss_grad(self, params, x, y, weights, decay):
        hidden, pres = self._forward(x, params)
        residual = hidden[-1] @ params['head']+params['out_bias']-y
        loss = float(np.sum(residual**2*weights)/(len(x)*weights.sum()))
        g = 2*residual*weights/(len(x)*weights.sum())
        gradients = dict(head=hidden[-1].T @ g, out_bias=g.sum(axis=0))
        g = g @ params['head'].T
        for i in reversed(range(len(self.widths))):
            if self.activations[i] == 'relu':
                g = g*(pres[i] > 0)
            gradients[f'w{i}'] = hidden[i].T @ g
            if self.biases[i]:
                gradients[f'b{i}'] = g.sum(axis=0)
            if i:
                g = g @ params[f'w{i}'].T
        for key in params:
            if key == 'head' or key.startswith('w'):
                loss += .5*decay*float(np.sum(params[key]**2))
                gradients[key] += decay*params[key]
        return loss, gradients

    def rotate(self, params, orthogonal):
        if orthogonal.shape != (self.width, self.width) or not np.allclose(orthogonal.T @ orthogonal, np.eye(self.width)):
            raise ValueError('rotation must be orthogonal and match final hidden width')
        result = {key: value.copy() for key, value in params.items()}
        last = len(self.widths)-1
        result[f'w{last}'] = result[f'w{last}'] @ orthogonal
        if self.biases[last]:
            result[f'b{last}'] = result[f'b{last}'] @ orthogonal
        # Compensate the native head. This preserves the linearized predictor;
        # after ReLU it is an intervention and independent heads are refitted.
        result['head'] = orthogonal.T @ result['head']
        return result

    def shift_final_bias(self, params, shift):
        last = len(self.widths)-1
        if not self.biases[last]:
            raise ValueError('activity matching requires a final preactivation bias')
        result = {key: value.copy() for key, value in params.items()}
        result[f'b{last}'] -= shift
        return result

    def linearized(self):
        """Remove only the last gate; earlier gates, if present, stay fixed."""
        return replace(self, activations=(*self.activations[:-1], 'linear'))

    def preactivation_weight(self, params):
        # For the old linear compression/expansion, expose its effective map.
        if all(a == 'linear' for a in self.activations[:-1]):
            weight = params['w0']
            for i in range(1, len(self.widths)):
                weight = weight @ params[f'w{i}']
            return weight
        return params[f'w{len(self.widths)-1}']


def _mlp(spec, activation):
    widths = tuple(spec['widths'])
    return DenseRepresentation(widths, (activation,)*len(widths), (True,)*len(widths))


def _bottleneck(spec, activation):
    return DenseRepresentation((spec['compression'], spec['width']),
                               ('linear', activation), (False, True), True)


register_model('mlp', _mlp)
register_model('bottleneck', _bottleneck)


def default_models(compression=4, width=16):
    return [dict(name='bottleneck', family='bottleneck', compression=compression, width=width),
            dict(name='mlp_wide', family='mlp', widths=[width]),
            dict(name='mlp_narrow', family='mlp', widths=[compression])]


def train_model(model, initial, x, y, weights, steps, batch, lr, decay, seed):
    """Common float64 Adam and checkpoint selection for every architecture."""
    params = {key: value.copy() for key, value in initial.items()}
    first = {key: np.zeros_like(value) for key, value in params.items()}
    second = {key: np.zeros_like(value) for key, value in params.items()}
    rng = np.random.default_rng(seed)
    best, best_loss, history = None, float('inf'), []
    for step in range(steps+1):
        if step % 50 == 0 or step == steps:
            value, _ = model.loss_grad(params, x, y, weights, decay)
            if not np.isfinite(value):
                raise FloatingPointError('nonfinite representation training objective')
            if value < best_loss:
                best_loss, best = value, {key: value.copy() for key, value in params.items()}
            history.append(dict(step=step, objective=value, best_objective=best_loss))
        if step == steps:
            break
        idx = rng.integers(len(x), size=min(batch, len(x)))
        _, gradients = model.loss_grad(params, x[idx], y[idx], weights, decay)
        if gradients.keys() != params.keys():
            raise ValueError('model must return a gradient for every parameter')
        for key, grad in gradients.items():
            t = step+1
            first[key] = .9*first[key]+.1*grad
            second[key] = .999*second[key]+.001*grad**2
            params[key] -= lr*(first[key]/(1-.9**t))/(np.sqrt(second[key]/(1-.999**t))+1e-8)
    return best, history
