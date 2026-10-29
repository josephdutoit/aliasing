"""Controlled coactivation laws and probability-weighted pair quadrature.

The tasks are the individually centered, variance-one latent coordinates.
Changing pair coactivation preserves individual marginals, not covariance.
Parity controls preserve all joint laws on at most three coordinates per block.
"""
from __future__ import annotations

import itertools
import numpy as np


def pair_probabilities(marginals, fraction):
    a, b = map(float, marginals)
    if not (0 < a < 1 and 0 < b < 1 and 0 <= fraction <= 1):
        raise ValueError('invalid pair marginals or coactivation fraction')
    lower, upper = max(0., a+b-1.), min(a, b)
    q = lower+fraction*(upper-lower)
    return np.array([1-a-b+q, a-q, b-q, q])


def moments(spec):
    p = spec['features']
    if spec['law'] == 'pair':
        if p % 2:
            raise ValueError('pair law requires an even feature count')
        pi = np.tile(spec['marginals'], p//2).astype(float)
        second = np.outer(pi, pi)
        q = pair_probabilities(spec['marginals'], spec['coactivation'])[3]
        for j in range(0, p, 2):
            second[j, j+1] = second[j+1, j] = q
    elif spec['law'] == 'parity':
        if p % 4 or not 0 < spec['rho'] <= 1 or not 0 <= spec['eta'] <= 1:
            raise ValueError('parity requires p divisible by four and valid rho/eta')
        pi = np.full(p, spec['rho']/2)
        second = np.outer(pi, pi)
        for j in range(0, p, 4):
            second[j:j+4, j:j+4] = spec['rho']/4
    else:
        raise ValueError('unknown activation law')
    np.fill_diagonal(second, pi*13/12)
    covariance = second-np.outer(pi, pi)
    return pi, covariance


def targets(x, spec):
    mean, covariance = moments(spec)
    return (x-mean)/np.sqrt(np.diag(covariance))


def sample(spec, count, seed):
    rng = np.random.default_rng(seed)
    p = spec['features']
    active = np.zeros((count, p), dtype=bool)
    if spec['law'] == 'pair':
        probabilities = pair_probabilities(spec['marginals'], spec['coactivation'])
        patterns = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=bool)
        for j in range(0, p, 2):
            active[:, j:j+2] = patterns[rng.choice(4, count, p=probabilities)]
    else:
        patterns = np.array(list(itertools.product([0, 1], repeat=4)), dtype=bool)
        even, odd = patterns[patterns.sum(1) % 2 == 0], patterns[patterns.sum(1) % 2 == 1]
        for j in range(0, p, 4):
            enabled = rng.random(count) < spec['rho']
            parity = rng.random(count) < spec['eta']
            index = rng.integers(8, size=count)
            active[:, j:j+4] = np.where(parity[:, None], even[index], odd[index])*enabled[:, None]
    x = active*rng.uniform(.5, 1.5, size=(count, p))
    return x, targets(x, spec)


def pair_quadrature(spec, order):
    """Gauss--Legendre integration on each support; NOT exact across ReLU kinks."""
    if spec['law'] != 'pair' or spec['features'] != 2 or order < 2:
        raise ValueError('quadrature requires a two-feature pair law and order >=2')
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = 1+nodes/2, weights/2
    probabilities = pair_probabilities(spec['marginals'], spec['coactivation'])
    x, mass = [np.zeros((1, 2))], [np.array([probabilities[0]])]
    x += [np.column_stack((nodes, np.zeros(order))), np.column_stack((np.zeros(order), nodes))]
    mass += [probabilities[1]*weights, probabilities[2]*weights]
    a, b = np.meshgrid(nodes, nodes, indexing='ij')
    x.append(np.column_stack((a.ravel(), b.ravel())))
    mass.append(probabilities[3]*np.outer(weights, weights).ravel())
    x, mass = np.concatenate(x), np.concatenate(mass)
    keep = mass > 0
    x, mass = x[keep], mass[keep]
    mass /= mass.sum()
    return x, targets(x, spec), mass


def support_labels(x):
    """Four exact cases for p=2; active-count strata for larger laws."""
    if x.shape[1] == 2:
        return (x[:, 0] > 0).astype(int)+2*(x[:, 1] > 0).astype(int)
    return np.count_nonzero(x, axis=1)
