"""Seven independently launchable studies; all include learned representations."""
from __future__ import annotations

import hashlib
import itertools
import json
import numpy as np


EXPERIMENTS = ('phase', 'finite', 'ablations', 'tasks', 'blocks', 'landscape', 'higher_order')


def defaults(profile):
    smoke = profile == 'smoke'
    config = dict(profile=profile, seed=21029, num_seeds=2 if smoke else 8,
        restarts=2 if smoke else 4, steps=30 if smoke else 3000, batch_size=128,
        learning_rate=.005, checkpoint_every=30 if smoke else 100,
        training_order=4 if smoke else 20, reference_orders=[6, 10] if smoke else [40, 80],
        evaluation_order=14 if smoke else 120,
        reference_sizes=[128, 256] if smoke else [16384, 65536],
        n_test=128 if smoke else 16384, n_validation=64 if smoke else 2048,
        sample_sizes=[3, 8, 32] if smoke else [3, 4, 5, 8, 9, 15, 16, 17, 18, 31, 32, 33, 34, 64, 128, 256, 512, 2048, 4096],
        draws=2 if smoke else 8, rotations=1 if smoke else 3,
        threshold_offsets=[-.5, .5] if smoke else [-1., -.5, .5, 1., 2.],
        penalties=[.001, .1] if smoke else [1e-6, 1e-4, .01, .1, 1., 10.],
        bootstrap=100 if smoke else 1000, evaluate_starts=True,
        experiments=list(EXPERIMENTS))
    if profile == 'extended':
        config.update(num_seeds=24, restarts=12, steps=8000, training_order=32, draws=16)
    return config


def scenarios(config):
    smoke = config['profile'] == 'smoke'
    result = []

    def add(experiment, q=.5, p=2, compression=1, width=2, marginals=(.1, .1),
            training='quadrature', count=0, decay=1e-5, weights=None, angle=None,
            law='pair', eta=.5, rho=.3):
        distribution = dict(law=law, features=p)
        distribution.update(dict(marginals=list(marginals), coactivation=q) if law == 'pair' else dict(eta=eta, rho=rho))
        row = dict(experiment=experiment, distribution=distribution, compression=compression,
                   width=width, training=training, n_representation=count, decay=decay,
                   task_weights=weights or [1.]*p)
        if angle is not None:
            row['fixed_angle'] = angle
        row['scenario'] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:12]
        result.append(row)

    if smoke:
        for experiment in config['experiments']:
            if experiment in ('blocks', 'higher_order'):
                add(experiment, p=4, compression=2, width=4, training='empirical', count=64,
                    law='parity' if experiment == 'higher_order' else 'pair')
            elif experiment == 'finite':
                add(experiment, training='empirical', count=64)
            elif experiment == 'tasks':
                add(experiment, weights=[1., .25])
            elif experiment == 'landscape':
                add(experiment, angle=float(np.pi/4))
            else:
                add(experiment)
        return result
    for experiment in config['experiments']:
        if experiment == 'phase':
            for q, width, decay in itertools.product([0., .02, .05, .1, .25, .5, .75, 1.], [2, 4], [0., 1e-5, 1e-3]):
                add(experiment, q=q, width=width, decay=decay)
        elif experiment == 'finite':
            for q, n in itertools.product([0., .1, .5, 1.], [64, 256, 2048, 8192]):
                add(experiment, q=q, training='empirical', count=n)
        elif experiment == 'ablations':
            for q, width in itertools.product([0., .1, .5, 1.], [2, 4]):
                add(experiment, q=q, width=width)
        elif experiment == 'tasks':
            for q, marginal, importance in itertools.product([0., .25, 1.], [(.1, .1), (.1, .3)], [(1., 1.), (1., .1), (.1, 1.)]):
                add(experiment, q=q, marginals=marginal, weights=list(importance))
        elif experiment == 'blocks':
            for p, fraction, expansion, q, pi in itertools.product([8, 16], [4, 2], [1, 2], [0., .25, 1.], [.1, .3]):
                add(experiment, q=q, p=p, compression=p//fraction, width=p*expansion,
                    marginals=(pi, pi), training='empirical', count=8192)
        elif experiment == 'landscape':
            for q, angle in itertools.product([0., .25, 1.], np.linspace(0., np.pi, 17, endpoint=False)):
                add(experiment, q=q, angle=float(angle))
        elif experiment == 'higher_order':
            for p, eta, rho in itertools.product([8, 16], [0., .25, .5, .75, 1.], [.2, .6]):
                add(experiment, p=p, compression=p//4, width=p, training='empirical', count=8192,
                    law='parity', eta=eta, rho=rho)
    return result


def worlds(config):
    return [scenario | dict(seed=seed) for scenario in scenarios(config) for seed in range(config['num_seeds'])]


def validate(config):
    counts = ('num_seeds', 'restarts', 'steps', 'batch_size', 'checkpoint_every', 'training_order',
              'evaluation_order', 'n_test', 'n_validation', 'draws', 'bootstrap')
    if any(not isinstance(config[k], int) or isinstance(config[k], bool) or config[k] < 1 for k in counts):
        raise ValueError('counts must be positive integers')
    if config['num_seeds'] < 2 or config['training_order'] < 2 or config['rotations'] < 0 or config['seed'] < 0:
        raise ValueError('require >=2 seeds, quadrature order >=2 and nonnegative seed/rotations')
    for key in ('sample_sizes', 'reference_sizes', 'reference_orders'):
        values = config[key]
        if not values or values != sorted(set(values)) or any(not isinstance(v, int) or v < 2 for v in values):
            raise ValueError(key+' must contain increasing, distinct integers >=2')
    if config['reference_orders'][-1] >= config['evaluation_order']:
        raise ValueError('evaluation quadrature must be finer than every reference quadrature')
    if not np.isfinite(config['learning_rate']) or config['learning_rate'] <= 0:
        raise ValueError('learning rate must be positive and finite')
    if any(not np.isfinite(p) or p <= 0 for p in config['penalties']):
        raise ValueError('ridge penalties must be positive and finite')
    if not config['experiments'] or len(set(config['experiments'])) != len(config['experiments']) or any(e not in EXPERIMENTS for e in config['experiments']):
        raise ValueError('invalid experiment selection')
    return config
