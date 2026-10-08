# Experiments

The first experiment is a fixed-feature linear baseline. It sweeps the number
of modeled features through the interpolation threshold and records:

- test mean-squared error;
- the label-independent aliasing-operator norm;
- model insufficiency from omitted features;
- data insufficiency from the training null space;
- aliasing from omitted signal redirected through the modeled fit;
- fitted training-noise contribution;
- the residual cross-term.

The cross-term is intentional: component energies do not generally add to
the total risk unless the relevant projection conditions make them orthogonal.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
python experiments/01_linear_aliasing_double_descent.py --output-dir outputs/linear_baseline
```

The output directory contains `metrics.csv` and `linear_aliasing.png`.

This baseline is the first rung of the planned ladder:

1. fixed-feature linear/GLM models;
2. frozen random-feature networks;
3. feature-learning MLPs;
4. modular addition with Fourier diagnostics;
5. time-dependent aliasing during grokking.
