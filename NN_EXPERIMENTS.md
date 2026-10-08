# Neural-network experiments

These experiments extend the GLM work in two controlled steps.

## 1. Random-feature ReLU network

`experiments/02_random_feature_aliasing.py` uses a fixed random ReLU
hidden layer and fits a minimum-norm linear readout. The hidden map is
nonlinear in the input, but the readout is still a GLM, so the existing
aliasing decomposition applies exactly to the feature matrix.

Run a feature-capacity sweep:

```bash
python -m pip install -e ".[dev]"
python experiments/02_random_feature_aliasing.py \
  --sweep featurewise \
  --num-seeds 100 \
  --num-workers 32 \
  --output-dir outputs/random_feature_featurewise
```

Run a samplewise sweep:

```bash
python experiments/02_random_feature_aliasing.py \
  --sweep samplewise \
  --sample-sizes 8,16,24,32,40,48,56,64,72,80,88 \
  --modeled-features 40 \
  --num-seeds 100 \
  --num-workers 32 \
  --output-dir outputs/random_feature_samplewise
```

Each run writes raw per-seed results, means, standard deviations, and a plot.
The random-feature matrix is generated once per seed for featurewise sweeps,
and fresh training samples are generated for each sample size in samplewise
sweeps.

## 2. Feature-learning MLP

`experiments/03_feature_learning_mlp.py` trains a two-layer ReLU MLP on a
teacher-generated regression task. At checkpoints it records:

- train and test MSE;
- hidden-feature rank and log condition number;
- tangent-like representation drift;
- the fraction of hidden readout lying in the training null space;
- the test-visible size of the training-null subspace;
- the test effect of the learned head's null-space component.

Run a model-wise sweep:

```bash
python -m pip install -r requirements-neural.txt
python experiments/03_feature_learning_mlp.py \
  --widths 16,32,64,128,256 \
  --num-seeds 100 \
  --num-workers 32 \
  --torch-threads 1 \
  --steps 5000 \
  --output-dir outputs/mlp_feature_learning
```

Use `--checkpoints` to control checkpoint locations, for example
`0,10,100,500,1000,2500,5000`.

## HPC guidance

Use one process per independent seed/width task and keep each worker's BLAS
and Torch thread count small. For CPU jobs, set environment variables before
launching:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

Then choose `--num-workers` to match the allocated cores. Do not use a large
worker count together with a large `--torch-threads` value.

For a cluster scheduler, it is also reasonable to split the seed range across
job-array tasks and give each task a separate output directory.
