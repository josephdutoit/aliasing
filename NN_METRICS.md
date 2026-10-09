# Neural-network diagnostics

The neural experiments use the following quantities.

- `train_mse`, `test_mse`: empirical and held-out squared risk.
- `hidden_rank`, `hidden_singular_value_max`, `hidden_singular_value_min`, `hidden_condition_log10`: singular-value/Fisher geometry of the hidden feature matrix.
- `head_null_fraction`: fraction of the learned output head in the nullspace of the training hidden matrix.
- `test_null_response_rms`: held-out effect of that finite-sample nullspace component; this is the direct aliasing diagnostic.
- `feature_coherence_max`, `feature_effective_rank`: simple feature-sharing/superposition diagnostics.
- `ntk_drift`: relative Frobenius drift of the empirical tangent kernel from initialization.
- `feature_drift`: relative drift of the first-layer weights.
- `feature_transport_train`, `feature_transport_test`: relative movement of the hidden feature matrices from initialization.
- `null_projector_drift`: spectral-norm movement of the training-data nullspace projector.
- `alias_operator_norm`, `initial_alias_operator_norm`, `alias_operator_drift`: spectral norm of the current test-visible nullspace operator, its initialization value, and its movement from initialization.
- `interpolates`: whether training MSE is below the configured threshold.

The MLP experiment is intentionally a feature-learning bridge between fixed/frozen features and the later modular-addition transformer. It does not claim to reproduce grokking yet. The checkpointed risk and geometry traces are designed to reveal whether any late generalization improvement is associated with decreasing aliasing, changing rank/coherence, or tangent-kernel drift.

Install the optional dependency after the base project:

```bash
python -m pip install -e ".[dev]"
python -m pip install -r requirements-neural.txt
```

Run a small smoke experiment:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python experiments/03_feature_learning_mlp.py \
  --widths 16,32 --num-seeds 4 --num-workers 4 \
  --steps 200 --checkpoints 0,10,50,200 \
  --output-dir outputs/mlp_smoke
```

For a larger array job, the Cartesian product of `--num-seeds` and `--widths` is distributed as independent processes. Keep `--torch-threads 1` when using many workers, or use fewer workers and increase it when a scheduler gives each task a whole node.
