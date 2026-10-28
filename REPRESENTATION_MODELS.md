# Plug-in representation benchmark (experiment 16)

Compare the compressed toy architecture with ordinary MLPs using the same
sparse-data generator, optimizer, independent splits, frozen-map GAD evaluator,
gate interventions, readout probes and reports. This run is self-contained;
it does not need checkpoint shards from experiments 14 or 15.

## Run on the node

From the repository checkout, with its existing `.venv` installed:

```bash
git pull
.venv/bin/python experiments/16_representation_model_benchmark.py \
  --profile smoke --output-dir outputs/model_smoke --workers 2
sbatch representation_models.sbatch
```

The default batch job requests 28 CPUs, 64 GB and 12 hours. Adjust the account,
partition and allocation options to match your cluster. BLAS uses one thread
per worker. The time limit is an allocation request, not a measured runtime.
The older omnibus `overnight.sbatch` skips experiment 16 and points to this
dedicated job, which runs the complete benchmark with its own resume rules.

Results appear in `outputs/representation_models/JOB_ID/`. Start with
`RESULTS.md`, then inspect `readouts_paired.csv`, `probes_paired.csv` and the
seed tables. The script generates four figures and `results_bundle.zip`.
The bundle contains all summaries, seed means, paired contrasts, configuration,
manifest and figures; it omits raw shards and trained checkpoints. Keep the
full output on the node. Upload the ZIP for detailed analysis.

Resume an interrupted run using the same checkout:

```bash
GAD_MODEL_OUTPUT=outputs/representation_models/JOB_ID \
GAD_MODEL_RESUME=1 sbatch representation_models.sbatch
```

The saved scientific configuration is reused automatically. Resume requires
the same code/configuration and verifies checksums for completed shard files.
Failed worlds are retried; successful worlds are skipped. A missing or changed
file in a completed shard raises an error rather than accepting damaged data.
Workers may change. To summarize available shards without training:

```bash
.venv/bin/python experiments/16_representation_model_benchmark.py \
  --output-dir outputs/representation_models/JOB_ID --resume --summarize-only
```

Partial reports are labeled PARTIAL and the command exits nonzero if any
world is missing. For plotting existing summary tables only:

```bash
.venv/bin/python experiments/plot_representation_model_benchmark.py \
  --output-dir outputs/representation_models/JOB_ID
```

## Architectures and hypotheses

The default overnight run uses p=16 latent features, 32 observed input
coordinates, compression m=4, and expansion/standard width w=16.
Column-vector notation below includes an affine final head.

| Model name | Predictor | Question |
| --- | --- | --- |
| bottleneck | V ReLU(D E x + b) + c, E has 4 rows | Existing compression followed by nonlinear expansion |
| mlp_wide | V ReLU(U x + b) + c, U has 16 rows | Does the behavior persist without the rank constraint on U? |
| mlp_narrow | V ReLU(U x + b) + c, U has 4 rows | What happens when the final hidden representation itself has only four coordinates? |

For every architecture, train both an all-linear and a ReLU version. Save
their untrained random representations as paired controls. Training uses the
same representation examples, minibatch-index stream, updates, Adam
hyperparameters and objective. Checkpoints minimize the representation-training
objective evaluated every 50 updates and at the end. Test, reference and
validation data do not select representation checkpoints. ReLU biases are
initialized from the representation-training 70th preactivation percentile.

Budgets match data and updates, **not parameter count or optimization
difficulty**. Default total parameter counts are 480 (bottleneck), 800 (wide)
and 212 (narrow). Actual counts are recorded for every map. The JSON example
also offers a 9-unit MLP with 457 parameters as an approximate budget control.
This differs in width as well as architecture. No comparison isolates every
architectural difference at once.
Weight decay penalizes each factor separately in a factorized model; it is
not an identical penalty on every architecture's effective input map.

The wide all-linear model can represent the full synthetic target map. Its
population approximation floor is zero. The narrow MLP's outputs lie in an
affine subspace of dimension at most four, even with nonlinear activations;
the covariance-tail bound still applies. A four-dimensional compression
followed by 16 nonlinear features has a different restriction. Do not expect
every architecture to reproduce the same sparse recovery advantage.

Two privileged population-linear oracle representations are included: rank
m and rank p. Finite-sample heads are refitted on them too. These are labeled
oracles, not learned competitors. Output-subspace lower bounds are unweighted
population bounds; finite test estimates can fall slightly below them.

## Overnight design

- 24 seeds, exact active counts k=1,2,4,8,12,16; balanced task importance.
- Exactly k uniformly selected nonzero coordinates, independent Uniform[1/2,3/2]
  amplitudes, and scaling sqrt(p/(k*13/12)). Expected squared input norm is p.
  Means and covariance still change with k; this does not hold every input
  statistic fixed. Targets are centered, mixed by an orthogonal teacher and
  normalized using analytic per-query variances, as in experiments 14/15.
- 2,048 representation-training samples, 2,500 Adam updates, batch size 128,
  learning rate .005, weight decay 1e-5 on weight matrices only.
- Independent readout pool of 4,096 samples; n=3,4,5,8,15,16,17,18,32,64,128,
  256,512,1024,2048,4096, with three orders of the same pool. These sizes cover
  nominal readout dimensions 5 and 17; actual function rank is measured.
- Nested independent references of 16,384 and 65,536 samples; validation 2,048;
  test 8,192. The sampler is called once per full split and prefixes are then
  selected. All models see the same samples within a world.
- Each frozen map receives independent MP and validation-selected ridge heads
  at each n. Ridge penalties are 1e-6,1e-4,.01,.1,1,10, using training-only
  feature standardization and an unpenalized intercept.
- Native heads and MP heads on the identical representation-training samples
  provide a separate, explicitly labeled diagnostic.
- Three orthogonal final-preactivation rotations plus rate-matched versions;
  a final-gate-removal intervention. Every intervened map receives fresh heads.
- Exact isotropic perturbation energy and worst RMS gain; eight shared Gaussian
  and quadratic directions and eight map-residualized quadratic directions.
- Repeated readout orders, rotations and perturbation directions are averaged
  within seed. Bootstrap intervals and paired differences resample seeds.
  Means, medians, maxima and top-five shares are saved for test MSE and MP
  aliasing. Intervals are pointwise exploratory intervals.

The smoke profile uses smaller dimensions, samples, training budgets and two
seeds. It checks execution and identities; it is not scientific evidence of
convergence or an overnight hypothesis result.

## GAD definitions and interpretation

For each frozen representation, H includes an intercept. A separate reference
fit estimates beta*, and r_T=Y_T-H_T beta*. The canonical prediction identity is

```text
H_P beta_MP - Y_P
 = H_P(H_T^+ H_T-I) beta*       [data insufficiency]
 + H_P H_T^+ r_T               [MP generalized aliasing]
 + (H_P beta*-Y_P)             [signed model residual].
```

Ridge/native heads add H_P(beta_candidate-beta_MP). All four energies and all
six signed pairwise cross terms are saved. `mp_aliasing` is always the MP
baseline, including on ridge rows; it is not ridge's actual residual leakage.
The reported risk uses direct predictions. Identity diagnostics retain
absolute, risk-relative and component-scaled errors and cancellation ratios.

Finite reference fits are estimated projections. Their model energy and tiny
aliasing terms need convergence checks; cross terms are not assumed zero.
The reference tables compare prediction and alias-vector differences to the
largest fit at n closest to 128 and the largest n.

Pure rotations preserve the preactivation Gram and weight singular values.
Compensated linearized native predictions are checked too. Activity matching
changes bias and is a distinct intervention; its error is recorded because
ties can prevent exact matching. Controlled perturbation sensitivities concern
K=H_P H_T^+, not the full omitted-feature GAD operator. They do not prove
uniform suppression of aliasing. Lower model error and lower sensitivity are
separate hypotheses.

## Add a model without changing the evaluator

For another width or depth, write a JSON list and pass it with `--models-file`:

```json
[
  {"name": "bottleneck", "family": "bottleneck", "compression": 4, "width": 16},
  {"name": "mlp_deep", "family": "mlp", "widths": [16, 16]}
]
```

The first default comparison plus optional deeper/budget arms is provided in
`experiments/configs/representation_models.json`. Select names explicitly:

```bash
sbatch representation_models.sbatch \
  --models-file experiments/configs/representation_models.json \
  --models bottleneck,mlp_wide,mlp_narrow,mlp_deep,mlp_budget
```

The built-in MLP supports arbitrary hidden-depth lists. `linear` removes all
gates when training its baseline; `no_final_gate` removes only the last gate
from a learned ReLU stack, leaving earlier activations intact.

The current plug-in interface supports NumPy ReLU/linear representations with
an affine readout. Other frameworks need an adapter implementing the same
operations; a raw PyTorch module cannot be passed directly.

For a different architecture, implement `RepresentationModel` in
`src/aliasing/representation_models.py`'s protocol and register a factory in
your own importable module:

```python
from aliasing.representation_models import register_model

def factory(spec, activation):
    return YourRepresentation(spec, activation)

register_model("your_family", factory)
```

Pass `--model-module your_module` and a JSON model with `family=your_family`.
Workers import the module before building models. Factories accept both
`relu` and `linear` baseline requests. Models expose a finite float64 parameter
dictionary, its loss/gradients, hidden features with an intercept, the affine
native head, final preactivations, final-gate removal and an orthogonal final
preactivation intervention. `preactivation_weight` exposes the matrix whose
singular values the intervention preserves. Optionally implement
`shift_final_bias(params, shift)` to enable the activity-matched arm; otherwise
only that arm is skipped. Factory/model source and plug-in module sources
are included in the run fingerprint. No evaluator changes are needed.

The shared benchmark lives in `src/aliasing/representation_benchmark.py`,
models and Adam in `representation_models.py`, and seed aggregation in
`benchmark_results.py`. Experiments 14/15 and their source modules are kept
unchanged so their recorded code hashes and old checkpoints remain usable.

Run the scientific controls with:

```bash
.venv/bin/python -m pytest tests/test_representation_models.py \
  tests/test_controlled_superposition.py tests/test_mechanism_checks.py
```

Install the existing optional development dependencies (`pip install -e '.[dev]'`)
if pytest is not present. The experiment itself adds no dependencies.
