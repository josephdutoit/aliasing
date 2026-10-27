# Chapter 2: additional static-representation experiments

These scripts extend experiments 05–11 on `chapter2-static-geometry-plan-20261021`.
They test spectral and real-language-model questions not answered by the
existing sparse-dictionary and random-feature sweeps. All conclusions are
conditional on the stated task, decoder class, sampling law, and checkpoint.

Install from the repository root:

```bash
python -m pip install -e .
python -m pip install torch transformers
```

## 12: same weight spectrum, different task risk

```bash
python experiments/12_matched_spectrum_tasks.py \
  --dimension 8 --sample-sizes 4,8,16,32,64,128 \
  --num-seeds 30 --num-draws 3 \
  --output-dir outputs/chapter2/matched_spectrum
```

The duplicated dictionary `[I,I]` and its orthogonal rotation in latent
coordinates have the *same complete singular spectrum*. On a one-hot latent
support, the first has an exact collision between two states, while the
rotated one generically does not. The contrast task gives those states
different targets; the common task gives them equal targets. With the default
pair mass 0.4, the exact duplicated/contrast Bayes floor is 0.4. All other
Bayes floors should be zero. A random unit-column dictionary supplies a
descriptive weight-spectrum comparison; it is not asserted to follow MP
exactly or to exhibit HTSR. The script fails if the matched spectra or GAD
identities disagree beyond tolerance.

Outputs: `raw.csv` (individual seed/draw), `mean.csv`, `std.csv`, `ci95.csv`
(pointwise bootstrap intervals over *dictionary seeds*, after averaging draws),
`matched_spectrum_tasks.png`, `run_manifest.json`. Compare Bayes floor,
linear-readout model insufficiency, training-null aliasing, and their cross
terms, not only total error. Keep large raw outputs off Git if needed.

## 13: a fixed OPT checkpoint and a defined next-token probe

Supply varied UTF-8 text with **at least six independent documents** separated
by blank lines. More documents are preferable: the train, reference and test
splits are by document. Use several hundred or more tokens in each split.

```bash
python experiments/13_frozen_opt_gad.py \
  --text-file corpus.txt --device cuda:0 --model-id facebook/opt-125m \
  --layers 0,5,11 --widths 8,16,32,64,128 \
  --sample-sizes 16,32,64,128,256,512 \
  --num-readout-draws 5 --max-tokens-per-split 4096 \
  --output-dir outputs/chapter2/frozen_opt
```

The target is **whether the next token belongs to a set of frequent tokens**
defined using train documents alone. The encoder is frozen. PCA coordinates
are fit to *unlabeled reference documents*, then nested feature counts and
training sample counts are swept. A separate reference-set head defines a
fixed comparator; the resulting GAD error-vector identity is exact on the
test set, with all risk cross terms saved. Its model-insufficiency estimate
is finite-reference and should not be called an oracle population term.
Repeated readout draws share the same documents and checkpoint: their
confidence intervals are conditional, **not independent corpus CIs**.
Repeat with different document splits/seeds and alternative tasks for a
general claim. PCA width changes a probe, not the OPT architecture.

`fc1` raw-weight participation rank, leading energy fraction, and leading
eigenvalue are saved beside each layer/task risk and a diagnostic scatter
plot. These are descriptive HTSR-related summaries; neither a fitted power
law nor identification of an MP noise bulk follows from them. Compare GAD
terms at matched probe width/sample count across layers before interpreting
the scatter.

Outputs: `raw.csv`, `mean.csv`, `std.csv`, `ci95.csv` (pointwise over readout
draws), `frozen_opt_gad.png`, `weight_spectrum_vs_gad.png`, and
`run_manifest.json`. Validate `max_risk_identity_error` in the manifest before
interpreting risk components. A held-out next-token *binary probe* is not
the pretrained model's language-model loss.

You can also evaluate saved arrays without downloading OPT by using
`--from-npz features.npz`. Required keys are `train_X`, `ref_X`, `test_X`
(each `[tokens, selected_layers, hidden]`) and `train_y`, `ref_y`, `test_y`.
Choose `--layers` matching the number and order of saved layers. Weight
statistics are omitted for this route.

## Cluster submission

From `/home/jcdutoit/aliasing`, submit either Python invocation above through
your usual SLURM wrapper. The existing `run.sh` dispatches experiments 04–11,
so it does not yet dispatch 12 or 13. For example, after installing dependencies:

```bash
mkdir -p slurm_logs
sbatch --job-name=ch2-spectrum --time=03:00:00 --cpus-per-task=8 --mem=32G \
  --output=slurm_logs/%x-%j.out \
  --wrap='cd /home/jcdutoit/aliasing && .venv/bin/python experiments/12_matched_spectrum_tasks.py --output-dir outputs/chapter2/matched_spectrum'
```

For OPT use the interactive/accelerator queue or adjust SLURM GPU options to
the cluster's policies; pass `--device cuda:0` only when a GPU is allocated.
