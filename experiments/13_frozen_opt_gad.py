"""Frozen OPT layer features: samplewise/featurewise GAD on a next-token task.

Requires varied UTF-8 text with blank lines separating independent documents.
The binary target asks whether the next token belongs to a fixed frequent-token
set selected using TRAIN documents alone. This is a reproducible probe of a
specific task, not a measurement of the language model's overall loss or a
claim that an MLP neuron is a concept. Feature width means nested PCA readout
coordinates, NOT a change in the OPT architecture.

Run: python experiments/13_frozen_opt_gad.py --text-file corpus.txt \
       --output-dir outputs/chapter2/frozen_opt --layers 0,5,11
Replay a custom activation dump: --from-npz saved.npz (keys train_X,
ref_X,test_X,train_y,ref_y,test_y; X arrays shape [samples,layers,hidden]).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aliasing.static_geometry import finite_support_gad
from aliasing.summaries import aggregate_seed_rows


def ints(value: str) -> list[int]:
    try:
        result = [int(x) for x in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or min(result) < 0:
        raise argparse.ArgumentTypeError("expected nonnegative comma-separated integers")
    return result


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def gather(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    docs = [s.strip() for s in args.text_file.read_text(encoding="utf-8").split("\n\n") if s.strip()]
    if len(docs) < 6:
        raise ValueError("need >=6 independent documents separated by blank lines")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id).to(args.device)
    model.eval()
    n_layers = len(model.model.decoder.layers)
    if any(i >= n_layers for i in args.layers):
        raise ValueError(f"requested layer outside 0..{n_layers-1}")
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(len(docs))
    a, b = max(1, int(.6 * len(docs))), max(2, int(.8 * len(docs)))
    partitions = {"train": shuffled[:a], "ref": shuffled[a:b], "test": shuffled[b:]}
    if any(not len(v) for v in partitions.values()):
        raise ValueError("document split has no reference or test documents")
    windows = {}
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    width = min(args.max_length, model.config.max_position_embeddings) - len(bos)
    if width < 3:
        raise ValueError("max-length is too small")
    for split, ids in partitions.items():
        chunks = []
        for i in ids:
            tokens = tokenizer.encode(docs[int(i)], add_special_tokens=False)
            # Discard the last token in each window, since it lacks a target.
            for start in range(0, len(tokens), width):
                piece = tokens[start:start+width]
                if len(piece) >= 3:
                    chunks.append(bos + piece)
        if not chunks:
            raise ValueError(f"{split} contains no usable token windows")
        windows[split] = chunks
    # The class definition is determined by training next-token frequencies.
    counts = Counter(t for chunk in windows["train"] for t in chunk[1:])
    selected, total = set(), 0
    for t, c in counts.most_common():
        selected.add(t); total += c
        if total >= args.positive_fraction * sum(counts.values()):
            break
    result = {}
    with torch.inference_mode():
        for split, chunks in windows.items():
            features, labels, collected = [], [], 0
            for chunk in chunks:
                if collected >= args.max_tokens_per_split:
                    break
                ids = torch.tensor([chunk], device=args.device)
                out = model.model.decoder(input_ids=ids, output_hidden_states=True,
                                          use_cache=False, return_dict=True)
                remaining = args.max_tokens_per_split - collected
                # Context state at position t predicts token t+1.
                count = min(len(chunk)-1, remaining)
                X = torch.stack([out.hidden_states[layer+1][0, :count].float().cpu()
                                 for layer in args.layers], dim=1).numpy()
                features.append(X)
                labels.extend(float(t in selected) for t in chunk[1:count+1])
                collected += count
            result[split+"_X"] = np.concatenate(features)
            result[split+"_y"] = np.asarray(labels, dtype=float)
    weights = {}
    for i in args.layers:
        W = model.model.decoder.layers[i].fc1.weight.detach().float().cpu().numpy()
        singular = np.linalg.svd(W, compute_uv=False)
        eig = singular ** 2
        weights[i] = {"fc1_top_eigenvalue": float(eig[0]),
                      "fc1_participation_rank": float(eig.sum()**2 / (eig@eig)),
                      "fc1_top10_energy_fraction": float(eig[:max(1,len(eig)//10)].sum()/eig.sum())}
    return result, weights, {"document_counts": {k: len(v) for k,v in partitions.items()},
                             "positive_vocabulary_size": len(selected)}


def evaluate(data, layers: list[int], widths: list[int], sizes: list[int], seed: int,
             num_draws: int, weights=None):
    for name in ("train", "ref", "test"):
        X, y = data[name+"_X"], data[name+"_y"]
        if X.ndim != 3 or len(X) != len(y) or X.shape[1] != len(layers):
            raise ValueError(f"invalid {name} arrays")
    if max(sizes) > len(data["train_y"]):
        raise ValueError("largest sample size exceeds collected train tokens")
    orderings = [np.random.default_rng(seed + 104729 * i).permutation(len(data["train_y"]))
                 for i in range(num_draws)]
    rows = []
    for col, layer in enumerate(layers):
        ref = data["ref_X"][:, col, :].astype(np.float64)
        center = ref.mean(axis=0)
        # PCA is fit on an independent, UNLABELED reference document split.
        _, sv, vt = np.linalg.svd(ref-center, full_matrices=False)
        if max(widths) > len(sv):
            raise ValueError("width exceeds reference covariance rank limit")
        scores = {name: (data[name+"_X"][:, col, :].astype(np.float64)-center) @ vt.T
                  for name in ("train", "ref", "test")}
        for width in widths:
            # The intercept is always present and is counted separately from width.
            zref = np.column_stack([np.ones(len(ref)), scores["ref"][:, :width]])
            ztest = np.column_stack([np.ones(len(data["test_y"])), scores["test"][:, :width]])
            ztrain = np.column_stack([np.ones(len(data["train_y"])), scores["train"][:, :width]])
            head = np.linalg.lstsq(zref, data["ref_y"], rcond=None)[0]
            for draw, ordering in enumerate(orderings):
                for n in sizes:
                    idx = ordering[:n]
                    g = finite_support_gad(ztrain[idx], data["train_y"][idx], ztest,
                                           data["test_y"], np.full(len(ztest), 1/len(ztest)),
                                           population_head=head)
                    if g["risk_identity_error"] > 1e-6 * max(1., g["test_mse"]):
                        raise AssertionError("test GAD risk identity failed")
                    rows.append({"layer": layer, "pca_width": width, "sample_size": n,
                             "readout_draw": draw,
                             "train_positive_fraction": float(data["train_y"].mean()),
                             "ref_positive_fraction": float(data["ref_y"].mean()),
                             "test_positive_fraction": float(data["test_y"].mean()),
                             "constant_test_mse": float(np.mean((data["test_y"]-data["ref_y"].mean())**2)),
                             "ref_pca_variance_fraction": float(np.sum(sv[:width]**2)/np.sum(sv**2)),
                             **(weights or {}).get(layer, {}), **g})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--text-file", type=Path)
    source.add_argument("--from-npz", type=Path)
    ap.add_argument("--model-id", default="facebook/opt-125m")
    ap.add_argument("--device", default="cpu", help="cpu or cuda:0")
    ap.add_argument("--layers", type=ints, default=ints("0,5,11"))
    ap.add_argument("--widths", type=ints, default=ints("8,16,32,64,128"))
    ap.add_argument("--sample-sizes", type=ints, default=ints("16,32,64,128,256,512"))
    ap.add_argument("--num-readout-draws", type=int, default=5)
    ap.add_argument("--positive-fraction", type=float, default=.5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-tokens-per-split", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if not 0 < args.positive_fraction < 1 or min(args.max_length, args.max_tokens_per_split) < 3:
        ap.error("invalid token limit or positive fraction")
    if args.num_readout_draws < 2:
        ap.error("need at least two readout draws for conditional uncertainty intervals")
    if len(args.layers) != len(set(args.layers)) or len(args.widths) != len(set(args.widths)):
        ap.error("duplicate layer or width")
    if args.from_npz:
        with np.load(args.from_npz) as f:
            data = {k: f[k] for k in f.files}
        weights, meta = {}, {"source": str(args.from_npz)}
    else:
        data, weights, meta = gather(args)
    rows = evaluate(data, args.layers, args.widths, args.sample_sizes, args.seed,
                    args.num_readout_draws, weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.output_dir/"raw.csv", rows)
    grouping = ("layer", "pca_width", "sample_size")
    means, stds, cis = aggregate_seed_rows(rows, group_keys=grouping,
        excluded_keys={"readout_draw", *grouping}, n_bootstrap=1000)
    for name, items in (("mean", means), ("std", stds), ("ci95", cis)):
        save_csv(args.output_dir/f"{name}.csv", items)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, metric in zip(axes.flat, ("test_mse", "model_insufficiency", "aliasing", "data_algorithm")):
        for layer in args.layers:
            for width in (min(args.widths), max(args.widths)):
                rr = sorted((r for r in means if r["layer"] == layer and r["pca_width"] == width),
                            key=lambda r: r["sample_size"])
                ax.plot([r["sample_size"] for r in rr], [r[metric] for r in rr],
                        marker=".", label=f"layer {layer}, width {width}")
        ax.set(xlabel="probe training tokens", ylabel=metric.replace("_", " "))
        ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8)
    fig.savefig(args.output_dir/"frozen_opt_gad.png", dpi=170)
    plt.close(fig)
    if weights:
        fig, ax = plt.subplots(figsize=(7, 5))
        for layer in args.layers:
            r = next(v for v in means if v["layer"] == layer and
                     v["pca_width"] == max(args.widths) and v["sample_size"] == max(args.sample_sizes))
            ax.scatter(r["fc1_participation_rank"], r["test_mse"], s=65)
            ax.annotate(str(layer), (r["fc1_participation_rank"], r["test_mse"]))
        ax.set(xlabel="fc1 weight participation rank (descriptive)",
               ylabel="probe held-out MSE", title="Weight spectrum versus task-specific risk")
        ax.grid(alpha=.3)
        fig.savefig(args.output_dir/"weight_spectrum_vs_gad.png", dpi=170)
        plt.close(fig)
    with (args.output_dir/"run_manifest.json").open("w") as f:
        json.dump({"experiment": "13_frozen_opt_gad", "config": {k: str(v) for k,v in vars(args).items()},
                   "metadata": meta, "split_sizes": {k: len(data[k+"_y"]) for k in ("train","ref","test")},
                   "rows": len(rows), "max_risk_identity_error": max(r["risk_identity_error"] for r in rows),
                   "warning": "test distribution is empirical; reference head is estimated on independent docs; cross terms are retained. CIs over readout draws condition on this one document split and one checkpoint."},f,indent=2)
    print(f"Saved {len(rows)} rows in {args.output_dir}")


if __name__ == "__main__":
    main()
