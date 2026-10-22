"""Train sparse-feature autoencoders and analyze final representations.

Training produces the representation; all reported theory-facing metrics are
computed at the final checkpoint. This is a static-geometry experiment. The
feature directions are known because the input coordinates are ground-truth
latent features.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from aliasing.static_geometry import (
    covariance_pair_metrics,
    feature_direction_metrics,
    finite_support_gad,
)
from aliasing.summaries import aggregate_seed_rows
from aliasing.superposition import representation_bayes_mse


torch.set_default_dtype(torch.float64)


class SparseAutoencoder(nn.Module):
    def __init__(self, feature_count: int, bottleneck: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(feature_count, bottleneck)
        self.decoder = nn.Linear(bottleneck, feature_count)

    def hidden_pre(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.hidden_pre(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.hidden(x))


def parse_ints(value: str) -> list[int]:
    try:
        result = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("all values must be positive")
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def binary_codebook(feature_count: int, active_probability: float) -> tuple[np.ndarray, np.ndarray]:
    if feature_count > 16:
        raise ValueError("exact codebook diagnostics are limited to feature_count <= 16")
    codes = ((np.arange(1 << feature_count)[:, None] >> np.arange(feature_count)) & 1).astype(float)
    counts = codes.sum(axis=1)
    probs = active_probability**counts * (1.0 - active_probability) ** (feature_count - counts)
    probs /= probs.sum()
    return codes, probs


def fit_one(
    seed: int,
    feature_count: int,
    bottleneck: int,
    n_train: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    set_seed(args.seed + seed * 100003 + feature_count * 1009 + bottleneck * 17)
    rng = np.random.default_rng(args.seed + seed)
    maximum_train = max(args.sample_sizes)
    x_train_all = (rng.random((maximum_train, feature_count)) < args.active_probability).astype(float)
    x_test = (rng.random((args.n_test, feature_count)) < args.active_probability).astype(float)
    beta = rng.normal(size=feature_count) / np.sqrt(feature_count)
    y_train_all = x_train_all @ beta
    y_test = x_test @ beta
    x_train = torch.as_tensor(x_train_all[:n_train])
    x_test_t = torch.as_tensor(x_test)

    model = SparseAutoencoder(feature_count, bottleneck)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    for _ in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x_train) - x_train) ** 2)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pre_train = model.hidden_pre(x_train).cpu().numpy()
        hidden_train = model.hidden(x_train).cpu().numpy()
        hidden_test = model.hidden(x_test_t).cpu().numpy()
        reconstruction_mse = float(torch.mean((model(x_test_t) - x_test_t) ** 2).item())

    directions = np.zeros((bottleneck, feature_count))
    with torch.no_grad():
        baseline = model.hidden(x_test_t)
        for feature in range(feature_count):
            absent = x_test[:, feature] == 0
            if not np.any(absent):
                continue
            intervened = x_test[absent].copy()
            intervened[:, feature] = 1.0
            effect = model.hidden(torch.as_tensor(intervened)) - baseline[torch.as_tensor(absent)]
            directions[:, feature] = effect.mean(dim=0).cpu().numpy()

    # Freeze the learned encoder and evaluate the specified linear task head.
    readout = np.linalg.pinv(hidden_train) @ y_train_all[:n_train]
    train_prediction = hidden_train @ readout
    test_prediction = hidden_test @ readout
    hidden_support_t, probabilities = binary_codebook(feature_count, args.active_probability)
    with torch.no_grad():
        hidden_support = model.hidden(torch.as_tensor(hidden_support_t)).cpu().numpy()
    target_support = hidden_support_t @ beta
    bayes_mse, aliased_mass = representation_bayes_mse(
        hidden_support, target_support, probabilities, atol=args.collision_tolerance
    )
    gad = finite_support_gad(
        hidden_train,
        y_train_all[:n_train],
        hidden_support,
        target_support,
        probabilities,
    )
    return {
        "seed": float(seed),
        "feature_count": float(feature_count),
        "bottleneck": float(bottleneck),
        "sample_size": float(n_train),
        "feature_pressure": float(feature_count / bottleneck),
        "active_probability": args.active_probability,
        "autoencoder_test_mse": reconstruction_mse,
        "probe_train_mse": float(np.mean((train_prediction - y_train_all[:n_train]) ** 2)),
        "probe_test_mse": float(np.mean((test_prediction - y_test) ** 2)),
        "representation_bayes_mse": bayes_mse,
        "aliased_probability_mass": aliased_mass,
        **feature_direction_metrics(directions),
        **covariance_pair_metrics(pre_train),
        **gad,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no results to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(means: list[dict[str, float]], output: Path) -> None:
    configs = sorted({(row["feature_count"], row["bottleneck"]) for row in means})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    panels = [
        ("probe_test_mse", "held-out linear-probe MSE"),
        ("aliasing", "GAD alias component MSE"),
        ("feature_direction_coherence", "feature-direction coherence"),
        ("post_cov_stable_rank", "post-ReLU sample Gram stable rank"),
    ]
    for axis, (metric, label) in zip(axes.flat, panels):
        for feature_count, bottleneck in configs:
            selected = [row for row in means if row["feature_count"] == feature_count and row["bottleneck"] == bottleneck]
            selected.sort(key=lambda row: row["sample_size"])
            axis.plot(
                [row["sample_size"] for row in selected],
                [row[metric] for row in selected],
                marker="o",
                label=f"p={int(feature_count)}, m={int(bottleneck)}",
            )
        axis.set_xlabel("training sample count")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-counts", type=parse_ints, default=parse_ints("8,12"))
    parser.add_argument("--bottlenecks", type=parse_ints, default=parse_ints("3,4"))
    parser.add_argument("--sample-sizes", type=parse_ints, default=parse_ints("32,64,128"))
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--active-probability", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--collision-tolerance", type=float, default=1e-9)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.num_seeds < 2 or args.n_test < 1 or args.epochs < 1:
        raise ValueError("num-seeds must be >=2; n-test and epochs must be positive")
    if any(p > 16 for p in args.feature_counts):
        raise ValueError("feature counts above 16 exceed the exact codebook limit")
    rows: list[dict[str, float]] = []
    for p in args.feature_counts:
        for m in args.bottlenecks:
            if m >= p:
                continue
            for n_train in args.sample_sizes:
                for seed in range(args.num_seeds):
                    rows.append(fit_one(seed, p, m, n_train, args))

    keys = ("feature_count", "bottleneck", "sample_size")
    means, stds, cis = aggregate_seed_rows(
        rows,
        group_keys=keys,
        excluded_keys={"seed", *keys},
        n_bootstrap=1000,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "raw.csv", rows)
    write_csv(args.output_dir / "mean.csv", means)
    write_csv(args.output_dir / "std.csv", stds)
    write_csv(args.output_dir / "ci95.csv", cis)
    write_csv(args.output_dir / "config.csv", [{
        "feature_counts": ",".join(map(str, args.feature_counts)),
        "bottlenecks": ",".join(map(str, args.bottlenecks)),
        "sample_sizes": ",".join(map(str, args.sample_sizes)),
        "num_seeds": args.num_seeds,
        "n_test": args.n_test,
        "epochs": args.epochs,
        "active_probability": args.active_probability,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "collision_tolerance": args.collision_tolerance,
        "seed": args.seed,
    }])
    plot_summary(means, args.output_dir / "trained_sparse_feature_geometry.png")
    print(f"Wrote {len(rows)} trained static-geometry runs to {args.output_dir}")


if __name__ == "__main__":
    main()
