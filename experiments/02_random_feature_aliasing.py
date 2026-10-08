from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

from aliasing.linear import evaluate_width, run_sweep
from aliasing.random_features import make_random_feature_problem


def parse_int_list(value: str) -> List[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", choices=("featurewise", "samplewise"), required=True)
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--sample-sizes", type=parse_int_list,
                        default=parse_int_list("8,16,24,32,40,48,56,64,72,80,88"))
    parser.add_argument("--modeled-features", type=int, default=40)
    parser.add_argument("--full-features", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--n-test", type=int, default=4096)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _one_featurewise(seed: int, args: argparse.Namespace) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    _, _, z_train, z_test, y_train, theta = make_random_feature_problem(
        rng=rng,
        n_train=args.n_train,
        n_test=args.n_test,
        input_dim=args.input_dim,
        full_features=args.full_features,
        noise_std=args.noise_std,
    )
    rows = run_sweep(
        x_train=z_train,
        x_test=z_test,
        theta_full=theta,
        y_train=y_train,
        widths=range(1, args.full_features + 1),
    )
    for row in rows:
        row.update({"seed": float(seed), "sweep_value": row["d"]})
    return rows


def _one_samplewise(seed: int, args: argparse.Namespace) -> List[Dict[str, float]]:
    rows = []
    for sample_size in args.sample_sizes:
        rng = np.random.default_rng(seed * 1_000_003 + sample_size)
        _, _, z_train, z_test, y_train, theta = make_random_feature_problem(
            rng=rng,
            n_train=sample_size,
            n_test=args.n_test,
            input_dim=args.input_dim,
            full_features=args.full_features,
            noise_std=args.noise_std,
        )
        row = evaluate_width(
            x_train=z_train,
            x_test=z_test,
            theta_full=theta,
            y_train=y_train,
            d=args.modeled_features,
        )
        row.update({"seed": float(seed), "sweep_value": float(sample_size)})
        rows.append(row)
    return rows


def _worker(task: tuple[str, int, argparse.Namespace]) -> List[Dict[str, float]]:
    sweep, seed, args = task
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    return _one_featurewise(seed, args) if sweep == "featurewise" else _one_samplewise(seed, args)


def flatten(rows: Iterable[List[Dict[str, float]]]) -> List[Dict[str, float]]:
    return [row for batch in rows for row in batch]


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: List[Dict[str, float]]) -> tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sweep_value"]].append(row)

    mean_rows = []
    std_rows = []
    for value in sorted(grouped):
        group = grouped[value]
        numeric_keys = [
            key for key, item in group[0].items()
            if key not in {"seed", "sweep_value"}
        ]
        mean_row = {"sweep_value": value, "num_runs": float(len(group))}
        std_row = {"sweep_value": value, "num_runs": float(len(group))}
        for key in numeric_keys:
            values = np.array([item[key] for item in group], dtype=float)
            mean_row[key] = float(np.nanmean(values))
            std_row[key] = float(np.nanstd(values))
        mean_rows.append(mean_row)
        std_rows.append(std_row)
    return mean_rows, std_rows


def plot_means(mean_rows: List[Dict[str, float]], args: argparse.Namespace) -> None:
    x = np.array([row["sweep_value"] for row in mean_rows])
    test_mse = np.array([row["test_mse"] for row in mean_rows])
    alias_norm = np.array([row["alias_operator_norm"] for row in mean_rows])
    aliasing = np.array([row["aliasing"] for row in mean_rows])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    axes[0].plot(x, test_mse, marker=".")
    axes[0].set_yscale("log")
    axes[0].set_title("Test MSE")
    axes[1].plot(x, alias_norm, marker=".", color="tab:orange")
    axes[1].set_yscale("log")
    axes[1].set_title("Aliasing operator norm")
    axes[2].plot(x, aliasing, marker=".", color="tab:green")
    axes[2].set_yscale("log")
    axes[2].set_title("Aliasing energy")
    xlabel = "modeled feature count" if args.sweep == "featurewise" else "training sample count"
    for axis in axes:
        axis.set_xlabel(xlabel)
        axis.grid(alpha=0.25)
    fig.suptitle(f"Random-feature ReLU: {args.sweep} sweep")
    fig.savefig(args.output_dir / "random_feature_aliasing.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be positive")
    if args.num_workers is None:
        args.num_workers = min(args.num_seeds, os.cpu_count() or 1)

    tasks = [(args.sweep, seed, args) for seed in range(args.num_seeds)]
    context = mp.get_context("spawn")
    with context.Pool(processes=args.num_workers) as pool:
        batches = pool.map(_worker, tasks)

    rows = flatten(batches)
    mean_rows, std_rows = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "raw.csv", rows)
    write_csv(args.output_dir / "mean.csv", mean_rows)
    write_csv(args.output_dir / "std.csv", std_rows)
    plot_means(mean_rows, args)
    print(f"Wrote {len(rows)} raw rows to {args.output_dir}")


if __name__ == "__main__":
    main()
