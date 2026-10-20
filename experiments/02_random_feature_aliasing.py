from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import matplotlib.pyplot as plt
import numpy as np

from aliasing.linear import evaluate_width, run_sweep
from aliasing.random_features import make_random_feature_problem
from aliasing.summaries import aggregate_seed_rows, adjacent_paired_contrasts


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


def write_csv(path: Path, rows: List[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: List[Dict[str, float]]):
    return aggregate_seed_rows(
        rows,
        group_keys=("sweep_value",),
        excluded_keys={"seed", "sweep_value"},
    )


def plot_means(
    mean_rows: List[Dict[str, float]],
    ci_rows: List[Dict[str, float]],
    args: argparse.Namespace,
) -> None:
    x = np.array([row["sweep_value"] for row in mean_rows])
    tiny = np.finfo(float).tiny
    ci_by_x = {row["sweep_value"]: row for row in ci_rows}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for axis, key, title, color in (
        (axes[0], "test_mse", "Test MSE", "tab:blue"),
        (axes[1], "alias_operator_rms_norm", "RMS alias-operator norm", "tab:orange"),
        (axes[2], "aliasing", "Aliasing energy", "tab:green"),
    ):
        mean = np.asarray([row[key] for row in mean_rows])
        low = np.asarray([ci_by_x[value][f"{key}_ci95_low"] for value in x])
        high = np.asarray([ci_by_x[value][f"{key}_ci95_high"] for value in x])
        axis.plot(x, np.maximum(mean, tiny), marker=".", color=color)
        axis.fill_between(x, np.maximum(low, tiny), np.maximum(high, tiny), color=color, alpha=0.2)
        axis.set_yscale("log")
        axis.set_title(title)
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
    mean_rows, std_rows, ci_rows = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "raw.csv", rows)
    write_csv(args.output_dir / "mean.csv", mean_rows)
    write_csv(args.output_dir / "std.csv", std_rows)
    write_csv(args.output_dir / "ci95.csv", ci_rows)
    if args.sweep == "featurewise":
        contrasts = adjacent_paired_contrasts(
            rows,
            level_key="sweep_value",
            match_keys=("seed",),
            excluded_keys={"seed", "sweep_value", "d"},
        )
        write_csv(args.output_dir / "paired_contrasts.csv", contrasts)
    plot_means(mean_rows, ci_rows, args)
    print(f"Wrote {len(rows)} raw rows to {args.output_dir}")


if __name__ == "__main__":
    main()

