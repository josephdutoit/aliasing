"""Run parallel feature-learning MLP aliasing experiments.

Each task fixes a data seed and student width, trains a two-layer ReLU MLP,
and records geometry diagnostics at several checkpoints.  The outer process
parallelism is over independent (seed, width) tasks; each worker should use a
single BLAS/PyTorch thread on a cluster node.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np

from aliasing.mlp import (
    TwoLayerMLP,
    hidden_aliasing_metrics,
    make_teacher_problem,
    seed_everything,
)


def parse_positive_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def parse_nonnegative_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of nonnegative integers"
        )
    return values


def _set_worker_threads(torch_threads: int) -> None:
    # These defaults prevent oversubscription when a scheduler launches many
    # Python processes.  Users can still export explicit values before launch.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    import torch

    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting inter-op threads only before parallel work;
        # this is harmless when a worker receives multiple tasks.
        pass


def train_one(seed: int, width: int, args: argparse.Namespace) -> list[dict[str, float]]:
    _set_worker_threads(args.torch_threads)
    import torch

    x_train, x_test, y_train, y_test = make_teacher_problem(
        seed=seed,
        n_train=args.n_train,
        n_test=args.n_test,
        input_dim=args.input_dim,
        teacher_width=args.teacher_width,
        noise_std=args.noise_std,
    )

    # Width-dependent initialization is reproducible while the data remain
    # identical across widths for a fixed seed.
    seed_everything(seed + 10_000_019 * width)
    model = TwoLayerMLP(args.input_dim, width)
    initial_fc1_weight = model.fc1.weight.detach().clone()
    initial_fc1_bias = model.fc1.bias.detach().clone()
    initial_fc2_weight = model.fc2.weight.detach().clone()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    checkpoints = set(args.checkpoints)
    rows: list[dict[str, float]] = []

    for step in range(args.steps + 1):
        if step in checkpoints:
            metrics = hidden_aliasing_metrics(
                model,
                x_train,
                x_test,
                y_train,
                y_test,
                initial_fc1_weight,
                initial_fc1_bias,
                initial_fc2_weight,
                interpolation_threshold=args.interpolation_threshold,
            )
            rows.append({"seed": float(seed), "width": float(width), "step": float(step), **metrics})

        if step == args.steps:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x_train) - y_train) ** 2)
        loss.backward()
        optimizer.step()
    return rows


def _worker(task: tuple[int, int, argparse.Namespace]) -> list[dict[str, float]]:
    seed, width, args = task
    return train_one(seed, width, args)


def flatten(groups: Iterable[list[dict[str, float]]]) -> list[dict[str, float]]:
    return [row for group in groups for row in group]


def write_rows(path: Path, rows: list[Mapping[str, float]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, float]]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    grouped: dict[tuple[float, float], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[(row["width"], row["step"])].append(row)
    mean_rows: list[dict[str, float]] = []
    std_rows: list[dict[str, float]] = []
    excluded = {"seed", "width", "step"}
    for (width, step), group in sorted(grouped.items()):
        mean_row = {"width": width, "step": step}
        std_row = {"width": width, "step": step}
        for key in group[0]:
            if key in excluded:
                continue
            values = np.asarray([row[key] for row in group], dtype=float)
            mean_row[key] = float(np.nanmean(values))
            std_row[key] = float(np.nanstd(values))
        mean_rows.append(mean_row)
        std_rows.append(std_row)
    return mean_rows, std_rows


def plot_summary(mean_rows: list[dict[str, float]], output: Path) -> None:
    widths = sorted({row["width"] for row in mean_rows})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [
        ("test_mse", "test MSE", True),
        ("train_mse", "train MSE", True),
        ("test_null_response_rms", "test null-head response RMS", True),
        ("feature_drift", "hidden-weight drift", False),
    ]
    for axis, (key, label, log_y) in zip(axes.flat, panels):
        for width in widths:
            selected = [row for row in mean_rows if row["width"] == width]
            selected.sort(key=lambda row: row["step"])
            x = [row["step"] for row in selected]
            y = [max(row[key], np.finfo(float).tiny) for row in selected]
            axis.plot(x, y, marker="o", label=f"width={int(width)}")
        axis.set_xlabel("optimization step")
        axis.set_ylabel(label)
        if log_y:
            axis.set_yscale("log")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=parse_positive_int_list, default=parse_positive_int_list("16,32,64,128,256"))
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--checkpoints", type=parse_nonnegative_int_list, default=parse_nonnegative_int_list("0,10,30,100,300,1000,3000,5000"))
    parser.add_argument("--input-dim", type=int, default=20)
    parser.add_argument("--teacher-width", type=int, default=64)
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--n-test", type=int, default=4096)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--interpolation-threshold", type=float, default=1e-10)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_seeds < 1 or args.num_workers is not None and args.num_workers < 1:
        raise ValueError("num-seeds and num-workers must be positive")
    if args.steps < 0 or args.torch_threads < 1:
        raise ValueError("steps must be nonnegative and torch-threads must be positive")
    if args.steps not in args.checkpoints:
        args.checkpoints = sorted(set(args.checkpoints + [args.steps]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(seed, width, args) for seed in range(args.num_seeds) for width in args.widths]
    context = mp.get_context("spawn")
    with context.Pool(processes=args.num_workers) as pool:
        rows = flatten(pool.map(_worker, tasks))

    rows.sort(key=lambda row: (row["width"], row["step"], row["seed"]))
    write_rows(args.output_dir / "raw.csv", rows)
    mean_rows, std_rows = aggregate(rows)
    write_rows(args.output_dir / "mean.csv", mean_rows)
    write_rows(args.output_dir / "std.csv", std_rows)
    plot_summary(mean_rows, args.output_dir / "mlp_aliasing.png")


if __name__ == "__main__":
    main()
