from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aliasing.linear import run_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--n-test", type=int, default=4096)
    parser.add_argument("--n-full", type=int, default=128)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/linear_baseline")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_full < 2:
        raise ValueError("--n-full must be at least 2")
    if args.n_train < 1 or args.n_test < 1:
        raise ValueError("sample counts must be positive")

    rng = np.random.default_rng(args.seed)
    x_train = rng.normal(size=(args.n_train, args.n_full))
    x_test = rng.normal(size=(args.n_test, args.n_full))

    theta_full = rng.normal(size=args.n_full)
    theta_full /= np.sqrt(np.arange(1, args.n_full + 1))
    theta_full /= np.linalg.norm(theta_full)

    y_clean = x_train @ theta_full
    y_train = y_clean + args.noise_std * rng.normal(size=args.n_train)

    rows = run_sweep(
        x_train=x_train,
        x_test=x_test,
        theta_full=theta_full,
        y_train=y_train,
        widths=range(1, args.n_full + 1),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    widths = np.array([row["d"] for row in rows])
    test_mse = np.array([row["test_mse"] for row in rows])
    alias_norm = np.array([row["alias_operator_norm"] for row in rows])
    model = np.array([row["model_insufficiency"] for row in rows])
    data = np.array([row["data_insufficiency"] for row in rows])
    alias = np.array([row["aliasing"] for row in rows])
    noise = np.array([row["fitted_noise"] for row in rows])
    cross = np.array([row["component_cross_term"] for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(widths, test_mse, marker=".", label="test MSE")
    axes[0, 0].axvline(args.n_train, color="black", linestyle="--", alpha=0.5)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Risk across modeled capacity")
    axes[0, 0].set_xlabel("modeled feature count d")
    axes[0, 0].legend()

    axes[0, 1].plot(widths, alias_norm, marker=".", color="tab:orange")
    axes[0, 1].axvline(args.n_train, color="black", linestyle="--", alpha=0.5)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Aliasing operator norm")
    axes[0, 1].set_xlabel("modeled feature count d")

    axes[1, 0].plot(widths, model, label="model insufficiency")
    axes[1, 0].plot(widths, data, label="data insufficiency")
    axes[1, 0].plot(widths, alias, label="aliasing")
    axes[1, 0].plot(widths, noise, label="fitted noise")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Component energies")
    axes[1, 0].set_xlabel("modeled feature count d")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(widths, cross, label="cross-term")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Non-orthogonality diagnostic")
    axes[1, 1].set_xlabel("modeled feature count d")
    axes[1, 1].legend()

    fig.suptitle("Fixed-feature generalized aliasing baseline")
    fig.savefig(args.output_dir / "linear_aliasing.png", dpi=160)
    plt.close(fig)

    print(f"Wrote {metrics_path}")
    print(f"Wrote {args.output_dir / 'linear_aliasing.png'}")


if __name__ == "__main__":
    main()
