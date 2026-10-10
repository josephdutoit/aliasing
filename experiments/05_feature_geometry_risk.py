from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replicated test of finite-dimensional GAD risk predictions across "
            "feature covariances and nested training sample sizes."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=4096)
    parser.add_argument("--n-directions", type=int, default=32)
    parser.add_argument("--p", type=int, default=60)
    parser.add_argument(
        "--sample-sizes",
        type=parse_int_list,
        default=parse_int_list("6,12,18,24,30,36,42,48,54,60"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/chapter2_geometry_risk")
    )
    return parser.parse_args()


def null_basis(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    """Return an orthonormal basis for the numerical right nullspace."""
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    if singular_values.size == 0:
        return np.eye(matrix.shape[1]), 0
    tolerance = max(matrix.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    return vh[rank:].T, rank


def covariance_scenarios(p: int) -> dict[str, np.ndarray]:
    power_law = np.arange(1, p + 1, dtype=float) ** -1.0
    low_rank = np.zeros(p)
    low_rank[: max(2, p // 5)] = 1.0
    return {
        "isotropic": np.ones(p),
        "power_law": power_law,
        "low_rank": low_rank,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_replicates(args: argparse.Namespace) -> list[dict[str, object]]:
    rng = np.random.default_rng(args.seed)
    raw: list[dict[str, object]] = []
    covariances = covariance_scenarios(args.p)
    radius = 1.0

    for scenario, eigenvalues in covariances.items():
        root = np.diag(np.sqrt(eigenvalues))
        for replicate in range(args.replicates):
            # The same nested design and held-out sample are reused across n.
            x_all = rng.normal(size=(args.p, args.p)) @ root
            x_test = rng.normal(size=(args.n_test, args.p)) @ root

            for n_train in args.sample_sizes:
                x_train = x_all[:n_train]
                basis, rank = null_basis(x_train)
                nullity = basis.shape[1]
                if nullity == 0:
                    raw.append(
                        {
                            "scenario": scenario,
                            "replicate": replicate,
                            "n_train": n_train,
                            "rank": rank,
                            "nullity": 0,
                            "operator_norm": 0.0,
                            "trace": 0.0,
                            "top_expected_risk": 0.0,
                            "top_test_risk": 0.0,
                            "random_trace_prediction": 0.0,
                            "random_quadratic_risk": 0.0,
                            "random_test_risk": 0.0,
                            "train_null_residual": 0.0,
                        }
                    )
                    continue

                # Work in nullspace coordinates to avoid forming a dense
                # p-by-p projector for every replicate and sample size.
                restricted_covariance = basis.T @ (eigenvalues[:, None] * basis)
                values, vectors = np.linalg.eigh(restricted_covariance)
                values = np.maximum(values, 0.0)
                operator_norm = float(values[-1])
                trace = float(values.sum())

                top_direction = basis @ vectors[:, -1]
                top_delta = radius * top_direction
                top_expected = 0.5 * float(top_delta @ (eigenvalues * top_delta))
                top_test = 0.5 * float(np.mean((x_test @ top_delta) ** 2))

                coefficients = rng.normal(size=(nullity, args.n_directions))
                coefficients /= np.linalg.norm(coefficients, axis=0, keepdims=True)
                directions = basis @ coefficients
                deltas = radius * directions
                random_expected = 0.5 * np.sum(
                    deltas * (eigenvalues[:, None] * deltas), axis=0
                )
                random_test = 0.5 * np.mean((x_test @ deltas) ** 2, axis=0)
                random_trace_prediction = 0.5 * radius**2 * trace / nullity
                train_null_residual = float(np.linalg.norm(x_train @ deltas, ord=2))

                raw.append(
                    {
                        "scenario": scenario,
                        "replicate": replicate,
                        "n_train": n_train,
                        "rank": rank,
                        "nullity": nullity,
                        "operator_norm": operator_norm,
                        "trace": trace,
                        "top_expected_risk": top_expected,
                        "top_test_risk": top_test,
                        "random_trace_prediction": random_trace_prediction,
                        "random_quadratic_risk": float(np.mean(random_expected)),
                        "random_test_risk": float(np.mean(random_test)),
                        "train_null_residual": train_null_residual,
                    }
                )
    return raw


def summarize(raw: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in raw:
        key = (str(row["scenario"]), int(row["n_train"]))
        groups.setdefault(key, []).append(row)

    metrics = (
        "rank",
        "nullity",
        "operator_norm",
        "trace",
        "top_expected_risk",
        "top_test_risk",
        "random_trace_prediction",
        "random_quadratic_risk",
        "random_test_risk",
        "train_null_residual",
    )
    summary: list[dict[str, object]] = []
    for (scenario, n_train), rows in sorted(groups.items()):
        result: dict[str, object] = {
            "scenario": scenario,
            "n_train": n_train,
            "replicates": len(rows),
        }
        for metric in metrics:
            values = np.array([float(row[metric]) for row in rows])
            mean = float(values.mean())
            standard_error = (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
            result[f"{metric}_mean"] = mean
            result[f"{metric}_se"] = standard_error
            result[f"{metric}_ci95_low"] = mean - 1.96 * standard_error
            result[f"{metric}_ci95_high"] = mean + 1.96 * standard_error
        summary.append(result)
    return summary


def plot_summary(summary: list[dict[str, object]], output: Path) -> None:
    colors = {"isotropic": "tab:blue", "power_law": "tab:orange", "low_rank": "tab:green"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    for scenario, color in colors.items():
        rows = [row for row in summary if row["scenario"] == scenario]
        rows.sort(key=lambda row: int(row["n_train"]))
        n = np.array([int(row["n_train"]) for row in rows])
        for axis, metric, label in (
            (axes[0], "trace", "trace: average aliasing"),
            (axes[1], "operator_norm", "operator norm: worst-case aliasing"),
        ):
            mean = np.array([float(row[f"{metric}_mean"]) for row in rows])
            low = np.array([float(row[f"{metric}_ci95_low"]) for row in rows])
            high = np.array([float(row[f"{metric}_ci95_high"]) for row in rows])
            axis.plot(n, mean, marker="o", color=color, label=scenario)
            axis.fill_between(n, low, high, color=color, alpha=0.15)
            axis.set_xlabel("training sample size n")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)

    axes[0].set_title("Average aliasing")
    axes[0].legend()
    axes[1].set_title("Worst-case aliasing")
    axes[1].legend()

    markers = {"isotropic": "o", "power_law": "s", "low_rank": "^"}
    for scenario, color in colors.items():
        rows = [row for row in summary if row["scenario"] == scenario]
        top_expected = np.array([float(row["top_expected_risk_mean"]) for row in rows])
        top_test = np.array([float(row["top_test_risk_mean"]) for row in rows])
        random_expected = np.array([float(row["random_trace_prediction_mean"]) for row in rows])
        random_test = np.array([float(row["random_test_risk_mean"]) for row in rows])
        axes[2].scatter(
            top_expected,
            top_test,
            color=color,
            marker=markers[scenario],
            label=f"{scenario}: top direction",
        )
        axes[2].scatter(
            random_expected,
            random_test,
            color=color,
            marker=markers[scenario],
            facecolors="none",
            label=f"{scenario}: random directions",
        )
    upper = max(
        max(float(row["top_test_risk_mean"]) for row in summary),
        max(float(row["top_expected_risk_mean"]) for row in summary),
        1e-12,
    )
    axes[2].plot([0, upper], [0, upper], "k--", linewidth=1)
    axes[2].set_xlabel("GAD-predicted excess risk")
    axes[2].set_ylabel("measured held-out excess risk")
    axes[2].set_title("Risk prediction check")
    axes[2].legend(fontsize=7)
    fig.suptitle("Feature geometry and train-null population risk")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.replicates < 2:
        raise ValueError("--replicates must be at least 2 to report uncertainty")
    if args.n_test < 1 or args.n_directions < 1 or args.p < 2:
        raise ValueError("n-test and n-directions must be positive and p must be at least 2")
    if any(n < 1 or n > args.p for n in args.sample_sizes):
        raise ValueError("every sample size must be between 1 and p")
    if len(set(args.sample_sizes)) != len(args.sample_sizes):
        raise ValueError("sample sizes must be unique")
    args.sample_sizes = sorted(args.sample_sizes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = run_replicates(args)
    summary = summarize(raw)
    write_csv(args.output_dir / "replicates.csv", raw)
    write_csv(args.output_dir / "summary.csv", summary)
    plot_summary(summary, args.output_dir / "feature_geometry_risk.png")

    max_residual = max(float(row["train_null_residual"]) for row in raw)
    print(f"Wrote {len(raw)} replicate rows to {args.output_dir}")
    print(f"Maximum train-null residual: {max_residual:.3e}")
    print(f"Wrote {args.output_dir / 'feature_geometry_risk.png'}")


if __name__ == "__main__":
    main()
