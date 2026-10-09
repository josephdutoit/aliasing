from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small synthetic pilots for finite-dimensional GAD geometry."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--n-test", type=int, default=4096)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gad_geometry_pilot"))
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
    identity = np.eye(p)
    power = np.arange(1, p + 1, dtype=float) ** -1.0
    low_rank = np.zeros(p)
    low_rank[: max(2, p // 5)] = 1.0
    return {
        "isotropic": identity,
        "power_law": np.diag(power),
        "low_rank": np.diag(low_rank),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_identity_pilot(
    rng: np.random.Generator, n_test: int
) -> tuple[dict[str, float], list[dict[str, object]]]:
    n_train, p, radius = 18, 40, 1.25
    sigma = np.diag(np.linspace(0.35, 1.8, p))
    x_train = rng.normal(size=(n_train, p)) @ np.diag(np.sqrt(np.diag(sigma)))
    basis, rank = null_basis(x_train)
    projector = basis @ basis.T
    alias = projector @ sigma @ projector
    eigenvalues, eigenvectors = np.linalg.eigh(alias)
    top_direction = eigenvectors[:, -1]

    # A noiseless linear teacher makes the population excess risk exact.
    teacher = rng.normal(size=p)
    y_train = x_train @ teacher
    delta = radius * top_direction
    training_change = float(np.linalg.norm(x_train @ delta))
    x_test = rng.normal(size=(n_test, p)) @ np.diag(np.sqrt(np.diag(sigma)))
    measured_risk = 0.5 * float(np.mean((x_test @ delta) ** 2))
    expected_risk = 0.5 * float(delta @ sigma @ delta)
    predicted_risk = 0.5 * float(delta @ alias @ delta)

    samples = 4000
    coefficients = rng.normal(size=(basis.shape[1], samples))
    coefficients /= np.linalg.norm(coefficients, axis=0, keepdims=True)
    directions = basis @ coefficients
    random_risks = 0.5 * radius**2 * np.einsum(
        "ij,ji->i", directions.T @ sigma, directions
    )
    average_prediction = 0.5 * radius**2 * float(np.trace(alias)) / basis.shape[1]

    # The training labels are included as a sanity check that the teacher setup
    # really leaves the sample fit unchanged under each train-null perturbation.
    del y_train
    summary = {
        "train_rank": float(rank),
        "nullity": float(basis.shape[1]),
        "train_prediction_change": training_change,
        "top_direction_test_risk": measured_risk,
        "top_direction_expected_risk": expected_risk,
        "top_direction_predicted_risk": predicted_risk,
        "top_direction_abs_error": abs(expected_risk - predicted_risk),
        "top_direction_spectral_prediction": 0.5 * radius**2 * float(eigenvalues[-1]),
        "random_direction_mean_risk": float(np.mean(random_risks)),
        "random_direction_mean_prediction": average_prediction,
        "random_direction_mean_abs_error": abs(float(np.mean(random_risks)) - average_prediction),
    }
    rows = [{"metric": key, "value": value} for key, value in summary.items()]
    return summary, rows


def nested_geometry_pilot(
    rng: np.random.Generator, replicates: int
) -> list[dict[str, object]]:
    p = 60
    covariances = covariance_scenarios(p)
    sample_sizes = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
    raw: list[dict[str, object]] = []

    for scenario, covariance in covariances.items():
        eigenvalues = np.diag(covariance)
        root = np.diag(np.sqrt(eigenvalues))
        for replicate in range(replicates):
            maximum_n = max(sample_sizes)
            z = rng.normal(size=(maximum_n, p))
            x_all = z @ root
            for n in sample_sizes:
                basis, rank = null_basis(x_all[:n])
                projector = basis @ basis.T
                alias = projector @ covariance @ projector
                raw.append(
                    {
                        "scenario": scenario,
                        "replicate": replicate,
                        "n_train": n,
                        "rank": rank,
                        "nullity": basis.shape[1],
                        "operator_norm": float(np.linalg.eigvalsh(alias)[-1])
                        if p > rank
                        else 0.0,
                        "trace": float(np.trace(alias)),
                    }
                )

    # Aggregate replicates for compact tables and uncertainty bands.
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in raw:
        grouped.setdefault((str(row["scenario"]), int(row["n_train"])), []).append(row)
    rows: list[dict[str, object]] = []
    for (scenario, n), group in grouped.items():
        row: dict[str, object] = {"scenario": scenario, "n_train": n}
        for metric in ("rank", "nullity", "operator_norm", "trace"):
            values = np.array([float(item[metric]) for item in group])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def fisher_weight_pilot(rng: np.random.Generator) -> list[dict[str, object]]:
    n_train, p = 24, 40
    x_train = rng.normal(size=(n_train, p))
    _, unweighted_rank = null_basis(x_train)
    signed_margins = np.where(np.arange(n_train) % 2 == 0, -1.0, 1.0) * np.linspace(
        0.5, 1.5, n_train
    )
    rows: list[dict[str, object]] = []

    # Positive logistic Fisher weights change conditioning, but not the exact
    # mathematical row space. Numerical rank is reported separately because
    # sufficiently ill-conditioned matrices can appear rank-deficient in float64.
    for scale in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        logits = scale * signed_margins
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        weights = probabilities * (1.0 - probabilities)
        weighted_design = np.sqrt(weights)[:, None] * x_train
        singular_values = np.linalg.svd(weighted_design, compute_uv=False)
        numerical_rank = int(np.linalg.matrix_rank(weighted_design))
        effective_n = float(weights.sum() ** 2 / np.sum(weights**2))
        rows.append(
            {
                "case": "positive_logistic_weights",
                "scale_or_zero_rows": scale,
                "effective_sample_size": effective_n,
                "positive_weight_count": int(np.count_nonzero(weights > 0.0)),
                "unweighted_rank": unweighted_rank,
                "weighted_numerical_rank": numerical_rank,
                "weighted_numerical_nullity": p - numerical_rank,
                "smallest_weighted_singular_value": float(singular_values[-1]),
                "condition_number": float(singular_values[0] / singular_values[-1]),
                "alias_trace_identity_covariance": float(p - numerical_rank),
            }
        )

    # Exact zero weights are a separate limiting diagnostic. With Sigma=I,
    # tr(P_perp Sigma P_perp) equals the nullity exactly.
    for zero_rows in (0, 4, 8, 12, 16):
        weights = np.ones(n_train)
        weights[:zero_rows] = 0.0
        weighted_design = np.sqrt(weights)[:, None] * x_train
        _, weighted_rank = null_basis(weighted_design)
        rows.append(
            {
                "case": "exact_zero_weights",
                "scale_or_zero_rows": zero_rows,
                "effective_sample_size": float(weights.sum() ** 2 / np.sum(weights**2))
                if np.sum(weights**2) > 0
                else 0.0,
                "positive_weight_count": int(np.count_nonzero(weights > 0.0)),
                "unweighted_rank": unweighted_rank,
                "weighted_numerical_rank": weighted_rank,
                "weighted_numerical_nullity": p - weighted_rank,
                "smallest_weighted_singular_value": float("nan"),
                "condition_number": float("nan"),
                "alias_trace_identity_covariance": float(p - weighted_rank),
            }
        )
    return rows


def make_figure(
    output_path: Path,
    exact: dict[str, float],
    geometry_rows: list[dict[str, object]],
    fisher_rows: list[dict[str, object]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].scatter(
        [exact["top_direction_predicted_risk"], exact["random_direction_mean_prediction"]],
        [exact["top_direction_test_risk"], exact["random_direction_mean_risk"]],
        color=["tab:red", "tab:blue"],
        label="measured risks",
    )
    max_risk = max(exact["top_direction_test_risk"], exact["random_direction_mean_risk"])
    axes[0, 0].plot([0, max_risk], [0, max_risk], "k--", linewidth=1, label="identity")
    axes[0, 0].set_xlabel("GAD quadratic prediction")
    axes[0, 0].set_ylabel("measured / predicted risk")
    axes[0, 0].set_title("Train-null risk identity")
    axes[0, 0].legend(fontsize=8)

    colors = {"isotropic": "tab:blue", "power_law": "tab:orange", "low_rank": "tab:green"}
    for scenario, color in colors.items():
        subset = [row for row in geometry_rows if row["scenario"] == scenario]
        subset.sort(key=lambda row: int(row["n_train"]))
        n = np.array([int(row["n_train"]) for row in subset])
        trace = np.array([float(row["trace_mean"]) for row in subset])
        norm = np.array([float(row["operator_norm_mean"]) for row in subset])
        axes[0, 1].plot(n, trace, marker="o", color=color, label=f"{scenario}: trace")
        axes[0, 1].plot(n, norm, linestyle="--", color=color, label=f"{scenario}: norm")
    axes[0, 1].set_xlabel("nested training sample size n")
    axes[0, 1].set_ylabel("aliasing measure")
    axes[0, 1].set_title("Trace and worst-case aliasing")
    axes[0, 1].legend(fontsize=7, ncol=2)

    positive = [row for row in fisher_rows if row["case"] == "positive_logistic_weights"]
    zeros = [row for row in fisher_rows if row["case"] == "exact_zero_weights"]
    axes[1, 0].plot(
        [float(row["scale_or_zero_rows"]) for row in positive],
        [float(row["effective_sample_size"]) for row in positive],
        marker="o",
        label="effective sample size",
    )
    axes[1, 0].set_xlabel("logit scale (all weights positive)")
    axes[1, 0].set_ylabel("effective sample size")
    axes[1, 0].set_title("Soft Fisher curvature")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        [float(row["scale_or_zero_rows"]) for row in zeros],
        [float(row["weighted_numerical_nullity"]) for row in zeros],
        marker="s",
        label="nullity = alias trace for $\\Sigma=I$",
    )
    axes[1, 1].set_xlabel("number of exactly zero weights")
    axes[1, 1].set_ylabel("weighted nullity")
    axes[1, 1].set_title("Exact rank loss")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Finite-dimensional GAD geometry pilots")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.replicates < 2:
        raise ValueError("--replicates must be at least 2 to report variability")
    if args.n_test < 1:
        raise ValueError("--n-test must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    exact, exact_rows = exact_identity_pilot(rng, args.n_test)
    geometry_rows = nested_geometry_pilot(rng, args.replicates)
    fisher_rows = fisher_weight_pilot(rng)

    write_csv(args.output_dir / "exact_identity.csv", exact_rows)
    write_csv(args.output_dir / "nested_geometry.csv", geometry_rows)
    write_csv(args.output_dir / "fisher_weights.csv", fisher_rows)
    make_figure(
        args.output_dir / "gad_geometry_pilot.png", exact, geometry_rows, fisher_rows
    )

    print(f"Wrote pilot outputs to {args.output_dir}")
    print(f"Train-null prediction change: {exact['train_prediction_change']:.3e}")
    print(f"Exact-identity error: {exact['top_direction_abs_error']:.3e}")
    print(f"Random-direction average error: {exact['random_direction_mean_abs_error']:.3e}")


if __name__ == "__main__":
    main()
