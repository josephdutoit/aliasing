"""Exact toy experiment for feature superposition and representation aliasing.

This is a controlled, static representation experiment, not a trained neural
network result.  It varies how many orthogonal directions one feature spans
and whether the data support contains the corresponding alternative code.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aliasing.superposition import (
    controlled_code_distribution,
    dictionary_geometry,
    representation_bayes_mse,
    structured_overcomplete_dictionary,
)


def run_grid(dimension: int, collision_mass: float) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for spread in range(1, dimension + 1):
        geometry = dictionary_geometry(
            structured_overcomplete_dictionary(dimension, spread)
        )
        for max_sparsity in range(1, dimension + 1):
            representations, targets, probabilities = controlled_code_distribution(
                dimension,
                spread,
                max_sparsity,
                collision_mass=collision_mass,
            )
            bayes_mse, aliased_mass = representation_bayes_mse(
                representations, targets, probabilities
            )
            rows.append(
                {
                    "spread": float(spread),
                    "max_sparsity": float(max_sparsity),
                    **geometry,
                    "collision_mass": collision_mass,
                    "support_contains_alias_pair": float(max_sparsity >= spread),
                    "aliased_probability_mass": aliased_mass,
                    "representation_bayes_mse": bayes_mse,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_grid(rows: list[dict[str, float]], path: Path) -> None:
    spreads = sorted({int(row["spread"]) for row in rows})
    sparsities = sorted({int(row["max_sparsity"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    for sparsity in sparsities:
        selected = [row for row in rows if int(row["max_sparsity"]) == sparsity]
        selected.sort(key=lambda row: row["spread"])
        axes[0].plot(
            [row["spread"] for row in selected],
            [row["representation_bayes_mse"] for row in selected],
            marker=".",
            label=f"max sparsity={sparsity}",
        )
    axes[0].set_xlabel("number of axes in superposed feature")
    axes[0].set_ylabel("representation Bayes MSE")
    axes[0].set_title("Aliasing requires the alternative code in the support")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)

    selected_geometry = [row for row in rows if int(row["max_sparsity"]) == 1]
    selected_geometry.sort(key=lambda row: row["spread"])
    axes[1].plot(
        [row["spread"] for row in selected_geometry],
        [row["mutual_coherence"] for row in selected_geometry],
        marker="o",
        label="dictionary coherence",
    )
    axes[1].plot(
        [row["spread"] for row in selected_geometry],
        1.0 / np.asarray(spreads, dtype=float),
        marker="s",
        label="1 / participation ratio",
    )
    axes[1].set_xlabel("number of axes in superposed feature")
    axes[1].set_ylabel("geometry statistic")
    axes[1].set_title("Overlap geometry")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--collision-mass", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.dimension < 2:
        raise ValueError("--dimension must be at least 2")
    if not 0.0 <= args.collision_mass <= 1.0:
        raise ValueError("--collision-mass must lie in [0, 1]")

    rows = run_grid(args.dimension, args.collision_mass)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "superposition_aliasing.csv", rows)
    plot_grid(rows, args.output_dir / "superposition_aliasing.png")
    print(f"Wrote {len(rows)} exact finite-support cases to {args.output_dir}")


if __name__ == "__main__":
    main()
