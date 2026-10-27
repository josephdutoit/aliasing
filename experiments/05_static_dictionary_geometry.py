"""Controlled sparse-dictionary study of representation and sample aliasing.

This experiment keeps dictionary overlap, code support, sample coverage, and
task risk separately measurable. It is a finite-support linear readout study,
not evidence about feature learning in a neural network.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aliasing.static_geometry import (
    finite_support_gad,
    make_dictionary,
    restricted_min_singular_value,
    sparse_binary_support,
)
from aliasing.summaries import bootstrap_mean_interval
from aliasing.superposition import dictionary_geometry, representation_bayes_mse


FAMILY_IDS = {"orthogonal": 0, "random": 1, "structured": 2}


def comma_ints(value: str) -> list[int]:
    try:
        values = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def run(args: argparse.Namespace) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for seed, family, feature_count, sparsity, sample_size, cooccur in product(
        range(args.num_seeds),
        args.families,
        args.feature_counts,
        args.max_sparsities,
        args.sample_sizes,
        (False, True),
    ):
        if family == "orthogonal" and feature_count > args.dimension:
            continue
        if family == "structured" and feature_count <= args.dimension:
            continue
        if sparsity > feature_count:
            continue

        dictionary = make_dictionary(
            args.dimension,
            feature_count,
            family,
            seed=args.seed + seed,
            spread=args.spread,
        )
        codes, probabilities = sparse_binary_support(
            feature_count, sparsity, allow_cooccurrence=cooccur
        )
        hidden_support = codes @ dictionary.T
        rng = np.random.default_rng(args.seed + 104729 * seed + 17 * feature_count)
        beta = rng.normal(size=feature_count) / np.sqrt(feature_count)
        target_support = codes @ beta
        train_indices = rng.choice(
            len(codes), size=sample_size, replace=True, p=probabilities
        )
        gad = finite_support_gad(
            hidden_support[train_indices],
            target_support[train_indices],
            hidden_support,
            target_support,
            probabilities,
        )
        bayes_mse, aliased_mass = representation_bayes_mse(
            hidden_support, target_support, probabilities, atol=args.collision_tolerance
        )
        geometry = dictionary_geometry(dictionary)
        kappa = restricted_min_singular_value(dictionary, 2 * sparsity)
        row: dict[str, float | str] = {
            "seed": float(seed),
            "dictionary_family": family,
            "family_id": float(FAMILY_IDS[family]),
            "dimension": float(args.dimension),
            "feature_count": float(feature_count),
            "max_sparsity": float(sparsity),
            "sample_size": float(sample_size),
            "allow_cooccurrence": float(cooccur),
            "restricted_min_singular_value_2k": kappa,
            "representation_bayes_mse": bayes_mse,
            "aliased_probability_mass": aliased_mass,
            **geometry,
            **gad,
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no compatible configurations were produced")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    group_keys = (
        "family_id", "dimension", "feature_count", "max_sparsity",
        "sample_size", "allow_cooccurrence",
    )
    excluded = set(group_keys) | {"seed", "dictionary_family"}
    grouped: dict[tuple[float, ...], list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(float(row[key]) for key in group_keys)].append(row)
    means: list[dict[str, object]] = []
    intervals: list[dict[str, object]] = []
    metric_keys = [key for key in rows[0] if key not in excluded]
    for group_index, (group, samples) in enumerate(sorted(grouped.items())):
        mean_row: dict[str, object] = dict(zip(group_keys, group))
        ci_row: dict[str, object] = dict(zip(group_keys, group))
        mean_row["dictionary_family"] = ci_row["dictionary_family"] = samples[0]["dictionary_family"]
        mean_row["num_seeds"] = ci_row["num_seeds"] = len(samples)
        for metric_index, metric in enumerate(metric_keys):
            values = np.asarray([float(item[metric]) for item in samples])
            mean_row[metric] = float(np.nanmean(values))
            lo, hi = bootstrap_mean_interval(
                values,
                seed=20261022 + group_index * 101 + metric_index,
                n_bootstrap=1000,
            )
            ci_row[f"{metric}_ci95_low"] = lo
            ci_row[f"{metric}_ci95_high"] = hi
        means.append(mean_row)
        intervals.append(ci_row)
    return means, intervals


def plot_summary(means: list[dict[str, object]], output: Path) -> None:
    configurations = sorted({
        (row["dictionary_family"], row["feature_count"], row["max_sparsity"], row["allow_cooccurrence"])
        for row in means
    })
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    metrics = [
        ("test_mse", "held-out MSE"),
        ("aliasing", "GAD alias component MSE"),
        ("representation_bayes_mse", "representation Bayes MSE"),
    ]
    for axis, (metric, label) in zip(axes, metrics):
        for family, p, k, cooccur in configurations:
            selected = [
                row for row in means
                if row["dictionary_family"] == family
                and row["feature_count"] == p
                and row["max_sparsity"] == k
                and row["allow_cooccurrence"] == cooccur
            ]
            if not selected:
                continue
            selected.sort(key=lambda row: float(row["sample_size"]))
            axis.plot(
                [row["sample_size"] for row in selected],
                [row[metric] for row in selected],
                marker="o",
                label=f"{family}, p={int(p)}, k={int(k)}, cooccur={int(cooccur)}",
            )
        axis.set_xlabel("training sample count")
        axis.set_ylabel(label)
        axis.set_xscale("log", base=2)
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=4)
    parser.add_argument("--feature-counts", type=comma_ints, default=comma_ints("4,5,8"))
    parser.add_argument("--max-sparsities", type=comma_ints, default=comma_ints("1,2"))
    parser.add_argument("--sample-sizes", type=comma_ints, default=comma_ints("8,16,32"))
    parser.add_argument("--families", type=lambda s: s.split(","), default=["orthogonal", "random", "structured"])
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--spread", type=int, default=2)
    parser.add_argument("--collision-tolerance", type=float, default=1e-12)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.dimension < 2 or args.num_seeds < 2:
        raise ValueError("dimension must be >=2 and num-seeds >=2")
    if any(family not in FAMILY_IDS for family in args.families):
        raise ValueError(f"families must be selected from {tuple(FAMILY_IDS)}")
    if args.collision_tolerance < 0:
        raise ValueError("collision tolerance must be nonnegative")

    rows = run(args)
    means, intervals = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "raw.csv", rows)
    write_csv(args.output_dir / "mean.csv", means)
    write_csv(args.output_dir / "ci95.csv", intervals)
    write_csv(args.output_dir / "config.csv", [{
        "dimension": args.dimension,
        "feature_counts": ",".join(map(str, args.feature_counts)),
        "max_sparsities": ",".join(map(str, args.max_sparsities)),
        "sample_sizes": ",".join(map(str, args.sample_sizes)),
        "families": ",".join(args.families),
        "num_seeds": args.num_seeds,
        "seed": args.seed,
        "spread": args.spread,
        "collision_tolerance": args.collision_tolerance,
    }])
    plot_summary(means, args.output_dir / "static_dictionary_geometry.png")
    print(f"Wrote {len(rows)} configurations to {args.output_dir}")


if __name__ == "__main__":
    main()
