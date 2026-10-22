"""Factorial test of geometry, task alignment, co-occurrence, and sample aliasing.

This is a finite-support, frozen-representation experiment. It separates
irreducible representation ambiguity from linear-model insufficiency and
finite-sample GAD components. No learned features are used here; experiment 08
tests the corresponding claims in trained representations.
"""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aliasing.static_geometry import finite_support_gad, make_dictionary
from aliasing.summaries import aggregate_seed_rows
from aliasing.superposition import dictionary_geometry, representation_bayes_mse

FAMILY_ID = {"random": 0, "structured": 1}
TASK_ID = {name: i for i, name in enumerate(
    ("random", "sum", "pair_contrast", "max_coherence_contrast", "decoder_null")
)}


def ints(value: str) -> list[int]:
    try:
        out = [int(v) for v in value.split(",") if v]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not out or min(out) < 1:
        raise argparse.ArgumentTypeError("values must be positive")
    return out


def floats(value: str) -> list[float]:
    try:
        out = [float(v) for v in value.split(",") if v]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not out:
        raise argparse.ArgumentTypeError("expected at least one value")
    return out


def correlated_binary_support(p: int, marginal: float, joint: float) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate a Bernoulli codebook with a controlled (0,1) feature pair."""
    if not max(0.0, 2 * marginal - 1) <= joint <= marginal:
        raise ValueError("joint probability must preserve the specified Bernoulli marginals")
    codes = ((np.arange(1 << p)[:, None] >> np.arange(p)) & 1).astype(float)
    a, b = codes[:, 0], codes[:, 1]
    pair_prob = np.select(
        [((a == 1) & (b == 1)), ((a == 1) & (b == 0)), ((a == 0) & (b == 1))],
        [joint, marginal - joint, marginal - joint],
        default=1.0 - 2.0 * marginal + joint,
    )
    rest = codes[:, 2:]
    rest_count = rest.sum(axis=1)
    probs = pair_prob * marginal**rest_count * (1 - marginal) ** (p - 2 - rest_count)
    probs /= probs.sum()
    return codes, probs


def task_vector(kind: str, dictionary: np.ndarray, seed: int) -> np.ndarray:
    d, p = dictionary.shape
    rng = np.random.default_rng(seed)
    if kind == "random":
        beta = rng.normal(size=p)
    elif kind == "sum":
        beta = np.ones(p)
    elif kind == "pair_contrast":
        beta = np.zeros(p)
        beta[:2] = (1.0, -1.0)
    elif kind == "max_coherence_contrast":
        gram = np.abs(dictionary.T @ dictionary)
        np.fill_diagonal(gram, -np.inf)
        i, j = np.unravel_index(np.argmax(gram), gram.shape)
        beta = np.zeros(p)
        beta[i], beta[j] = 1.0, -1.0
    elif kind == "decoder_null":
        # This target varies along a coefficient-space direction annihilated by
        # the linear decoder. On a finite codebook it can still be recoverable
        # by a nonlinear function of the representation if there are no exact
        # collisions, so Bayes risk and linear model insufficiency are separate.
        _, _, vh = np.linalg.svd(dictionary, full_matrices=True)
        null = vh[d:, :]
        beta = null[0] if null.size else rng.normal(size=p)
    else:
        raise ValueError(f"unknown task: {kind}")
    return beta / max(float(np.linalg.norm(beta)), np.finfo(float).eps)


def one_run(args: argparse.Namespace, seed: int, family: str, p: int, n: int,
            joint: float, task: str) -> dict[str, float | str]:
    d = args.dimension
    dictionary = make_dictionary(d, p, family, seed=seed + args.seed, spread=args.spread)
    codes, probabilities = correlated_binary_support(p, args.marginal, joint)
    beta = task_vector(task, dictionary, seed + 17)
    target = codes @ beta
    hidden = codes @ dictionary.T
    rng = np.random.default_rng(args.seed + 1000003 * seed + 1009 * p + 37 * n + int(joint * 1e6))
    indices = rng.choice(len(codes), size=n, replace=True, p=probabilities)
    train_h, train_y = hidden[indices], target[indices]
    gad = finite_support_gad(train_h, train_y, hidden, target, probabilities)
    bayes_mse, alias_mass = representation_bayes_mse(
        hidden, target, probabilities, atol=args.collision_tolerance
    )
    geom = dictionary_geometry(dictionary)
    singular = np.linalg.svd(dictionary, compute_uv=False)
    nonzero = singular[singular > args.rank_tolerance * singular.max()]
    rank = int(nonzero.size)
    cond = float(nonzero.max() / nonzero.min()) if rank else float("inf")
    _, counts = np.unique(indices, return_counts=True)
    decoder_rowspace = dictionary.T @ np.linalg.pinv(dictionary.T)
    task_linear_residual = beta - decoder_rowspace @ beta
    return {
        "seed": float(seed), "family": family, "task": task,
        "family_id": float(FAMILY_ID[family]), "task_id": float(TASK_ID[task]),
        "dimension": float(d), "feature_count": float(p),
        "sample_size": float(n), "marginal_activity": float(args.marginal),
        "joint_activity": float(joint), "feature_cooccurrence": float(joint / args.marginal),
        "unique_training_states": float(counts.size),
        "state_coverage_probability": float(probabilities[np.unique(indices)].sum()),
        "dictionary_rank": float(rank), "dictionary_condition_nonzero": cond,
        "task_linear_decoder_residual_norm": float(np.linalg.norm(task_linear_residual)),
        "representation_bayes_mse": bayes_mse,
        "aliased_probability_mass": alias_mass,
        **geom, **gad,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no experiment rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=6)
    parser.add_argument("--feature-counts", type=ints, default=ints("6,8,10"))
    parser.add_argument("--sample-sizes", type=ints, default=ints("8,16,24,32,48,64,96,128"))
    parser.add_argument("--joint-probabilities", type=floats, default=floats("0,0.05,0.10,0.15,0.20"))
    parser.add_argument("--tasks", default="random,sum,pair_contrast,max_coherence_contrast,decoder_null")
    parser.add_argument("--families", default="random,structured")
    parser.add_argument("--marginal", type=float, default=0.2)
    parser.add_argument("--spread", type=int, default=2)
    parser.add_argument("--num-seeds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--rank-tolerance", type=float, default=1e-10)
    parser.add_argument("--collision-tolerance", type=float, default=1e-10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.dimension < 2 or args.num_seeds < 2 or not 0 < args.marginal < 0.5:
        raise ValueError("dimension >=2, num-seeds >=2, and marginal in (0, .5) required")
    if any(p < 2 or p > 16 for p in args.feature_counts):
        raise ValueError("feature-counts must be between 2 and 16 (exact codebook enumeration)")
    if any(not 0 <= q <= args.marginal for q in args.joint_probabilities):
        raise ValueError("joint probabilities must be in [0, marginal]")

    rows = []
    for family, p, n, joint, task, seed in product(
        args.families.split(","), args.feature_counts, args.sample_sizes,
        args.joint_probabilities, args.tasks.split(","), range(args.num_seeds),
    ):
        if task == "decoder_null" and p <= args.dimension:
            continue
        if family == "structured" and p <= args.dimension:
            continue
        if family == "random" and p < 2:
            continue
        rows.append(one_run(args, seed, family, p, n, joint, task))

    keys = ("family_id", "task_id", "feature_count", "sample_size", "joint_activity")
    means, stds, cis = aggregate_seed_rows(
        rows, group_keys=keys, excluded_keys={"seed", "family", "task", *keys}, n_bootstrap=2000,
    )
    for group in (means, stds, cis):
        for row in group:
            row["family"] = next(k for k,v in FAMILY_ID.items() if v == int(row["family_id"]))
            row["task"] = next(k for k,v in TASK_ID.items() if v == int(row["task_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "raw.csv", rows)
    write_csv(args.output_dir / "mean.csv", means)
    write_csv(args.output_dir / "std.csv", stds)
    write_csv(args.output_dir / "ci95.csv", cis)
    write_csv(args.output_dir / "config.csv", [{
        "dimension": args.dimension, "feature_counts": ",".join(map(str, args.feature_counts)),
        "sample_sizes": ",".join(map(str, args.sample_sizes)),
        "joint_probabilities": ",".join(map(str, args.joint_probabilities)),
        "tasks": args.tasks, "families": args.families, "marginal": args.marginal,
        "num_seeds": args.num_seeds, "seed": args.seed,
    }])

    # Plot only the most theory-facing quantities. Each line is one family/task/
    # co-occurrence condition; uncertainty bands are seed-bootstrap intervals.
    panels = (("test_mse", "held-out risk"), ("model_insufficiency", "linear model insufficiency"),
              ("aliasing", "finite-sample alias component"),
              ("representation_bayes_mse", "representation Bayes risk"))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, (metric, label) in zip(axes.flat, panels):
        for family in args.families.split(","):
            for task in args.tasks.split(","):
                for joint in args.joint_probabilities:
                    selected = [r for r in means if int(float(r["family_id"])) == FAMILY_ID[family]
                                and int(float(r["task_id"])) == TASK_ID[task]
                                and float(r["joint_activity"]) == joint
                                and int(float(r["feature_count"])) == args.feature_counts[-1]]
                    selected.sort(key=lambda r: float(r["sample_size"]))
                    if not selected:
                        continue
                    ax.plot([r["sample_size"] for r in selected], [r[metric] for r in selected],
                            marker=".", label=f"{family}/{task}/q={joint:g}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("symlog", linthresh=1e-10) if metric == "representation_bayes_mse" else None
        ax.set_xlabel("training sample count")
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=5, ncol=2)
    fig.savefig(args.output_dir / "static_geometry_factorial.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {len(rows)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
