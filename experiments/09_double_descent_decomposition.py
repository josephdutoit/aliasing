"""Replicated featurewise/samplewise double-descent with GAD diagnostics.

One random-feature design is paired across the full width-by-sample grid for
each seed. At every point we record rank and conditioning and decompose the
population test error into model, alias, and data/algorithm components. This
tests whether observed risk spikes are actually carried by the alias term.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aliasing.static_geometry import finite_support_gad
from aliasing.summaries import aggregate_seed_rows


def ints(value: str) -> list[int]:
    try: out = [int(x) for x in value.split(",") if x]
    except ValueError as exc: raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not out or min(out) < 1: raise argparse.ArgumentTypeError("all values must be positive")
    return out


def covariance(d: int, kind: str) -> np.ndarray:
    if kind == "isotropic": return np.eye(d)
    if kind == "spiked": return np.diag(np.geomspace(16.0, 0.25, d))
    if kind == "low_rank":
        vals = np.r_[np.ones(max(1, d // 4)), np.full(d - max(1, d // 4), 1e-3)]
        return np.diag(vals)
    raise ValueError(kind)


def evaluate_design(args: argparse.Namespace, seed: int, cov_kind: str,
                    task: str) -> list[dict[str, float | str]]:
    rng = np.random.default_rng(args.seed + seed * 1000003)
    cov = covariance(args.input_dim, cov_kind)
    root = np.linalg.cholesky(cov)
    x_train_all = rng.normal(size=(max(args.sample_sizes), args.input_dim)) @ root.T
    x_test = rng.normal(size=(args.n_test, args.input_dim)) @ root.T
    teacher = np.zeros(args.input_dim)
    if task == "first_pc": teacher[0] = 1.0
    elif task == "random": teacher = rng.normal(size=args.input_dim)
    else: raise ValueError(task)
    teacher /= np.linalg.norm(teacher)
    y_train_all = x_train_all @ teacher
    y_test = x_test @ teacher

    # Shared random directions make width comparisons paired and nested.
    w = rng.normal(size=(args.input_dim, max(args.widths)))
    w /= np.maximum(np.linalg.norm(w, axis=0, keepdims=True), np.finfo(float).eps)
    z_train_all = np.maximum(x_train_all @ w, 0.0) / np.sqrt(args.input_dim)
    z_test_all = np.maximum(x_test @ w, 0.0) / np.sqrt(args.input_dim)
    # The population projection depends on the frozen representation and task,
    # not the training sample size. Compute it once per width, not once per grid
    # cell; this keeps the large replicated sweep practical.
    population_heads = {
        width: np.linalg.lstsq(z_test_all[:, :width], y_test, rcond=None)[0]
        for width in args.widths
    }

    rows = []
    for n, width in product(args.sample_sizes, args.widths):
        h_train, h_test = z_train_all[:n, :width], z_test_all[:, :width]
        y_train = y_train_all[:n]
        singular = np.linalg.svd(h_train, compute_uv=False)
        tol = max(h_train.shape) * np.finfo(float).eps * (singular[0] if singular.size else 0.0)
        positive = singular[singular > tol]
        rank = int(positive.size)
        condition = float(positive.max() / positive.min()) if positive.size else float("inf")
        gad = finite_support_gad(
            h_train, y_train, h_test, y_test, np.full(args.n_test, 1.0 / args.n_test),
            population_head=population_heads[width],
        )
        fitted = np.linalg.pinv(h_train) @ y_train
        rows.append({
            "seed": float(seed), "covariance": cov_kind, "task": task,
            "input_dim": float(args.input_dim), "sample_size": float(n), "width": float(width),
            "n_over_width": float(n / width), "training_rank": float(rank),
            "training_nullity": float(width - rank), "training_condition_nonzero": condition,
            "train_mse": float(np.mean((h_train @ fitted - y_train)**2)),
            **gad,
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows: raise ValueError("no result rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--widths", type=ints, default=ints("8,16,24,32,40,48,56,64,80,96,128"))
    ap.add_argument("--sample-sizes", type=ints, default=ints("8,16,24,32,40,48,56,64,80,96,128,192,256"))
    ap.add_argument("--input-dim", type=int, default=20)
    ap.add_argument("--covariances", default="isotropic,spiked,low_rank")
    ap.add_argument("--tasks", default="first_pc,random")
    ap.add_argument("--num-seeds", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args()
    if a.num_seeds < 2 or a.input_dim < 2 or a.n_test < 100:
        raise ValueError("need >=2 seeds, input dimension >=2, and at least 100 test points")
    if any(w > a.sample_sizes[-1] * 2 for w in a.widths):
        raise ValueError("unexpectedly wide grid; check sample sizes and widths")
    rows = []
    for cov, task, seed in product(a.covariances.split(","), a.tasks.split(","), range(a.num_seeds)):
        rows.extend(evaluate_design(a, seed, cov, task))
    # Aggregate over seeds only; rows within a seed share the same inputs and
    # random features and are therefore deliberately treated as paired.
    # Generic helper expects numeric grouping keys, so create stable IDs.
    cov_ids = {v:i for i,v in enumerate(a.covariances.split(","))}
    task_ids = {v:i for i,v in enumerate(a.tasks.split(","))}
    for row in rows:
        row["covariance_id"] = float(cov_ids[str(row["covariance"])])
        row["task_id"] = float(task_ids[str(row["task"])])
    numeric_keys = ("covariance_id", "task_id", "sample_size", "width")
    means, stds, cis = aggregate_seed_rows(
        rows, group_keys=numeric_keys,
        excluded_keys={"seed", "covariance", "task", *numeric_keys}, n_bootstrap=2000,
    )
    for group in (means, stds, cis):
        for row in group:
            row["covariance"] = next(k for k,v in cov_ids.items() if v == int(row["covariance_id"]))
            row["task"] = next(k for k,v in task_ids.items() if v == int(row["task_id"]))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(a.output_dir/"raw.csv", rows); write_csv(a.output_dir/"mean.csv", means)
    write_csv(a.output_dir/"std.csv", stds); write_csv(a.output_dir/"ci95.csv", cis)
    write_csv(a.output_dir/"config.csv", [{
        "widths": ",".join(map(str,a.widths)), "sample_sizes": ",".join(map(str,a.sample_sizes)),
        "input_dim": a.input_dim, "covariances": a.covariances, "tasks": a.tasks,
        "num_seeds": a.num_seeds, "n_test": a.n_test, "seed": a.seed,
    }])
    metric_info = (
        ("test_mse", "held-out MSE"),
        ("aliasing", "GAD alias component"),
        ("data_insufficiency", "GAD data-insufficiency component"),
        ("training_condition_nonzero", "nonzero singular condition"),
    )
    ci_lookup = {
        (r["covariance"], r["task"], int(float(r["sample_size"])), int(float(r["width"]))): r
        for r in cis
    }
    for cov in a.covariances.split(","):
        for task in a.tasks.split(","):
            fig, axes = plt.subplots(1, 4, figsize=(19, 4.5), constrained_layout=True)
            for ax, (metric, label) in zip(axes, metric_info):
                for width in a.widths:
                    sel = [r for r in means if int(float(r["width"])) == width
                           and r["covariance"] == cov and r["task"] == task]
                    sel.sort(key=lambda r: float(r["sample_size"]))
                    if not sel:
                        continue
                    x = np.asarray([float(r["sample_size"]) for r in sel])
                    y = np.asarray([float(r[metric]) for r in sel])
                    intervals = [ci_lookup[(cov, task, int(float(r["sample_size"])), width)] for r in sel]
                    lo = np.asarray([float(r[f"{metric}_ci95_low"]) for r in intervals])
                    hi = np.asarray([float(r[f"{metric}_ci95_high"]) for r in intervals])
                    line, = ax.plot(x, y, marker=".", linewidth=1.1, label=f"w={width}")
                    ax.fill_between(x, lo, hi, color=line.get_color(), alpha=.12, linewidth=0)
                ax.set_xlabel("training sample count")
                ax.set_ylabel(label)
                ax.grid(alpha=.25)
                # Interpolation-region spikes can be orders of magnitude larger
                # than the surrounding curve. Symlog keeps them visible without
                # making the finite, non-spike regime unreadable.
                ax.set_yscale("symlog", linthresh=1e-8)
            axes[0].legend(fontsize=7, ncol=2)
            fig.suptitle(f"Frozen random features: covariance={cov}, task={task}; shaded bands are pointwise seed-bootstrap 95% CIs")
            suffix = f"{cov}_{task}".replace("/", "_")
            fig.savefig(a.output_dir/f"double_descent_{suffix}.png", dpi=170, bbox_inches="tight")
            if cov == a.covariances.split(",")[0] and task == a.tasks.split(",")[0]:
                fig.savefig(a.output_dir/"double_descent_decomposition.png", dpi=170, bbox_inches="tight")
            plt.close(fig)
    max_reconstruction_error = max(float(r["decomposition_reconstruction_error"]) for r in rows)
    max_identity_error = max(float(r["risk_identity_error"]) for r in rows)
    with (a.output_dir/"run_manifest.json").open("w") as f:
        json.dump({"experiment": "09_double_descent_decomposition",
                   "args": vars(a) | {"output_dir": str(a.output_dir)}, "rows": len(rows),
                   "max_decomposition_reconstruction_error": max_reconstruction_error,
                   "max_risk_identity_error": max_identity_error,
                   "plot_configs": [f"{c}_{t}" for c in a.covariances.split(",") for t in a.tasks.split(",")]},
                  f, indent=2, default=str)
    print(f"Wrote {len(rows)} rows and {len(a.covariances.split(','))*len(a.tasks.split(','))} configuration plots to {a.output_dir}; "
          f"max risk identity error={max_identity_error:.3g}")


if __name__ == "__main__": main()
