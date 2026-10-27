"""A falsifiable static test of weight spectra versus task-dependent GAD risk.

Uses known one-hot latent features. Two dictionaries have IDENTICAL singular
values: D=[I,I] and D@R for an orthogonal latent-coordinate rotation R. Their
sparse-code collisions and task risks need not agree. Also includes a random
unit-column dictionary as a descriptive spectral baseline. Nothing here
asserts that random weights or an empirical bulk constitute HTSR.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aliasing.static_geometry import finite_support_gad
from aliasing.summaries import aggregate_seed_rows
from aliasing.superposition import representation_bayes_mse


def ints(s: str) -> list[int]:
    try:
        out = [int(v) for v in s.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not out or min(out) < 1:
        raise argparse.ArgumentTypeError("sample sizes must be positive")
    return out


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)


def run(args: argparse.Namespace) -> list[dict]:
    m, d = args.dimension, 2 * args.dimension
    codes = np.eye(d)
    probabilities = np.full(d, (1 - args.pair_mass) / (d - 2))
    probabilities[[0, m]] = args.pair_mass / 2
    base = np.tile(np.eye(m), (1, 2))
    rows = []
    for seed in range(args.num_seeds):
        rng = np.random.default_rng(args.seed + seed * 104729)
        rotation, _ = np.linalg.qr(rng.normal(size=(d, d)))
        random_d = rng.normal(size=(m, d))
        random_d /= np.linalg.norm(random_d, axis=0, keepdims=True)
        dictionaries = {"duplicate": base, "rotated": base @ rotation, "random": random_d}
        spectra = {name: np.linalg.svd(D, compute_uv=False) ** 2 for name, D in dictionaries.items()}
        if not np.allclose(spectra["duplicate"], spectra["rotated"], atol=1e-11):
            raise AssertionError("orthogonal rotation did not preserve the singular spectrum")
        for name, D in dictionaries.items():
            h = codes @ D.T
            eig = spectra[name]
            gram = D.T @ D
            off = np.abs(gram / np.outer(np.linalg.norm(D, axis=0), np.linalg.norm(D, axis=0)))
            np.fill_diagonal(off, 0)
            for task in ("contrast", "common"):
                y = np.zeros(d)
                y[0] = 1.0
                y[m] = -1.0 if task == "contrast" else 1.0
                bayes, collision_mass = representation_bayes_mse(h, y, probabilities, atol=1e-10)
                for n in args.sample_sizes:
                    for draw in range(args.num_draws):
                        idx = rng.choice(d, size=n, replace=True, p=probabilities)
                        gad = finite_support_gad(h[idx], y[idx], h, y, probabilities)
                        if gad["risk_identity_error"] > 1e-8 or gad["decomposition_reconstruction_error"] > 1e-8:
                            raise AssertionError("GAD identity failed")
                        # Population covariance and task direction relative to
                        # training nullspace matter even at fixed weight spectrum.
                        rows.append({
                            "seed": seed, "draw": draw, "dictionary": name,
                            "dictionary_id": ("duplicate", "rotated", "random").index(name),
                            "task": task, "task_id": ("contrast", "common").index(task),
                            "sample_size": n, "dimension": m, "feature_count": d,
                            "pair_mass": args.pair_mass,
                            "weight_top_eigenvalue": float(eig[0]),
                            "weight_nonzero_eigenvalue_std": float(eig.std()),
                            "weight_participation_rank": float(eig.sum() ** 2 / (eig @ eig)),
                            "weight_coherence": float(off.max()),
                            "representation_bayes_mse": float(bayes),
                            "collision_mass": float(collision_mass),
                            **gad,
                        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dimension", type=int, default=8)
    ap.add_argument("--pair-mass", type=float, default=.4)
    ap.add_argument("--sample-sizes", type=ints, default=ints("4,8,16,32,64,128"))
    ap.add_argument("--num-seeds", type=int, default=30)
    ap.add_argument("--num-draws", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args()
    if a.dimension < 2 or not 0 < a.pair_mass < 1 or a.num_seeds < 2 or a.num_draws < 1:
        ap.error("dimension >=2, pair mass in (0,1), >=2 seeds, >=1 draws required")
    rows = run(a)
    # Draws reuse one dictionary and must be collapsed before a seed CI.
    by = {}
    for r in rows:
        key = (r["seed"], r["dictionary_id"], r["task_id"], r["sample_size"])
        by.setdefault(key, []).append(r)
    collapsed = []
    for group in by.values():
        rec = {k: v for k, v in group[0].items() if k not in {"draw", "dictionary", "task"}}
        for k in rec:
            if k not in {"seed", "dictionary_id", "task_id", "sample_size"}:
                rec[k] = float(np.mean([float(g[k]) for g in group]))
        collapsed.append(rec)
    keys = ("dictionary_id", "task_id", "sample_size")
    means, std, ci = aggregate_seed_rows(collapsed, group_keys=keys,
        excluded_keys={"seed", *keys}, n_bootstrap=1000)
    for group in (means, std, ci):
        for r in group:
            r["dictionary"] = ("duplicate", "rotated", "random")[int(r["dictionary_id"])]
            r["task"] = ("contrast", "common")[int(r["task_id"])]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (("raw", rows), ("mean", means), ("std", std), ("ci95", ci)):
        save_csv(a.output_dir / f"{name}.csv", records)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for col, metric in enumerate(("representation_bayes_mse", "model_insufficiency", "aliasing")):
        for row_ix, task in enumerate(("contrast", "common")):
            ax = axes[row_ix, col]
            for name in ("duplicate", "rotated", "random"):
                selected = sorted((r for r in means if r["task"] == task and r["dictionary"] == name),
                                  key=lambda r: r["sample_size"])
                ax.plot([r["sample_size"] for r in selected], [r[metric] for r in selected],
                        marker="o", label=name)
            ax.set(xlabel="readout training samples", ylabel=metric.replace("_", " "), title=task)
            ax.grid(alpha=.3)
    axes[0, 0].legend()
    fig.savefig(a.output_dir / "matched_spectrum_tasks.png", dpi=160)
    plt.close(fig)
    max_error = max(r["risk_identity_error"] for r in rows)
    with (a.output_dir / "run_manifest.json").open("w") as f:
        json.dump({"experiment": "12_matched_spectrum_tasks", "config": {k: str(v) for k,v in vars(a).items()},
                   "max_risk_identity_error": max_error, "num_raw_rows": len(rows),
                   "note": "duplicate and rotated share all singular values; Bayes risk is target/support dependent"}, f, indent=2)
    print(f"Saved {len(rows)} rows; max GAD identity error={max_error:.3g}; output={a.output_dir}")


if __name__ == "__main__":
    main()
