"""Test whether trained feature superposition predicts downstream aliasing.

Sparse binary codes have a controlled correlated feature pair. An autoencoder
is trained once per representation seed, then frozen while task, readout sample
size, and task/feature alignment are varied. This separates encoder geometry
from readout sampling and measures exact code collisions directly.
"""

from __future__ import annotations

import argparse
import csv
import random
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from aliasing.static_geometry import feature_direction_metrics, finite_support_gad
from aliasing.summaries import aggregate_seed_rows
from aliasing.superposition import representation_bayes_mse

torch.set_default_dtype(torch.float64)
TASK_ID = {name: i for i, name in enumerate(("sum", "pair_contrast", "single_feature", "random"))}


class SparseAE(nn.Module):
    def __init__(self, p: int, m: int):
        super().__init__()
        self.encoder = nn.Linear(p, m)
        self.decoder = nn.Linear(m, p)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.hidden(x))


def parse_ints(value: str) -> list[int]:
    try:
        result = [int(x) for x in value.split(",") if x]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or min(result) < 1:
        raise argparse.ArgumentTypeError("all values must be positive")
    return result


def parse_floats(value: str) -> list[float]:
    try:
        result = [float(x) for x in value.split(",") if x]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one value is required")
    return result


def codebook(p: int, marginal: float, joint: float) -> tuple[np.ndarray, np.ndarray]:
    if not max(0.0, 2 * marginal - 1) <= joint <= marginal:
        raise ValueError("joint probability is incompatible with marginal activity")
    z = ((np.arange(1 << p)[:, None] >> np.arange(p)) & 1).astype(float)
    a, b = z[:, 0], z[:, 1]
    pair = np.select([a*b == 1, (a == 1)&(b == 0), (a == 0)&(b == 1)],
                     [joint, marginal-joint, marginal-joint], 1-2*marginal+joint)
    k = z[:, 2:].sum(axis=1)
    prob = pair * marginal**k * (1-marginal)**(p-2-k)
    prob /= prob.sum()
    return z, prob


def target(beta_kind: str, z: np.ndarray, seed: int) -> np.ndarray:
    p = z.shape[1]
    rng = np.random.default_rng(seed)
    if beta_kind == "sum":
        beta = np.ones(p)
    elif beta_kind == "pair_contrast":
        beta = np.zeros(p); beta[:2] = (1.0, -1.0)
    elif beta_kind == "single_feature":
        beta = np.zeros(p); beta[-1] = 1.0
    elif beta_kind == "random":
        beta = rng.normal(size=p)
    else:
        raise ValueError(beta_kind)
    beta /= np.linalg.norm(beta)
    return z @ beta


def one_representation(args: argparse.Namespace, seed: int, p: int, m: int,
                       joint: float) -> list[dict[str, float | str]]:
    seed_value = args.seed + seed * 100003 + p * 1009 + m * 17 + round(joint*1e6)
    random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)
    z, prob = codebook(p, args.marginal, joint)
    rng = np.random.default_rng(seed_value + 5)
    train_indices = rng.choice(len(z), size=args.encoder_samples, replace=True, p=prob)
    z_encoder = torch.as_tensor(z[train_indices])
    model = SparseAE(p, m)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model.train()
    for _ in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        loss = ((model(z_encoder) - z_encoder) ** 2).mean()
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        h = model.hidden(torch.as_tensor(z)).cpu().numpy()
        rec = float(((model(torch.as_tensor(z)) - torch.as_tensor(z))**2 * torch.as_tensor(prob[:,None])).sum().item() / p)

    # Causal feature directions average the hidden change from toggling a
    # feature from absent to present, weighted over observed contexts.
    directions = np.zeros((m, p))
    with torch.no_grad():
        base = model.hidden(torch.as_tensor(z))
        for j in range(p):
            mask = z[:, j] == 0
            w = prob[mask]; w = w / w.sum()
            changed = z[mask].copy(); changed[:, j] = 1
            effect = model.hidden(torch.as_tensor(changed)) - base[torch.as_tensor(mask)]
            directions[:, j] = (effect * torch.as_tensor(w[:, None])).sum(dim=0).cpu().numpy()
    geom = feature_direction_metrics(directions)

    rows = []
    for task in args.tasks.split(","):
        y = target(task, z, seed_value + 101)
        bayes, alias_mass = representation_bayes_mse(h, y, prob, atol=args.collision_tolerance)
        sqrt_prob = np.sqrt(prob)
        population_head = np.linalg.lstsq(h * sqrt_prob[:, None], y * sqrt_prob, rcond=None)[0]
        for n, readout_seed in product(args.sample_sizes, range(args.num_readout_draws)):
            rng_readout = np.random.default_rng(seed_value + readout_seed * 65537 + n * 19)
            idx = rng_readout.choice(len(z), size=n, replace=True, p=prob)
            gad = finite_support_gad(
                h[idx], y[idx], h, y, prob, population_head=population_head
            )
            rows.append({
                "representation_seed": float(seed), "readout_draw": float(readout_seed),
                "feature_count": float(p), "bottleneck": float(m), "feature_pressure": float(p/m),
                "joint_activity": float(joint), "feature_cooccurrence": float(joint/args.marginal),
                "task": task, "task_id": float(TASK_ID[task]),
                "sample_size": float(n), "encoder_samples": float(args.encoder_samples),
                "autoencoder_reconstruction_mse": rec,
                "representation_bayes_mse": bayes, "aliased_probability_mass": alias_mass,
                **geom, **gad,
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows: raise ValueError("no results to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feature-counts", type=parse_ints, default=parse_ints("8,10,12"))
    ap.add_argument("--bottlenecks", type=parse_ints, default=parse_ints("2,3,4"))
    ap.add_argument("--sample-sizes", type=parse_ints, default=parse_ints("16,32,64,128,256"))
    ap.add_argument("--joint-probabilities", type=parse_floats, default=parse_floats("0,0.05,0.10,0.15,0.20"))
    ap.add_argument("--tasks", default="sum,pair_contrast,single_feature,random")
    ap.add_argument("--marginal", type=float, default=0.2)
    ap.add_argument("--encoder-samples", type=int, default=4096)
    ap.add_argument("--num-seeds", type=int, default=20)
    ap.add_argument("--num-readout-draws", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--learning-rate", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--collision-tolerance", type=float, default=1e-9)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args()
    if a.num_seeds < 2 or a.num_readout_draws < 1 or a.epochs < 1:
        raise ValueError("need >=2 representation seeds and positive draws/epochs")
    if any(p > 12 for p in a.feature_counts):
        raise ValueError("exact collision diagnostics enumerate all codes; feature count must be <=12")
    if not 0 < a.marginal < 0.5 or any(not 0 <= q <= a.marginal for q in a.joint_probabilities):
        raise ValueError("invalid marginal or joint activity probability")
    rows = []
    for p, m, q, seed in product(a.feature_counts, a.bottlenecks, a.joint_probabilities, range(a.num_seeds)):
        if m >= p: continue
        rows.extend(one_representation(a, seed, p, m, q))
    # Readout draws share one learned representation. Average them within each
    # representation seed, then form intervals over independent seeds.
    draw_keys = ("representation_seed", "feature_count", "bottleneck", "joint_activity", "task_id", "sample_size")
    draw_groups: dict[tuple[float, ...], list[dict[str, float | str]]] = {}
    for row in rows:
        draw_groups.setdefault(tuple(float(row[k]) for k in draw_keys), []).append(row)
    collapsed: list[dict[str, float | str]] = []
    excluded_draw = {"readout_draw", "task"}
    for key, group in draw_groups.items():
        row = dict(zip(draw_keys, key))
        row["task"] = group[0]["task"]
        metric_keys = [k for k in group[0] if k not in excluded_draw | set(draw_keys)]
        for metric in metric_keys:
            row[metric] = float(np.mean([float(item[metric]) for item in group]))
        collapsed.append(row)
    keys = ("feature_count", "bottleneck", "joint_activity", "task_id", "sample_size")
    means, stds, cis = aggregate_seed_rows(
        collapsed, group_keys=keys,
        excluded_keys={"representation_seed", "readout_draw", "task", *keys}, n_bootstrap=2000,
    )
    for group in (means, stds, cis):
        for row in group:
            row["task"] = next(k for k, v in TASK_ID.items() if v == int(row["task_id"]))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(a.output_dir/"raw.csv", rows); write_csv(a.output_dir/"mean.csv", means)
    write_csv(a.output_dir/"std.csv", stds); write_csv(a.output_dir/"ci95.csv", cis)
    write_csv(a.output_dir/"config.csv", [{
        "feature_counts": ",".join(map(str,a.feature_counts)), "bottlenecks": ",".join(map(str,a.bottlenecks)),
        "sample_sizes": ",".join(map(str,a.sample_sizes)), "joint_probabilities": ",".join(map(str,a.joint_probabilities)),
        "tasks": a.tasks, "marginal": a.marginal, "encoder_samples": a.encoder_samples,
        "num_seeds": a.num_seeds, "num_readout_draws": a.num_readout_draws, "epochs": a.epochs, "seed": a.seed,
    }])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, key, label in zip(axes.flat,
        ("probe_test_mse", "aliasing", "representation_bayes_mse", "feature_direction_coherence"),
        ("held-out risk", "GAD alias component", "representation Bayes risk", "feature-direction coherence")):
        for task in a.tasks.split(","):
            sel = [r for r in means if r["task"] == task and float(r["joint_activity"]) == a.joint_probabilities[-1]
                   and int(float(r["feature_count"])) == a.feature_counts[-1]
                   and int(float(r["bottleneck"])) == a.bottlenecks[0]]
            sel.sort(key=lambda r: float(r["sample_size"]))
            if sel: ax.plot([r["sample_size"] for r in sel],[r[key] for r in sel],marker="o",label=task)
        ax.set_xlabel("readout training sample count"); ax.set_ylabel(label); ax.grid(alpha=.25)
    axes[0,0].legend()
    fig.savefig(a.output_dir/"trained_superposition_aliasing.png",dpi=170,bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {len(rows)} rows to {a.output_dir}")


if __name__ == "__main__": main()
