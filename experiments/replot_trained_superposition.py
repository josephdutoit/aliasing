"""Recover experiment 08's figure from saved mean.csv without training or Torch.

python experiments/replot_trained_superposition.py --results-dir outputs/.../08
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    with (args.results_dir / "mean.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        parser.error("mean.csv is empty")
    if float(rows[0].get("gad_definition_version", "0")) != 2:
        parser.error("legacy GAD labels: run relabel_gad_results.py into a new directory first")
    p = max(float(r["feature_count"]) for r in rows)
    rows = [r for r in rows if float(r["feature_count"]) == p]
    tasks = sorted({r["task"] for r in rows})
    metrics = ("test_mse", "model_insufficiency", "aliasing")
    fig, axes = plt.subplots(len(tasks), 3, figsize=(15, 4 * len(tasks)), squeeze=False, constrained_layout=True)
    for i, task in enumerate(tasks):
        groups = sorted({(float(r["bottleneck"]), float(r["joint_activity"])) for r in rows if r["task"] == task})
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            for m, q in groups:
                group = sorted((r for r in rows if r["task"] == task and float(r["bottleneck"]) == m and float(r["joint_activity"]) == q), key=lambda r: float(r["sample_size"]))
                ax.plot([float(r["sample_size"]) for r in group], [float(r[metric]) for r in group], marker="o", label=f"width {m:g}, joint {q:g}")
            ax.set(title=f"{task}: {metric}", xlabel="readout samples")
            if metric == "aliasing" and max(abs(float(r[metric])) for r in rows if r["task"] == task) < 1e-12:
                ax.set(title=f"{task}: aliasing (numerically zero)", ylim=(0, 1e-12))
            ax.grid(alpha=.25)
        axes[i, 0].legend(fontsize=7)
    fig.suptitle(f"Experiment 08: saved means, {p:g} possible features; maximum joint activity makes pair contrast identically zero")
    output = args.results_dir / "trained_superposition_aliasing.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
