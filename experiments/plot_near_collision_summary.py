"""Regenerate the near-collision sample-size figure from saved summaries.

This plotting-only utility reads mean.csv, ci95.csv, and config.csv from an
existing experiment 10 output directory. It does not rerun any simulations.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def close(a: float, b: float) -> bool:
    return bool(np.isclose(a, b, rtol=1e-12, atol=1e-14))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="existing experiment 10 output directory")
    parser.add_argument("--zero-tolerance", type=float, default=1e-12,
                        help="display numerical alias values below this threshold as zero")
    args = parser.parse_args()

    mean_rows = read_csv(args.output_dir / "mean.csv")
    ci_rows = read_csv(args.output_dir / "ci95.csv")
    config_rows = read_csv(args.output_dir / "config.csv")
    if len(config_rows) != 1:
        raise ValueError("config.csv must contain exactly one configuration row")
    config = config_rows[0]
    if not mean_rows or not ci_rows:
        raise ValueError("mean.csv and ci95.csv must contain results")
    required = {"test_mse", "aliasing", "data_algorithm"}
    if not required.issubset(mean_rows[0]):
        raise ValueError(f"mean.csv is missing columns: {sorted(required - set(mean_rows[0]))}")

    prior = min((float(x) for x in config["priors"].split(",")), key=lambda x: abs(x - 0.5))
    separation = max(float(x) for x in config["separations"].split(","))
    noises = [float(x) for x in config["noise_levels"].split(",")]
    sample_sizes = sorted(float(x) for x in config["sample_sizes"].split(","))
    ci_lookup = {
        (as_float(r, "separation"), as_float(r, "observation_noise"),
         as_float(r, "prior_positive"), as_float(r, "sample_size")): r
        for r in ci_rows
    }

    panels = (("test_mse", "finite-sample affine-readout risk"),
              ("aliasing", "GAD training-null-space alias component"),
              ("data_algorithm", "GAD data/algorithm component"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for noise in noises:
        selected = [r for r in mean_rows
                    if close(as_float(r, "observation_noise"), noise)
                    and close(as_float(r, "prior_positive"), prior)
                    and close(as_float(r, "separation"), separation)]
        selected.sort(key=lambda r: as_float(r, "sample_size"))
        if len(selected) != len(sample_sizes):
            raise ValueError(f"missing sample-size rows for noise={noise:g}")
        x = np.asarray([as_float(r, "sample_size") for r in selected])
        label = f"noise={noise:g}"
        for ax, (metric, ylabel) in zip(axes, panels):
            y = np.asarray([as_float(r, metric) for r in selected])
            interval_rows = [ci_lookup[(as_float(r, "separation"),
                                        as_float(r, "observation_noise"),
                                        as_float(r, "prior_positive"),
                                        as_float(r, "sample_size"))] for r in selected]
            low = np.asarray([as_float(r, f"{metric}_ci95_low") for r in interval_rows])
            high = np.asarray([as_float(r, f"{metric}_ci95_high") for r in interval_rows])
            if metric == "aliasing":
                # Full-rank two-coordinate affine features have no training
                # null space here; values around 1e-32 are floating-point zero.
                y[np.abs(y) < args.zero_tolerance] = 0.0
                low[np.abs(low) < args.zero_tolerance] = 0.0
                high[np.abs(high) < args.zero_tolerance] = 0.0
            line, = ax.plot(x, y, marker="o", label=label)
            ax.fill_between(x, low, high, color=line.get_color(), alpha=0.13)
            ax.set_xlabel("readout training sample count")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Sample-size convergence at separation={separation:g}, prior={prior:g}; pointwise 95% seed-bootstrap CIs")
    destination = args.output_dir / "near_collision_sample_convergence.png"
    fig.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Regenerated {destination} from mean.csv and ci95.csv; no simulations were run.")


if __name__ == "__main__":
    main()
