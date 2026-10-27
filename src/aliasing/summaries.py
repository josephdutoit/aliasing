"""Seed-level summaries for simulation outputs."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np


def bootstrap_mean_interval(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 4000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a percentile-bootstrap interval for the mean of seed outcomes.

    This interval treats independent simulation seeds as the sampling units.
    It is pointwise and does not correct for selecting extrema across a sweep.
    """
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if array.size < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_bootstrap, array.size))
    bootstrap_means = array[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    return float(low), float(high)


def paired_difference_summary(
    differences: Iterable[float],
    *,
    seed: int = 0,
    n_bootstrap: int = 4000,
) -> dict[str, float]:
    """Summarize paired per-seed differences with a bootstrap interval."""
    values = np.asarray(list(differences), dtype=float)
    values = values[np.isfinite(values)]
    low, high = bootstrap_mean_interval(
        values, seed=seed, n_bootstrap=n_bootstrap
    )
    return {
        "num_pairs": float(values.size),
        "mean_difference": float(np.mean(values)) if values.size else float("nan"),
        "std_difference": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
        "ci95_low": low,
        "ci95_high": high,
    }


def aggregate_seed_rows(
    rows: Sequence[Mapping[str, float]],
    *,
    group_keys: Sequence[str],
    excluded_keys: set[str],
    seed: int = 20261019,
    n_bootstrap: int = 4000,
) -> tuple[list[dict[str, float]], list[dict[str, float]], list[dict[str, float]]]:
    """Aggregate seed-level rows into means, sample SDs, and pointwise CIs."""
    if not rows:
        raise ValueError("cannot summarize an empty result")
    grouped: dict[tuple[float, ...], list[Mapping[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(float(row[key]) for key in group_keys)].append(row)

    mean_rows: list[dict[str, float]] = []
    std_rows: list[dict[str, float]] = []
    ci_rows: list[dict[str, float]] = []
    for group_index, (group_values, group) in enumerate(sorted(grouped.items())):
        keys = [key for key in group[0] if key not in excluded_keys]
        mean_row = dict(zip(group_keys, group_values))
        std_row = dict(zip(group_keys, group_values))
        ci_row = dict(zip(group_keys, group_values))
        mean_row["num_runs"] = std_row["num_runs"] = ci_row["num_runs"] = float(len(group))
        for metric_index, key in enumerate(keys):
            values = np.asarray([row[key] for row in group], dtype=float)
            finite = values[np.isfinite(values)]
            mean_row[key] = float(np.mean(finite)) if finite.size else float("nan")
            std_row[key] = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
            low, high = bootstrap_mean_interval(
                values,
                seed=seed + group_index * 1009 + metric_index,
                n_bootstrap=n_bootstrap,
            )
            ci_row[f"{key}_ci95_low"] = low
            ci_row[f"{key}_ci95_high"] = high
        mean_rows.append(mean_row)
        std_rows.append(std_row)
        ci_rows.append(ci_row)
    return mean_rows, std_rows, ci_rows


def adjacent_paired_contrasts(
    rows: Sequence[Mapping[str, float]],
    *,
    level_key: str,
    match_keys: Sequence[str],
    excluded_keys: set[str],
    seed: int = 20261019,
    n_bootstrap: int = 4000,
) -> list[dict[str, float | str]]:
    """Compare adjacent sweep levels using matched seed-level outcomes."""
    if not rows:
        return []
    levels = sorted({float(row[level_key]) for row in rows})
    by_level: dict[float, dict[tuple[float, ...], Mapping[str, float]]] = {}
    for level in levels:
        by_level[level] = {
            tuple(float(row[key]) for key in match_keys): row
            for row in rows
            if float(row[level_key]) == level
        }
    metrics = [key for key in rows[0] if key not in excluded_keys | {level_key, *match_keys}]
    results: list[dict[str, float | str]] = []
    contrast_index = 0
    for left_level, right_level in zip(levels[:-1], levels[1:]):
        left_rows = by_level[left_level]
        right_rows = by_level[right_level]
        common = sorted(set(left_rows) & set(right_rows))
        for metric in metrics:
            differences = [right_rows[key][metric] - left_rows[key][metric] for key in common]
            summary = paired_difference_summary(
                differences,
                seed=seed + contrast_index,
                n_bootstrap=n_bootstrap,
            )
            results.append({
                "from_level": left_level,
                "to_level": right_level,
                "metric": metric,
                **summary,
            })
            contrast_index += 1
    return results
