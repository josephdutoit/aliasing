#!/usr/bin/env python3
"""Aggregate completed and partial Chapter 2 v3 shards into a review bundle."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime import CSV_FIELDS, NA, atomic_json, csv_bytes, file_lock, sha256_file, utc_now  # noqa: E402


METRICS = (
    "M_err", "A_err", "A_spec", "A_dep", "A_ind", "D0", "D_e", "S_e",
    "R_direct_pop", "R_direct_ls", "R_metric_min_norm", "R_oracle_nearest_interpolant",
    "R_euclidean_min_norm",
    "R_test_mc", "A_expected_gaussian", "D0_expected_gaussian",
    "rel_decomposition_error", "signed_identity_error", "cond_K", "min_eig_K", "trace_K_inv",
    "residual_second_moment", "A_normalized", "model_error_fraction",
    "reconstruction_loss", "bottom_eigenmode_share_A", "oracle_residual_feature_cross_norm",
    "oracle_residual_second_moment_se", "M_err_small_reference",
    "oracle_residual_feature_cross_norm_small_reference",
)
PLOT_NAMES = {
    "A": ("A_gaussian_theory", "A_gauge_invariance", "A_identity_checks"),
    "B": ("B_sparse_aliasing", "B_spectral_vs_dependence", "B_eigenmode_contributions", "B_failures"),
    "C": ("C_learned_vs_random", "C_kernel_vs_residual_dependence", "C_reconstruction_vs_aliasing", "C_metric_sensitivity"),
    "D1": ("D1_samplewise_total_risk", "D1_samplewise_gad_components", "D1_samplewise_rank_condition", "D1_samplewise_quantiles"),
    "D2": ("D2_featurewise_total_risk", "D2_featurewise_components", "D2_learned_vs_nested", "D2_peak_location_amplitude", "D2_model_insufficiency_vs_width"),
    "P": ("P_real_probe_transfer",),
}
FIGURE_INTERPRETATIONS = {
    "A_gaussian_theory": "Observed A and D0 are compared with their Gaussian analytic expectations away from undefined mean boundaries.",
    "A_gauge_invariance": "Absolute paired A differences show whether whitening the same feature row space preserves the intrinsic metric result.",
    "A_identity_checks": "Relative decomposition residuals expose numerical failures in the exact interpolating identity.",
    "B_sparse_aliasing": "Residual-scale-normalized aliasing is compared across exact-k input sparsities at matched second moments.",
    "B_spectral_vs_dependence": "A_spec, A_dep, and independent-residual A_ind separate kernel amplification from residual dependence.",
    "B_eigenmode_contributions": "The bottom kernel eigenmodes' share of A shows whether small-eigenvalue directions dominate aliasing.",
    "B_failures": "Per-cell invalid-trial fractions show where sparse designs become singular or numerically unstable.",
    "C_learned_vs_random": "Learned, whitened-gauge, Gaussian, and orthonormal controls compare normalized A under the same probe draws.",
    "C_kernel_vs_residual_dependence": "Spectral, dependence, and independent-residual terms compare the sources of learned-control aliasing.",
    "C_reconstruction_vs_aliasing": "Validation reconstruction quality is plotted against residual-scale-normalized readout aliasing.",
    "C_metric_sensitivity": "D_e and signed S_e show how the Euclidean readout metric changes risk relative to the population metric.",
    "D1_samplewise_total_risk": "Direct population risk traces the samplewise curve across actual n/d for each frozen representation control.",
    "D1_samplewise_gad_components": "A, D0, D_e, and S_e appear only where the exact interpolating decomposition applies.",
    "D1_samplewise_rank_condition": "Measured rank_H and whitened-design condition show the observed interpolation boundary and conditioning.",
    "D1_samplewise_quantiles": "Probe-level mean, median, and 5th–95th percentiles display spike variability with W-seed uncertainty in the CSV.",
    "D2_featurewise_total_risk": "Direct risk varies with actual readout width d at fixed downstream sample sizes.",
    "D2_featurewise_components": "Model insufficiency and interpolating GAD terms separate representation capacity from sample aliasing.",
    "D2_learned_vs_nested": "Independently learned widths are compared with truncations of a fixed learned or random feature bank.",
    "D2_peak_location_amplitude": "Risk peaks are located against actual d for each fixed n and representation design.",
    "D2_model_insufficiency_vs_width": "M_err shows how accessible target signal changes as the readout width increases.",
    "P_real_probe_transfer": "Held-out risk and estimated GAD components summarize the optional cached-activation transfer probe.",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float | None:
    if value in (None, "", NA, "NONFINITE", "None"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _key(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row.get(field, NA)) for field in fields)


def grouped_seed_means(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    keys = ("lane", "condition_id", "control_name", "representation_seed", "representation_family", "target_family",
            "input_law", "importance_profile", "N", "M", "d", "n", "noise_level",
            "active_count", "sparsity_p", "feature_order")
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("representation_seed") in (None, NA, ""):
            continue
        groups.setdefault(_key(row, keys), []).append(row)
    output: list[dict[str, Any]] = []
    for group_key, values in sorted(groups.items()):
        item: dict[str, Any] = dict(zip(keys, group_key))
        item["rows"] = len(values)
        item["valid_rows"] = sum(row.get("trial_valid") == "True" for row in values)
        item["invalid_rows"] = item["rows"] - item["valid_rows"]
        item["failure_fraction"] = item["invalid_rows"] / item["rows"] if item["rows"] else NA
        valid = [row for row in values if row.get("trial_valid") == "True"]
        ranks = [rank for row in valid if (rank := as_float(row.get("rank_H"))) is not None]
        item["mean_rank_H"] = float(np.mean(ranks)) if ranks else NA
        for metric in METRICS:
            observed = [number for row in valid if (number := as_float(row.get(metric))) is not None]
            item[metric] = float(np.mean(observed)) if observed else NA
        probe_risks = [number for row in valid if (number := as_float(row.get("R_direct_ls"))) is not None]
        item["sample_mean"] = float(np.mean(probe_risks)) if probe_risks else NA
        item["sample_median"] = float(np.median(probe_risks)) if probe_risks else NA
        item["sample_q05"] = float(np.quantile(probe_risks, 0.05)) if probe_risks else NA
        item["sample_q95"] = float(np.quantile(probe_risks, 0.95)) if probe_risks else NA
        output.append(item)
    return output


def failure_rates(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Count valid/invalid trials and failure states per predeclared cell."""
    keys = ("lane", "condition_id", "control_name", "target_family", "input_law",
            "importance_profile", "N", "M", "d", "n", "noise_level", "active_count",
            "sparsity_p", "feature_order", "n_fixed", "reference_size")
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(_key(row, keys), []).append(row)
    output = []
    for key, values in sorted(groups.items()):
        valid = sum(row.get("trial_valid") == "True" for row in values)
        invalid = sum(row.get("trial_valid") == "False" for row in values)
        attempted = valid + invalid
        states: dict[str, int] = {}
        for row in values:
            state = row.get("status", "UNKNOWN")
            states[state] = states.get(state, 0) + 1
        output.append(dict(zip(keys, key)) | {
            "trial_rows": len(values), "attempted_trial_rows": attempted,
            "valid_trial_rows": valid, "invalid_trial_rows": invalid,
            "unavailable_trial_rows": len(values) - attempted,
            "failure_fraction": invalid / attempted if attempted else NA,
            "status_counts": states,
        })
    return output


def summarize_seed_means(seed_rows: list[dict[str, Any]], bootstrap_replicates: int,
                         seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    group_fields = ("lane", "condition_id", "control_name", "representation_family", "target_family", "input_law",
                    "importance_profile", "N", "M", "d", "n", "noise_level", "active_count",
                    "sparsity_p", "feature_order")
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in seed_rows:
        groups.setdefault(_key(row, group_fields), []).append(row)
    means: list[dict[str, Any]] = []
    quantiles: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    for index, (group_key, values) in enumerate(sorted(groups.items())):
        base = dict(zip(group_fields, group_key))
        base["valid_representation_seeds"] = len({v["representation_seed"] for v in values
                                                   if int(v.get("valid_rows", 0)) > 0})
        for metric in METRICS:
            observed = [number for v in values if (number := as_float(v.get(metric))) is not None]
            if not observed:
                continue
            means.append(base | {"metric": metric, "mean": float(np.mean(observed)),
                                 "valid_seeds": len(observed)})
            quantiles.append(base | {
                "metric": metric, "mean": float(np.mean(observed)), "median": float(np.median(observed)),
                "q05": float(np.quantile(observed, 0.05)), "q95": float(np.quantile(observed, 0.95)),
                "valid_seeds": len(observed),
            })
            rng = np.random.default_rng(seed + index * 7919 + sum(map(ord, metric)))
            if len(observed) < 2 or bootstrap_replicates < 1:
                lower = upper = NA
            else:
                samples = rng.choice(observed, size=(bootstrap_replicates, len(observed)), replace=True)
                boot_means = np.mean(samples, axis=1)
                lower, upper = (float(value) for value in np.quantile(boot_means, [0.025, 0.975]))
            intervals.append(base | {"metric": metric, "mean": float(np.mean(observed)),
                                     "lower_95": lower, "upper_95": upper,
                                     "valid_seeds": len(observed), "bootstrap_unit": "representation_seed"})
    return means, quantiles, intervals


def gaussian_theory_points(rows: list[dict[str, str]], bootstrap_replicates: int,
                           seed: int) -> list[dict[str, Any]]:
    """Build W-level Lane A theory points and paired difference intervals."""
    selected = [row for row in rows if row.get("lane") == "A" and row.get("input_law") == "gaussian"]
    per_seed = grouped_seed_means(selected)
    keys = ("condition_id", "control_name", "representation_family", "target_family",
            "N", "M", "d", "n", "noise_level", "feature_order")
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in per_seed:
        groups.setdefault(_key(row, keys), []).append(row)
    output: list[dict[str, Any]] = []
    for group_index, (key, values) in enumerate(sorted(groups.items())):
        base = dict(zip(keys, key)) | {"lane": "A"}
        for label, observed_field, expected_field in (
                ("A_err", "A_err", "A_expected_gaussian"),
                ("D0", "D0", "D0_expected_gaussian")):
            paired = [(as_float(row.get(observed_field)), as_float(row.get(expected_field)))
                      for row in values]
            paired = [(observed, expected) for observed, expected in paired
                      if observed is not None and expected is not None]
            if not paired:
                continue
            observed_values = np.asarray([pair[0] for pair in paired], dtype=float)
            expected_values = np.asarray([pair[1] for pair in paired], dtype=float)
            difference_values = observed_values - expected_values
            if len(paired) >= 2 and bootstrap_replicates > 0:
                rng = np.random.default_rng(seed + group_index * 7907 + sum(map(ord, label)))
                sampled = rng.integers(0, len(paired), size=(bootstrap_replicates, len(paired)))
                boot_observed = np.mean(observed_values[sampled], axis=1)
                boot_difference = np.mean(difference_values[sampled], axis=1)
                obs_low, obs_high = (float(x) for x in np.quantile(boot_observed, [0.025, 0.975]))
                diff_low, diff_high = (float(x) for x in np.quantile(boot_difference, [0.025, 0.975]))
            else:
                obs_low = obs_high = diff_low = diff_high = NA
            expected_column = "A_expected_gaussian" if label == "A_err" else "D0_expected_gaussian"
            output.append(base | {
                "metric": label, expected_column: float(np.mean(expected_values)),
                "observed_mean": float(np.mean(observed_values)),
                "lower_95": obs_low, "upper_95": obs_high,
                "mean_difference": float(np.mean(difference_values)),
                "difference_lower_95": diff_low, "difference_upper_95": diff_high,
                "valid_representation_seeds": len(paired),
                "bootstrap_unit": "representation_seed",
            })
    return output


def d1_quantile_summary(rows: list[dict[str, str]], bootstrap_replicates: int,
                        seed: int) -> list[dict[str, Any]]:
    per_seed = grouped_seed_means([row for row in rows if row.get("lane") == "D1"])
    keys = ("condition_id", "control_name", "representation_family", "target_family", "input_law", "importance_profile",
            "sparsity_p", "N", "M", "d", "n", "noise_level", "feature_order")
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in per_seed:
        groups.setdefault(_key(row, keys), []).append(row)
    output: list[dict[str, Any]] = []
    stat_fields = ("sample_mean", "sample_median", "sample_q05", "sample_q95")
    for group_index, (key, values) in enumerate(sorted(groups.items())):
        base = dict(zip(keys, key))
        base["n_over_d"] = float(base["n"]) / float(base["d"])
        base["valid_representation_seeds"] = len({row["representation_seed"] for row in values
                                                   if int(row.get("valid_rows", 0)) > 0})
        for field in stat_fields:
            observed = np.asarray([number for row in values
                                   if (number := as_float(row.get(field))) is not None], dtype=float)
            if observed.size == 0:
                base[field] = NA
                base[f"{field}_lower_95"] = NA
                base[f"{field}_upper_95"] = NA
                continue
            base[field] = float(np.mean(observed))
            if observed.size >= 2 and bootstrap_replicates > 0:
                rng = np.random.default_rng(seed + group_index * 3571 + sum(map(ord, field)))
                sampled = rng.choice(observed, size=(bootstrap_replicates, observed.size), replace=True)
                low, high = (float(value) for value in np.quantile(np.mean(sampled, axis=1), [0.025, 0.975]))
                base[f"{field}_lower_95"], base[f"{field}_upper_95"] = low, high
            else:
                base[f"{field}_lower_95"] = base[f"{field}_upper_95"] = NA
        output.append(base)
    return output


def paired_contrasts(rows: list[dict[str, str]], bootstrap_replicates: int, seed: int) -> list[dict[str, Any]]:
    pair_fields = ("lane", "target_family", "input_law", "importance_profile", "N", "M", "d", "n",
                   "n_over_d", "noise_level", "active_count", "sparsity_p", "representation_seed",
                   "probe_seed", "feature_order")
    groups: dict[tuple[str, ...], dict[str, dict[str, str]]] = {}
    for row in rows:
        control = row.get("control_name", "")
        if not control:
            continue
        if row.get("lane") == "D2" and row.get("target_family") == "in_span":
            # The in-span teacher is representation-dependent for width-matched controls.
            continue
        if row.get("trial_valid") != "True":
            continue
        key = list(_key(row, pair_fields))
        groups.setdefault(tuple(key), {})[control] = row
    raw: list[dict[str, Any]] = []
    comparisons = (("learned", "random"), ("learned", "whitened_gauge"),
                   ("learned", "orthonormal"), ("nested_learned", "nested_random"),
                   ("independently_learned", "width_matched_random"))
    for key, controls in groups.items():
        for left, right in comparisons:
            if left not in controls or right not in controls:
                continue
            for metric in ("A_err", "A_normalized", "A_dep", "M_err", "R_direct_pop", "R_direct_ls", "R_test_mc"):
                lv, rv = as_float(controls[left].get(metric)), as_float(controls[right].get(metric))
                if lv is None or rv is None:
                    continue
                raw.append(dict(zip(pair_fields, key)) | {
                    "contrast": f"{left}-{right}", "left_control": left, "right_control": right,
                    "left_representation_family": controls[left].get("representation_family", NA),
                    "right_representation_family": controls[right].get("representation_family", NA),
                    "metric": metric, "left": lv, "right": rv, "difference": lv - rv,
                })
    # Add seed-level bootstrap intervals while preserving paired trial rows above.
    by_seed: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    ci_fields = ("lane", "contrast", "metric", "target_family", "N", "M", "d", "n", "noise_level",
                 "active_count", "sparsity_p", "feature_order", "left_representation_family",
                 "right_representation_family", "representation_seed")
    for row in raw:
        by_seed.setdefault(_key(row, ci_fields), []).append(row)
    compact: list[dict[str, Any]] = []
    for key, values in sorted(by_seed.items()):
        diffs = [float(row["difference"]) for row in values]
        compact.append(dict(zip(ci_fields, key)) | {
            "mean_difference": float(np.mean(diffs)), "probe_draws": len(diffs),
        })
    # Store paired per-seed means; the collector's CI file is representation-seed based.
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    summary_fields = ci_fields[:-1]
    for row in compact:
        grouped.setdefault(_key(row, summary_fields), []).append(row)
    for i, (key, values) in enumerate(sorted(grouped.items())):
        numbers = [float(row["mean_difference"]) for row in values]
        if len(numbers) >= 2 and bootstrap_replicates > 0:
            rng = np.random.default_rng(seed + i * 6151)
            boot = np.mean(rng.choice(numbers, size=(bootstrap_replicates, len(numbers)), replace=True), axis=1)
            low, high = (float(value) for value in np.quantile(boot, [0.025, 0.975]))
        else:
            low = high = NA
        compact.append(dict(zip(summary_fields, key)) | {
            "mean_difference": float(np.mean(numbers)), "lower_95": low, "upper_95": high,
            "valid_representation_seeds": len(numbers), "bootstrap_unit": "representation_seed",
        })
    return raw + compact


def gaussian_sparse_contrasts(rows: list[dict[str, str]], bootstrap_replicates: int,
                              seed: int) -> list[dict[str, Any]]:
    """Paired Lane B k=N versus Lane A Gaussian control, sharing W/data/noise seeds."""
    key_fields = ("N", "M", "d", "n", "n_over_d", "sigma_epsilon", "target_family",
                  "representation_family", "representation_seed", "probe_seed")
    buckets: dict[tuple[str, ...], dict[str, dict[str, str]]] = {}
    for row in rows:
        if row.get("trial_valid") != "True":
            continue
        lane = row.get("lane")
        if lane == "A" and row.get("control_name") == "learned":
            group = buckets.setdefault(_key(row, key_fields), {})
            group["A"] = row
        elif lane == "B" and as_float(row.get("active_count")) == as_float(row.get("N")):
            group = buckets.setdefault(_key(row, key_fields), {})
            group["B"] = row
    output: list[dict[str, Any]] = []
    for key, group in buckets.items():
        if "A" not in group or "B" not in group:
            continue
        for metric in ("A_err", "A_spec", "A_dep", "A_normalized", "M_err", "D0", "D_e", "S_e"):
            left, right = as_float(group["B"].get(metric)), as_float(group["A"].get(metric))
            if left is None or right is None:
                continue
            output.append(dict(zip(key_fields, key)) | {
                "lane": "B", "contrast": "B_kN-A_Gaussian", "metric": metric,
                "left_control": "sparse_k=N", "right_control": "Gaussian",
                "left": left, "right": right, "difference": left - right,
                "paired_representation_seed": group["A"].get("representation_seed"),
                "paired_probe_seed": group["A"].get("probe_seed"),
            })
    per_seed_fields = ("contrast", "metric", "N", "M", "d", "n", "n_over_d",
                       "sigma_epsilon", "target_family", "representation_family", "control_name",
                       "representation_seed")
    per_seed: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in output:
        per_seed.setdefault(_key(row, per_seed_fields), []).append(row)
    seed_means = [dict(zip(per_seed_fields, key)) | {
        "mean_difference": float(np.mean([float(row["difference"]) for row in values])),
        "probe_draws": len(values),
    } for key, values in per_seed.items()]
    summary_fields = per_seed_fields[:-1]
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in seed_means:
        groups.setdefault(_key(row, summary_fields), []).append(row)
    summaries: list[dict[str, Any]] = []
    for index, (key, values) in enumerate(sorted(groups.items())):
        differences = [float(row["mean_difference"]) for row in values]
        if len(differences) >= 2 and bootstrap_replicates > 0:
            rng = np.random.default_rng(seed + index * 4001)
            bootstrap = np.mean(rng.choice(differences, size=(bootstrap_replicates, len(differences)),
                                           replace=True), axis=1)
            low, high = (float(value) for value in np.quantile(bootstrap, [0.025, 0.975]))
        else:
            low = high = NA
        summaries.append(dict(zip(summary_fields, key)) | {
            "mean_difference": float(np.mean(differences)), "lower_95": low, "upper_95": high,
            "valid_representation_seeds": len(differences), "bootstrap_unit": "representation_seed",
        })
    return output + seed_means + summaries


def summarized_contrasts(contrasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in contrasts
            if row.get("valid_representation_seeds") not in (None, NA, "")
            and row.get("representation_seed") in (None, NA, "")]


def format_contrast(row: dict[str, Any]) -> str:
    effect = as_float(row.get("mean_difference"))
    low, high = as_float(row.get("lower_95")), as_float(row.get("upper_95"))
    interval = f"95% CI [{low:.4g}, {high:.4g}]" if low is not None and high is not None else "95% CI unavailable"
    seeds = row.get("valid_representation_seeds", "NA")
    condition = (f"target={row.get('target_family', 'NA')}, n={row.get('n', 'NA')}, "
                 f"d={row.get('d', 'NA')}, M={row.get('M', 'NA')}, "
                 f"law={row.get('input_law', 'NA')}, p={row.get('sparsity_p', 'NA')}, "
                 f"importance={row.get('importance_profile', 'NA')}, "
                 f"noise={row.get('noise_level', row.get('sigma_epsilon', 'NA'))}")
    return (f"`{row.get('contrast', 'comparison')}` on `{row.get('metric', 'metric')}`: "
            f"mean paired difference {effect:.4g} ({interval}; {seeds} representation seeds), "
            f"at {condition}. Alternative explanation: "
            "residual scale, model error, or numerical conditioning may account for the contrast.") if effect is not None else "not established"


def peak_description(rows: list[dict[str, str]], lane: str) -> str:
    metric = "R_direct_ls"
    grouped: dict[tuple[str, ...], list[tuple[float, float | None]]] = {}
    seed_rows = grouped_seed_means([row for row in rows if row.get("lane") == lane])
    for row in seed_rows:
        value, n, width, features = (as_float(row.get(metric)), as_float(row.get("n")),
                                     as_float(row.get("M")), as_float(row.get("d")))
        if value is None or n is None or width is None or features is None:
            continue
        control = row.get("control_name", "unspecified")
        key = (control, row.get("target_family", NA), row.get("input_law", NA),
               row.get("importance_profile", NA), row.get("sparsity_p", NA),
               row.get("noise_level", NA), str(int(n)), str(int(width)), str(int(features)))
        rank = as_float(row.get("mean_rank_H"))
        grouped.setdefault(key, []).append((value, rank))
    peaks = []
    for key, values in grouped.items():
        risks = [value for value, _ in values]
        ranks = [rank for _, rank in values if rank is not None]
        peaks.append((float(np.mean(risks)), key, float(np.median(risks)),
                      float(np.mean(ranks)) if ranks else None, len(values)))
    if not peaks:
        return "not established: no completed direct-risk rows"
    descriptions = []
    for mean, key, median, rank, count in sorted(peaks, reverse=True)[:4]:
        n, width, features = key[6], key[7], key[8]
        rank_text = f", mean rank_H={rank:.2f}" if rank is not None else ""
        descriptions.append(f"{key[0]} target={key[1]}, p={key[4]}, noise={key[5]}: n={n}, M={width}, d={features}{rank_text}, mean={mean:.4g}, median={median:.4g}, W seeds={count}")
    return "; ".join(descriptions)


def _plot_source_rows(lane: str, rows: list[dict[str, str]], metric: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("lane") == lane
                and row.get("trial_valid") == "True" and as_float(row.get(metric)) is not None]
    selected.sort(key=lambda row: (as_float(row.get("n_over_d")) or 0.0,
                                   row.get("control_name", ""), row.get("representation_seed", "")))
    return selected


def gauge_invariance_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Pair learned and whitened-gauge rows for the Lane A invariant plot."""
    pair_fields = ("representation_family", "target_family", "representation_seed", "probe_seed",
                   "draw_index", "input_law", "importance_profile", "N", "M", "n", "d",
                   "n_over_d", "noise_level")
    buckets: dict[tuple[str, ...], dict[str, dict[str, str]]] = {}
    for row in rows:
        control = row.get("control_name")
        if row.get("lane") != "A" or row.get("trial_valid") != "True":
            continue
        if control not in ("learned", "whitened_gauge"):
            continue
        buckets.setdefault(_key(row, pair_fields), {})[control] = row
    paired: list[dict[str, Any]] = []
    for key, controls in sorted(buckets.items()):
        if "learned" not in controls or "whitened_gauge" not in controls:
            continue
        learned = as_float(controls["learned"].get("A_err"))
        whitened = as_float(controls["whitened_gauge"].get("A_err"))
        if learned is None or whitened is None:
            continue
        paired.append(dict(zip(pair_fields, key)) | {
            "lane": "A", "control_name": "learned-vs-whitened_gauge",
            "A_learned": learned, "A_whitened_gauge": whitened,
            "A_difference": learned - whitened,
            "absolute_A_difference": abs(learned - whitened),
            "trial_valid": "True", "gad_definition": "population_metric_v3",
            "is_oracle_exact": "True",
        })
    return paired


def write_gauge_invariance_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    groups: dict[str, dict[float, list[float]]] = {}
    for row in rows:
        x = as_float(row.get("n_over_d"))
        error = as_float(row.get("absolute_A_difference"))
        if x is not None and error is not None:
            groups.setdefault(_series_label(row), {}).setdefault(x, []).append(error)
    for label, points in sorted(groups.items()):
        ordered = sorted((x, float(np.mean(values))) for x, values in points.items())
        ax.plot([x for x, _ in ordered], [y for _, y in ordered], marker="o", label=label)
    if groups:
        ax.legend(title="Paired condition", fontsize="small")
    else:
        ax.text(0.5, 0.5, "No valid learned/gauge pairs collected", ha="center", va="center",
                transform=ax.transAxes)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("n / actual readout feature count d")
    ax.set_ylabel("Absolute paired difference in A")
    ax.set_title(f"Lane A whitened-gauge invariance ({len(rows)} paired rows; exact oracle)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_d1_rank_condition_plot(path: Path, rows: list[dict[str, str]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row.get("trial_valid") == "True"
                and as_float(row.get("rank_H")) is not None]
    fig, (rank_ax, condition_ax) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    grouped: dict[str, dict[int, list[dict[str, str]]]] = {}
    for row in selected:
        n = as_float(row.get("n"))
        if n is not None:
            label = f"{_series_label(row)}, d={row.get('d', 'NA')}"
            grouped.setdefault(label, {}).setdefault(int(n), []).append(row)
    for label, by_n in sorted(grouped.items()):
        ns = sorted(by_n)
        ranks = [float(np.mean([value for row in by_n[n]
                                if (value := as_float(row.get("rank_H"))) is not None])) for n in ns]
        conditions = []
        for n in ns:
            values = [value for row in by_n[n]
                      if (value := as_float(row.get("condition_H_whitened"))) is not None and value > 0]
            conditions.append(float(np.median(values)) if values else float("nan"))
        rank_ax.plot(ns, ranks, marker="o", linewidth=1.1, label=label)
        condition_ax.plot(ns, conditions, marker=".", linewidth=1.0, label=label)
    if grouped:
        condition_ax.set_yscale("log")
        condition_ax.legend(title="Method / readout width", fontsize="xx-small", ncol=2)
        maximum_n = max((max(points) for points in grouped.values()), default=1)
        rank_ax.plot([0, maximum_n], [0, maximum_n], linestyle="--", color="black",
                     linewidth=0.8, label="rank = n")
        rank_ax.legend(title="Observed rank_H", fontsize="xx-small", ncol=2)
    else:
        rank_ax.text(0.5, 0.5, "No valid rank rows collected", ha="center", va="center",
                     transform=rank_ax.transAxes)
    rank_ax.set_ylabel("Observed rank_H")
    rank_ax.set_title(f"Lane D1 actual rank and whitened-design conditioning ({len(selected)} valid trials)")
    condition_ax.set_xlabel("Actual training sample count n")
    condition_ax.set_ylabel("Median condition of whitened H")
    rank_ax.grid(True, alpha=0.25)
    condition_ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _series_label(row: dict[str, str]) -> str:
    pieces = [row.get("control_name", "unspecified")]
    if row.get("lane") == "A" and row.get("representation_family") not in (None, "", NA):
        pieces.append(f"family={row['representation_family']}")
    if row.get("lane") not in ("D2", "P") and row.get("M") not in (None, "", NA):
        pieces.append(f"M={row['M']}")
    for field, label in (("target_family", "target"), ("noise_level", "noise"),
                         ("active_count", "k"), ("feature_order", "order"),
                         ("n_fixed", "n"), ("sparsity_p", "p"),
                         ("importance_profile", "importance"), ("reference_size", "reference")):
        value = row.get(field)
        if value not in (None, "", NA, "native") or (field == "feature_order" and value == "native"):
            pieces.append(f"{label}={value}")
    return ", ".join(pieces)


def write_plot(path: Path, title: str, rows: list[dict[str, str]], x_field: str,
               y_field: str, ylabel: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    plot_rows = [row for row in rows if "trial_valid" not in row or row.get("trial_valid") == "True"]
    if plot_rows:
        groups: dict[str, dict[float, list[float]]] = {}
        for row in plot_rows:
            name = _series_label(row)
            x = as_float(row.get(x_field))
            y = as_float(row.get(y_field))
            if x is not None and y is not None:
                groups.setdefault(name, {}).setdefault(x, []).append(y)
        for name, values in sorted(groups.items()):
            points = sorted((x, float(np.mean(ys))) for x, ys in values.items())
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", linewidth=1.3,
                        alpha=0.75, label=name)
        ax.legend(title="Method", fontsize="small")
        count = len(plot_rows)
        exact = "estimated oracle" if any(r.get("oracle_estimated") == "True" for r in plot_rows) else "exact oracle"
        ns = sorted({int(float(r["n"])) for r in plot_rows if as_float(r.get("n")) is not None})
        n_text = f"; n={ns[0]}–{ns[-1]}" if ns else ""
        ax.set_title(f"{title} ({count} trial rows{n_text}; {exact})")
    else:
        ax.text(0.5, 0.5, "No completed trial rows collected", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(f"{title} (no results yet)")
    ax.set_xlabel("n / actual readout feature count d" if x_field == "n_over_d" else x_field)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_multi_plot(path: Path, title: str, rows: list[dict[str, str]], x_field: str,
                     y_fields: tuple[str, ...]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    grouped: dict[tuple[str, str], dict[float, list[float]]] = {}
    for row in rows:
        if "trial_valid" in row and row.get("trial_valid") != "True":
            continue
        method = _series_label(row)
        for metric in y_fields:
            x, y = as_float(row.get(x_field)), as_float(row.get(metric))
            if x is not None and y is not None:
                grouped.setdefault((method, metric), {}).setdefault(x, []).append(y)
    for (method, metric), values in sorted(grouped.items()):
        points = sorted((x, float(np.mean(ys))) for x, ys in values.items())
        ax.plot([x for x, _ in points], [y for _, y in points], marker="o", linewidth=1.2,
                alpha=0.75, label=f"{method}: {metric}")
    if grouped:
        ax.legend(title="Method / metric", fontsize="x-small")
    else:
        ax.text(0.5, 0.5, "No completed trial rows collected", ha="center", va="center",
                transform=ax.transAxes)
    exact = "estimated oracle" if any(row.get("oracle_estimated") == "True" for row in rows) else "exact oracle"
    ax.set_title(f"{title} ({len(rows)} trial rows; {exact})")
    ax.set_xlabel("n / actual readout feature count d" if x_field == "n_over_d" else x_field)
    ax.set_ylabel("Population risk component")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_theory_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    series = (
        ("A", "A_expected_gaussian", "observed_mean", "A_err"),
        ("D0", "D0_expected_gaussian", "observed_mean", "D0"),
    )
    point_count = 0
    for ax, (label, expected_key, observed_key, metric) in zip(axes, series):
        selected = [row for row in rows if row.get("metric") == metric
                    and as_float(row.get(expected_key)) is not None
                    and as_float(row.get(observed_key)) is not None]
        point_count += len(selected)
        if selected:
            values = [as_float(row[key]) for row in selected for key in (expected_key, observed_key)]
            low, high = min(value for value in values if value is not None), max(
                value for value in values if value is not None)
            by_method: dict[str, list[dict[str, Any]]] = {}
            for row in selected:
                by_method.setdefault(_series_label(row), []).append(row)
            for method, method_rows in sorted(by_method.items()):
                method_rows.sort(key=lambda row: as_float(row.get(expected_key)) or 0.0)
                xs = [float(row[expected_key]) for row in method_rows]
                ys = [float(row[observed_key]) for row in method_rows]
                bounds = [(as_float(row.get("lower_95")), as_float(row.get("upper_95")))
                          for row in method_rows]
                error = None
                if all(lower is not None and upper is not None for lower, upper in bounds):
                    error = np.asarray([[max(0.0, y - float(lower)) for y, (lower, _) in zip(ys, bounds)],
                                        [max(0.0, float(upper) - y) for y, (_, upper) in zip(ys, bounds)]])
                ax.errorbar(xs, ys, yerr=error, marker="o", alpha=0.75, capsize=2, label=method)
            ax.plot([low, high], [low, high], linestyle="--", color="black", label="y=x")
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) <= 12:
                ax.legend(fontsize="x-small")
        else:
            ax.text(0.5, 0.5, "No rows satisfy this analytic condition", ha="center",
                    va="center", transform=ax.transAxes)
        ax.set_xlabel(f"Gaussian analytic E[{label}]")
        ax.set_ylabel(f"Observed {label}")
        ax.grid(True, alpha=0.25)
    seeds = max((int(row.get("valid_representation_seeds", 0)) for row in rows), default=0)
    fig.suptitle(f"Gaussian theory checks ({point_count} condition points; up to {seeds} independent W seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_d1_quantiles_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    grouped: dict[str, dict[float, list[dict[str, Any]]]] = {}
    for row in rows:
        x = as_float(row.get("n_over_d"))
        if x is not None:
            grouped.setdefault(_series_label(row), {}).setdefault(x, []).append(row)
    for name, points in sorted(grouped.items()):
        merged = []
        for x, values in sorted(points.items()):
            merged.append((x, {field: float(np.mean([number for row in values
                if (number := as_float(row.get(field))) is not None]))
                for field in ("sample_mean", "sample_median", "sample_q05", "sample_q95")
                if any(as_float(row.get(field)) is not None for row in values)}))
        mean_points = [(x, value["sample_mean"]) for x, value in merged if "sample_mean" in value]
        median_points = [(x, value["sample_median"]) for x, value in merged if "sample_median" in value]
        if mean_points:
            ax.plot([x for x, _ in mean_points], [y for _, y in mean_points], linestyle="--",
                    marker=".", linewidth=1.0, alpha=0.7, label=f"{name}: mean")
        if median_points:
            ax.plot([x for x, _ in median_points], [y for _, y in median_points], marker="o",
                    linewidth=1.4, label=f"{name}: median")
        band = [(x, values["sample_q05"], values["sample_q95"])
                for x, values in merged if "sample_q05" in values and "sample_q95" in values]
        if band:
            ax.fill_between([point[0] for point in band], [point[1] for point in band],
                            [point[2] for point in band], alpha=0.12)
    if grouped:
        ax.legend(fontsize="xx-small", ncol=2)
    else:
        ax.text(0.5, 0.5, "No valid probe risks collected", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("n / actual readout feature count d")
    ax.set_ylabel("Direct population risk")
    ax.set_title("Lane D1 probe mean, median, and within-representation 5th–95th percentiles")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def create_figures(run_dir: Path, lane_rows: dict[str, list[dict[str, str]]],
                   plan: dict[str, Any]) -> list[Path]:
    generated: list[Path] = []
    for lane, names in PLOT_NAMES.items():
        rows = lane_rows.get(lane, [])
        for index, name in enumerate(names):
            metric_by_lane = {
                "A": ("A_err", "A_err", "rel_decomposition_error"),
                "B": ("A_normalized", "A_dep", "bottom_eigenmode_share_A", "trial_valid"),
                "C": ("A_normalized", "A_dep", "A_err", "S_e"),
                "D1": ("R_direct_ls", "D_e", "cond_K", "R_direct_ls"),
                "D2": ("R_direct_ls", "A_err", "R_direct_ls", "R_direct_ls", "M_err"),
                "P": ("R_direct_ls",),
            }
            metric = metric_by_lane[lane][index]
            source = _plot_source_rows(lane, rows, metric)
            x_field = "d" if lane == "D2" else "n_over_d"
            if lane == "B" and index == 3:
                source = rows
                metric = "trial_valid"
            plot_path = run_dir / "figures" / f"{name}.png"
            if lane == "A" and index == 0:
                source = gaussian_theory_points(rows, int(plan["config"].get("bootstrap_replicates", 0)),
                                                int(plan["config"]["seed"]))
                write_theory_plot(plot_path, source)
            elif lane == "A" and index == 1:
                source = gauge_invariance_rows(rows)
                write_gauge_invariance_plot(plot_path, source)
            elif lane == "B" and index == 1:
                source = [row for row in rows if as_float(row.get("A_spec")) is not None]
                write_multi_plot(plot_path, name.replace("_", " "), source, "n_over_d", ("A_spec", "A_dep", "A_ind"))
            elif lane == "C" and index == 1:
                source = [row for row in rows if as_float(row.get("A_spec")) is not None]
                write_multi_plot(plot_path, name.replace("_", " "), source, "n_over_d", ("A_spec", "A_dep", "A_ind"))
            elif lane == "D1" and index == 1:
                source = rows
                write_multi_plot(plot_path, name.replace("_", " "), source, "n_over_d", ("A_err", "D0", "D_e", "S_e"))
            elif lane == "D1" and index == 2:
                source = rows
                write_d1_rank_condition_plot(plot_path, source)
            elif lane == "D2" and index == 1:
                source = rows
                write_multi_plot(plot_path, name.replace("_", " "), source, "d", ("M_err", "A_err", "D0", "D_e"))
            elif lane == "C" and index == 2:
                source = [row for row in rows if as_float(row.get("reconstruction_loss")) is not None]
                write_plot(plot_path, name.replace("_", " "), source, "reconstruction_loss", "A_normalized", "A / residual second moment")
            elif lane == "B" and index == 3:
                source = failure_rates(rows)
                x_field = "active_count"
                write_plot(plot_path, name.replace("_", " "), source, x_field,
                           "failure_fraction", "failure fraction")
            else:
                if lane == "D1" and index == 0:
                    metric = "R_direct_ls"
                    source = _plot_source_rows(lane, rows, metric)
                elif lane == "D1" and index == 2:
                    metric = "condition_H_whitened"
                    source = _plot_source_rows(lane, rows, metric)
                elif lane == "D1" and index == 3:
                    source = d1_quantile_summary(rows,
                        int(plan["config"].get("bootstrap_replicates", 0)), int(plan["config"]["seed"]))
                    write_d1_quantiles_plot(plot_path, source)
                    metric = "sample_median"
                elif lane == "D2" and index in (0, 2, 3):
                    metric = "R_direct_ls"
                    source = _plot_source_rows(lane, rows, metric)
                elif lane == "D2" and index == 4:
                    metric = "M_err"
                    source = _plot_source_rows(lane, rows, metric)
                elif lane == "A" and index == 2:
                    metric = "rel_decomposition_error"
                    source = _plot_source_rows(lane, rows, metric)
                elif lane == "C" and index == 3:
                    source = rows
                    write_multi_plot(plot_path, name.replace("_", " "), source, "n_over_d", ("D_e", "S_e"))
                    metric = "S_e"
                else:
                    if lane == "A" and index == 1:
                        metric = "A_err"
                        source = [row for row in rows if row.get("control_name") in ("learned", "whitened_gauge")]
                    write_plot(plot_path, name.replace("_", " "), source, x_field, metric, metric)
                if not (lane == "C" and index == 3) and not (lane == "D1" and index == 3):
                    write_plot(plot_path, name.replace("_", " "), source, x_field, metric, metric)
            figure_data = run_dir / "figure_data" / f"{name}.csv"
            figure_data.parent.mkdir(parents=True, exist_ok=True)
            fields = list(dict.fromkeys(key for row in source for key in row)) if source else CSV_FIELDS
            atomic_bytes = csv_bytes(source, fields=fields)
            atomic_write(figure_data, atomic_bytes)
            lane_figure = run_dir / "lanes" / lane / "figures" / f"{name}.png"
            lane_data = run_dir / "lanes" / lane / "figure_data" / f"{name}.csv"
            lane_figure.parent.mkdir(parents=True, exist_ok=True)
            lane_data.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(plot_path, lane_figure)
            shutil.copy2(figure_data, lane_data)
            generated.extend((plot_path, figure_data))
    return generated


def report_files(run_dir: Path, plan: dict[str, Any], manifest: dict[str, Any],
                 all_rows: list[dict[str, str]], seed_means: list[dict[str, Any]],
                 contrasts: list[dict[str, Any]]) -> None:
    lane_counts = manifest["lane_status_counts"]
    total_rows = len(all_rows)
    valid_rows = sum(row.get("trial_valid") == "True" for row in all_rows)
    completed_jobs = sum(value.get("status") in ("COMPLETED", "SKIPPED_DEPENDENCY")
                         for value in manifest["jobs"])
    expected_jobs = len(manifest["jobs"])
    actual_completed_lanes = ", ".join(f"{lane}: {counts}" for lane, counts in sorted(lane_counts.items())) or "none"
    partial = manifest["status"] == "PARTIAL"
    failures = sum(row.get("trial_valid") == "False" for row in all_rows)
    contrast_rows = summarized_contrasts(contrasts)
    def select_contrast(lane: str, contrast: str | None = None,
                        metric: str | None = None) -> dict[str, Any] | None:
        candidates = [row for row in contrast_rows if row.get("lane") == lane
                      and (contrast is None or row.get("contrast") == contrast)
                      and (metric is None or row.get("metric") == metric)]
        return max(candidates, key=lambda row: abs(as_float(row.get("mean_difference")) or 0.0)) if candidates else None

    theory_rows = gaussian_theory_points(all_rows, int(plan["config"].get("bootstrap_replicates", 0)),
                                         int(plan["config"]["seed"]))
    a_theory = [row for row in theory_rows if row.get("metric") == "A_err"]
    d0_theory = [row for row in theory_rows if row.get("metric") == "D0"]
    a_difference = float(np.mean([row["mean_difference"] for row in a_theory])) if a_theory else None
    d0_difference = float(np.mean([row["mean_difference"] for row in d0_theory])) if d0_theory else None
    def interval_envelope(values: list[dict[str, Any]], low_field: str, high_field: str) -> str:
        lows = [value for row in values if (value := as_float(row.get(low_field))) is not None]
        highs = [value for row in values if (value := as_float(row.get(high_field))) is not None]
        return f"condition-level 95% W-seed intervals span [{min(lows):.4g}, {max(highs):.4g}]" if lows and highs else "95% intervals unavailable"
    finding_a = (f"Lane A: observed-minus-analytic A = {a_difference:.4g} across {len(a_theory)} eligible condition points "
                 f"({interval_envelope(a_theory, 'difference_lower_95', 'difference_upper_95')}); "
                 f"D0 difference = {d0_difference:.4g} across {len(d0_theory)} points "
                 f"({interval_envelope(d0_theory, 'difference_lower_95', 'difference_upper_95')}).") if a_difference is not None else "Lane A Gaussian analytic agreement: not established; no eligible finite-mean rows."
    finding_b_row = select_contrast("B", "B_kN-A_Gaussian", "A_normalized") or select_contrast(
        "B", "B_kN-A_Gaussian", "A_spec")
    finding_b = f"Lane B matched k=N contrast: {format_contrast(finding_b_row)}" if finding_b_row else "Lane B matched-second-moment contrast: not established."
    finding_c_row = select_contrast("C", "learned-random", "A_normalized") or select_contrast(
        "C", "learned-random", "A_err")
    finding_c = f"Lane C learned-versus-random: {format_contrast(finding_c_row)}" if finding_c_row else "Lane C learned-versus-random contrast: not established."
    finding_d1_row = select_contrast("D1", "learned-random", "R_direct_ls")
    finding_d1 = f"Lane D1 learned-versus-random total risk: {format_contrast(finding_d1_row)}" if finding_d1_row else "Lane D1 learned-versus-random risk contrast: not established."
    finding_d2_row = select_contrast("D2", "independently_learned-width_matched_random", "R_direct_ls")
    finding_d2 = f"Lane D2 independent learned-width versus width-matched random: {format_contrast(finding_d2_row)}" if finding_d2_row else "Lane D2 paired width contrast: not established."
    evidence = {"A": finding_a, "B": finding_b, "C": finding_c, "D1": finding_d1, "D2": finding_d2}
    top_contrasts = sorted(contrast_rows, key=lambda row: abs(as_float(row.get("mean_difference")) or 0.0), reverse=True)[:5]
    highlight_text = "\n".join(f"- {format_contrast(row)}" for row in top_contrasts) or "- No completed paired contrast is available."
    report_status = "PARTIAL" if partial else "COMPLETE"
    multiplicity_plan = plan["config"].get("multiplicity_plan", "No multiplicity plan was recorded in this profile.")

    morning = f"""# Chapter 2 v3 morning readme

**Status:** {report_status}. **Run:** `{run_dir.name}`. **Profile:** `{plan['profile']}`.

This folder contains {total_rows} aggregated trial rows from {completed_jobs}/{expected_jobs} planned jobs. {valid_rows} trial rows are valid and {failures} are explicitly marked failed or invalid. Per-lane job status: {actual_completed_lanes}. Lane P is optional and is recorded as `SKIPPED_DEPENDENCY` when no activation cache is configured.

No experimental claim should be made from this report unless actual completed rows are present. Inspect `MANIFEST.json`, `AUDIT_PASS.json`, per-cell `failure_rates.csv`, each lane's `run_manifest.json`, `seed_means.csv`, `paired_contrasts.csv`, and the source CSVs in `figure_data/`. The independent unit is the representation seed; probes are repeated measurements. Exact synthetic oracles and estimated Lane P oracles are separately marked.

The planner records resource availability as `NOT_QUERIED` unless Slurm metadata is observed in actual jobs. This code does not infer cluster partitions, memory, GPU availability, or scratch locations.
"""
    (run_dir / "MORNING_README.md").write_text(morning, encoding="utf-8")

    audit = json.loads((run_dir / "AUDIT_PASS.json").read_text(encoding="utf-8")) if (run_dir / "AUDIT_PASS.json").is_file() else {"status": "NOT_RUN"}
    audit_status = audit.get("status", "NOT_RUN")
    key_contrasts = len(contrast_rows)
    decision = f"""# Decision memo: Chapter 2 v3

## 1. Executive verdict

{('No scientific verdict is established because one or more planned jobs are missing or failed.' if partial else 'The collected evidence is descriptive; causal or mechanism claims require the registered contrasts and completed audit.')}
The available results contain {total_rows} rows and {key_contrasts} representation-seed paired-contrast summaries. They do not establish a publishable lead unless the R-stage audit passed and the discovery/confirmation distinction is satisfied. The working verdict is **not established**.

## 2. What actually ran

- Planned profile: `{plan['profile']}`; plan hash: `{plan['plan_hash']}`; Git SHA: `{plan['git']['git_sha']}`; code hash: `{plan['code_sha256']}`.
- Jobs: {completed_jobs}/{expected_jobs} completed or intentionally skipped; trial rows: {total_rows}; invalid/failed rows: {failures}.
- Per-lane job statuses: {actual_completed_lanes}.
- Observed Slurm resources are recorded in shard provenance. No resource value is inferred from a profile target.
- Confirmation is a fresh seed range only if this profile is `confirmation`; otherwise findings remain exploratory.
- Multiplicity plan: {multiplicity_plan}

## 3. Validated foundations

R-stage audit status is `{audit_status}`. Read `AUDIT_PASS.json` for check-by-check results and numerical maxima. This collector does not treat a successful synthetic identity as evidence that the full experiment ran. Gauge invariance, Gaussian inverse-Wishart agreement, sparse-moment checks, and decomposition errors must be supported by measured audit outputs.

## 4. Five key findings

1. **Gaussian benchmark:** {evidence['A']} See `A_gaussian_theory.png` and `figure_data/A_gaussian_theory.csv`.
2. **Sparse inputs:** {evidence['B']} See `B_sparse_aliasing.png` and `B_spectral_vs_dependence.png`.
3. **Learned versus random:** {evidence['C']} See `C_learned_vs_random.png`; model insufficiency, actual rank and residual-scale alternatives are in the paired table.
4. **Samplewise double descent:** {evidence['D1']} Peak candidates: {peak_description(all_rows, 'D1')}. See `D1_samplewise_rank_condition.png` and the `is_interpolating` applicability mask.
5. **Featurewise double descent:** {evidence['D2']} Peak candidates: {peak_description(all_rows, 'D2')}. See `D2_learned_vs_nested.png`; widths are static representation comparisons.

The five strongest available paired contrasts are descriptive only:

{highlight_text}

Missing cells and singular designs remain visible; no observations are deleted to smooth spikes. The intervals bootstrap independent representation seeds, not individual probe draws. Nonconverged learned representations remain in raw data with `trial_valid=False` and are omitted from scientific means and paired contrasts.

## 5. Samplewise double descent

Peak location and amplitude are **not established** unless completed D1 rows span both sides of the observed rank boundary. Use actual `n`, `d`, and `rank_H`; the quantile figure shows per-probe mean, median, and within-representation 5th–95th percentiles, with uncertainty across representation seeds. For `n>d`, `A`, `D0`, `D_e`, and `S_e` are `NA` by design.

## 6. Featurewise double descent

Compare `independently_learned`, `nested_learned`, `nested_random`, and `width_matched_random` at fixed raw-data seeds. `M_err` is the population model-insufficiency estimate and is reported separately from readout risk. Do not describe re-trained widths as training-time dynamics.

## 7. Superposition mechanism

The relevant evidence is learned-versus-random contrasts conditioned on input law, target, actual rank, residual scale, and independent representation seeds, plus kernel/dependence diagnostics. `A` alone does not establish a superposition mechanism. Compare `A_spec`, `A_dep`, `A_ind`, kernel eigenvalues, and reconstruction/coherence diagnostics.

## 8. Counterexamples and null results

Reported invalid, singular, nonconverged, and missing rows are preserved in lane raw tables, per-cell `failure_rates.csv`, and `failed_jobs/`. Any absent comparison is **not established**, not a null result. Read `MATH_SPEC_REVIEW.md` for the derivation of the brief's `theta_0` equation.

## 9. Ranking of next directions

No direction is ranked as publishable until a discovery contrast replicates in a fresh confirmation seed range. Candidate directions are (a) residual-kernel dependence under sparse inputs, (b) learned-versus-random model-insufficiency controls, and (c) rank-adjusted samplewise/featurewise risk attribution. Each should be rejected if its representation-seed interval includes effects too small to matter or if the matched controls explain it.

## 10. One next decisive experiment and one theorem candidate

**Experiment:** repeat the strongest completed learned-versus-random contrast with fresh representation seeds, paired raw samples and noise, and enough draws around measured rank to estimate the peak and its interval. If no such contrast completed, do not start confirmation; finish the smoke and discovery audit first.

**Theorem candidate:** not established. A candidate is a conditional expectation identity for `A` under independent Gaussian residuals and full-row-rank Gaussian designs; its assumptions and finite-mean boundary must be checked against the validated Lane A controls before stating a theorem.
"""
    (run_dir / "DECISION_MEMO.md").write_text(decision, encoding="utf-8")

    (run_dir / "NEXT_THEOREM.md").write_text(
        "# Next theorem\n\n**Not established from this run.** A potential candidate is the conditional inverse-Wishart expectation of intrinsic aliasing for an independent Gaussian design, away from the singular boundary. It should be stated only after Lane A confirms the exact rank, covariance, and observation-noise assumptions and reports uncertainty across independent designs.\n",
        encoding="utf-8")

    closing = ("Please critically audit these experiments for mathematical validity, confounders, statistical robustness, and novelty. Using the attached `results_bundle.zip`, identify the strongest publishable Chapter 2 story and the smallest decisive next theorem or experiment. Do not assume the plots prove the mechanism; check the manifests, seed-level contrasts, exact-vs-estimated oracle status, and definition-version warnings first.")
    handoff = f"""# New-chat handoff: Chapter 2 v3 static GAD experiments

This package studies whether frozen learned linear representations change the intrinsic finite-sample aliasing of squared-loss linear readouts after controlling for target accessibility, effective feature dimension, and population second-moment geometry. The repository is `https://github.com/josephdutoit/aliasing`; this run used local Git SHA `{plan['git']['git_sha']}` and code hash `{plan['code_sha256']}`.

GAD-v3 uses the population feature second moment `Sigma = E[m(x)m(x)^T]`, the exact synthetic population oracle `theta_star`, and the intrinsic sample kernel `K = H Sigma^-1 H.T`. For a full-row-rank interpolating design, `A = 1/2 r.T K^-1 r`; the population risk is `R(theta_e)=M+A+D_e`, with signed readout displacement `S_e=D_e-D0`. Here `D0` is the squared population-metric norm of the oracle component in the sample nullspace. For `n>d`, only the general risk `M + 1/2 ||theta_hat-theta_star||_Sigma^2` is reported; the interpolating components are `NA`. These definitions are tagged `population_metric_v3` and remain separate from the repository's legacy GAD-v2 prediction-space components. Lane A compares observed `A` and `D0` with their respective Gaussian expectations.

**Run status:** {report_status}; {completed_jobs}/{expected_jobs} jobs; {total_rows} trial rows; {valid_rows} valid rows; {failures} invalid rows. Profile `{plan['profile']}` is exploratory unless the manifests document a fresh confirmation run. Cached real-model activation probing is optional and is marked `SKIPPED_DEPENDENCY` when absent. No Slurm resources are assumed; consult actual shard provenance.

Multiplicity plan: {multiplicity_plan}

Inspect `MANIFEST.json` and `AUDIT_PASS.json` first, then `failure_rates.csv`, lane-level `run_manifest.json`, `seed_means.csv`, `paired_contrasts.csv`, and `figure_data/`. Curves report the row count and exact-versus-estimated oracle status in figure titles. Read `D1_samplewise_rank_condition.png` and each row's `rank_H` before interpreting an interpolation peak. Check `trial_valid`, `status`, `warning_codes`, `is_oracle_exact`, and `oracle_reference_size`; do not remove singular or failed cells. The historical GAD naming caveat is preserved in `GAD_DEFINITIONS_AUDIT.md` and the pre-edit `REPO_AUDIT.md`.

Quantitative highlights:

{highlight_text}

Missing comparisons are “not established,” not null effects. Any result from `overnight_discovery` remains exploratory; holdout confirmation requires the fresh seed range in the `confirmation` profile.

Files to inspect: `DECISION_MEMO.md`, `NEXT_THEOREM.md`, `FIGURE_GALLERY.md`, `summary_matrix.csv`, `AUDIT_PASS.json`, lane tables, and the linked source CSVs. The ZIP contains curated summaries and code/config evidence but omits bulky checkpoint and shard archives; the full run folder retains those artifacts.

> {closing}
"""
    (run_dir / "NEW_CHAT_HANDOFF.md").write_text(handoff, encoding="utf-8")

    figure_lines = ["# Figure gallery", "", "Each image has an adjacent numeric source CSV in `figure_data/`. Titles report the number of trial rows and whether the oracle is exact or estimated.", ""]
    for lane, names in PLOT_NAMES.items():
        for name in names:
            interpretation = FIGURE_INTERPRETATIONS[name]
            figure_lines.append(f"- [`{name}.png`](figures/{name}.png) — {interpretation} Inspect its source CSV and seed-level intervals before interpreting a pattern.")
    (run_dir / "FIGURE_GALLERY.md").write_text("\n".join(figure_lines) + "\n", encoding="utf-8")


def build_bundle(run_dir: Path) -> Path:
    bundle = run_dir / "results_bundle.zip"
    include_roots = (
        "MANIFEST.json", "plan.json", "AUDIT_PASS.json", "REPO_AUDIT.md", "REFACTOR_CHANGELOG.md",
        "MATH_SPEC_REVIEW.md", "MORNING_README.md", "DECISION_MEMO.md", "NEXT_THEOREM.md",
        "NEW_CHAT_HANDOFF.md", "FIGURE_GALLERY.md", "summary_matrix.csv", "failure_rates.csv", "figures",
        "figure_data", "lanes", "failed_jobs", "audit_junit.xml", "audit_pytest.log",
    )
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in include_roots:
            path = run_dir / name
            if path.is_file():
                archive.write(path, arcname=path.relative_to(run_dir))
            elif path.is_dir():
                for member in sorted(path.rglob("*")):
                    if not member.is_file():
                        continue
                    relative = member.relative_to(run_dir)
                    if any(part in ("shards", "checkpoints", "attempts") for part in relative.parts):
                        continue
                    if member.suffix in (".npz", ".npy", ".pkl", ".pickle"):
                        continue
                    if member.name == "results_bundle.zip":
                        continue
                    archive.write(member, arcname=relative)
        source_files = (
            ROOT / "src/aliasing/intrinsic_gad.py", ROOT / "src/aliasing/chapter2_v3.py",
            ROOT / "experiments/chapter2_v3/plan.py", ROOT / "experiments/chapter2_v3/run.py",
            ROOT / "experiments/chapter2_v3/collect.py", ROOT / "experiments/chapter2_v3/runtime.py",
            ROOT / "experiments/chapter2_v3/audit.py",
            ROOT / "scripts/chapter2_v3/launch_overnight.sh",
            ROOT / "scripts/chapter2_v3/array_cpu.sbatch",
            ROOT / "scripts/chapter2_v3/array_train.sbatch",
            ROOT / "scripts/chapter2_v3/aggregate.sbatch",
            ROOT / "REFACTOR_CHANGELOG.md", ROOT / "MATH_SPEC_REVIEW.md",
            ROOT / "configs/chapter2_v3" / f"{json.loads((run_dir / 'plan.json').read_text())['profile']}.json",
            ROOT / "GAD_DEFINITIONS_AUDIT.md", ROOT / "CHAPTER2_V3_RUN_GUIDE.md",
        )
        for path in source_files:
            if path.is_file():
                archive.write(path, arcname=f"implementation/{path.relative_to(ROOT)}")
        audit_path = ROOT / "tests"
        if audit_path.is_dir():
            for path in sorted(audit_path.glob("test_*.py")):
                archive.write(path, arcname=f"implementation/tests/{path.name}")
    return bundle


def observed_resources(job_statuses: list[dict[str, Any]], previous: dict[str, Any]) -> dict[str, Any]:
    """Summarize scheduler resources from actual task environments only."""
    fields = {
        "partition": "slurm_partition",
        "cpus_per_task": "slurm_cpus_per_task",
        "memory_per_node": "slurm_mem_per_node",
        "time_limit": "slurm_time_limit",
    }
    result: dict[str, Any] = {"gpu": "NOT_REQUESTED_BY_LAUNCHER"}
    observed_jobs: set[str] = set()
    for output_name, provenance_name in fields.items():
        values = set()
        for job in job_statuses:
            provenance = job.get("provenance", {})
            value = provenance.get(provenance_name)
            if value not in (None, "", "NOT_IN_SLURM"):
                values.add(str(value))
                observed_jobs.add(str(job.get("job_id", "unknown")))
        result[output_name] = sorted(values) if values else previous.get(output_name, "NOT_QUERIED")
    result["tasks_with_observed_resources"] = len(observed_jobs)
    return result


def collect(run_dir: Path) -> tuple[dict[str, Any], int]:
    plan_path = run_dir / "plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"missing plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    manifest_path = run_dir / "MANIFEST.json"
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    job_statuses: list[dict[str, Any]] = []
    rows_by_lane: dict[str, list[dict[str, str]]] = {lane: [] for lane in PLOT_NAMES}
    failed_dir = run_dir / "failed_jobs"
    failed_dir.mkdir(parents=True, exist_ok=True)
    for job in plan["jobs"]:
        status_path = run_dir / "lanes" / job["lane"] / "shards" / job["job_id"] / "job_status.json"
        if not status_path.is_file():
            status = {"job_id": job["job_id"], "job_index": job["job_index"],
                      "lane": job["lane"], "status": "MISSING", "attempt": NA}
            job_statuses.append(status)
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status |= {"expected_job_index": job["job_index"]}
        job_statuses.append(status)
        attempt_dir = status_path.parent / status.get("attempt", "")
        raw_path = attempt_dir / "raw.csv"
        rows_by_lane[job["lane"]].extend(read_csv(raw_path))
        if status.get("status") not in ("COMPLETED", "SKIPPED_DEPENDENCY"):
            destination = failed_dir / f"{job['job_id']}_{status.get('attempt','unknown')}"
            destination.mkdir(parents=True, exist_ok=True)
            for source_name in ("status.json", "traceback.txt", "raw.csv"):
                source = attempt_dir / source_name
                if source.is_file():
                    shutil.copy2(source, destination / source_name)
    status_counts: dict[str, dict[str, int]] = {}
    for job_status in job_statuses:
        lane = str(job_status.get("lane", "unknown"))
        state = str(job_status.get("status", "MISSING"))
        status_counts.setdefault(lane, {})[state] = status_counts.setdefault(lane, {}).get(state, 0) + 1
    expected_lanes = {job["lane"] for job in plan["jobs"]}
    required_lanes = {"A", "B", "C", "D1", "D2"}
    incomplete = any(status["status"] not in ("COMPLETED", "SKIPPED_DEPENDENCY")
                     for status in job_statuses)
    missing_required_lane = any(not any(j["lane"] == lane and s.get("status") == "COMPLETED"
                                        for j, s in zip(plan["jobs"], job_statuses))
                                for lane in required_lanes if lane in expected_lanes)
    partial = incomplete or missing_required_lane
    lane_names = sorted(expected_lanes)
    lane_status_counts = {lane: status_counts.get(lane, {}) for lane in lane_names}
    all_rows = [row for lane in lane_names for row in rows_by_lane.get(lane, [])]
    all_contrasts = paired_contrasts(all_rows, int(plan["config"].get("bootstrap_replicates", 0)),
                                     int(plan["config"]["seed"]))
    all_contrasts.extend(gaussian_sparse_contrasts(
        all_rows, int(plan["config"].get("bootstrap_replicates", 0)), int(plan["config"]["seed"])))
    manifest = old_manifest | {
        "schema": plan["schema"], "run_id": run_dir.name, "profile": plan["profile"],
        "plan_hash": plan["plan_hash"], "git": plan["git"], "code_sha256": plan["code_sha256"],
        "config_sha256": plan["config_sha256"],
        "status": "PARTIAL" if partial else "COMPLETE", "expected_jobs": len(plan["jobs"]),
        "completed_jobs": sum(s.get("status") == "COMPLETED" for s in job_statuses),
        "skipped_jobs": sum(s.get("status") == "SKIPPED_DEPENDENCY" for s in job_statuses),
        "failed_jobs": sum(s.get("status") in ("FAILED", "TIMEOUT", "DEPENDENCY_MISSING", "INCOMPLETE", "MISSING") for s in job_statuses),
        "job_status_counts": {state: sum(s.get("status") == state for s in job_statuses)
                               for state in sorted({str(s.get("status")) for s in job_statuses})},
        "lane_status_counts": lane_status_counts, "jobs": job_statuses,
        "resources": observed_resources(
            job_statuses, old_manifest.get("resources", {"partition": "NOT_QUERIED"})
        ),
        "collected_utc": utc_now(), "trial_rows": len(all_rows),
        "invalid_trial_rows": sum(row.get("trial_valid") == "False" for row in all_rows),
    }
    atomic_json(manifest_path, manifest)

    all_rows_by_lane = rows_by_lane
    bootstrap_n = int(plan["config"].get("bootstrap_replicates", 0))
    seed_rows_all: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    contrast_all: list[dict[str, Any]] = []
    for lane in lane_names:
        lane_dir = run_dir / "lanes" / lane
        lane_dir.mkdir(parents=True, exist_ok=True)
        lane_rows = all_rows_by_lane.get(lane, [])
        lane_failures = failure_rates(lane_rows)
        lane_raw = lane_dir / "raw.csv"
        from runtime import atomic_write as write_bytes
        write_bytes(lane_raw, csv_bytes(lane_rows))
        write_bytes(lane_dir / "failure_rates.csv", csv_bytes(lane_failures,
            fields=list(dict.fromkeys(key for row in lane_failures for key in row)) if lane_failures
            else ["lane", "condition_id", "trial_rows", "valid_trial_rows", "invalid_trial_rows", "failure_fraction"]))
        lane_seed_means = grouped_seed_means(lane_rows)
        lane_means, lane_quantiles, lane_ci = summarize_seed_means(
            lane_seed_means, bootstrap_n, int(plan["config"]["seed"]))
        lane_contrasts = [row for row in all_contrasts if row.get("lane") == lane]
        for table_name, table_rows in (("seed_means.csv", lane_seed_means), ("mean.csv", lane_means),
                                       ("median_quantiles.csv", lane_quantiles), ("ci95.csv", lane_ci),
                                       ("paired_contrasts.csv", lane_contrasts)):
            if table_rows:
                fields = list(dict.fromkeys(key for row in table_rows for key in row))
            else:
                fields = ["lane", "condition_id", "metric", "status"]
            write_bytes(lane_dir / table_name, csv_bytes(table_rows, fields=fields))
        lane_jobs = [j for j in job_statuses if j["lane"] == lane]
        lane_manifest = {
            "run_id": run_dir.name, "lane": lane, "expected_jobs": sum(j["lane"] == lane for j in plan["jobs"]),
            "job_status_counts": lane_status_counts.get(lane, {}), "trial_rows": len(lane_rows),
            "invalid_trial_rows": sum(row.get("trial_valid") == "False" for row in lane_rows),
            "failure_fraction": (sum(row.get("trial_valid") == "False" for row in lane_rows) / len(lane_rows)
                                 if lane_rows else NA),
            "jobs": lane_jobs, "plan_hash": plan["plan_hash"], "code_sha256": plan["code_sha256"],
        }
        atomic_json(lane_dir / "run_manifest.json", lane_manifest)
        seed_rows_all.extend(lane_seed_means)
        contrast_all.extend(lane_contrasts)
        summary_rows.append({
            "lane": lane, "hypothesis": {
                "A": "Gaussian inverse-Wishart and exact invariants",
                "B": "Sparse input law at matched raw second moment",
                "C": "Learned superposition representation versus matched controls",
                "D1": "Samplewise double descent at fixed frozen representation",
                "D2": "Featurewise double descent: learned width versus nested/random controls",
                "P": "Optional frozen real-model probe transfer",
            }.get(lane, lane),
            "planned_jobs": sum(j["lane"] == lane for j in plan["jobs"]),
            "completed_jobs": sum(j.get("status") == "COMPLETED" for j in lane_jobs),
            "trial_rows": len(lane_rows),
            "valid_trial_rows": sum(row.get("trial_valid") == "True" for row in lane_rows),
            "invalid_trial_rows": sum(row.get("trial_valid") == "False" for row in lane_rows),
            "independent_representation_seeds": len({row.get("representation_seed") for row in lane_rows
                                                       if row.get("representation_seed") not in (NA, None, "")}),
            "status": "PARTIAL" if any(j.get("status") not in ("COMPLETED", "SKIPPED_DEPENDENCY")
                                         for j in lane_jobs) else "COMPLETE_OR_SKIPPED",
        })
    for contrast in summarized_contrasts(contrast_all):
        summary_rows.append({
            "lane": contrast.get("lane", "NA"), "hypothesis": "paired scientific contrast",
            "comparison": contrast.get("contrast", "NA"), "metric": contrast.get("metric", "NA"),
            "mean_difference": contrast.get("mean_difference", NA),
            "lower_95": contrast.get("lower_95", NA), "upper_95": contrast.get("upper_95", NA),
            "valid_representation_seeds": contrast.get("valid_representation_seeds", 0),
            "target_family": contrast.get("target_family", NA), "n": contrast.get("n", NA),
            "d": contrast.get("d", NA), "M": contrast.get("M", NA),
            "status": "SUMMARIZED",
        })
    write_bytes(run_dir / "summary_matrix.csv", csv_bytes(summary_rows,
        fields=list(dict.fromkeys(key for row in summary_rows for key in row))))
    failure_rows = failure_rates(all_rows)
    write_bytes(run_dir / "failure_rates.csv", csv_bytes(failure_rows,
        fields=list(dict.fromkeys(key for row in failure_rows for key in row)) if failure_rows
        else ["lane", "condition_id", "trial_rows", "valid_trial_rows", "invalid_trial_rows", "failure_fraction"]))
    create_figures(run_dir, rows_by_lane, plan)
    report_files(run_dir, plan, manifest, all_rows, seed_rows_all, contrast_all)
    manifest["bundle"] = {"path": "results_bundle.zip"}
    atomic_json(manifest_path, manifest)
    bundle = build_bundle(run_dir)
    (run_dir / "results_bundle.sha256").write_text(f"{sha256_file(bundle)}  {bundle.name}\n", encoding="utf-8")
    return manifest, 2 if partial else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest, status = collect(args.run_dir.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"run_dir": str(args.run_dir.resolve()), "status": manifest["status"],
                      "completed_jobs": manifest["completed_jobs"], "expected_jobs": manifest["expected_jobs"],
                      "bundle": manifest.get("bundle")}, sort_keys=True, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
