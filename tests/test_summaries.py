import numpy as np

from aliasing.summaries import (
    aggregate_seed_rows,
    adjacent_paired_contrasts,
    bootstrap_mean_interval,
)


def test_bootstrap_interval_is_reproducible_and_contains_mean_for_constant_data():
    first = bootstrap_mean_interval([3.0] * 8, seed=4)
    second = bootstrap_mean_interval([3.0] * 8, seed=4)
    assert first == second == (3.0, 3.0)


def test_aggregation_uses_sample_standard_deviation_and_ci_fields():
    rows = [
        {"seed": 0.0, "width": 2.0, "risk": 1.0},
        {"seed": 1.0, "width": 2.0, "risk": 3.0},
        {"seed": 2.0, "width": 2.0, "risk": 5.0},
    ]
    means, stds, cis = aggregate_seed_rows(
        rows, group_keys=("width",), excluded_keys={"seed", "width"}, n_bootstrap=1000
    )
    assert means[0]["risk"] == 3.0
    assert np.isclose(stds[0]["risk"], 2.0)
    assert cis[0]["risk_ci95_low"] <= means[0]["risk"] <= cis[0]["risk_ci95_high"]


def test_adjacent_contrasts_are_paired_by_seed():
    rows = [
        {"seed": 0.0, "width": 2.0, "risk": 10.0},
        {"seed": 1.0, "width": 2.0, "risk": 20.0},
        {"seed": 0.0, "width": 4.0, "risk": 7.0},
        {"seed": 1.0, "width": 4.0, "risk": 16.0},
    ]
    contrasts = adjacent_paired_contrasts(
        rows,
        level_key="width",
        match_keys=("seed",),
        excluded_keys={"seed", "width"},
        n_bootstrap=1000,
    )
    assert len(contrasts) == 1
    assert contrasts[0]["num_pairs"] == 2.0
    assert contrasts[0]["mean_difference"] == -3.5
