import json
from pathlib import Path
import sys

import numpy as np
import pytest

from aliasing.chapter2_v3 import (
    bernoulli_probabilities,
    bernoulli_uniform_moments,
    rng_for,
    sparse_fixed_k,
    train_tied_sparse_autoencoder,
)
from aliasing.intrinsic_gad import population_linear_oracle
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "chapter2_v3"))
from plan import build_jobs, validate_profile  # noqa: E402
from runtime import atomic_write, covariance_moments, file_lock  # noqa: E402
from run import run_one  # noqa: E402


def test_exact_k_gaussian_samples_and_analytic_second_moments():
    rng = np.random.default_rng(7)
    samples = sparse_fixed_k(12000, 8, 2, rng)
    assert np.all(np.count_nonzero(samples, axis=1) == 2)
    mean, second, centered = covariance_moments("exact_k_gaussian", 8)
    assert np.array_equal(mean, np.zeros(8))
    assert np.array_equal(second, np.eye(8))
    assert np.array_equal(centered, np.eye(8))
    empirical = samples.T @ samples / samples.shape[0]
    assert np.allclose(empirical, np.eye(8), atol=0.07, rtol=0.0)


def test_bernoulli_uniform_raw_and_centered_moments_include_heterogeneous_p():
    probabilities = bernoulli_probabilities(0.2, 8, "two_tier")
    mean, second = bernoulli_uniform_moments(probabilities)
    variance = probabilities / 3.0 - probabilities**2 / 4.0
    assert np.allclose(mean, probabilities / 2.0)
    assert np.allclose(np.diag(second), probabilities / 3.0)
    assert np.allclose(second - np.outer(mean, mean), np.diag(variance))


def test_bernoulli_intercept_feature_moments_are_exact():
    probabilities = bernoulli_probabilities(0.2, 6, "two_tier")
    mean, second = bernoulli_uniform_moments(probabilities)
    W = np.arange(18, dtype=np.float64).reshape(3, 6) / 10.0
    teacher = np.linspace(-1.0, 1.0, 6)
    oracle = population_linear_oracle(W, second, teacher, mean_x=mean, intercept=True)
    expected = np.empty((4, 4), dtype=np.float64)
    expected[0, 0] = 1.0
    expected[0, 1:] = W @ mean
    expected[1:, 0] = expected[0, 1:]
    expected[1:, 1:] = W @ second @ W.T
    assert np.allclose(oracle["Sigma"], expected, atol=1e-12, rtol=1e-12)


def test_rng_streams_are_stable_and_coordinate_separated():
    first = rng_for(11, "teacher", 4).normal(size=8)
    replay = rng_for(11, "teacher", 4).normal(size=8)
    other = rng_for(11, "probe", 4).normal(size=8)
    assert np.array_equal(first, replay)
    assert not np.array_equal(first, other)


def test_tied_autoencoder_training_resumes_identically_from_optimizer_checkpoint():
    train = np.random.default_rng(13).uniform(size=(24, 6))
    validation = np.random.default_rng(17).uniform(size=(12, 6))
    settings = dict(steps=12, batch_size=8, learning_rate=0.001,
                    sparsity_penalty=0.02, weight_decay=0.001, patience=100)
    uninterrupted = train_tied_sparse_autoencoder(train, validation, 3, 19, **settings)
    checkpoints = []
    train_tied_sparse_autoencoder(train, validation, 3, 19,
        **(settings | {"steps": 5}), checkpoint_callback=checkpoints.append)
    resumed = train_tied_sparse_autoencoder(train, validation, 3, 19,
        **settings, resume_state=checkpoints[-1])
    assert np.array_equal(uninterrupted["W"], resumed["W"])
    assert np.array_equal(uninterrupted["bias"], resumed["bias"])
    assert uninterrupted["finite"] and resumed["finite"]


def test_smoke_plan_has_unique_world_shards_and_d1_sources():
    config = json.loads((ROOT / "configs/chapter2_v3/smoke.json").read_text())
    jobs = build_jobs(config)
    identifiers = [job["job_id"] for job in jobs]
    assert len(identifiers) == len(set(identifiers))
    c_jobs = {job["job_id"] for job in jobs if job["lane"] == "C"}
    assert all(job["source_job_id"] in c_jobs for job in jobs
               if job["lane"] == "D1" and job.get("source_job_id"))
    assert any(job["world"].get("design") == "gaussian_baseline"
               for job in jobs if job["lane"] == "D1")
    d2 = [job for job in jobs if job["lane"] == "D2"]
    # Each frozen representation is one shard; configured probe blocks stay inside it.
    assert len(d2) == len({(job["world"].get("design"), job["world"].get("family"),
                            job["world"].get("M"), job.get("representation_seed")) for job in d2})


def test_d2_requires_raw_targets_fixed_across_widths():
    config = json.loads((ROOT / "configs/chapter2_v3/overnight_discovery.json").read_text())
    assert "in_span" not in config["lane_D2"]["target_kinds"]
    invalid = config | {"lane_D2": config["lane_D2"] | {"target_kinds": ["in_span"]}}
    with pytest.raises(ValueError, match="raw teacher fixed across widths"):
        validate_profile(invalid)


def test_atomic_shard_write_lock_is_released_and_reusable(tmp_path):
    lock = tmp_path / "world.lock"
    data = tmp_path / "raw.csv"
    with file_lock(lock):
        atomic_write(data, b"trial_valid\nTrue\n")
    with file_lock(lock):
        assert data.read_bytes() == b"trial_valid\nTrue\n"


def test_completed_dependency_skip_is_idempotent_and_creates_one_attempt(tmp_path):
    run_dir = tmp_path / "smoke"
    job = {
        "job_id": "p_optional", "job_index": 0, "lane": "P",
        "world": {"activation_cache": None, "optional": True},
        "representation_seed": None, "probe_seed_block": 0,
        "config": {"activation_cache": None},
    }
    plan = {
        "schema": "chapter2_v3_plan_1", "profile": "smoke", "jobs": [job],
        "git": {"git_sha": "test"}, "code_sha256": "code", "config_sha256": "config",
        "config": {"seed": 1},
    }
    run_dir.mkdir()
    (run_dir / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    first = run_one(run_dir, 0)
    second = run_one(run_dir, 0)
    shard = run_dir / "lanes/P/shards/p_optional"
    assert first["status"] == "SKIPPED_DEPENDENCY"
    assert second["reused"] is True
    assert len(list(shard.glob("attempt_[0-9][0-9][0-9][0-9]"))) == 1


def test_optional_lane_p_missing_cache_is_recorded_as_dependency_skip(tmp_path):
    run_dir = tmp_path / "smoke"
    missing_cache = tmp_path / "not_present.npz"
    job = {
        "job_id": "p_missing_cache", "job_index": 0, "lane": "P",
        "world": {"activation_cache": str(missing_cache), "optional": True},
        "representation_seed": None, "probe_seed_block": 0,
        "config": {"activation_cache": str(missing_cache)},
    }
    plan = {
        "schema": "chapter2_v3_plan_1", "profile": "smoke", "jobs": [job],
        "git": {"git_sha": "test"}, "code_sha256": "code", "config_sha256": "config",
        "config": {"seed": 1},
    }
    run_dir.mkdir()
    (run_dir / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    status = run_one(run_dir, 0)
    assert status["status"] == "SKIPPED_DEPENDENCY"
    shard = run_dir / "lanes/P/shards/p_missing_cache/attempt_0001/raw.csv"
    assert "activation_cache_missing" in shard.read_text(encoding="utf-8")
