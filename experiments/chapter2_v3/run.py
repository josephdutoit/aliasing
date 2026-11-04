#!/usr/bin/env python3
"""Run one planned Chapter 2 v3 world shard on CPU in float64."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import traceback
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aliasing.chapter2_v3 import (  # noqa: E402
    feature_matrix,
    random_rows,
    rng_for,
    superposition_diagnostics,
    train_tied_sparse_autoencoder,
)
from aliasing.intrinsic_gad import (  # noqa: E402
    GAD_DEFINITION,
    independent_residual_control,
    interpolating_population_gad,
    noninterpolating_risk,
    spectral_dependence_diagnostics,
)
from runtime import (  # noqa: E402
    NA,
    atomic_json,
    atomic_write,
    base_row,
    build_trial_row,
    covariance_moments,
    csv_bytes,
    draw_labels,
    file_lock,
    ordinary_least_squares,
    population_oracle,
    runtime_provenance,
    sample_inputs,
    sample_sizes,
    sha256_file,
    teacher_for,
    utc_now,
)

PREEMPT_REQUESTED = False


def _handle_preempt(_signum: int, _frame: Any) -> None:
    global PREEMPT_REQUESTED
    PREEMPT_REQUESTED = True


def _raise_if_preempted() -> None:
    if PREEMPT_REQUESTED:
        raise TimeoutError("TIMEOUT: preemption signal received; retry the immutable shard")


def _mark_nonconverged(row: dict[str, Any]) -> None:
    row["status"] = "NONCONVERGED_REPRESENTATION"
    row["trial_valid"] = False
    existing = str(row.get("warning_codes", ""))
    row["warning_codes"] = "|".join(filter(None, (existing, "NONCONVERGED_REPRESENTATION")))


def load_plan(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "plan.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing immutable plan: {path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != "chapter2_v3_plan_1":
        raise ValueError("unsupported plan schema")
    plan["run_id"] = run_dir.name
    return plan


def _rng(plan: dict[str, Any], *coordinates: Any) -> np.random.Generator:
    return rng_for(int(plan["config"]["seed"]), "chapter2_v3", *coordinates)


def _shared_probe_key(job: dict[str, Any]) -> Any:
    if job.get("source_job_id"):
        return job["source_job_id"]
    if job["lane"] == "C":
        return job["job_id"]
    world = job["world"]
    return ("paired_synthetic", world.get("input_dimension"), world.get("M"),
            world.get("family"), job.get("representation_seed"),
            world.get("p"), world.get("importance_profile"))


def _world_W(plan: dict[str, Any], job: dict[str, Any], family: str | None = None) -> np.ndarray:
    world = job["world"]
    width = int(world.get("M", world.get("max_width", 0)))
    dimension = int(world.get("input_dimension", job["config"].get("input_dimension", 0)))
    representation_seed = int(job.get("representation_seed", 0))
    row_family = family or world.get("family", "gaussian")
    return random_rows(width, dimension,
                       _rng(plan, "fixed_W", dimension, width, row_family, representation_seed),
                       "orthonormal" if row_family == "orthonormal" else "gaussian")


def _test_data(plan: dict[str, Any], job: dict[str, Any], law: str, teacher: np.ndarray,
               n_test: int, noise: float, *, p: float, importance: str,
               active_count: int | None, block: int, draw: int) -> tuple[np.ndarray, np.ndarray]:
    rng_x = _rng(plan, "probe", _shared_probe_key(job), block, draw, "test_X")
    X = sample_inputs(law, n_test, teacher.size, rng_x, active_count=active_count,
                      p=p, importance=importance)
    y = draw_labels(X, teacher, _rng(plan, "probe", _shared_probe_key(job), block, draw, "test_y"), 0.0)
    return X, y


def _row_trials(plan: dict[str, Any], job: dict[str, Any], *, W: np.ndarray,
                law: str, target_kinds: list[str], noise_levels: list[float],
                ratios: list[float], offsets: list[int], block: int, draws: int,
                n_test: int, n_oracle: int, active_count: int | None = None,
                p: float = 0.1, importance: str = "uniform", intercept: bool = False,
                control_specs: list[dict[str, Any]] | None = None,
                representation_converged: Any = NA, reconstruction_loss: Any = NA,
                max_samples: int | None = None) -> list[dict[str, Any]]:
    """Evaluate paired target/noise/probe conditions for one fixed representation bank."""
    control_specs = control_specs or [{"name": "learned", "W": W}]
    d = W.shape[0] + int(intercept)
    ns = sample_sizes(d, ratios, offsets, maximum=max_samples)
    n_max = max((n for n, _ in ns), default=0)
    rows: list[dict[str, Any]] = []
    _, _, centered = covariance_moments(law, W.shape[1], p=p, importance=importance)

    for target_kind in target_kinds:
        if job.get("source_job_id"):
            target_key = job["source_job_id"]
        elif job["lane"] == "C":
            target_key = job["job_id"]
        else:
            target_key = (
                job["world"].get("input_dimension"), job["world"].get("M"),
                job["world"].get("family"), job["world"].get("p"),
                job["world"].get("importance_profile"), job.get("representation_seed"),
            )
        teacher = teacher_for(target_kind, W.shape[1], W, centered,
                              _rng(plan, "target", target_key, target_kind), importance)
        for sigma_epsilon in noise_levels:
            for n, ratio_actual in ns:
                for draw in range(draws):
                    _raise_if_preempted()
                    shared_key = _shared_probe_key(job)
                    probe_seed = int(_rng(plan, "seed_label", shared_key, block, draw,
                                          target_kind, sigma_epsilon, n).integers(0, 2**31 - 1))
                    oracle_seed = int(_rng(plan, "seed_label", shared_key, block, draw,
                                           "oracle", target_kind, sigma_epsilon, n).integers(0, 2**31 - 1))
                    X_all = sample_inputs(law, n_max, W.shape[1],
                        _rng(plan, "probe", shared_key, block, draw, "train_X"),
                        active_count=active_count, p=p, importance=importance)
                    noise_all = _rng(plan, "probe", shared_key, block, draw,
                                     target_kind, sigma_epsilon, "observation_noise").normal(
                                         scale=sigma_epsilon, size=n_max)
                    y_all = X_all @ teacher + noise_all
                    X_residual = sample_inputs(law, n_oracle, W.shape[1],
                        _rng(plan, "oracle", shared_key, block, draw, "residual_X"),
                        active_count=active_count, p=p, importance=importance)
                    residual_noise = _rng(plan, "oracle", shared_key, block, draw,
                                           target_kind, sigma_epsilon, "residual_noise").normal(
                                               scale=sigma_epsilon, size=n_oracle)
                    residual_labels = X_residual @ teacher + residual_noise
                    X_test, y_test = _test_data(plan, job, law, teacher, n_test, 0.0,
                        p=p, importance=importance, active_count=active_count, block=block, draw=draw)
                    for control in control_specs:
                        row = build_trial_row(
                            plan, job, W=control["W"], X_train=X_all[:n], y_train=y_all[:n],
                            X_residual=X_residual, residual_labels=residual_labels,
                            teacher=teacher, law=law, importance=importance, p=p,
                            active_count=active_count, target_kind=target_kind,
                            noise_sigma=float(sigma_epsilon), target_ratio=ratio_actual,
                            probe_seed=probe_seed, oracle_seed=oracle_seed, draw_index=draw,
                            intercept=intercept, control_name=control["name"],
                            representation_converged=control.get("converged", representation_converged),
                            reconstruction_loss=control.get("reconstruction_loss", reconstruction_loss),
                            feature_order=control.get("feature_order", "native"),
                            feature_transform=control.get("transform"), X_test=X_test, y_test=y_test,
                        )
                        row["condition_id"] = f"{target_kind}|noise={sigma_epsilon:g}|n={n}|{control['name']}"
                        row["noise_level"] = float(sigma_epsilon)
                        row["active_count"] = active_count if active_count is not None else NA
                        rows.append(row)
    return rows


def run_lane_a(plan: dict[str, Any], job: dict[str, Any]) -> list[dict[str, Any]]:
    config = job["config"]
    family = job["world"]["family"]
    W = _world_W(plan, job, family)
    whitening = _whitening_transform(plan, "gaussian", W, 0.1, "uniform", intercept=False)
    rows = _row_trials(
        plan, job, W=W, law="gaussian", target_kinds=config["target_kinds"],
        noise_levels=config["noise_levels"], ratios=config["ratios"],
        offsets=config.get("near_threshold_offsets", []), block=int(job["probe_seed_block"]),
        draws=int(config["probe_draws"]), n_test=int(config["n_test"]),
        n_oracle=int(config["n_oracle"]), intercept=False,
        control_specs=[{"name": "learned", "W": W},
                       {"name": "whitened_gauge", "W": W, "transform": whitening}],
    )
    for row in rows:
        row["representation_family"] = family
        row["condition_id"] = f"{row['condition_id']}|family={family}"
    return rows


def run_lane_b(plan: dict[str, Any], job: dict[str, Any]) -> list[dict[str, Any]]:
    config = job["config"]
    family = job["world"]["family"]
    W = _world_W(plan, job, family)
    rows: list[dict[str, Any]] = []
    for active_count in config["active_counts"]:
        rows.extend(_row_trials(
            plan, job, W=W, law="exact_k_gaussian", target_kinds=config["target_kinds"],
            noise_levels=config["noise_levels"], ratios=config["ratios"], offsets=[],
            block=int(job["probe_seed_block"]), draws=int(config["probe_draws"]),
            n_test=int(config["n_test"]), n_oracle=int(config["n_oracle"]),
            active_count=int(active_count), intercept=False,
            control_specs=[{"name": family, "W": W}],
        ))
    return rows


def _whitening_transform(plan: dict[str, Any], law: str, W: np.ndarray, p: float,
                         importance: str, *, intercept: bool = True) -> np.ndarray:
    oracle = population_oracle(W, law, np.zeros(W.shape[1]), intercept=intercept,
                               p=p, importance=importance)
    chol = np.linalg.cholesky(oracle["Sigma"])
    return np.linalg.solve(chol, np.eye(chol.shape[0]))


def _random_controls(plan: dict[str, Any], job: dict[str, Any], learned_W: np.ndarray,
                     names: list[str], *, law: str, p: float, importance: str) -> list[dict[str, Any]]:
    controls = [{"name": "learned", "W": learned_W}]
    control_key = job.get("source_job_id") or job.get("job_id")
    for name in names:
        if name == "learned":
            continue
        if name == "whitened_gauge":
            controls.append({"name": name, "W": learned_W,
                             "transform": _whitening_transform(plan, law, learned_W, p, importance)})
        elif name in ("random", "gaussian"):
            controls.append({"name": "random", "W": random_rows(learned_W.shape[0], learned_W.shape[1],
                _rng(plan, "control_W", control_key, "random"), "gaussian")})
        elif name == "orthonormal":
            controls.append({"name": name, "W": random_rows(learned_W.shape[0], learned_W.shape[1],
                _rng(plan, "control_W", control_key, "orthonormal"), "orthonormal")})
        else:
            raise ValueError(f"unknown representation control: {name}")
    return controls


def _train_representation(plan: dict[str, Any], job: dict[str, Any], run_dir: Path, *, p: float,
                          importance: str, width: int, law: str = "bernoulli_uniform") -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    config = job["config"]
    dimension = int(job["world"]["input_dimension"])
    seed = int(job["representation_seed"])
    train_x = sample_inputs(law, int(config["train_samples"]), dimension,
        _rng(plan, "representation_data", dimension, p, importance, seed, "train"),
        p=p, importance=importance)
    validation_x = sample_inputs(law, int(config["validation_samples"]), dimension,
        _rng(plan, "representation_data", dimension, p, importance, seed, "validation"),
        p=p, importance=importance)
    progress_path = run_dir / "checkpoints" / f"{job['job_id']}.training.npz"
    metadata_path = progress_path.with_suffix(".training.json")
    resume_state: dict[str, Any] | None = None
    if progress_path.is_file() and metadata_path.is_file():
        progress_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if progress_meta.get("plan_hash") != plan["plan_hash"]:
            raise ValueError("training checkpoint plan hash does not match this run")
        if sha256_file(progress_path) != progress_meta.get("sha256"):
            raise ValueError("training checkpoint checksum mismatch")
        with np.load(progress_path, allow_pickle=False) as archive:
            resume_state = {key: np.asarray(archive[key], dtype=np.float64)
                            for key in ("W", "bias", "first_W", "second_W", "first_b", "second_b",
                                        "best_W", "best_bias")}
        resume_state.update(progress_meta["state"])

    def save_progress(state: dict[str, Any]) -> None:
        arrays = {key: np.asarray(state[key]) for key in (
            "W", "bias", "first_W", "second_W", "first_b", "second_b", "best_W", "best_bias")}
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        payload = buffer.getvalue()
        metadata = {
            "job_id": job["job_id"], "plan_hash": plan["plan_hash"],
            "sha256": hashlib.sha256(payload).hexdigest(),
            "state": {key: state[key] for key in (
                "step", "best_validation", "best_step", "history", "plateau", "converged", "finite", "rng_state")},
            "saved_utc": utc_now(),
        }
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(progress_path.with_suffix(".lock")):
            atomic_write(progress_path, payload)
            atomic_json(metadata_path, metadata)

    trained = train_tied_sparse_autoencoder(
        train_x, validation_x, width, seed,
        steps=int(config["training_steps"]), batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        sparsity_penalty=float(config["sparsity_penalty"]),
        weight_decay=float(config["weight_decay"]), patience=int(config["patience"]),
        resume_state=resume_state, checkpoint_callback=save_progress,
        should_stop=lambda: PREEMPT_REQUESTED,
    )
    if trained.get("paused"):
        raise TimeoutError(f"TIMEOUT: saved resumable training checkpoint at step {trained['steps_completed']}")
    progress_path.unlink(missing_ok=True)
    metadata_path.unlink(missing_ok=True)
    return trained, train_x, validation_x


def _save_checkpoint(run_dir: Path, job: dict[str, Any], trained: dict[str, Any],
                     train_x: np.ndarray, validation_x: np.ndarray) -> dict[str, Any]:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{job['job_id']}.npz"
    buffer = io.BytesIO()
    np.savez_compressed(buffer, W=trained["W"], bias=trained["bias"],
                        representation_seed=np.asarray(job["representation_seed"], dtype=np.int64))
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    with file_lock(checkpoint_dir / f"{job['job_id']}.lock"):
        if path.exists():
            existing_metadata = checkpoint_dir / f"{job['job_id']}.json"
            if not existing_metadata.is_file():
                # A preemption between the two atomic renames can leave this
                # generated file without its sidecar. It is not a committed
                # checkpoint; remove it and publish the deterministic pair.
                path.unlink()
            else:
                existing = json.loads(existing_metadata.read_text(encoding="utf-8"))
                if sha256_file(path) != existing.get("sha256"):
                    raise RuntimeError("existing checkpoint checksum does not match metadata")
                with np.load(path, allow_pickle=False) as archive:
                    if not np.array_equal(np.asarray(archive["W"]), trained["W"]):
                        raise RuntimeError("existing checkpoint differs from deterministic retraining")
                return existing
        atomic_write(path, payload)
        metadata = {
            "job_id": job["job_id"], "sha256": digest, "shape": list(trained["W"].shape),
            "dtype": "float64", "representation_seed": job["representation_seed"],
            "input_law": "bernoulli_prob_uniform_0_1", "decoder": "relu(W.T @ W @ x + b)",
            "train_data_sha256": hashlib.sha256(np.ascontiguousarray(train_x).tobytes()).hexdigest(),
            "validation_data_sha256": hashlib.sha256(np.ascontiguousarray(validation_x).tobytes()).hexdigest(),
            "training_history": trained["history"], "converged": trained["converged"],
            "finite": trained["finite"], "best_step": trained["best_step"],
            "validation_reconstruction_mse": trained["validation_reconstruction_mse"],
        }
        atomic_json(checkpoint_dir / f"{job['job_id']}.json", metadata)
    return metadata


def run_lane_c(plan: dict[str, Any], job: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    world, config = job["world"], job["config"]
    p, importance = float(world["p"]), world["importance_profile"]
    trained, train_x, validation_x = _train_representation(
        plan, job, run_dir, p=p, importance=importance, width=int(world["M"]))
    W = trained["W"]
    checkpoint = _save_checkpoint(run_dir, job, trained, train_x, validation_x)
    controls = _random_controls(plan, job, W, config["controls"], law="bernoulli_uniform",
                                p=p, importance=importance)
    for control in controls:
        control["converged"] = trained["converged"] if control["name"] in ("learned", "whitened_gauge") else NA
        control["reconstruction_loss"] = trained["validation_reconstruction_mse"] if control["name"] in (
            "learned", "whitened_gauge") else NA
    rows: list[dict[str, Any]] = []
    blocks = config.get("probe_seed_blocks", [0])
    for block in blocks:
        rows.extend(_row_trials(
            plan, job, W=W, law="bernoulli_uniform", target_kinds=config["target_kinds"],
            noise_levels=config["noise_levels"], ratios=config["ratios"], offsets=[],
            block=int(block), draws=int(config["probe_draws"]), n_test=int(config["n_test"]),
            n_oracle=int(config["n_oracle"]), p=p, importance=importance, intercept=True,
            control_specs=controls, representation_converged=trained["converged"],
            reconstruction_loss=trained["validation_reconstruction_mse"],
        ))
    for row in rows:
        if row.get("control_name") in ("learned", "whitened_gauge"):
            row["representation_family"] = "tied_sparse_autoencoder"
            row["representation_checkpoint_sha256"] = checkpoint["sha256"]
            if not trained["converged"]:
                _mark_nonconverged(row)
        elif row.get("control_name") == "orthonormal":
            row["representation_family"] = "orthonormal_random"
        else:
            row["representation_family"] = "gaussian_random"
        control_W = next(control["W"] for control in controls if control["name"] == row["control_name"])
        row.update(superposition_diagnostics(control_W))
    return rows


def _load_source_W(run_dir: Path, source_job_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    checkpoint = run_dir / "checkpoints" / f"{source_job_id}.npz"
    metadata_path = run_dir / "checkpoints" / f"{source_job_id}.json"
    if not checkpoint.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"DEPENDENCY_MISSING: frozen representation checkpoint {source_job_id}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha256_file(checkpoint) != metadata["sha256"]:
        raise ValueError("checkpoint checksum does not match its metadata")
    with np.load(checkpoint, allow_pickle=False) as archive:
        W = np.asarray(archive["W"], dtype=np.float64)
    return W, metadata


def run_lane_d1(plan: dict[str, Any], job: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    config, world = job["config"], job["world"]
    if world.get("design") == "gaussian_baseline":
        W = _world_W(plan, job, "gaussian")
        rows = _row_trials(
            plan, job, W=W, law="gaussian", target_kinds=config["target_kinds"],
            noise_levels=config["noise_levels"], ratios=config["ratios"],
            offsets=config["near_threshold_offsets"], block=int(job["probe_seed_block"]),
            draws=int(config["probe_draws"]), n_test=int(config["n_test"]),
            n_oracle=int(config["n_oracle"]), intercept=False,
            control_specs=[{"name": "gaussian_baseline", "W": W}],
            max_samples=int(config["max_samples"]),
        )
        for row in rows:
            row["representation_family"] = "gaussian_baseline"
        return rows

    W, metadata = _load_source_W(run_dir, job["source_job_id"])
    p, importance = float(world["p"]), world["importance_profile"]
    controls = _random_controls(plan, job, W, config["controls"], law="bernoulli_uniform",
                                p=p, importance=importance)
    for control in controls:
        control["converged"] = metadata.get("converged", False) if control["name"] in (
            "learned", "whitened_gauge") else NA
        control["reconstruction_loss"] = metadata.get("validation_reconstruction_mse", NA)
    rows = _row_trials(
        plan, job, W=W, law="bernoulli_uniform", target_kinds=config["target_kinds"],
        noise_levels=config["noise_levels"], ratios=config["ratios"],
        offsets=config["near_threshold_offsets"], block=int(job["probe_seed_block"]),
        draws=int(config["probe_draws"]), n_test=int(config["n_test"]),
        n_oracle=int(config["n_oracle"]), p=p, importance=importance, intercept=True,
        control_specs=controls, max_samples=int(config["max_samples"]),
        representation_converged=metadata.get("converged", False),
        reconstruction_loss=metadata.get("validation_reconstruction_mse", NA),
    )
    for row in rows:
        if row.get("control_name") in ("learned", "whitened_gauge"):
            row["representation_checkpoint_sha256"] = metadata["sha256"]
            if not metadata.get("converged", False):
                _mark_nonconverged(row)
        row["representation_sha256"] = hashlib.sha256(np.ascontiguousarray(W).tobytes()).hexdigest()
    return rows


def _nested_projection(plan: dict[str, Any], job: dict[str, Any], W: np.ndarray,
                       law: str, p: float, importance: str, order: str,
                       seed: int) -> np.ndarray:
    if order == "native":
        return W
    if order == "random":
        permutation = _rng(plan, "feature_order", job["job_id"], seed, "random").permutation(W.shape[0])
        return W[permutation]
    if order == "pca":
        sample = sample_inputs(law, max(4 * W.shape[0], 512), W.shape[1],
            _rng(plan, "feature_order", job["job_id"], seed, "pca_data"), p=p, importance=importance)
        hidden = sample @ W.T
        hidden = hidden - hidden.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(hidden, full_matrices=True)
        return vt @ W
    raise ValueError(f"unknown nested feature order: {order}")


def run_lane_d2(plan: dict[str, Any], job: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    config, world = job["config"], job["world"]
    design = world["design"]
    dimension = int(world["input_dimension"])
    p = float(config.get("probability", 0.05))
    importance = str(config.get("importance_profile", "uniform"))
    law = str(config.get("input_law", "bernoulli_uniform"))
    seed = int(job["representation_seed"])
    width_values = [int(world["M"])] if "M" in world else [int(x) for x in config["widths"]]
    n_fixed_values = ([int(world["n_fixed"])] if "n_fixed" in world
                      else [int(value) for value in config["n_fixed"]])
    trained: dict[str, Any] | None = None
    checkpoint_metadata: dict[str, Any] | None = None
    train_x: np.ndarray | None = None
    val_x: np.ndarray | None = None
    base_bank: np.ndarray
    if design in ("independent_learned_width", "nested_fixed_bank") and (
            design == "independent_learned_width" or world.get("family") == "learned"):
        max_width = int(world["M"]) if design == "independent_learned_width" else max(width_values)
        train_job = job | {"world": world | {"M": max_width}}
        trained, train_x, val_x = _train_representation(plan, train_job, run_dir, p=p,
                                                        importance=importance, width=max_width, law=law)
        base_bank = trained["W"]
        checkpoint_metadata = _save_checkpoint(run_dir, train_job, trained, train_x, val_x)
    else:
        max_width = max(width_values)
        base_bank = random_rows(max_width, dimension,
            _rng(plan, "d2_nested_random_bank", seed, max_width, world.get("family", "gaussian")),
            "gaussian")

    rows: list[dict[str, Any]] = []
    blocks = [int(value) for value in config.get("probe_seed_blocks", [0])]
    orders = ["native"] if design != "nested_fixed_bank" else [
        str(value) for value in config.get("feature_orders", ["native", "random", "pca"])]
    for order in orders:
        ordered_bank = _nested_projection(plan, job, base_bank, law, p, importance, order, seed)
        for width in width_values:
            W = ordered_bank[:width] if design == "nested_fixed_bank" else base_bank
            if design == "independent_learned_width":
                W = base_bank
            if design == "width_matched_random":
                W = random_rows(width, dimension,
                    _rng(plan, "d2_random_width", seed, width), "gaussian")
            for target_kind in config["target_kinds"]:
                _, _, centered = covariance_moments(law, dimension, p=p, importance=importance)
                teacher_bank = base_bank if design == "nested_fixed_bank" else W
                teacher = teacher_for(target_kind, dimension, teacher_bank, centered,
                    _rng(plan, "d2_teacher", seed, target_kind), importance)
                for n_fixed in n_fixed_values:
                    for block in blocks:
                        for draw in range(int(config["probe_draws"])):
                            _raise_if_preempted()
                            n = n_fixed
                            X = sample_inputs(law, n, dimension,
                                _rng(plan, "d2_probe", seed, block, draw, n_fixed, "train_X"),
                                p=p, importance=importance)
                            residual_x = sample_inputs(law, int(config["n_oracle"]), dimension,
                                _rng(plan, "d2_oracle", seed, block, draw, "residual_X"),
                                p=p, importance=importance)
                            test_x = sample_inputs(law, int(config["n_test"]), dimension,
                                _rng(plan, "d2_probe", seed, block, draw, "test_X"),
                                p=p, importance=importance)
                            test_y = test_x @ teacher
                            single = job | {"world": world | {"M": width, "family": world.get("family", design)}}
                            control_name = ("independently_learned" if design == "independent_learned_width"
                                            else "nested_" + str(world.get("family", "random"))
                                            if design == "nested_fixed_bank" else "width_matched_random")
                            for noise_sigma in config.get("noise_levels", [0.05]):
                                y = draw_labels(X, teacher,
                                    _rng(plan, "d2_probe", seed, block, draw, "labels"), float(noise_sigma))
                                residual_y = draw_labels(residual_x, teacher,
                                    _rng(plan, "d2_oracle", seed, block, draw, "residual_y"), float(noise_sigma))
                                row = build_trial_row(
                                    plan, single, W=W, X_train=X, y_train=y, X_residual=residual_x,
                                    residual_labels=residual_y, teacher=teacher, law=law, importance=importance,
                                    p=p, active_count=None, target_kind=target_kind, noise_sigma=float(noise_sigma),
                                    target_ratio=n_fixed / (width + 1), probe_seed=int(seed * 100000 + block * 100 + draw),
                                    oracle_seed=int(seed * 100000 + block * 100 + draw + 1), draw_index=draw,
                                    intercept=True, control_name=control_name,
                                    representation_converged=trained["converged"] if trained else NA,
                                    reconstruction_loss=trained["validation_reconstruction_mse"] if trained else NA,
                                    feature_order=order, X_test=test_x, y_test=test_y,
                                )
                                row.update({"condition_id": f"{design}|{target_kind}|n={n_fixed}|M={width}|{order}",
                                            "noise_level": float(noise_sigma), "n_fixed": n_fixed,
                                            "representation_family": world.get("family", design)})
                                if trained:
                                    row.update(superposition_diagnostics(W))
                                    if (not trained.get("converged", False)
                                            and control_name in ("independently_learned", "nested_learned")):
                                        _mark_nonconverged(row)
                                if checkpoint_metadata and trained and control_name in (
                                        "independently_learned", "nested_learned"):
                                    row["representation_checkpoint_sha256"] = checkpoint_metadata["sha256"]
                                rows.append(row)
    return rows


def run_lane_p(plan: dict[str, Any], job: dict[str, Any]) -> list[dict[str, Any]]:
    cache = job["config"].get("activation_cache")
    if not cache:
        return []
    path = Path(cache).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        return []
    with np.load(path, allow_pickle=False) as archive:
        required = ("train_X", "train_y", "ref_X", "ref_y", "test_X", "test_y")
        absent = [name for name in required if name not in archive]
        if absent:
            raise ValueError(f"activation cache missing keys: {absent}")
        arrays = {name: np.asarray(archive[name], dtype=np.float64) for name in required}
    if any(arrays[k].ndim not in (2, 3) for k in ("train_X", "ref_X", "test_X")):
        raise ValueError("cached feature arrays must be [samples,features] or [samples,layers,features]")
    if any(arrays[k].ndim == 3 for k in ("train_X", "ref_X", "test_X")) and not all(
            arrays[k].ndim == 3 for k in ("train_X", "ref_X", "test_X")):
        raise ValueError("cached train/ref/test arrays must use the same layer convention")
    has_layer_axis = arrays["train_X"].ndim == 3
    layers = arrays["train_X"].shape[1] if has_layer_axis else 1
    rows: list[dict[str, Any]] = []
    for layer in range(min(layers, int(job["config"].get("max_layers", 3)))):
        Xtr = arrays["train_X"][:, layer, :] if has_layer_axis else arrays["train_X"]
        Xref = arrays["ref_X"][:, layer, :] if has_layer_axis else arrays["ref_X"]
        Xte = arrays["test_X"][:, layer, :] if has_layer_axis else arrays["test_X"]
        ytr, yref, yte = (arrays[name].reshape(-1) for name in ("train_y", "ref_y", "test_y"))
        split = Xref.shape[0] // 2
        if split < 2 or Xref.shape[0] - split < 2:
            raise ValueError("Lane P needs at least four independent reference examples")
        ref_fit = np.column_stack((np.ones(split), Xref[:split]))
        ref_eval = np.column_stack((np.ones(Xref.shape[0] - split), Xref[split:]))
        theta_star, _, _, _ = ordinary_least_squares(ref_fit, yref[:split])
        sigma = (ref_eval.T @ ref_eval) / ref_eval.shape[0]
        ref_residual = yref[split:] - ref_eval @ theta_star
        model_error = 0.5 * float(np.mean(ref_residual**2))
        residual_cross = ref_eval.T @ ref_residual / ref_eval.shape[0]
        residual_second_se = (float(np.std(ref_residual**2, ddof=1) / np.sqrt(ref_eval.shape[0]))
                              if ref_eval.shape[0] > 1 else NA)
        small_reference_size = max(2, split // 2)
        small_ref_fit = np.column_stack((np.ones(small_reference_size), Xref[:small_reference_size]))
        theta_small, _, _, _ = ordinary_least_squares(small_ref_fit, yref[:small_reference_size])
        small_residual = yref[split:] - ref_eval @ theta_small
        model_error_small_reference = 0.5 * float(np.mean(small_residual**2))
        try:
            sigma_factor = np.linalg.cholesky(0.5 * (sigma + sigma.T))
            residual_cross_norm = float(np.linalg.norm(np.linalg.solve(sigma_factor, residual_cross)))
            small_cross = ref_eval.T @ small_residual / ref_eval.shape[0]
            small_residual_cross_norm = float(np.linalg.norm(np.linalg.solve(sigma_factor, small_cross)))
        except np.linalg.LinAlgError:
            residual_cross_norm = NA
            small_residual_cross_norm = NA
        d = Xtr.shape[1] + 1
        for n, ratio in sample_sizes(d, job["config"]["sample_ratios"], maximum=Xtr.shape[0]):
            _raise_if_preempted()
            H = np.column_stack((np.ones(n), Xtr[:n]))
            theta_hat, rank, tolerance, _ = ordinary_least_squares(H, ytr[:n])
            test_H = np.column_stack((np.ones(Xte.shape[0]), Xte))
            risk_test = 0.5 * float(np.mean((test_H @ theta_hat - yte) ** 2))
            reference = {
                "status": "COMPLETED_ESTIMATED_ORACLE", "A_err": NA, "A_spec": NA,
                "A_dep": NA, "A_ind": NA, "D0": NA, "D_e": NA, "S_e": NA,
                "R_direct_pop": NA, "R_direct_e": NA, "R_direct_ls": NA,
                "R_metric_min_norm": NA, "R_euclidean_min_norm": NA,
                "cond_K": NA, "min_eig_K": NA, "trace_K_inv": NA,
                "rel_decomposition_error": NA, "interpolation_residual": NA,
                "reference_eval_size": ref_eval.shape[0],
                "small_oracle_reference_size": small_reference_size,
                "oracle_residual_feature_cross_norm": residual_cross_norm,
                "oracle_residual_second_moment_se": residual_second_se,
                "M_err_small_reference": model_error_small_reference,
                "oracle_residual_feature_cross_norm_small_reference": small_residual_cross_norm,
                "warning_codes": "ESTIMATED_ORACLE|APPROXIMATE_ORACLE_CROSS_TERM",
            }
            if n <= d:
                try:
                    fitted = interpolating_population_gad(H, ytr[:n], sigma, theta_star, model_error)
                    spectral = spectral_dependence_diagnostics(fitted["K"],
                                                              ytr[:n] - H @ theta_star,
                                                              ref_residual)
                    independent = independent_residual_control(
                        fitted["K"], ref_residual,
                        _rng(plan, "P_independent_residual", layer, n), draws=1)
                    theta_circle_delta = fitted["theta_circle"] - theta_star
                    theta_zero_delta = fitted["theta_0"] - theta_star
                    theta_e_delta = fitted["theta_e"] - theta_star
                    risk_circle = model_error - float(theta_circle_delta @ residual_cross) + 0.5 * float(
                        theta_circle_delta @ sigma @ theta_circle_delta)
                    risk_metric_min_norm = model_error - float(theta_zero_delta @ residual_cross) + 0.5 * float(
                        theta_zero_delta @ sigma @ theta_zero_delta)
                    risk_euclidean = model_error - float(theta_e_delta @ residual_cross) + 0.5 * float(
                        theta_e_delta @ sigma @ theta_e_delta)
                    decomposition = model_error + fitted["A"] + fitted["D_e"]
                    relative_decomposition = abs(risk_euclidean - decomposition) / max(
                        1.0, abs(risk_euclidean), abs(decomposition)
                    )
                    reference.update({
                        "A_err": fitted["A"], "A_spec": spectral["A_spec"],
                        "A_dep": spectral["A_dep"], "A_ind": independent["A_ind"],
                        "D0": fitted["D0"], "D_e": fitted["D_e"], "S_e": fitted["S_e"],
                        "R_direct_pop": risk_euclidean, "R_direct_e": risk_euclidean,
                        "R_direct_ls": risk_euclidean,
                        "R_metric_min_norm": risk_metric_min_norm,
                        "R_oracle_nearest_interpolant": risk_circle,
                        "R_euclidean_min_norm": risk_euclidean,
                        "condition_H_whitened": fitted["diagnostics"]["condition_H_whitened"],
                        "cond_K": fitted["diagnostics"]["condition_K"],
                        "min_eig_K": float(np.linalg.eigvalsh(fitted["K"])[0]),
                        "trace_K_inv": float(np.trace(np.linalg.solve(fitted["K"], np.eye(n)))),
                        "rel_decomposition_error": relative_decomposition,
                        "signed_identity_error": fitted["signed_identity_error"],
                        "interpolation_residual": fitted["interpolation_residual"],
                        "warning_codes": "ESTIMATED_ORACLE|ESTIMATED_GAD_COMPONENTS|APPROXIMATE_ORACLE_CROSS_TERM",
                    })
                    interpolation_tolerance = 1e-10 + 1e-8 * max(1.0, float(np.linalg.norm(ytr[:n])))
                    if (not fitted["diagnostics"]["finite"]
                            or fitted["interpolation_residual"] > interpolation_tolerance):
                        reference.update({"status": "NUMERICAL_FAILURE", "trial_valid": False,
                                          "warning_codes": "ESTIMATED_ORACLE|NUMERICAL_FAILURE|INTERPOLATION_RESIDUAL"})
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    reference.update({"status": "SINGULAR_DESIGN",
                                      "trial_valid": False,
                                      "warning_codes": "ESTIMATED_ORACLE|REFERENCE_OR_SINGULAR_DESIGN",
                                      "status_reason": str(exc)})
            else:
                try:
                    estimated = noninterpolating_risk(H, ytr[:n], sigma, theta_star,
                                                       theta_hat, model_error)
                    estimate_delta = theta_hat - theta_star
                    direct_risk = model_error - float(estimate_delta @ residual_cross) + 0.5 * float(
                        estimate_delta @ sigma @ estimate_delta
                    )
                    reference.update({"R_direct_pop": direct_risk,
                                      "R_direct_e": direct_risk,
                                      "R_direct_ls": direct_risk,
                                      "R_euclidean_min_norm": direct_risk})
                    reference["interpolation_residual"] = estimated["interpolation_residual"]
                    reference["warning_codes"] = "ESTIMATED_ORACLE|INTERPOLATING_COMPONENTS_NA"
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    reference.update({"status": "SINGULAR_DESIGN",
                                      "trial_valid": False,
                                      "warning_codes": "ESTIMATED_ORACLE|REFERENCE_COVARIANCE_SINGULAR",
                                      "status_reason": str(exc)})
            row = base_row(plan, job) | {
                "input_law": "cached_real_activations", "target_family": "cached_labels",
                "lane": "P", "N": Xtr.shape[1], "M": layers, "d": d, "n": n,
                "n_over_d": n / d, "rank_H": rank, "effective_rank": rank,
                "rank_tolerance": tolerance, "rank_basis": "unwhitened_H",
                "solver_path": "raw_design_svd_least_squares",
                "M_err": model_error, "R_test_mc": risk_test,
                "is_oracle_exact": False,
                "oracle_estimated": True, "oracle_reference_size": split,
                "reference_size": split, "is_interpolating": n <= rank,
                "trial_valid": True, "condition_id": f"P|layer={layer}|n={n}",
                "control_name": "cached_model", "target_kind": "cached_labels",
                "estimator_name": "ordinary_least_squares", "ratio_target": ratio,
                "input_cache_sha256": sha256_file(path),
            } | reference
            rows.append(row)
    return rows


def execute_job(plan: dict[str, Any], job: dict[str, Any], run_dir: Path) -> tuple[list[dict[str, Any]], str]:
    lane = job["lane"]
    if lane == "A":
        rows = run_lane_a(plan, job)
    elif lane == "B":
        rows = run_lane_b(plan, job)
    elif lane == "C":
        rows = run_lane_c(plan, job, run_dir)
    elif lane == "D1":
        rows = run_lane_d1(plan, job, run_dir)
    elif lane == "D2":
        rows = run_lane_d2(plan, job, run_dir)
    elif lane == "P":
        rows = run_lane_p(plan, job)
        if not rows:
            return [], "SKIPPED_DEPENDENCY"
    else:
        raise ValueError(f"unknown lane: {lane}")
    return rows, "COMPLETED"


def run_one(run_dir: Path, job_index: int) -> dict[str, Any]:
    plan = load_plan(run_dir)
    jobs = plan["jobs"]
    if not 0 <= job_index < len(jobs):
        raise IndexError(f"job index {job_index} is outside [0,{len(jobs)})")
    job = jobs[job_index]
    if int(job["job_index"]) != job_index:
        raise ValueError("job index does not match immutable plan ordering")
    shard_dir = run_dir / "lanes" / job["lane"] / "shards" / job["job_id"]
    shard_dir.mkdir(parents=True, exist_ok=True)
    status_path = shard_dir / "job_status.json"
    with file_lock(shard_dir / ".lock"):
        if status_path.is_file():
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            if previous.get("status") in ("COMPLETED", "SKIPPED_DEPENDENCY"):
                return previous | {"reused": True}
        attempts = sorted(shard_dir.glob("attempt_[0-9][0-9][0-9][0-9]"))
        attempt = shard_dir / f"attempt_{len(attempts) + 1:04d}"
        attempt.mkdir(parents=True)
        started = utc_now()
        provenance = runtime_provenance()
        try:
            rows, status = execute_job(plan, job, run_dir)
            if status == "SKIPPED_DEPENDENCY":
                skip_reason = ("activation_cache_not_configured"
                               if not job["config"].get("activation_cache")
                               else "activation_cache_missing")
                rows = [base_row(plan, job) | {
                    "status": status, "status_reason": skip_reason,
                    "trial_valid": NA, "warning_codes": "SKIPPED_DEPENDENCY",
                }]
            elif not rows:
                rows = [base_row(plan, job) | {"status": "INCOMPLETE", "trial_valid": False,
                                               "warning_codes": "EMPTY_SHARD"}]
                status = "INCOMPLETE"
            csv_payload = csv_bytes(rows)
            atomic_write(attempt / "raw.csv", csv_payload)
            shard_status = {
                "job_id": job["job_id"], "job_index": job_index, "lane": job["lane"],
                "status": status, "attempt": attempt.name, "rows": len(rows),
                "started_utc": started, "finished_utc": utc_now(),
                "raw_sha256": hashlib.sha256(csv_payload).hexdigest(),
                "provenance": provenance, "warnings": sorted(set(
                    str(row.get("warning_codes", "")) for row in rows if row.get("warning_codes"))),
            }
            atomic_json(attempt / "status.json", shard_status)
            atomic_json(status_path, shard_status)
            return shard_status
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            if "DEPENDENCY_MISSING" in reason:
                status = "DEPENDENCY_MISSING"
            elif "TIMEOUT" in reason:
                status = "TIMEOUT"
            else:
                status = "FAILED"
            failed_row = base_row(plan, job) | {"status": status, "status_reason": reason,
                                               "trial_valid": False, "warning_codes": status}
            payload = csv_bytes([failed_row])
            atomic_write(attempt / "raw.csv", payload)
            atomic_write(attempt / "traceback.txt", traceback.format_exc().encode("utf-8"))
            shard_status = {
                "job_id": job["job_id"], "job_index": job_index, "lane": job["lane"],
                "status": status, "attempt": attempt.name, "rows": 1,
                "started_utc": started, "finished_utc": utc_now(), "reason": reason,
                "raw_sha256": hashlib.sha256(payload).hexdigest(), "provenance": provenance,
            }
            atomic_json(attempt / "status.json", shard_status)
            atomic_json(status_path, shard_status)
            return shard_status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--lane", choices=("A", "B", "C", "D1", "D2", "P"))
    parser.add_argument("--profile", choices=("smoke", "overnight_discovery", "confirmation"))
    parser.add_argument("--job-index", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, _handle_preempt)
        selected_plan = load_plan(args.run_dir.resolve())
        if args.profile and args.profile != selected_plan["profile"]:
            parser.error("--profile does not match run plan")
        if not 0 <= args.job_index < len(selected_plan["jobs"]):
            parser.error("--job-index is outside the run plan")
        if args.lane and selected_plan["jobs"][args.job_index]["lane"] != args.lane:
            parser.error("--lane does not match selected plan job")
        status = run_one(args.run_dir.resolve(), args.job_index)
        print(json.dumps(status, sort_keys=True, indent=2))
        return 0 if status.get("status") in ("COMPLETED", "SKIPPED_DEPENDENCY") else 1
    except (OSError, ValueError, IndexError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
