"""Shared deterministic shard, data, and risk helpers for Chapter 2 v3."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any, Iterator

import numpy as np

from aliasing.chapter2_v3 import (
    bernoulli_probabilities,
    bernoulli_uniform,
    bernoulli_uniform_moments,
    feature_matrix,
    normalized_teacher,
    random_rows,
    rng_for,
    sparse_fixed_k,
)
from aliasing.intrinsic_gad import (
    GAD_DEFINITION,
    SingularDesignError,
    independent_residual_control,
    interpolating_population_gad,
    noninterpolating_risk,
    population_linear_oracle,
    spectral_dependence_diagnostics,
)


NA = "NA"
CSV_FIELDS = [
    "run_id", "git_sha", "code_hash", "config_hash", "gad_definition", "experiment", "lane",
    "profile", "world_id", "status", "status_reason", "representation_family",
    "representation_seed", "probe_seed", "oracle_seed", "input_law", "target_family",
    "importance_profile", "N", "M", "d", "n", "n_over_d", "rank_H", "sigma_epsilon",
    "sparsity_k", "sparsity_p", "cond_Sigma", "condition_H_whitened", "cond_K", "min_eig_K", "trace_K_inv",
    "residual_second_moment", "M_err", "A_err", "A_spec", "A_dep", "A_ind", "D0", "D_e",
    "S_e", "R_direct_pop", "R_direct_e", "R_direct_ls", "R_metric_min_norm",
    "R_oracle_nearest_interpolant", "R_euclidean_min_norm", "A_expected_gaussian",
    "D0_expected_gaussian",
    "rel_decomposition_error", "signed_identity_error", "interpolation_residual",
    "solver_path", "rank_tolerance", "rank_basis",
    "representation_converged", "reconstruction_loss", "elapsed_seconds", "warning_codes",
    "is_interpolating", "is_oracle_exact", "oracle_reference_size", "trial_valid", "effective_rank",
    "condition_id", "control_name", "target_kind", "noise_level", "active_count", "ratio_target",
    "n_requested", "seed_block", "draw_index", "target_signal_variance", "model_error_fraction",
    "A_normalized", "bottom_eigenmode_share_A", "sample_mean", "sample_median", "sample_q05",
    "sample_q95", "estimator_name", "oracle_estimated", "reference_size", "feature_order",
    "R_test_mc", "n_fixed", "input_cache_sha256", "representation_checkpoint_sha256",
    "reference_eval_size", "small_oracle_reference_size", "oracle_residual_feature_cross_norm",
    "oracle_residual_second_moment_se", "M_err_small_reference",
    "oracle_residual_feature_cross_norm_small_reference",
    "encoder_coherence", "encoder_stable_rank", "encoder_norm_min", "encoder_norm_max",
    "train_data_sha256", "residual_data_sha256", "teacher_sha256", "feature_matrix_sha256",
    "labels_sha256", "representation_sha256",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, (json.dumps(value, sort_keys=True, indent=2, default=json_default) + "\n").encode())


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def csv_bytes(rows: list[dict[str, Any]], fields: list[str] | None = None) -> bytes:
    columns = fields or CSV_FIELDS
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: csv_value(row.get(key, NA)) for key in columns})
    return buffer.getvalue().encode("utf-8")


def csv_value(value: Any) -> Any:
    if value is None:
        return NA
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), separators=(",", ":"))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return "NONFINITE"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def file_lock(path: Path) -> Iterator[None]:
    """Context manager yielding under an exclusive flock; process death releases it."""
    class Lock:
        def __enter__(self) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = path.open("a+")
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)

        def __exit__(self, *_: Any) -> None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()

    return Lock()  # type: ignore[return-value]


def base_row(plan: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    world = job["world"]
    return {
        "run_id": plan.get("run_id", plan["profile"]), "git_sha": plan["git"]["git_sha"],
        "code_hash": plan["code_sha256"], "config_hash": plan["config_sha256"],
        "gad_definition": GAD_DEFINITION, "experiment": "chapter2_v3", "lane": job["lane"],
        "profile": plan["profile"], "world_id": job["job_id"], "status": "COMPLETED",
        "status_reason": "", "representation_seed": job.get("representation_seed", NA),
        "seed_block": job.get("probe_seed_block", NA), "N": world.get("input_dimension", NA),
        "M": world.get("M", NA), "importance_profile": world.get("importance_profile", NA),
        "representation_converged": NA, "reconstruction_loss": NA, "oracle_reference_size": "NA",
        "is_oracle_exact": True, "oracle_estimated": False, "trial_valid": True,
        "warning_codes": "",
    }


def covariance_moments(law: str, dimension: int, *, p: float = 0.1,
                       importance: str = "uniform") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if law == "gaussian" or law == "exact_k_gaussian":
        mean = np.zeros(dimension, dtype=np.float64)
        second = np.eye(dimension, dtype=np.float64)
        return mean, second, second
    if law == "bernoulli_uniform":
        probabilities = bernoulli_probabilities(p, dimension, importance)
        mean, second = bernoulli_uniform_moments(probabilities)
        return mean, second, second - np.outer(mean, mean)
    raise ValueError(f"unknown input law: {law}")


def sample_inputs(law: str, count: int, dimension: int, rng: np.random.Generator,
                  *, active_count: int | None = None, p: float = 0.1,
                  importance: str = "uniform") -> np.ndarray:
    if count < 1:
        raise ValueError("sample count must be positive")
    if law == "gaussian":
        return rng.normal(size=(count, dimension))
    if law == "exact_k_gaussian":
        if active_count is None:
            raise ValueError("exact-k input requires active_count")
        if active_count == dimension:
            return rng.normal(size=(count, dimension))
        return sparse_fixed_k(count, dimension, active_count, rng)
    if law == "bernoulli_uniform":
        probabilities = bernoulli_probabilities(p, dimension, importance)
        return bernoulli_uniform(count, probabilities, rng)
    raise ValueError(f"unknown input law: {law}")


def teacher_for(kind: str, dimension: int, W: np.ndarray, centered_covariance: np.ndarray,
                rng: np.random.Generator, importance: str = "uniform") -> np.ndarray:
    return normalized_teacher(kind, dimension, W, centered_covariance, rng, importance)


def population_oracle(W: np.ndarray, law: str, teacher: np.ndarray, *, intercept: bool,
                      p: float = 0.1, importance: str = "uniform") -> dict[str, Any]:
    mean, second, _ = covariance_moments(law, W.shape[1], p=p, importance=importance)
    return population_linear_oracle(W, second, teacher, intercept=intercept, mean_x=mean)


def draw_labels(X: np.ndarray, teacher: np.ndarray, rng: np.random.Generator,
                noise_sigma: float) -> np.ndarray:
    labels = X @ teacher
    if noise_sigma:
        labels = labels + rng.normal(scale=noise_sigma, size=X.shape[0])
    return np.asarray(labels, dtype=np.float64)


def sample_sizes(dimension: int, ratios: list[float], offsets: list[int] | None = None,
                 maximum: int | None = None, minimum: int = 2) -> list[tuple[int, float]]:
    requested: dict[int, float] = {}
    for ratio in ratios:
        n = max(minimum, int(round(float(ratio) * dimension)))
        requested.setdefault(n, float(ratio))
    for offset in offsets or []:
        n = dimension + int(offset)
        if n >= minimum:
            requested[n] = n / dimension
    if maximum is not None:
        requested = {n: r for n, r in requested.items() if n <= maximum}
    return sorted(requested.items())


def build_trial_row(plan: dict[str, Any], job: dict[str, Any], *, W: np.ndarray,
                    X_train: np.ndarray, y_train: np.ndarray, X_residual: np.ndarray,
                    residual_labels: np.ndarray, teacher: np.ndarray, law: str,
                    importance: str, p: float, active_count: int | None,
                    target_kind: str, noise_sigma: float, target_ratio: float,
                    probe_seed: int, oracle_seed: int, draw_index: int,
                    intercept: bool, control_name: str = "learned",
                    representation_converged: Any = NA, reconstruction_loss: Any = NA,
                    feature_order: str = "native", feature_transform: np.ndarray | None = None,
                    X_test: np.ndarray | None = None, y_test: np.ndarray | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    row = base_row(plan, job)
    hidden_train = feature_matrix(X_train, W, intercept=intercept)
    hidden_residual = feature_matrix(X_residual, W, intercept=intercept)
    oracle = population_oracle(W, law, teacher, intercept=intercept, p=p, importance=importance)
    sigma = oracle["Sigma"]
    theta_star = oracle["theta_star"]
    if feature_transform is not None:
        transform = np.asarray(feature_transform, dtype=np.float64)
        if transform.shape != sigma.shape:
            raise ValueError("feature transform must match augmented feature dimension")
        hidden_train = hidden_train @ transform.T
        hidden_residual = hidden_residual @ transform.T
        sigma = transform @ sigma @ transform.T
        theta_star = np.linalg.solve(transform.T, theta_star)
    model_error = float(oracle["M"])
    train_residual = y_train - hidden_train @ theta_star
    residual_pool = residual_labels - hidden_residual @ theta_star
    residual_second = max(0.0, 2.0 * model_error + noise_sigma**2)
    n, d = hidden_train.shape
    row.update({
        "representation_family": job["world"].get("family", control_name),
        "control_name": control_name, "target_family": target_kind,
        "importance_profile": importance, "input_law": law, "N": int(W.shape[1]),
        "M": int(W.shape[0]), "d": int(d), "n": int(n), "n_over_d": n / d,
        "sigma_epsilon": float(noise_sigma), "sparsity_k": active_count if active_count is not None else NA,
        "sparsity_p": p if law == "bernoulli_uniform" else NA,
                "M_err": model_error, "residual_second_moment": residual_second,
        "representation_converged": representation_converged,
        "reconstruction_loss": reconstruction_loss, "probe_seed": int(probe_seed),
        "oracle_seed": int(oracle_seed), "draw_index": int(draw_index),
        "ratio_target": float(target_ratio), "n_requested": n,
        "target_signal_variance": float(teacher @ covariance_moments(law, W.shape[1], p=p,
                                             importance=importance)[2] @ teacher),
        "model_error_fraction": model_error / max(model_error + 0.5 * float(teacher @ covariance_moments(
            law, W.shape[1], p=p, importance=importance)[2] @ teacher), 1e-15),
        "cond_Sigma": float(np.linalg.cond(sigma)),
        "is_oracle_exact": True, "oracle_reference_size": "NA", "trial_valid": True,
        "feature_order": feature_order,
        "train_data_sha256": sha256_array(np.column_stack((X_train, y_train))),
        "residual_data_sha256": sha256_array(np.column_stack((X_residual, residual_labels))),
        "teacher_sha256": sha256_array(teacher),
        "feature_matrix_sha256": sha256_array(hidden_train),
        "labels_sha256": sha256_array(y_train),
        "representation_sha256": sha256_array(W),
    })
    try:
        if n <= d:
            result = interpolating_population_gad(hidden_train, y_train, sigma, theta_star, model_error)
            diagnostics = result["diagnostics"]
            row.update({
                "rank_H": diagnostics["rank_H"], "rank_tolerance": diagnostics["rank_tolerance"],
                "solver_path": diagnostics["solver_path"], "rank_basis": diagnostics["rank_basis"],
                "condition_H_whitened": diagnostics["condition_H_whitened"],
                "cond_K": diagnostics["condition_K"],
                "min_eig_K": float(np.linalg.eigvalsh(result["K"])[0]),
                "trace_K_inv": float(np.trace(np.linalg.solve(result["K"], np.eye(n)))),
                "A_err": result["A"], "D0": result["D0"], "D_e": result["D_e"],
                "S_e": result["S_e"], "R_direct_pop": result["R_direct"],
                "R_direct_e": result["R_direct"], "R_direct_ls": result["R_direct"],
                # theta_0 is the interpolant with minimum population-metric norm.
                # Its risk includes both intrinsic aliasing and D0. The
                # oracle-nearest interpolant theta_circle has risk M + A.
                "R_metric_min_norm": model_error + result["A"] + result["D0"],
                "R_oracle_nearest_interpolant": model_error + result["A"],
                "R_euclidean_min_norm": result["R_direct"],
                "rel_decomposition_error": result["rel_decomposition_error"],
                "signed_identity_error": result["signed_identity_error"],
                "interpolation_residual": result["interpolation_residual"],
                "is_interpolating": True, "effective_rank": diagnostics["rank_H"],
            })
            independent = independent_residual_control(result["K"], residual_pool,
                            rng_for(int(plan["config"]["seed"]), "independent_residual", probe_seed),
                            draws=1)
            spectral = spectral_dependence_diagnostics(result["K"], train_residual, residual_pool)
            row.update({"A_ind": independent["A_ind"], "A_spec": spectral["A_spec"],
                        "A_dep": spectral["A_dep"],
                        "bottom_eigenmode_share_A": spectral["bottom_eigenmode_share_A"],
                        "A_normalized": result["A"] / max(residual_second, 1e-15)})
            if law == "gaussian" and not intercept:
                row["D0_expected_gaussian"] = 0.5 * (1.0 - n / d) * float(theta_star @ sigma @ theta_star)
                if n < d - 1:
                    row["A_expected_gaussian"] = 0.5 * residual_second * n / (d - n - 1)
            row["residual_second_moment"] = spectral["residual_second_moment"]
            row["sample_mean"] = result["R_direct"]
            row["sample_median"] = result["R_direct"]
            interpolation_tolerance = 1e-10 + 1e-8 * max(1.0, float(np.linalg.norm(y_train)))
            identity_scale = max(1.0, abs(result["R_direct"]), abs(result["R_decomposition"]),
                                 abs(result["D0"]), abs(result["D_e"]))
            identity_tolerance = 1e-10 + 1e-8 * identity_scale
            if (not diagnostics["finite"]
                    or result["interpolation_residual"] > interpolation_tolerance):
                row.update({"status": "NUMERICAL_FAILURE", "trial_valid": False,
                            "warning_codes": "NUMERICAL_FAILURE|INTERPOLATION_RESIDUAL"})
            elif (result["rel_decomposition_error"] > 1e-8
                  or result["signed_identity_error"] > identity_tolerance):
                row.update({"status": "INVALID_MATH", "trial_valid": False,
                            "warning_codes": "INVALID_MATH|IDENTITY_RESIDUAL"})
            else:
                row["status"] = "COMPLETED"
            if X_test is not None and y_test is not None:
                test_features = feature_matrix(X_test, W, intercept=intercept)
                if feature_transform is not None:
                    test_features = test_features @ feature_transform.T
                row["R_test_mc"] = 0.5 * float(np.mean((test_features @ result["theta_e"] - y_test) ** 2))
        else:
            theta_hat, rank, tolerance, condition = ordinary_least_squares(hidden_train, y_train)
            result = noninterpolating_risk(hidden_train, y_train, sigma, theta_star, theta_hat, model_error)
            row.update({
                "rank_H": rank, "effective_rank": rank, "rank_tolerance": tolerance,
                "solver_path": "raw_design_svd_least_squares", "rank_basis": "unwhitened_H",
                "cond_K": NA, "min_eig_K": NA,
                "trace_K_inv": NA, "A_err": NA, "A_spec": NA, "A_dep": NA, "A_ind": NA,
                "D0": NA, "D_e": NA, "S_e": NA, "R_direct_pop": result["R_direct"],
                "R_direct_ls": result["R_direct"], "R_direct_e": result["R_direct"],
                "R_metric_min_norm": NA, "R_oracle_nearest_interpolant": NA,
                "R_euclidean_min_norm": result["R_direct"],
                "rel_decomposition_error": NA,
                "interpolation_residual": result["interpolation_residual"],
                "is_interpolating": False, "sample_mean": result["R_direct"],
                "sample_median": result["R_direct"], "warning_codes": "INTERPOLATING_COMPONENTS_NA",
            })
            factor = np.linalg.cholesky(sigma)
            whitened = np.linalg.solve(factor, hidden_train.T).T
            singular_values = np.linalg.svd(whitened, compute_uv=False)
            row["condition_H_whitened"] = (
                float(singular_values[0] / singular_values[rank - 1])
                if rank > 0 and singular_values.size >= rank else NA
            )
            if X_test is not None and y_test is not None:
                test_features = feature_matrix(X_test, W, intercept=intercept)
                if feature_transform is not None:
                    test_features = test_features @ feature_transform.T
                row["R_test_mc"] = 0.5 * float(np.mean((test_features @ theta_hat - y_test) ** 2))
        if row.get("status") in (None, ""):
            row["status"] = "COMPLETED"
    except SingularDesignError as exc:
        diag = exc.diagnostics
        row.update({
            "status": "SINGULAR_DESIGN", "status_reason": str(exc), "trial_valid": False,
            "rank_H": diag.get("rank_H", NA), "rank_tolerance": diag.get("rank_tolerance", NA),
            "condition_H_whitened": diag.get("condition_H_whitened", NA),
            "solver_path": diag.get("solver_path", "raw_design_svd_rank_cholesky_whitened_kernel"),
            "rank_basis": diag.get("rank_basis", "unwhitened_H"),
            "is_interpolating": False, "A_err": NA, "A_spec": NA, "A_dep": NA, "A_ind": NA,
            "D0": NA, "D_e": NA, "S_e": NA, "R_direct_pop": NA,
            "warning_codes": "SINGULAR_DESIGN",
        })
    except (np.linalg.LinAlgError, FloatingPointError) as exc:
        row.update({"status": "NUMERICAL_FAILURE", "status_reason": str(exc),
                    "trial_valid": False, "warning_codes": "NUMERICAL_FAILURE"})
    row["elapsed_seconds"] = time.perf_counter() - started
    return row


def ordinary_least_squares(H: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int, float, float]:
    """SVD least-squares fit with an explicit, recorded numerical rank rule."""
    U, values, Vt = np.linalg.svd(H, full_matrices=False)
    largest = float(values[0]) if values.size else 0.0
    tolerance = max(H.shape) * np.finfo(np.float64).eps * largest
    keep = values > tolerance
    rank = int(np.count_nonzero(keep))
    coefficients = (U[:, keep].T @ y) / values[keep] if rank else np.zeros(0)
    theta = Vt[keep, :].T @ coefficients if rank else np.zeros(H.shape[1])
    condition = float(values[0] / values[rank - 1]) if rank else float("inf")
    return theta, rank, tolerance, condition


def actual_feature_rank(H: np.ndarray) -> tuple[int, float, float]:
    singular_values = np.linalg.svd(H, compute_uv=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = max(H.shape) * np.finfo(np.float64).eps * largest
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition = float(singular_values[0] / singular_values[rank - 1]) if rank else float("inf")
    return rank, tolerance, condition


def runtime_provenance() -> dict[str, Any]:
    try:
        import scipy
        scipy_version: str | None = scipy.__version__
    except ImportError:
        scipy_version = None
    return {
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy_version,
        "platform": platform.platform(), "device": "cpu_numpy",
        "precision": "float64", "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NOT_IN_SLURM"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "NOT_IN_SLURM"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", "NOT_IN_SLURM"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", "NOT_IN_SLURM"),
        "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE", "NOT_IN_SLURM"),
        "slurm_time_limit": os.environ.get("SLURM_TIMELIMIT", "NOT_IN_SLURM"),
        "started_utc": utc_now(),
    }
