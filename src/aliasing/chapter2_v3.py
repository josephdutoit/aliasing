"""Shared deterministic data and representation helpers for Chapter 2 v3."""
from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np


def stable_seed(seed: int, *coordinates: Any) -> int:
    """Derive a stable 32-bit seed from named independent RNG coordinates."""
    payload = "\x1f".join([str(int(seed)), *(str(value) for value in coordinates)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def rng_for(seed: int, *coordinates: Any) -> np.random.Generator:
    return np.random.default_rng(stable_seed(seed, *coordinates))


def sparse_fixed_k(count: int, dimension: int, active_count: int, rng: np.random.Generator) -> np.ndarray:
    """Sample exact-k Gaussian inputs with E[x]=0 and E[xx.T]=I."""
    if count < 1 or dimension < 1 or not 1 <= active_count <= dimension:
        raise ValueError("invalid sample count, dimension, or active_count")
    selected = np.argpartition(rng.random((count, dimension)), active_count - 1, axis=1)[:, :active_count]
    values = rng.normal(0.0, np.sqrt(dimension / active_count), size=(count, active_count))
    result = np.zeros((count, dimension), dtype=np.float64)
    result[np.arange(count)[:, None], selected] = values
    return result


def bernoulli_probabilities(base_probability: float, dimension: int, profile: str) -> np.ndarray:
    if dimension < 2 or not 0 < base_probability < 1:
        raise ValueError("dimension must be >=2 and probability must be in (0,1)")
    if profile == "uniform":
        return np.full(dimension, base_probability, dtype=np.float64)
    if profile == "two_tier":
        result = np.full(dimension, 0.5 * base_probability, dtype=np.float64)
        result[: dimension // 2] = 1.5 * base_probability
        if np.any(result >= 1):
            raise ValueError("two_tier profile requires base_probability < 2/3")
        return result
    raise ValueError("profile must be 'uniform' or 'two_tier'")


def bernoulli_uniform(count: int, probabilities: Any, rng: np.random.Generator) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or p.size < 1 or np.any((p <= 0) | (p >= 1)):
        raise ValueError("probabilities must be a vector with entries in (0,1)")
    active = rng.random((count, p.size)) < p[None, :]
    amplitude = rng.random((count, p.size))
    return (active * amplitude).astype(np.float64, copy=False)


def bernoulli_uniform_moments(probabilities: Any) -> tuple[np.ndarray, np.ndarray]:
    """Exact mean and raw second moment for Bernoulli(p)*Uniform(0,1)."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or p.size < 1 or np.any((p <= 0) | (p >= 1)):
        raise ValueError("probabilities must be a vector with entries in (0,1)")
    mean = p / 2.0
    variance = p / 3.0 - p**2 / 4.0
    second = np.diag(variance) + np.outer(mean, mean)
    return mean, second


def normalized_teacher(
    kind: str,
    dimension: int,
    W: np.ndarray,
    covariance: np.ndarray,
    rng: np.random.Generator,
    importance_profile: str = "uniform",
) -> np.ndarray:
    """Construct a raw-input linear target with unit population variance."""
    if kind == "in_span":
        raw = W.T @ rng.normal(size=W.shape[0])
    elif kind == "dense":
        raw = rng.normal(size=dimension)
    elif kind == "sparse":
        raw = np.zeros(dimension, dtype=np.float64)
        count = max(1, min(dimension, int(np.ceil(np.sqrt(dimension)))))
        indices = rng.choice(dimension, size=count, replace=False)
        raw[indices] = rng.normal(size=count)
    else:
        raise ValueError("target kind must be in_span, dense, or sparse")
    if importance_profile == "two_tier":
        raw[: dimension // 2] *= 1.5
        raw[dimension // 2 :] *= 0.5
    elif importance_profile != "uniform":
        raise ValueError("unknown importance profile")
    variance = float(raw @ covariance @ raw)
    if not np.isfinite(variance) or variance <= np.finfo(float).eps:
        raise FloatingPointError("teacher has zero or nonfinite population variance")
    return raw / np.sqrt(variance)


def feature_matrix(inputs: Any, W: Any, intercept: bool = True) -> np.ndarray:
    x = np.asarray(inputs, dtype=np.float64)
    weight = np.asarray(W, dtype=np.float64)
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("inputs and W must be matrices with matching input dimensions")
    hidden = x @ weight.T
    return np.column_stack((np.ones(x.shape[0]), hidden)) if intercept else hidden


def random_rows(width: int, dimension: int, rng: np.random.Generator, family: str = "gaussian") -> np.ndarray:
    if not 1 <= width <= dimension:
        raise ValueError("width must be between one and dimension")
    if family == "gaussian":
        return rng.normal(size=(width, dimension)) / np.sqrt(dimension)
    if family == "orthonormal":
        q, _ = np.linalg.qr(rng.normal(size=(dimension, width)), mode="reduced")
        return q.T
    raise ValueError("family must be gaussian or orthonormal")


def autoencoder_loss(
    x: np.ndarray,
    W: np.ndarray,
    bias: np.ndarray,
    sparsity_penalty: float,
    weight_decay: float,
) -> tuple[float, float]:
    z = x @ W.T
    reconstruction = np.maximum(z @ W + bias, 0.0)
    reconstruction_mse = float(np.mean((reconstruction - x) ** 2))
    sparsity = float(np.mean(np.abs(z)))
    objective = reconstruction_mse + sparsity_penalty * sparsity + 0.5 * weight_decay * float(np.sum(W**2))
    return objective, reconstruction_mse


def train_tied_sparse_autoencoder(
    x_train: Any,
    x_validation: Any,
    width: int,
    seed: int,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    sparsity_penalty: float,
    weight_decay: float,
    patience: int,
    resume_state: dict[str, Any] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Train the tied map x->Wx with decoder ReLU(W.T W x+b).

    Parameters are optimized in NumPy with deterministic Adam so the central
    representation lane remains portable to CPU-only cluster environments.
    Convergence means a finite validation objective followed by the configured
    plateau window; reaching the step limit without that condition is reported
    as a nonconverged representation rather than silently accepted.
    """
    train = np.asarray(x_train, dtype=np.float64)
    validation = np.asarray(x_validation, dtype=np.float64)
    if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != validation.shape[1]:
        raise ValueError("training and validation inputs must be compatible matrices")
    if not (1 <= width <= train.shape[1] and steps >= 1 and batch_size >= 1 and patience >= 1):
        raise ValueError("invalid width or training limits")
    if learning_rate <= 0 or sparsity_penalty < 0 or weight_decay < 0:
        raise ValueError("invalid optimizer or regularization settings")
    if not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise ValueError("training inputs contain nonfinite values")

    rng = rng_for(seed, "tied_autoencoder", width)
    W = rng.normal(size=(width, train.shape[1])) / np.sqrt(train.shape[1])
    initial_z = train @ W.T
    initial_pre = initial_z @ W
    bias = -np.quantile(initial_pre, 0.70, axis=0)
    initial_objective, initial_mse = autoencoder_loss(
        validation, W, bias, sparsity_penalty, weight_decay
    )
    first_W, second_W = np.zeros_like(W), np.zeros_like(W)
    first_b, second_b = np.zeros_like(bias), np.zeros_like(bias)
    best_W, best_bias = W.copy(), bias.copy()
    best_validation, best_step = initial_objective, 0
    history: list[dict[str, float]] = []
    interval = max(1, steps // 50)
    checkpoint_interval = max(1, min(interval, 100))
    plateau = 0
    converged = False
    finite = True
    start_step = 1
    steps_completed = 0
    if resume_state is not None:
        W = np.asarray(resume_state["W"], dtype=np.float64).copy()
        bias = np.asarray(resume_state["bias"], dtype=np.float64).copy()
        first_W = np.asarray(resume_state["first_W"], dtype=np.float64).copy()
        second_W = np.asarray(resume_state["second_W"], dtype=np.float64).copy()
        first_b = np.asarray(resume_state["first_b"], dtype=np.float64).copy()
        second_b = np.asarray(resume_state["second_b"], dtype=np.float64).copy()
        best_W = np.asarray(resume_state["best_W"], dtype=np.float64).copy()
        best_bias = np.asarray(resume_state["best_bias"], dtype=np.float64).copy()
        best_validation = float(resume_state["best_validation"])
        best_step = int(resume_state["best_step"])
        history = list(resume_state["history"])
        plateau = int(resume_state["plateau"])
        converged = bool(resume_state.get("converged", False))
        finite = bool(resume_state.get("finite", True))
        steps_completed = int(resume_state["step"])
        start_step = steps_completed + 1
        rng.bit_generator.state = resume_state["rng_state"]

    def checkpoint(step: int) -> dict[str, Any]:
        return {
            "step": step, "W": W.copy(), "bias": bias.copy(),
            "first_W": first_W.copy(), "second_W": second_W.copy(),
            "first_b": first_b.copy(), "second_b": second_b.copy(),
            "best_W": best_W.copy(), "best_bias": best_bias.copy(),
            "best_validation": float(best_validation), "best_step": int(best_step),
            "history": list(history), "plateau": int(plateau), "converged": bool(converged),
            "finite": bool(finite), "rng_state": rng.bit_generator.state,
        }

    for step in range(start_step, steps + 1):
        indices = rng.integers(0, train.shape[0], size=min(batch_size, train.shape[0]))
        batch = train[indices]
        z = batch @ W.T
        pre = z @ W + bias
        reconstruction = np.maximum(pre, 0.0)
        error = reconstruction - batch
        d_pre = (2.0 / error.size) * error * (pre > 0.0)
        dW_decoder = z.T @ d_pre
        dz = d_pre @ W.T
        dW_encoder = dz.T @ batch
        dW = dW_decoder + dW_encoder
        dW += (sparsity_penalty / max(1, z.size)) * (np.sign(z).T @ batch)
        dW += weight_decay * W
        db = np.sum(d_pre, axis=0)
        if not (np.isfinite(dW).all() and np.isfinite(db).all()):
            finite = False
            break

        first_W = 0.9 * first_W + 0.1 * dW
        second_W = 0.999 * second_W + 0.001 * dW**2
        first_b = 0.9 * first_b + 0.1 * db
        second_b = 0.999 * second_b + 0.001 * db**2
        correction1, correction2 = 1.0 - 0.9**step, 1.0 - 0.999**step
        W -= learning_rate * (first_W / correction1) / (np.sqrt(second_W / correction2) + 1e-8)
        bias -= learning_rate * (first_b / correction1) / (np.sqrt(second_b / correction2) + 1e-8)
        steps_completed = step

        if step % interval == 0 or step == steps:
            objective, validation_mse = autoencoder_loss(
                validation, W, bias, sparsity_penalty, weight_decay
            )
            train_objective, train_mse = autoencoder_loss(
                train, W, bias, sparsity_penalty, weight_decay
            )
            if not np.isfinite([objective, validation_mse, train_objective, train_mse]).all():
                finite = False
                break
            history.append({
                "step": float(step), "train_objective": train_objective,
                "train_reconstruction_mse": train_mse,
                "validation_objective": objective,
                "validation_reconstruction_mse": validation_mse,
            })
            threshold = max(1e-12, abs(best_validation) * 1e-7)
            if objective < best_validation - threshold:
                best_validation, best_step = objective, step
                best_W, best_bias = W.copy(), bias.copy()
                plateau = 0
            else:
                plateau += interval
            if plateau >= patience:
                converged = True
        if checkpoint_callback is not None and (step % checkpoint_interval == 0 or step == steps):
            checkpoint_callback(checkpoint(step))
        if should_stop is not None and step % checkpoint_interval == 0 and should_stop():
            reconstruction = np.maximum((validation @ best_W.T) @ best_W + best_bias, 0.0)
            return {
                "W": best_W, "bias": best_bias, "converged": False, "finite": finite,
                "paused": True, "best_step": int(best_step),
                "validation_reconstruction_mse": float(np.mean((reconstruction - validation) ** 2)),
                "initial_validation_objective": float(initial_objective),
                "initial_validation_reconstruction_mse": float(initial_mse),
                "history": history, "steps_completed": steps_completed,
            }
        if converged:
            break

    # Keep the best finite validation checkpoint, even when the optimizer hit
    # its step limit. The flag communicates whether the plateau criterion met.
    reconstruction = np.maximum((validation @ best_W.T) @ best_W + best_bias, 0.0)
    validation_mse = float(np.mean((reconstruction - validation) ** 2))
    return {
        "W": best_W,
        "bias": best_bias,
        "converged": bool(converged and finite),
        "finite": bool(finite),
        "paused": False,
        "best_step": int(best_step),
        "validation_reconstruction_mse": validation_mse,
        "initial_validation_objective": float(initial_objective),
        "initial_validation_reconstruction_mse": float(initial_mse),
        "history": history,
        "steps_completed": steps_completed,
    }


def superposition_diagnostics(W: Any) -> dict[str, float]:
    weight = np.asarray(W, dtype=np.float64)
    if weight.ndim != 2:
        raise ValueError("W must be a matrix")
    norms = np.linalg.norm(weight, axis=1)
    valid = norms > np.finfo(float).eps
    if valid.sum() < 2:
        coherence = 0.0
    else:
        normalized = weight[valid] / norms[valid, None]
        gram = np.abs(normalized @ normalized.T)
        np.fill_diagonal(gram, 0.0)
        coherence = float(np.max(gram))
    singular = np.linalg.svd(weight, compute_uv=False)
    stable_rank = float(np.sum(singular**2) / singular[0]**2) if singular.size and singular[0] > 0 else 0.0
    return {
        "encoder_coherence": coherence,
        "encoder_stable_rank": stable_rank,
        "encoder_norm_min": float(np.min(norms)) if norms.size else 0.0,
        "encoder_norm_max": float(np.max(norms)) if norms.size else 0.0,
    }
