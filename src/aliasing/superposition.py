"""Frozen superposition benchmarks with exact population moments.

All heads use squared loss. Singular population coordinates are quotiented out
only for the population projection; Euclidean heads retain their original metric.
Population moments for independent pairs and uniform [0,1] amplitudes are
integrated exactly by polynomial Gaussian quadrature on two triangles.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODES = ("shared", "shared_scaled", "drop", "linear", "positive", "uncompressed")


def encode(s: np.ndarray, mode: str) -> np.ndarray:
    left, right = s[:, ::2], s[:, 1::2]
    if mode in ("shared", "shared_scaled"):
        z = left - right
        u = np.stack((np.maximum(z, 0), np.maximum(-z, 0)), axis=-1)
        if mode == "shared_scaled":
            u = u * np.array([10., .1])
        u = u.reshape(len(s), -1)
    elif mode == "drop":
        u = left
    elif mode == "linear":
        u = left - right
    elif mode == "positive":
        u = left + right
    elif mode == "uncompressed":
        u = s
    else:
        raise ValueError(f"unknown encoder: {mode}")
    return np.column_stack((np.ones(len(s)), u))


def probabilities(p: float, q: float) -> np.ndarray:
    if not 0 < p < 1 or not max(0.0, 2 * p - 1) <= q <= p:
        raise ValueError("require 0<p<1 and max(0,2p-1)<=q<=p")
    return np.array([1 - 2 * p + q, p - q, p - q, q])


def sample(rng: np.random.Generator, n: int, pairs: int, p: float,
           q: float, amplitudes: str = "uniform") -> np.ndarray:
    states = rng.choice(4, size=(n, pairs), p=probabilities(p, q))
    supports = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])[states]
    if amplitudes == "uniform":
        supports = supports * rng.uniform(0, 1, size=supports.shape)
    elif amplitudes != "binary":
        raise ValueError("amplitudes must be uniform or binary")
    return supports.reshape(n, 2 * pairs).astype(float)


def pair_quadrature(p: float, q: float, amplitudes: str):
    weights = probabilities(p, q)
    if amplitudes == "binary":
        return np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float), weights
    if amplitudes != "uniform":
        raise ValueError("amplitudes must be uniform or binary")
    nodes, w = np.polynomial.legendre.leggauss(4)
    nodes, w = (nodes + 1) / 2, w / 2
    s, mass = [[0., 0.]], [weights[0]]
    for a, wa in zip(nodes, w):
        s.extend([[a, 0], [0, a]])
        mass.extend([weights[1] * wa, weights[2] * wa])
        for b, wb in zip(nodes, w):
            # (a,a*b) and its reflection cover the unit square. Jacobian a.
            s.extend([[a, a * b], [a * b, a]])
            mass.extend([q * wa * wb * a, q * wa * wb * a])
    return np.array(s), np.array(mass)


def pinv(matrix: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    return np.linalg.pinv(matrix, rcond=rcond)


@dataclass
class Population:
    G: np.ndarray
    cross: np.ndarray  # E[s phi^T]
    H: np.ndarray      # E[s s^T]
    rcond: float = 1e-12

    def __post_init__(self):
        values, vectors = np.linalg.eigh(self.G)
        keep = values > self.rcond * max(float(values[-1]), 1e-30)
        self.rank = int(keep.sum())
        self.root = vectors[:, keep] / np.sqrt(values[keep])
        self.L = self.cross @ self.root @ self.root.T
        bias = self.H - self.L @ self.cross.T
        self.B = (bias + bias.T) / 2

    def fit(self, s: np.ndarray, phi: np.ndarray, rule: str = "euclidean"):
        white = phi @ self.root
        pw = pinv(white, self.rcond)
        residual = s - phi @ self.L.T
        displacement = pw @ residual
        circ = self.L.T + self.root @ displacement
        if rule == "euclidean":
            head = pinv(phi, self.rcond) @ s
        elif rule == "population":
            head = self.root @ pw @ s
        else:
            raise ValueError(f"unknown head rule: {rule}")
        delta = head - circ
        A = displacement.T @ displacement
        D = delta.T @ self.G @ delta
        R = (self.H - self.cross @ head - head.T @ self.cross.T
             + head.T @ self.G @ head)
        R = (R + R.T) / 2
        singular = np.linalg.svd(white, compute_uv=False)
        rank = int(np.sum(singular > self.rcond * singular[0]))
        eig_min = float(singular[rank - 1] ** 2) if rank else 0.
        return Fit(self, head, A, D, R, phi @ head - s, rank, eig_min,
                   float(singular[0] ** 2))


@dataclass
class Fit:
    population: Population
    head: np.ndarray
    A: np.ndarray
    D: np.ndarray
    R: np.ndarray
    training_error: np.ndarray
    rank: int
    min_nonzero_eigenvalue: float
    max_eigenvalue: float

    def evaluate(self, beta: np.ndarray) -> dict[str, float | int]:
        B = float(beta @ self.population.B @ beta)
        A, D, R = (float(beta @ matrix @ beta) for matrix in
                   (self.A, self.D, self.R))
        error = float(np.linalg.norm(self.training_error @ beta))
        # Absolute+relative tolerance in original target units.
        scale = np.sqrt(len(self.training_error) * max(
            float(beta @ self.population.H @ beta), 1e-30))
        feasible = error <= 1e-9 * (1 + scale)
        return {
            "B": B, "A": A if feasible else float("nan"),
            "D": D if feasible else float("nan"), "R": R,
            "interpolates": int(feasible), "train_residual_norm": error,
            # These projected-label quantities remain diagnostic outside
            # interpolation, but are NOT the original label-interpolation GAD.
            "projected_A": A, "projected_D": D,
            "signed_remainder": R - B - A - D,
            "train_rank": self.rank, "population_rank": self.population.rank,
            "min_nonzero_S_eigenvalue": self.min_nonzero_eigenvalue,
            "max_S_eigenvalue": self.max_eigenvalue,
        }


def population(pairs: int, p: float, q: float, mode: str,
               amplitudes: str = "uniform") -> Population:
    s, weights = pair_quadrature(p, q, amplitudes)
    u = encode(s, mode)[:, 1:]
    sm, um = weights @ s, weights @ u
    ss = s.T @ (weights[:, None] * s)
    uu = u.T @ (weights[:, None] * u)
    su = s.T @ (weights[:, None] * u)
    d, k = 2 * pairs, u.shape[1] * pairs
    sms, ums = np.tile(sm, pairs), np.tile(um, pairs)
    H, block = np.outer(sms, sms), np.outer(ums, ums)
    cross = np.column_stack((sms, np.outer(sms, ums)))
    for j in range(pairs):
        si = slice(2 * j, 2 * j + 2)
        ui = slice(u.shape[1] * j, u.shape[1] * (j + 1))
        H[si, si] = ss
        block[ui, ui] = uu
        cross[si, slice(1 + ui.start, 1 + ui.stop)] = su
    G = np.empty((k + 1, k + 1))
    G[0, 0], G[0, 1:], G[1:, 0], G[1:, 1:] = 1., ums, ums, block
    assert cross.shape == (d, k + 1)
    return Population(G, cross, H)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_list(text: str, cast=float) -> list:
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


def summarize(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    result = []
    for group, values in groups.items():
        out = dict(zip(keys, group))
        out["attempts"] = len(values)
        for metric in metrics:
            numbers = np.array([row[metric] for row in values], float)
            numbers = numbers[np.isfinite(numbers)]
            out[f"{metric}_count"] = len(numbers)
            out[f"{metric}_mean"] = float(numbers.mean()) if len(numbers) else float("nan")
            out[f"{metric}_se"] = (float(numbers.std(ddof=1) / np.sqrt(len(numbers)))
                                  if len(numbers) > 1 else float("nan"))
            if metric in ("R", "delta_R"):
                out[f"{metric}_median"] = float(np.median(numbers)) if len(numbers) else float("nan")
                out[f"{metric}_q95"] = float(np.quantile(numbers, .95)) if len(numbers) else float("nan")
        result.append(out)
    return result
