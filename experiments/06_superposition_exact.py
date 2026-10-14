#!/usr/bin/env python3
"""Exact binary GAD, correlation tradeoffs, support coverage and interventions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib.pyplot as plt
import numpy as np
from aliasing.superposition import encode, parse_list, population, probabilities, write_csv

STATES = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
TASKS = {"sum": np.array([1., 1.]), "contrast": np.array([1., -1.]),
         "feature": np.array([1., 0.])}


def exact_case(p, q, mode, beta, rule="euclidean"):
    pop = population(1, p, q, mode, "binary")
    probs = probabilities(p, q)
    mean = {key: 0. for key in ("B", "A", "D", "R", "signed_remainder")}
    for i, mass in enumerate(probs):
        if mass == 0:
            continue
        row = pop.fit(STATES[i:i + 1], encode(STATES[i:i + 1], mode), rule).evaluate(beta)
        if not row["interpolates"]:
            raise AssertionError("one sample with an intercept must interpolate")
        for key in mean:
            mean[key] += mass * row[key]
    rho, t, _, joint = probs
    k = rho + joint
    if mode == "shared":
        theory_B = beta.sum() ** 2 * rho * joint / k if k else 0.
        theory_A = beta.sum() ** 2 * rho * joint
    elif mode == "drop":
        theory_B = beta[1] ** 2 * (rho * t / (rho + t) + t * joint / (t + joint))
        theory_A = beta[1] ** 2 * t * k
    else:
        theory_B = theory_A = 0.
    if not np.allclose([mean["B"], mean["A"]], [theory_B, theory_A], atol=1e-10):
        raise AssertionError("exact binary formulas disagree with population projection")
    return dict(p=p, q=q, q_over_p=q / p, mode=mode, rule=rule,
                B_theory=theory_B, A_theory=theory_A, **mean)


def coverage(p, q, sizes):
    probs = probabilities(p, q)
    mass = np.zeros(16)
    mass[0] = 1.
    feasible = {}
    for mode in ("shared", "drop", "uncompressed"):
        pop = population(1, p, q, mode, "binary")
        for task, beta in TASKS.items():
            flags = np.ones(16)
            for mask in range(1, 16):
                ids = [i for i in range(4) if mask & (1 << i) and probs[i] > 0]
                if ids:
                    flags[mask] = pop.fit(STATES[ids], encode(STATES[ids], mode)).evaluate(beta)["interpolates"]
            feasible[mode, task] = flags
    rows = []
    for n in range(1, max(sizes) + 1):
        next_mass = np.zeros(16)
        for mask in range(16):
            for i in range(4):
                next_mass[mask | (1 << i)] += mass[mask] * probs[i]
        mass = next_mass
        if n not in sizes:
            continue
        rho = probs[0]
        theory = (1 - rho) ** n + (1 - q) ** n - (1 - rho - q) ** n
        for (mode, task), flags in feasible.items():
            value = float(mass @ flags)
            if mode == "shared" and task == "sum" and not np.isclose(value, theory, atol=1e-10):
                raise AssertionError("support-coverage law failed")
            rows.append(dict(p=p, q=q, n=n, mode=mode, task=task,
                             interpolation_probability=value,
                             shared_sum_theory=theory))
    return rows


def interventions(p, q):
    """Exactly enumerate one baseline example plus one additional example."""
    probs = probabilities(p, q)
    rows = []
    for mode in ("shared", "drop", "uncompressed"):
        pop = population(1, p, q, mode, "binary")
        for task, beta in TASKS.items():
            for scheme in ("random", "targeted_conflict"):
                for i in range(4):
                    if probs[i] == 0:
                        continue
                    before = pop.fit(STATES[i:i+1], encode(STATES[i:i+1], mode)).evaluate(beta)
                    # On the shared code, 00 and 11 collide. If one was seen,
                    # add its counterpart; otherwise add a joint activation.
                    target = 0 if i == 3 else 3
                    additions = [(j, probs[j]) for j in range(4)] if scheme == "random" else [(target, 1.)]
                    for j, weight in additions:
                        if weight == 0:
                            continue
                        ss = STATES[[i, j]]
                        after = pop.fit(ss, encode(ss, mode)).evaluate(beta)
                        rows.append(dict(p=p, q=q, mode=mode, task=task, scheme=scheme,
                                         baseline_state=i, added_state=j,
                                         probability=probs[i] * weight,
                                         R_before=before["R"], delta_R=after["R"]-before["R"],
                                         **after))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marginals", default="0.05,0.1,0.2,0.2113248654,0.3,0.5")
    parser.add_argument("--cooccurrence-fractions", default="0,0.1,0.2,0.22,0.4,0.6,0.8,1")
    parser.add_argument("--sample-sizes", default="1,2,4,8,16,32,64")
    parser.add_argument("--angles", type=int, default=72)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/superposition_exact"))
    args = parser.parse_args()
    ps = parse_list(args.marginals)
    fractions = parse_list(args.cooccurrence_fractions)
    sizes = parse_list(args.sample_sizes, int)
    if not ps or any(not 0 < p <= .5 for p in ps) or not fractions or any(not 0 <= f <= 1 for f in fractions):
        parser.error("marginals must be in (0,.5], fractions in [0,1]")
    if not sizes or min(sizes) < 1 or args.angles < 2:
        parser.error("sample sizes must be positive and angles at least 2")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    exact, angles, cover, interventions_raw = [], [], [], []
    for p in ps:
        qs = sorted(set([p*p] + [p*f for f in fractions]))
        for q in qs:
            for task, beta in TASKS.items():
                for mode in ("shared", "drop", "uncompressed"):
                    for rule in ("euclidean", "population"):
                        exact.append(dict(task=task, **exact_case(p, q, mode, beta, rule)))
            cover.extend(coverage(p, q, sizes))
        for angle in np.linspace(0, 2*np.pi, args.angles, endpoint=False):
            beta = np.array([np.cos(angle), np.sin(angle)])
            for mode in ("shared", "drop", "uncompressed"):
                angles.append(dict(angle=angle, beta1=beta[0], beta2=beta[1],
                                   **exact_case(p, p*p, mode, beta)))
    for p, q in ((.1, .022), (.2, .04)):
        interventions_raw.extend(interventions(p, q))
    interventions_mean = []
    for key in sorted({(r["p"], r["q"], r["mode"], r["task"], r["scheme"]) for r in interventions_raw}):
        group = [r for r in interventions_raw if (r["p"],r["q"],r["mode"],r["task"],r["scheme"]) == key]
        row = dict(zip(("p","q","mode","task","scheme"), key))
        for metric in ("R", "delta_R", "interpolates", "projected_A", "projected_D"):
            row[metric] = sum(r["probability"]*r[metric] for r in group)
        interventions_mean.append(row)
    for name, rows in (("exact.csv", exact), ("tasks.csv", angles), ("coverage.csv", cover),
                       ("interventions.csv", interventions_raw), ("intervention_summary.csv", interventions_mean)):
        write_csv(out/name, rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for mode in ("shared", "drop"):
        group = sorted([r for r in exact if r["task"] == "sum" and r["mode"] == mode
                        and r["rule"] == "euclidean" and np.isclose(r["q"],r["p"]**2)], key=lambda r:r["p"])
        for ax, metric in zip(axes, ("B", "A", "R")):
            ax.plot([r["p"] for r in group], [r[metric] for r in group], "o-", label=mode)
            ax.set(xlabel="Marginal activity p", ylabel=metric, title="Independent features; sum task")
            ax.axvline((1-1/np.sqrt(3))/2, color="gray", linestyle=":")
            ax.legend()
    fig.tight_layout(); fig.savefig(out/"independent.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for mode in ("shared", "drop"):
        group = sorted([r for r in exact if r["task"] == "sum" and r["mode"] == mode
                        and r["rule"] == "euclidean" and np.isclose(r["p"],.1)], key=lambda r:r["q"])
        if group:
            for ax, metric in zip(axes, ("B", "A")):
                ax.plot([r["q"] for r in group], [r[metric] for r in group], "o-", label=mode)
                ax.set(xlabel="Joint probability q (p=.1 fixed)", ylabel=metric); ax.legend()
    fig.tight_layout(); fig.savefig(out/"correlation.png", dpi=160); plt.close(fig)
    metadata = {**vars(args), "output_dir": str(out), "population": "exact four-state enumeration",
                "B_A_boundary_p": (1-1/np.sqrt(3))/2, "seed": "none: deterministic enumeration",
                "interpretation": "A,D are NaN outside interpolation; projected terms fit projected labels"}
    (out/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"Wrote exact GAD, task, coverage and intervention CSVs and figures to {out}", flush=True)


if __name__ == "__main__":
    main()
