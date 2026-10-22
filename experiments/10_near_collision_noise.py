"""Test how small representation separation and observation noise interact.

Two latent states have different targets and hidden means separated by delta.
An isotropic scalar observation-noise channel turns exact collisions into
near-collisions. The script compares the exact Bayes risk (Gauss-Hermite
quadrature) with a finite-sample linear readout and its GAD components.
"""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aliasing.static_geometry import finite_support_gad
from aliasing.summaries import aggregate_seed_rows


def floats(value: str) -> list[float]:
    try: values = [float(x) for x in value.split(",") if x]
    except ValueError as exc: raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not values: raise argparse.ArgumentTypeError("at least one value is required")
    return values


def ints(value: str) -> list[int]:
    try: values = [int(x) for x in value.split(",") if x]
    except ValueError as exc: raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or min(values) < 1: raise argparse.ArgumentTypeError("values must be positive")
    return values


def bayes_risk(delta: float, noise: float, prior: float, target_gap: float, order: int) -> float:
    """Exact E Var(Y|X) for two Gaussian observation channels."""
    if noise == 0:
        return prior * (1-prior) * target_gap**2 if delta == 0 else 0.0
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    x_plus = delta/2 + np.sqrt(2)*noise*nodes
    x_minus = -delta/2 + np.sqrt(2)*noise*nodes
    # Posterior probability of the positive latent state, evaluated stably.
    log_odds_plus = np.log(prior/(1-prior)) + delta*x_plus/(noise**2)
    log_odds_minus = np.log(prior/(1-prior)) + delta*x_minus/(noise**2)
    post_plus = 1/(1+np.exp(-np.clip(log_odds_plus, -700, 700)))
    post_minus = 1/(1+np.exp(-np.clip(log_odds_minus, -700, 700)))
    conditional_var_plus = target_gap**2 * post_plus*(1-post_plus)
    conditional_var_minus = target_gap**2 * post_minus*(1-post_minus)
    expectation_plus = np.dot(weights, conditional_var_plus)/np.sqrt(np.pi)
    expectation_minus = np.dot(weights, conditional_var_minus)/np.sqrt(np.pi)
    return float(prior*expectation_plus + (1-prior)*expectation_minus)


def run_one(args: argparse.Namespace, seed: int, delta: float, noise: float,
            prior: float, n: int) -> dict[str, float]:
    rng = np.random.default_rng(args.seed + seed*1000003 + int(delta*1e6) + int(noise*1e5))
    states = rng.random(n) < prior
    signs = np.where(states, 1.0, -1.0)
    x_train = signs*delta/2 + rng.normal(scale=noise, size=n)
    y_train = signs*args.target_gap/2
    test_states = rng.random(args.n_test) < prior
    test_signs = np.where(test_states, 1.0, -1.0)
    x_test = test_signs*delta/2 + rng.normal(scale=noise, size=args.n_test)
    y_test = test_signs*args.target_gap/2
    # The constant coordinate allows the optimal affine linear readout to
    # represent non-balanced class priors without forcing a zero intercept.
    h_train = np.column_stack((np.ones(n), x_train))
    h_test = np.column_stack((np.ones(args.n_test), x_test))
    gad = finite_support_gad(
        h_train, y_train, h_test, y_test, np.full(args.n_test, 1/args.n_test)
    )
    return {
        "seed": float(seed), "separation": float(delta), "observation_noise": float(noise),
        "prior_positive": float(prior), "target_gap": float(args.target_gap),
        "sample_size": float(n), "separation_to_noise": float(delta/noise) if noise else float("inf"),
        "exact_bayes_risk": bayes_risk(delta, noise, prior, args.target_gap, args.quadrature_order),
        **gad,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows: raise ValueError("no results")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--separations",type=floats,default=floats("0,0.01,0.03,0.1,0.3,1,2"))
    ap.add_argument("--noise-levels",type=floats,default=floats("0.01,0.03,0.1,0.3,1"))
    ap.add_argument("--sample-sizes",type=ints,default=ints("4,8,16,32,64,128,256"))
    ap.add_argument("--priors",type=floats,default=floats("0.2,0.5,0.8"))
    ap.add_argument("--target-gap",type=float,default=1.0)
    ap.add_argument("--num-seeds",type=int,default=100)
    ap.add_argument("--n-test",type=int,default=20000)
    ap.add_argument("--quadrature-order",type=int,default=100)
    ap.add_argument("--seed",type=int,default=20261023)
    ap.add_argument("--output-dir",type=Path,required=True)
    a=ap.parse_args()
    if a.num_seeds<2 or a.n_test<100 or a.target_gap<=0 or a.quadrature_order<10:
        raise ValueError("require >=2 seeds, >=100 test points, positive gap, quadrature order >=10")
    if any(d<0 for d in a.separations) or any(s<0 for s in a.noise_levels):
        raise ValueError("separations and noise levels must be nonnegative")
    if any(not 0<p<1 for p in a.priors): raise ValueError("priors must lie in (0,1)")
    rows=[]
    for delta, noise, prior, n, seed in product(a.separations,a.noise_levels,a.priors,a.sample_sizes,range(a.num_seeds)):
        rows.append(run_one(a,seed,delta,noise,prior,n))
    keys=("separation","observation_noise","prior_positive","sample_size")
    means,stds,cis=aggregate_seed_rows(rows,group_keys=keys,excluded_keys={"seed",*keys},n_bootstrap=2000)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    write_csv(a.output_dir/"raw.csv",rows);write_csv(a.output_dir/"mean.csv",means)
    write_csv(a.output_dir/"std.csv",stds);write_csv(a.output_dir/"ci95.csv",cis)
    write_csv(a.output_dir/"config.csv",[{
        "separations":",".join(map(str,a.separations)),"noise_levels":",".join(map(str,a.noise_levels)),
        "sample_sizes":",".join(map(str,a.sample_sizes)),"priors":",".join(map(str,a.priors)),
        "target_gap":a.target_gap,"num_seeds":a.num_seeds,"n_test":a.n_test,
        "quadrature_order":a.quadrature_order,"seed":a.seed,
    }])
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    chosen_prior=min(a.priors,key=lambda x:abs(x-0.5))
    for noise in a.noise_levels:
        sel=[r for r in means if float(r["observation_noise"])==noise and float(r["prior_positive"])==chosen_prior
             and int(float(r["sample_size"]))==max(a.sample_sizes)]
        sel.sort(key=lambda r:float(r["separation"]))
        label=f"noise={noise:g}"
        axes[0].plot([r["separation"] for r in sel],[r["exact_bayes_risk"] for r in sel],marker="o",label=label)
        axes[1].plot([r["separation"] for r in sel],[r["test_mse"] for r in sel],marker="o",label=label)
        axes[2].plot([r["separation"] for r in sel],[r["aliasing"] for r in sel],marker="o",label=label)
    for ax,label in zip(axes,("exact Bayes risk","finite-sample linear readout risk","GAD alias component")):
        ax.set_xlabel("hidden-state separation");ax.set_ylabel(label);ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.savefig(a.output_dir/"near_collision_noise.png",dpi=170,bbox_inches="tight");plt.close(fig)
    print(f"Wrote {len(rows)} rows to {a.output_dir}")


if __name__=="__main__":main()
