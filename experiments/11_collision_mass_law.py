"""Exact and near-collision controls with a known Bayes-risk law.

Two target-distinct states have hidden representations separated by epsilon;
both are mixed with a zero state. At epsilon=0, the exact Bayes risk for the
contrast task is c*rho*(1-rho)*gap^2. A common-target control has zero
collision risk. This tests the representation floor separately from finite
sample GAD aliasing.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aliasing.static_geometry import finite_support_gad
from aliasing.summaries import aggregate_seed_rows
from aliasing.superposition import representation_bayes_mse


def floats(value: str) -> list[float]:
    try: out=[float(x) for x in value.split(",") if x]
    except ValueError as exc: raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not out: raise argparse.ArgumentTypeError("at least one value is required")
    return out


def ints(value: str) -> list[int]:
    try: out=[int(x) for x in value.split(",") if x]
    except ValueError as exc: raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not out or min(out)<1: raise argparse.ArgumentTypeError("values must be positive")
    return out


def one(args: argparse.Namespace, seed: int, mass: float, rho: float,
        epsilon: float, task: str, n: int) -> dict[str,float|str]:
    # v and u are orthonormal. epsilon therefore is the Euclidean distance
    # between the two nonzero represented states.
    h=np.array([[1.0,0.0],[1.0,epsilon],[0.0,0.0]])
    if task=="contrast": y=np.array([args.target_gap/2,-args.target_gap/2,0.0])
    elif task=="common": y=np.array([args.target_gap/2,args.target_gap/2,0.0])
    else: raise ValueError(task)
    probabilities=np.array([mass*rho,mass*(1-rho),1-mass],dtype=float)
    # This experiment tests exact collisions. Near-collisions stay distinct;
    # their conditioning is visible in the readout metrics rather than being
    # silently merged by a numerical distance threshold.
    bayes,aliased_mass=representation_bayes_mse(h,y,probabilities,atol=0.0)
    # Known population-optimal affine head: at a collision it predicts the
    # conditional target mean; at positive separation both tasks are exact.
    if epsilon == 0.0:
        population_head=np.array([
            args.target_gap*(rho-0.5) if task=="contrast" else args.target_gap/2,
            0.0,
        ])
    elif task=="contrast":
        population_head=np.array([args.target_gap/2,-args.target_gap/epsilon])
    else:
        population_head=np.array([args.target_gap/2,0.0])
    rng=np.random.default_rng(args.seed+seed*1000003+int(mass*1e6)+int(rho*1e5)+int(epsilon*1e4))
    idx=rng.choice(3,size=n,replace=True,p=probabilities)
    gad=finite_support_gad(h[idx],y[idx],h,y,probabilities,population_head=population_head)
    formula=mass*rho*(1-rho)*args.target_gap**2 if epsilon==0.0 and task=="contrast" else 0.0
    exact_alias_control=(epsilon==0.0 and task=="contrast" and abs(rho-0.5)<1e-14)
    if exact_alias_control and gad["aliasing"] > args.control_tolerance:
        raise RuntimeError(
            "balanced exact-collision control should have zero GAD alias component; "
            f"got {gad['aliasing']:.6g} (seed={seed}, n={n})."
        )
    return {
        "seed":float(seed),"collision_mass":mass,"conditional_positive_mass":rho,
        "target_task":task,"target_gap":args.target_gap,"separation":epsilon,
        "sample_size":float(n),"formula_bayes_risk":formula,
        "formula_error":abs(bayes-formula),"representation_bayes_mse":bayes,
        "aliased_probability_mass":aliased_mass,
        "population_head_first":float(population_head[0]),
        "population_head_second":float(population_head[1]),
        "population_head_risk":float(np.dot(probabilities,(h@population_head-y)**2)),
        "within_collision_tolerance":float(epsilon<=args.collision_tolerance),
        "balanced_exact_alias_control":float(exact_alias_control),**gad,
    }


def write_csv(path:Path,rows:list[dict[str,object]])->None:
    if not rows:raise ValueError("no results")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main()->None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collision-masses",type=floats,default=floats("0.1,0.25,0.5,0.75,0.9"))
    ap.add_argument("--conditional-masses",type=floats,default=floats("0.1,0.25,0.5,0.75,0.9"))
    ap.add_argument("--separations",type=floats,default=floats("0,1e-8,1e-5,1e-3,0.01,0.1,0.5"))
    ap.add_argument("--tasks",default="contrast,common")
    ap.add_argument("--sample-sizes",type=ints,default=ints("4,8,16,32,64,128"))
    ap.add_argument("--target-gap",type=float,default=1.0)
    ap.add_argument("--num-seeds",type=int,default=100)
    ap.add_argument("--collision-tolerance",type=float,default=1e-10)
    ap.add_argument("--control-tolerance",type=float,default=1e-10)
    ap.add_argument("--seed",type=int,default=20261023)
    ap.add_argument("--output-dir",type=Path,required=True)
    a=ap.parse_args()
    if a.num_seeds<2 or a.target_gap<=0 or a.control_tolerance<=0:raise ValueError("need >=2 seeds, positive target gap, and positive control tolerance")
    if any(not 0<=c<=1 for c in a.collision_masses):raise ValueError("collision masses must be in [0,1]")
    if any(not 0<r<1 for r in a.conditional_masses):raise ValueError("conditional masses must be in (0,1)")
    if any(e<0 for e in a.separations):raise ValueError("separations must be nonnegative")
    rows=[]
    for c,rho,eps,task,n,seed in product(a.collision_masses,a.conditional_masses,a.separations,
                                         a.tasks.split(","),a.sample_sizes,range(a.num_seeds)):
        rows.append(one(a,seed,c,rho,eps,task,n))
    max_formula_error=max(float(r["formula_error"]) for r in rows)
    max_reconstruction_error=max(float(r["decomposition_reconstruction_error"]) for r in rows)
    max_identity_error=max(float(r["risk_identity_error"]) for r in rows)
    if max_formula_error > a.control_tolerance:
        raise RuntimeError(f"Bayes-risk formula check failed: max error={max_formula_error:.6g}")
    keys=("collision_mass","conditional_positive_mass","separation","target_task","sample_size")
    # Convert categorical task labels to numeric IDs for the shared aggregator.
    task_ids={name:i for i,name in enumerate(a.tasks.split(","))}
    for row in rows:row["task_id"]=float(task_ids[str(row["target_task"])])
    numeric=("collision_mass","conditional_positive_mass","separation","task_id","sample_size")
    means,stds,cis=aggregate_seed_rows(rows,group_keys=numeric,
        excluded_keys={"seed","target_task",*numeric},n_bootstrap=2000)
    for group in (means,stds,cis):
        for row in group:row["target_task"]=next(k for k,v in task_ids.items() if v==int(row["task_id"]))
    a.output_dir.mkdir(parents=True,exist_ok=True)
    write_csv(a.output_dir/"raw.csv",rows);write_csv(a.output_dir/"mean.csv",means)
    write_csv(a.output_dir/"std.csv",stds);write_csv(a.output_dir/"ci95.csv",cis)
    with (a.output_dir/"run_manifest.json").open("w") as f:
        json.dump({"experiment":"11_collision_mass_law","args":vars(a)|{"output_dir":str(a.output_dir)},
                   "rows":len(rows),"max_formula_error":max_formula_error,
                   "max_decomposition_reconstruction_error":max_reconstruction_error,
                   "max_risk_identity_error":max_identity_error,"checks_passed":True},
                  f,indent=2,default=str)
    write_csv(a.output_dir/"config.csv",[{
        "collision_masses":",".join(map(str,a.collision_masses)),
        "conditional_masses":",".join(map(str,a.conditional_masses)),
        "separations":",".join(map(str,a.separations)),"tasks":a.tasks,
        "sample_sizes":",".join(map(str,a.sample_sizes)),"target_gap":a.target_gap,
        "num_seeds":a.num_seeds,"collision_tolerance":a.collision_tolerance,"seed":a.seed,
    }])
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    rho=min(a.conditional_masses,key=lambda v:abs(v-.5));mass=a.collision_masses[len(a.collision_masses)//2]
    ci_lookup={(r["target_task"],float(r["collision_mass"]),float(r["conditional_positive_mass"]),
                float(r["separation"]),float(r["sample_size"])):r for r in cis}
    for task in a.tasks.split(","):
        sel=[r for r in means if r["target_task"]==task and float(r["collision_mass"])==mass
             and float(r["conditional_positive_mass"])==rho and int(float(r["sample_size"]))==max(a.sample_sizes)]
        sel.sort(key=lambda r:float(r["separation"]))
        for ax,key,label in zip(axes,("representation_bayes_mse","test_mse","aliasing"),
                                ("representation Bayes risk","linear-readout risk","GAD alias component")):
            x=np.array([float(r["separation"]) for r in sel])
            yvals=np.array([float(r[key]) for r in sel])
            lows=[];highs=[]
            for r in sel:
                ci=ci_lookup[(task,mass,rho,float(r["separation"]),float(max(a.sample_sizes)))]
                lows.append(float(ci[f"{key}_ci95_low"]));highs.append(float(ci[f"{key}_ci95_high"]))
            ax.plot(x,yvals,marker="o",label=task)
            if key!="representation_bayes_mse": ax.fill_between(x,lows,highs,alpha=.18)
            if key=="representation_bayes_mse" and task=="contrast":
                formula=[mass*rho*(1-rho)*a.target_gap**2 if e==0.0 else 0.0 for e in x]
                ax.plot(x,formula,"k--",alpha=.7,label="exact formula")
            ax.set_xlabel("separation between hidden states");ax.set_ylabel(label);ax.grid(alpha=.25)
    axes[0].legend()
    fig.suptitle("Exact collision law; bands are pointwise 95% bootstrap CIs across seeds")
    fig.savefig(a.output_dir/"collision_mass_law.png",dpi=170,bbox_inches="tight");plt.close(fig)
    print(f"Wrote {len(rows)} rows to {a.output_dir}; max formula error={max_formula_error:.3g}; "
          f"max risk identity error={max_identity_error:.3g}")


if __name__=="__main__":main()
