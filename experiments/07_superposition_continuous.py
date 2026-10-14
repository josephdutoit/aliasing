#!/usr/bin/env python3
"""Continuous paired superposition: exact moments, GAD matrices and task transfer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib.pyplot as plt
import numpy as np
from aliasing.superposition import (
    MODES, encode, parse_list, population, sample, summarize, write_csv,
)


def teachers(pairs, rng, directions):
    d = 2*pairs
    tasks = {"sum": np.ones(d)/np.sqrt(d),
             "contrast": np.tile([1., -1.], pairs)/np.sqrt(d),
             "feature": np.eye(d)[0]}
    for j in range(directions):
        beta = rng.normal(size=d)
        tasks[f"random_{j}"] = beta/np.linalg.norm(beta)
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="4")
    parser.add_argument("--marginals", default="0.1,0.3")
    parser.add_argument("--cooccurrence-fractions", default="0,0.1,0.22,0.6")
    parser.add_argument("--sample-sizes", default="1,4,8,16,32")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--head-rules", default="euclidean,population")
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--validation-replicates", type=int, default=30)
    parser.add_argument("--n-directions", type=int, default=4)
    parser.add_argument("--n-test", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/superposition_continuous"))
    args = parser.parse_args()
    widths = parse_list(args.pairs, int)
    ps, fractions = parse_list(args.marginals), parse_list(args.cooccurrence_fractions)
    sizes = sorted(set(parse_list(args.sample_sizes, int)))
    modes, rules = parse_list(args.modes, str), parse_list(args.head_rules, str)
    if not widths or min(widths) < 1 or not sizes or min(sizes) < 1:
        parser.error("pairs and sample sizes must be positive")
    if not ps or any(not 0<p<=.5 for p in ps) or not fractions or any(not 0<=f<=1 for f in fractions):
        parser.error("marginals must be in (0,.5], cooccurrence fractions in [0,1]")
    if not set(("shared", "drop")).issubset(modes) or any(m not in MODES for m in modes):
        parser.error("modes must include shared and drop, and use documented encoder names")
    if not rules or any(r not in ("euclidean", "population") for r in rules):
        parser.error("head rules must be euclidean or population")
    if min(args.replicates, args.validation_replicates, args.n_test) < 2 or args.n_directions < 1:
        parser.error("replicates, validation replicates and n-test must be >=2; directions >=1")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out/"metadata.json").write_text(json.dumps({**vars(args),"output_dir":str(out),
                                               "run_state":"running"},indent=2)+"\n")
    # Distinct streams keep teachers, fitting data, evaluation data and
    # prospective eigenvector validation independent.
    streams = np.random.SeedSequence(args.seed).spawn(4)
    teacher_rng, train_rng, test_rng, val_rng = map(np.random.default_rng, streams)
    raw, paired, pop_rows, spectrum, validation, vector_rows, semantic_rows = [], [], [], [], [], [], []
    matrices = {}
    for pairs in widths:
        tasks = teachers(pairs, teacher_rng, args.n_directions)
        for task, beta in tasks.items():
            vector_rows.append(dict(pairs=pairs, condition="all", task=task,
                                    beta=json.dumps(beta.tolist())))
        for p in ps:
            for fraction in fractions:
                q = p*fraction
                tag = f"m{pairs}_p{p:g}_q{q:g}"
                context = dict(pairs=pairs, p=p, q=q, q_over_p=fraction)
                pops = {mode:population(pairs,p,q,mode) for mode in modes}
                test_s = sample(test_rng,args.n_test,pairs,p,q)
                test_phi = {mode:encode(test_s,mode) for mode in modes}
                test_joint = np.any((test_s[:,::2]>0)&(test_s[:,1::2]>0),axis=1)
                for mode, pop in pops.items():
                    for task, beta in tasks.items():
                        b = beta[::2]+beta[1::2]
                        # E[min(a,b)^2 | joint] = 1/6. Independent pairs.
                        overlap_mean = q/3
                        overlap_second = q/6
                        bound = ((overlap_second-overlap_mean**2)*np.dot(b,b)
                                 + overlap_mean**2*b.sum()**2)
                        pop_rows.append(dict(**context,mode=mode,task=task,
                            population_rank=pop.rank, hidden_width=(2*pairs if mode=="uncompressed" else pairs),
                            B=float(beta@pop.B@beta), target_energy=float(beta@pop.H@beta),
                            overlap_second_moment_bound=float(bound) if mode.startswith("shared") else float("nan")))
                bank = {(n,rule,mode):np.zeros((2*pairs,2*pairs))
                        for n in sizes for rule in rules for mode in ("shared","drop")}
                component_bank = {(n,rule,mode,term):np.zeros((2*pairs,2*pairs))
                        for n in sizes for rule in rules for mode in ("shared","drop")
                        for term in ("projected_A","projected_D")}
                semantic_bank = {(n,rule,mode):np.zeros((2*pairs,4))
                        for n in sizes for rule in rules for mode in modes}
                for replicate in range(args.replicates):
                    train_s = sample(train_rng,max(sizes),pairs,p,q)
                    for n in sizes:
                        current = {}
                        for mode,pop in pops.items():
                            ss = train_s[:n]
                            phi = encode(ss,mode)
                            pair_joint = (ss[:,::2]>0)&(ss[:,1::2]>0)
                            for rule in rules:
                                fit = pop.fit(ss,phi,rule)
                                current[mode,rule] = fit
                                if mode in ("shared","drop"):
                                    bank[n,rule,mode] += fit.R/args.replicates
                                    component_bank[n,rule,mode,"projected_A"] += fit.A/args.replicates
                                    component_bank[n,rule,mode,"projected_D"] += fit.D/args.replicates
                                error_vectors = test_s-test_phi[mode]@fit.head
                                semantic_bank[n,rule,mode][:,0] += np.diag(fit.R)/args.replicates
                                semantic_bank[n,rule,mode][:,1] += np.mean(error_vectors**2,axis=0)/args.replicates
                                if test_joint.any():
                                    semantic_bank[n,rule,mode][:,2] += np.mean(error_vectors[test_joint]**2,axis=0)/args.replicates
                                else:
                                    semantic_bank[n,rule,mode][:,2] = np.nan
                                semantic_bank[n,rule,mode][:,3] += np.array([
                                    fit.evaluate(np.eye(2*pairs)[j])["interpolates"] for j in range(2*pairs)])/args.replicates
                                # Direct held-out predictions of all semantic targets.
                                for task,beta in tasks.items():
                                    values = fit.evaluate(beta)
                                    heldout_loss = (error_vectors@beta)**2
                                    row = dict(**context,replicate=replicate,n=n,mode=mode,rule=rule,
                                               task=task,target_energy=float(beta@pop.H@beta),
                                               heldout_R=float(heldout_loss.mean()),
                                               heldout_R_se=float(heldout_loss.std(ddof=1)/np.sqrt(args.n_test)),
                                               heldout_joint_R=float(heldout_loss[test_joint].mean()) if test_joint.any() else float("nan"),
                                               heldout_no_joint_R=float(heldout_loss[~test_joint].mean()) if (~test_joint).any() else float("nan"),
                                               cooccurrence_pairs_seen=int(np.any(pair_joint,axis=0).sum()),
                                               train_joint_rate=float(np.any(pair_joint,axis=1).mean()),
                                               **values)
                                    raw.append(row)
                        for rule in rules:
                            for task,beta in tasks.items():
                                a=current["shared",rule].evaluate(beta)
                                b=current["drop",rule].evaluate(beta)
                                paired.append(dict(**context,replicate=replicate,n=n,rule=rule,task=task,
                                                   delta_B=a["B"]-b["B"],delta_A=a["A"]-b["A"],
                                                   delta_D=a["D"]-b["D"],delta_R=a["R"]-b["R"],
                                                   both_interpolate=a["interpolates"]*b["interpolates"]))
                    if (replicate+1)%max(1,args.replicates//5)==0:
                        print(f"{tag}: fitting {replicate+1}/{args.replicates}",flush=True)
                vectors = {}
                for (n,rule,mode), values in semantic_bank.items():
                    for j in range(2*pairs):
                        semantic_rows.append(dict(**context,n=n,rule=rule,mode=mode,semantic_feature=j,
                            B=float(pops[mode].B[j,j]),R=float(values[j,0]),heldout_R=float(values[j,1]),
                            heldout_joint_R=float(values[j,2]),interpolation_fraction=float(values[j,3])))
                for n in sizes:
                    for rule in rules:
                        difference = bank[n,rule,"shared"]-bank[n,rule,"drop"]
                        eig, vec = np.linalg.eigh((difference+difference.T)/2)
                        matrices[f"{tag}_n{n}_{rule}_delta_R"] = difference
                        for mode in ("shared","drop"):
                            matrices[f"{tag}_n{n}_{rule}_{mode}_R"] = bank[n,rule,mode]
                            matrices[f"{tag}_n{n}_{rule}_{mode}_B"] = pops[mode].B
                            for term in ("projected_A","projected_D"):
                                matrices[f"{tag}_n{n}_{rule}_{mode}_{term}"] = component_bank[n,rule,mode,term]
                        for j,value in enumerate(eig):
                            spectrum.append(dict(**context,n=n,rule=rule,eigen_index=j,
                                                 eigenvalue=float(value),calibration_replicates=args.replicates))
                        for label,j in (("lowest",0),("highest",len(eig)-1)):
                            beta=vec[:,j]
                            vectors[n,rule,label]=(beta,float(eig[j]))
                            vector_rows.append(dict(pairs=pairs,condition=tag,task=f"n{n}_{rule}_{label}",
                                                    beta=json.dumps(beta.tolist())))
                # New tasks selected by the calibration matrices are tested on
                # NEW training datasets. No validation draw enters task selection.
                val_raw=[]
                for replicate in range(args.validation_replicates):
                    train_s=sample(val_rng,max(sizes),pairs,p,q)
                    for n in sizes:
                        for rule in rules:
                            fits={mode:pops[mode].fit(train_s[:n],encode(train_s[:n],mode),rule)
                                  for mode in ("shared","drop")}
                            for label in ("lowest","highest"):
                                beta,predicted=vectors[n,rule,label]
                                shared=fits["shared"].evaluate(beta)
                                dropped=fits["drop"].evaluate(beta)
                                val_raw.append(dict(**context,n=n,rule=rule,direction=label,
                                    replicate=replicate,predicted_delta_R=predicted,
                                    delta_R=shared["R"]-dropped["R"],
                                    both_interpolate=shared["interpolates"]*dropped["interpolates"]))
                validation.extend(summarize(val_raw,("pairs","p","q","q_over_p","n","rule","direction"),
                                            ("predicted_delta_R","delta_R","both_interpolate")))
                # Save each completed condition, including raw attempts. An
                # interrupted long run retains completed work and explicit NaNs.
                write_csv(out/"replicates.csv",raw)
                write_csv(out/"paired.csv",paired)
                write_csv(out/"task_validation.csv",validation)
                print(f"Completed {tag}; CSV checkpoint written",flush=True)
    summary = summarize(raw,("pairs","p","q","q_over_p","n","mode","rule","task"),
                        ("B","A","D","R","interpolates","heldout_R","train_residual_norm",
                         "heldout_joint_R","heldout_no_joint_R","cooccurrence_pairs_seen","train_joint_rate",
                         "projected_A","projected_D","signed_remainder","min_nonzero_S_eigenvalue"))
    paired_summary=summarize(paired,("pairs","p","q","q_over_p","n","rule","task"),
                             ("delta_B","delta_A","delta_D","delta_R","both_interpolate"))
    for name,rows in (("summary.csv",summary),("paired_summary.csv",paired_summary),
                      ("population.csv",pop_rows),("task_spectrum.csv",spectrum),("task_vectors.csv",vector_rows),
                      ("semantic_errors.csv",semantic_rows)):
        write_csv(out/name,rows)
    np.savez_compressed(out/"risk_matrices.npz",**matrices)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for task in ("sum","contrast","feature"):
        group=sorted([r for r in paired_summary if r["pairs"]==widths[0] and r["p"]==ps[0]
                      and r["n"]==sizes[0] and r["rule"]==rules[0] and r["task"]==task],key=lambda r:r["q"])
        for ax,metric in zip(axes,("delta_B","delta_A","delta_R")):
            ax.plot([r["q"] for r in group],[r[f"{metric}_mean"] for r in group],"o-",label=task)
            ax.axhline(0,color="gray",linestyle=":")
            ax.set(xlabel="Joint probability q; marginals fixed",ylabel=metric+" (sharing - dropping)")
            ax.legend()
    fig.tight_layout();fig.savefig(out/"continuous_tradeoffs.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5))
    x=np.array([r["predicted_delta_R_mean"] for r in validation])
    y=np.array([r["delta_R_mean"] for r in validation])
    se=np.array([r["delta_R_se"] for r in validation])
    ax.errorbar(x,y,yerr=se,fmt=".",alpha=.6)
    lower,upper=min(x.min(),y.min()),max(x.max(),y.max())
    ax.plot([lower,upper],[lower,upper],"k:")
    ax.set_xscale("symlog",linthresh=.1)
    ax.set_yscale("symlog",linthresh=.1)
    ax.set(xlabel="Calibration matrix prediction",ylabel="New-dataset mean risk difference",
           title="Prospective tasks (symlog axes; error bars: 1 SE)")
    fig.tight_layout();fig.savefig(out/"task_validation.png",dpi=160);plt.close(fig)
    metadata={**vars(args),"output_dir":str(out),"run_state":"completed","amplitudes":"independent uniform [0,1] on active coordinates",
              "population_moments":"exact polynomial quadrature; independent pairs",
              "paired_comparison":"same semantic training and test samples across encoders and heads",
              "validation":"independent teacher, calibration, test and validation RNG streams",
              "head_rules":{"euclidean":"min Euclidean norm LS, gradient descent from zero",
                            "population":"min population-G norm LS; controlled known-metric diagnostic"},
              "outside_interpolation":"A,D are NaN; projected_A,D describe projected-label fitting",
              "finite_sample_warning":"All attempts retained. SEs are descriptive; pseudoinverse risk can have heavy tails."}
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(f"Finished. Results in {out}",flush=True)


if __name__=="__main__":
    main()
