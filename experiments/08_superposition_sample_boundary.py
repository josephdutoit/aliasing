#!/usr/bin/env python3
"""Exact sample-size preference reversal for fixed binary superposition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import matplotlib.pyplot as plt
import numpy as np
from aliasing.finite_sample import (
    P_SMALL,P_POPULATION,binary_cells,expected_risk,interpolation_probability,sufficient_bound,
)
from aliasing.superposition import parse_list,write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marginals",default="0.12,0.16,0.18,0.2,0.21,0.24,0.3")
    parser.add_argument("--max-n",type=int,default=64)
    parser.add_argument("--output-dir",type=Path,default=Path("outputs/superposition_sample_boundary"))
    args=parser.parse_args()
    ps=parse_list(args.marginals)
    if args.max_n<1 or not ps or any(not 0<p<=.5 for p in ps):
        parser.error("max-n must be positive; marginals must be in (0,.5]")
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    rows,gaps=[],[]
    for p in ps:
        q=p*p
        for n in range(1,args.max_n+1):
            risks={}
            for mode in ("shared","drop"):
                result=expected_risk(n,*binary_cells(p,q,mode))
                result.update(p=p,q=q,n=n,mode=mode,
                    interpolation_probability=interpolation_probability(n,p,q,mode),
                    A_original=result["sampling_excess"] if n==1 else float("nan"),
                    D_original=result["unobserved_excess"] if n==1 else float("nan"))
                rows.append(result);risks[mode]=result
            bias_gap=risks["drop"]["B"]-risks["shared"]["B"]
            bound=sufficient_bound(n,p)
            gaps.append(dict(p=p,q=q,n=n,delta_R=risks["shared"]["R"]-risks["drop"]["R"],
                             population_advantage=bias_gap,shared_excess_upper_bound=bound,
                             sharing_certified_better=int(bound<bias_gap)))
        print(f"Exact curves completed for p={p:g}",flush=True)
    write_csv(out/"curves.csv",rows);write_csv(out/"boundaries.csv",gaps)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    chosen=min(ps,key=lambda p:abs(p-.2))
    for mode in ("shared","drop"):
        group=[r for r in rows if r["p"]==chosen and r["mode"]==mode]
        axes[0].plot([r["n"] for r in group],[r["R"] for r in group],label=mode)
        axes[0].axhline(group[0]["B"],linestyle=":",alpha=.5)
    axes[0].set(xlabel="Training size n",ylabel="Exact expected population risk",
                title=f"Fixed encoders and head rule; p={chosen:g}");axes[0].legend()
    for p in ps:
        group=[r for r in gaps if r["p"]==p]
        axes[1].plot([r["n"] for r in group],[r["delta_R"] for r in group],label=f"p={p:g}")
    axes[1].axhline(0,color="gray",linestyle=":")
    axes[1].set(xlabel="Training size n",ylabel="Risk difference: sharing - dropping")
    axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/"sample_boundary.png",dpi=160);plt.close(fig)
    metadata={**vars(args),"output_dir":str(out),"calculation":"exact finite occupancy sums, floating-point evaluation",
        "target":"s1+s2; independent Bernoulli(p) semantic features",
        "fitting_rule":"minimum Euclidean-norm least squares in the same coordinates as study 06",
        "one_sample_boundary":P_SMALL,"population_boundary":P_POPULATION,
        "interpretation":"Original A,D apply at n=1. Larger-n sampling/unobserved terms are LS extensions; feasibility is reported.",
        "claim":"A preference reversal exists for every p strictly between the two boundaries; no unique crossing is asserted."}
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(f"Wrote exact curves, sufficient-bound certificates and figure to {out}",flush=True)


if __name__=="__main__":
    main()
