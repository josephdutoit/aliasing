"""Exact finite-cell minimum-norm least-squares risk, including unseen cells.

Sampling and unseen-cell excess equal original A,D with one sample. For larger
samples they describe projected-label least squares, not original GAD on
incompatible datasets. All expectations are finite occupancy sums.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np


def subsets(mask):
    value = mask
    while True:
        yield value
        if value == 0:
            break
        value = (value - 1) & mask


@lru_cache(maxsize=16384)
def inverse_count(n: int, a: float, b: float) -> float:
    """Sum_{j=1}^n binom(n,j) a^j b^(n-j)/j; a+b may be <1."""
    if a == 0:
        return 0.
    if b == 0:
        return a**n/n
    loga, logb = math.log(a), math.log(b)
    return sum(math.exp(math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)
                        + j*loga+(n-j)*logb-math.log(j)) for j in range(1,n+1))


def expected_risk(n, pi, mu, variance, design):
    """Exact E[R] for a saturated finite-cell class and a Euclidean min-norm head.

    design contains one readout vector per encoded cell. Its rows must be
    independent on positive-mass cells, so observed cell means can be fitted.
    """
    if n < 1:
        raise ValueError("n must be positive")
    positive=np.asarray(pi)>0
    pi,mu,variance=(np.asarray(x,float)[positive] for x in (pi,mu,variance))
    design=np.asarray(design,float)[positive]
    if np.linalg.matrix_rank(design) != len(pi):
        raise ValueError("positive-mass cell rows must be independent")
    B=float(pi@variance)
    sampling=unseen=total_probability=0.
    cells=len(pi)
    for mask in range(1,1<<cells):
        observed=[j for j in range(cells) if mask>>j&1]
        probability=sum((-1)**(len(observed)-a.bit_count())
                        *sum(pi[j] for j in range(cells) if a>>j&1)**n
                        for a in subsets(mask))
        total_probability+=probability
        transfer=design@np.linalg.pinv(design[observed],rcond=1e-12)
        miss=transfer@mu[observed]-mu
        missing=np.array([not (mask>>j&1) for j in range(cells)])
        unseen+=probability*float(pi[missing]@(miss[missing]**2))
        for col,j in enumerate(observed):
            other=mask^(1<<j)
            reciprocal=sum((-1)**(other.bit_count()-a.bit_count())
                           *inverse_count(n,float(pi[j]),float(sum(pi[k] for k in range(cells) if a>>k&1)))
                           for a in subsets(other))
            sampling+=pi[j]*variance[j]*reciprocal
            unseen+=variance[j]*float(pi[missing]@(transfer[missing,col]**2))*reciprocal
    if not np.isclose(total_probability,1,atol=1e-10):
        raise ArithmeticError("occupancy probabilities do not sum to one")
    return dict(B=B,sampling_excess=float(sampling),unobserved_excess=float(unseen),
                R=float(B+sampling+unseen))


def binary_cells(p,q,mode):
    if not 0<p<1 or not max(0,2*p-1)<=q<=p:
        raise ValueError("invalid binary co-occurrence probabilities")
    rho,t,k=1-2*p+q,p-q,1-2*p+2*q
    if mode=="shared":
        return (np.array([k,t,t]),np.array([2*q/k if k else 0.,1.,1.]),
                np.array([4*rho*q/k**2 if k else 0.,0.,0.]),
                np.array([[1.,0.,0.],[1.,1.,0.],[1.,0.,1.]]))
    if mode=="drop":
        return (np.array([1-p,p]),np.array([t/(1-p),(t+2*q)/p]),
                np.array([rho*t/(1-p)**2,t*q/p**2]),
                np.array([[1.,0.],[1.,1.]]))
    raise ValueError("mode must be shared or drop")


def interpolation_probability(n,p,q,mode):
    rho,t=1-2*p+q,p-q
    if mode=="shared":
        return (1-rho)**n+(1-q)**n-(1-rho-q)**n
    if mode!="drop":
        raise ValueError("mode must be shared or drop")
    # Feasible supports choose at most one member of each conflicting cell:
    # {00,01} and {10,11}. Union of four allowed-support events.
    allowed=(0b0011,0b1001,0b0110,0b1100)
    masses=np.array([rho,t,t,q])
    value=0.
    for mask in range(1,16):
        intersection=15
        for j in range(4):
            if mask>>j&1:
                intersection &= allowed[j]
        mass=sum(masses[j] for j in range(4) if intersection>>j&1)
        value+=(-1)**(mask.bit_count()+1)*mass**n
    return value


P_SMALL=(19-math.sqrt(337))/4
P_POPULATION=(1-1/math.sqrt(3))/2


def sufficient_bound(n,p):
    """Upper bound on shared R-B for independent binary features and sum task."""
    q=p*p;rho=(1-p)**2;t=p*(1-p);k=rho+q
    B=4*rho*q/k
    mu=2*q/k
    c=max((.5-mu)**2,(2/3-mu)**2)
    return 2*B/((n+1)*k)+2*t*(1-t)**n+k*c*(1-k)**n
