import math
from fractions import Fraction
import unittest

import numpy as np

from aliasing.finite_sample import (
    P_SMALL,P_POPULATION,binary_cells,expected_risk,sufficient_bound,
)
from aliasing.superposition import encode,population,probabilities


class TestSampleBoundary(unittest.TestCase):
    def test_exact_expectation_matches_all_training_count_vectors(self):
        p,q=.2,.04
        probs=probabilities(p,q)
        states=np.array([[0,0],[1,0],[0,1],[1,1]],float)
        beta=np.ones(2)
        for n in (1,2,5):
            for mode in ("shared","drop"):
                pop=population(1,p,q,mode,"binary")
                direct=np.zeros(3)
                for a in range(n+1):
                    for b in range(n-a+1):
                        for c in range(n-a-b+1):
                            counts=[a,b,c,n-a-b-c]
                            mass=math.factorial(n)
                            for j,count in enumerate(counts):
                                mass=mass/math.factorial(count)*probs[j]**count
                            s=np.repeat(states,counts,axis=0)
                            row=pop.fit(s,encode(s,mode)).evaluate(beta)
                            direct+=mass*np.array([row["R"],row["projected_A"],row["projected_D"]])
                exact=expected_risk(n,*binary_cells(p,q,mode))
                np.testing.assert_allclose(direct,[exact["R"],exact["sampling_excess"],
                                                   exact["unobserved_excess"]],atol=1e-11)

    def test_one_sample_boundary_and_later_reversal(self):
        for p in (.12,.18,.2,.24):
            q=p*p
            shared=expected_risk(1,*binary_cells(p,q,"shared"))
            dropped=expected_risk(1,*binary_cells(p,q,"drop"))
            expected=p*(1-p)*(-3+19*p-2*p*p)/4
            np.testing.assert_allclose(shared["R"]-dropped["R"],expected,atol=1e-12)
        assert P_SMALL<.2<P_POPULATION
        assert expected_risk(1,*binary_cells(.2,.04,"shared"))["R"] > expected_risk(1,*binary_cells(.2,.04,"drop"))["R"]
        assert expected_risk(16,*binary_cells(.2,.04,"shared"))["R"] < expected_risk(16,*binary_cells(.2,.04,"drop"))["R"]

    def test_n47_sufficient_certificate_with_exact_rational_arithmetic(self):
        B=Fraction(64,425);k=Fraction(17,25);t=Fraction(4,25)
        c=Fraction(28,51)**2
        upper=2*B/(48*k)+2*t*(1-t)**47+k*c*(1-k)**47
        gap=Fraction(4,25)-B
        assert upper<gap
        np.testing.assert_allclose(sufficient_bound(47,.2),float(upper),atol=1e-13)

    def test_risk_excess_bound_and_degenerate_support(self):
        for p in (.18,.2,.21):
            for n in (1,2,8,32,64):
                row=expected_risk(n,*binary_cells(p,p*p,"shared"))
                assert row["R"]-row["B"]<=sufficient_bound(n,p)+1e-12
        for p,q in ((.1,0),(.1,.1),(.5,.5)):
            for mode in ("shared","drop"):
                row=expected_risk(8,*binary_cells(p,q,mode))
                assert np.isfinite(row["R"])
                assert row["R"]>=row["B"]-1e-12


if __name__=="__main__":
    unittest.main()
