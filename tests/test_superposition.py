import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import unittest
import tempfile
import shutil

from aliasing.superposition import encode, population, probabilities, sample

ROOT = Path(__file__).resolve().parents[1]


def load_exact():
    spec = importlib.util.spec_from_file_location("exact_study", ROOT/"experiments/06_superposition_exact.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSuperposition(unittest.TestCase):
    def test_exact_tradeoff_and_independent_boundary(self):
        exact = load_exact()
        beta = np.array([1., 1.])
        shared = exact.exact_case(.1, .022, "shared", beta)
        dropped = exact.exact_case(.1, .022, "drop", beta)
        np.testing.assert_allclose(shared["B"], .08570616113744076)
        np.testing.assert_allclose(shared["A"], .072336)
        assert shared["B"] < dropped["B"]
        assert shared["A"] > dropped["A"]
        assert shared["B"]+shared["A"] > dropped["B"]+dropped["A"]
        p = (1-1/np.sqrt(3))/2
        shared = exact.exact_case(p, p*p, "shared", beta)
        dropped = exact.exact_case(p, p*p, "drop", beta)
        np.testing.assert_allclose(shared["B"], dropped["B"])
        np.testing.assert_allclose(shared["A"], dropped["A"])


    def test_algorithm_selected_D_reverses_B_plus_A_benefit(self):
        exact = load_exact()
        beta = np.array([1., 1.])
        shared = exact.exact_case(.2, .04, "shared", beta)
        dropped = exact.exact_case(.2, .04, "drop", beta)
        assert shared["B"]+shared["A"] < dropped["B"]+dropped["A"]
        assert shared["D"] > dropped["D"]
        np.testing.assert_allclose(shared["R"], .5152)
        np.testing.assert_allclose(dropped["R"], .4864)


    def test_uniform_moments_match_analytic_integrals(self):
        p, q = .2, .06
        pop = population(1,p,q,"shared")
        t = p-q
        np.testing.assert_allclose(pop.G[0,1], t/2+q/6)
        np.testing.assert_allclose(pop.G[1,1], t/3+q/12)
        np.testing.assert_allclose(pop.G[1,2], 0,atol=1e-15)
        np.testing.assert_allclose(pop.cross[0,1], t/3+q/8)
        np.testing.assert_allclose(pop.cross[1,1], q/24)
        np.testing.assert_allclose(pop.H[0,1], q/4)
        np.testing.assert_allclose(pop.H[0,0], p/3)


    def test_multifeature_task_matrix_matches_independent_direct_population_risk(self):
        rng = np.random.default_rng(91)
        p,q,pairs = .3,.12,3
        pop=population(pairs,p,q,"shared")
        ss=sample(rng,5,pairs,p,q)
        fit=pop.fit(ss,encode(ss,"shared"))
        # Integrate predictions on an independently constructed exact product grid;
        # compare with the general matrix formula, not only B+A+D itself.
        from aliasing.superposition import pair_quadrature
        s,w=pair_quadrature(p,q,"uniform")
        indexes=np.array(np.meshgrid(*([np.arange(len(s))]*pairs),indexing="ij")).reshape(pairs,-1).T
        grid=s[indexes].reshape(len(indexes),2*pairs)
        weights=np.prod(w[indexes],axis=1)
        for beta in (np.ones(6),np.tile([1.,-1.],3),rng.normal(size=6)):
            row=fit.evaluate(beta)
            prediction=encode(grid,"shared")@fit.head@beta
            direct=float(weights@((grid@beta-prediction)**2))
            np.testing.assert_allclose(row["R"], direct,atol=1e-10)
            np.testing.assert_allclose(row["R"], row["B"]+row["projected_A"]+row["projected_D"],atol=1e-10)
        contrast=np.tile([1.,-1.],3)
        np.testing.assert_allclose(contrast@pop.B@contrast, 0,atol=1e-12)
        np.testing.assert_allclose(contrast@fit.A@contrast, 0,atol=1e-12)


    def test_coordinate_change_preserves_B_A_but_can_change_euclidean_D(self):
        rng=np.random.default_rng(32)
        ss=sample(rng,3,3,.3,.09)
        beta=rng.normal(size=6)
        rows=[]
        for mode in ("shared","shared_scaled"):
            pop=population(3,.3,.09,mode)
            rows.append(pop.fit(ss,encode(ss,mode)).evaluate(beta))
        np.testing.assert_allclose(rows[0]["B"], rows[1]["B"],atol=1e-10)
        np.testing.assert_allclose(rows[0]["A"], rows[1]["A"],atol=1e-10)
        assert abs(rows[0]["D"]-rows[1]["D"])>1e-5


    def test_impossible_interpolation_is_not_reported_as_original_GAD(self):
        ss=np.array([[0.,0.],[1.,1.]])
        pop=population(1,.1,.022,"shared","binary")
        row=pop.fit(ss,encode(ss,"shared")).evaluate(np.ones(2))
        assert not row["interpolates"]
        assert np.isnan(row["A"]) and np.isnan(row["D"])
        assert row["train_residual_norm"]>1
        assert np.isfinite(row["projected_A"])
        contrast=pop.fit(ss,encode(ss,"shared")).evaluate(np.array([1.,-1.]))
        assert contrast["interpolates"]


    def test_degenerate_population_and_coverage_endpoints(self):
        exact=load_exact()
        for p,q in ((.1,0),(.5,.5),(.1,.1)):
            rows=exact.coverage(p,q,[1,2,4,8])
            assert all(-1e-12<=r["interpolation_probability"]<=1+1e-12 for r in rows)
            exact.exact_case(p,q,"shared",np.ones(2))
        rows=exact.coverage(.1,.022,[2])
        row=next(r for r in rows if r["mode"]=="shared" and r["task"]=="sum")
        np.testing.assert_allclose(row["interpolation_probability"], 1-2*.822*.022)


    def test_runner_smoke_writes_real_csvs_and_accepts_options(self):
        tmp_path=Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree,tmp_path)
        subprocess.run(["bash",str(ROOT/"run.sh"),"07","--pairs","2","--marginals","0.1",
            "--cooccurrence-fractions","0,0.22","--sample-sizes","1,8","--replicates","2",
            "--validation-replicates","2","--n-directions","1","--n-test","32",
            "--output-dir",str(tmp_path)],check=True,capture_output=True,text=True,
            env={**__import__("os").environ,"PYTHON":sys.executable})
        for name in ("replicates.csv","summary.csv","paired_summary.csv","population.csv",
                     "task_spectrum.csv","task_validation.csv","risk_matrices.npz","metadata.json"):
            assert (tmp_path/name).stat().st_size>10

    def test_slurm_staged_runner_resolves_repository_and_writes_exact_csv(self):
        import os
        tmp_path=Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree,tmp_path)
        staged=tmp_path/"staged-run.sh"
        shutil.copyfile(ROOT/"run.sh",staged)
        output=tmp_path/"results"
        subprocess.run(["bash",str(staged),"06","--marginals","0.1",
            "--cooccurrence-fractions","0.22","--sample-sizes","1,2",
            "--angles","4","--output-dir",str(output)],check=True,
            capture_output=True,text=True,cwd=tmp_path,
            env={**os.environ,"SLURM_SUBMIT_DIR":str(ROOT),"SLURM_CPUS_PER_TASK":"2",
                 "PYTHON":sys.executable})
        assert (output/"exact.csv").stat().st_size>10


if __name__ == "__main__":
    unittest.main()
