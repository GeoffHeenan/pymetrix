"""The vine copula's goodness-of-fit test (the Rosenblatt transform, VineCopula's ECP2 Cramer-von Mises statistic,
a parametric bootstrap that refits the vine), checked independently:

- against R's VineCopula 2.6.1 on a seeded data set and the vine fitted to it here (R's numbers written here, so R
  is not needed): RVinePIT, the Rosenblatt transform, at the first two observations (1e-8), RVineLogLik (1e-9) and
  RVineGofTest(..., method = "ECP2", statistic = "CvM") (relative 1e-3; they agree to about 1e-13; R's own p-value
  resamples the data with the parameters fixed, so it is not compared);
- the Rosenblatt transform inverts the simulation (1e-9);
- size on data from the null: a seeded study (30 replications of 200 observations from a three-series vine, 19
  bootstrap samples each, the parameters refitted: at most 5 rejections at 5%; 100 replications with
  PYMETRIX_SLOW=1); power against a wrong family (Clayton-vine data fitted by Gaussian pair copulas, p < 0.05);
- through the API: vine_copula(..., gof=True), the summary's note, the result reproducible with its seed.
"""

import importlib
import os

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, vine_copula

V = importlib.import_module("pymetrix_finance.vines")
cp = importlib.import_module("pymetrix_finance.copulas")

M = [[4, 0, 0, 0], [1, 3, 0, 0], [2, 1, 2, 0], [3, 2, 1, 1]]
F = [[0, 0, 0, 0], [3, 0, 0, 0], [1, 24, 0, 0], [2, 4, 5, 0]]
P = [[0, 0, 0, 0], [1.2, 0, 0, 0], [0.3, -1.4, 0, 0], [0.6, 1.8, 4.0, 0]]
Q = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [5.0, 0, 0, 0]]
FAMS = ["indep", "gaussian", "t", "clayton", "gumbel", "frank", "joe", "clayton180", "gumbel180", "joe180", "clayton90",
        "gumbel90", "joe90", "clayton270", "gumbel270", "joe270"]  # RVineGofTest's families
# with the fitted vine's to_vinecopula() as RVM: RVineGofTest(u, RVM, method = "ECP2", statistic = "CvM", B = 0)$CvM
R_ECP2 = 0.04465750241558359
R_LOGLIK = 462.5135008611424
R_PIT = [[0.32061501019425559, 0.8491564324294979, 0.92937785393772354, 0.59102244389027436],
         [0.84144470262135174, 0.0060031603358029657, 0.89535621009386979, 0.29675810473815462]]


# the vine fitted to the data below by fit_vine(U, families=FAMS), as to_vinecopula() wrote it for R (fixed here, so
# optimiser differences between SciPy versions cannot move the R comparison; the matrix sets the transform's order)
FIT_M = [[1, 0, 0, 0], [4, 2, 0, 0], [3, 4, 3, 0], [2, 3, 4, 4]]
FIT_F = [[0, 0, 0, 0], [3, 0, 0, 0], [23, 1, 0, 0], [5, 4, 2, 0]]
FIT_P = [[0, 0, 0, 0], [1.3055273176641193, 0, 0, 0], [-0.56960955210913899, 0.29932673639105678, 0, 0],
         [3.8886148018278188, 1.7763584526011307, 0.6563738659322701, 0]]
FIT_Q = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2.7241759833191987, 0]]


@pytest.fixture(scope="module")
def fitted():
    trees = V.from_matrix(M, F, P, Q)
    U = cp.pseudo_observations(V.simulate_vine(trees, 4, 400, np.random.default_rng(7)))
    return U, V.from_matrix(FIT_M, FIT_F, FIT_P, FIT_Q)


def test_against_vinecopula(fitted):
    U, trees = fitted
    ours = V.fit_vine(U, families=FAMS)  # the same vine again, within the optimiser's tolerance
    assert V.propagate(U, ours)[0].sum() == pytest.approx(R_LOGLIK, rel=1e-6)
    np.testing.assert_allclose(V.rosenblatt(U, trees, FIT_M)[:2], R_PIT, rtol=1e-8)  # 2e-9 with SciPy 1.11 (its t quantile)
    assert V.propagate(U, trees)[0].sum() == pytest.approx(R_LOGLIK, rel=1e-9)
    assert V.vine_gof_statistic(U, trees, FIT_M) == pytest.approx(R_ECP2, rel=1e-3)
    # the fitted vine's own matrix is that one, so its statistic is R's too
    assert V.vine_gof_statistic(U, ours) == pytest.approx(R_ECP2, rel=1e-3)


def test_rosenblatt_inverts_the_simulation(fitted):
    _, trees = fitted
    W = np.random.default_rng(3).uniform(size=(500, 4))
    U = V.simulate_vine(trees, 4, 500, np.random.default_rng(0), w=W)
    Z = V.rosenblatt(U, trees)
    Mx = V.to_matrix(trees, 4)
    for k in range(4):
        np.testing.assert_allclose(Z[:, Mx[3 - k, 3 - k]], W[:, k], atol=1e-9)


TRUE = V.from_matrix([[3, 0, 0], [1, 2, 0], [2, 1, 1]], [[0, 0, 0], [5, 0, 0], [3, 4, 0]], [[0, 0, 0], [2.0, 0, 0], [1.5, 1.6, 0]])


def _size(reps, seed):
    rng = np.random.default_rng(seed)
    p = []
    for r in range(reps):
        U = cp.pseudo_observations(V.simulate_vine(TRUE, 3, 200, rng))
        p.append(V.vine_gof_test(U, V.refit(U, TRUE), 19, r)["pvalue"])
    return int((np.array(p) < 0.05).sum())


def test_size_small_study():
    assert _size(30, 123) <= 5  # Binomial(30, 0.05): 99% band [0, 5]


@pytest.mark.skipif(not os.environ.get("PYMETRIX_SLOW"), reason="a few minutes: set PYMETRIX_SLOW=1")
def test_size_100_replications():
    assert _size(100, 5) <= 12  # Binomial(100, 0.05): 99.9% band [0, 12]


def test_power():
    clayton = V.from_matrix([[3, 0, 0], [1, 2, 0], [2, 1, 1]], [[0, 0, 0], [3, 0, 0], [3, 3, 0]], [[0, 0, 0], [1.0, 0, 0], [4.0, 4.0, 0]])
    U = cp.pseudo_observations(V.simulate_vine(clayton, 3, 300, np.random.default_rng(1)))
    assert V.vine_gof_test(U, V.fit_vine(U, families=["gaussian"]), 19, 0)["pvalue"] < 0.05


def test_through_the_api():
    U = V.simulate_vine(TRUE, 3, 300, np.random.default_rng(8))
    x = np.log(U / (1 - U)) * 0.01
    ds = Dataset.from_frame(pd.DataFrame(x, columns=["a", "b", "c"], index=pd.period_range("2001-01-01", periods=300, freq="D", name="time")))
    fit = vine_copula(ds, ["a", "b", "c"], families=["clayton", "gumbel", "frank"], gof=True, n_boot=19, seed=2)
    g = fit.gof_result
    assert g["n_boot"] == 19 and 0 < g["pvalue"] < 1 and g["statistic"] == pytest.approx(V.vine_gof_statistic(fit.u, fit.trees))
    assert "ECP2" in str(fit.summary())
    assert fit.goodness_of_fit(19, 2) == g
    assert fit.rosenblatt().shape == (300, 3)
    fit_t = vine_copula(ds, ["a", "b", "c"], families=["clayton", "gumbel", "frank"], margins="t", gof=True, n_boot=5)
    assert fit_t.gof_result["n_boot"] == 5


def test_the_dialog_replays():
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS

    U = V.simulate_vine(TRUE, 3, 200, np.random.default_rng(9))
    ds = Dataset.from_frame(pd.DataFrame(np.log(U / (1 - U)) * 0.01, columns=["a", "b", "c"],
                                         index=pd.period_range("2001-01-01", periods=200, freq="D", name="time")))
    s = Session()
    s.objects["data"] = ds
    s.run(COMMANDS["FitVineCopula"].cls("a, b, c", families="clayton, gumbel, frank", gof=True))
    code = s.history[-1].code
    assert "gof=True" in code
    namespace = dict(s.namespace) | {"data": ds}
    exec("from pymetrix_finance import *\n" + code, namespace)
    assert namespace["vine_fit"].gof_result == s.objects["vine_fit"].gof_result
    assert "families='all'" in COMMANDS["FitVineCopula"].cls("a, b, c", families="all").to_code()
    assert "'tawn1'" in COMMANDS["FitVineCopula"].cls("a, b, c", families="gaussian, tawn1").to_code()
