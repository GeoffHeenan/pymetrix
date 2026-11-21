"""The multiplier bootstrap of the copula goodness-of-fit test (Kojadinovic and Yan 2011), checked independently:

- against R's copula 1.1-7, gofCopula(cop, x, N = 4000, simulation = "mult", method = "Sn") on seeded data (its
  numbers written here, so R is not needed): the statistic S_n to relative 1e-3 (agreeing to about 1e-6) for
  Clayton, Gumbel, Frank and Joe by maximum pseudo-likelihood, Clayton and Gumbel in three dimensions, Clayton and
  Gumbel and the Gaussian by inversion of Kendall's tau; the p-value within Monte Carlo error of R's (both from 4,000
  multiplier replicates: 0.05 absolute, below 0.01 where R's is). (The Gaussian by maximum likelihood is left out:
  its correlation here is the normal scores' rather than R's pseudo-likelihood one, asymptotically the same
  estimator but not numerically; and on Gumbel data R's fitCopula stops at a Clayton parameter of lower likelihood
  than the maximum, 16.05 against 26.62.)
- size on data from the null: a small seeded study (60 replications of 200 observations, 200 multipliers each:
  rejections at 5% inside the 99% binomial band) and, with PYMETRIX_SLOW=1, 400 replications for two families
  (Clayton 26 of 400; Gumbel 6 of 400, conservative at n = 200, so only bounded above there);
  power against a wrong family; the p-value close to the parametric bootstrap's on the same data;
- through the API: copula(..., gof=True, bootstrap="multiplier"), compare_copulas, the summary's note, the error
  for the t copula (its correlations from Kendall's tau, its degrees of freedom by maximum likelihood).
"""

import importlib
import os

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, compare_copulas, copula
from pymetrix.errors import SpecificationError

cp = importlib.import_module("pymetrix_finance.copulas")


def _data():
    rng = np.random.default_rng(2026)
    x2 = cp.sample_copula("clayton", {"theta": 2.0}, 300, 2, rng)
    x3 = cp.sample_copula("clayton", {"theta": 1.5}, 250, 3, rng)
    x4 = cp.sample_copula("gumbel", {"theta": 1.6}, 300, 2, rng)
    return {2: x2, 3: x3, 4: x4}


# set.seed(1); gofCopula(cop, x, N = 4000, simulation = "mult", method = "Sn", estim.method = est)
R_MULT = [  # family, data, method, R's statistic, R's p-value, R's parameter
    ("clayton", 2, "ml", 0.018188773198109524, 0.394026, 1.84304836),
    ("gumbel", 2, "ml", 0.1516033936382111, 0.000125, 1.714473679),
    ("frank", 2, "ml", 0.074951449620463323, 0.000125, 5.271376522),
    ("joe", 2, "ml", 0.46079909085457155, 0.000125, 1.745563146),
    ("clayton", 2, "itau", 0.018388409643109581, 0.276306, 1.804393927),
    ("gumbel", 2, "itau", 0.11758203409357704, 0.000125, 1.902196963),
    ("clayton", 3, "ml", 0.057974941394748937, 0.050112, 1.541346107),
    ("gumbel", 3, "ml", 0.26747309493936433, 0.000125, 1.736047733),
    ("gumbel", 4, "ml", 0.030715973962380721, 0.075856, 1.615734677),
    ("gaussian", 4, "itau", 0.064710380638529047, 0.000625, 0.5237671331),
]


@pytest.fixture(scope="module")
def data():
    return _data()


@pytest.mark.parametrize("case", R_MULT, ids=[f"{c[0]}-{c[1]}-{c[2]}" for c in R_MULT])
def test_against_r(data, case):
    fam, k, method, stat, pvalue, par = case
    U = cp.pseudo_observations(data[k])
    fit = cp.fit_copula(U, fam, method)
    assert (fit["R"][0, 1] if fam == "gaussian" else fit["theta"]) == pytest.approx(par, rel=1e-5)
    out = cp.gof_test(U, fam, method, 4000, 1, bootstrap="multiplier")
    assert out["statistic"] == pytest.approx(stat, rel=1e-3)
    if pvalue < 0.005:
        assert out["pvalue"] < 0.01
    else:
        assert abs(out["pvalue"] - pvalue) < 0.05


def _size(fam, par, reps, n=200, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    rejected = 0
    for r in range(reps):
        U = cp.pseudo_observations(cp.sample_copula(fam, par, n, 2, rng))
        rejected += cp.multiplier_test(U, fam, "ml", n_boot, r)["pvalue"] < 0.05
    return rejected


def test_size_small_study():
    assert _size("clayton", {"theta": 2.0}, 60) <= 8  # Binomial(60, 0.05): 99% band [0, 8]


@pytest.mark.skipif(not os.environ.get("PYMETRIX_SLOW"), reason="a minute or two: set PYMETRIX_SLOW=1")
def test_size_400_replications():
    # Binomial(400, 0.05): 99.9% band [8, 33]. At n = 200 the multiplier test is conservative for Gumbel (6 of 400
    # here, as Kojadinovic and Yan's simulations find for small samples): not too liberal, and not degenerate
    assert 8 <= _size("clayton", {"theta": 2.0}, 400, seed=1) <= 33
    assert 2 <= _size("gumbel", {"theta": 1.5}, 400, seed=2) <= 33


def test_power_and_against_the_parametric_bootstrap(data):
    U = cp.pseudo_observations(data[4])  # Gumbel data
    assert cp.gof_test(U, "clayton", n_boot=500, bootstrap="multiplier")["pvalue"] < 0.01
    U = cp.pseudo_observations(data[2])  # Clayton data
    mult = cp.gof_test(U, "clayton", n_boot=1000, seed=3, bootstrap="multiplier")
    par = cp.gof_test(U, "clayton", n_boot=200, seed=3)
    assert mult["statistic"] == par["statistic"]
    assert abs(mult["pvalue"] - par["pvalue"]) < 0.15
    assert cp.gof_test(U, "clayton", n_boot=300, seed=4, bootstrap="multiplier") == cp.gof_test(U, "clayton", n_boot=300, seed=4, bootstrap="multiplier")


def test_through_the_api(data):
    x = np.log(data[2]) * 0.01  # any increasing transform of the margins: the same ranks
    ds = Dataset.from_frame(pd.DataFrame(x, columns=["a", "b"], index=pd.period_range("2001-01-01", periods=300, freq="D", name="time")))
    fit = copula(ds, ["a", "b"], family="clayton", gof=True, n_boot=500, bootstrap="multiplier")
    assert fit.gof_bootstrap == "multiplier" and fit.gof_result["n_boot"] == 500
    assert "multiplier bootstrap" in str(fit.summary())
    table = compare_copulas(ds, ["a", "b"], families=["clayton", "gumbel"], gof=True, n_boot=300, bootstrap="multiplier")
    assert "multiplier" in " ".join(table.notes)
    with pytest.raises(SpecificationError, match="multiplier"):
        copula(ds, ["a", "b"], family="t", gof=True, bootstrap="multiplier")
    with pytest.raises(SpecificationError, match="bootstrap is one of"):
        copula(ds, ["a", "b"], family="clayton", gof=True, bootstrap="wild")


def test_the_dialogs_replay(data):
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS

    x = np.log(data[2]) * 0.01
    ds = Dataset.from_frame(pd.DataFrame(x, columns=["a", "b"], index=pd.period_range("2001-01-01", periods=300, freq="D", name="time")))
    s = Session()
    s.objects["data"] = ds
    s.run(COMMANDS["FitCopula"].cls("a, b", family="clayton", gof=True, bootstrap="multiplier"))
    code = s.history[-1].code
    assert "bootstrap='multiplier'" in code and "n_boot=1000" in code
    namespace = dict(s.namespace) | {"data": ds}
    exec("from pymetrix_finance import *\n" + code, namespace)
    assert namespace["copula_fit"].gof_result == s.objects["copula_fit"].gof_result
    s.run(COMMANDS["CompareCopulas"].cls("a, b", families="clayton, tawn1", gof=True, bootstrap="multiplier"))
    code = s.history[-1].code
    assert "bootstrap='multiplier'" in code and "'tawn1'" in code
    exec("from pymetrix_finance import *\n" + code, namespace)
    assert namespace["copula_comparison"].rows == s.objects["copula_comparison"].rows
