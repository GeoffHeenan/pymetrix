"""Regression extras, checked independently: Cochrane-Orcutt against statsmodels' GLSAR at the converged rho (and rho
the residuals' own AR(1) coefficient); Prais-Winsten against statsmodels' GLS with the AR(1) covariance; multiplicative
heteroskedasticity against the likelihood maximised by a general optimiser, and recovering a simulated variance
function; the stochastic frontier against its likelihood maximised by a general optimiser, recovering simulated
sigma_u, sigma_v and efficiency; the J and Cox tests against statsmodels' compare_j and compare_cox."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import optimize
from scipy import stats as sps

from pymetrix import CochraneOrcutt, Dataset, Equation, MultiplicativeHeteroskedasticity, PraisWinsten, StochasticFrontier, cox_test, j_test
from pymetrix.errors import SpecificationError

sm = pytest.importorskip("statsmodels.api")


def _ar1_data(T=200, seed=0, rho=0.6):
    rng = np.random.default_rng(seed)
    x, z = rng.normal(size=T), rng.normal(size=T)
    u, e = np.zeros(T), rng.normal(size=T)
    for t in range(1, T):
        u[t] = rho * u[t - 1] + e[t]
    y = 1 + 2 * x + 0.5 * z + u
    return Dataset.from_frame(pd.DataFrame({"y": y, "x": x, "z": z}, index=pd.period_range("1950Q1", periods=T, freq="Q"))), y, x, z


def test_cochrane_orcutt_against_glsar():
    data, y, x, _ = _ar1_data()
    fit = Equation("y ~ x").fit(data, method=CochraneOrcutt())
    rho = fit.stats["rho"]
    ref = sm.GLSAR(y, sm.add_constant(x), rho=[rho]).fit()
    assert fit.params.values == pytest.approx(ref.params, rel=1e-10) and fit.params.se == pytest.approx(ref.bse, rel=1e-10)
    u = fit.resid.to_numpy()
    assert rho == pytest.approx(u[1:] @ u[:-1] / (u[:-1] @ u[:-1]), abs=1e-9)  # the fixed point
    one = Equation("y ~ x").fit(data, method=CochraneOrcutt(iterate=False))
    assert one.stats["iterations"] == 1


def test_prais_winsten_is_gls():
    data, y, x, _ = _ar1_data(seed=1)
    fit = Equation("y ~ x").fit(data, method=PraisWinsten())
    r = fit.stats["rho"]
    t = np.arange(len(y))
    V = r ** np.abs(t[:, None] - t[None, :]) / (1 - r**2)
    ref = sm.GLS(y, sm.add_constant(x), sigma=V).fit()
    assert fit.params.values == pytest.approx(ref.params, rel=1e-10) and fit.params.se == pytest.approx(ref.bse, rel=1e-8)
    assert r == pytest.approx(0.6, abs=0.15)


def test_multiplicative_heteroskedasticity():
    rng = np.random.default_rng(2)
    n = 2000
    x, z = rng.normal(size=n), rng.uniform(0, 2, n)
    y = 1 + x + np.exp(0.5 * (-1 + 1.5 * z)) * rng.normal(size=n)
    data = Dataset.from_frame(pd.DataFrame({"id": range(n), "y": y, "x": x, "z": z}), entity="id")
    fit = Equation("y ~ x").fit(data, method=MultiplicativeHeteroskedasticity(variance=["z"]))
    X, Z = sm.add_constant(x), sm.add_constant(z)

    def negll(th):
        h = Z @ th[2:]
        return 0.5 * np.sum(np.log(2 * np.pi) + h + (y - X @ th[:2]) ** 2 * np.exp(-h))

    ref = optimize.minimize(negll, np.zeros(4), method="BFGS", options={"gtol": 1e-10})
    ours = np.r_[fit.params.values, fit.stats["variance.const"], fit.stats["variance.z"]]
    assert ours == pytest.approx(ref.x, abs=1e-5) and fit.stats["loglike"] == pytest.approx(-ref.fun, rel=1e-10)
    assert fit.stats["variance.z"] == pytest.approx(1.5, abs=0.15) and fit.stats["lr_homoskedastic_pvalue"] < 1e-10
    with pytest.raises(SpecificationError, match="variance"):
        Equation("y ~ x").fit(data, method=MultiplicativeHeteroskedasticity())


def test_the_stochastic_frontier():
    rng = np.random.default_rng(3)
    n = 3000
    x = rng.normal(size=n)
    u, v = np.abs(rng.normal(size=n)) * 0.8, rng.normal(size=n) * 0.4
    y = 2 + 0.7 * x + v - u
    data = Dataset.from_frame(pd.DataFrame({"id": range(n), "y": y, "x": x}), entity="id")
    fit = Equation("y ~ x").fit(data, method=StochasticFrontier())
    X = sm.add_constant(x)

    def negll(th):  # parametrised by sigma_v and sigma_u directly: a different route from ours
        sv, su = abs(th[2]), abs(th[3])
        sig, lam = np.hypot(sv, su), su / sv
        e = y - X @ th[:2]
        return -np.sum(np.log(2) - np.log(sig) + sps.norm.logpdf(e / sig) + sps.norm.logcdf(-e * lam / sig))

    ref = optimize.minimize(negll, [2.5, 0.7, 0.5, 0.5], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    assert fit.params.values == pytest.approx(ref.x[:2], abs=1e-5)
    assert [fit.stats["sigma_v"], fit.stats["sigma_u"]] == pytest.approx(np.abs(ref.x[2:]), abs=1e-5)
    assert fit.stats["sigma_u"] == pytest.approx(0.8, abs=0.06) and fit.stats["mean_efficiency"] == pytest.approx(np.mean(np.exp(-u)), abs=0.02)
    assert np.corrcoef(fit.extra["inefficiency"], u)[0, 1] > 0.7
    costs = Dataset.from_frame(pd.DataFrame({"id": range(n), "c": 2 + 0.7 * x + v + u, "x": x}), entity="id")
    cost = Equation("c ~ x").fit(costs, method=StochasticFrontier(cost=True))  # the mirror image: inefficiency raises cost
    assert cost.stats["sigma_u"] == pytest.approx(0.8, abs=0.08) and cost.params["x"] == pytest.approx(0.7, abs=0.03)


def test_non_nested_against_statsmodels():
    from statsmodels.stats.diagnostic import compare_cox, compare_j

    data, y, x, z = _ar1_data(seed=4)
    r1, r2 = sm.OLS(y, sm.add_constant(x)).fit(), sm.OLS(y, sm.add_constant(z)).fit()
    j = j_test(Equation("y ~ z"), Equation("y ~ x"), data)  # statsmodels' compare_j(x, z) tests the second model
    def reference(compare):  # statsmodels before 0.14.2 returns (statistic, p value) and has no result_object
        try:
            return compare(r1, r2, result_object=True)
        except TypeError:
            stat, p = compare(r1, r2)[:2]
            return SimpleNamespace(statistic=stat, pvalue=p)

    ref_j = reference(compare_j)
    assert j.statistic == pytest.approx(ref_j.statistic, rel=1e-10) and j.p_value == pytest.approx(ref_j.pvalue, rel=1e-8)
    cox = cox_test(Equation("y ~ x"), Equation("y ~ z"), data)
    ref_c = reference(compare_cox)
    assert cox.statistic == pytest.approx(ref_c.statistic, rel=1e-10) and cox.p_value == pytest.approx(ref_c.pvalue, rel=1e-8)
    with pytest.raises(SpecificationError, match="same dependent variable"):
        j_test(Equation("y ~ x"), Equation("x ~ z"), data)
