"""Copula estimation, model checking and parametric margins, checked independently:

- maximum likelihood recovers the parameters of every family from 2,000 draws with known (uniform) margins, within
  three standard errors from the observed information; from the ranks the estimates and Kendall's-tau (itau)
  estimates agree roughly; the four-series Archimedean copulas likewise;
- fits on one data set against R's copula package 1.1-7 (fitCopula by maximum pseudo-likelihood and by itau, and
  gofCopula's Cramer-von Mises statistic S_n): Gumbel, Frank and Joe to 1e-6 (Clayton: R's optimiser stops at its
  starting value there, with a lower likelihood than ours);
- the goodness-of-fit test: its size on data from the null family (60 replications here; 200 with PYMETRIX_SLOW=1,
  rejections at 5% within a binomial band), its power against a clearly wrong family, and its p-value reproducible;
- Hansen's skewed t: its density integrates to one with mean 0 and variance 1 (quadrature), its CDF is the integral of
  the density, its quantiles invert the CDF, and with no skewness it is the standardised t;
- IFM and joint maximum likelihood on data simulated from known margins and copula recover both; joint ML's total
  likelihood is at least IFM's; GARCH-t margins recover the degrees of freedom and the copula;
- portfolio VaR and ES through the margins: the model at its true parameters against a direct simulation written
  here (conditional inversion of the Clayton copula, scipy's t), within a Monte Carlo band, and the fitted model
  against the same within its estimation error;
- the comparison table (likelihoods equal to the single fits, the true family preferred by BIC), the errors, and the
  dialogs, whose recorded code reproduces their results.
"""

import importlib
import os
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from pymetrix import Dataset, compare_copulas, copula
from pymetrix.errors import SpecificationError

cp = importlib.import_module("pymetrix_finance.copulas")

TRUE = {"clayton": {"theta": 2.0}, "gumbel": {"theta": 1.8}, "frank": {"theta": 5.0}, "joe": {"theta": 2.5},
        "bb1": {"theta": 0.8, "delta": 1.5}, "bb7": {"theta": 1.6, "delta": 1.2}, "clayton180": {"theta": 2.0},
        "gumbel180": {"theta": 1.8}, "joe180": {"theta": 2.5}, "clayton90": {"theta": 2.0}, "gumbel270": {"theta": 1.8},
        "joe270": {"theta": 2.5}}


def _data(X, names=None):
    idx = pd.period_range("2000-01-01", periods=len(X), freq="D", name="time")
    return Dataset.from_frame(pd.DataFrame(X, columns=names or [f"x{i}" for i in range(X.shape[1])], index=idx))


@pytest.mark.parametrize("family", list(TRUE) + ["frank-"])
def test_ml_recovers_the_parameters(family):
    fam, true = (family, TRUE[family]) if family != "frank-" else ("frank", {"theta": -4.0})
    U = cp.sample_copula(fam, true, 2000, 2, np.random.default_rng(31))
    fit = cp.fit_copula(U, fam)  # known margins: the observed information's standard errors apply
    se = cp._copula_se(U, fam, fit, "ml")
    for key, value in true.items():
        assert abs(fit[key] - value) < 3 * se[key], (key, fit[key], se[key])
    ranks = copula(_data(stats.norm.ppf(U)), ["x0", "x1"], family=fam)  # from the ranks, through the API
    assert ranks.params["theta"] == pytest.approx(fit["theta"], rel=0.1, abs=0.1)
    assert ranks.aic == pytest.approx(2 * ranks.k - 2 * ranks.loglike) and ranks.bic == pytest.approx(ranks.k * np.log(2000) - 2 * ranks.loglike)
    if len(true) == 1:
        itau = copula(_data(stats.norm.ppf(U)), ["x0", "x1"], family=fam, method="itau")
        assert itau.params["theta"] == pytest.approx(ranks.params["theta"], rel=0.1)
        assert itau.se == {}


@pytest.mark.parametrize("family, theta", [("clayton", 1.5), ("gumbel", 1.7), ("frank", 4.0), ("joe", 2.2)])
def test_four_series(family, theta):
    U = cp.sample_copula(family, {"theta": theta}, 2000, 4, np.random.default_rng(32))
    fit = cp.fit_copula(U, family)
    se = cp._copula_se(U, family, fit, "ml")
    assert abs(fit["theta"] - theta) < 3 * se["theta"]
    itau = cp.fit_copula(cp.pseudo_observations(U), family, "itau")
    assert itau["theta"] == pytest.approx(theta, rel=0.1)
    assert "Exchangeable" in str(copula(_data(stats.norm.ppf(U)), ["x0", "x1", "x2", "x3"], family=family).summary())


def test_fits_against_r():
    U = cp.sample_copula("gumbel", {"theta": 1.6}, 300, 2, np.random.default_rng(11))
    P = cp.pseudo_observations(U)
    r = {"gumbel": (1.671782159, 1.687612884, 0.01590614727), "frank": (4.298782453, 4.264555592, 0.03292796196),
         "joe": (1.973386747, 2.258774329, 0.06170810861)}
    for family, (ml, itau, sn) in r.items():
        fit = cp.fit_copula(P, family)
        assert fit["theta"] == pytest.approx(ml, abs=1e-6)
        assert cp.fit_copula(P, family, "itau")["theta"] == pytest.approx(itau, abs=1e-6)
        assert cp.cramer_von_mises(P, family, fit) == pytest.approx(sn, abs=1e-7)
    fit = cp.fit_copula(P, "clayton")  # R's fitCopula stops at 1.375225768, log likelihood 22.5428
    assert fit["loglike"] > 22.5428 + 10 and cp.clayton_logpdf(P, 1.375225768).sum() == pytest.approx(22.54280646, abs=1e-6)


def _size(family, params, reps, n_boot, n=100):
    rejections = 0
    for i in range(reps):
        U = cp.sample_copula(family, params, n, 2, np.random.default_rng(1000 + i))
        rejections += cp.gof_test(cp.pseudo_observations(U), family, "ml", n_boot, seed=i)["pvalue"] < 0.05
    return rejections


def test_goodness_of_fit():
    assert _size("clayton", {"theta": 1.0}, 60, 40) <= 8  # P(Binomial(60, 0.05) > 8) = 0.003
    U = cp.sample_copula("gumbel", {"theta": 1.8}, 300, 2, np.random.default_rng(33))
    data = _data(stats.norm.ppf(U))
    wrong = copula(data, ["x0", "x1"], family="clayton", gof=True)
    right = copula(data, ["x0", "x1"], family="gumbel", gof=True)
    assert wrong.gof_result["pvalue"] < 0.01 and right.gof_result["pvalue"] > 0.05
    assert right.goodness_of_fit(100, seed=0) == right.gof_result  # seeded
    assert "goodness of fit p-value" in str(right.summary())
    t = copula(data, ["x0", "x1"], family="t", gof=True, n_boot=40)  # the t copula's CDF by quadrature
    assert 0 < t.gof_result["pvalue"] <= 1


@pytest.mark.skipif(not os.environ.get("PYMETRIX_SLOW"), reason="about two minutes: set PYMETRIX_SLOW=1")
def test_goodness_of_fit_size_200_replications():
    for family, params in (("clayton", {"theta": 1.0}), ("frank", {"theta": 3.0})):
        assert 2 <= _size(family, params, 200, 100) <= 19  # Binomial(200, 0.05): 10 expected, 99% band [3, 18]


def test_hansen_skewed_t():
    for nu, lam in ((5.0, -0.3), (8.0, 0.5), (30.0, 0.0)):
        pdf = lambda z: np.exp(cp.skewt_logpdf(z, nu, lam))  # noqa: E731
        assert integrate.quad(pdf, -np.inf, np.inf)[0] == pytest.approx(1.0, abs=1e-8)
        assert integrate.quad(lambda z: z * pdf(z), -np.inf, np.inf)[0] == pytest.approx(0.0, abs=1e-7)
        assert integrate.quad(lambda z: z * z * pdf(z), -np.inf, np.inf)[0] == pytest.approx(1.0, abs=1e-6)
        for z in (-2.5, -0.4, 0.0, 0.3, 1.7):
            assert cp.skewt_cdf(z, nu, lam) == pytest.approx(integrate.quad(pdf, -np.inf, z)[0], abs=1e-9)
        q = np.array([0.001, 0.05, 0.3, 0.5, 0.8, 0.999])
        assert cp.skewt_cdf(cp.skewt_ppf(q, nu, lam), nu, lam) == pytest.approx(q, abs=1e-9)  # old SciPy's t.ppf: ~1e-10
    z = np.linspace(-3, 3, 7)
    k = np.sqrt(6.0 / 4.0)
    assert np.exp(cp.skewt_logpdf(z, 6.0, 0.0)) == pytest.approx(stats.t.pdf(z * k, 6.0) * k)
    assert cp.skewt_cdf(-1.0, 5.0, -0.4) > cp.skewt_cdf(-1.0, 5.0, 0.0)  # negative lambda: more mass on the left


def _clayton_t_sample(n, theta, nus, scales, locs, rng):
    """Clayton by conditional inversion (its h-function inverted by hand) and t margins by scipy: independent of the module."""
    u, w = rng.uniform(size=n), rng.uniform(size=n)
    v = ((w ** (-theta / (1 + theta)) - 1) * u ** (-theta) + 1) ** (-1 / theta)
    return np.column_stack([locs[j] + scales[j] * stats.t.ppf(x, nus[j]) for j, x in enumerate((u, v))])


def test_ifm_and_joint_recover_margins_and_copula():
    rng = np.random.default_rng(34)
    U = cp.sample_copula("gumbel", {"theta": 1.8}, 4000, 2, rng)
    X = np.column_stack([0.01 * stats.t.ppf(U[:, 0], 5) * np.sqrt(3 / 5), 0.02 + 0.03 * cp.skewt_ppf(U[:, 1], 6.0, -0.3)])
    data = _data(X, ["a", "b"])
    ifm = copula(data, ["a", "b"], family="gumbel", margins="skewt")
    joint = copula(data, ["a", "b"], family="gumbel", margins="skewt", method="joint")
    for fit in (ifm, joint):
        a, b = (m.params for m in fit.margin_fits)
        assert a["sigma"] == pytest.approx(0.01, rel=0.05) and a["nu"] == pytest.approx(5.0, abs=1.5) and abs(a["lambda"]) < 0.06
        assert b["mu"] == pytest.approx(0.02, abs=0.003) and b["sigma"] == pytest.approx(0.03, rel=0.05)
        assert b["nu"] == pytest.approx(6.0, abs=2.0) and b["lambda"] == pytest.approx(-0.3, abs=0.06)
        assert abs(fit.params["theta"] - 1.8) < 3 * joint.se["theta"]
    assert joint.se["theta"] >= ifm.se["theta"]  # IFM's ignores the margins' estimation
    assert joint.total_loglike >= ifm.total_loglike - 1e-6
    assert joint.params["theta"] == pytest.approx(ifm.params["theta"], abs=2 * joint.se["theta"])
    t_joint = copula(_data(_clayton_t_sample(1500, 2.0, (4.0, 8.0), (0.01, 0.02), (0.0, 0.001), rng)), ["x0", "x1"], family="t",
                     margins="t", method="joint")
    assert set(t_joint.se) == {"rho[0,1]", "nu"} and t_joint.params["R"][0, 1] > 0.5


def test_garch_t_margins():
    rng = np.random.default_rng(35)
    n, nu = 2000, 6.0
    z = stats.t.ppf(cp.sample_copula("clayton", {"theta": 1.5}, n, 2, rng), nu) * np.sqrt((nu - 2) / nu)
    r, h = np.zeros((n, 2)), np.full(2, 0.05 / 0.02)
    for t in range(n):
        r[t] = 0.1 + np.sqrt(h) * z[t]
        h = 0.05 + 0.08 * (r[t] - 0.1) ** 2 + 0.9 * h
    fit = copula(_data(r / 100, ["a", "b"]), ["a", "b"], family="clayton", garch=True, margins="t")
    for m in fit.margin_fits:
        assert m.params["nu"] == pytest.approx(nu, abs=2.0)
        assert m.params["sigma"] == pytest.approx(np.sqrt((m.params["nu"] - 2) / m.params["nu"]))  # the standardised t
    assert abs(fit.params["theta"] - 1.5) < 3 * fit.se["theta"]
    assert fit.portfolio_var([0.5, 0.5])["var"] > 0
    with pytest.raises(SpecificationError, match="GARCH's own errors"):
        copula(_data(r / 100, ["a", "b"]), ["a", "b"], family="clayton", garch=True, margins="t", dist="normal")


def _quantile_se(x, p):
    q = np.quantile(x, p)
    f = np.mean(np.abs(x - q) < 0.05 * np.std(x)) / (0.1 * np.std(x))  # the density at the quantile
    return np.sqrt(p * (1 - p) / len(x)) / f


def test_portfolio_var_through_the_margins():
    rng = np.random.default_rng(36)
    theta, nus, scales, locs = 2.0, (4.0, 8.0), (0.01, 0.02), (0.0, 0.001)
    w = np.array([0.6, 0.4])
    direct = _clayton_t_sample(1_000_000, theta, nus, scales, locs, rng) @ w
    var_true, es_true = -np.quantile(direct, 0.01), -direct[direct <= np.quantile(direct, 0.01)].mean()
    se = _quantile_se(direct, 0.01)
    fit = copula(_data(_clayton_t_sample(5000, theta, nus, scales, locs, rng)), ["x0", "x1"], family="clayton", margins="t")
    true_margins = tuple(cp.Margin("t", {"mu": locs[j], "sigma": scales[j], "nu": nus[j]}) for j in range(2))
    at_truth = replace(fit, params={**fit.params, "theta": theta}, margin_fits=true_margins)
    got = at_truth.portfolio_var(w, 0.99, n=1_000_000, seed=1)
    assert abs(got["var"] - var_true) < 4 * np.hypot(se, _quantile_se(direct, 0.01))
    assert got["es"] == pytest.approx(es_true, rel=0.01)
    est = fit.portfolio_var(w, 0.99, n=400_000)
    assert est["var"] == pytest.approx(var_true, rel=0.06) and est["es"] == pytest.approx(es_true, rel=0.08)
    assert fit.portfolio_var(w, 0.99, n=400_000, margins="empirical")["var"] == pytest.approx(var_true, rel=0.1)


def test_compare_copulas():
    U = cp.sample_copula("clayton", {"theta": 2.0}, 1500, 2, np.random.default_rng(37))
    data = _data(stats.norm.ppf(U))
    table = compare_copulas(data, ["x0", "x1"])
    frame = table.to_frame()
    assert list(frame.index) == cp.default_families(2, 0.5)
    assert frame["BIC"].idxmin() == "clayton"
    for family in ("t", "clayton", "bb7", "gumbel180"):
        single = copula(data, ["x0", "x1"], family=family)
        assert frame.loc[family, "log likelihood"] == pytest.approx(single.loglike)
        assert frame.loc[family, "AIC"] == pytest.approx(single.aic)
    small = compare_copulas(data, ["x0", "x1"], families=["clayton", "frank"], gof=True, n_boot=30).to_frame()
    assert small.loc["clayton", "p-value"] > small.loc["frank", "p-value"]
    negative = compare_copulas(_data(-stats.norm.ppf(U) * [1, -1]), ["x0", "x1"]).to_frame()
    assert "clayton90" in negative.index and "clayton" not in negative.index
    assert negative["AIC"].idxmin() in ("clayton90", "clayton270")
    three = compare_copulas(_data(stats.norm.ppf(cp.sample_copula("frank", {"theta": 4.0}, 600, 3, np.random.default_rng(38)))),
                            ["x0", "x1", "x2"], method="itau").to_frame()
    assert list(three.index) == ["gaussian", "t", "clayton", "gumbel", "frank", "joe"]


def test_errors():
    U = cp.sample_copula("clayton", {"theta": 2.0}, 300, 3, np.random.default_rng(39))
    data = _data(stats.norm.ppf(U))
    with pytest.raises(SpecificationError, match="two parameters"):
        copula(data, ["x0", "x1"], family="bb1", method="itau")
    with pytest.raises(SpecificationError, match="parametric margins"):
        copula(data, ["x0", "x1"], family="clayton", method="joint")
    with pytest.raises(SpecificationError, match="two series"):
        copula(data, ["x0", "x1", "x2"], family="gumbel180")
    with pytest.raises(SpecificationError, match="margins is one of"):
        copula(data, ["x0", "x1"], margins="kernel")
    with pytest.raises(SpecificationError, match="unknown copula families"):
        compare_copulas(data, ["x0", "x1"], families=["clayton", "plackett"])


def test_the_dialogs_replay():
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS

    U = cp.sample_copula("frank", {"theta": 4.0}, 500, 2, np.random.default_rng(40))
    data = _data(stats.norm.ppf(U) * 0.01, ["a", "b"])
    s = Session()
    s.objects["data"] = data
    s.run(COMMANDS["FitCopula"].cls("a, b", family="frank", margins="t", method="itau", gof=True))
    code = s.history[-1].code
    assert "margins='t'" in code and "method='itau'" in code and "gof=True" in code
    namespace = dict(s.namespace) | {"data": data}
    exec("from pymetrix_finance import *\n" + code, namespace)
    fit = s.objects["copula_fit"]
    assert namespace["copula_fit"].params["theta"] == pytest.approx(fit.params["theta"])
    assert namespace["copula_fit"].gof_result == fit.gof_result
    s.run(COMMANDS["CompareCopulas"].cls("a, b", families="clayton, frank, bb1", margins="normal"))
    table = s.objects["copula_comparison"]
    assert list(table.index) == ["clayton", "frank", "bb1"]
    namespace = dict(s.namespace) | {"data": data}
    exec("from pymetrix_finance import *\n" + s.history[-1].code, namespace)
    assert namespace["copula_comparison"].to_frame()["AIC"].tolist() == pytest.approx(table.to_frame()["AIC"].tolist())
    s.run(COMMANDS["FitCopula"].cls("a, b", family="t", garch=True, margins="t"))
    assert "dist" not in s.history[-1].code and s.objects["copula_fit"].margin_fits[0].kind == "t"
