"""Copulas (the finance plugin), checked independently: each copula density integrates to one (numerical
integration) and matches scipy's multivariate densities; parameters recovered from simulated copulas, and the samplers
against Kendall's tau; the t copula's tail dependence against simulation; and the Gaussian copula with normal margins
giving the closed-form normal portfolio VaR."""

import importlib

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from pymetrix import Dataset, copula
from pymetrix.errors import SpecificationError

cp = importlib.import_module("pymetrix_finance.copulas")


def test_the_densities():
    for logpdf in (lambda u: cp.clayton_logpdf(u, 2.0), lambda u: cp.gumbel_logpdf(u, 1.8),
                   lambda u: cp.t_logpdf(u, np.array([[1, 0.5], [0.5, 1]]), 4.0)):
        total = integrate.dblquad(lambda b, a: np.exp(logpdf(np.array([[a, b]])))[0], 1e-9, 1 - 1e-9, 1e-9, 1 - 1e-9, epsabs=1e-7)[0]
        assert total == pytest.approx(1.0, abs=2e-3)
    R = np.array([[1, 0.3, -0.2], [0.3, 1, 0.4], [-0.2, 0.4, 1]])
    u = np.random.default_rng(0).uniform(0.05, 0.95, (5, 3))
    z, x = stats.norm.ppf(u), stats.t.ppf(u, 6)
    assert cp.gaussian_logpdf(u, R) == pytest.approx(stats.multivariate_normal(cov=R).logpdf(z) - stats.norm.logpdf(z).sum(axis=1))
    assert cp.t_logpdf(u, R, 6) == pytest.approx(stats.multivariate_t(shape=R, df=6).logpdf(x) - stats.t.logpdf(x, 6).sum(axis=1))


def _uniform_data(U):
    idx = pd.period_range("2000-01-01", periods=len(U), freq="D", name="time")
    return Dataset.from_frame(pd.DataFrame(stats.norm.ppf(U) * 0.01, columns=[f"x{i}" for i in range(U.shape[1])], index=idx))


def test_parameters_recovered():
    rng = np.random.default_rng(1)
    R = np.array([[1, 0.6, 0.3], [0.6, 1, 0.5], [0.3, 0.5, 1]])
    t_fit = copula(_uniform_data(cp.sample_copula("t", {"R": R, "nu": 5.0}, 4000, 3, rng)), ["x0", "x1", "x2"], family="t")
    assert t_fit.params["R"] == pytest.approx(R, abs=0.04) and t_fit.params["nu"] == pytest.approx(5.0, abs=1.5)
    g_fit = copula(_uniform_data(cp.sample_copula("gaussian", {"R": R}, 4000, 3, rng)), ["x0", "x1", "x2"], family="gaussian")
    assert g_fit.params["R"] == pytest.approx(R, abs=0.04)
    for family, theta, tau in (("clayton", 2.0, 2.0 / 4.0), ("gumbel", 2.0, 0.5)):
        U = cp.sample_copula(family, {"theta": theta}, 4000, 2, rng)
        assert stats.kendalltau(U[:, 0], U[:, 1])[0] == pytest.approx(tau, abs=0.03)  # the sampler has the right tau
        fit = copula(_uniform_data(U), ["x0", "x1"], family=family)
        assert fit.params["theta"] == pytest.approx(theta, rel=0.1)
    with pytest.raises(SpecificationError, match="two series"):
        copula(_uniform_data(cp.sample_copula("gaussian", {"R": R}, 200, 3, rng)), ["x0", "x1", "x2"], family="bb1")


def test_tail_dependence_by_simulation():
    params = {"R": np.array([[1, 0.5], [0.5, 1]]), "nu": 3.0}
    lam = cp.tail_dependence("t", params)["lower"][0, 1]
    U = cp.sample_copula("t", params, 2_000_000, 2, np.random.default_rng(2))
    q = 0.002
    empirical = np.mean((U[:, 0] < q) & (U[:, 1] < q)) / q
    assert empirical == pytest.approx(lam, abs=0.04)
    assert cp.tail_dependence("clayton", {"theta": 2.0})["lower"] == pytest.approx(2 ** -0.5)


def test_gaussian_copula_portfolio_var_is_the_normal_var():
    from pymetrix_finance.risk import normal

    rng = np.random.default_rng(3)
    C = np.array([[1.0, 0.4], [0.4, 2.0]]) * 1e-4
    r = rng.multivariate_normal([0, 0], C, 3000)
    data = Dataset.from_frame(pd.DataFrame(r, columns=["a", "b"], index=pd.period_range("2000-01-01", periods=3000, freq="D", name="time")))
    fit = copula(data, ["a", "b"], family="gaussian")
    w = np.array([0.6, 0.4])
    got = fit.portfolio_var(w, 0.99, n=400_000, margins="normal")
    m, s = r.mean(axis=0), r.std(axis=0, ddof=1)
    Rz = fit.params["R"]
    S = np.outer(s, s) * Rz
    assert (got["var"], got["es"]) == pytest.approx(normal(float(w @ m), float(np.sqrt(w @ S @ w)), 0.01), rel=0.01)


def test_the_dialog_and_garch_margins():
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS

    rng = np.random.default_rng(4)
    r = rng.multivariate_normal([0, 0], [[1e-4, 5e-5], [5e-5, 1e-4]], 600)
    s = Session()
    s.objects["data"] = Dataset.from_frame(pd.DataFrame(r, columns=["a", "b"], index=pd.period_range("2000-01-01", periods=600, freq="D", name="time")))
    s.run(COMMANDS["FitCopula"].cls("a, b", family="clayton", garch=True, dist="t"))
    fit = s.objects["copula_fit"]
    assert fit.garch and fit.family == "clayton" and np.all(fit.scale_next > 0)
    assert "garch=True" in s.history[-1].code
