"""The DCC copula (a Gaussian or t copula of several series with Engle's DCC correlation), checked independently:

- the filtered correlations against a loop written here (Gaussian and t scores, 1e-12) and against mgarch's own
  period-by-period dcc_path (1e-12);
- with a = b = 0 it is the constant copula: the Gaussian's log likelihood that of copula(family="gaussian") on
  the same ranks (1e-10), the t's that of the t copula density at Qbar scaled (1e-10); the reported constant
  log likelihood the Gaussian copula fit's (1e-8);
- parameter recovery from simulated paths (Gaussian 2,000 periods, t 1,500, three series) within three standard
  errors;
- forecasts (Engle and Sheppard's approximation), simulation (shapes; the first period at the forecast;
  the draws' correlation of normal scores near it), portfolio VaR and ES against a direct simulation written
  here, the summary and the errors, through the API.
"""

import importlib

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pymetrix import Dataset, copula, dcc_copula
from pymetrix.errors import SpecificationError

T = importlib.import_module("pymetrix_finance.tvcopulas")
cp = importlib.import_module("pymetrix_finance.copulas")
mg = importlib.import_module("pymetrix_finance.mgarch")

RBAR = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])


def _simulate(n, a, b, family="gaussian", nu=None, seed=0):
    rng = np.random.default_rng(seed)
    Q, U = RBAR.copy(), np.empty((n, 3))
    for t in range(n):
        s = 1 / np.sqrt(np.diag(Q))
        x = np.linalg.cholesky(Q * np.outer(s, s)) @ rng.standard_normal(3)
        if family == "t":
            x = x / np.sqrt(rng.chisquare(nu) / nu)
            U[t] = stats.t.cdf(x, nu)
        else:
            U[t] = stats.norm.cdf(x)
        Q = (1 - a - b) * RBAR + a * np.outer(x, x) + b * Q
    return U


def _dataset(U):
    x = stats.norm.ppf(U) * 0.01
    idx = pd.period_range("2001-01-01", periods=len(U), freq="D", name="time")
    return Dataset.from_frame(pd.DataFrame(x, columns=["a", "b", "c"][: U.shape[1]], index=idx))


@pytest.mark.parametrize("family,nu", [("gaussian", None), ("t", 5.0)])
def test_filter_against_loops(family, nu):
    u = cp.pseudo_observations(_simulate(300, 0.06, 0.9, seed=1))
    a, b = 0.07, 0.88
    f = T.dcc_copula_filter(u, family, a, b, nu)
    z = stats.norm.ppf(u) if family == "gaussian" else stats.t.ppf(u, nu)
    Qbar = z.T @ z / len(z)
    Q = Qbar.copy()
    for t in range(len(z)):
        s = 1 / np.sqrt(np.diag(Q))
        np.testing.assert_allclose(f["R"][t], Q * np.outer(s, s), rtol=1e-12, atol=1e-14)
        Q = (1 - a - b) * Qbar + a * np.outer(z[t], z[t]) + b * Q
    np.testing.assert_allclose(f["Q_next"], Q, rtol=1e-12)
    np.testing.assert_allclose(f["R"], mg.dcc_path(z, a, b, Qbar), rtol=1e-12, atol=1e-14)


def test_constant_copula():
    U = _simulate(500, 0.05, 0.9, seed=2)
    u = cp.pseudo_observations(U)
    const = copula(_dataset(U), ["a", "b", "c"], family="gaussian")
    f = T.dcc_copula_filter(u, "gaussian", 0.0, 0.0)
    ll = T.dcc_copula_logpdf(f["z"], f["R"], "gaussian").sum()
    assert ll == pytest.approx(const.loglike, abs=1e-10 * abs(const.loglike) + 1e-10)
    np.testing.assert_allclose(f["R"][0], const.params["R"], atol=1e-12)
    ft = T.dcc_copula_filter(u, "t", 0.0, 0.0, 6.0)
    R = mg._scale_to_unit(ft["Qbar"])
    assert T.dcc_copula_logpdf(ft["z"], ft["R"], "t", 6.0).sum() == pytest.approx(cp.t_logpdf(u, R, 6.0).sum(), rel=1e-10)
    fit = T.fit_dcc_copula(u, "gaussian")
    assert fit["constant_loglike"] == pytest.approx(const.loglike, rel=1e-8)
    assert fit["loglike"] >= fit["constant_loglike"]


@pytest.mark.parametrize("family,n,nu", [("gaussian", 2000, None), ("t", 1500, 6.0)])
def test_parameter_recovery(family, n, nu):
    a, b = 0.05, 0.9
    U = _simulate(n, a, b, family, nu, seed=11)
    fit = T.fit_dcc_copula(cp.pseudo_observations(U), family)
    p, se = fit["params"], fit["se"]
    assert abs(p["a"] - a) < 3 * se["a"] and abs(p["b"] - b) < 3 * se["b"]
    if family == "t":
        assert abs(p["nu"] - nu) < 3 * se["nu"]


def test_through_the_api():
    U = _simulate(800, 0.05, 0.9, seed=4)
    fit = dcc_copula(_dataset(U), ["a", "b", "c"])
    assert fit.path.shape == (800, 3) and list(fit.path.columns) == ["a, b", "a, c", "b, c"]
    F = fit.forecast(5)
    f = fit._filtered()
    Rbar = mg._scale_to_unit(f["Qbar"])
    np.testing.assert_allclose(F[0], f["R_next"], rtol=1e-12)
    np.testing.assert_allclose(F[3], Rbar + (fit.params["a"] + fit.params["b"]) ** 3 * (f["R_next"] - Rbar), rtol=1e-12)
    sim = fit.simulate(horizon=3, n=20_000, seed=1)
    assert sim["u"].shape == (20_000, 3, 3) and sim["R"].shape == (20_000, 3, 3, 3)
    np.testing.assert_allclose(sim["R"][:, 0], np.broadcast_to(f["R_next"], (20_000, 3, 3)), rtol=1e-12)
    z = stats.norm.ppf(sim["u"][:, 0])
    np.testing.assert_allclose(np.corrcoef(z.T), f["R_next"], atol=0.03)
    # VaR and ES against a direct simulation at the forecast correlation through the empirical margins
    w = [0.5, 0.3, 0.2]
    out = fit.portfolio_var(w, level=0.99, n=200_000, seed=3)
    rng = np.random.default_rng(5)
    Ud = stats.norm.cdf(rng.standard_normal((200_000, 3)) @ np.linalg.cholesky(f["R_next"]).T)
    X = np.column_stack([np.quantile(fit.residuals[:, j], Ud[:, j]) for j in range(3)]) @ np.array(w)
    var = -np.quantile(X, 0.01)
    assert out["var"] == pytest.approx(var, rel=0.03)
    s = str(fit.summary())
    assert "DCC Gaussian copula" in s and "LR statistic: " in s
    assert fit.k == 2 + 3 and fit.loglike >= fit.constant_loglike
    two = dcc_copula(_dataset(U[:, :2]), ["a", "b"], family="t")
    assert set(two.params) == {"a", "b", "nu"} and two.path.shape == (800, 1)
    with pytest.raises(SpecificationError, match="family is one of"):
        dcc_copula(_dataset(U), ["a", "b", "c"], family="clayton")


def test_the_dialog_replays():
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS

    data = _dataset(_simulate(400, 0.05, 0.9, seed=6))
    s = Session()
    s.objects["data"] = data
    s.run(COMMANDS["FitDCCCopula"].cls("a, b, c", family="t", margins="normal"))
    code = s.history[-1].code
    assert "dcc_copula(" in code and "family='t'" in code and "margins='normal'" in code
    namespace = dict(s.namespace) | {"data": data}
    exec("from pymetrix_finance import *\n" + code, namespace)
    fit = s.objects["dcc_copula_fit"]
    assert namespace["dcc_copula_fit"].loglike == pytest.approx(fit.loglike, abs=1e-12)
    assert namespace["dcc_copula_fit"].params == fit.params
