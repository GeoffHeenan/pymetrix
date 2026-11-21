"""Multivariate GARCH (the finance plugin), checked independently: known parameters recovered from simulated DCC and
BEKK processes; the DCC likelihood against a plain loop; DCC with a = b = 0 is CCC; forecasts approach their
long-run values; the portfolio VaR is the normal VaR of w'Hw; every conditional covariance is positive definite."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset
from pymetrix_finance import FittedMGARCH, mgarch
import importlib

mg = importlib.import_module("pymetrix_finance.mgarch")  # the module (the package exports its function under the same name)


def _data(r, names):
    idx = pd.period_range("2001-01-01", periods=len(r), freq="D", name="time")
    return Dataset.from_frame(pd.DataFrame(r, columns=names, index=idx))


def _simulate_dcc(T, a, b, rho, seed):
    """Two GARCH(1,1) series whose standardised surprises have DCC correlation (a, b) around rho."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = np.array([1e-6, 2e-6]), np.array([0.08, 0.06]), np.array([0.9, 0.92])
    Qbar = np.array([[1.0, rho], [rho, 1.0]])
    Q, h = Qbar.copy(), omega / (1 - alpha - beta)
    r = np.zeros((T, 2))
    for t in range(T):
        R = mg._scale_to_unit(Q)
        z = np.linalg.cholesky(R) @ rng.standard_normal(2)
        r[t] = np.sqrt(h) * z
        h = omega + alpha * r[t] ** 2 + beta * h
        Q = (1 - a - b) * Qbar + a * np.outer(z, z) + b * Q
    return r


@pytest.fixture(scope="module")
def dcc_data():
    return _data(_simulate_dcc(3000, 0.05, 0.90, 0.5, 3), ["x", "y"])


def test_dcc_recovers_its_parameters(dcc_data):
    fit = mgarch(dcc_data, ["x", "y"], model="dcc")
    v = dict(zip(fit.names, fit.values))
    assert v["dcc a"] == pytest.approx(0.05, abs=0.02) and v["dcc b"] == pytest.approx(0.90, abs=0.05)
    assert fit.correlations.iloc[:, 0].mean() == pytest.approx(0.5, abs=0.08)
    assert np.all(np.linalg.eigvalsh(fit.H) > 0)  # every conditional covariance a covariance
    assert isinstance(fit, FittedMGARCH) and fit.summary().index[-1] == "dcc b"


def test_the_dcc_likelihood_and_the_ccc_special_case():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((300, 3))
    Qbar = z.T @ z / 300
    a, b = 0.04, 0.9
    total, Q = 0.0, Qbar.copy()  # the second-stage likelihood, by a plain loop
    for t in range(300):
        d = np.diag(1 / np.sqrt(np.diag(Q)))
        R = d @ Q @ d
        total += np.log(np.linalg.det(R)) + z[t] @ np.linalg.solve(R, z[t]) - z[t] @ z[t]
        Q = (1 - a - b) * Qbar + a * np.outer(z[t], z[t]) + b * Q
    assert mg.dcc_objective(np.array([a, b]), z, Qbar) == pytest.approx(0.5 * total, rel=1e-10)
    still = mg.dcc_path(z, 0.0, 0.0, Qbar)  # no dynamics: the constant correlation of the residuals
    assert np.allclose(still, mg._scale_to_unit(Qbar)) and np.allclose(still[0], np.corrcoef(z.T), atol=0.05)


def test_ccc_and_the_forecasts(dcc_data):
    ccc = mgarch(dcc_data, ["x", "y"], model="ccc")
    assert np.allclose(ccc.correlations.iloc[:, 0], ccc.correlations.iloc[0, 0])  # constant
    far = ccc.forecast(2000)
    s = ccc.state
    long_var = s["omega"] / (1 - s["alpha"] - s["beta"])
    assert np.diag(far[-1]) == pytest.approx(long_var, rel=1e-3)  # variances approach their long-run level
    dcc = mgarch(dcc_data, ["x", "y"], model="dcc")
    corr = [f[0, 1] / np.sqrt(f[0, 0] * f[1, 1]) for f in dcc.forecast(3000)]
    assert corr[-1] == pytest.approx(mg._scale_to_unit(dcc.state["Qbar"])[0, 1], abs=1e-4)  # and the correlation to Qbar's
    w = np.array([0.6, 0.4])
    out = dcc.portfolio_var(w, level=0.99)
    H = dcc.forecast(1)[0]
    from pymetrix_finance.risk import normal

    assert (out["var"], out["es"]) == pytest.approx(normal(float(w @ dcc.mu), float(np.sqrt(w @ H @ w)), 0.01))


def test_scalar_and_diagonal_bekk():
    rng = np.random.default_rng(5)
    T, a, b = 3000, 0.06, 0.9
    S = np.array([[1.0, 0.4], [0.4, 2.0]]) * 1e-4
    e, H = np.zeros((T, 2)), S.copy()
    for t in range(T):
        e[t] = np.linalg.cholesky(H) @ rng.standard_normal(2)
        H = (1 - a - b) * S + a * np.outer(e[t], e[t]) + b * H
    data = _data(e, ["u", "v"])
    scalar = mgarch(data, ["u", "v"], model="bekk")
    assert scalar.values[0] == pytest.approx(a, abs=0.02) and scalar.values[1] == pytest.approx(b, abs=0.04)
    assert np.all(np.linalg.eigvalsh(scalar.H) > 0)
    far = scalar.forecast(3000)[-1]
    assert far == pytest.approx(scalar.state["S"], rel=1e-3)  # targeted: the forecast returns to the sample covariance
    diag = mgarch(data, ["u", "v"], model="diagonal_bekk")
    assert np.all(diag.values[:2] ** 2 == pytest.approx([a, a], abs=0.03)) and diag.stats["loglike"] >= scalar.stats["loglike"] - 1
    assert list(diag.names) == ["a u", "a v", "b u", "b v"]


def test_refusals_and_the_dialog(dcc_data):
    from pymetrix.errors import SpecificationError

    with pytest.raises(SpecificationError, match="at least two"):
        mgarch(dcc_data, ["x"])
    with pytest.raises(SpecificationError, match="model is one of"):
        mgarch(dcc_data, ["x", "y"], model="bekk_full")
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS

    s = Session()
    s.objects["data"] = dcc_data
    s.run(COMMANDS["FitMGARCH"].cls("x, y", model="ccc"))
    assert s.history[-1].code == "mgarch_fit = mgarch(data, ['x', 'y'], model='ccc')"
    from pymetrix.views import sections

    assert [t for t, _ in sections(s.objects["mgarch_fit"])][:1] == ["Summary"]
