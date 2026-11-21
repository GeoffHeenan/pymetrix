"""Student-t and GJR GARCH in value at risk, and asymmetric DCC (the finance plugin), checked independently: the
GJR-t GARCH's next VaR and ES against the arch package's fit and forecast; the GJR variance update by hand; the
standardised t's quantile and tail mean by simulation; asymmetric DCC with no asymmetry equal to DCC, its
stationarity bound against the eigenvalues of Qbar^-1 Nbar, and its asymmetry recovered from a simulated process."""

import importlib

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pymetrix import Dataset, mgarch, value_at_risk
from pymetrix.errors import SpecificationError

risk = importlib.import_module("pymetrix_finance.risk")
mg = importlib.import_module("pymetrix_finance.mgarch")


def _gjr_t(T=2500, seed=0, omega=0.05, alpha=0.04, gamma=0.1, beta=0.88, nu=6.0):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, T) * np.sqrt((nu - 2) / nu)
    r, h = np.zeros(T), omega / (1 - alpha - gamma / 2 - beta)
    for t in range(T):
        r[t] = 0.02 + np.sqrt(h) * z[t]
        e = r[t] - 0.02
        h = omega + (alpha + gamma * (e < 0)) * e**2 + beta * h
    return pd.Series(r, index=pd.period_range("2000-01-01", periods=T, freq="D", name="time"), name="r")


def test_gjr_t_against_arch():
    arch = pytest.importorskip("arch")
    r = _gjr_t()
    ours = risk._fit_garch(r, dist="t", asymmetric=True)
    ref = arch.arch_model(r.to_numpy(), mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
    pv = ref.params
    assert (ours.alpha, ours.gamma, ours.beta, ours.nu) == pytest.approx((pv["alpha[1]"], pv["gamma[1]"], pv["beta[1]"], pv["nu"]), rel=0.05, abs=0.01)
    var_next = float(ref.forecast(horizon=1).variance.iloc[-1, 0])
    assert ours.var_next == pytest.approx(var_next, rel=0.03)
    q = stats.t.ppf(0.01, pv["nu"]) * np.sqrt((pv["nu"] - 2) / pv["nu"])  # arch's standardised t quantile
    assert ours.var_es(0.01)[0] == pytest.approx(-(pv["mu"] + np.sqrt(var_next) * q), rel=0.03)
    assert (ours.alpha, ours.gamma, ours.beta) == pytest.approx((0.04, 0.1, 0.88), abs=0.04)


def test_the_gjr_step_and_the_standardised_t():
    g = risk._Garch(mu=0.0, omega=0.1, alpha=0.05, beta=0.9, var_next=1.0, z=np.zeros(0), gamma=0.1, nu=5.0)
    g.step(-2.0)
    assert g.var_next == pytest.approx(0.1 + 0.15 * 4 + 0.9)
    g.step(2.0)
    assert g.var_next == pytest.approx(0.1 + 0.05 * 4 + 0.9 * 1.6)
    draws = np.random.default_rng(0).standard_t(5, 4_000_000) * np.sqrt(3 / 5) * np.sqrt(g.var_next)
    v, e = g.var_es(0.01)
    q = np.quantile(draws, 0.01)
    assert v == pytest.approx(-q, rel=0.01) and e == pytest.approx(-draws[draws <= q].mean(), rel=0.01)


def test_fat_tails_and_asymmetry_reduce_the_violations():
    ff = Dataset.example("french")  # in the tests, the simulated stand-in: a GJR-GARCH market with t(4) errors
    normal = value_at_risk(ff, "mkt_rf", level=0.99, method="garch", window=120)
    fat = value_at_risk(ff, "mkt_rf", level=0.99, method="garch", window=120, dist="t", asymmetric=True)
    assert fat.violations < normal.violations and fat.method == "garch (GJR, t errors)"
    assert (fat.frame["var"] > normal.frame["var"]).mean() > 0.5  # the t's tail puts the 99% VaR further out, mostly
    with pytest.raises(SpecificationError, match="GARCH methods"):
        value_at_risk(ff, "mkt_rf", method="normal", dist="t")


def test_asymmetric_dcc():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((400, 3))
    Qbar = z.T @ z / 400
    n = np.minimum(z, 0)
    Nbar = n.T @ n / 400
    assert mg.dcc_objective(np.array([0.04, 0.9, 0.0]), z, Qbar, Nbar) == pytest.approx(mg.dcc_objective(np.array([0.04, 0.9]), z, Qbar))
    assert mg.adcc_bound(Qbar, Nbar) == pytest.approx(np.max(np.linalg.eigvals(np.linalg.solve(Qbar, Nbar)).real))
    # a process whose correlation rises after joint falls
    a, b, g = 0.03, 0.9, 0.06
    T = 4000
    Qbar = np.array([[1.0, 0.3], [0.3, 1.0]])
    big = rng.standard_normal((1_000_000, 2)) @ np.linalg.cholesky(Qbar).T
    Nbar = np.minimum(big, 0).T @ np.minimum(big, 0) / len(big)  # E[n n'] when the correlation is Qbar's
    assert np.all(np.linalg.eigvalsh((1 - a - b) * Qbar - g * Nbar) > 0) and a + b + mg.adcc_bound(Qbar, Nbar) * g < 1
    Q, Z = Qbar.copy(), np.zeros((T, 2))
    for t in range(T):
        Z[t] = np.linalg.cholesky(mg._scale_to_unit(Q)) @ rng.standard_normal(2)
        Q = mg.dcc_step(Q, Z[t], a, b, Qbar, g, Nbar)
    data = Dataset.from_frame(pd.DataFrame(Z * 0.01, columns=["x", "y"], index=pd.period_range("2000-01-01", periods=T, freq="D", name="time")))
    fit = mgarch(data, ["x", "y"], model="adcc")
    v = dict(zip(fit.names, fit.values))
    assert v["adcc g"] == pytest.approx(g, abs=0.06) and v["adcc b"] == pytest.approx(b, abs=0.06)
    assert np.all(np.linalg.eigvalsh(fit.H) > 0) and len(fit.forecast(5)) == 5
    gjr = mgarch(data, ["x", "y"], model="dcc", asymmetric=True)
    assert "x: gamma" in gjr.names


def test_the_dialogs_pass_the_options():
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS

    s = Session()
    s.objects["data"] = Dataset.from_frame(_gjr_t(T=600).to_frame())
    s.run(COMMANDS["ValueAtRisk"].cls("r", method="garch", window=400, refit=100, dist="t"))
    assert "dist='t'" in s.history[-1].code and s.objects["risk"].method == "garch (t errors)"
    s.run(COMMANDS["ValueAtRisk"].cls("r", method="normal", dist="t"))  # ignored where it does not apply
    assert "dist" not in s.history[-1].code
