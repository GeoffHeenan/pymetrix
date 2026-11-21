"""Term structure (the finance plugin), checked independently: forwards that average to the yield (numerical
integration); discount factors and zero-coupon bonds by their definitions; a bond at the par coupon priced at 100; the
duration and convexity of a zero; the Fisher-Weil duration against a finite difference of the price; the dynamic
Nelson-Siegel likelihood against statsmodels' Kalman filter, with missing yields; and its parameters recovered from a
simulated model."""

import importlib

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

from pymetrix import Dataset, dynamic_nelson_siegel, yield_curve

ts = importlib.import_module("pymetrix_finance.termstructure")
yl = importlib.import_module("pymetrix_finance.yields")

NAMES = ["m3", "m6", "y1", "y2", "y3", "y5", "y7", "y10", "y20", "y30"]
YEARS = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])


def _simulate(T=300, lam=0.6, seed=1):
    rng = np.random.default_rng(seed)
    mu, A, sd = np.array([5.0, -1.5, 0.5]), np.diag([0.97, 0.93, 0.8]), np.array([0.15, 0.25, 0.5])
    F = np.zeros((T, 3))
    F[0] = mu
    for t in range(1, T):
        F[t] = mu + A @ (F[t - 1] - mu) + rng.normal(0, sd)
    Y = F @ yl.loadings(YEARS, lam).T + rng.normal(0, 0.05, (T, len(YEARS)))
    idx = pd.period_range("1990-01", periods=T, freq="M", name="time")
    return Dataset.from_frame(pd.DataFrame(Y, columns=NAMES, index=idx)), F


def test_forwards_discounts_and_bonds():
    data, _ = _simulate()
    yc = yield_curve(data, NAMES, YEARS, model="svensson", lam=0.9, lam2=0.2)
    for m in (0.5, 4.0, 17.0):  # the yield is the average forward rate from 0 to m
        avg = integrate.quad(lambda s: ts.forwards(yc, [s]).iloc[0], 0, m)[0] / m
        assert avg == pytest.approx(yc.curve(maturities=[m]).iloc[0], rel=1e-9)
    y5 = yc.curve(maturities=[5]).iloc[0] / 100
    assert ts.discount(yc, [5]).iloc[0] == pytest.approx(np.exp(-5 * y5))
    zero = ts.bond(yc, coupon=0, maturity=5, freq=1)
    assert zero["price"] == pytest.approx(100 * np.exp(-5 * y5)) and zero["yield"] == pytest.approx(y5)
    assert zero["duration"] == pytest.approx(5) and zero["convexity"] == pytest.approx(25)
    c = ts.par_coupon(yc, 10, freq=2)
    assert ts.bond(yc, c, 10)["price"] == pytest.approx(100, abs=1e-9)


def test_the_fisher_weil_duration():
    times, cf = ts._cash_flows(5.0, 7.0, 2)
    zeros = 0.03 + 0.002 * times
    b = ts.bond_measures(times, cf, zeros)
    h = 1e-6
    up, down = ts.bond_measures(times, cf, zeros + h)["price"], ts.bond_measures(times, cf, zeros - h)["price"]
    assert b["Fisher-Weil duration"] == pytest.approx(-(up - down) / (2 * h * b["price"]), rel=1e-6)
    flat = ts.bond_measures(times, cf, np.full_like(times, 0.04))  # on a flat curve the two durations agree
    assert flat["duration"] == pytest.approx(flat["Fisher-Weil duration"]) and flat["yield"] == pytest.approx(0.04)


def test_the_dns_likelihood_against_statsmodels():
    from statsmodels.tsa.statespace.mlemodel import MLEModel

    data, _ = _simulate(T=120)
    Y = data.frame[NAMES].to_numpy(float).copy()
    Y[5:20, 8] = np.nan  # a maturity missing for a while
    Y[40, :3] = np.nan
    rng = np.random.default_rng(0)
    theta = np.concatenate([[np.log(0.55)], [5, -1.4, 0.3], np.arctanh([0.95, 0.9, 0.7]), rng.normal(0, 0.1, 6) - [1, 0, 1, 0, 0, 1][:6],
                            np.log(np.full(10, 0.06))])
    ours = ts.dns_loglike(theta, Y, YEARS, NAMES, full=False)
    parts = ts._unpack(theta, 3, 10, False)
    Z = yl.loadings(YEARS, parts["lam"])
    mod = MLEModel(Y, k_states=3)
    mod["design"], mod["obs_intercept"], mod["obs_cov"] = Z, Z @ parts["mu"], parts["H"]
    mod["transition"], mod["selection"], mod["state_cov"] = parts["A"], np.eye(3), parts["Q"]
    mod.ssm.initialize_stationary()
    assert ours == pytest.approx(mod.ssm.loglike(), rel=1e-9)


def test_dns_recovers_its_parameters():
    data, F = _simulate(T=300)
    fit = dynamic_nelson_siegel(data, NAMES, YEARS)
    assert fit.lam == pytest.approx(0.6, abs=0.03) and fit.converged
    assert fit.mu == pytest.approx([5.0, -1.5, 0.5], abs=0.6)
    assert np.diag(fit.A) == pytest.approx([0.97, 0.93, 0.8], abs=0.08)
    assert np.sqrt(np.diag(fit.Q)) == pytest.approx([0.15, 0.25, 0.5], rel=0.2)
    assert np.sqrt(np.diag(fit.H)) == pytest.approx(np.full(10, 0.05), rel=0.25)
    assert np.abs(fit.factors.to_numpy() - F).mean(axis=0) == pytest.approx([0, 0, 0], abs=0.2)
    f1 = fit.mu + fit.A @ (fit.filtered.iloc[-1].to_numpy() - fit.mu)  # one step ahead by hand
    assert fit.forecast(2).iloc[0].to_numpy() == pytest.approx(yl.loadings(YEARS, fit.lam) @ f1)
    assert fit.summary().index[0] == "decay"
    yc = yield_curve(data, NAMES, YEARS)  # the curve's own methods are the module's functions
    assert yc.bond(4, 10) == ts.bond(yc, 4, 10) and yc.discount([5]).equals(ts.discount(yc, [5]))
    assert yc.par_coupon(7) == pytest.approx(ts.par_coupon(yc, 7)) and yc.forwards([3]).equals(ts.forwards(yc, [3]))


def test_the_dialog():
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS

    data, _ = _simulate(T=80)
    s = Session()
    s.objects["data"] = data
    s.run(COMMANDS["FitDynamicNelsonSiegel"].cls(", ".join(NAMES), ", ".join(f"{m:g}" for m in YEARS)))
    assert s.history[-1].code.startswith("dns = dynamic_nelson_siegel(data, ['m3'")
