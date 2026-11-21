"""Quantile VARs, checked independently: the linear-programme quantile regression against statsmodels' QuantReg, each
equation (contemporaneous terms included) likewise; the responses as the quantile system's moving average by hand; at
the median on Gaussian data near the OLS VAR's recursive unit-shock responses; and in a location-scale model where
shocks to x widen y's spread, the responses at the 10% and 90% quantiles on opposite sides of the median's."""

import importlib

import numpy as np
import pandas as pd
import pytest

from pymetrix import VAR, Dataset, QuantileVAR
from pymetrix.errors import SpecificationError

qv = importlib.import_module("pymetrix.quantile_var")
sm = pytest.importorskip("statsmodels.api")


@pytest.fixture(scope="module")
def gaussian():
    rng = np.random.default_rng(0)
    y = np.zeros((400, 2))
    for t in range(1, 400):
        y[t] = np.array([[0.5, 0.1], [0.3, 0.4]]) @ y[t - 1] + np.array([[1, 0], [0.5, 1]]) @ rng.normal(size=2)
    return Dataset.from_frame(pd.DataFrame(y, columns=["a", "b"], index=pd.period_range("1900Q1", periods=400, freq="Q"))), y


def test_equations_against_quantreg(gaussian):
    data, y = gaussian
    fit = QuantileVAR(["a", "b"], lags=1, quantile=0.3).fit(data)
    X = np.column_stack([np.ones(399), y[1:, 0], y[:-1]])
    ref = sm.QuantReg(y[1:, 1], X).fit(q=0.3, max_iter=5000, p_tol=1e-10).params
    assert np.r_[fit.intercepts[1], -fit.A0[1, 0], fit.A[0, 1]] == pytest.approx(ref, abs=1e-5)
    ref_a = sm.QuantReg(y[1:, 0], np.column_stack([np.ones(399), y[:-1]])).fit(q=0.3, max_iter=5000, p_tol=1e-10).params
    assert np.r_[fit.intercepts[0], fit.A[0, 0]] == pytest.approx(ref_a, abs=1e-5)
    A0inv = np.linalg.inv(fit.A0)
    R = A0inv @ fit.A[0]
    irf = fit.irf(3).values
    assert irf[2] == pytest.approx(R @ R @ A0inv, abs=1e-12)  # the moving average by hand


def test_the_median_is_near_the_ols_var(gaussian):
    data, _ = gaussian
    med = QuantileVAR(["a", "b"], lags=1, quantile=0.5).fit(data).irf(8).values
    var = VAR(["a", "b"], lags=1).fit(data)
    P = np.linalg.cholesky(var.estimate.sigma)
    ols = var.irf(8, draws=0).values @ np.linalg.inv(P) @ (P / np.diag(P))  # recursive, unit shocks
    assert med == pytest.approx(ols, abs=0.15)
    with pytest.raises(SpecificationError, match="between 0 and 1"):
        QuantileVAR(["a"], quantile=1.2)


def test_quantiles_differ_where_the_spread_responds():
    rng = np.random.default_rng(1)
    T = 1500
    x, y = np.zeros(T), np.zeros(T)
    for t in range(1, T):
        x[t] = 0.5 * x[t - 1] + rng.normal()
        y[t] = 0.3 * y[t - 1] + (1.5 + 0.5 * x[t - 1]) * rng.normal()  # last period's x scales the spread (positive in practice)
    data = Dataset.from_frame(pd.DataFrame({"x": x, "y": y}, index=pd.period_range("1500Q1", periods=T, freq="Q")))
    lo, hi = (QuantileVAR(["x", "y"], lags=1, quantile=q).fit(data).irf(2).values[1, 1, 0] for q in (0.1, 0.9))
    assert lo == pytest.approx(-0.64, abs=0.2) and hi == pytest.approx(0.64, abs=0.2)  # 0.5 z: the 10% quantile falls, the 90% rises
