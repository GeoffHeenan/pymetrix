"""Ng-Perron, ERS point-optimal and Perron break tests, checked independently: the GLS detrending through arch's DF-GLS
statistic (the ADF t on our detrended series equal to arch's); the M statistics by their formulas from the detrended
series and the long-run variance; the M and PT tests' size under a random walk at the tabulated 5% critical values
and their power against a stationary AR(1); and Perron's simulated critical values against his 1989 tables
(lambda = 0.5, T = 100), the statistic as the t ratio from statsmodels OLS."""

import importlib

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset
from pymetrix.errors import SpecificationError

ur = importlib.import_module("pymetrix.stats.unit_root_extra")


def _dataset(y, start="1950Q1"):
    return Dataset.from_frame(pd.DataFrame({"y": y}, index=pd.period_range(start, periods=len(y), freq="Q")))


def _walk(T=300, seed=0, drift=0.05):
    return np.cumsum(np.random.default_rng(seed).normal(size=T)) + drift * np.arange(T)


@pytest.mark.parametrize("trend", ["c", "ct"])
def test_gls_detrending_through_arch_dfgls(trend):
    arch = pytest.importorskip("arch.unitroot")
    y = _walk()
    yd, _ = ur.gls_detrend(y, trend)
    X, dy = ur._adf_design(yd, 3, 3)
    b = np.linalg.lstsq(X, dy, rcond=None)[0]
    e = dy - X @ b
    t = b[0] / np.sqrt(e @ e / (len(e) - X.shape[1]) * np.linalg.inv(X.T @ X)[0, 0])
    assert t == pytest.approx(arch.DFGLS(y, lags=3, trend=trend).stat, rel=1e-10)


def test_the_m_statistics_by_their_formulas():
    y = _walk(seed=1)
    res = _dataset(y).test("ng_perron", "y", trend="ct", lags=2)
    yd, _ = ur.gls_detrend(y, "ct")
    T = len(y)
    dx = np.diff(yd)
    X = np.column_stack([yd[2:-1], dx[1:-1], dx[:-2]])
    b = np.linalg.lstsq(X, dx[2:], rcond=None)[0]
    e = dx[2:] - X @ b
    s2 = (e @ e / len(e)) / (1 - b[1:].sum()) ** 2
    kappa = np.sum(yd[:-1] ** 2) / T**2
    MZa = (yd[-1] ** 2 / T - s2) / (2 * kappa)
    MSB = np.sqrt(kappa / s2)
    assert res.statistic == pytest.approx(MZa, rel=1e-10) and res.details["MSB"] == pytest.approx(MSB, rel=1e-10)
    assert res.details["MZt"] == pytest.approx(MZa * MSB, rel=1e-10)
    assert res.details["MPT"] == pytest.approx((13.5**2 * kappa + 14.5 * yd[-1] ** 2 / T) / s2, rel=1e-10)
    assert res.critical_values["5%"] == -17.3


@pytest.mark.parametrize("trend", ["c", "ct"])
def test_size_and_power(trend):
    rng = np.random.default_rng(7)
    R = 300
    size = {"MZa": 0, "MZt": 0, "MSB": 0, "MPT": 0, "PT": 0}
    power = 0
    for _ in range(R):
        x = np.cumsum(rng.normal(size=400))
        s = ur.m_statistics(x, trend)
        crit = ur.NG_PERRON_CRITICAL[trend]
        for k in ("MZa", "MZt", "MSB", "MPT"):
            size[k] += s[k] < crit[k][1]
        size["PT"] += _dataset(x).test("ers_point_optimal", "y", trend=trend).statistic < ur.ERS_CRITICAL[trend][1]
    for _ in range(100):
        e = rng.normal(size=400)
        x = np.zeros(400)
        for t in range(1, 400):
            x[t] = 0.9 * x[t - 1] + e[t]
        power += ur.m_statistics(x, trend)["MZa"] < ur.NG_PERRON_CRITICAL[trend]["MZa"][1]
    for k, v in size.items():
        assert v / R == pytest.approx(0.05, abs=0.035), k
    assert power / 100 > 0.6


def test_perron_against_his_tables_and_ols():
    import statsmodels.api as sm

    tables = {"A": -3.76, "B": -3.96, "C": -4.24}  # Perron (1989), 5%, lambda = 0.5
    for model, value in tables.items():
        sims = ur.perron_simulated(100, 49, model, 0, reps=3000, seed=1)
        assert np.quantile(sims, 0.05) == pytest.approx(value, abs=0.12), model
    rng = np.random.default_rng(3)
    t = np.arange(1, 121)
    y = 0.5 * t / 10 + 3.0 * (t > 60) + np.cumsum(rng.normal(size=120)) * 0.3 + rng.normal(size=120)
    data = _dataset(y)
    res = data.test("perron_break", "y", date=str(data.frame.index[60]), model="A", lags=1, reps=500)
    dy = np.diff(y)
    DU, DTB = (t > 60).astype(float), (t == 61).astype(float)
    X = np.column_stack([np.ones(118), t[2:], DU[2:], DTB[2:], y[1:-1], dy[:-1]])
    ols = sm.OLS(dy[1:], X).fit()
    assert res.statistic == pytest.approx(ols.tvalues[4], rel=1e-10) and res.details["break fraction"] == pytest.approx(0.5)
    assert res.p_value < 0.05  # trend stationary with a level shift


def test_refusals():
    data = _dataset(_walk())
    with pytest.raises(SpecificationError, match="date"):
        data.test("perron_break", "y")
    with pytest.raises(SpecificationError, match="not in the sample"):
        data.test("perron_break", "y", date="2200Q1")
