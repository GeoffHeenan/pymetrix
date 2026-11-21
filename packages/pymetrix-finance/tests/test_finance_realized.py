"""Realised volatility and HAR (the finance plugin), checked independently: on a simulated diffusion with known
daily integrated variance, realised variance and bipower variation estimate it, bipower variation removes most of a jump while
realised variance adds its square, and the jump test rejects about as often as it should without jumps and finds
large ones; each measure against its formula by a plain loop; the HAR against least squares on hand-made averages,
and its forecasts against the iteration by hand."""

import importlib
from math import gamma, pi

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, har, realized_measures
from pymetrix.errors import SpecificationError

rz = importlib.import_module("pymetrix_finance.realized")


def _intraday(days=300, n=78, jump_days=(), jump=0.0, seed=0):
    rng = np.random.default_rng(seed)
    lv = np.zeros(days)
    lv[0] = np.log(1e-4)
    for t in range(1, days):
        lv[t] = 0.05 * np.log(1e-4) + 0.95 * lv[t - 1] + rng.normal(0, 0.2)
    iv = np.exp(lv)
    r = rng.normal(0, 1, (days, n)) * np.sqrt(iv[:, None] / n)
    for d in jump_days:
        r[d, n // 2] += jump
    start = pd.bdate_range("2020-01-02", periods=days)
    idx = pd.PeriodIndex([pd.Period(d + pd.Timedelta(minutes=570 + 5 * j), "5min") for d in start for j in range(n)], name="time")
    return Dataset.from_frame(pd.DataFrame({"r": r.ravel()}, index=idx)), iv, r


def test_each_measure_by_its_formula():
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.001, 50)
    m = rz.day_measures(r)
    n = len(r)
    rv = sum(x * x for x in r)
    bv = pi / 2 * n / (n - 1) * sum(abs(r[j]) * abs(r[j - 1]) for j in range(1, n))
    mu = 2 ** (2 / 3) * gamma(7 / 6) / gamma(0.5)
    tq = n * mu**-3 * n / (n - 2) * sum((abs(r[j]) * abs(r[j - 1]) * abs(r[j - 2])) ** (4 / 3) for j in range(2, n))
    z = ((rv - bv) / rv) / np.sqrt((pi**2 / 4 + pi - 5) * max(1, tq / bv**2) / n)
    assert (m["rv"], m["bv"], m["tq"], m["jump z"]) == pytest.approx((rv, bv, tq, z), rel=1e-12)
    assert mu == pytest.approx(np.mean(np.abs(np.random.default_rng(0).normal(size=2_000_000)) ** (4 / 3)), rel=2e-3)


def test_realised_measures_estimate_the_integrated_variance():
    jump_days = list(range(10, 300, 20))
    data, iv, r = _intraday(jump_days=jump_days, jump=0.03)
    daily = realized_measures(data, "r").frame
    calm = np.setdiff1d(np.arange(300), jump_days)
    assert np.mean(daily["rv"].to_numpy()[calm] / iv[calm]) == pytest.approx(1, abs=0.03)
    assert np.mean(daily["bv"].to_numpy()[calm] / iv[calm]) == pytest.approx(1, abs=0.04)
    excess_bv = np.mean(daily["bv"].to_numpy()[jump_days] - iv[jump_days])  # a jump leaks into bipower only through its
    excess_rv = np.mean(daily["rv"].to_numpy()[jump_days] - iv[jump_days])  # neighbours: a small part of what it adds to RV
    assert 0 < excess_bv < 0.2 * excess_rv
    assert np.mean(daily["rv"].to_numpy()[jump_days] - iv[jump_days]) == pytest.approx(0.03**2, rel=0.15)  # RV adds its square
    assert (daily["jump"].to_numpy()[jump_days] > 0).mean() >= 0.9
    assert daily["rvol"].to_numpy() == pytest.approx(np.sqrt(daily["rv"].to_numpy() * 252))
    assert daily["n"].eq(78).all() and len(daily) == 300


def test_the_jump_test_size_without_jumps():
    data, _, _ = _intraday(days=1500, n=288, seed=5)
    daily = realized_measures(data, "r", jump_level=0.95).frame
    assert 0.02 <= (daily["jump p"] < 0.05).mean() <= 0.08
    with pytest.raises(SpecificationError, match="several returns a day"):
        realized_measures(daily_dataset(daily), "rv")


def daily_dataset(frame):
    return Dataset.from_frame(frame)


def test_har_against_least_squares_and_its_forecasts():
    data, _, _ = _intraday(days=400, seed=7)
    daily = realized_measures(data, "r")
    fit = har(daily, "rv")
    rv = daily["rv"].to_numpy()
    X = np.column_stack([np.ones(len(rv)), rv, pd.Series(rv).rolling(5).mean(), pd.Series(rv).rolling(22).mean()])[21:-1]
    b = np.linalg.lstsq(X, rv[22:], rcond=None)[0]
    assert fit.params.to_numpy() == pytest.approx(b, rel=1e-8)
    hist = list(rv)
    for _ in range(3):  # three days ahead by hand
        hist.append(b[0] + b[1] * hist[-1] + b[2] * np.mean(hist[-5:]) + b[3] * np.mean(hist[-22:]))
    assert fit.forecast(3).to_numpy() == pytest.approx(hist[-3:], rel=1e-8)
    from pymetrix_finance.risk import normal

    assert fit.value_at_risk(0.99)["var"] == pytest.approx(normal(0.0, np.sqrt(hist[-3]), 0.01)[0])
    logfit = har(daily, "rv", log=True)
    lb = np.linalg.lstsq(np.column_stack([np.ones(len(rv)), np.log(rv), pd.Series(np.log(rv)).rolling(5).mean(),
                                          pd.Series(np.log(rv)).rolling(22).mean()])[21:-1], np.log(rv[22:]), rcond=None)[0]
    assert logfit.params.to_numpy() == pytest.approx(lb, rel=1e-8)


def test_the_dialogs():
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS
    from pymetrix.views import sections

    data, _, _ = _intraday(days=120, seed=1)
    s = Session()
    s.objects["data"] = data
    s.run(COMMANDS["RealizedVolatility"].cls("r"))
    assert s.history[-1].code == "daily = realized_measures(data, 'r')"
    s.run(COMMANDS["FitHAR"].cls(log=True))
    assert s.objects["har_fit"].log and [t for t, _ in sections(s.objects["har_fit"])][0] == "Summary"
