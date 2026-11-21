"""Smooth local projections, checked independently: with no penalty, exactly the ordinary local projection on the
same periods (the spline has a function for every horizon); a large penalty making the response a straight line in
the horizon (order 2) or a constant (order 1); cross-validation choosing a penalty that brings the response nearer the
truth than the ordinary projection, on average; Newey-West standard errors against the stacked regression's by hand."""

import importlib

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, LocalProjection
from pymetrix.errors import SpecificationError

slp = importlib.import_module("pymetrix.smooth_lp")
H = 12
THETA = np.exp(-0.3 * np.arange(H + 1)) * np.sin(np.arange(H + 1) / 2 + 0.5) * 2


def _data(T=300, seed=0, noise=1.5):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=T)
    y = np.convolve(e, THETA)[:T] + rng.normal(0, noise, T)
    return Dataset.from_frame(pd.DataFrame({"y": y, "x": e}, index=pd.period_range("1950Q1", periods=T, freq="Q", name="time")))


def test_no_penalty_is_the_ordinary_projection():
    data = _data()
    lp = LocalProjection("y", "x", horizons=H, lags=2)
    zero = lp.fit_smooth(data, penalty=0.0)
    idx = data.frame.index
    ordinary = lp.fit(data, span=(str(idx[2]), str(idx[len(idx) - 1 - H])))  # the same periods at every horizon
    assert zero.irf["response"].to_numpy() == pytest.approx(ordinary.irf["response"].to_numpy(), abs=1e-10)
    assert np.linalg.matrix_rank(slp.spline_basis(H)) == H + 1  # any response at the horizons can be matched


def test_a_large_penalty_gives_a_polynomial():
    data = _data(seed=1)
    lp = LocalProjection("y", "x", horizons=H, lags=1)
    line = lp.fit_smooth(data, penalty=1e12, order=2).irf["response"].to_numpy()
    assert np.max(np.abs(np.diff(line, 2))) < 1e-6  # a straight line in h
    flat = lp.fit_smooth(data, penalty=1e12, order=1).irf["response"].to_numpy()
    assert np.max(np.abs(np.diff(flat))) < 1e-6  # a constant


def test_cross_validation_smooths_towards_the_truth():
    errors = []
    for seed in range(5):
        fit = LocalProjection("y", "x", horizons=H, lags=1).fit_smooth(_data(seed=seed + 10))
        errors.append((np.mean((fit.irf["response"] - THETA) ** 2), np.mean((fit.irf["ordinary"] - THETA) ** 2)))
        assert fit.penalty >= 0 and fit.cv is not None and len(fit.cv) == 31
    smooth, ordinary = np.mean(errors, axis=0)
    assert smooth < ordinary


def test_standard_errors_by_hand_and_refusals():
    data = _data(T=200, seed=2)
    lp = LocalProjection("y", "x", horizons=3, lags=1)
    fit = lp.fit_smooth(data, penalty=0.0)
    Y, (x, W), index = slp._stacked(lp, data, None)
    B = slp.spline_basis(3)
    X, start = slp._design(x, W, B, np.arange(len(index)), 3)
    y = Y.T.reshape(-1)
    theta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ theta
    T = len(index)
    S = sum(X[h * T:(h + 1) * T] * e[h * T:(h + 1) * T, None] for h in range(4))
    meat = S.T @ S + sum((1 - k / 5) * (S[k:].T @ S[:-k] + S[:-k].T @ S[k:]) for k in range(1, 5))
    bread = np.linalg.pinv(X.T @ X)  # the spline's coefficients are not unique without a penalty; its values are
    V = (bread @ meat @ bread)[start:, start:]
    assert fit.irf["se"].to_numpy() == pytest.approx(np.sqrt(np.diag(B @ V @ B.T)), rel=1e-8)
    with pytest.raises(SpecificationError, match="time series"):
        LocalProjection("y", "x", horizons=3, state="x").fit_smooth(data)


def test_the_dialog():
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS
    from pymetrix.views import sections

    s = Session()
    s.objects["data"] = _data(T=200)
    s.run(COMMANDS["FitLocalProjection"].cls("y", "x", horizons=4, lags=1, smooth=True))
    assert ".fit_smooth(data)" in s.history[-1].code
    assert [t for t, _ in sections(s.objects[s.history[-1].outputs[0]])][:1] == ["Responses"]
