"""Residual correlograms of VARs and VECMs.

- A VAR's residual auto- and cross-correlations against statsmodels' VARResults.resid_acorr, to 1e-12; a VECM's
  against the same formula by hand on its residuals.
- The table (a column per pair, the ±1.96/sqrt(T) band and how many fall outside it), the K x K panels, the result-window
  tabs, refusals and the recorded call."""

import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.api import VAR as SMVAR

from pymetrix import VAR, VECM, Dataset
from pymetrix.errors import SpecificationError
from pymetrix.views import sections


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(3)
    T = 200
    A = np.array([[0.5, 0.1, 0.0], [0.2, 0.4, 0.1], [0.0, 0.3, 0.6]])
    y = np.zeros((T, 3))
    for t in range(1, T):
        y[t] = 0.1 + A @ y[t - 1] + rng.normal(size=3)
    return pd.DataFrame(y, columns=["a", "b", "c"], index=pd.period_range("1970Q1", periods=T, freq="Q", name="time"))


def test_var_against_statsmodels(frame):
    var = VAR(["a", "b", "c"], lags=2).fit(Dataset.from_frame(frame))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = SMVAR(frame).fit(2)
    rc = var.residual_correlogram(10)
    assert np.allclose(rc.values, ref.resid_acorr(10)[1:], atol=1e-12)
    table = rc.table().to_frame()
    assert table.shape == (10, 9) and list(table.columns[:2]) == ["a, a[-h]", "a, b[-h]"]
    assert np.isclose(table["b, c[-h]"].iloc[2], rc.values[2, 1, 2])
    assert f"±{1.96 / np.sqrt(len(ref.resid)):.3f}" in " ".join(rc.table().notes)
    assert len(rc.plot().panels) == 9
    tabs = dict(sections(var))
    assert "Residual correlogram" in tabs and "Residual correlogram graph" in tabs
    with pytest.raises(SpecificationError, match="lags must be between"):
        var.residual_correlogram(0)


def test_vecm_by_hand(frame):
    levels = frame.cumsum().assign(b=lambda f: f.a + np.random.default_rng(1).normal(size=len(f)))
    vecm = VECM(["a", "b", "c"], lags=1, rank=1).fit(Dataset.from_frame(levels))
    U = np.asarray(vecm.results.resid)
    x = U - U.mean(0)
    T = len(x)
    c0 = x.T @ x / T
    h = 3
    expected = (x[h:].T @ x[:-h] / T) / np.sqrt(np.outer(np.diag(c0), np.diag(c0)))
    assert np.allclose(vecm.residual_correlogram(6).values[h - 1], expected, atol=1e-12)
    assert "Residual correlogram" in dict(sections(vecm))
