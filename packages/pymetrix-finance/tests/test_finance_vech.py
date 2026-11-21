"""The full VECH (the finance plugin), checked independently: its likelihood by a plain loop over the vech recursion;
no lower than the diagonal VECH it nests (and equal to it at diagonal A and B); forecasts returning to the sample
covariance; refused for more than three series."""

import importlib

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset
from pymetrix.errors import SpecificationError
from pymetrix_finance import mgarch

mg = importlib.import_module("pymetrix_finance.mgarch")


@pytest.fixture(scope="module")
def returns():
    rng = np.random.default_rng(6)
    T = 1500
    S = np.array([[1.0, 0.3], [0.3, 1.5]]) * 1e-4
    A, B = np.array([[0.08, 0.04], [0.04, 0.05]]), np.array([[0.88, 0.9], [0.9, 0.92]])
    e, H = np.zeros((T, 2)), S.copy()
    for t in range(T):
        e[t] = np.linalg.cholesky(H) @ rng.standard_normal(2)
        H = (1 - A - B) * S + A * np.outer(e[t], e[t]) + B * H
    idx = pd.period_range("2001-01-01", periods=T, freq="D", name="time")
    return Dataset.from_frame(pd.DataFrame(e, columns=["u", "v"], index=idx))


def test_the_full_vech(returns):
    full = mgarch(returns, ["u", "v"], model="vech")
    diag = mgarch(returns, ["u", "v"], model="diagonal_vech")
    assert full.loglike >= diag.loglike - 1e-8 and len(full.values) == 18
    e = full.returns - full.mu
    S = e.T @ e / len(e)
    A, B = full.state["vech_A"], full.state["vech_B"]
    iu = np.triu_indices(2)
    h, ll = S[iu].copy(), 0.0
    c = (np.eye(3) - A - B) @ S[iu]
    for t in range(len(e)):  # the likelihood by a plain loop
        Ht = np.array([[h[0], h[1]], [h[1], h[2]]])
        ll += -0.5 * (2 * np.log(2 * np.pi) + np.linalg.slogdet(Ht)[1] + e[t] @ np.linalg.solve(Ht, e[t]))
        h = c + A @ np.outer(e[t], e[t])[iu] + B @ h
    assert full.loglike == pytest.approx(ll, rel=1e-10)
    da = np.diag(mg._bekk_matrices(diag.values, "diagonal_vech", 2)[0][iu])  # diagonal A and B: the diagonal VECH
    db = np.diag(mg._bekk_matrices(diag.values, "diagonal_vech", 2)[1][iu])
    assert mg.gaussian_loglike(e, mg.vech_path(e, S, da, db)) == pytest.approx(diag.loglike, rel=1e-10)
    assert full.forecast(4000)[-1] == pytest.approx(S, rel=1e-3)


def test_refused_for_many_series():
    rng = np.random.default_rng(7)
    data = Dataset.from_frame(pd.DataFrame(rng.normal(size=(200, 4)) * 0.01, columns=list("abcd"),
                                           index=pd.period_range("2001-01-01", periods=200, freq="D", name="time")))
    with pytest.raises(SpecificationError, match="two or three"):
        mgarch(data, list("abcd"), model="vech")
