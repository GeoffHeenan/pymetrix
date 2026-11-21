"""Dynamic equicorrelation and the diagonal VECH (the finance plugin), checked independently: with two series DECO is
DCC (the same parameters and likelihood); its closed-form likelihood against numpy's determinant and inverse of the
equicorrelation matrices, rho_t the mean of DCC's correlations; DECO recovering a simulated common correlation; the
diagonal VECH nesting the scalar BEKK (its likelihood no lower), its likelihood by a plain loop and above the true parameters',
and its estimates within three standard errors of a simulated VECH's."""

import importlib

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset
from pymetrix_finance import mgarch

mg = importlib.import_module("pymetrix_finance.mgarch")


def _data(r, names):
    idx = pd.period_range("2001-01-01", periods=len(r), freq="D", name="time")
    return Dataset.from_frame(pd.DataFrame(r, columns=names, index=idx))


def _equicorrelated(T, N, rho, seed):
    rng = np.random.default_rng(seed)
    R = (1 - rho) * np.eye(N) + rho * np.ones((N, N))
    z = rng.standard_normal((T, N)) @ np.linalg.cholesky(R).T
    h = np.ones(N) * 1e-4
    r = np.zeros((T, N))
    for t in range(T):
        r[t] = np.sqrt(h) * z[t]
        h = 1e-6 + 0.08 * r[t] ** 2 + 0.9 * h
    return r


def test_deco_with_two_series_is_dcc():
    data = _data(_equicorrelated(1500, 2, 0.4, 0), ["x", "y"])
    deco, dcc = mgarch(data, ["x", "y"], model="deco"), mgarch(data, ["x", "y"], model="dcc")
    assert deco.values[-2:] == pytest.approx(dcc.values[-2:], abs=1e-5) and deco.loglike == pytest.approx(dcc.loglike, rel=1e-8)
    assert deco.H == pytest.approx(dcc.H, rel=1e-4)


def test_the_deco_likelihood_and_a_common_correlation():
    data = _data(_equicorrelated(2000, 4, 0.5, 1), ["a", "b", "c", "d"])
    fit = mgarch(data, ["a", "b", "c", "d"], model="deco")
    a, b = fit.state["a"], fit.state["b"]
    sd = fit.volatilities.to_numpy()
    z = (fit.returns - fit.mu) / sd
    Q = fit.state["Qbar"]
    R = mg.dcc_path(z, a, b, Q)
    rho = (R.sum(axis=(1, 2)) - 4) / 12
    E = (1 - rho)[:, None, None] * np.eye(4) + rho[:, None, None]
    ll = -0.5 * np.sum(np.linalg.slogdet(E)[1] + np.einsum("ti,tij,tj->t", z, np.linalg.inv(E), z) - np.sum(z**2, axis=1))
    assert -mg.deco_objective(np.array([a, b]), z, Q) == pytest.approx(ll, rel=1e-10)
    assert fit.correlations.to_numpy().mean() == pytest.approx(0.5, abs=0.06)
    corr = fit.correlations.to_numpy()
    assert np.allclose(corr, corr[:, :1])  # every pair the same at each date
    assert fit.forecast(5).shape == (5, 4, 4)


def test_diagonal_vech():
    rng = np.random.default_rng(2)
    T = 3000
    S = np.array([[1.0, 0.3], [0.3, 1.5]]) * 1e-4
    A = np.array([[0.08, 0.04], [0.04, 0.05]])
    B = np.array([[0.88, 0.9], [0.9, 0.92]])
    e, H = np.zeros((T, 2)), S.copy()
    for t in range(T):
        e[t] = np.linalg.cholesky(H) @ rng.standard_normal(2)
        H = (1 - A - B) * S + A * np.outer(e[t], e[t]) + B * H
    data = _data(e, ["u", "v"])
    vech = mgarch(data, ["u", "v"], model="diagonal_vech")
    scalar = mgarch(data, ["u", "v"], model="bekk")
    assert vech.loglike >= scalar.loglike - 1e-6  # it nests the scalar BEKK
    assert list(vech.names) == ["a u, u", "a u, v", "a v, v", "b u, u", "b u, v", "b v, v"]
    truth = np.r_[A[np.triu_indices(2)], B[np.triu_indices(2)]]
    assert np.all(np.abs(vech.values - truth) < 3 * vech.se)
    resid = vech.returns - vech.mu
    Sh = resid.T @ resid / T
    Am, Bm = mg._bekk_matrices(vech.values, "diagonal_vech", 2)
    h, ll = Sh.copy(), 0.0
    for t in range(T):  # the likelihood by a plain loop
        ll += -0.5 * (2 * np.log(2 * np.pi) + np.linalg.slogdet(h)[1] + resid[t] @ np.linalg.solve(h, resid[t]))
        h = (1 - Am - Bm) * Sh + Am * np.outer(resid[t], resid[t]) + Bm * h
    assert vech.loglike == pytest.approx(ll, rel=1e-10)
    assert vech.loglike >= mg.gaussian_loglike(resid, mg.bekk_path(resid, Sh, A, B))  # the maximum: above the truth's likelihood
