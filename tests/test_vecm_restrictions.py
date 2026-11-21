"""Restrictions on a VECM's cointegrating vectors and adjustment, checked independently: the reduced-rank regression's
cointegrating vectors and log likelihood equal statsmodels' in all five deterministic cases; the beta test equal to
twice the fall in the log likelihood when the system is re-estimated by least squares with the restricted beta; the
weak-exogeneity test equal to a reduced-rank regression of the other variables conditional on the exogenous one's
differences, done by hand; both keeping their size in repeated samples and rejecting false restrictions."""

import importlib

import numpy as np
import pandas as pd
import pytest
from scipy import linalg

from pymetrix import VECM, Dataset
from pymetrix.errors import SpecificationError

vr = importlib.import_module("pymetrix.vecm_restrictions")
pytest.importorskip("statsmodels")


def _data(T=500, seed=0):
    """b and e are random walks (weakly exogenous); a - b and c - 0.5 b - 0.8 e are stationary."""
    rng = np.random.default_rng(seed)
    b = np.cumsum(rng.normal(size=T)) + 0.02 * np.arange(T)
    e = np.cumsum(rng.normal(size=T))
    a, c = np.zeros(T), np.zeros(T)
    for t in range(1, T):
        a[t] = a[t - 1] - 0.3 * (a[t - 1] - b[t - 1]) + (b[t] - b[t - 1]) + 0.5 * rng.normal()
        c[t] = c[t - 1] - 0.4 * (c[t - 1] - 0.5 * b[t - 1] - 0.8 * e[t - 1]) + 0.5 * rng.normal()
    idx = pd.period_range("1900Q1", periods=T, freq="Q")
    return Dataset.from_frame(pd.DataFrame({"a": a, "b": b, "c": c, "e": e}, index=idx))


TRUE_H = [[1, 0], [-1, -0.5], [0, 1], [0, -0.8]]


@pytest.fixture(scope="module")
def fitted():
    return VECM(["a", "b", "c", "e"], lags=1, rank=2, deterministic="co").fit(_data())


@pytest.mark.parametrize("det", ["n", "co", "ci", "coli", "colo"])
def test_the_reduced_rank_regression_is_statsmodels(det):
    f = VECM(["a", "b", "c", "e"], lags=2, rank=2, deterministic=det).fit(_data(seed=1))
    rr = vr.reduced_rank(f)
    ll = -rr.T / 2 * (4 * np.log(2 * np.pi) + 4 + np.linalg.slogdet(rr.S00)[1] + np.sum(np.log(1 - rr.eigenvalues[:2])))
    beta = rr.vectors[:, :2] @ np.linalg.inv(rr.vectors[:2, :2])
    assert rr.T == f.results.nobs and ll == pytest.approx(float(np.real(f.results.llf)), rel=1e-10)
    assert beta[:4] == pytest.approx(np.asarray(f.results.beta), abs=1e-9)
    if det in ("ci", "coli"):  # statsmodels keeps the restricted constant or trend apart from beta
        assert beta[4] == pytest.approx(np.asarray(f.results.det_coef_coint)[0], abs=1e-9)


def test_beta_against_least_squares_with_the_restricted_beta(fitted):
    H = np.array([[1, 0], [-1, 0], [0, 1], [0, -0.8]])  # b's weight in the second relation fixed at 0 (false: it is -0.5)
    res = fitted.test_beta(H)
    beta = res.details["restricted beta"].to_numpy()
    y = np.asarray(fitted.results.model.endog, float)
    dy = np.diff(y, axis=0)

    def logdet(b):  # least squares of the differences on beta'y[-1], the lagged difference and a constant
        X = np.column_stack([y[1:-1] @ b, dy[:-1], np.ones(len(dy) - 1)])
        U = dy[1:] - X @ np.linalg.lstsq(X, dy[1:], rcond=None)[0]
        return np.linalg.slogdet(U.T @ U / len(U))[1]

    unrestricted = np.asarray(fitted.results.beta)
    assert res.statistic == pytest.approx(len(dy[1:]) * (logdet(beta) - logdet(unrestricted)), rel=1e-8)
    assert res.df == 4 and res.p_value < 0.01
    assert fitted.test_beta(np.eye(4)).statistic == pytest.approx(0, abs=1e-8)


def test_weak_exogeneity_by_a_conditional_reduced_rank_regression(fitted):
    res = fitted.test_weak_exogeneity(["b", "e"])
    y = np.asarray(fitted.results.model.endog, float)
    dy = np.diff(y, axis=0)
    Z0, Z1 = dy[1:], y[1:-1]
    T = len(Z0)
    Z2 = np.column_stack([dy[:-1], np.ones(T)])

    def resid(M, X):
        return M - X @ np.linalg.lstsq(X, M, rcond=None)[0]

    def eigenvalues(R0, R1):
        S00, S01, S11 = R0.T @ R0 / T, R0.T @ R1 / T, R1.T @ R1 / T
        return np.sort(linalg.eigvalsh(S01.T @ np.linalg.solve(S00, S01), S11))[::-1]

    lam = eigenvalues(resid(Z0, Z2), resid(Z1, Z2))
    conditioning = np.column_stack([Z2, Z0[:, [1, 3]]])  # the exogenous variables' differences as regressors
    lam_star = eigenvalues(resid(Z0[:, [0, 2]], conditioning), resid(Z1, conditioning))
    assert res.statistic == pytest.approx(T * np.sum(np.log((1 - lam_star[:2]) / (1 - lam[:2]))), rel=1e-8)
    assert res.df == 4 and res.p_value > 0.01
    assert fitted.test_weak_exogeneity("a").p_value < 1e-6  # a adjusts


def test_size_in_repeated_samples():
    p_beta, p_weak = [], []
    for seed in range(60):
        f = VECM(["a", "b", "c", "e"], lags=1, rank=2, deterministic="co").fit(_data(T=400, seed=100 + seed))
        p_beta.append(f.test_beta(TRUE_H).p_value)
        p_weak.append(f.test_weak_exogeneity(["b", "e"]).p_value)
    assert np.mean(np.array(p_beta) < 0.05) == pytest.approx(0.05, abs=0.07)
    assert np.mean(np.array(p_weak) < 0.05) == pytest.approx(0.05, abs=0.07)


def test_refusals(fitted):
    with pytest.raises(SpecificationError, match="4 rows"):
        fitted.test_beta([[1, 0], [0, 1]])
    with pytest.raises(SpecificationError, match="at least 2"):
        fitted.test_beta([[1], [-1], [0], [0]])
    with pytest.raises(SpecificationError, match="not variables"):
        fitted.test_weak_exogeneity(["q"])
    with pytest.raises(SpecificationError, match="fewer than the rank"):
        fitted.test_weak_exogeneity(["a", "b", "c"])
