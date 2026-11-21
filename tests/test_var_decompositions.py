"""Variance and historical decompositions of BVARs with stochastic volatility, Markov-switching VARs and quantile VARs,
checked independently:

- a BVAR-SV draw with constant volatilities is a VAR: its decompositions equal ``FittedVAR``'s (least-squares
  coefficients, the covariance's LDL factors as A and the volatilities); a fitted draw's shares at a date equal the
  Cholesky shares of that date's covariance A^-1 D_t^2 A^-T;
- every draw's historical decomposition adds up to the data to 1e-10, and its contributions do not depend on the
  volatilities (they scale each shock and its impact in opposite directions);
- an MS-VAR whose regimes are identical is a VAR: its decompositions equal ``FittedVAR``'s, whatever the regime path;
  a fitted MS-VAR's contributions equal a brute-force sum over the regime path's companion-matrix products;
- a median quantile VAR of Gaussian data decomposes the variance about as the least-squares VAR does.
"""

import dataclasses
import importlib

import numpy as np
import pandas as pd
import pytest

from pymetrix import BVARSV, MSVAR, VAR, Dataset, QuantileVAR
from pymetrix.errors import SpecificationError
from pymetrix.ms_var import FittedMSVAR

bsv = importlib.import_module("pymetrix.bvar_sv")


def _gaussian(T=300, seed=0):
    rng = np.random.default_rng(seed)
    A1, A2 = np.array([[0.5, 0.1], [0.2, 0.4]]), np.array([[-0.1, 0.05], [0.0, 0.1]])
    C = np.array([[1.0, 0.0], [0.5, 0.8]])
    y = np.zeros((T, 2))
    for t in range(2, T):
        y[t] = np.array([0.3, -0.2]) + A1 @ y[t - 1] + A2 @ y[t - 2] + C @ rng.normal(size=2)
    return Dataset.from_frame(pd.DataFrame(y, columns=["a", "b"], index=pd.period_range("1950Q1", periods=T, freq="Q")))


@pytest.fixture(scope="module")
def data():
    return _gaussian()


@pytest.fixture(scope="module")
def var(data):
    return VAR(["a", "b"], lags=2).fit(data)


def _sv_from_var(var, data, volatilities=None):
    """A one-draw BVAR-SV at the VAR's estimates: A^-1 D = chol(Sigma), D constant (or the given (T, K) log sds)."""
    K, p = 2, 2
    Yall = data.frame[["a", "b"]].to_numpy()
    Y, X = bsv._design(Yall, p)
    L = np.linalg.cholesky(var.estimate.sigma)
    A = np.linalg.inv(L / np.diag(L))
    h = np.tile(np.log(np.diag(L)), (len(Y), 1)) if volatilities is None else volatilities
    return bsv.FittedBVARSV(BVARSV(["a", "b"], lags=p), var.estimate.B.T[None], A[None], h[None], np.ones((1, K)),
                            data.frame.index[p:], X, Y)


def test_sv_with_constant_volatility_is_the_var(var, data):
    sv = _sv_from_var(var, data)
    for period in (None, "1960Q1"):
        f = sv.fevd(10, period=period)
        np.testing.assert_allclose(f.values, var.fevd(10).values, atol=1e-12)
        np.testing.assert_allclose(f.lower, f.values, atol=1e-12)  # one draw: the band is the point
    hd, ref = sv.historical_decomposition(), var.historical_decomposition()
    for v in ("a", "b"):
        np.testing.assert_allclose(hd.contributions[v].to_numpy(), ref.contributions[v].to_numpy(), atol=1e-10)
    np.testing.assert_allclose(hd.baseline.to_numpy(), ref.baseline.to_numpy(), atol=1e-10)
    np.testing.assert_allclose(hd.shocks.to_numpy(), ref.shocks.to_numpy(), atol=1e-10)  # P = A^-1 D = chol(Sigma)


@pytest.fixture(scope="module")
def sv(data):
    return BVARSV(["a", "b"], lags=2).fit(data, draws=40, burn=40, seed=2)


def _draw(fit, d):
    return dataclasses.replace(fit, B=fit.B[d:d + 1], A=fit.A[d:d + 1], h=fit.h[d:d + 1], w=fit.w[d:d + 1])


def test_sv_fevd_at_a_date_is_that_dates_cholesky_decomposition(sv):
    from pymetrix.var import _ma

    t = 100
    full = sv.fevd(8, period=str(sv.index[t]), coverage=0.9)
    assert full.values.sum(axis=2) == pytest.approx(np.ones((8, 2)), abs=1e-12)
    assert np.all(full.lower <= full.values + 1e-12) and np.all(full.values <= full.upper + 1e-12)
    for d in (0, 17, 39):
        Ainv = np.linalg.inv(sv.A[d])
        sigma = Ainv @ np.diag(np.exp(2 * sv.h[d, t])) @ Ainv.T  # that date's covariance
        lagm = np.stack([sv.B[d][:, 1 + 2 * l:3 + 2 * l] for l in range(2)])
        theta = _ma(lagm, 7) @ np.linalg.cholesky(sigma)
        c = np.cumsum(theta**2, axis=0)
        np.testing.assert_allclose(_draw(sv, d).fevd(8, period=str(sv.index[t])).values, c / c.sum(axis=2, keepdims=True), atol=1e-12)
    assert not np.allclose(sv.fevd(8, period=str(sv.index[0])).values, full.values)  # the date matters
    with pytest.raises(SpecificationError, match="not a period"):
        sv.fevd(8, period="1800Q1")


def test_sv_historical_decomposition_adds_up_draw_by_draw(sv):
    Y = sv.Y
    for d in (0, 21, 39):
        hd = _draw(sv, d).historical_decomposition()
        total = hd.baseline.to_numpy() + np.stack([hd.contributions[v].sum(axis=1).to_numpy() for v in sv.variables], axis=1)
        np.testing.assert_allclose(total, Y, atol=1e-10)
    hd = sv.historical_decomposition(coverage=0.9)
    assert hd.has_bands and hd.coverage == 0.9 and hd.band_source == "40 posterior draws"
    assert float(hd.residual.abs().to_numpy().max()) < 1e-10  # the mean over draws adds up too
    for v in sv.variables:
        assert np.all(hd.lower[v].to_numpy() <= hd.median[v].to_numpy() + 1e-12)
        assert np.all(hd.median[v].to_numpy() <= hd.upper[v].to_numpy() + 1e-12)
    assert sv.historical_decomposition(draws=5).band_source == "5 posterior draws"
    # the contributions do not depend on the volatilities; the standardised shocks do
    one = _draw(sv, 3)
    moved = dataclasses.replace(one, h=one.h + np.random.default_rng(0).normal(size=one.h.shape))
    a, b = one.historical_decomposition(), moved.historical_decomposition()
    np.testing.assert_allclose(a.contributions["b"].to_numpy(), b.contributions["b"].to_numpy(), atol=1e-10)
    assert not np.allclose(a.shocks.to_numpy(), b.shocks.to_numpy())


def _ms_from_var(var, data, smoothed):
    K, p = 2, 2
    A = var.coefs
    coefs = np.hstack(list(A))  # (K, K p): lags side by side
    probs = pd.DataFrame(smoothed, index=data.frame.index[p:], columns=["regime 0", "regime 1"])
    c = var.estimate.B[0]
    return FittedMSVAR(MSVAR(["a", "b"], lags=p, regimes=2, switching="all"), np.stack([c, c]), np.stack([coefs, coefs]),
                       np.stack([var.estimate.sigma] * 2), np.array([[0.9, 0.1], [0.2, 0.8]]), 0.0, 0, probs, probs,
                       len(probs), data.frame[["a", "b"]])


def test_ms_var_with_identical_regimes_is_the_var(var, data):
    T = len(data.frame) - 2
    w = np.random.default_rng(1).uniform(size=T)
    ms = _ms_from_var(var, data, np.column_stack([w, 1 - w]))  # the most likely regime switches often
    assert (np.diff(ms.regime_path().to_numpy()) != 0).sum() > 20
    for regime in (0, 1):
        np.testing.assert_allclose(ms.fevd(regime, 12).values, var.fevd(12).values, atol=1e-12)
    hd, ref = ms.historical_decomposition(), var.historical_decomposition()
    for v in ("a", "b"):
        np.testing.assert_allclose(hd.contributions[v].to_numpy(), ref.contributions[v].to_numpy(), atol=1e-10)
    np.testing.assert_allclose(hd.baseline.to_numpy(), ref.baseline.to_numpy(), atol=1e-10)
    np.testing.assert_allclose(hd.shocks.to_numpy(), ref.shocks.to_numpy(), atol=1e-10)
    with pytest.raises(SpecificationError, match="regime is 0 to 1"):
        ms.fevd(2)
    with pytest.raises(SpecificationError, match="fit it again"):
        dataclasses.replace(ms, sample_rows=None).historical_decomposition()


def _two_regimes(T=300, seed=3):
    rng = np.random.default_rng(seed)
    s, y = 0, np.zeros((T, 2))
    c = [np.array([-1.0, 0.5]), np.array([1.5, -0.5])]
    A = [np.array([[0.5, 0.1], [0.0, 0.3]]), np.array([[0.2, -0.1], [0.3, 0.5]])]
    sd = [np.array([[0.5, 0.0], [0.2, 0.4]]), np.array([[1.0, 0.0], [-0.3, 1.2]])]
    for t in range(1, T):
        if rng.uniform() < 0.05:
            s = 1 - s
        y[t] = c[s] + A[s] @ y[t - 1] + sd[s] @ rng.normal(size=2)
    return Dataset.from_frame(pd.DataFrame(y, columns=["a", "b"], index=pd.period_range("1950Q1", periods=T, freq="Q")))


@pytest.mark.parametrize("switching", ["all", "intercept_variance"])
def test_ms_var_historical_decomposition_by_brute_force(switching):
    data = _two_regimes()
    ms = MSVAR(["a", "b"], lags=1, regimes=2, switching=switching).fit(data, starts=2)
    hd = ms.historical_decomposition()
    assert float(hd.residual.abs().to_numpy().max()) < 1e-10
    s = ms.regime_path().to_numpy()
    assert 0 < s.mean() < 1
    Yall = data.frame[["a", "b"]].to_numpy()
    Y, T = Yall[1:], len(Yall) - 1
    A, c, P = ms.coefs[s], ms.intercepts[s], np.linalg.cholesky(ms.sigmas)[s]
    e = np.array([np.linalg.solve(P[t], Y[t] - c[t] - A[t] @ Yall[t]) for t in range(T)])
    for t in (0, 5, 150, T - 1):  # contribution of shock j at t: sum over r <= t of A_t ... A_r+1 P_r[:, j] e_j,r
        brute = np.zeros((2, 2))
        for r in range(t + 1):
            prod = np.eye(2)
            for q in range(r + 1, t + 1):
                prod = A[q] @ prod
            brute += prod @ P[r] * e[r]
        got = np.column_stack([hd.contributions[v].iloc[t].to_numpy() for v in ("a", "b")]).T
        np.testing.assert_allclose(got, brute, atol=1e-10)
    # the regimes' intercepts are in the baseline
    base = hd.baseline.to_numpy()
    np.testing.assert_allclose(base[0], c[0] + A[0] @ Yall[0], atol=1e-12)
    f = ms.fevd(1, 10)
    assert f.values.sum(axis=2) == pytest.approx(np.ones((10, 2)), abs=1e-12)


def test_quantile_var_fevd_at_the_median_of_a_gaussian_var():
    data = _gaussian(T=800, seed=5)
    q = QuantileVAR(["a", "b"], lags=2, quantile=0.5).fit(data)
    ols = VAR(["a", "b"], lags=2).fit(data).fevd(10).values
    f = q.fevd(10)
    assert f.values.sum(axis=2) == pytest.approx(np.ones((10, 2)), abs=1e-12)
    assert f.values == pytest.approx(ols, abs=0.03)
    u = q.residuals()
    assert len(u) == len(data.frame) - 2 and abs(float((u > 0).mean().mean()) - 0.5) < 0.02
    low = QuantileVAR(["a", "b"], lags=2, quantile=0.1).fit(data)
    assert float((low.residuals() > 0).mean().mean()) > 0.85  # not centred: hence no historical decomposition
    assert not hasattr(low, "historical_decomposition")
