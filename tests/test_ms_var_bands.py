"""Bands for a Markov-switching VAR's responses, variance and historical decompositions: draws of the parameters from
the ML estimates' asymptotic normal distribution (pymetrix.ms_var_bands).

Checks: the batched exact log likelihood equals the fit's; the standard errors (inverse observed information) equal
statsmodels' MarkovRegression's for one equation; when the regimes share their dynamics and covariance (only the
intercepts switch, the regimes well separated) the band widths are close to those of the plain VAR's parametric
bootstrap with the regime as a dummy; bands are ordered, contain the estimates and are reproducible by seed; every
draw is a valid set of covariance and transition matrices, and its variance shares add up to one; the historical
decomposition's bands are those of the point decomposition under each drawn parameter set."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from test_var_decompositions import _two_regimes

from pymetrix import MSVAR, VAR, Dataset
from pymetrix.errors import SpecificationError
from pymetrix.ms_var_bands import loglike_batch, pack
from pymetrix.var import _ma
from pymetrix.var_decompositions import fevd_shares


@pytest.fixture(scope="module")
def ms():
    return MSVAR(["a", "b"], lags=1, regimes=2, switching="intercept_variance").fit(_two_regimes(), starts=2)


def _intercept_only(T=300, seed=11):
    """Two regimes with the same dynamics and covariance, intercepts far apart: the ML fit and the regime path. (The
    fit before data-driven starts missed these regimes, so this fit was once EM started from the true path.)"""
    rng = np.random.default_rng(seed)
    A, L = np.array([[0.3, 0.1], [-0.1, 0.2]]), np.array([[1.0, 0.0], [0.5, 0.8]])
    c = [np.zeros(2), np.array([4.0, -3.0])]
    s, y = np.zeros(T, int), np.zeros((T, 2))
    for t in range(1, T):
        s[t] = s[t - 1] if rng.uniform() < 0.97 else 1 - s[t - 1]
        y[t] = c[s[t]] + A @ y[t - 1] + L @ rng.normal(size=2)
    idx = pd.period_range("1900Q1", periods=T, freq="Q")
    frame = pd.DataFrame({"a": y[:, 0], "b": y[:, 1], "d": s.astype(float)}, index=idx)
    fit = MSVAR(["a", "b"], lags=1, regimes=2, switching="intercept").fit(Dataset.from_frame(frame))
    return fit, Dataset.from_frame(frame), s


def test_loglike_and_standard_errors_match_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    data = _two_regimes()
    y = data.frame["a"].to_numpy()
    for switching in ("intercept_variance", "intercept"):
        fit = MSVAR(["a"], lags=1, regimes=2, switching=switching).fit(data, starts=1)
        assert loglike_batch(pack(fit), fit)[0] == pytest.approx(fit.loglike, abs=1e-9)
        nS = 2 if switching != "intercept" else 1
        P = fit.transition
        start = np.r_[P[0, 0], P[1, 0], fit.intercepts[:, 0], fit.coefs[0, 0, 0], fit.sigmas[:nS, 0, 0]]
        ref = sm.tsa.MarkovRegression(y[1:], k_regimes=2, exog=y[:-1], switching_variance=switching != "intercept",
                                      switching_exog=False).fit(start_params=start, search_reps=0)  # its own optimum and Hessian
        assert ref.llf == pytest.approx(fit.loglike, abs=1e-5)
        se = np.sqrt(np.diag(fit.parameter_covariance()))
        # statsmodels: p[0->0], p[1->0], const 0, const 1, x1, sigma2 (0, 1); ours: const 0, const 1, x1, log sd (0, 1),
        # logits of P[0, 0] and P[1, 0]
        bse = ref.bse
        np.testing.assert_allclose(fit.intercepts[:, 0], ref.params[[2, 3]], rtol=1e-4)
        np.testing.assert_allclose(se[[0, 1, 2]], bse[[2, 3, 4]], rtol=2e-3)
        sig2 = fit.sigmas[:nS, 0, 0]
        np.testing.assert_allclose(2 * sig2 * se[3:3 + nS], bse[5:5 + nS], rtol=2e-3)  # d sigma2 / d log sd = 2 sigma2
        np.testing.assert_allclose(se[3 + nS:] * P[:, 0] * P[:, 1], bse[[0, 1]], rtol=2e-3)  # d p / d logit = p (1 - p)


def test_bands_with_regimes_sharing_dynamics_are_the_var_bootstrap_bands():
    fit, data, s = _intercept_only()
    assert (fit.regime_path().to_numpy() == s[1:]).all()
    var = VAR(["a", "b"], lags=1, exog=["d"]).fit(data)  # the regime known, as a dummy
    np.testing.assert_allclose(fit.fevd(0, 8).values, var.fevd(8).values, atol=1e-7)
    boot = list(var._bootstrap(800, 1))  # the VAR's parametric bootstrap
    ours = fit.irf(0, 8, draws=800, coverage=0.9, seed=1)
    lo, hi = np.quantile(np.stack([_ma(A, 8) @ np.linalg.cholesky(S) for A, S in boot]), [0.05, 0.95], axis=0)
    w, wr = ours.upper - ours.lower, hi - lo
    ratio = w[wr > 1e-6] / wr[wr > 1e-6]
    assert 0.8 < ratio.min() and ratio.max() < 1.2 and abs(np.median(ratio) - 1) < 0.06  # 0.92 to 1.10, median 1.00
    f = fit.fevd(0, 8, draws=800, coverage=0.9, seed=1)
    lo, hi = np.quantile(np.stack([fevd_shares(_ma(A, 7) @ np.linalg.cholesky(S)) for A, S in boot]), [0.05, 0.95], axis=0)
    ratio = (f.upper - f.lower)[hi - lo > 1e-3] / (hi - lo)[hi - lo > 1e-3]
    assert 0.7 < ratio.min() and ratio.max() < 1.2  # 0.81 to 0.96


def test_draws_are_valid_and_bands_ordered_and_reproducible(ms):
    d = ms.parameter_draws(300, seed=4)
    assert np.all(np.linalg.eigvalsh(d["sigmas"]) > 0)
    np.testing.assert_allclose(d["transition"].sum(axis=2), 1, atol=1e-12)
    assert np.all((d["transition"] > 0) & (d["transition"] < 1))
    se = np.sqrt(np.diag(ms.parameter_covariance()))[:4].reshape(2, 2)
    assert np.all(np.abs(d["intercepts"].mean(axis=0) - ms.intercepts) < 4 * se / np.sqrt(300))
    for regime in (0, 1):
        K = 2
        shares = np.stack([fevd_shares(_ma(np.stack([c[:, :K]]), 9) @ np.linalg.cholesky(S)) for c, S in zip(d["coefs"][:, regime], d["sigmas"][:, regime])])
        np.testing.assert_allclose(shares.sum(axis=3), 1, atol=1e-12)
        f = ms.fevd(regime, 10, draws=300, coverage=0.8, seed=4)
        lo, hi = np.quantile(shares, [0.1, 0.9], axis=0)
        np.testing.assert_allclose(f.lower, lo, atol=1e-12)
        np.testing.assert_allclose(f.upper, hi, atol=1e-12)
        assert np.all(f.lower <= f.values + 1e-12) and np.all(f.values <= f.upper + 1e-12)
        i = ms.irf(regime, 10, draws=300, coverage=0.9, seed=4)
        assert np.all(i.lower <= i.values + 1e-12) and np.all(i.values <= i.upper + 1e-12)
        assert i.coverage == 0.9 and "asymptotic normal" in i.identification
    a, b, c = (ms.fevd(1, 6, draws=100, seed=s) for s in (7, 7, 8))
    np.testing.assert_array_equal(a.upper, b.upper)
    assert not np.allclose(a.upper, c.upper)
    assert a.has_bands and a.coverage == 0.68 and "100 draws" in a.band_source
    assert not ms.fevd(1, 6).has_bands


def test_historical_decomposition_bands_are_draw_by_draw_decompositions(ms):
    hd = ms.historical_decomposition(draws=5, coverage=0.999999, seed=2)
    assert float(hd.residual.abs().to_numpy().max()) < 1e-10  # the point decomposition is unchanged
    ref = ms.historical_decomposition()
    for v in ("a", "b"):
        np.testing.assert_allclose(hd.contributions[v].to_numpy(), ref.contributions[v].to_numpy(), atol=1e-12)
    d = ms.parameter_draws(5, seed=2)
    each = [dataclasses.replace(ms, intercepts=d["intercepts"][n], coefs=d["coefs"][n], sigmas=d["sigmas"][n]).historical_decomposition()
            for n in range(5)]
    for v in ("a", "b"):
        stack = np.stack([h.contributions[v].to_numpy() for h in each])
        np.testing.assert_allclose(hd.lower[v].to_numpy(), stack.min(axis=0), atol=1e-4 * np.abs(stack).max())
        np.testing.assert_allclose(hd.upper[v].to_numpy(), stack.max(axis=0), atol=1e-4 * np.abs(stack).max())
        assert all(float(h.residual.abs().to_numpy().max()) < 1e-10 for h in each)  # every draw adds up to the data
    assert "regime path held" in hd.band_source
    hd68 = ms.historical_decomposition(draws=200, seed=2)
    inside = np.mean([((hd68.lower[v] <= hd68.contributions[v] + 1e-12) & (hd68.contributions[v] <= hd68.upper[v] + 1e-12)).to_numpy().mean()
                      for v in ("a", "b")])
    assert inside > 0.95


def test_boundary_transition_rows_are_held_and_errors():
    data = Dataset.example("sample_macro").derive(dy="400*dlog(y)", di="400*dlog(i)")
    fit = MSVAR(["dy", "di"], lags=1, regimes=2, switching="intercept").fit(data, starts=2)
    assert fit.transition[1].min() < 1e-6  # regime 1: one-period outliers, P[1, 1] at 0
    se = np.sqrt(np.diag(fit.parameter_covariance()))
    assert se[-1] == 0 and np.all(se[:-1] > 0)
    f = fit.fevd(1, 6, draws=50)
    assert "transition rows 1 at the boundary held fixed" in f.band_source
    np.testing.assert_allclose(fit.parameter_draws(20)["transition"][:, 1], np.broadcast_to(fit.transition[1], (20, 2)))
    with pytest.raises(SpecificationError, match="coverage"):
        fit.fevd(0, 6, draws=50, coverage=1.5)
    with pytest.raises(SpecificationError, match="at least 2 draws"):
        fit.irf(0, 6, draws=1)
    with pytest.raises(SpecificationError, match="fit it again"):
        dataclasses.replace(fit, sample_rows=None).fevd(0, 6, draws=10)
    # identical regimes: the likelihood is flat in the transition probabilities, so no bands
    same = dataclasses.replace(fit, intercepts=np.stack([fit.intercepts[0]] * 2), transition=np.array([[0.9, 0.1], [0.2, 0.8]]))
    with pytest.raises(SpecificationError, match="flat"):
        same.irf(0, 6, draws=10)
