"""Estimation by simulation, checked independently: an MA(1) by indirect inference (AR(3) auxiliary) and by the
efficient method of moments near exact maximum likelihood (statsmodels); geometric Brownian motion's drift and
volatility by indirect inference equal to their closed-form maximum likelihood estimates as the paths grow; the
simulated method of moments recovering a normal's mean and standard deviation; the reported standard errors against
the spread of estimates across simulated samples; and refusals."""

import numpy as np
import pytest

from pymetrix import simulation_estimators as se
from pymetrix.errors import SpecificationError


def _ma1(theta=0.5, T=500, seed=0):
    e = np.random.default_rng(seed).normal(size=T + 1)
    return e[1:] + theta * e[:-1]


def _sim_ma(theta, shocks):
    return shocks[1:] + theta[0] * shocks[:-1]


def _ar3(y):
    return se.ar_coefficients(y, 3)[1:4]


def test_ma1_by_indirect_inference_and_emm_near_ml():
    from statsmodels.tsa.arima.model import ARIMA

    y = _ma1()
    ml = ARIMA(y, order=(0, 0, 1), trend="n").fit().params[0]
    ii = se.indirect_inference(y, _sim_ma, _ar3, [0.2], paths=20)
    emm = se.efficient_method_of_moments(y, _sim_ma, lambda z: se.ar_coefficients(z, 3), se.ar_score(3), [0.2], length=20)
    for fit in (ii, emm):
        assert fit.values[0] == pytest.approx(ml, abs=1.5 * fit.se[0]) and fit.J is not None and fit.J_df > 0
    assert ii.summary().title.endswith("indirect inference")


def test_gbm_by_indirect_inference_is_the_closed_form():
    rng = np.random.default_rng(1)
    mu, sigma, dt, T = 0.08, 0.2, 1 / 252, 2000
    r = (mu - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(T)  # log returns of geometric Brownian motion

    def simulate(theta, shocks):  # the exact discretisation
        return (theta[0] - theta[1] ** 2 / 2) * dt + abs(theta[1]) * np.sqrt(dt) * shocks[1:]

    fit = se.indirect_inference(r, simulate, lambda z: np.array([z.mean(), z.var()]), [0.0, 0.1], paths=200, weight="identity")
    s_ml = r.std()  # maximum likelihood: the log returns are normal
    mu_ml = (r.mean() + r.var() / 2) / dt
    assert abs(fit.values[1]) == pytest.approx(s_ml / np.sqrt(dt), rel=0.02)
    assert fit.values[0] == pytest.approx(mu_ml, abs=0.05)


def test_simulated_moments_recover_a_normal():
    x = np.random.default_rng(2).normal(2.0, 1.5, 800)
    fit = se.simulated_moments(x, lambda th, s: th[0] + abs(th[1]) * s[1:], lambda z: np.column_stack([z, z**2, z**3]),
                               [1.0, 1.0], paths=50, names=["mu", "sigma"])
    assert fit.values[0] == pytest.approx(x.mean(), abs=2 * fit.se[0] / 3) and abs(fit.values[1]) == pytest.approx(x.std(), rel=0.03)
    assert fit.names == ("mu", "sigma") and fit.J_df == 1


def test_standard_errors_against_the_spread_of_estimates():
    estimates, errors = [], []
    for seed in range(40):
        fit = se.indirect_inference(_ma1(T=300, seed=100 + seed), _sim_ma, _ar3, [0.3], paths=10, seed=seed)
        estimates.append(fit.values[0])
        errors.append(fit.se[0])
    assert np.mean(errors) == pytest.approx(np.std(estimates, ddof=1), rel=0.35)
    assert np.mean(estimates) == pytest.approx(0.5, abs=0.05)


def test_refusals():
    y = _ma1(T=200)
    with pytest.raises(SpecificationError, match="cannot identify"):
        se.indirect_inference(y, lambda th, s: s[1:] + th[0] * s[:-1] + th[1], lambda z: np.array([z.var()]), [0.1, 0.0])
    with pytest.raises(SpecificationError, match="weight"):
        se.indirect_inference(y, _sim_ma, _ar3, [0.2], weight="optimal")


def _all_three(data):
    """The three estimators on ``data``, short runs: their estimates, standard errors and J statistics."""
    moments = lambda z: np.column_stack([z, z**2, np.r_[z[1:] * z[:-1], 0.0]])  # noqa: E731
    fits = (se.indirect_inference(data, _sim_ma, _ar3, [0.2], paths=4, maxiter=400),
            se.simulated_moments(data, _sim_ma, moments, [0.2], paths=4, maxiter=400),
            se.efficient_method_of_moments(data, _sim_ma, lambda z: se.ar_coefficients(z, 2), se.ar_score(2), [0.2], length=3,
                                           maxiter=400))
    return [(f.values, f.se, f.J, f.nobs) for f in fits]


def test_a_series_a_dataset_column_or_a_one_column_frame_equal_the_array():
    """A pandas Series (with periods or without), a dataset's column, a one-column DataFrame or a one-series Dataset
    give what the same numbers as an ndarray give. Before, the recorded call turned pandas data into a Dataset, which
    np.asarray could not read ("setting an array element with a sequence"), or refused a Series without periods."""
    import pandas as pd

    from pymetrix import Dataset

    y = _ma1(T=200, seed=3)
    periods = pd.period_range("2000Q1", periods=len(y), freq="Q", name="time")
    data = Dataset.from_frame(pd.DataFrame({"y": y, "x": np.arange(len(y), dtype=float)}, index=periods))
    want = _all_three(y)
    for given in (pd.Series(y, index=periods, name="y"), pd.Series(y), data["y"], pd.DataFrame({"y": y}, index=periods),
                  pd.DataFrame({"y": y}), data.keep(["y"])):
        for (v, s, J, n), (v0, s0, J0, n0) in zip(_all_three(given), want):
            np.testing.assert_allclose(v, v0, rtol=0, atol=1e-12)
            np.testing.assert_allclose(s, s0, rtol=0, atol=1e-12)
            assert n == n0 and (J is None) == (J0 is None) and (J is None or abs(J - J0) < 1e-12)
