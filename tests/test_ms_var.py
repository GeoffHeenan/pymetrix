"""Markov-switching VARs, checked independently: a one-equation MSIH-AR(1) equal to statsmodels' MarkovRegression
(intercepts, slope, standard deviations, transition probabilities; its likelihood at our estimates is ours, and its
optimum is no higher), the likelihood at statsmodels' estimates through our filter equal to theirs; a simulated
two-regime VAR's regimes and transition matrix recovered, the smoothed probabilities tracking the true regimes; the
switching-everything model's likelihood no lower than the intercept-switching one it nests; the filter's
probabilities summing to one; refusals."""

import importlib

import numpy as np
import pandas as pd
import pytest

from pymetrix import MSVAR, Dataset
from pymetrix.errors import SpecificationError

ms = importlib.import_module("pymetrix.ms_var")
sm = pytest.importorskip("statsmodels.api")


def _regimes(T, seed):
    rng = np.random.default_rng(seed)
    P = np.array([[0.95, 0.05], [0.1, 0.9]])
    s = np.zeros(T, int)
    for t in range(1, T):
        s[t] = rng.choice(2, p=P[s[t - 1]])
    return s, rng


def test_one_equation_is_statsmodels_markov_regression():
    s, rng = _regimes(500, 0)
    y = np.zeros(500)
    for t in range(1, 500):
        y[t] = [-0.5, 1.5][s[t]] + 0.5 * y[t - 1] + [1.0, 0.5][s[t]] * rng.normal()
    data = Dataset.from_frame(pd.DataFrame({"y": y}, index=pd.period_range("1900Q1", periods=500, freq="Q")))
    fit = MSVAR(["y"], lags=1, switching="intercept_variance").fit(data)
    mod = sm.tsa.MarkovRegression(y[1:], k_regimes=2, exog=y[:-1], trend="c", switching_exog=False, switching_variance=True)
    # from its own start, without the random search (search_reps), statsmodels' fit is the same every run and reaches
    # the maximum here; the search added nothing but now and then a crash ("Steady-state probabilities could not be
    # constructed") and run-to-run noise of up to about 1e-6. Compare likelihoods where any such noise does not
    # matter: theirs at our estimates is ours, and theirs finds no higher
    ref = mod.fit(disp=False)
    ours = [fit.transition[0, 0], fit.transition[1, 0], *fit.intercepts[:, 0], fit.coefs[0, 0, 0], *fit.sigmas[:, 0, 0]]
    assert mod.loglike(np.array(ours)) == pytest.approx(fit.loglike, abs=1e-8)
    assert fit.loglike >= ref.llf - 1e-7
    p00, p10, c0, c1, b, v0, v1 = np.asarray(ref.params)
    if c0 > c1:  # our regimes are ordered by the intercept
        p00, p10, c0, c1, v0, v1 = 1 - p10, 1 - p00, c1, c0, v1, v0
    assert fit.intercepts[:, 0] == pytest.approx([c0, c1], abs=1e-3) and fit.coefs[0, 0, 0] == pytest.approx(b, abs=1e-3)
    assert np.sqrt(fit.sigmas[:, 0, 0]) == pytest.approx(np.sqrt([v0, v1]), abs=1e-3)
    assert fit.transition[:, 0] == pytest.approx([p00, p10], abs=1e-3)
    means = np.stack([np.asarray(ref.params)[2] + b * y[:-1], np.asarray(ref.params)[3] + b * y[:-1]])[:, :, None]
    dens = ms._densities(y[1:, None], means, np.array([[[np.asarray(ref.params)[5]]], [[np.asarray(ref.params)[6]]]]))
    Pm = np.array([[ref.params[0], 1 - ref.params[0]], [ref.params[1], 1 - ref.params[1]]])
    filt, _, ll = ms.hamilton_filter(dens, Pm)
    assert ll == pytest.approx(ref.llf, abs=1e-6) and filt.sum(axis=1) == pytest.approx(np.ones(499), abs=1e-12)


def test_a_two_regime_var():
    s, rng = _regimes(500, 1)
    Y = np.zeros((500, 2))
    for t in range(1, 500):
        c = np.array([[-0.5, 0.2], [1.0, -0.4]])[s[t]]
        Y[t] = c + np.array([[0.4, 0.1], [0.0, 0.3]]) @ Y[t - 1] + rng.normal(size=2) * 0.6
    data = Dataset.from_frame(pd.DataFrame(Y, columns=["a", "b"], index=pd.period_range("1900Q1", periods=500, freq="Q")))
    fit = MSVAR(["a", "b"], lags=1, switching="intercept").fit(data, starts=4)
    assert fit.intercepts == pytest.approx(np.array([[-0.5, 0.2], [1.0, -0.4]]), abs=0.15)
    assert fit.transition == pytest.approx(np.array([[0.95, 0.05], [0.1, 0.9]]), abs=0.05)
    assert np.corrcoef(fit.smoothed["regime 1"], s[1:])[0, 1] > 0.9
    full = MSVAR(["a", "b"], lags=1, switching="all").fit(data, starts=3)
    assert full.loglike >= fit.loglike - 1e-6  # it nests the intercept-switching model
    assert fit.irf(0, 8).values.shape == (9, 2, 2) and fit.durations[0] == pytest.approx(1 / (1 - fit.transition[0, 0]))
    with pytest.raises(SpecificationError, match="two regimes"):
        MSVAR(["a"], regimes=1)
