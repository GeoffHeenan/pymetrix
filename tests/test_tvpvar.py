"""TVP-VARs with stochastic volatility, checked independently: Kim, Shephard and Chib's mixture against the exact
log chi-squared(1) density and its mean; the forward-filtering backward-sampling draws against the Kalman smoother's
mean and variance; a volatility break recovered from a simulated VAR; coefficients averaging near least squares; the responses at a date from the kept draws by hand."""

import importlib

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from pymetrix import Dataset, TVPVAR

tv = importlib.import_module("pymetrix.tvpvar")


def test_the_ksc_mixture():
    exact = lambda x: np.exp(x / 2 - np.exp(x) / 2) / np.sqrt(2 * np.pi)  # noqa: E731 - the density of log chi-squared(1)
    x = np.linspace(-15, 4, 4001)
    mix = sum(w * stats.norm.pdf(x, m, np.sqrt(v)) for w, m, v in zip(tv.KSC_WEIGHTS, tv.KSC_MEANS, tv.KSC_VARIANCES))
    assert np.max(np.abs(mix - exact(x))) < 0.011 and tv.KSC_WEIGHTS.sum() == pytest.approx(1, abs=1e-12)
    assert float(tv.KSC_WEIGHTS @ tv.KSC_MEANS) == pytest.approx(integrate.quad(lambda u: u * exact(u), -60, 10)[0], abs=1e-4)


def test_ffbs_against_the_kalman_smoother():
    from pymetrix.statespace import SystemMatrices, kalman_filter, kalman_smoother

    rng = np.random.default_rng(0)
    T = 60
    y = np.cumsum(rng.normal(0, 0.3, T)) + rng.normal(0, 1, T)
    m = SystemMatrices(("s",), ("y",), ("e",), np.eye(1), np.zeros(1), np.eye(1), np.array([[0.09]]), np.eye(1), np.zeros(1), np.eye(1))
    kf = kalman_filter(y[:, None], m, np.zeros(1), np.array([[10.09]]))
    mean, var = kalman_smoother(kf, m)
    draws = np.array([tv.ffbs(y[:, None], np.ones((T, 1, 1)), np.ones((T, 1, 1)), np.array([[0.09]]), np.zeros(1), np.array([[10.0]]), rng)[:, 0]
                      for _ in range(4000)])
    se = np.sqrt(var[:, 0, 0] / 4000)
    assert np.all(np.abs(draws.mean(axis=0) - mean[:, 0]) < 4.5 * se)
    assert np.median(draws.var(axis=0) / var[:, 0, 0]) == pytest.approx(1, abs=0.05)


def _simulated(T=240, seed=0):
    rng = np.random.default_rng(seed)
    sd1 = np.where(np.arange(T) < 140, 0.5, 1.5)  # the first shock's volatility triples
    A = np.array([[0.5, 0.1], [0.2, 0.4]])
    y = np.zeros((T, 2))
    for t in range(1, T):
        y[t] = A @ y[t - 1] + np.array([[1, 0], [0.5, 1]]) @ (np.array([sd1[t], 1.0]) * rng.normal(size=2))
    return Dataset.from_frame(pd.DataFrame(y, columns=["a", "b"], index=pd.period_range("1950Q1", periods=T, freq="Q", name="time")))


@pytest.fixture(scope="module")
def fit():
    return TVPVAR(["a", "b"], lags=1).fit(_simulated(), draws=250, burn=250, training=40, seed=1)


def test_the_volatility_break_is_found(fit):
    v = fit.volatilities()
    t = np.arange(len(v)) + 41  # positions in the simulation
    early, late = v["a"][t < 120], v["a"][t > 165]
    assert early.median() == pytest.approx(0.5, abs=0.15) and late.median() == pytest.approx(1.5, abs=0.4)
    b_early, b_late = v["b"][t < 120].median(), v["b"][t > 165].median()
    assert v["b"].median() == pytest.approx(1.0, abs=0.2) and 0.6 < b_late / b_early < 1.6  # the other shock does not triple


def test_the_responses_by_hand(fit):
    from pymetrix.var import _ma

    irf = fit.irf(str(fit.index[100]), periods=3, coverage=0.5)
    d, t, K = 7, 100, 2
    B = fit.beta[d, t].reshape(K, fit.m).T
    A1 = B[1:3].T
    Am = np.eye(2)
    Am[1, 0] = fit.a[d, t, 0]
    one = _ma(A1[None], 3) @ (np.linalg.inv(Am) @ np.diag(np.exp(fit.h[d, t])))
    sims = []
    for k in range(len(fit.beta)):
        Bk = fit.beta[k, t].reshape(K, fit.m).T
        Ak = np.eye(2)
        Ak[1, 0] = fit.a[k, t, 0]
        sims.append(_ma(Bk[1:3].T[None], 3) @ (np.linalg.inv(Ak) @ np.diag(np.exp(fit.h[k, t]))))
    assert irf.values == pytest.approx(np.median(np.stack(sims), axis=0)) and one.shape == irf.values.shape


def test_coefficients_centre_on_least_squares_and_the_dialog():
    data = _simulated(seed=2)
    fit = TVPVAR(["a", "b"], lags=1).fit(data, draws=150, burn=150, training=40, seed=2)
    Y = data.frame[["a", "b"]].to_numpy()[41:]  # the estimation sample: after the lag and the training periods
    X = np.column_stack([np.ones(len(Y)), data.frame[["a", "b"]].to_numpy()[40:-1]])
    ols = np.linalg.lstsq(X, Y, rcond=None)[0].T.reshape(-1)  # equation by equation, as the draws are
    average = np.median(fit.beta, axis=0).mean(axis=0)
    assert average == pytest.approx(ols, abs=0.25)  # the prior from 40 training periods pulls the paths a little
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS
    from pymetrix.views import sections

    s = Session()
    s.objects["data"] = _simulated(T=120, seed=3)
    s.run(COMMANDS["FitTVPVAR"].cls("a, b", draws=20, burn=20))
    assert s.history[-1].code.startswith("tvp = TVPVAR(['a', 'b']")
    assert [t for t, _ in sections(s.objects["tvp"])][0] == "Summary"


def test_the_responses_over_time(fit):
    """Every date's responses at once: at each date, the median over the kept draws of what irf gives there; with fewer
    draws, the median of those evenly spaced draws. Drawn as a surface per response and shock, horizons by dates."""
    from pymetrix.figures import Surface
    from pymetrix.var import _ma
    from pymetrix.views import view

    paths = fit.irf_paths(periods=6, draws=len(fit.beta))
    assert paths.shape == (len(fit.index), 7, 2, 2)
    for t in (0, 77, len(fit.index) - 1):
        assert paths[t] == pytest.approx(fit.irf(str(fit.index[t]), periods=6).values)
    few = fit.irf_paths(periods=2, draws=5)
    picked = np.unique(np.linspace(0, len(fit.beta) - 1, 5).round().astype(int))
    t = 30
    sims = []
    for k in picked:  # by hand, from the five draws
        B = fit.beta[k, t].reshape(2, fit.m).T
        Am = np.eye(2)
        Am[1, 0] = fit.a[k, t, 0]
        sims.append(_ma(B[1:3].T[None], 2) @ (np.linalg.inv(Am) @ np.diag(np.exp(fit.h[k, t]))))
    assert few[t] == pytest.approx(np.median(np.stack(sims), axis=0))
    table = fit.irf_over_time("b", "a", periods=6, draws=len(fit.beta))
    assert table.shape == (len(fit.index), 7) and list(table.columns) == list(range(7)) and table.index.equals(fit.index.rename("date"))
    assert table.to_numpy() == pytest.approx(paths[:, :, 1, 0])
    fig = view(fit, "Responses over time")
    assert len(fig.panels) == 4 and fig.columns == 2 and [p.title for p in fig.panels][1] == "a, shock to b"
    surface = fig.panels[2].layers[0]
    assert isinstance(surface, Surface) and surface.values.to_numpy() == pytest.approx(fit.irf_over_time("b", "a").to_numpy())
    drawn = fig.draw()
    assert sum(a.name == "3d" for a in drawn.axes) == 4
    with pytest.raises(ValueError, match="no variable 'c'"):
        fit.irf_over_time("c", "a")
