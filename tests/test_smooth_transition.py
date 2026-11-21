"""Smooth-transition autoregressions, checked independently: known LSTAR and ESTAR parameters recovered from simulated
processes; the concentrated sum of squares equal to the fitted model's residuals; with a very fast transition, LSTAR's
sum of squares equal to the threshold AR's at the same threshold; the Gauss-Newton covariance against the numerical
Hessian of the sum of squares (but for gamma's); the one-step forecast's mean against the model's prediction."""

import importlib

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, SmoothTransitionAR, ThresholdAR
from pymetrix.errors import SpecificationError

st = importlib.import_module("pymetrix.smooth_transition")
PHI1, PHI2 = np.array([0.5, 0.8]), np.array([-0.5, 0.2])


def _simulate(kind, gamma, T=2000, seed=1):
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    for t in range(1, T):
        G = st.transition(y[t - 1], gamma, 0.0, 1.0, kind)
        y[t] = (PHI1 @ [1, y[t - 1]]) * (1 - G) + (PHI2 @ [1, y[t - 1]]) * G + rng.normal()
    return Dataset.from_frame(pd.DataFrame({"y": y}, index=pd.period_range("1500Q1", periods=T, freq="Q", name="time")))


@pytest.mark.parametrize("kind, gamma", [("logistic", 4.0), ("exponential", 1.5)])
def test_known_parameters_recovered(kind, gamma):
    fit = SmoothTransitionAR("y", lags=1, transition=kind).fit(_simulate(kind, gamma))
    se = np.sqrt(np.diag(fit.cov))
    assert np.all(np.abs(fit.phi1 - PHI1) < 3.5 * se[:2]) and np.all(np.abs(fit.phi2 - PHI2) < 3.5 * se[2:4])
    raw = fit.gamma / (fit.scale if kind == "logistic" else fit.scale**2)  # back in the simulation's units
    assert abs(raw - gamma) < 3.5 * se[4] / (fit.scale if kind == "logistic" else fit.scale**2) and abs(fit.c) < 3.5 * se[5]
    assert fit.ssr < fit.ssr_linear and fit.converged


def test_the_sums_of_squares():
    data = _simulate("logistic", 4.0, T=800, seed=2)
    fit = SmoothTransitionAR("y", lags=1).fit(data)
    _, y, X, q = fit.spec._arrays(data, None)
    s = q[1]
    G = st.transition(s, fit.gamma, fit.c, fit.scale)
    resid = y - (X @ fit.phi1) * (1 - G) - (X @ fit.phi2) * G
    assert resid @ resid == pytest.approx(fit.ssr, rel=1e-10)
    tar = ThresholdAR("y", lags=1, delay=1).fit(data)  # a very fast transition is a threshold
    fast = st._concentrated(y, X, s, 1e12, tar.threshold + 1e-7, fit.scale, "logistic")[0]  # G is 0 or 1 at every observation
    assert fast == pytest.approx(tar.ssr, rel=1e-8)


def test_the_covariance_against_the_hessian():
    data = _simulate("logistic", 4.0, T=1500, seed=3)
    fit = SmoothTransitionAR("y", lags=1).fit(data)
    _, y, X, q = fit.spec._arrays(data, None)
    s = q[1]
    theta = np.concatenate([fit.phi1, fit.phi2, [fit.gamma, fit.c]])

    def ssr(t):
        G = st.transition(s, t[4], t[5], fit.scale)
        e = y - (X @ t[:2]) * (1 - G) - (X @ t[2:4]) * G
        return e @ e

    H = np.zeros((6, 6))
    h = 1e-4 * np.maximum(np.abs(theta), 1.0)
    for i in range(6):
        for j in range(6):
            ei, ej = np.eye(6)[i] * h[i], np.eye(6)[j] * h[j]
            H[i, j] = (ssr(theta + ei + ej) - ssr(theta + ei - ej) - ssr(theta - ei + ej) + ssr(theta - ei - ej)) / (4 * h[i] * h[j])
    newton = fit.sigma2 * np.linalg.inv(H / 2)  # at the minimum, J'J is half the Hessian, to first order
    keep = [0, 1, 2, 3, 5]  # not gamma: the sum of squares is far from quadratic in it, as the summary warns
    assert np.sqrt(np.diag(fit.cov))[keep] == pytest.approx(np.sqrt(np.diag(newton))[keep], rel=0.1)


def test_the_forecast_and_the_dialog():
    data = _simulate("logistic", 4.0, T=600, seed=4)
    fit = SmoothTransitionAR("y", lags=1).fit(data)
    last = data.frame.index[-1]
    fc = fit.forecast(data, (str(last + 1), str(last + 3)), draws=20000).frame
    y_last = data.frame["y"].iloc[-1]
    G = st.transition(y_last, fit.gamma, fit.c, fit.scale)
    one = (fit.phi1 @ [1, y_last]) * (1 - G) + (fit.phi2 @ [1, y_last]) * G
    assert fc["forecast"].iloc[0] == pytest.approx(one, abs=4 * np.sqrt(fit.sigma2 / 20000))
    with pytest.raises(SpecificationError, match="logistic"):
        SmoothTransitionAR("y", transition="quadratic")
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS
    from pymetrix.views import sections

    s = Session()
    s.objects["data"] = data
    s.run(COMMANDS["FitSmoothTransitionAR"].cls("y", lags=1))
    assert s.history[-1].code == "star = SmoothTransitionAR('y').fit(data)"
    assert [t for t, _ in sections(s.objects["star"])][0] == "Summary"
