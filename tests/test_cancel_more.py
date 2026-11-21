"""Cancel reaches the other long computations: model estimation, state-space fits, optimal simple rules, seasonal
adjustment (between series) and the Kalman filter. Each reports as it goes, and stops once cancelled."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Model, osr
from pymetrix.metadata import Role
from pymetrix.progress import Cancelled, task


def _quarters(n: int) -> pd.PeriodIndex:
    return pd.period_range("1950Q1", periods=n, freq="Q", name="time")


@pytest.fixture(scope="module")
def ar1():
    rng = np.random.default_rng(1)
    x = np.zeros(120)
    for t in range(1, 120):
        x[t] = 0.7 * x[t - 1] + 0.5 * rng.normal()
    data = Dataset.from_frame(pd.DataFrame({"x": x}, index=_quarters(120)))
    model = Model([Equation("x ~ rho*x[-1] + e", params=["rho"])], params={"rho": 0.3}, roles={"e": Role.SHOCK})
    return model, data


def _stops(run, says: str) -> None:
    """Run reports what it does (``says``) and raises Cancelled once Cancel is pressed."""
    with task() as t:
        run()
        assert says in t.message
    with task() as t:
        t.cancel()
        with pytest.raises(Cancelled):
            run()


def test_model_estimation_stops(ar1):
    model, data = ar1
    _stops(lambda: model.estimate(data, ["x"], estimate=["rho"], shock_sd={"e": 0.5}), "likelihood evaluation")


def test_a_state_space_fit_stops(ar1):
    from pymetrix import StateSpace

    _, data = ar1
    ss = StateSpace(["s"], ["s = rho*s[-1] + e"], ["x = s"], {"e": "sd"}, ["rho", "sd"])
    _stops(lambda: ss.fit(data), "likelihood evaluation")


def test_an_optimal_simple_rule_search_stops():
    gali = Model([Equation("pi ~ bet*pi[+1] + kap*x + u", params=["bet", "kap"]),
                  Equation("x ~ x[+1] - sig*(i - pi[+1])", params=["sig"]),
                  Equation("u ~ rho*u[-1] + eu", params=["rho"]),
                  Equation("i ~ phi*pi", params=["phi"])],
                 params=dict(bet=0.99, kap=0.1, sig=1.0, rho=0.8, phi=1.5), roles={"eu": Role.SHOCK})
    _stops(lambda: osr(gali, ["phi"], {"pi": 1.0, "x": 0.25}, bounds={"phi": (1.01, 50.0)}), "rule coefficients")


def test_seasonal_adjustment_stops_between_series():
    pytest.importorskip("statsmodels")
    from pymetrix import STL

    idx = pd.period_range("2005-01", "2024-12", freq="M", name="time")
    factor = np.tile(1 + 0.1 * np.sin(2 * np.pi * np.arange(12) / 12), len(idx) // 12)
    data = Dataset.from_frame(pd.DataFrame({"sales": 100 * factor, "other": 50 * factor}, index=idx))
    _stops(lambda: data.seasonal_adjust(["sales", "other"], STL(), derived=False), "seasonal adjustment: other")


def test_the_kalman_filter_stops_on_a_long_sample():
    from pymetrix import StateSpace
    from pymetrix.statespace import kalman_filter

    system = StateSpace(["s"], ["s = rho*s[-1] + e"], ["x = s"], {"e": 1.0}, ["rho"]).system({"rho": 0.5})
    y = np.zeros((200, 1))
    a0, P0 = np.zeros(1), np.eye(1)
    kalman_filter(y, system, a0, P0)  # outside a task: no checks to speak of
    with task() as t:
        t.cancel()
        with pytest.raises(Cancelled):
            kalman_filter(y, system, a0, P0)
