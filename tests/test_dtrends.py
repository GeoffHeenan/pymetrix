"""Deterministic trends in measurement equations (IRIS !dtrends): the filter, the likelihood and forecasts with a
trend equal the same model on data with the trend taken out by hand."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, read_iris, templates
from pymetrix.models import MeasurementMap, ModelError

TREND = {"rate": "0.3 + 0.05*ttrend"}


@pytest.fixture(scope="module")
def setup():
    nk = templates.new_keynesian()
    truth = nk.solve().simulate(80, nk.shock_sd, seed=7)
    idx = pd.period_range("2000Q1", periods=80, freq="Q", name="time")
    truth.index = idx
    plain = nk.measurement.apply(Dataset.from_frame(truth))
    ttrend = np.arange(80) - 4  # trend_base 2001Q1: ttrend is -4 in 2000Q1
    trended = plain.with_series(rate=plain["rate"] + 0.3 + 0.05 * ttrend)
    m = nk.measurement
    with_trend = MeasurementMap(m.equations, m.params, m.errors, m.sources, m.units, m.labels, TREND, "2001Q1")
    return nk, plain, trended, with_trend


def test_the_filter_and_likelihood_take_the_trend_out(setup):
    nk, plain, trended, with_trend = setup
    sol = nk.solve()
    a = sol.system(nk.measurement, nk.shock_sd)
    b = sol.system(with_trend, nk.shock_sd)
    assert b.loglike(trended) == pytest.approx(a.loglike(plain), abs=1e-9)
    fa, fb = a.filter(plain).frame, b.filter(trended).frame
    for v in ("y", "pi", "i"):
        assert np.allclose(fa[v], fb[v], atol=1e-10)
    assert np.allclose(fb["rate_fitted"] - fa["rate_fitted"], trended["rate"] - plain["rate"], atol=1e-10)
    c = b.contributions(trended)
    assert np.allclose(c.contributions["y"].sum(axis=1), fb["y"] - sol.model.steady.values["y"], atol=1e-10)


def test_estimation_and_forecasts_with_the_trend(setup):
    nk, plain, trended, with_trend = setup
    fa = nk.estimate(plain, estimate=["rho"])
    fb = nk.estimate(trended, estimate=["rho"], measurement=with_trend)
    assert fb.params["rho"] == pytest.approx(fa.params["rho"], abs=1e-6)
    span = ("2020Q1", "2020Q4")
    ga, gb = fa.forecast(span).frame, fb.forecast(span).frame
    periods = pd.period_range("2020Q1", periods=4, freq="Q")
    expected_trend = 0.3 + 0.05 * (periods.asi8 - pd.Period("2001Q1", freq="Q").ordinal)
    assert np.allclose(gb.loc[periods, "rate"].to_numpy() - ga.loc[periods, "rate"].to_numpy(), expected_trend, atol=1e-6)


def test_iris_dtrends_and_errors():
    text = """
    !transition_variables x
    !transition_shocks e
    !parameters rho = 0.5, g = 0.25, c = 1
    !measurement_variables obs
    !transition_equations
        x = rho*x{-1} + e;
    !measurement_equations
        obs = x;
    !dtrends
        obs += c;
        obs += g*ttrend;
    """
    t = read_iris(text, trend_base="2000Q1")
    assert dict(t.measurement.trends) == {"obs": "(c) + (g*ttrend)"} and t.measurement.trend_base == "2000Q1"
    idx = pd.period_range("2000Q1", periods=4, freq="Q")
    assert np.allclose(t.measurement.trend_values(idx)[:, 0], 1 + 0.25 * np.arange(4))
    with pytest.raises(ModelError, match="trend_base"):
        read_iris(text)
    with pytest.raises(ModelError, match="parameters and ttrend only"):
        MeasurementMap(["obs = x"], trends={"obs": "x + 1"})
    with pytest.raises(ModelError, match="trends given for"):
        MeasurementMap(["obs = x"], trends={"other": "1"})
