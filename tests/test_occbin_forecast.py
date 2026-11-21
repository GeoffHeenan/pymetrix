"""Forecasts that respect occasionally binding constraints: FittedModel.forecast(..., occbin=True), by OccBin from the
OccBin-smoothed state (the default for a model estimated with the OccBin filter).

Checked independently: with an anticipated shock path, the forecast is the nonlinear perfect-foresight solution
(stacked Newton on the model with its max) from the same state; a constraint that never binds gives the linear
forecast; the floor holds in the forecast and in every draw, where the linear forecast breaks it."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, PerfectForesight, Scenario, occbin, templates
from pymetrix.errors import SpecificationError
from pymetrix.models import MeasurementMap, ModelError

NOISY = MeasurementMap.direct(["y", "pi", "i"], {"y": 0.02, "pi": 0.02, "i": 0.02})
SPAN = ("2025Q1", "2027Q4")
RECESSION = Scenario(set={"ey": {"2025Q1": -3.0, "2025Q2": -3.0}})  # announced: the floor binds for a while


def _data(template, seed=1, n=100):
    rng = np.random.default_rng(seed)
    shocks = {s: {t + 1: float(v) for t, v in enumerate(rng.normal(0, template.shock_sd[s], n))} for s in template.shocks}
    path = occbin(template, shocks, periods=n)
    frame = path.frame[["y", "pi", "i"]].copy()
    frame.index = pd.period_range("2000Q1", periods=n, freq="Q", name="time")
    return Dataset.from_frame(frame + rng.normal(0, 0.02, frame.shape))


@pytest.fixture(scope="module")
def fitted():
    zlb = templates.new_keynesian(lower_bound=2.0)
    return zlb, zlb.estimate(_data(zlb), ["rho"], measurement=NOISY, occbin=True)


def test_the_forecast_is_the_nonlinear_perfect_foresight_solution(fitted):
    _, fit = fitted
    forecast = fit.forecast(SPAN, RECESSION)
    assert forecast.provenance.method["type"] == "piecewise-linear (OccBin)"
    assert forecast.provenance.method["binding"]["policy (max)"]  # the floor binds in the forecast
    history = fit.smoothed.frame[["y", "pi", "i"]]
    full = pd.period_range(history.index[0], "2032Q4", freq="Q", name="time")
    shells = Dataset.from_frame(history.reindex(full).assign(ey=0.0, epi=0.0, ei=0.0))
    exact = fit.model.solve(shells, ("2025Q1", "2032Q4"), RECESSION, solver=PerfectForesight(terminal="steady_state"))
    np.testing.assert_allclose(forecast.frame.loc["2025Q1":"2027Q4", ["y", "pi", "i"]].to_numpy(),
                               exact.frame.loc["2025Q1":"2027Q4", ["y", "pi", "i"]].to_numpy(), atol=1e-8)


def test_the_floor_holds_where_the_linear_forecast_breaks_it(fitted):
    _, fit = fitted
    floor = fit.model.parameters["i_floor"].value
    constrained = fit.forecast(SPAN, RECESSION)
    linear = fit.forecast(SPAN, RECESSION, occbin=False)
    assert constrained["i"].loc["2025Q1":].min() >= floor - 1e-10
    assert linear["i"].loc["2025Q1":].min() < floor - 0.05
    assert "y" in constrained.frame.columns and "pi" in constrained.frame.columns  # observed series through the measurement


def test_a_constraint_that_never_binds_gives_the_linear_forecast(fitted):
    zlb, _ = fitted
    far = templates.new_keynesian(lower_bound=-100.0)
    fit = far.estimate(_data(zlb), ["rho"], measurement=NOISY, occbin=True)
    constrained, linear = fit.forecast(SPAN, RECESSION), fit.forecast(SPAN, RECESSION, occbin=False)
    np.testing.assert_allclose(constrained.frame.loc["2025Q1":, ["y", "pi", "i"]].to_numpy(),
                               linear.frame.loc["2025Q1":, ["y", "pi", "i"]].to_numpy(), atol=1e-10)


def test_draws_respect_the_floor(fitted):
    _, fit = fitted
    floor = fit.model.parameters["i_floor"].value
    first = fit.forecast(SPAN, draws=50, seed=4)
    again = fit.forecast(SPAN, draws=50, seed=4)
    i = first.simulation.names.index("i")
    assert first.simulation.values.shape == (50, 12, 3) and first.simulation.info["failed"] == 0
    assert np.min(first.simulation.values[:, :, i]) >= floor - 1e-10
    assert np.min(first.simulation.values[:, :, i]) <= floor + 1e-10  # some draws reach it
    np.testing.assert_array_equal(first.simulation.values, again.simulation.values)


def test_default_follows_the_estimation_and_refusals(fitted):
    zlb, fit = fitted
    assert fit.forecast(SPAN).provenance.method["type"] == "piecewise-linear (OccBin)"
    linear_fit = zlb.estimate(_data(zlb), ["rho"], measurement=NOISY)
    assert linear_fit.forecast(SPAN).provenance.method["type"] != "piecewise-linear (OccBin)"
    assert linear_fit.forecast(SPAN, occbin=True).provenance.method["type"] == "piecewise-linear (OccBin)"
    with pytest.raises(ModelError, match="cannot hold the targets"):  # below the floor: the policy shock does nothing there
        fit.forecast(SPAN, Scenario(hold={"i": {"2025Q1": -1.0}}, free=["ei"]))
    with pytest.raises(ModelError, match="fix=, adjust="):
        fit.forecast(SPAN, Scenario(fix={"i": {"2025Q1": 0.0}}))
    with pytest.raises(SpecificationError, match="tunes"):
        fit.forecast(SPAN, data=_data(zlb), tunes={"y": {"2024Q4": 0.0}})
    with pytest.raises(ModelError, match="all anticipated or all unanticipated"):
        fit.forecast(SPAN, Scenario(set={"ey": {"2025Q1": -1.0}, "epi": {"2025Q1": 0.5}}, unanticipated=["ey"]))
    with pytest.raises(ModelError, match="the period after the data end"):
        fit.forecast(("2026Q1", "2026Q4"))


def test_the_dialog(fitted):
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import ForecastModel

    _, fit = fitted
    s = Session()
    s.objects["estimated"] = fit
    s.run(ForecastModel("estimated", "2025Q1", "2027Q4", set_to="ey[2025Q1:2025Q2]=-3", name="fc"))
    assert s["fc"].provenance.method["type"] == "piecewise-linear (OccBin)"
    s.run(ForecastModel("estimated", "2025Q1", "2027Q4", set_to="ey[2025Q1:2025Q2]=-3", occbin=False, name="linear"))
    assert "occbin=False" in s.script() and s["linear"]["i"].min() < s["fc"]["i"].min()
