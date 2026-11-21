"""Equations with ARMA errors or GARCH variance in models.

- Solved deterministically, a model of one such equation gives the equation's own dynamic forecast: ARMA errors
  (AR, ARMA and MA), ARMA with GARCH, GARCH alone, and after a gap between the estimation sample and the simulation.
- Stochastic simulation gives the equation's forecast variances (ARMA weights times the GARCH variance path; the
  formulas checked against arch and simulation in test_arma_garch.py), with normal, t and bootstrapped errors, and its
  means within Monte Carlo error."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Model
from pymetrix.errors import SpecificationError
from pymetrix.estimation.timeseries import ARMA, GARCH
from pymetrix.models.solvers import ModelError

H = 12


def _series(n=1500, seed=5):
    rng = np.random.default_rng(seed)
    e, h, u = np.zeros(n), np.full(n, 0.5), np.zeros(n)
    z, x = rng.standard_normal(n), rng.normal(size=n)
    for t in range(1, n):
        h[t] = 0.05 + 0.08 * e[t - 1] ** 2 + 0.88 * h[t - 1]
        e[t] = np.sqrt(h[t]) * z[t]
        u[t] = 0.5 * u[t - 1] + e[t] + 0.3 * e[t - 1]
    idx = pd.period_range("1900-01", periods=n, freq="M", name="time")
    full = Dataset.from_frame(pd.DataFrame({"y": 0.2 + 0.5 * x + u, "x": x}, index=idx))
    data = Dataset.from_frame(full.frame.assign(y=np.r_[full.frame.y.to_numpy()[:-H], np.full(H, np.nan)]))
    return full, data, (str(idx[-H]), str(idx[-1]))


@pytest.fixture(scope="module")
def series():
    return _series()


SPECS = [dict(errors=ARMA(1, 0)), dict(errors=ARMA(1, 1)), dict(errors=ARMA(0, 2)), dict(errors=ARMA(1, 1), variance=GARCH()),
         dict(variance=GARCH())]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: "+".join(s))
def test_a_model_solves_to_the_equations_forecast(series, spec):
    full, data, span = series
    fit = Equation("y ~ x", **spec).fit(Dataset.from_frame(full.frame.iloc[:-H]))
    expected = fit.forecast(full, span=span, dynamic=True)["forecast"].to_numpy()
    assert np.allclose(Model([fit]).solve(data, span).solved()["y"].to_numpy(), expected, atol=1e-10)


def test_after_a_gap_the_innovations_come_from_the_data(series):
    full, data, span = series
    fit = Equation("y ~ x", errors=ARMA(1, 1)).fit(Dataset.from_frame(full.frame.iloc[:-H - 40]))  # 40 periods before the span
    sol = Model([fit]).solve(data, span)
    assert np.allclose(sol.solved()["y"].to_numpy(), fit.forecast(full, span=span, dynamic=True)["forecast"].to_numpy(), atol=1e-10)
    assert "e_y" in sol.solved().names  # the innovations, a variable of the model (zero ahead)
    assert np.allclose(sol.solved()["e_y"].to_numpy(), 0.0)


@pytest.mark.parametrize("spec, errors", [(dict(errors=ARMA(1, 1)), "normal"), (dict(errors=ARMA(1, 1), variance=GARCH()), "normal"),
                                          (dict(variance=GARCH(dist="t")), "normal"), (dict(errors=ARMA(1, 1), variance=GARCH()), "bootstrap")],
                         ids=["arma", "arma-garch", "garch-t", "arma-garch-bootstrap"])
def test_stochastic_simulation_gives_the_forecast_variances(series, spec, errors):
    full, data, span = series
    fit = Equation("y ~ x", **spec).fit(Dataset.from_frame(full.frame.iloc[:-H]))
    forecast = fit.forecast(full, span=span, dynamic=True)
    draws = 20_000
    sol = Model([fit]).solve(data, span, draws=draws, seed=1, errors=errors)
    v = sol.simulation.values[:, :, sol.simulation.names.index("y")]
    se = forecast["se"].to_numpy()
    assert np.all(np.abs(v.mean(axis=0) - forecast["forecast"].to_numpy()) < 4 * se / np.sqrt(draws))
    assert np.allclose(v.var(axis=0), se ** 2, rtol=0.05)  # fat-tailed errors: a wider margin than four standard errors of normal ones
    if "variance" in spec:
        assert sol.simulation.info["GARCH"] == ["y"]


def test_refusals(series):
    full, data, span = series
    with pytest.raises(ModelError, match="fit it before using it in a model"):
        Model([Equation("y ~ x", errors=ARMA(1, 0))], params={"const": 0.0, "x": 0.5})
    fit = Equation("y ~ x", variance=GARCH()).fit(Dataset.from_frame(full.frame.iloc[:-H]))
    with pytest.raises(SpecificationError, match="GARCH variance, which sets the size"):
        Model([fit]).solve(data, span, draws=10, shock_sd={"y": 1.0})
