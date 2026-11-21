"""Dynamic forecasts of equations with lags of the dependent variable among the regressors and ARMA errors or GARCH
variance.

- y = c + b y[-1] + u with ARMA(1, 1) errors is an ARMA(2, 1) for y: its dynamic forecasts and standard errors against
  statsmodels' ARIMA(2, 0, 1) at the same parameters (exact Kalman filter), to 1e-6.
- y = c + b y[-1] with GARCH(1, 1) variance: forecasts and forecast variances against arch's AR(1)-GARCH(1, 1) at the
  same parameters (its analytic multi-step variances: impulse responses times the variance forecasts), to 1e-4.
- With other regressors, two lags and logs (dlog(y) ~ dlog(y)[-1] + x): the forecasts against the equation solved as
  a model, a separate code path, to 1e-10, and the variances against the model's stochastic simulation, within 5%.
- Regressors that depend on the dependent variable in other ways (y ~ d(y)[-1] + x) are solved as a model: the
  forecasts against the recursion written out by hand, without standard errors.
- Static forecasts of such equations (which were all missing) are one step ahead, with the innovation variance."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Model
from pymetrix.estimation.timeseries import ARMA, GARCH

H = 12


def _data(n=800, seed=2, garch=False, logs=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    e, h, u, y = np.zeros(n), np.full(n, 0.4), np.zeros(n), np.zeros(n)
    for t in range(2, n):
        h[t] = 0.05 + 0.1 * e[t - 1] ** 2 + 0.85 * h[t - 1] if garch else 1.0
        e[t] = np.sqrt(h[t]) * rng.standard_normal()
        u[t] = (0.0 if garch else 0.4 * u[t - 1] + 0.3 * e[t - 1]) + e[t]
        y[t] = 0.5 + 0.6 * y[t - 1] - 0.2 * y[t - 2] * logs + 0.4 * x[t] * logs + u[t]
    if logs:
        y = np.exp(np.cumsum(0.1 * (y - y.mean())))
    idx = pd.period_range("1900-01", periods=n, freq="M", name="time")
    return Dataset.from_frame(pd.DataFrame({"y": y, "x": x}, index=idx)), (str(idx[-H]), str(idx[-1]))


def test_arx_with_arma_errors_against_statsmodels_arima():
    from statsmodels.tsa.arima.model import ARIMA

    data, span = _data()
    estimation = Dataset.from_frame(data.frame.iloc[:-H])
    fit = Equation("y ~ y[-1]", errors=ARMA(1, 1)).fit(estimation)
    ours = fit.forecast(data, span, dynamic=True).frame
    v = fit.params.as_dict()
    b, phi, theta = v["y[-1]"], v["ar[1]"], v["ma[1]"]
    mean = v["const"] / (1 - b)
    model = ARIMA(estimation.frame.y.to_numpy(), order=(2, 0, 1), trend="c")
    res = model.smooth(np.array([mean, b + phi, -b * phi, theta, v["sigma2"]]))
    theirs = res.get_forecast(H)
    assert np.allclose(ours["forecast"].to_numpy(), theirs.predicted_mean, atol=1e-6)
    assert np.allclose(ours["se"].to_numpy(), theirs.se_mean, rtol=1e-6)
    static = fit.forecast(data, span).frame  # one step ahead from the actual lags
    assert np.all(np.isfinite(static["forecast"])) and np.allclose(static["se"], np.sqrt(v["sigma2"]), rtol=1e-6)


def test_arx_with_garch_against_arch():
    arch = pytest.importorskip("arch")
    data, span = _data(garch=True)
    estimation = Dataset.from_frame(data.frame.iloc[:-H])
    fit = Equation("y ~ y[-1]", variance=GARCH()).fit(estimation)
    ours = fit.forecast(data, span, dynamic=True).frame
    v = fit.params.as_dict()
    model = arch.arch_model(estimation.frame.y.to_numpy(), mean="AR", lags=1, vol="GARCH")
    res = model.fix([v["const"], v["y[-1]"], v["omega"], v["alpha[1]"], v["beta[1]"]])
    theirs = res.forecast(horizon=H, reindex=False)
    assert np.allclose(ours["forecast"].to_numpy(), theirs.mean.to_numpy()[-1], atol=1e-4)
    assert np.allclose(ours["variance"].to_numpy(), theirs.variance.to_numpy()[-1], rtol=1e-4)


@pytest.mark.parametrize("spec", [dict(errors=ARMA(1, 1)), dict(errors=ARMA(1, 0), variance=GARCH()), dict(variance=GARCH())],
                         ids=["arma", "arma-garch", "garch"])
def test_logs_two_lags_and_other_regressors_against_the_model(spec):
    data, span = _data(garch="variance" in spec, logs=True)
    fit = Equation("dlog(y) ~ dlog(y)[-1] + dlog(y)[-2] + x", **spec).fit(Dataset.from_frame(data.frame.iloc[:-H]))
    ours = fit.forecast(data, span, dynamic=True).frame
    model = Model([fit])
    solved = model.solve(data.frame.assign(y=np.r_[data.frame.y.to_numpy()[:-H], np.full(H, np.nan)]).pipe(Dataset.from_frame), span)
    y = pd.concat([data.frame.y.iloc[:-H], solved.solved().frame.y])
    assert np.allclose(ours["forecast"].to_numpy(), np.log(y).diff().to_numpy()[-H:], atol=1e-10)
    draws = 20_000
    sim = model.solve(Dataset.from_frame(data.frame.assign(y=y)), span, draws=draws, seed=3).simulation
    assert sim.info["failed"] == 0  # draws far from the deterministic path in levels: Newton steps halved, none dropped
    paths = np.diff(np.log(np.concatenate([np.broadcast_to(y.to_numpy()[-H - 1], (draws, 1)),
                                           sim.values[:, :, sim.names.index("y")]], axis=1)), axis=1)
    assert np.allclose(paths.var(axis=0), ours["se"].to_numpy() ** 2, rtol=0.05)


def test_other_functions_of_the_dependent_variable_are_solved_as_a_model():
    data, span = _data()
    fit = Equation("y ~ d(y)[-1] + x", errors=ARMA(1, 0)).fit(Dataset.from_frame(data.frame.iloc[:-H]))
    ours = fit.forecast(data, span, dynamic=True)
    v = fit.params.as_dict()
    y, x = data.frame.y.to_numpy()[:-H].tolist(), data.frame.x.to_numpy()
    n = len(y)
    u = y[-1] - v["const"] - v["d(y)[-1]"] * (y[-2] - y[-3]) - v["x"] * x[n - 1]
    for h in range(1, H + 1):
        y.append(v["const"] + v["d(y)[-1]"] * (y[-1] - y[-2]) + v["x"] * x[n - 1 + h] + v["ar[1]"] ** h * u)
    assert np.allclose(ours.frame["forecast"].to_numpy(), y[-H:], atol=1e-10)
    assert ours.frame["se"].isna().all() and "no standard errors" in ours.meta.description
