"""Seasonal ARIMA, the theta method and STL forecasting (through statsmodels), checked independently: the airline
model against R's published estimates and forecasts, ARIMA forecasts and their variances by hand from the
estimated parameters, the theta method's drift against a least-squares trend, and STL forecasts as the adjusted
series' forecast plus last year's seasonal component."""

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from pymetrix import SARIMA, STLForecast, Dataset, Theta, backtest  # noqa: E402
from pymetrix.errors import SpecificationError  # noqa: E402

# Box and Jenkins' international airline passengers, monthly, 1949-1960 (R's AirPassengers)
AIR = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
       145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166, 171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
       196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201, 204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
       242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278, 284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
       315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336, 340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
       360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405, 417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432]


@pytest.fixture(scope="module")
def air():
    idx = pd.period_range("1949-01", periods=len(AIR), freq="M", name="time")
    return Dataset.from_frame(pd.DataFrame({"air": AIR, "lair": np.log(AIR)}, index=idx))


@pytest.fixture(scope="module")
def ar_data():
    rng = np.random.default_rng(7)
    n = 160
    y = np.zeros(n)
    e = rng.normal(0, 1, n)
    for t in range(5, n):  # (1 - 0.5L)(1 - 0.3L^4)(y - 10) = e
        y[t] = 0.5 * y[t - 1] + 0.3 * y[t - 4] - 0.15 * y[t - 5] + e[t]
    idx = pd.period_range("1980Q1", periods=n, freq="Q", name="time")
    x = rng.normal(0, 1, n)
    return Dataset.from_frame(pd.DataFrame({"y": 10 + y, "yx": 10 + y + 2 * x, "x": x}, index=idx))


def _fit(spec, data, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return spec.fit(data, **kw)


def test_the_airline_model_matches_r(air):
    # R: arima(log(AirPassengers), c(0, 1, 1), seasonal = c(0, 1, 1)) gives ma1 -0.4018 (s.e. 0.0896), sma1 -0.5569
    # (0.0731), sigma^2 0.001348, log likelihood 244.7, AIC -483.4; predict(n.ahead = 1) 6.110186 (s.e. 0.03671)
    fit = _fit(SARIMA("lair", (0, 1, 1), (0, 1, 1)), air)
    table = fit.params_table()
    est = dict(zip(table.index, (row[0] for row in table.rows)))
    se = dict(zip(table.index, (row[1] for row in table.rows)))
    assert est["MA(1)"] == pytest.approx(-0.4018, abs=5e-4) and est["seasonal MA(12)"] == pytest.approx(-0.5569, abs=5e-4)
    assert se["MA(1)"] == pytest.approx(0.0896, abs=5e-4) and se["seasonal MA(12)"] == pytest.approx(0.0731, abs=5e-4)
    assert fit.stats["sigma2"] == pytest.approx(0.001348, rel=2e-3) and fit.stats["loglike"] == pytest.approx(244.7, abs=0.05)
    assert fit.stats["aic"] == pytest.approx(-483.4, abs=0.05)
    first = fit.forecast(horizon=1).frame.iloc[0]
    assert first["forecast"] == pytest.approx(6.110186, abs=2e-5) and first["se"] == pytest.approx(0.03671, abs=2e-5)
    assert fit.spec.describe(fit.season) == "SARIMA(0,1,1)(0,1,1)12" and fit.season == 12


def test_the_fit_uses_bfgs_unless_told_otherwise(air):
    # statsmodels' default L-BFGS stopped at MA(1) -0.4135 (log likelihood 0.01 short of the maximum) with
    # statsmodels 0.14 on OpenBLAS's generic kernels; BFGS reaches R's estimates on every CPU tried
    spec = SARIMA("lair", (0, 1, 1), (0, 1, 1))
    fit = _fit(spec, air)
    assert fit.provenance.options["stages"]["fit"]["method"] == "bfgs"
    lbfgs = _fit(SARIMA("lair", (0, 1, 1), (0, 1, 1), backend_options={"method": "lbfgs"}), air)
    assert lbfgs.provenance.options["stages"]["fit"]["method"] == "lbfgs"  # still the user's to choose
    assert fit.stats["loglike"] >= lbfgs.stats["loglike"] - 1e-9


def test_seasonal_ar_forecasts_and_variances_by_hand(ar_data):
    fit = _fit(SARIMA("y", (1, 0, 0), (1, 0, 0)), ar_data, span=(None, "2015Q4"))
    names = list(fit.params.names)
    v = dict(zip(names, fit.params.values))
    phi, Phi, c, s2 = v["AR(1)"], v["seasonal AR(4)"], v["constant"], v["variance"]
    mu = c / ((1 - phi) * (1 - Phi))  # the constant is on the left: (1 - phi L)(1 - Phi L^4) y = c + e
    y = list(fit.data.to_numpy() - mu)
    for _ in range(8):  # y_t = phi y_t-1 + Phi y_t-4 - phi Phi y_t-5 (in deviations from the mean)
        y.append(phi * y[-1] + Phi * y[-4] - phi * Phi * y[-5])
    fc = fit.forecast(ar_data, ("2016Q1", "2017Q4")).frame
    assert fc["forecast"].to_numpy() == pytest.approx(np.array(y[-8:]) + mu, rel=1e-6)
    ar = np.zeros(13)  # the AR polynomial's lags, for the MA(infinity) weights
    ar[1], ar[4], ar[5] = phi, Phi, -phi * Phi
    psi = [1.0]
    for j in range(1, 8):
        psi.append(sum(ar[i] * psi[j - i] for i in range(1, min(j, 5) + 1)))
    expected_var = s2 * np.cumsum(np.square(psi))
    assert (fc["se"] ** 2).to_numpy() == pytest.approx(expected_var, rel=1e-6)  # var_h = sigma^2 sum psi_j^2
    assert fc["actual"].notna().all()


def test_regressors_take_their_future_values_from_the_data(ar_data):
    fit = _fit(SARIMA("yx", (1, 0, 0), exog=("x",)), ar_data, span=(None, "2015Q4"))
    v = dict(zip(fit.params.names, fit.params.values))
    assert v["x"] == pytest.approx(2.0, abs=0.15)
    one = fit.forecast(ar_data, ("2016Q1", "2016Q1")).frame["forecast"].iloc[0]
    # SARIMAX's form: (1 - phi L)(y_t - b x_t) = c + e_t, the constant in the error's AR equation, so the one-step
    # forecast is c + b x_T+1 + phi (y_T - b x_T)
    error_last = fit.data.iloc[-1] - v["x"] * ar_data["x"].loc["2015Q4"]
    assert one == pytest.approx(v["constant"] + v["x"] * ar_data["x"].loc["2016Q1"] + v["AR(1)"] * error_last, rel=1e-6)
    with pytest.raises(SpecificationError, match="needs data"):
        fit.forecast(horizon=2)


def test_the_theta_drift_is_half_the_trend_slope(air):
    fit = _fit(Theta("lair", deseasonalize="no"), air)
    slope = np.polyfit(np.arange(len(AIR)), np.log(AIR), 1)[0]  # least squares, independently of statsmodels
    assert fit.params.values[0] == pytest.approx(slope, rel=1e-8)
    fc = fit.forecast(horizon=6).frame["forecast"].to_numpy()
    assert np.diff(fc) == pytest.approx(np.full(5, slope / 2), rel=1e-8)  # theta = 2: the forecasts climb by b0 / 2
    se = fit.forecast(horizon=6).frame["se"].to_numpy()
    assert np.all(np.diff(se) > 0)
    seasonal = _fit(Theta("air"), air)
    assert seasonal.deseasonalized and "seasonal adjustment" in seasonal.summary().notes[-1]


def test_stl_forecasts_are_the_adjusted_forecast_plus_last_years_season(air):
    from statsmodels.tsa.arima.model import ARIMA

    fit = _fit(STLForecast("lair", model="arima", order=(1, 1, 0)), air)
    comp = fit.components
    assert (comp["trend"] + comp["seasonal"] + comp["remainder"]).to_numpy() == pytest.approx(fit.data.to_numpy(), rel=1e-10)
    adjusted = ARIMA(comp["adjusted"].to_numpy(), order=(1, 1, 0), trend="n").fit()  # the adjusted series' own model
    fc = fit.forecast(horizon=12).frame
    expected = adjusted.forecast(12) + comp["seasonal"].iloc[-12:].to_numpy()
    assert fc["forecast"].to_numpy() == pytest.approx(expected, rel=1e-8)
    assert fit.spec.describe() == "STL + ARIMA(1,1,0)"
    ets = _fit(STLForecast("lair", model="ets"), air)
    assert ets.forecast(horizon=4).frame["forecast"].notna().all() and ets.spec.describe() == "STL + ETS(A,Ad,N)"


def test_backtests_and_refusals(air, ar_data):
    fit = _fit(SARIMA("lair", (0, 1, 1), (0, 1, 1)), air, span=(None, "1958-12"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bt = backtest(fit, air, ("1959-01", "1959-06"), horizon=3)
    assert len(bt.errors) > 0
    with pytest.raises(SpecificationError, match="three numbers"):
        SARIMA("y", (1, 0))
    with pytest.raises(SpecificationError, match="after the sample"):
        fit.forecast(air, ("1958-06", "1959-01"))
    gappy = Dataset.from_frame(ar_data.frame.assign(y=ar_data["y"].where(ar_data["y"].index != ar_data["y"].index[50])))
    with pytest.raises(SpecificationError, match="unbroken"):
        _fit(SARIMA("y"), gappy)
    annual = Dataset.from_frame(pd.DataFrame({"y": np.arange(30.0)}, index=pd.period_range("1990", periods=30, freq="Y", name="time")))
    with pytest.raises(SpecificationError, match="seasonal period"):
        _fit(SARIMA("y", seasonal_order=(1, 0, 0)), annual)


def test_the_dialogs_record_the_calls(air):
    from pymetrix.commands import FitSARIMA, FitSTLForecast, FitTheta, Forecast, Session

    s = Session()
    s.objects["air"] = air
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.run(FitSARIMA("lair", data="air", order="0, 1, 1", seasonal_order="0, 1, 1", end="1959-12"))
        s.run(FitTheta("air", data="air"))
        s.run(FitSTLForecast("lair", data="air"))
        s.run(Forecast("sarima", "1960-01", "1960-12", data="air", name="fc"))
    assert s.history[0].code == ("sarima = SARIMA(series='lair', order=[0, 1, 1], seasonal_order=[0, 1, 1])"
                                 ".fit(air, span=(None, '1959-12'))")
    assert type(s.objects["theta"]).__name__ == "FittedTheta" and type(s.objects["stl"]).__name__ == "FittedSTLForecast"
    assert s.objects["fc"].frame["forecast"].notna().all()
    namespace = {"air": air}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(s.script(), namespace)  # noqa: S102 - the script reproduces the estimates
    assert namespace["sarima"].stats["loglike"] == pytest.approx(s.objects["sarima"].stats["loglike"])
