"""ARMA errors (against statsmodels SARIMAX) and GARCH (by recovery and an independent likelihood)."""

import numpy as np
import pandas as pd
import pytest
pytest.importorskip("statsmodels")  # reference library for these tests
import statsmodels.api as sm  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: E402

from pymetrix import ARMA, GARCH, ML, OLS, Dataset, Equation


def _quarterly(values: dict) -> Dataset:
    n = len(next(iter(values.values())))
    return Dataset.from_frame(pd.DataFrame(values, index=pd.period_range("1950Q1", periods=n, freq="Q", name="time")))


@pytest.fixture
def arma_data() -> Dataset:
    rng = np.random.default_rng(7)
    n = 400
    x = rng.normal(size=n)
    eps = rng.normal(size=n + 1)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.6 * u[t - 1] + eps[t + 1] + 0.3 * eps[t]
    ar2 = np.zeros(n)
    for t in range(2, n):
        ar2[t] = 0.5 * ar2[t - 1] + 0.3 * ar2[t - 2] + eps[t]
    return _quarterly({"y": 1 + 0.5 * x + u, "x": x, "a": 2 + ar2})


def _sarimax(data, dep, order, exog=None):
    df = data.frame
    ex = sm.add_constant(df[exog]) if exog else (pd.Series(1.0, index=df.index, name="const") if exog == [] else None)
    return sm.tsa.SARIMAX(df[dep], exog=ex, order=order).fit(disp=0, cov_type="approx")


def test_regression_with_arma_errors_matches_sarimax(arma_data):
    fit = Equation("y ~ x", errors=ARMA(ar=1, ma=1)).fit(arma_data)
    ref = _sarimax(arma_data, "y", (1, 0, 1), ["x"])
    assert fit.params.names == ("const", "x", "ar[1]", "ma[1]", "sigma2")
    assert np.allclose(fit.params.values, ref.params.to_numpy(), atol=1e-4)
    assert np.allclose(fit.params.se, ref.bse.to_numpy(), rtol=2e-3)
    assert np.isclose(fit.stats["loglike"], ref.llf)
    assert np.allclose(fit.resid, ref.resid, atol=1e-3)
    assert isinstance(fit.method, ML)


def test_pure_ar_and_ma_models(arma_data):
    ar = Equation("a ~ 1", errors=ARMA(ar=2)).fit(arma_data)
    ref = _sarimax(arma_data, "a", (2, 0, 0), [])
    assert np.allclose(ar.params.values, ref.params.to_numpy(), atol=1e-4)
    data = arma_data.with_series(m="a - 2")
    ma = Equation("m ~ 0", errors=ARMA(ma=1)).fit(data)
    ref_ma = sm.tsa.SARIMAX(data["m"], order=(0, 0, 1)).fit(disp=0)
    assert np.allclose(ma.params.values, ref_ma.params.to_numpy(), atol=1e-4)


def test_subset_lags(arma_data):
    fit = Equation("a ~ 1", errors=ARMA(ar=[1, 4])).fit(arma_data)
    assert fit.params.names == ("const", "ar[1]", "ar[4]", "sigma2")
    assert fit.params["ar[1]"] > 0.5


def test_ljung_box(ts, arma_data):
    ols = Equation("y ~ x + z").fit(ts)
    ref = acorr_ljungbox(ols.resid, lags=[10])
    assert np.isclose(ols.test("ljung_box", lags=10).statistic, ref["lb_stat"].iloc[0])
    arma = Equation("y ~ x", errors=ARMA(ar=1, ma=1)).fit(arma_data)
    lb = arma.test("ljung_box", lags=10)
    assert lb.df == 8 and not lb.reject()
    assert [r[0] for r in arma.diagnostics().rows][:2] == ["Ljung-Box Q", "McLeod-Li (squared residuals)"]


def test_error_model_rules(arma_data):
    with pytest.raises(ValueError, match="use method=ML"):
        Equation("y ~ x", errors=ARMA(ar=1)).fit(arma_data, method=OLS())
    with pytest.raises(ValueError, match="explicit form is planned"):
        Equation("y ~ a + b*x", params=["a", "b"], errors=ARMA(ar=1))
    gappy = arma_data.with_series(y=arma_data["y"].mask(arma_data["y"].index == pd.Period("1960Q1", "Q")))
    with pytest.raises(ValueError, match="consecutive periods"):
        Equation("y ~ x", errors=ARMA(ar=1)).fit(gappy)


def test_arma_forecasts_match_sarimax(arma_data):
    df = arma_data.frame
    fit = Equation("y ~ x", errors=ARMA(ar=1, ma=1)).fit(arma_data, span=("1950Q1", str(df.index[379])))
    span = (str(df.index[380]), str(df.index[-1]))
    est = df.iloc[:380]
    ref = sm.tsa.SARIMAX(est["y"], exog=sm.add_constant(est["x"]), order=(1, 0, 1)).fit(disp=0)
    exog = sm.add_constant(df["x"])
    dynamic = ref.get_forecast(20, exog=exog.iloc[380:]).summary_frame()
    static = ref.apply(df["y"], exog=exog).get_prediction(start=380, end=399).summary_frame()
    for ours, theirs in ((fit.forecast(arma_data, span, dynamic=True), dynamic), (fit.forecast(arma_data, span), static)):
        assert np.allclose(ours["forecast"], theirs["mean"], atol=1e-3)
        assert np.allclose(ours["se"], theirs["mean_se"], atol=1e-3)
        assert np.allclose(ours["actual"], df["y"].iloc[380:])
    assert np.allclose(fit.predict(arma_data, span), fit.forecast(arma_data, span)["forecast"])


def test_pure_arma_forecast_beyond_the_data(arma_data):
    fit = Equation("a ~ 1", errors=ARMA(ar=2)).fit(arma_data)
    fc = fit.forecast(arma_data, ("2050Q1", "2074Q4"), dynamic=True)
    assert len(fc) == 100 and np.isnan(fc["actual"]).all()
    assert abs(fc["forecast"].iloc[-1] - fit.params["const"]) < 1e-3  # reverts to the mean
    assert fc["se"].is_monotonic_increasing
    with pytest.raises(ValueError, match="must start after"):
        fit.forecast(arma_data, ("1950Q1", "1951Q1"))


def test_fit_command_with_error_models(arma_data):
    from pymetrix.commands import FitEquation, Session

    session = Session()
    session.objects["data"] = arma_data
    assert FitEquation("y ~ x", ar="1", ma="1").validate(session)  # needs method ML
    fit = session.run(FitEquation("y ~ x", method="ML", ar="1", ma="1", name="arma"))
    session.run(FitEquation("y ~ x", method="ML", variance="GJR-GARCH", dist="t", name="gjr"))
    script = session.script()
    assert "Equation('y ~ x', errors=ARMA(ar=1, ma=1)).fit(data, method=ML())" in script
    assert "variance=GARCH(asym=1, dist='t')" in script
    ns = {"data": arma_data}
    exec(script, ns)
    assert np.allclose(ns["arma"].params.values, fit.params.values)


# GARCH


def _simulate_garch(n, omega, alpha, gamma, beta, nu=None, seed=1):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu) if nu else rng.normal(size=n)
    e, s2 = np.zeros(n), np.full(n, omega / (1 - alpha - gamma / 2 - beta))
    for t in range(n):
        if t:
            s2[t] = omega + (alpha + gamma * (e[t - 1] < 0)) * e[t - 1] ** 2 + beta * s2[t - 1]
        e[t] = np.sqrt(s2[t]) * z[t]
    index = pd.period_range("2000-01-01", periods=n, freq="D", name="time")
    return Dataset.from_frame(pd.DataFrame({"r": 0.2 + e}, index=index))


def _loop_loglike(r, mu, omega, alpha, beta, backcast):
    """Independent GARCH(1,1) normal log likelihood, written as a plain loop."""
    e = r - mu
    s2_prev, e2_prev, total = backcast, backcast, 0.0
    for et in e:
        s2 = omega + alpha * e2_prev + beta * s2_prev
        total += -0.5 * (np.log(2 * np.pi) + np.log(s2) + et**2 / s2)
        s2_prev, e2_prev = s2, et**2
    return total


def test_garch_recovers_parameters_and_maximises_the_likelihood():
    data = _simulate_garch(3000, 0.05, 0.08, 0.0, 0.88)
    fit = Equation("r ~ 1", variance=GARCH()).fit(data)
    truth = {"const": 0.2, "omega": 0.05, "alpha[1]": 0.08, "beta[1]": 0.88}
    for (name, value), se in zip(fit.params.as_dict().items(), fit.params.se):
        assert abs(value - truth[name]) < 4 * se, name
    p = fit.params.as_dict()
    r = data["r"].to_numpy()
    args = (p["const"], p["omega"], p["alpha[1]"], p["beta[1]"])
    ll = _loop_loglike(r, *args, fit.extra["backcast"])
    assert np.isclose(ll, fit.stats["loglike"])
    for i in range(4):
        for step in (-1e-4, 1e-4):
            moved = list(args)
            moved[i] += step * max(abs(moved[i]), 0.01)
            assert _loop_loglike(r, *moved, fit.extra["backcast"]) <= ll + 1e-6
    assert fit.extra["volatility"].shape == (3000,)
    assert not fit.test("arch").reject()


def test_garch_forecasts():
    data = _simulate_garch(2000, 0.05, 0.08, 0.0, 0.88, seed=4)
    fit = Equation("r ~ 1", variance=GARCH()).fit(data, span=("2000-01-01", "2005-01-31"))
    vol = fit.extra["volatility"]
    one_step = fit.forecast(data, ("2000-01-02", "2005-01-31"))
    assert np.allclose(one_step["se"], vol.iloc[1:])  # static forecasts reproduce the fitted volatility
    p = fit.params.as_dict()
    w, a, b, mu = p["omega"], p["alpha[1]"], p["beta[1]"], p["const"]
    e_last = data["r"].loc[pd.Period("2005-01-31", "D")] - mu
    s2_next = w + a * e_last**2 + b * vol.iloc[-1] ** 2
    h = np.arange(1, 31)
    closed = w * (1 - (a + b) ** (h - 1)) / (1 - a - b) + (a + b) ** (h - 1) * s2_next
    multi = fit.forecast(data, ("2005-02-01", "2005-03-02"), dynamic=True)
    assert np.allclose(multi["variance"], closed)
    assert np.allclose(multi["forecast"], mu)
    assert np.allclose(multi["upper"] - multi["forecast"], 1.959964 * np.sqrt(closed))


def test_gjr_garch_with_t_errors():
    data = _simulate_garch(3000, 0.05, 0.03, 0.10, 0.88, nu=8, seed=2)
    fit = Equation("r ~ 1", variance=GARCH(asym=1, dist="t")).fit(data, method=ML(cov="white"))
    assert fit.params.names == ("const", "omega", "alpha[1]", "gamma[1]", "beta[1]", "nu")
    assert fit.params["gamma[1]"] > 2 * fit.params.se[3]
    assert 5 < fit.params["nu"] < 15
    assert fit.stats["persistence"] < 1
    assert fit.test("jarque_bera").reject()  # t innovations are not normal
