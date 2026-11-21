"""Exponential smoothing (ETS, through statsmodels): the recursions by hand, the textbook forecast variance, seasonal
forecasts from the states, backtests, and the dialog. Elliott and Timmermann (2016), section 7.5."""

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from pymetrix import Dataset, ExponentialSmoothing, backtest  # noqa: E402
from pymetrix.errors import SpecificationError  # noqa: E402


@pytest.fixture(scope="module")
def macro():
    return Dataset.example("sample_macro")


@pytest.fixture(scope="module")
def seasonal():
    rng = np.random.default_rng(2)
    idx = pd.period_range("2005Q1", "2024Q4", freq="Q", name="time")
    n = len(idx)
    season = np.tile([3.0, -1.0, -4.0, 2.0], n // 4)
    y = 100 + 0.5 * np.arange(n) + season + rng.normal(0, 0.8, n)
    return Dataset.from_frame(pd.DataFrame({"y": y}, index=idx))


def _params(fit):
    return dict(zip(fit.result.param_names, np.asarray(fit.result.params, float)))


def test_simple_smoothing_is_the_level_recursion_with_the_textbook_variance(macro):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing("y").fit(macro, span=(None, "2020Q4"))
    p = _params(fit)
    alpha, level = p["smoothing_level"], p["initial_level"]
    fitted = []
    for y in fit.data.to_numpy():
        fitted.append(level)
        level = level + alpha * (y - level)  # l_t = l_(t-1) + alpha e_t
    assert fit.fitted.to_numpy() == pytest.approx(fitted, rel=1e-9)
    fc = fit.forecast(macro, ("2021Q1", "2021Q4")).frame
    assert fc["forecast"].to_numpy() == pytest.approx([level] * 4, rel=1e-9)  # flat at the last level
    ratio = (fc["se"] ** 2 / fc["se"].iloc[0] ** 2).to_numpy()
    assert ratio == pytest.approx(1 + np.arange(4) * alpha**2, rel=1e-9)  # var_h = sigma^2 (1 + (h - 1) alpha^2)
    assert fc["actual"].to_numpy() == pytest.approx(macro["y"].loc["2021Q1":"2021Q4"].to_numpy())


def test_damped_trend_forecasts_by_hand(macro):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing("c", trend="add", damped=True).fit(macro, span=(None, "2020Q4"))
    p = _params(fit)
    a, b, phi = p["smoothing_level"], p["smoothing_trend"], p["damping_trend"]
    level, slope = p["initial_level"], p["initial_trend"]
    for y in fit.data.to_numpy():  # error-correction form: e = y - (l + phi b); l += phi b + a e; b = phi b + b* e
        e = y - (level + phi * slope)
        level, slope = level + phi * slope + a * e, phi * slope + b * e
    expected = [level + sum(phi**j for j in range(1, h + 1)) * slope for h in range(1, 5)]
    assert fit.forecast(macro, ("2021Q1", "2021Q4")).frame["forecast"].to_numpy() == pytest.approx(expected, rel=1e-8)
    assert fit.spec.describe() == "ETS(A,Ad,N)"


def test_holt_winters_forecasts_from_the_final_states(seasonal):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing("y", trend="add", seasonal="add").fit(seasonal, span=(None, "2022Q4"))
    states = fit.states
    level, slope = states["level"].iloc[-1], states["trend"].iloc[-1]
    seasons = states["seasonal"].iloc[-4:].to_numpy()  # the last four seasonal factors
    expected = [level + h * slope + seasons[(h - 1) % 4] for h in range(1, 9)]
    fc = fit.forecast(seasonal, ("2023Q1", "2024Q4")).frame
    assert fc["forecast"].to_numpy() == pytest.approx(expected, rel=1e-9)
    assert np.mean(np.abs(fc["forecast"] - fc["actual"])) < 2.0  # it tracks the seasonal pattern
    assert fit.summary().index[0] == "alpha (level)" and "gamma (seasonal)" in fit.summary().index


def test_specification_errors(macro, seasonal):
    with pytest.raises(SpecificationError, match="only a trend"):
        ExponentialSmoothing("y", damped=True).fit(macro)
    with pytest.raises(SpecificationError, match="no series"):
        ExponentialSmoothing("nope").fit(macro)
    with pytest.raises(SpecificationError, match="start after the sample"):
        ExponentialSmoothing("y").fit(macro, span=(None, "2020Q4")).forecast(macro, ("2020Q3", "2021Q1"))
    shifted = Dataset.from_frame(pd.DataFrame({"y": seasonal["y"] - 200}, index=seasonal.frame.index))
    with pytest.raises(SpecificationError, match="positive"):
        ExponentialSmoothing("y", error="mul").fit(shifted)


def test_backtests_re_estimate_exponential_smoothing(macro):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing("c", trend="add").fit(macro, span=("2001Q1", "2020Q4"))
        bt = backtest(fit, macro, ("2016Q1", "2017Q4"), horizon=2, label="ets")
        rolling = backtest(fit, macro, ("2016Q1", "2016Q2"), horizon=1, label="ets rolling", window=40)
        row = bt.errors.iloc[5]
        own = ExponentialSmoothing("c", trend="add").fit(macro, span=("2001Q1", row["origin"]))
        expected = own.forecast(macro, (str(row["target"]), str(row["target"]))).frame
    assert row["forecast"] == pytest.approx(float(expected["forecast"].iloc[0]), rel=1e-9)  # from the fit's own start
    assert row["sd"] == pytest.approx(float(expected["se"].iloc[0]), rel=1e-9) and row["density"] == "normal"
    assert bt.density_scores().loc[("c", 1), "n"] == 8
    assert rolling.errors.iloc[0]["forecast"] == pytest.approx(float(
        ExponentialSmoothing("c", trend="add").fit(macro, span=("2006Q2", "2016Q1")).forecast(macro, ("2016Q2", "2016Q2")).frame["forecast"].iloc[0]),
        rel=1e-9)


def test_the_dialog_replays(macro):
    from pymetrix.commands import FitExponentialSmoothing, Forecast, Session

    session = Session()
    session.objects["data"] = macro
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.run(FitExponentialSmoothing("c", trend="add", damped=True, end="2020Q4"))
        session.run(Forecast("ets", "2021Q1", "2021Q4"))
        namespace = dict(session.namespace) | {"data": macro}
        exec("\n".join(r.code for r in session.history), namespace)
    assert namespace["forecast"].frame["forecast"].to_numpy() == pytest.approx(session["forecast"].frame["forecast"].to_numpy())
    assert session.history[0].code == "ets = ExponentialSmoothing(series='c', trend='add', damped=True).fit(data, span=(None, '2020Q4'))"
