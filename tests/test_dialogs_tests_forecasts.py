"""Dialogs for tests that need the data (weak instruments, non-nested, Gregory-Hansen), the quantile process, panel mean
group and CCE, superior predictive ability, binary forecast scores, real-time backtests from vintages, and forecasts from
smooth-transition autoregressions.

Each dialog's run equals the API call it stands for, records that call (the same one typing its code records), and the
session's script replays to the same result."""

import os

import numpy as np
import pandas as pd
import pytest

from pymetrix import (GMM, IV, Dataset, Equation, SmoothTransitionAR, anderson_rubin, anderson_rubin_set, backtest, binary_scores, c_test,
                      cce, cox_test, gregory_hansen, j_test, mean_group, pooled_mean_group, quantile_process, stock_wright)
from pymetrix.commands import (BinaryForecastScores, FitPanelMeanGroup, GregoryHansenTest, InstrumentTests, NonNestedTest, QuantileProcessPlot,
                               RunBacktest, Session, SuperiorPredictiveAbility)
from pymetrix.commands.builtin import Forecast

EQ = "lwage ~ educ + exper + black + south"
GMM_IV = ["nearc4", "nearc2", "exper", "black", "south"]


def _replay(s, name):
    ns = dict(s.objects)
    exec(s.script(), ns)
    return ns[name]


def _typed_again(s, start):
    """The dialog's code, typed in a fresh session from ``start()``, records the same call."""
    record = s.history[-1]
    assert record.ok, record.error
    assert record.call is not None and record.code == record.call.to_code()
    other = start()
    other.run_code(record.code)
    typed = other.history[-1]
    assert typed.ok, typed.error
    assert type(typed.command).__name__ == "Call" and typed.call.same_as(record.call)


def _same_test(a, b):
    assert a.statistic == pytest.approx(b.statistic, rel=1e-10)
    assert (a.p_value is None and b.p_value is None) or a.p_value == pytest.approx(b.p_value, rel=1e-10, abs=1e-15)


def _iv_session():
    s = Session()
    s.objects["data"] = Dataset.example("card")
    s.run_code(f"ivfit = Equation({EQ!r}).fit(data, method=IV(endog=['educ'], instruments=['nearc4', 'nearc2']))")
    s.run_code(f"gfit = Equation({EQ!r}).fit(data, method=GMM(instruments={GMM_IV!r}))")
    return s


@pytest.fixture(scope="module")
def iv():
    return _iv_session()


def test_anderson_rubin_and_its_confidence_set(iv):
    data, eq = iv["data"], Equation(EQ)
    iv.run(InstrumentTests("ivfit", value=0.1, cov="white", name="ar"))
    _same_test(iv["ar"], anderson_rubin(eq, data, "educ", ["nearc4", "nearc2"], 0.1, cov="white"))
    _typed_again(iv, _iv_session)
    _same_test(_replay(iv, "ar"), iv["ar"])
    iv.run(InstrumentTests("ivfit", test="anderson_rubin_set", grid="0, 0.4, 41", alpha=0.1, name="arset"))
    by_api = anderson_rubin_set(eq, data, "educ", ["nearc4", "nearc2"], np.linspace(0, 0.4, 41), alpha=0.1)
    np.testing.assert_allclose(iv["arset"]["p-value"], by_api["p-value"], rtol=1e-10)
    assert iv["arset"]["in set"].tolist() == by_api["in set"].tolist()
    pd.testing.assert_frame_equal(_replay(iv, "arset"), iv["arset"])
    # a GMM fit: the excluded instruments are those not among the regressors
    iv.run(InstrumentTests("gfit", value=0.1, name="ar_gmm"))
    assert "['nearc4', 'nearc2']" in iv.history[-1].code
    _same_test(iv["ar_gmm"], anderson_rubin(eq, data, "educ", ["nearc4", "nearc2"], 0.1))


def test_stock_wright_and_the_c_test(iv):
    data, eq, gmm = iv["data"], Equation(EQ), GMM(instruments=GMM_IV)
    values = "const=4, educ=0.1, exper=0.04, black=-0.2, south=-0.1"
    iv.run(InstrumentTests("gfit", test="stock_wright", values=values, name="sw"))
    _same_test(iv["sw"], stock_wright(eq, data, gmm, {"const": 4, "educ": 0.1, "exper": 0.04, "black": -0.2, "south": -0.1}))
    _typed_again(iv, _iv_session)
    iv.run(InstrumentTests("gfit", test="c_test", suspect="nearc2", name="ct"))
    _same_test(iv["ct"], c_test(eq, data, gmm, ["nearc2"]))
    _same_test(_replay(iv, "ct"), iv["ct"])
    _same_test(_replay(iv, "sw"), iv["sw"])
    # Stock-Wright and C need a GMM fit; every parameter's value; the suspect instruments
    assert "GMM" in InstrumentTests("ivfit", test="c_test", suspect="nearc2").validate(iv)[0]
    assert InstrumentTests("gfit", test="stock_wright").validate(iv)
    assert InstrumentTests("gfit", test="c_test").validate(iv)
    assert InstrumentTests("data").validate(iv)


def test_the_quantile_process(iv):
    iv.run(QuantileProcessPlot("ivfit", quantiles="0.25, 0.5, 0.75", vcov="iid", name="qp"))
    by_api = quantile_process(Equation(EQ), iv["data"], [0.25, 0.5, 0.75], vcov="iid")
    for q in (0.25, 0.5, 0.75):
        np.testing.assert_allclose(iv["qp"].fits[q].params.values, by_api.fits[q].params.values, rtol=1e-8)
    _typed_again(iv, _iv_session)
    np.testing.assert_allclose(_replay(iv, "qp").fits[0.5].params.values, by_api.fits[0.5].params.values, rtol=1e-8)
    assert QuantileProcessPlot("ivfit", quantiles="0.5, 1.2").validate(iv)


def _macro_session():
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro").derive(lc="log(c)", ly="log(y)")
    s.run_code("a = Equation('c ~ y + c[-1]').fit(data)")
    s.run_code("b = Equation('c ~ y + y[-1]').fit(data)")
    return s


def test_non_nested_tests():
    s = _macro_session()
    data, a, b = s["data"], Equation("c ~ y + c[-1]"), Equation("c ~ y + y[-1]")
    s.run(NonNestedTest("a", "b", name="j"))
    _same_test(s["j"], j_test(a, b, data))
    _typed_again(s, _macro_session)
    s.run(NonNestedTest("b", "a", test="cox", name="cox"))
    _same_test(s["cox"], cox_test(b, a, data))
    _same_test(_replay(s, "cox"), s["cox"])
    assert NonNestedTest("a", "a").validate(s)


def test_gregory_hansen():
    s = _macro_session()
    s.run(GregoryHansenTest("lc", "ly", model="regime", lags=2, reps=40, seed=3, name="gh"))
    by_api = gregory_hansen(s["data"], "lc", ["ly"], model="regime", lags=2, reps=40, seed=3)
    _same_test(s["gh"], by_api)
    assert s["gh"].details["break"] == by_api.details["break"]
    _typed_again(s, _macro_session)
    _same_test(_replay(s, "gh"), by_api)


def _panel_session():
    rng = np.random.default_rng(7)
    frames = []
    for u in range(6):
        x = np.cumsum(rng.normal(size=40))
        frames.append(pd.DataFrame({"country": f"c{u}", "time": pd.period_range("2000Q1", periods=40, freq="Q"), "x": x,
                                    "y": 1 + 0.5 * x + rng.normal(size=40)}))
    s = Session()
    s.objects["panel"] = Dataset.from_frame(pd.concat(frames), time="time", entity="country")
    return s


def _numbers(result):
    return result.summary().to_frame().select_dtypes("number").to_numpy(float)


@pytest.mark.parametrize("method, by_api", [("mean group", lambda d: mean_group(d, "y", ["x"], trend=True)),
                                            ("cce mean group", lambda d: cce(d, "y", ["x"])),
                                            ("cce pooled", lambda d: cce(d, "y", ["x"], pooled=True)),
                                            ("pooled mean group", lambda d: pooled_mean_group(d, "y", ["x"], lags=2))])
def test_panel_mean_group_and_cce(method, by_api):
    s = _panel_session()
    s.run(FitPanelMeanGroup("y", "x", data="panel", method=method, trend=True, lags=2, name="mg"))
    expected = _numbers(by_api(s["panel"]))
    np.testing.assert_allclose(_numbers(s["mg"]), expected, rtol=1e-8)
    _typed_again(s, _panel_session)
    np.testing.assert_allclose(_numbers(_replay(s, "mg")), expected, rtol=1e-8)
    if method == "pooled mean group":
        from pymetrix.views import view_sections

        assert "Hausman" in [t for t, _ in view_sections(s["mg"])]
    assert FitPanelMeanGroup("y", "x", data="panel").validate(_macro_session()) == ["no object named 'panel'"]


def _forecast_session():
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    s.run_code("a = Equation('c ~ c[-1]').fit(data)")
    s.run_code("b = Equation('c ~ y[-1] + c[-1]').fit(data)")
    s.run_code("bt_a = backtest(a, data, ('2018Q1', '2022Q4'), horizon=2)")
    s.run_code("bt_b = backtest(b, data, ('2018Q1', '2022Q4'), horizon=2)")
    return s


def test_superior_predictive_ability():
    pytest.importorskip("arch")
    s = _forecast_session()
    s.run(SuperiorPredictiveAbility("bt_a", "bt_b", "c", horizon=2, loss="absolute", reps=200, seed=4, name="spa_ab"))
    by_api = s["bt_a"].spa({"bt_b": s["bt_b"]}, "c", horizon=2, loss="absolute", reps=200, seed=4)
    _same_test(s["spa_ab"], by_api)
    from pymetrix import spa

    direct = spa(s["bt_a"].losses("c", 2, "absolute"), {"bt_b": s["bt_b"].losses("c", 2, "absolute")}, 2, 200, None, 4)
    _same_test(s["spa_ab"], direct)
    _typed_again(s, _forecast_session)
    _same_test(_replay(s, "spa_ab"), by_api)
    assert SuperiorPredictiveAbility("bt_a", "bt_a", "c").validate(s)


def test_binary_forecast_scores():
    rng = np.random.default_rng(0)
    idx = pd.period_range("2000Q1", periods=80, freq="Q")
    p = rng.uniform(size=80)

    def start():
        s = Session()
        s.objects["probs"] = Dataset.from_frame(pd.DataFrame({"p": p, "rec": (rng.uniform(size=80) < p).astype(float)}, index=idx))
        return s

    s = start()
    s.run(BinaryForecastScores("p", "rec", data="probs", threshold=0.4, name="scores"))
    frame = s["probs"].frame
    by_api = binary_scores(frame["p"], frame["rec"], threshold=0.4)
    assert s["scores"].rows == by_api.rows
    assert s.history[-1].code == "scores = binary_scores('p', 'rec', threshold=0.4, data=probs)"
    assert _replay(s, "scores").rows == by_api.rows
    assert BinaryForecastScores("p", "nothing", data="probs").validate(s)


def test_a_real_time_backtest_from_vintages():
    def start():
        s = Session()
        data = Dataset.example("sample_macro")
        s.objects["data"] = data
        s.run_code("fit = Equation('c ~ c[-1]').fit(data)")
        for o in pd.period_range("2019Q1", "2019Q4", freq="Q"):
            s.objects[f"v{o}".lower()] = data.with_series(c="c + 10").select(span=(None, str(o)))  # an early release ten higher
        return s

    s = start()
    vintages = "2019Q1=v2019q1, 2019Q2=v2019q2, 2019Q3=v2019q3, 2019Q4=v2019q4"
    s.run(RunBacktest("fit", "2019Q1", "2019Q4", horizon=2, vintages=vintages, name="rt"))
    by_api = backtest(s["fit"], s["data"], ("2019Q1", "2019Q4"), horizon=2, label="fit",
                      vintages={str(o): s[f"v{o}".lower()] for o in pd.period_range("2019Q1", "2019Q4", freq="Q")})
    pd.testing.assert_frame_equal(s["rt"].errors, by_api.errors)
    latest = backtest(s["fit"], s["data"], ("2019Q1", "2019Q4"), horizon=2, label="fit")
    assert not np.allclose(s["rt"].errors["forecast"], latest.errors["forecast"])  # the vintages were used
    _typed_again(s, start)
    pd.testing.assert_frame_equal(_replay(s, "rt").errors, by_api.errors)
    assert RunBacktest("fit", "2019Q1", "2019Q4", vintages="2019Q1").validate(s)
    assert RunBacktest("fit", "2019Q1", "2019Q4", vintages="2019Q1=nothing").validate(s) == ["vintages: no object named 'nothing'"]


def test_forecasts_from_a_smooth_transition_autoregression():
    def start():
        s = Session()
        s.objects["data"] = Dataset.example("sample_macro")
        s.run_code("star = SmoothTransitionAR('c', lags=1).fit(data)")
        return s

    s = start()
    s.run(Forecast("star", "2025Q1", "2025Q4", draws=300, seed=2, name="fstar"))
    by_api = s["star"].forecast(s["data"], ("2025Q1", "2025Q4"), draws=300, seed=2)
    pd.testing.assert_frame_equal(s["fstar"].frame, by_api.frame)
    _typed_again(s, start)
    pd.testing.assert_frame_equal(_replay(s, "fstar").frame, by_api.frame)
    assert isinstance(s["star"], type(SmoothTransitionAR("c", lags=1).fit(s["data"])))


def test_the_new_forms_offscreen(iv):
    pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import command_dialog

    form = command_dialog(InstrumentTests, iv)
    assert form.row_visible("value") and not form.row_visible("suspect") and not form.row_visible("grid")
    form.editors["test"].set("c_test")
    assert form.row_visible("suspect") and not form.row_visible("value")
    form.editors["test"].set("anderson_rubin_set")
    assert form.row_visible("grid") and form.row_visible("alpha")
    assert form.on_advanced_tab("instruments")
    panel = command_dialog(FitPanelMeanGroup, _panel_session())
    assert panel.row_visible("trend") and not panel.row_visible("lags")
    panel.editors["method"].set("pooled mean group")
    assert panel.row_visible("lags") and not panel.row_visible("trend")
    assert command_dialog(RunBacktest, _forecast_session()).row_visible("vintages")
    for cls in (NonNestedTest, GregoryHansenTest, QuantileProcessPlot, SuperiorPredictiveAbility, BinaryForecastScores):
        assert command_dialog(cls, iv) is not None


def test_menus_and_next_steps():
    from pymetrix.commands import COMMANDS
    from pymetrix.commands.menus import menu_order
    from pymetrix.ui.next_steps import steps_for

    places = {info.cls.__name__: path for info, path in menu_order(list(COMMANDS.values()))}
    assert places["InstrumentTests"] == "Test → Weak-instrument tests…"
    assert places["NonNestedTest"] == "Test → Non-nested tests (J, Cox)…"
    assert places["GregoryHansenTest"] == "Test → Gregory-Hansen cointegration test…"  # beside Johansen
    assert places["FitPanelMeanGroup"] == "Estimate → VARs and local projections → Panel mean group and CCE…"
    assert places["QuantileProcessPlot"] == "Estimate → Quantile process…"
    forecast = [n for n, p in places.items() if p.startswith("Forecast →")]
    assert forecast.index("SuperiorPredictiveAbility") == forecast.index("CompareManyForecasts") + 1
    assert forecast.index("BinaryForecastScores") == forecast.index("CompareManyForecasts") + 2
    s = _forecast_session()
    assert "SuperiorPredictiveAbility" in [c.__name__ for _, c, _, _ in steps_for(s["bt_a"])]
