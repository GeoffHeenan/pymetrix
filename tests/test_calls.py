"""API calls as commands (plan item 1): typed code and dialogs record the same typed step, which saves, replays
and renders as one canonical line of Python."""

import json
import time

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, OLS, VAR
from pymetrix._api import API_CALLS, api, call_migration, recording
from pymetrix.calls import Call, make_call
from pymetrix.codec import Ref
from pymetrix.commands import FitEquation, FitVAR, GenerateSeries, LoadExample, RunCode, Session, load_project, save_project
from pymetrix.commands.registry import from_json, to_json
from pymetrix.commands.ardl import FitARDL
from pymetrix.commands.builtin import (CheckModel, Correlogram, SelectOrder, SolveModel, StateSpaceForm, DefineEquation, DeriveSeries, Filter, FitRolling, FitVECM, Forecast, Hausman,
                                       ImpulseResponse, Johansen, RunTest, VarianceDecomposition, VARLagOrder, VECMLagOrder)
from pymetrix.commands.data_edit import DropSeries, FreezeSeries, RenameSeries, SeriesInfo
from pymetrix.commands import AdjustRound, CompareBacktests, CompareRounds, EditData, MakeRound, SaveRound, CombineForecasts, LoadTemplate, OccasionallyBinding, OptimalSimpleRule, ParameterSweep, RunBacktest, SolveOptimalPolicy
from pymetrix.commands.figures import (AnnotateFigure, CallAxesMethod, CoefficientPlot, CopyStyle, FigureOptions, PlotSeries,
                                       StyleFigure)
from pymetrix.commands.markov import FitMarkovSwitching
from pymetrix.commands.perturbation import SolveSecondOrder
from pymetrix.commands.midas import FitMIDAS
from pymetrix.commands.nowcast import FitBridge, FitDynamicFactor
from pymetrix.commands.panel import BalancePanel, PanelStatistics, PivotTable, Slice, ToLong, ToWide
from pymetrix.commands.projections import FitLocalProjection
from pymetrix.commands.samples import CheckData, NewSample, SetSample
from pymetrix.commands.seasonal import CalendarRegressors, SeasonalAdjust
from pymetrix.commands.selection import FitEach, SelectSeries, TestEach
from pymetrix.commands.statespace import (FitBMA, FitExponentialSmoothing, FitStateSpace, FitThresholdAR,
                                          FitUnobservedComponents)
from pymetrix.schema import catalogue


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMETRIX_HOME", str(tmp_path / "home"))


@pytest.fixture
def session():
    s = Session()
    s.run(LoadExample())
    return s


def kinds(s):
    return [type(r.command).__name__ for r in s.history]


def test_outside_a_session_nothing_changes():
    data = Dataset.example()
    fit = Equation("dlog(c) ~ dlog(y)").fit(data)
    assert fit.params.values.shape == (2,)
    assert Equation.fit.__name__ == "fit" and "Estimate the equation" in Equation.fit.__doc__
    assert Equation.fit.__api__.id == "pymetrix:equation.fit"
    import inspect

    assert list(inspect.signature(Equation.fit).parameters)[:3] == ["self", "data", "span"]


def test_the_decorator_costs_little_outside_a_session():
    def plain(x, y=1):
        return x

    recorded = api("pymetrix:tests.cost")(plain)
    try:
        n = 20_000
        t0 = time.perf_counter()
        for i in range(n):
            plain(i)
        t1 = time.perf_counter()
        for i in range(n):
            recorded(i)
        t2 = time.perf_counter()
        assert (t2 - t1 - (t1 - t0)) / n < 5e-6  # well under five microseconds a call
    finally:
        API_CALLS.pop("pymetrix:tests.cost")


def test_typed_calls_are_recorded_as_calls(session):
    session.run_code('data = data.with_series(ly="log(y)")\nfit = Equation("dlog(c) ~ dlog(y) + c[-1]").fit(data, method=OLS(cov="hac"))')
    assert kinds(session) == ["LoadExample", "Call", "Call"]
    with_series, fit = session.history[1:]
    assert with_series.code == "data = data.with_series(ly='log(y)')"
    assert fit.code == "fit = Equation('dlog(c) ~ dlog(y) + c[-1]').fit(data, method=OLS(cov='hac'))"
    assert fit.reads == {"data": with_series.writes["data"]}  # the second call read what the first wrote
    assert fit.call.arguments["missing"] == "drop"  # every argument, defaults included
    assert session.producer("fit") is fit


def test_code_that_is_more_than_calls_stays_typed_code(session):
    for code in ["import math\nx = math.pi", "x = [1, 2]", "for k in range(2):\n    data = data.with_series(z='y')",
                 "print(Equation('c ~ y').fit(data))"]:
        session.run_code(code)
        assert kinds(session)[-1] == "RunCode", code


def test_settings_defined_in_code_are_steps_that_replay_without_trust(tmp_path, session):
    session.run_code("m = OLS(cov='hac')\nfit = Equation('c ~ y').fit(data, method=m)")
    assert [r.code for r in session.history[-2:]] == ["m = OLS(cov='hac')", "fit = Equation('c ~ y').fit(data, method=m)"]
    assert kinds(session)[-2:] == ["Call", "Call"] and set(session.history[-1].reads) == {"data", "m"}
    session.run_code("ar = ARMA(ar=1)")
    session.run_code("eq = Equation('c ~ y', errors=ar)")
    assert session.history[-1].code == "eq = Equation('c ~ y', errors=ar)" and set(session.history[-1].reads) == {"ar"}
    opened = load_project(save_project(session, tmp_path / "settings.pymx"), trust=False)
    assert kinds(opened) == kinds(session) and np.allclose(opened["fit"].params.values, session["fit"].params.values)


def test_an_argument_that_is_not_a_setting_or_a_named_object_keeps_the_code_as_typed():
    s = Session()
    s.run_code("frame = pd.DataFrame({'y': [1.0, 2, 3, 4]}, index=pd.period_range('2000Q1', periods=4, freq='Q'))")
    s.run_code("data = Dataset.from_frame(frame)")  # not an API call: typed code
    s.run_code("data2 = data.with_series(z=frame['y'] * 2)")  # a Series argument: not a setting, not a named object
    assert kinds(s) == ["RunCode", "RunCode", "RunCode"]
    assert "z" in s["data2"]


def test_named_objects_are_recorded_by_name_and_settings_by_value(session):
    session.run_code('eq = Equation("c ~ y")')  # a plain constructor: typed code
    session.run_code("fit = eq.fit(data, method=OLS(cov='white'))")
    call = session.history[-1].call
    assert call.receiver == {"$ref": "eq"} and call.arguments["data"] == {"$ref": "data"}
    assert call.arguments["method"]["$type"] == "pymetrix:OLS"
    assert session.history[-1].code == "fit = eq.fit(data, method=OLS(cov='white'))"


def test_only_the_outermost_call_is_recorded():
    @api("pymetrix:tests.two_fits")
    def two_fits(data):
        return Equation("c ~ y").fit(data), Equation("c ~ g").fit(data)

    try:
        data, other = Dataset.example(), Dataset.example()
        with recording({"data": data}) as rec:
            two_fits(other)
        assert rec.calls == [] and rec.failed  # an unnamed dataset: not recordable, and its inner fits are part of it
        with recording({"data": data}) as rec:
            two_fits(data)
        assert [c.function for c, _ in rec.calls] == ["pymetrix:tests.two_fits"]
    finally:
        API_CALLS.pop("pymetrix:tests.two_fits")


DIALOGS = [
    FitEquation("dlog(c) ~ dlog(y) + c[-1]", cov="hac", start="1990Q1", end="2019Q4", name="fit"),
    FitEquation("c ~ b0 + b1*y", params="b0, b1", name="nls"),
    FitVAR("c, y", lags="2", name="var"),
    FitVAR("c, y", lags="aic", max_lags=4, bayesian=True, draws=200, seed=3, name="bvar"),
    GenerateSeries("ly", "log(y)"),
    GenerateSeries("ly", "log(y)", keep_updated=True),
    SetSample(start="1995Q1", end="2015Q4"),
    LoadExample("sample_monthly", name="monthly"),
    FitARDL("log(c) ~ log(y)", select="aic", max_lags=2, start="1990Q1"),
    FitMarkovSwitching("dlog(y) ~ 1", search=2, seed=1),
    FitLocalProjection("100*log(y)", "g", horizons=4, lags=2, controls="c"),
    FitUnobservedComponents("100*log(y)", cycle=0, end="2019Q4"),
    FitStateSpace("trend", "trend = trend[-1] + e", "y = trend + u", "e = sd_e, u = sd_u", params="sd_e, sd_u",
                  fixed="sd_u = 0.1"),
    FitExponentialSmoothing("c", trend="add", damped=True),
    FitBMA("dlog(c)", "dlog(y)[-1], dlog(i)[-1], dlog(g)[-1]", start="1990Q1"),
    FitThresholdAR("y", lags=2, delay=1),
    FitVECM("c, y", lags=2, rank="1"),
    SelectSeries(series="c, i", patterns="g*", name="sel"),
    FitEach("d{s} ~ d{s}[-1]", series="c, i", cov="white", start="1990Q1", keep_going=True),
    TestEach("adf", series="c, i", options="2, trend='ct'"),
    RenameSeries("data", "g", "gov"),
    DropSeries("data", "g, i"),
    DeriveSeries("dly", "dlog(y)"),
    Filter("y", method="HP", settings="lamb=1600"),
    Filter("c, y", keep_updated=False, name="filtered"),
    SeasonalAdjust("c", method="classical"),
    Johansen("c, y", lags=2, trend="linear trend", start="1990Q1"),
    VARLagOrder("c, y", max_lags=4),
    VECMLagOrder("c, y", max_lags=4),
    FitRolling("c ~ y", window=40, step=4),
    FitRolling("c ~ y", min_obs=20, cov="white", end="2019Q4"),
    CheckData(),
    DefineEquation("c ~ b0 + b1*y", params="b0, b1", name="ce"),
    NewSample(start="1990Q1", end="2000Q4", where="y > 0", name="sub"),
    SeriesInfo("data", "c", '{"units": "USD bn", "label": "Consumption"}'),
]


def _with_example():
    s = Session()
    s.run(LoadExample())
    return s


def _with_panel():
    """A session holding a wide dataset (y_A, y_B, c_A, c_B) and its long form, ``panel``."""
    s = Session()
    index = pd.period_range("2000Q1", periods=12, freq="Q")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(12, 4)).cumsum(axis=0), index=index, columns=["y_A", "y_B", "c_A", "c_B"])
    frame.iloc[0, 1] = np.nan  # a gap, for balancing
    s.objects["wide"] = Dataset.from_frame(frame)
    s.run_code("panel = wide.long(entity='country')")
    return s


def _same_call(start, dialog):
    """``dialog`` run in a session from ``start()``, and its code typed by hand in another, record the same call."""
    a = start()
    a.run(dialog)
    record = a.history[-1]
    assert record.ok, record.error
    assert record.call is not None and record.code == record.call.to_code()
    b = start()
    b.run_code(record.code)  # the code the dialog wrote, typed by hand
    typed = b.history[-1]
    assert type(typed.command).__name__ == "Call"
    assert typed.call.same_as(record.call)
    assert typed.code == record.code
    for name in record.writes:
        x, y = a[name], b[name]
        if hasattr(getattr(x, "params", None), "frame"):  # rolling fits: a dataset of estimates
            pd.testing.assert_frame_equal(x.params.frame, y.params.frame)
        elif hasattr(getattr(x, "params", None), "values"):
            assert np.allclose(x.params.values, y.params.values)
        elif hasattr(x, "frame"):
            pd.testing.assert_frame_equal(x.frame, y.frame)


@pytest.mark.parametrize("dialog", DIALOGS, ids=lambda d: f"{type(d).__name__}")
def test_a_dialog_and_its_code_record_the_same_call(dialog):
    _same_call(_with_example, dialog)


def test_freezing_a_derived_series_records_the_same_call():
    def start():
        s = _with_example()
        s.run_code("data = data.derive(ly='log(y)')")
        return s

    _same_call(start, FreezeSeries("data", "ly"))


def _with_two_frequencies():
    s = _with_example()
    s.run(LoadExample("sample_monthly", name="monthly"))
    s.run_code("data = data.with_series(gy='100*dlog(y)')")
    s.run_code("monthly = monthly.with_series(gip='100*dlog(ip)')")
    return s


@pytest.mark.parametrize("dialog", [
    FitBridge("gy ~ gip + pmi", low="data", high="monthly"),
    FitMIDAS("gy ~ gip", low="data", high="monthly", lags=6),
    FitDynamicFactor("gip, pmi", high="monthly", quarterly="gy", low="data"),
    FitDynamicFactor("gip, pmi", high="monthly"),
    CalendarRegressors("monthly", trading_days="one", easter=0),
], ids=lambda d: f"{type(d).__name__}")
def test_a_mixed_frequency_dialog_and_its_code_record_the_same_call(dialog):
    _same_call(_with_two_frequencies, dialog)


def test_a_hausman_test_names_both_fits():
    def start():
        s = _with_example()
        s.run_code("iv = Equation('c ~ y').fit(data, method=IV(endog=['y'], instruments=['g', 'i']))")
        s.run_code("ols = Equation('c ~ y').fit(data)")
        return s

    _same_call(start, Hausman("iv", "ols", sigma="separate"))
    s = start()
    s.run(Hausman("iv", "ols"))
    assert s.history[-1].code == "hausman = compare(iv, ols, test='hausman')"
    assert set(s.history[-1].reads) == {"iv", "ols"}


def _with_template():
    s = Session()
    s.run(LoadTemplate("new_keynesian", params="lower_bound=0", name="nk"))
    return s


@pytest.mark.parametrize("dialog", [
    LoadTemplate("rbc", params="alpha=0.3", name="rbc"),
    SolveOptimalPolicy("nk", "i", "pi=1, y=0.25", kind="discretion"),
    OptimalSimpleRule("nk", "phi_pi", "pi=1, y=0.25", lower="phi_pi=1.01", upper="phi_pi=10"),
    OccasionallyBinding("nk", "ey[1:3]=-4", periods=12, anticipated=True),
    ParameterSweep("nk", "phi_pi", "1.2, 2", what="moments"),
], ids=lambda d: f"{type(d).__name__}")
def test_a_model_dialog_and_its_code_record_the_same_call(dialog):
    _same_call(_with_template, dialog)


def test_templates_are_written_through_their_module():
    s = _with_template()
    assert s.history[-1].code == "nk = templates.new_keynesian(lower_bound=0.0)"
    assert type(s.history[-1].command).__name__ == "LoadTemplate" and s.history[-1].call is not None


def test_backtests_and_their_combination_name_what_they_use():
    def start():
        s = _with_example()
        s.run_code("ols = Equation('dlog(c) ~ dlog(c)[-1]').fit(data)")
        s.run(RunBacktest("ols", "2015Q1", "2018Q4", horizon=2, name="bt1"))
        s.run(RunBacktest("ols", "2015Q1", "2018Q4", horizon=2, window=40, name="bt2"))
        return s

    _same_call(start, RunBacktest("ols", "2015Q1", "2018Q4", horizon=2))
    _same_call(start, CombineForecasts("bt1, bt2", "inverse_mse"))
    s = start()
    s.run(CombineForecasts("bt1, bt2"))
    assert s.history[-1].code == "combination = combine_forecasts({'bt1': bt1, 'bt2': bt2})"
    assert set(s.history[-1].reads) == {"bt1", "bt2"}


def _with_fits():
    s = _with_example()
    s.run_code("fit = Equation('dlog(c) ~ dlog(c)[-1] + dlog(y)').fit(data, span=('1990Q1', '2019Q4'))")
    s.run_code("var = VAR(['c', 'y'], lags=2).fit(data, span=('1990Q1', '2019Q4'))")
    s.run_code("vecm = VECM(['c', 'y'], rank=1).fit(data, span=('1990Q1', '2019Q4'))")
    s.run_code("ets = ExponentialSmoothing('c', 'add').fit(data, span=('1990Q1', '2019Q4'))")
    return s


ACTING_ON_OBJECTS = [
    Forecast("fit", "2020Q1", "2021Q4", dynamic=True),
    Forecast("var", "2020Q1", "2021Q4", coverage=0.9),
    Forecast("vecm", "2020Q1", "2021Q4"),
    Forecast("ets", "2020Q1", "2021Q4"),
    ImpulseResponse("var", periods=8, draws=0),
    ImpulseResponse("var", periods=8, ordering="y, c", draws=50, seed=1),
    ImpulseResponse("var", periods=8, scheme="LongRun", draws=0, cumulative=True),
    ImpulseResponse("vecm", periods=8, ordering="c, y"),
    VarianceDecomposition("var", periods=8, ordering="y, c"),
    VarianceDecomposition("vecm", periods=8),
    Correlogram("fit", lags=8),
    Correlogram("data", variable="c", lags=8),
    Correlogram("data", variable="y", lags=8, squared=True),
    RunTest("fit", "ljung_box", options="lags=4"),
    RunTest("var", "normality"),
    RunTest("data", "adf", options="'y', lags=2"),
]


@pytest.mark.parametrize("dialog", ACTING_ON_OBJECTS, ids=lambda d: f"{type(d).__name__}-{getattr(d, 'fit', '') or getattr(d, 'var', '') or getattr(d, 'source', '')}")
def test_a_dialog_on_any_object_records_that_class_s_method(dialog):
    _same_call(_with_fits, dialog)


def test_the_call_names_the_method_of_the_object_s_class():
    s = _with_fits()
    for name, cid in (("fit", "pymetrix:fitted_equation.forecast"), ("var", "pymetrix:fitted_var.forecast"),
                      ("vecm", "pymetrix:fitted_vecm.forecast"), ("ets", "pymetrix:fitted_exponential_smoothing.forecast")):
        s.run(Forecast(name, "2020Q1", "2020Q4", name=f"{name}_fc"))
        assert s.history[-1].call.function == cid and s.history[-1].code.startswith(f"{name}_fc = {name}.forecast(data, ")
    s.run(ImpulseResponse("var", periods=8, ordering="y, c", draws=0))
    assert s.history[-1].code == "irf = var.identify(Cholesky(['y', 'c'])).irf(periods=8, draws=0)"
    assert s.history[-1].call.receiver["$call"]["function"] == "pymetrix:fitted_var.identify"


def _with_model():
    s = _with_example()
    s.run_code("cons = Equation('c ~ y + c[-1]').fit(data)")
    s.run_code("inv = Equation('i ~ y[-1]').fit(data)")
    s.run_code("identity = Equation('y = c + i + g')")
    s.run(LoadTemplate("new_keynesian", name="nk"))
    s.run(LoadTemplate("rbc", name="rbc"))
    return s


@pytest.mark.parametrize("dialog", [
    SolveModel("cons, inv, identity", "2015Q1", "2024Q4"),
    SolveModel("cons, inv, identity", "2015Q1", "2024Q4", shift="g=5"),
    CheckModel("cons, inv, identity", start="2015Q1", end="2024Q4"),
    SolveSecondOrder("rbc"),
    StateSpaceForm("nk", errors="growth=0.5"),
    SelectOrder("y", max_ar=2, max_ma=1),
], ids=lambda d: f"{type(d).__name__}")
def test_a_model_dialog_names_its_equations_and_model(dialog):
    _same_call(_with_model, dialog)


def test_a_model_from_named_equations_reads_them():
    s = _with_model()
    s.run(SolveModel("cons, inv, identity", "2015Q1", "2024Q4"))
    record = s.history[-1]
    assert record.code == "solution = Model([cons, inv, identity]).solve(data, ('2015Q1', '2024Q4'), solver=Newton())"
    assert set(record.reads) == {"cons", "inv", "identity", "data"}


def _with_forecasts():
    s = _with_example()
    s.run_code("ols = Equation('dlog(c) ~ dlog(c)[-1]').fit(data)")
    s.run(RunBacktest("ols", "2015Q1", "2018Q4", horizon=2, name="bt1"))
    s.run(RunBacktest("ols", "2015Q1", "2018Q4", horizon=2, window=40, name="bt2"))
    s.run_code("var = VAR(['c', 'y'], lags=2).fit(data, span=(None, '2022Q4'))")
    s.run_code("fc = var.forecast(data, ('2023Q1', '2024Q4'))")
    s.run(MakeRound("fc", "2022Q4 round", notes="baseline", name="rnd"))
    return s


@pytest.mark.parametrize("dialog", [
    CompareBacktests("bt1", "bt2", "c", horizon=2),
    CompareBacktests("bt1", "bt2", "c", test="encompassing"),
    MakeRound("fc", "copy", variables="c"),
    AdjustRound("rnd", "c", "2023Q1=-0.5, 2023Q2=-0.25", "port strike"),
], ids=lambda d: f"{type(d).__name__}")
def test_a_forecasting_dialog_and_its_code_record_the_same_call(dialog):
    _same_call(_with_forecasts, dialog)


def _with_figures():
    s = _with_example()
    s.run_code("fit = Equation('c ~ y').fit(data)")
    s.run_code("chart = data.plot(['c', 'y'], title='Spending').line('c', color='C1')")
    s.run_code("plain = data.plot(['c', 'y'])")
    return s


@pytest.mark.parametrize("dialog", [
    PlotSeries("c, i", kind="bar", stacked=True, start="2015Q1", name="bars"),
    FigureOptions("fit", title="Consumption", shade="2008Q3:2009Q2", name="titled"),
    StyleFigure("fit", series="actual (line)", color="C3", linewidth="2", grid="off", name="styled"),
    AnnotateFigure("chart", what="axvline", x="2020Q1", name="marked"),
    CallAxesMethod("fit", method="ticklabel_format", keywords="axis='y', style='sci'", name="sci"),
    CopyStyle("plain", style_from="chart", name="restyled"),
    CoefficientPlot("fit", name="coefs"),
], ids=lambda d: f"{type(d).__name__}")
def test_a_figure_dialog_records_its_chain_of_calls(dialog):
    _same_call(_with_figures, dialog)


def test_figure_steps_chain_figure_of_and_figure_methods():
    s = _with_figures()
    s.run(StyleFigure("fit", series="actual (line)", color="C3", grid="off", name="styled"))
    call = s.history[-1].call
    assert s.history[-1].code == "styled = figure_of(fit).line('actual', color='C3').grid(visible=False)"
    assert call.function == "pymetrix:figure.grid" and call.receiver["$call"]["function"] == "pymetrix:figure.line"


def test_edits_and_forecast_stores_record_the_same_call(tmp_path):
    _same_call(_with_example, EditData("data", '[["c", ["2020Q1"], 1.5], ["y", ["2020Q2"], null]]'))
    folder = str(tmp_path / "rounds")
    _same_call(_with_forecasts, SaveRound("rnd", folder, overwrite=True))
    s = _with_forecasts()
    s.run(SaveRound("rnd", folder, overwrite=True))
    assert s.history[-1].code == f"ForecastStore({folder!r}).save(rnd, overwrite=True)"
    s.run(MakeRound("fc", "second", name="rnd2"))
    s.run(SaveRound("rnd2", folder))
    _same_call(_with_forecasts, CompareRounds("2022Q4 round", "second", folder=folder, variables="c"))


# dialogs that act on whatever object they are given: (dialog, its object field, the method it calls on the object)
ON_ANY_OBJECT = [("Forecast", "fit", "forecast"), ("ImpulseResponse", "var", "irf"), ("VarianceDecomposition", "var", "fevd"),
                 ("Correlogram", "source", "correlogram"), ("RunTest", "source", "test"), ("RunMCMC", "source", "mcmc"),
                 ("CheckIdentification", "source", "identification"), ("DataContributions", "source", "contributions"),
                 ("ConditionalForecast", "fit", "conditional_forecast"), ("StateSpaceForm", "source", "system"),
                 ("SolveSecondOrder", "source", "solve"), ("EstimateModel", "source", "estimate")]


def test_every_class_a_dialog_offers_has_the_method_it_calls_recorded():
    """A new result type offered by Forecast (and the like) must mark the method with @api, or the dialog refuses it."""
    import importlib
    import inspect
    import pkgutil

    import pymetrix
    from pymetrix.commands.registry import COMMANDS
    from pymetrix.forms import form_spec

    classes = {}
    for info in pkgutil.walk_packages(pymetrix.__path__, "pymetrix."):
        if ".ui" in info.name or "mcp" in info.name:
            continue
        try:
            module = importlib.import_module(info.name)
        except ImportError:  # an optional library that is not installed
            continue
        classes.update({n: c for n, c in vars(module).items() if inspect.isclass(c) and c.__module__ == info.name})
    for dialog, field, method in ON_ANY_OBJECT:
        spec = next(f for f in form_spec(COMMANDS[dialog].cls) if f.name == field)
        for name in (n.strip() for n in spec.source.split(",")):
            if name == "Dataset" and method not in ("correlogram", "test"):
                continue
            cls = classes[name]
            assert hasattr(getattr(cls, method, None), "__api__"), f"{dialog} offers {name}, whose {method} is not a recorded call"


PANEL_DIALOGS = [
    ToLong("wide", entity="country"),
    ToWide("panel", series="y"),
    BalancePanel("panel", keep="entities"),
    Slice("panel", entity="A"),
    Slice("panel", period="2001Q1"),
    PanelStatistics("panel", series="y"),
    PanelStatistics("panel", show="variation"),
    PivotTable("panel", rows="country", values="y", agg="mean, count", margins=True),
]


@pytest.mark.parametrize("dialog", PANEL_DIALOGS, ids=lambda d: f"{type(d).__name__}")
def test_a_panel_dialog_and_its_code_record_the_same_call(dialog):
    _same_call(_with_panel, dialog)


@pytest.mark.parametrize("dialog", DIALOGS, ids=lambda d: f"{type(d).__name__}")
def test_calls_round_trip_as_json(dialog):
    call = dialog.to_call()
    value = json.loads(json.dumps(to_json(call), allow_nan=False))
    assert value["id"] == "pymetrix:call"
    back = from_json(value)
    assert back.same_as(call) and back.to_code() == call.to_code()


def test_a_system_names_its_equations(tmp_path, session):
    from pymetrix.commands.systems import FitSystem

    session.run_code("ce = Equation('c ~ y')")
    session.run_code("ie = Equation('i ~ y + c')")
    session.run(FitSystem("ce, ie", method="3sls", instruments="g", start="1990Q1"))
    record = session.history[-1]
    assert record.code == "sys = System([ce, ie], instruments=['g']).fit(data, method=ThreeSLS(), span=('1990Q1', None))"
    assert set(record.reads) == {"ce", "ie", "data"}  # the step follows the equations it names
    session.run_code(record.code)  # typed by hand: the same call
    assert type(session.history[-1].command).__name__ == "Call" and session.history[-1].call.same_as(record.call)
    opened = load_project(save_project(session, tmp_path / "system.pymx"), trust=True)  # the equations are typed code
    assert np.allclose(opened["sys"].params.values, session["sys"].params.values)


def test_typed_calls_replay_from_a_project_without_trust(tmp_path, session):
    session.run_code('data = data.with_series(ly="log(y)")')
    session.run_code("fit = Equation('dlog(c) ~ dlog(y) + c[-1]').fit(data, method=OLS(cov='hac'))")
    session.run_code("v = VAR(['c', 'y'], lags=2).fit(data)")
    path = save_project(session, tmp_path / "calls.pymx")
    from pymetrix.commands.project import project_code

    assert project_code(path).steps == ()  # API calls are not typed code: nothing to trust
    opened = load_project(path, trust=False)
    assert kinds(opened) == ["LoadExample", "Call", "Call", "Call"]
    assert np.allclose(opened["fit"].params.values, session["fit"].params.values)
    assert np.allclose(opened["v"].params.values, session["v"].params.values)
    assert opened.script() == session.script()


def test_a_dialog_step_saves_its_call_for_readers(tmp_path, session):
    import zipfile

    session.run(FitEquation("c ~ y", name="fit"))
    path = save_project(session, tmp_path / "dialog.pymx")
    with zipfile.ZipFile(path) as z:
        steps = json.loads(z.read("commands.json"))
    step = next(s for s in steps if s["id"] == "pymetrix:fit_equation")
    assert step["call"]["function"] == "pymetrix:equation.fit" and step["call"]["output"] == "fit"
    assert step["call"]["arguments"]["data"] == {"$ref": "data"}


def test_calls_saved_at_an_older_version_are_migrated():
    @api("pymetrix:tests.scaled", version=2)
    def scaled(data, factor=1.0):
        return data.with_series(scaled=f"y * {factor}")

    @call_migration("pymetrix:tests.scaled", 1)
    def _(arguments):
        arguments["factor"] = arguments.pop("multiplier")
        return arguments

    try:
        s = Session()
        s.run(LoadExample())
        old = Call("pymetrix:tests.scaled", 1, None, {"data": {"$ref": "data"}, "multiplier": 2.0}, output="out")
        s.run(old)
        assert np.allclose(s["out"]["scaled"], 2 * s["data"]["y"])
        assert old.to_code() == "out = scaled(data, factor=2.0)"
        newer = Call("pymetrix:tests.scaled", 3, None, {"data": {"$ref": "data"}}, output="out")
        with pytest.raises(Exception, match="newer"):
            s.run(newer)
    finally:
        API_CALLS.pop("pymetrix:tests.scaled")


def test_unknown_calls_and_missing_objects_are_refused():
    s = Session()
    with pytest.raises(Exception, match="no call"):
        s.run(Call("pymetrix:no.such_call"))
    with pytest.raises(Exception, match="no object named 'data'"):
        s.run(make_call("pymetrix:dataset.with_series", Ref("data"), output="data", z="1"))


def test_the_immediate_window_records_calls_too(session):
    before = session.begin_code()
    exec("fit = Equation('c ~ y').fit(data)", session.namespace)
    record = session.end_code(before, "fit = Equation('c ~ y').fit(data)", True)
    assert type(record.command).__name__ == "Call" and record.command.source == "immediate"
    assert "fit" in session.objects


def test_recording_can_be_switched_off(session):
    session.record_calls = False
    session.run_code("fit = Equation('c ~ y').fit(data)")
    assert kinds(session)[-1] == "RunCode"


def test_the_catalogue_lists_calls_with_schemas():
    entries = {e["id"]: e for e in catalogue()["entries"]}
    fit = entries["pymetrix:equation.fit"]
    assert fit["kind"] == "call" and fit["version"] == 1
    assert set(fit["schema"]["properties"]) == {"data", "span", "where", "method", "weights", "missing"}
    assert fit["schema"]["properties"]["missing"] == {"enum": ["drop", "raise"], "default": "drop"}
    for cid in ("pymetrix:dataset.read", "pymetrix:dataset.with_sample", "pymetrix:var.fit", "pymetrix:system.fit"):
        assert entries[cid]["kind"] == "call"


def test_generated_code_runs_to_the_same_call(session):
    session.run(FitEquation("dlog(c) ~ dlog(y)", cov="cluster", cluster="time", name="fit"))
    code = session.history[-1].code
    assert code == "fit = Equation('dlog(c) ~ dlog(y)').fit(data, method=OLS(cov=Cluster('time')))"
    session.run(RunCode(code))
    assert session.history[-1].call.same_as(session.history[-2].call)


def test_dialogs_not_yet_built_on_calls_only_shrink():
    from pathlib import Path

    from pymetrix.commands.registry import COMMAND_IDS

    listed = set(json.loads((Path(__file__).parent / "fixtures" / "dialogs_without_calls.json").read_text())["commands"])
    left = {cid for cid, info in COMMAND_IDS.items() if info.plugin == "pymetrix" and not hasattr(info.cls, "to_call")
            and cid not in ("pymetrix:call", "pymetrix:run_code")}
    assert left <= listed, f"new dialogs should build an API call (to_call): {sorted(left - listed)}"
    assert listed <= left | (listed - set(COMMAND_IDS)), "remove converted dialogs from the list"
    assert not (listed - left), f"converted: remove {sorted(listed - left)} from tests/fixtures/dialogs_without_calls.json"


def test_a_chain_of_calls_is_one_step(session):
    session.run_code("fit = Equation('c ~ z').fit(data.with_series(z='y'))")
    record = session.history[-1]
    assert type(record.command).__name__ == "Call"
    assert record.code == "fit = Equation('c ~ z').fit(data.with_series(z='y'))"
    assert record.call.arguments["data"]["$call"]["function"] == "pymetrix:dataset.with_series"
    assert record.reads == {"data": session.history[0].writes["data"]}
