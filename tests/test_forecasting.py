"""The forecast workflow: rounds, judgement, the store, fan charts, backtests and the Diebold-Mariano test."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pymetrix import VAR, Bayes, Dataset, Equation, Minnesota
from pymetrix.forecasting import Backtest, ForecastRound, ForecastStore, backtest, diebold_mariano


@pytest.fixture(scope="module")
def data():
    return Dataset.example("sample_macro").with_series(ly="100*log(y)", lc="100*log(c)")


@pytest.fixture(scope="module")
def round_2022(data):
    var = VAR(["ly", "lc"], lags=2).fit(data, span=(None, "2022Q4"))
    return ForecastRound.from_forecast("2022Q4 round", var.forecast(data, ("2023Q1", "2024Q4")), data=data.select(span=(None, "2022Q4")),
                                       notes="baseline")


def test_rounds_from_forecasts(data, round_2022):
    assert round_2022.variables == ["ly", "lc"] and round_2022.data_end == "2022Q4"
    assert list(round_2022.paths.columns) == ["ly", "ly_se", "lc", "lc_se"] and len(round_2022.paths) == 8
    eq = Equation("lc ~ ly + lc[-1]").fit(data, span=(None, "2022Q4"))
    single = ForecastRound.from_forecast("eq", eq.forecast(data, ("2023Q1", "2023Q4")), as_name="lc")
    assert single.variables == ["lc"] and "lc_se" in single.paths
    with pytest.raises(ValueError, match="as_name"):
        ForecastRound.from_forecast("eq", eq.forecast(data, ("2023Q1", "2023Q4")))


def test_judgement_is_logged_and_the_base_kept(round_2022):
    adjusted = round_2022.adjust("ly", {"2023Q1": -0.5, "2023Q2": -0.25}, reason="port strike")
    assert adjusted.path("ly").iloc[0] == pytest.approx(round_2022.path("ly").iloc[0] - 0.5)
    assert np.allclose(adjusted.base["ly"], round_2022.path("ly"))
    log = adjusted.judgement().to_frame()
    assert list(log["reason"]) == ["port strike", "port strike"] and np.allclose(log["change"], [-0.5, -0.25])
    fixed = adjusted.adjust("lc", 600.0, reason="hold flat", how="set")
    assert np.allclose(fixed.path("lc"), 600.0) and len(fixed.adjustments) == 2 + 8
    with pytest.raises(ValueError, match="give a reason"):
        round_2022.adjust("ly", 1.0, reason="")
    with pytest.raises(KeyError, match="not a forecast period"):
        round_2022.adjust("ly", {"2030Q1": 1.0}, reason="x")


def test_the_store_saves_compares_and_evaluates_rounds(data, round_2022, tmp_path):
    store = ForecastStore(tmp_path / "forecasts")
    store.save(round_2022.adjust("ly", {"2023Q1": -0.5}, reason="port strike"))
    var = VAR(["ly", "lc"], lags=2).fit(data, span=(None, "2023Q2"))
    store.save(ForecastRound.from_forecast("2023Q2 round", var.forecast(data, ("2023Q3", "2025Q2")), data=data.select(span=(None, "2023Q2"))))
    assert store.names() == ["2022Q4 round", "2023Q2 round"]
    with pytest.raises(FileExistsError, match="a different round"):
        store.save(round_2022)  # the stored one has judgement
    store.save(store.load("2022Q4 round"))  # the same round again: nothing to do
    back = store.load("2022Q4 round")
    assert back.adjustments[0]["reason"] == "port strike" and np.allclose(back.base["ly"], round_2022.path("ly"))
    assert back.paths.index.freqstr == round_2022.paths.index.freqstr
    revisions = store.compare("2022Q4 round", "2023Q2 round", ["ly"]).to_frame()
    assert revisions.loc["2023Q3", "ly revision"] == pytest.approx(revisions.loc["2023Q3", "ly (2023Q2 round)"] - revisions.loc["2023Q3", "ly (2022Q4 round)"])
    assert list(store.history("ly").columns) == ["2022Q4 round", "2023Q2 round"]
    accuracy = store.accuracy("ly", data).to_frame()
    assert list(accuracy["horizon"])[:2] == [1, 2] and accuracy["n"].iloc[0] == 2
    actual = data["ly"]
    expected_h1 = [actual[pd.Period("2023Q1", "Q")] - back.path("ly").iloc[0], actual[pd.Period("2023Q3", "Q")] - store.load("2023Q2 round").path("ly").iloc[0]]
    assert accuracy["RMSE"].iloc[0] == pytest.approx(np.sqrt(np.mean(np.square(expected_h1))))
    assert len(store.plot_history("ly", data).panels) == 1


def test_fan_chart(data, round_2022):
    pytest.importorskip("matplotlib")
    fig = round_2022.plot(data)
    bands = [layer for layer in fig.panels[0].drawn() if type(layer).__name__ == "Band"]  # the fan, as drawn
    assert [b.label for b in bands] == ["90%", "60%", "30%"]
    se = round_2022.paths["ly_se"]
    assert np.allclose(bands[0].upper.iloc[1:] - round_2022.paths["ly"], stats.norm.ppf(0.95) * se)
    last = data["ly"][data["ly"].index < round_2022.paths.index[0]].dropna()  # the fan starts at the last actual
    assert bands[0].upper.index[0] == last.index[-1] and bands[0].upper.iloc[0] == bands[0].lower.iloc[0] == last.iloc[-1]


def test_diebold_mariano():
    rng = np.random.default_rng(1)
    e1, e2 = rng.normal(0, 1, 120), rng.normal(0, 1.3, 120)
    # at horizon 1 the HLN statistic is the one-sample t-test of the loss differential
    dm = diebold_mariano(e1, e2, 1)
    t = stats.ttest_1samp(e1**2 - e2**2, 0.0)
    assert dm.statistic == pytest.approx(t.statistic) and dm.p_value == pytest.approx(t.pvalue)
    assert dm.statistic < 0 and dm.p_value < 0.05  # the first forecast is more accurate
    assert diebold_mariano(e2, e1, 1).statistic == pytest.approx(-dm.statistic)
    same = diebold_mariano(e1, e1, 4)
    assert same.statistic == 0 and same.p_value == 1
    # horizon h: the long-run variance includes h-1 autocovariances, by the formula
    d = np.abs(e1) - np.abs(e2)
    n, h = len(d), 3
    u = d - d.mean()
    lrv = np.sum(u * u) / n + 2 * sum(np.sum(u[k:] * u[:-k]) / n for k in (1, 2))
    raw = d.mean() / np.sqrt(lrv / n)
    assert diebold_mariano(e1, e2, h, "absolute", small_sample=False).statistic == pytest.approx(raw)
    assert diebold_mariano(e1, e2, h, "absolute").statistic == pytest.approx(raw * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n))


def test_backtests(data):
    var = VAR(["ly", "lc"], lags=2).fit(data)
    bt = backtest(var, data, ("2019Q1", "2022Q4"), horizon=4, label="VAR")
    assert isinstance(bt, Backtest) and bt.errors["origin"].nunique() == 16 and set(bt.errors["horizon"]) == {1, 2, 3, 4}
    row = bt.errors[(bt.errors["origin"] == "2020Q2") & (bt.errors["horizon"] == 2) & (bt.errors["variable"] == "lc")].iloc[0]
    direct = VAR(["ly", "lc"], lags=2).fit(data, span=(None, "2020Q2")).forecast(data, ("2020Q3", "2020Q4"))
    assert row["forecast"] == pytest.approx(direct["lc"].iloc[1])  # no look-ahead: estimated on data to the origin
    assert row["error"] == pytest.approx(data["lc"][pd.Period("2020Q4", "Q")] - row["forecast"])
    accuracy = bt.accuracy(["lc"]).to_frame()
    assert list(accuracy.index) == ["lc, h=1", "lc, h=2", "lc, h=3", "lc, h=4"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bvar = backtest(VAR(["ly", "lc"], lags=2).fit(data, method=Bayes(prior=Minnesota(), draws=50)), data, ("2019Q1", "2022Q4"), horizon=4)
    test = bt.compare(bvar, "lc", 2)
    assert test.details["observations"] == 16 and 0 <= test.p_value <= 1
    eq = backtest(Equation("lc ~ ly + lc[-1]").fit(data), data, ("2021Q1", "2022Q4"), horizon=2)
    assert set(eq.errors["variable"]) == {"lc"} and len(eq.errors) == 16


def test_real_time_backtests_use_each_vintage(data):
    var = VAR(["ly", "lc"], lags=1).fit(data)
    origins = pd.period_range("2021Q1", "2021Q4", freq="Q")
    vintages = {str(o): data.with_series(ly="ly + 10", lc="lc + 10") for o in origins}  # an early release ten points higher
    real_time = backtest(var, data, ("2021Q1", "2021Q4"), horizon=1, vintages=vintages)
    revised = backtest(var, data, ("2021Q1", "2021Q4"), horizon=1)
    shift = (revised.errors["error"] - real_time.errors["error"]).to_numpy()
    assert np.allclose(shift, 10, atol=1e-6)  # forecasts from the early release are ten higher; errors against the latest data
    with pytest.raises(KeyError, match="no vintage"):
        backtest(var, data, ("2021Q1", "2022Q1"), horizon=1, vintages=vintages)


def test_dialogs_record_and_replay(data, tmp_path):
    from pymetrix.commands import AdjustRound, CompareRounds, Forecast, FitVAR, MakeRound, RunBacktest, SaveRound, Session, load_project, save_project

    s = Session()
    s.objects["data"] = data
    s.run(FitVAR("ly, lc", lags="2", end="2022Q4"))
    s.run(Forecast("var", "2023Q1", "2024Q4", name="fc"))
    s.run(MakeRound("fc", "2022Q4 round", notes="baseline"))
    s.run(AdjustRound("forecast_round", "ly", "2023Q1=-0.5, 2023Q2=-0.25", "port strike"))
    folder = str(tmp_path / "store")
    s.run(SaveRound("forecast_round", folder=folder))
    s.run(MakeRound("fc", "copy", name="round2"))
    s.run(SaveRound("round2", folder=folder))
    s.run(CompareRounds("2022Q4 round", "copy", folder=folder, variables="ly"))
    s.run(RunBacktest("var", "2021Q1", "2022Q4", horizon=2))
    script = s.script()
    for line in ("forecast_round = ForecastRound.from_forecast('2022Q4 round', fc, data=data, notes='baseline')",
                 "forecast_round = forecast_round.adjust('ly', {'2023Q1': -0.5, '2023Q2': -0.25}, 'port strike')",
                 "backtest_result = backtest(var, data, ('2021Q1', '2022Q4'), horizon=2, label='var')"):
        assert line in script
    assert s["revisions"].to_frame().loc["2023Q1", "ly revision"] == pytest.approx(0.5)  # the copy has no judgement
    assert AdjustRound("forecast_round", "ly", "2023Q1 -0.5", "x").validate(s) == ["write each change as period=value, e.g. 2026Q4=-0.3 (not '2023Q1 -0.5')"]
    reopened = load_project(save_project(s, tmp_path / "fc.pymx"), trust=True)
    assert len(reopened["forecast_round"].adjustments) == 2
    from pymetrix.ui.render import view_sections

    titles = [t for t, _ in view_sections(s["forecast_round"])]
    assert titles[0] == "Paths" and titles[-1] == "Judgement"
    assert [t for t, _ in view_sections(s["backtest_result"])][0] == "Accuracy"


def test_rounds_keep_recent_history_for_their_charts(data, tmp_path):
    from pymetrix import VAR, ForecastRound, ForecastStore

    var = VAR(["ly", "lc"], lags=1).fit(data)
    rnd = ForecastRound.from_forecast("r", var.forecast(data, ("2025Q1", "2026Q4")), data=data)
    assert rnd.history is not None and len(rnd.history) == 16 and str(rnd.history.index[-1]) == "2024Q4"
    actual = [layer for layer in rnd.plot().panels[0].layers if getattr(layer, "label", "") == "actual"]
    assert actual and len(actual[0].y) == 16  # the chart shows the recent data without being given them
    store = ForecastStore(tmp_path / "s")
    store.save(rnd)
    back = store.load("r")
    np.testing.assert_allclose(back.history.to_numpy(), rnd.history.to_numpy())
    assert ForecastRound.from_forecast("bare", var.forecast(data, ("2025Q1", "2025Q4"))).history is None


def test_a_real_time_backtest_forecasts_past_each_vintage():
    """Each vintage ends at its origin, as real-time data do: forecasting past it must work for an equation with a
    lag of its target (the static forecast there is empty, and was compared with the dynamic one)."""
    idx = pd.period_range("2010Q1", periods=40, freq="Q")
    y = 101.0 + np.arange(40) + np.random.default_rng(0).normal(0, 1, 40)
    data = Dataset.from_frame(pd.DataFrame({"y": y}, index=idx))
    vintages = {o: Dataset.from_frame(pd.DataFrame({"y": y[:i + 1]}, index=idx[:i + 1])) for i, o in enumerate(idx) if str(o) >= "2015Q4"}
    result = backtest(Equation("y ~ y[-1]").fit(data), data, ("2015Q4", "2017Q4"), horizon=2, vintages=vintages)
    assert len(result.errors) == 18 and np.isfinite(result.errors["error"]).all()


def test_a_backtest_records_how_its_forecasts_were_made(data):
    """Provenance as every fit has it: the model and method re-estimated, the library, the forecasts not yet scored
    (no outcome), and which data were asked for, for the outcomes and at each origin."""
    origins = pd.period_range("2023Q1", "2024Q4", freq="Q")
    vintages = {o: data.select(span=(None, str(o))) for o in origins}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bt = backtest(Equation("lc ~ lc[-1]").fit(data), data, ("2023Q1", "2024Q4"), horizon=2, vintages=vintages)
    prov = bt.provenance
    assert prov.spec["model"]["text"] == "lc ~ lc[-1]" and prov.method["refit"]["type"] == "OLS"
    assert prov.backend == {"name": "pymetrix", "version": prov.package_version, "wraps": "pymetrix.OLS"}
    waiting = bt.errors["actual"].isna().sum()
    assert waiting > 0 and prov.dropped == waiting and prov.nobs == len(bt.errors) - waiting
    assert prov.dropped_rows["by_reason"] == {"no outcome yet": waiting}
    assert prov.rows == [["2023Q1", "2024Q4"]]
    assert prov.vintage_request["outcomes"]["policy"] == "example" and set(prov.vintage_request["origins"]) == {str(o) for o in origins}


def test_a_backtest_gathers_its_refits_warnings_by_origin():
    from types import SimpleNamespace

    from pymetrix.forecasting import _refit_record

    def fit(warns, backend):
        return SimpleNamespace(provenance=SimpleNamespace(spec={"type": "Equation"}, method={"type": "ML"}, warnings=warns, messages=(),
                                                          backend=backend, options=None, converged=not warns))

    record = _refit_record({"2020Q1": fit(("ConvergenceWarning: did not converge",), {"name": "statsmodels"}),
                            "2020Q2": fit((), {"name": "statsmodels"}),
                            "2020Q3": fit(("ConvergenceWarning: did not converge",), {"name": "statsmodels"})})
    assert record["warnings"] == ("ConvergenceWarning: did not converge (origins 2020Q1, 2020Q3)",)
    assert record["backend"] == {"name": "statsmodels"} and record["converged"] is False
