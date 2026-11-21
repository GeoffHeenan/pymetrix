"""Seasonal adjustment (statsmodels' STL and classical decomposition) and calendar regressors; X-13 and TRAMO-SEATS are
the pymetrix-seasonal plugin's (packages/pymetrix-seasonal/tests)."""

import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from statsmodels.tsa.seasonal import STL as SMSTL  # noqa: E402
from statsmodels.tsa.seasonal import seasonal_decompose  # noqa: E402

from pymetrix import STL, Classical, Dataset  # noqa: E402
from pymetrix.metadata import Adjustment  # noqa: E402
from pymetrix.seasonal import calendar_effects  # noqa: E402


@pytest.fixture(scope="module")
def monthly():
    rng = np.random.default_rng(0)
    idx = pd.period_range("2005-01", "2024-12", freq="M", name="time")
    n = len(idx)
    factor = np.tile(1 + 0.1 * np.sin(2 * np.pi * np.arange(12) / 12), n // 12)
    trend = 100 * np.exp(0.002 * np.arange(n))
    frame = pd.DataFrame({"sales": trend * factor * np.exp(rng.normal(0, 0.01, n)), "balance": rng.normal(0, 1, n) + 5 * (factor - 1)}, index=idx)
    return Dataset.from_frame(frame), factor


def test_stl_is_statsmodels_stl(monthly):
    data, factor = monthly
    out = data.seasonal_adjust("sales", STL(seasonal=13))
    ref = SMSTL(np.log(data["sales"].to_numpy()), period=12, seasonal=13, robust=True).fit()
    assert np.allclose(out["sales_seasonal"], np.exp(ref.seasonal))
    assert np.allclose(out["sales_sa"], data["sales"] / np.exp(ref.seasonal))
    assert np.corrcoef(out["sales_seasonal"], factor)[0, 1] > 0.99
    assert out.variables["sales_sa"].adjustment == Adjustment.SA and out.variables["sales_sa"].recipe is not None
    additive = data.seasonal_adjust("balance")  # negative values: additive, a component around zero
    ref = SMSTL(data["balance"].to_numpy(), period=12, seasonal=7, robust=True).fit()
    assert np.allclose(additive["balance_seasonal"], ref.seasonal)
    assert np.allclose(additive["balance_sa"], data["balance"] - ref.seasonal)


def test_adjusted_series_record_what_statsmodels_was_given(monthly):
    data, _ = monthly
    extra = data.seasonal_adjust("sales", STL(seasonal=13)).variables["sales_sa"].extra
    assert extra["backend"].startswith("statsmodels ") and extra["backend"].endswith("(statsmodels.tsa.seasonal.STL)")
    assert extra["backend_options"] == "period=12; seasonal=13; trend=None; robust=True"
    classical = data.seasonal_adjust("balance", Classical(), derived=False).variables["balance_seasonal"].extra
    assert classical["backend_options"] == "model='additive'; period=12"


def test_classical_is_seasonal_decompose(monthly):
    data, _ = monthly
    out = data.seasonal_adjust("sales", Classical())
    ref = seasonal_decompose(data["sales"].to_numpy(), model="multiplicative", period=12)
    assert np.allclose(out["sales_seasonal"], ref.seasonal) and np.allclose(out["sales_sa"], data["sales"] / ref.seasonal)


def test_adjusted_series_follow_their_source_and_panels(monthly):
    data, _ = monthly
    out = data.seasonal_adjust("sales")
    edited = out.edit({"sales": {"2024-12": 500.0}})
    assert edited["sales_sa"].iloc[-1] > 3 * out["sales_sa"].iloc[-1]
    frame = data.frame.reset_index()
    panel = Dataset.from_frame(pd.concat([frame.assign(region="north"), frame.assign(region="south", sales=frame.sales * 2)]),
                               time="time", entity="region")
    adjusted = panel.seasonal_adjust("sales")
    north = adjusted.frame.xs("north", level="region")["sales_sa"]
    south = adjusted.frame.xs("south", level="region")["sales_sa"]
    assert np.allclose(south, 2 * north)  # each region adjusted on its own; the factors are the same
    with pytest.raises(ValueError, match="monthly and quarterly"):
        Dataset.from_frame(pd.DataFrame({"y": np.arange(1.0, 30)}, index=pd.period_range("1990", periods=29, freq="Y"))).seasonal_adjust("y")
    with pytest.raises(ValueError, match="at least three years"):
        Dataset.from_frame(data.frame.iloc[:24]).seasonal_adjust("sales")


def test_calendar_regressors_match_the_calendar():
    months = calendar_effects(pd.period_range("2021-01", "2024-12", freq="M"))
    assert list(months.columns) == ["td_mon", "td_tue", "td_wed", "td_thu", "td_fri", "td_sat", "easter_8", "leap_year"]
    march = months.loc[pd.Period("2021-03", "M")]
    assert march["td_mon"] == 1 and march["td_sat"] == 0  # March 2021: 5 Mondays, 4 Saturdays, 4 Sundays
    # Easter 2021 is 4 April: the 8 days before it are 27 March to 3 April
    assert march["easter_8"] == 0.625 and months.loc[pd.Period("2021-04", "M"), "easter_8"] == 0.375
    assert months.loc[pd.Period("2024-03", "M"), "easter_8"] == 1.0  # Easter 31 March 2024
    assert months.loc[pd.Period("2024-02", "M"), "leap_year"] == 0.75 and months.loc[pd.Period("2023-02", "M"), "leap_year"] == -0.25
    assert np.allclose(months.groupby(months.index.year)["easter_8"].sum(), 1.0)  # the window falls in each year once
    quarters = calendar_effects(pd.period_range("2021Q1", "2021Q4", freq="Q"), trading_days="one", leap_year=False)
    assert list(quarters["td_weekdays"]) == [-1.0, 0.0, 1.0, 1.0]  # Q1 2021: 64 weekdays, 26 weekend days
    assert list(quarters["easter_8"]) == [0.625, 0.375, 0.0, 0.0]


def test_calendar_regressors_in_a_dataset_and_a_regression(monthly):
    from pymetrix import Equation

    data, _ = monthly
    with_cal = data.calendar(trading_days="one")
    assert {"td_weekdays", "easter_8", "leap_year"} <= set(with_cal.names)
    assert with_cal.variables["td_weekdays"].label == "Weekdays less 5/2 of weekend days"
    fit = Equation("sales ~ td_weekdays + easter_8").fit(with_cal)
    assert set(fit.params.names) == {"const", "td_weekdays", "easter_8"}


def test_dialogs_record_and_replay(monthly, tmp_path):
    from pymetrix.commands import (
        CalendarRegressors,
        SeasonalAdjust,
        Session,
        load_project,
        save_project,
    )

    data, _ = monthly
    s = Session()
    s.objects["data"] = data
    s.run(SeasonalAdjust("sales", settings="seasonal=13"))
    s.run(CalendarRegressors(trading_days="one", easter=0))
    script = s.script()
    assert "data = data.seasonal_adjust('sales', method=STL(seasonal=13))" in script
    assert "data = data.calendar('one', easter=None)" in script
    assert {"sales_sa", "sales_seasonal", "td_weekdays", "leap_year"} <= set(s["data"].names) and "easter_8" not in s["data"].names
    assert SeasonalAdjust("").validate(s) == ["choose the series to adjust"]
    reopened = load_project(save_project(s, tmp_path / "sa.pymx"))
    assert np.allclose(reopened["data"]["sales_sa"], s["data"]["sales_sa"])
    parquet = tmp_path / "sa.parquet"
    s["data"].write(str(parquet))  # the recipe survives a save
    assert Dataset.read(str(parquet)).variables["sales_sa"].recipe == s["data"].variables["sales_sa"].recipe
    assert os.path.getsize(parquet) > 0


def test_every_stl_and_classical_setting_reaches_statsmodels(monthly):
    data, _ = monthly
    y = data["sales"].to_numpy()
    settings = dict(low_pass=15, seasonal_deg=0, trend_deg=0, low_pass_deg=0, seasonal_jump=2, trend_jump=3, low_pass_jump=2)
    out = data.seasonal_adjust("sales", STL(seasonal=13, trend=25, inner_iter=3, outer_iter=4, **settings), derived=False)
    ref = SMSTL(np.log(y), period=12, seasonal=13, trend=25, robust=True, **settings).fit(inner_iter=3, outer_iter=4)
    assert np.allclose(out["sales_seasonal"], np.exp(ref.seasonal), rtol=1e-12)
    recorded = out.variables["sales_sa"].extra["backend_options"]
    assert "low_pass=15" in recorded and "inner_iter=3" in recorded
    plain = data.seasonal_adjust("sales", STL(seasonal=13), derived=False).variables["sales_sa"].extra["backend_options"]
    assert "low_pass" not in plain and "inner_iter" not in plain  # statsmodels' defaults stay out of the record
    classical = data.seasonal_adjust("balance", Classical(two_sided=False, extrapolate_trend=6), derived=False)
    ref = seasonal_decompose(data["balance"].to_numpy(), model="additive", period=12, two_sided=False, extrapolate_trend=6)
    assert np.allclose(classical["balance_seasonal"], ref.seasonal, rtol=1e-12)
    assert "two_sided=False" in classical.variables["balance_sa"].extra["backend_options"]


def test_a_project_that_used_x13_opens_without_the_seasonal_plugin(tmp_path):
    """X-13 and TRAMO-SEATS are the pymetrix-seasonal plugin's: without it, a project that used one opens with that
    step kept, not run, and the plugin named, as for any plugin's step."""
    import json
    import zipfile

    from pymetrix.commands import Session, load_project, save_project
    from pymetrix.commands.seasonal import SeasonalAdjust
    from pymetrix.plugins import registry

    if registry("seasonal").find("X13") is not None:
        pytest.skip("the seasonal plugin is installed")
    s = Session()
    s.objects["m"] = Dataset.example("sample_monthly")
    s.run(SeasonalAdjust(variables="retail", data="m", name="adj"))
    path = save_project(s, tmp_path / "p.pymx")
    forged = tmp_path / "x13.pymx"
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(forged, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "commands.json":  # as saved where the plugin was installed
                steps = json.loads(data)
                for step in steps:
                    if step.get("command") == "SeasonalAdjust":
                        step["fields"]["method"] = "X-13ARIMA-SEATS"
                data = json.dumps(steps).encode()
            dst.writestr(item, data)
    opened = load_project(forged)
    step = next(r for r in opened.history if type(r.command).__name__ == "SeasonalAdjust")
    assert step.skipped and "pymetrix-seasonal" in step.error and "pip install" in step.error
    assert any("pymetrix-seasonal" in k or "pymetrix-seasonal" in v for k, v in opened.missing_plugins.items())
