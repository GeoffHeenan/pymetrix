"""Panel data tools: group-wise functions, pivot tables, wide and long, slices, coverage, balancing, dialogs."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation
from pymetrix.expr import ExpressionError


@pytest.fixture()
def panel():
    rng = np.random.default_rng(0)
    rows = [{"country": c, "t": p, "y": 10.0 * j + i + rng.normal(), "c": 5.0 * j + i, "region": "Oceania" if c == "AUS" else "Pacific"}
            for j, c in enumerate(["AUS", "NZL", "WSM"]) for i, p in enumerate(pd.period_range("2018", periods=5, freq="Y"))]
    frame = pd.DataFrame(rows)
    frame = frame[~((frame.country == "WSM") & (frame.t == pd.Period("2018", "Y")))]  # Samoa starts a year later
    return Dataset.from_frame(frame, time="t", entity="country")


def test_group_wise_functions(panel):
    y = panel["y"]
    by_country = y.groupby(level="country")
    assert np.allclose(panel.eval("mean(y, country)"), by_country.transform("mean"))
    assert np.allclose(panel.eval("demean(y, country)"), y - by_country.transform("mean"))
    assert np.allclose(panel.eval("share(y, t)"), y / y.groupby(level="t").transform("sum"))
    assert np.allclose(panel.eval("mean(y)"), y.mean())
    assert np.allclose(panel.eval("sum(c, region)"), panel["c"].groupby(panel["region"]).transform("sum"))
    assert panel.eval("count(y, country)").loc["WSM"].iloc[0] == 4
    assert panel.eval("rank(y, t)").max() == 3
    lagged = panel.eval("mean(y, country)[-1]")  # the mean of last period's values, within country
    assert lagged.loc["AUS"].iloc[0] == pytest.approx(y.loc["AUS"].iloc[:4].mean())
    # derived series follow their inputs, including a grouping series
    derived = panel.derive(gap="y - mean(y, t)", reg="mean(c, region)")
    assert set(derived.derived["reg"].inputs(derived)) == {"c", "region"}
    changed = derived.edit({"y": {("AUS", "2020"): 100.0}})
    assert changed["gap"].loc[("AUS", pd.Period("2020", "Y"))] > 50
    # usable as regressors: deviations from cross-country means (time effects by hand)
    fit = Equation("demean(y, t) ~ demean(c, t) - 1").fit(panel)
    assert fit.params.names == ("demean(c, t)",)
    with pytest.raises(ExpressionError, match="group by an axis or a series"):
        panel.eval("mean(y, 2)")
    with pytest.raises(ExpressionError, match="neither an axis nor a series"):
        panel.eval("mean(y, continent)")


def test_pivot_tables(panel):
    table = panel.pivot("region", "t", "y", agg=["mean", "count"], margins=True)
    frame = table.to_frame()
    assert list(frame.index) == ["Oceania", "Pacific", "Total"]
    assert frame.loc["Pacific", "count | 2018"] == 1 and frame.loc["Total", "count | Total"] == 14
    assert frame.loc["Oceania", "mean | 2019"] == pytest.approx(panel["y"].loc[("AUS", pd.Period("2019", "Y"))])
    two = panel.pivot(["region", "country"], values=["y", "c"], agg={"y": "mean", "c": "sum"}).to_frame()
    assert list(two.index) == ["Oceania | AUS", "Pacific | NZL", "Pacific | WSM"] and two.loc["Pacific | WSM", "c"] == 50
    filled = panel.pivot("country", "t", "y", fill=0.0).to_frame()
    assert filled.loc["WSM", "2018"] == 0.0
    assert len(panel.pivot("country", "t", "y", span=("2021", None)).to_frame().columns) == 2
    with pytest.raises(KeyError, match="no axis or series 'continent'"):
        panel.pivot("continent")


def test_wide_and_long(panel):
    wide = panel.wide("y", "c")
    assert wide.kind == "time_series" and wide.names == ["y_AUS", "y_NZL", "y_WSM", "c_AUS", "c_NZL", "c_WSM"]
    assert wide.variables["y_WSM"].label == "y (WSM)" and np.isnan(wide["y_WSM"].iloc[0])
    back = wide.long(entity="country")
    assert back.kind == "panel" and set(back.names) == {"y", "c"}
    pd.testing.assert_series_equal(back["y"].dropna(), panel["y"], check_names=False)
    explicit = wide.long(entity="country", stubs=["y"])  # c_* columns stay: the same for every entity
    assert set(explicit.names) == {"y", "c_AUS", "c_NZL", "c_WSM"}
    sheet = Dataset.from_frame(pd.DataFrame({"t": pd.period_range("2020", periods=2, freq="Y"), "AUS": [1.0, 2.0], "NZL": [3.0, 4.0]}), time="t")
    gdp = sheet.long(entity="country", name="gdp")
    assert gdp["gdp"].loc[("NZL", pd.Period("2021", "Y"))] == 4.0
    named = Dataset.from_frame(pd.DataFrame({"id": ["United States", "New Zealand"], "x": [1.0, 2.0], "t": ["2020", "2020"]}), time="t", entity="id", freq="Y")
    assert named.wide().names == ["x_New_Zealand", "x_United_States"]
    with pytest.raises(ValueError, match="share a prefix"):
        sheet.long()


def test_slices(panel):
    samoa = panel.entity("WSM")
    assert samoa.kind == "time_series" and len(samoa) == 4 and "index" not in samoa.names
    section = panel.cross_section("2020")
    assert section.kind == "cross_section" and list(section.frame.index) == ["AUS", "NZL", "WSM"]
    assert len(panel.xs(country="AUS")) == 5
    with pytest.raises(KeyError, match="no country"):
        panel.entity("FJI")


def test_coverage_balance_and_complete(panel):
    cov = panel.coverage("y").to_frame()
    assert cov.loc["WSM", "first"] == "2019" and cov.loc["AUS", "observations"] == 5
    assert "Not balanced" in panel.coverage().notes[0]
    gappy = panel.edit({"y": {("NZL", "2020"): None}})
    assert gappy.coverage("y").to_frame().loc["NZL", "gaps"] == 1
    by_period = panel.balance()
    assert "Balanced" in by_period.coverage().notes[0] and len(by_period) == 12
    by_entity = panel.balance(keep="entities")
    assert sorted(by_entity.frame.index.get_level_values("country").unique()) == ["AUS", "NZL"]
    full = panel.complete()
    assert len(full) == 15 and np.isnan(full["y"].loc[("WSM", pd.Period("2018", "Y"))])
    stats = panel.describe_panel("y").to_frame()
    assert list(stats.index) == ["y overall", "y between", "y within"]
    assert stats.loc["y between", "sd"] > stats.loc["y within", "sd"]  # countries differ more than years


def test_dialogs_record_and_replay(panel, tmp_path):
    from pymetrix.commands import BalancePanel, PanelStatistics, PivotTable, Session, Slice, ToLong, ToWide, form_spec, load_project, save_project

    s = Session()
    s.objects["data"] = panel
    level = next(f for f in form_spec(PivotTable) if f.name == "rows")
    assert s.choices(level, {"data": "data"})[:4] == ["country", "t", "region", "y"]
    entity = next(f for f in form_spec(Slice) if f.name == "entity")
    assert s.choices(entity, {"data": "data"}) == ["", "AUS", "NZL", "WSM"]  # optional: blank first, nothing picked unasked
    s.run(PivotTable(rows="region", columns="t", values="y", agg="mean, count", margins=True))
    s.run(ToWide(series="y"))
    s.run(ToLong("wide", entity="country", name="back"))
    s.run(Slice(entity="WSM", name="samoa"))
    s.run(BalancePanel(keep="entities"))
    s.run(PanelStatistics(show="variation", series="y"))
    script = s.script()
    for line in ("pivot = data.pivot('region', columns='t', values='y', agg=['mean', 'count'], margins=True)",
                 "wide = data.wide('y')", "back = wide.long('country')", "samoa = data.entity('WSM')",
                 "balanced = data.balance(keep='entities')", "panel_stats = data.describe_panel('y')"):
        assert line in script
    assert Slice().validate(s) == ["choose either an entity or a period"]
    assert PivotTable().validate(s) == ["choose at least one row"]
    reopened = load_project(save_project(s, tmp_path / "panel.pymx"))
    assert reopened["samoa"].kind == "time_series" and reopened["pivot"].to_frame().shape == s["pivot"].to_frame().shape
