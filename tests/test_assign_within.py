"""An empty dated dataset (Dataset.empty, File → New dataset; EViews' wfcreate) and assignment within a sample
(with_series(within=...), Generate series → Within; EViews' smpl then y = ...), and a series of missing values
(with_series(y=None); EViews' series y).

Checks are against values written out by hand or computed with numpy (a closed form for the recursive assignment,
1.02 to a power), against the API call each dialog stands for, and the recorded steps' scripts and projects replay."""

import json

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Sample
from pymetrix.commands import GenerateSeries, Session, load_project, save_project
from pymetrix.commands.builtin import NewDataset
from pymetrix.commands.samples import NewSample
from pymetrix.commands.registry import from_json, to_json, upgrade
from pymetrix.errors import DataError


@pytest.fixture
def quarterly():
    index = pd.period_range("1999Q1", periods=12, freq="Q")
    return Dataset.from_frame(pd.DataFrame({"x": np.arange(1.0, 13.0)}, index=index))


@pytest.fixture
def panel():
    rows = [(c, str(p), float(i + 10 * k)) for k, c in enumerate(["DEU", "FRA"])
            for i, p in enumerate(pd.period_range("2000Q1", periods=6, freq="Q"))]
    frame = pd.DataFrame(rows, columns=["country", "time", "x"])
    return Dataset.from_frame(frame, time="time", entity="country", freq="Q")


def _replay(s, name="data", **objects):
    ns = dict(objects)
    exec(s.script(), ns)
    return ns[name]


# -- Dataset.empty --------------------------------------------------------------------------------------------------


def test_empty_has_every_period_and_no_series():
    data = Dataset.empty("Q", "1990Q1", "2030Q4")
    assert data.kind == "time_series" and data.names == [] and len(data) == 41 * 4
    periods = data.frame.index
    assert periods.name == "time" and str(periods[0]) == "1990Q1" and str(periods[-1]) == "2030Q4"
    assert (np.diff(periods.asi8) == 1).all()  # consecutive, no gaps
    assert data.time_axis.name == "time" and data.time_axis.freq.startswith("Q")
    for freq, start, end, n, first in (("Y", "1990", "2000", 11, "1990"), ("A", "1990", "1999", 10, "1990"),
                                       ("M", "2000-01", "2001-12", 24, "2000-01"), ("Y-JUN", "2000", "2003", 4, "2000"),
                                       ("D", "2024-02-27", "2024-03-01", 4, "2024-02-27"), ("q", "2000q1", "2000q4", 4, "2000Q1")):
        made = Dataset.empty(freq, start, end)
        assert len(made) == n and str(made.frame.index[0]) == first, freq
    assert len(Dataset.empty("Q", "2000Q1", "2000Q1")) == 1


def test_empty_refuses_what_is_not_a_range():
    with pytest.raises(DataError, match="ends .* before it starts"):
        Dataset.empty("Q", "2000Q1", "1990Q1")
    with pytest.raises(DataError, match="not a frequency"):
        Dataset.empty("ZZ", "1990", "1991")
    with pytest.raises(DataError, match="not periods of frequency Q"):
        Dataset.empty("Q", "1990Q5", "2000Q1")


def test_empty_then_series_as_in_eviews():
    """wfcreate q 1990q1 1990q4, series y, smpl 1990q2 1990q3, y = 1: missing, 1, 1, missing."""
    data = Dataset.empty("Q", "1990Q1", "1990Q4").with_series(y=None).with_series(y="1", within=("1990Q2", "1990Q3"))
    np.testing.assert_array_equal(data["y"].to_numpy(), [np.nan, 1.0, 1.0, np.nan])


def test_new_dataset_dialog_is_the_call_and_replays(tmp_path):
    s = Session()
    s.run(NewDataset(freq="M", start="2020-01", end="2021-06", name="scen"))
    expected = Dataset.empty("M", "2020-01", "2021-06")
    pd.testing.assert_frame_equal(s["scen"].frame, expected.frame)
    assert s.history[-1].code == "scen = Dataset.empty('M', '2020-01', '2021-06')"
    s.run(GenerateSeries("g", "1", data="scen", start="2021-01"))
    replayed = _replay(s, "scen")
    pd.testing.assert_frame_equal(replayed.frame, s["scen"].frame)
    path = save_project(s, tmp_path / "p.pymx")
    pd.testing.assert_frame_equal(load_project(path)["scen"].frame, s["scen"].frame)
    assert NewDataset(start="2000Q1", end="1990Q1").validate(s) == ["the dataset ends (1990Q1) before it starts (2000Q1)"]
    assert NewDataset(start="", end="1990Q1").validate(s) == ["give the first and last periods"]
    assert NewDataset(start="2000Q1", end="2001Q1", name="1x").validate(s)
    assert "target" not in NewDataset.__dataclass_fields__


# -- assignment within a sample -------------------------------------------------------------------------------------


def test_within_keeps_the_other_values(quarterly):
    out = quarterly.with_series(x="100", within=("2000Q1", "2000Q4"))
    expected = np.arange(1.0, 13.0)
    expected[4:8] = 100
    np.testing.assert_array_equal(out["x"].to_numpy(), expected)
    for within in ("2000Q1:2000Q4", Sample(("2000Q1", "2000Q4")), Sample(start="2000Q1", end="2000Q4")):
        np.testing.assert_array_equal(quarterly.with_series(x="100", within=within)["x"].to_numpy(), expected)
    np.testing.assert_array_equal(quarterly["x"].to_numpy(), np.arange(1.0, 13.0))  # the original is unchanged


def test_a_new_series_is_missing_outside(quarterly):
    out = quarterly.with_series(y="x*2", within=Sample(start="2001Q1"))
    expected = np.where(np.arange(12) >= 8, 2 * np.arange(1.0, 13.0), np.nan)
    np.testing.assert_array_equal(out["y"].to_numpy(), expected)
    assert out.variables["y"].derived_from == "x*2 within from 2001Q1"


def test_lags_reach_outside_the_sample(quarterly):
    out = quarterly.with_series(y="x[-1]", within=("2000Q1", "2000Q2"))  # x[-1] at 2000Q1 is 1999Q4's x, 4
    np.testing.assert_array_equal(out["y"].to_numpy()[4:6], [4.0, 5.0])


def test_own_lags_are_assigned_one_period_at_a_time(quarterly):
    """EViews computes y = y(-1)*1.02 under smpl observation by observation: 1.02**k from the last value before."""
    out = quarterly.with_series(x="x[-1]*1.02", within=Sample(start="2000Q1"))
    k = np.arange(1, 9)
    np.testing.assert_allclose(out["x"].to_numpy()[4:], 4.0 * 1.02 ** k, rtol=1e-14)
    np.testing.assert_array_equal(out["x"].to_numpy()[:4], [1.0, 2.0, 3.0, 4.0])
    # a lead sees the values before the call (not yet assigned); the series itself at t too
    lead = quarterly.with_series(x="x[+1] + x", within=("2000Q1", "2000Q2"))
    np.testing.assert_array_equal(lead["x"].to_numpy()[4:6], [6.0 + 5.0, 7.0 + 6.0])
    # a new series that uses itself starts from missing values, as in EViews
    assert quarterly.with_series(z="z[-1] + 1", within=Sample(start="2000Q1"))["z"].isna().all()
    # without within, as before: one pass over the values before the call
    once = quarterly.with_series(x="x[-1]*2")["x"].to_numpy()
    np.testing.assert_array_equal(once[1:], 2 * np.arange(1.0, 12.0))


def test_where_and_panels(panel, quarterly):
    out = quarterly.with_series(y="1", within=Sample(where="x > 9"))
    np.testing.assert_array_equal(out["y"].isna().to_numpy(), quarterly["x"].to_numpy() <= 9)
    # a panel: the sample's rows in every entity, lags within each entity, one entity by level
    out = panel.with_series(x="x[-1] + 100", within=Sample(start="2000Q3", country=["FRA"]))
    deu = out.frame.xs("DEU", level="country")["x"].to_numpy()
    fra = out.frame.xs("FRA", level="country")["x"].to_numpy()
    np.testing.assert_array_equal(deu, np.arange(0.0, 6.0))
    np.testing.assert_array_equal(fra, [10.0, 11.0, 111.0, 211.0, 311.0, 411.0])
    both = panel.with_series(y="x", within=Sample(windows={"DEU": ("2001Q1", None), "*": (None, "2000Q1")}))
    y = both.frame["y"].unstack("country")
    assert y["DEU"].notna().sum() == 2 and y["FRA"].notna().sum() == 1


def test_missing_values_and_series(quarterly):
    out = quarterly.with_series(y=None)
    assert out["y"].isna().all() and out["y"].dtype == float
    holes = quarterly.with_series(x=None, within=("2000Q1", "2000Q2"))
    assert holes["x"].isna().sum() == 2 and np.isnan(holes["x"].to_numpy()[4:6]).all()
    given = pd.Series([50.0, 60.0], index=pd.period_range("2000Q1", periods=2, freq="Q"))
    part = quarterly.with_series(x=given, within=("2000Q2", "2000Q4"))  # only 2000Q2 of the series is in the sample
    np.testing.assert_array_equal(part["x"].to_numpy()[4:8], [5.0, 60.0, np.nan, np.nan])


def test_metadata_and_derived_series(quarterly):
    data = quarterly.with_info("x", label="Output", units="bn").derive(dx="x*10")
    out = data.with_series(x="0", within=("2000Q1", "2000Q1"))
    assert out.variables["x"].label == "Output" and out.variables["x"].units == "bn"
    assert out.variables["x"].derived_from == "0 within 2000Q1 to 2000Q1"
    np.testing.assert_array_equal(out["dx"].to_numpy(), 10 * out["x"].to_numpy())  # dependents recomputed
    with pytest.raises(DataError, match="selects no rows"):
        quarterly.with_series(y="1", within=("2030Q1", "2030Q4"))
    with pytest.raises(DataError, match="within= is for with_series"):
        quarterly.with_each("x", "{x}_2", "{x}", derived=True, within=("2000Q1", None))
    each = quarterly.with_each("x", "{x}_2", "{x}*2", within=("2000Q1", "2000Q1"))
    assert each["x_2"].notna().sum() == 1


def test_recorded_code_replays(quarterly, tmp_path):
    s = Session()
    s.objects["data"] = quarterly
    for line in ('data = data.with_series(y="x[-1]*1.02", within=Sample(start="2000Q1"))', "data = data.with_series(e=None)",
                 'w = Dataset.empty("Q", "1990Q1", "2030Q4")'):
        s.run_code(line)
    assert [r.call.to_code() for r in s.history[-3:]] == [
        "data = data.with_series(within=Sample(start='2000Q1'), y='x[-1]*1.02')", "data = data.with_series(e=None)",
        "w = Dataset.empty('Q', '1990Q1', '2030Q4')"]
    replayed = _replay(s, data=quarterly)
    pd.testing.assert_frame_equal(replayed.frame, s["data"].frame)
    path = save_project(s, tmp_path / "p.pymx")
    opened = load_project(path)
    pd.testing.assert_frame_equal(opened["data"].frame, s["data"].frame)
    assert len(opened["w"]) == 164


# -- Generate series → Within ---------------------------------------------------------------------------------------


def test_generate_series_within_is_the_call(quarterly, tmp_path):
    s = Session()
    s.objects["data"] = quarterly
    s.run(GenerateSeries("x", "x[-1]*1.02", replace=True, start="2000Q1", end="2001Q2"))
    expected = quarterly.with_series(x="x[-1]*1.02", within=Sample(start="2000Q1", end="2001Q2"))
    pd.testing.assert_frame_equal(s["data"].frame, expected.frame)
    assert s.history[-1].code == "data = data.with_series(within=Sample(('2000Q1', '2001Q2')), x='x[-1]*1.02')"
    s.run(NewSample(name="pre", where="x < 3"))
    s.run(GenerateSeries("low", "1", within="pre"))
    pd.testing.assert_frame_equal(s["data"].frame, expected.with_series(low="1", within=Sample(where="x < 3")).frame)
    assert s.history[-1].code == "data = data.with_series(within=pre, low='1')"
    each = GenerateSeries("{x}_hi", "{x}", for_each="x", start="2001Q4")
    s.run(each)
    assert s.history[-1].code == "data = data.with_series(within=Sample(start='2001Q4'), x_hi='x')"  # the expansion
    assert each.to_code() == "data = data.with_each('x', '{x}_hi', '{x}', within=Sample(start='2001Q4'))"  # without the data
    named = GenerateSeries("{x}_lo", "{x}", for_each="x", within="pre").to_code()
    assert named == "data = data.with_each('x', '{x}_lo', '{x}', within=pre)"
    assert s["data"]["x_hi"].notna().sum() == 1
    replayed = _replay(s, data=quarterly)
    pd.testing.assert_frame_equal(replayed.frame, s["data"].frame)
    path = save_project(s, tmp_path / "p.pymx")
    pd.testing.assert_frame_equal(load_project(path)["data"].frame, s["data"].frame)


def test_generate_series_checks_and_schema(quarterly):
    s = Session()
    s.objects["data"] = quarterly
    assert GenerateSeries("y", "1", within="pre", start="2000Q1").validate(s) == [
        "Within: choose a named sample or give periods, not both"]
    assert "Keep updated" in GenerateSeries("y", "1", keep_updated=True, start="2000Q1").validate(s)[0]
    assert GenerateSeries("y", "1", within="nope").validate(s) == ["no sample named 'nope'"]
    assert GenerateSeries("y", "1", start="2000Q1").validate(s) == []
    # schema 3: a step saved at schema 2 opens with the Within fields empty and gives the same code as before
    assert GenerateSeries.schema_version == 3
    old = {"series": "z", "formula": "x*2", "data": "data", "keep_updated": False, "replace": False, "for_each": ""}
    assert upgrade("pymetrix:generate_series", 2, dict(old)) == old
    cmd = from_json({"kind": "command", "id": "pymetrix:generate_series", "version": 2, "fields": old})
    assert (cmd.within, cmd.start, cmd.end) == ("", "", "") and cmd.to_code() == "data = data.with_series(z='x*2')"
    new = GenerateSeries("z", "x", start="2000Q1")
    assert from_json(json.loads(json.dumps(to_json(new)))) == new
