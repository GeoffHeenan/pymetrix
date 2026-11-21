"""Derived series: recipes kept with the data and recomputed when their inputs change."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymetrix import HP, Dataset, Hamilton
from pymetrix.commands import DeriveSeries, FitEquation, Filter, Session, load_project, save_project
from pymetrix.data.derived import Filtered, Formula
from pymetrix.filters import trend_cycle
from pymetrix.metadata import Vintage

DATA = Path(__file__).parents[1] / "examples" / "sample_macro.parquet"


@pytest.fixture
def data():
    return Dataset.read(str(DATA))


def test_formulas_follow_their_inputs_in_order(data):
    d = data.derive(s="c / y", pct="s * 100", dly="dlog(y)")
    assert set(d.derived) == {"s", "pct", "dly"} and d.variables["pct"].derived_from == "s * 100"
    changed = d.with_series(y=d["y"] * 2)
    assert np.allclose(changed["s"], d["s"] / 2) and np.allclose(changed["pct"], changed["s"] * 100)
    assert np.allclose(changed["dly"], d["dly"], equal_nan=True)  # log differences ignore the scale
    untouched = d.with_series(g=d["g"] + 1)
    assert untouched["s"] is not None and np.allclose(untouched["s"], d["s"])


def test_filters_are_recomputed_and_revise_history(data):
    d = data.filter("y", HP())
    assert isinstance(d.variables["y_trend"].recipe, Filtered) and d.variables["y_cycle"].derived_from == "HP(lambda=1600) cycle of y"
    revised_y = d["y"].copy()
    revised_y.iloc[-1] *= 1.05  # a revision to the last observation...
    revised = d.with_series(y=revised_y)
    expected, _ = trend_cycle(revised, "y", HP())
    assert np.allclose(revised["y_trend"], expected)
    assert not np.isclose(revised["y_trend"].iloc[-10], d["y_trend"].iloc[-10])  # ...revises the two-sided trend's history
    static = data.filter("y", HP(), derived=False)
    assert not static.derived and np.allclose(static.with_series(y=revised_y)["y_trend"], d["y_trend"])


def test_panel_filters_run_within_entity():
    rng = np.random.default_rng(0)
    frames = [pd.DataFrame({"id": k, "t": pd.period_range("2000Q1", periods=40, freq="Q"), "y": np.cumsum(rng.normal(size=40))}) for k in "ab"]
    panel = Dataset.from_frame(pd.concat(frames), time="t", entity="id").filter("y", Hamilton())
    bumped = panel.with_series(y=panel["y"] + np.where(panel.frame.index.get_level_values("id") == "a", 1.0, 0.0))
    a = bumped.frame.xs("a", level="id")
    assert np.allclose(a["y_cycle"].dropna(), panel.frame.xs("a", level="id")["y_cycle"].dropna())  # a level shift leaves Hamilton's cycle unchanged
    assert np.allclose(bumped.frame.xs("b", level="id")["y_trend"], panel.frame.xs("b", level="id")["y_trend"], equal_nan=True)


def test_replacing_freezing_and_subsets(data):
    d = data.derive(s="c / y")
    plain = d.with_series(s=d["s"] * 0)
    assert not plain.derived and np.allclose(plain.with_series(y=d["y"] * 2)["s"], 0)
    assert not d.freeze("s").derived and np.allclose(d.freeze()["s"], d["s"])
    assert not d.select(span=("1990Q1", None)).derived  # a subset keeps values, not recipes
    assert np.allclose(d.recompute()["s"], d["s"])
    with pytest.raises(ValueError, match="circular"):
        data.derive(a="b + 1", b="a + 1")
    with pytest.raises(ValueError, match="no series nope"):
        data.derive(x="nope * 2")


def test_recipes_are_saved_with_the_data(data, tmp_path):
    d = data.derive(dly="dlog(y)").filter("c", HP(lamb=400))
    path = tmp_path / "derived.parquet"
    d.write(str(path))
    back = Dataset.read(str(path))
    assert back.derived == d.derived
    assert back.variables["c_trend"].recipe == Filtered("c", HP(lamb=400), "trend")
    assert np.allclose(back.with_series(c=back["c"] * 2)["c_trend"], d.with_series(c=d["c"] * 2)["c_trend"])


def test_vintage_of_a_derived_series_is_its_latest_input(data):
    from dataclasses import replace

    variables = dict(data.variables)
    variables["c"] = replace(variables["c"], vintage=Vintage(date(2026, 3, 1)))
    variables["y"] = replace(variables["y"], vintage=Vintage(date(2026, 6, 1)))
    dated = Dataset(data.frame, data.axes, variables)
    d = dated.derive(s="c / y", only_c="c * 2")
    assert d.variables["s"].vintage.released == date(2026, 6, 1) and d.variables["only_c"].vintage.released == date(2026, 3, 1)


def test_dialogs_sessions_and_projects(data, tmp_path):
    s = Session()
    s.objects["data"] = data
    s.run(DeriveSeries("dly", "dlog(y)"))
    s.run(Filter("c"))
    s.run(FitEquation("dly ~ dly[-1]", name="growth"))
    s.run(FitEquation("c_cycle ~ g", name="cycle"))
    assert "data = data.derive(dly='dlog(y)')" in s.script() and "data = data.filter('c', HP())" in s.script()
    s.run_code("data = data.with_series(y=data['y'] * 1.02 ** (data['c'] > data['c'].median()))")
    stale = s.stale()
    assert stale == {"growth": "dly in data has changed since growth was computed"}  # the cycle fit used c_cycle and g only
    s.recompute()
    assert s.stale() == {}
    reopened = load_project(save_project(s, tmp_path / "derived.pymx"))
    assert np.allclose(reopened["data"]["dly"], s["data"]["dly"], equal_nan=True) and reopened["data"].derived.keys() == s["data"].derived.keys()
