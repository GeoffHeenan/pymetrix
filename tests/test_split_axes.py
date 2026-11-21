"""Split axes: a value axis split to leave out a range of values (a spike that squashes the rest), or a time axis
split to leave out a run of periods; drawn in two parts with break marks, recorded, saved, and written by
matplotlib_code as the same split."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset
from pymetrix.errors import SpecificationError
from pymetrix.figures import period_break, value_break


@pytest.fixture
def data():
    idx = pd.period_range("1970Q1", periods=220, freq="Q", name="time")
    rng = np.random.default_rng(1)
    growth = 2 + rng.normal(0, 0.8, 220)
    growth[200] = 22.0  # a spike
    level = 6 + np.cumsum(rng.normal(0, 0.1, 220))
    return Dataset.from_frame(pd.DataFrame({"growth": growth, "unemp": level}, index=idx))


def _limits(ax, which):
    return tuple(round(float(v), 6) for v in getattr(ax, f"get_{which}lim")())


def test_the_break_is_parsed_and_checked():
    assert value_break("6:12") == (6.0, 12.0) and value_break((6, 12)) == (6.0, 12.0) and value_break(None) is None
    assert period_break("1986Q1:2004Q4") == ("1986Q1", "2004Q4") and period_break("1990:1999") == ("1990", "1999")
    assert period_break((3, 8)) == (3.0, 8.0)  # numbers for an undated axis
    for bad in ("12:6", "6", "a:b"):
        with pytest.raises(SpecificationError):
            value_break(bad)
    for bad in ("2004Q4:1986Q1", "1986Q1:5", "1986Q1"):
        with pytest.raises(SpecificationError):
            period_break(bad)


def test_a_split_value_axis_draws_two_parts_either_side_of_the_break(data):
    whole = data.plot(["growth"]).draw().axes[0]
    y0, y1 = whole.get_ylim()
    fig = data.plot(["growth"], title="Growth", ybreak="6:12")
    top, bottom = fig.draw().axes
    assert _limits(top, "y") == (12.0, round(y1, 6)) and _limits(bottom, "y") == (round(y0, 6), 6.0)
    assert top.get_position().y0 > bottom.get_position().y1  # one above the other
    ratio = (top.get_position().height / bottom.get_position().height)
    assert ratio == pytest.approx((y1 - 12) / (6 - y0), rel=0.1)  # each part as tall as its range of values
    assert not top.spines["bottom"].get_visible() and not any(t.get_visible() for t in top.get_xticklabels())
    marks = [l for ax in (top, bottom) for l in ax.get_lines() if l.get_gid() == "pymetrix-break"]
    assert len(marks) == 2
    assert [l.get_label() for l in top.get_lines() if l.get_gid() != "pymetrix-break"] == ["growth"]  # the series, in both
    assert [l.get_label() for l in bottom.get_lines() if l.get_gid() != "pymetrix-break"] == ["growth"]
    assert fig.data().equals(data.plot(["growth"]).data())  # the data are the same: only the drawing is split


def test_a_split_time_axis_leaves_out_the_periods(data):
    from pymetrix.figures import _x_number

    fig = data.plot(["unemp"], xbreak="1986Q1:2004Q4")
    left, right = fig.draw().axes
    assert left.get_xlim()[1] == pytest.approx(_x_number("1986Q1", "Q"))
    assert right.get_xlim()[0] == pytest.approx(_x_number("2004Q4", "Q", end=True))
    assert _limits(left, "y") == _limits(right, "y")  # one value scale
    assert not right.spines["left"].get_visible() and not any(t.get_visible() for t in right.get_yticklabels())
    assert left.get_position().x1 < right.get_position().x0  # side by side


def test_panels_the_break_does_not_fit_are_drawn_whole(data):
    fig = data.plot(["unemp", "growth"], layout="each", ybreak="6:12")  # unemployment never reaches 6 to 12
    axes = fig.draw().axes
    assert len(axes) == 3  # unemployment whole; growth in two parts
    annual = Dataset.from_frame(data.frame.groupby(data.frame.index.year).mean().set_axis(
        pd.period_range("1970", periods=55, freq="Y", name="time")))
    assert len(annual.plot(["unemp"], xbreak="1986:2004").draw().axes) == 2  # years on an annual axis
    assert len(data.plot(["unemp"], xbreak="1986:2004").draw().axes) == 2  # and on a quarterly one: whole years
    assert all(l.get_gid() != "pymetrix-break" for l in axes[0].get_lines())


def test_recorded_styles_reach_both_parts_and_the_legend_one(data):
    fig = data.plot(["growth"], ybreak="6:12").grid(axis="y", linestyle=":").axhline(2, color="red").legend(loc="lower left")
    top, bottom = fig.draw().axes
    assert top.get_legend() is not None and bottom.get_legend() is None
    for ax in (top, bottom):
        assert any(l.get_color() == "red" for l in ax.get_lines())  # the line at 2: drawn on both, shown where 2 is


def test_split_and_right_hand_axes(data):
    fig = data.plot(["unemp", "growth"], right=["growth"], xbreak="1986Q1:2004Q4")
    axes = fig.draw().axes
    assert len(axes) == 4  # two parts, each with its right-hand scale
    with pytest.raises(SpecificationError, match="right-hand axis"):
        data.plot(["growth", "unemp"], right=["unemp"], ybreak="6:12").draw()
    with pytest.raises(SpecificationError, match="not both"):
        data.plot(["growth"]).with_options(ybreak="6:12", xbreak="1986Q1:2004Q4")


def test_matplotlib_code_draws_the_same_split(data):
    import matplotlib

    matplotlib.use("Agg")
    for fig in (data.plot(["growth"], ybreak="6:12").grid(axis="y", linestyle=":").axhline(2, color="red"),
                data.plot(["unemp", "growth"], right=["growth"], xbreak="1986Q1:2004Q4").set(0, side="right", ylabel="growth")):
        namespace = {"__name__": "script"}
        exec(fig.matplotlib_code(), namespace)  # noqa: S102 - the generated script runs on its own

        def key(ax):
            return _limits(ax, "x"), _limits(ax, "y"), ax.get_ylabel(), len(ax.get_lines())

        assert sorted(map(key, fig.draw().axes)) == sorted(map(key, namespace["fig"].axes))


def test_the_split_is_recorded_saved_and_set_in_the_dialogs(data, tmp_path):
    from pymetrix import load_result, save_result
    from pymetrix.commands import FigureOptions, PlotSeries, Session

    s = Session()
    s.objects["data"] = data
    s.run(PlotSeries(series="growth", split_values="6:12", name="fig"))
    assert "fig = data.plot('growth', ybreak='6:12')" in s.script()
    assert s["fig"].ybreak == (6.0, 12.0)
    assert PlotSeries(series="growth", split_values="12:6").validate(s) != []
    assert PlotSeries(series="growth", split_values="6:12", split_dates="1990:1999").validate(s) == ["split the values or the dates, not both"]
    dialog = FigureOptions.for_figure("fig", s["fig"], is_figure=True)
    assert dialog.split_values == "6:12" and dialog.split_dates == ""
    s.run(FigureOptions("fig", split_dates="1986Q1:2004Q4", name="fig"))
    assert s["fig"].xbreak == ("1986Q1", "2004Q4") and s["fig"].ybreak is None
    path = save_result(s["fig"], tmp_path / "fig.pymxr")
    assert load_result(path).views[0][1].xbreak == ("1986Q1", "2004Q4")
