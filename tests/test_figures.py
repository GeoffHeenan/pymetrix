"""Figures: building, drawing, saving, the figures of results, the Plot dialog, and windows."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from pymetrix import HP, VAR, Dataset, Equation, LocalProjection, UnobservedComponents  # noqa: E402
from pymetrix.figures import Band, Bars, Ecdf, Fan, Figure, Histogram, Line, Panel, Rule, Scatter, correlogram_figure, filter_figure  # noqa: E402
from pymetrix.errors import SpecificationError  # noqa: E402

DATA = Path(__file__).parents[1] / "examples" / "sample_macro.parquet"


@pytest.fixture(scope="module")
def data():
    return Dataset.read(str(DATA)).with_series(ly="100*log(y)", lc="100*log(c)")


def _layers(fig, kind):
    return [layer for p in fig.panels for layer in p.layers if isinstance(layer, kind)]


def test_figures_draw_save_and_give_their_data(data, tmp_path):
    fig = data.plot(["c", "y"])
    assert len(fig.panels) == 1 and [l.label for l in _layers(fig, Line)] == ["c", "y"]
    assert list(fig.data().columns) == ["c", "y"] and len(fig.data()) == len(data)
    for suffix in ("png", "svg", "pdf"):
        fig.save(str(tmp_path / f"f.{suffix}"))
        assert (tmp_path / f"f.{suffix}").stat().st_size > 1000
    assert fig._repr_png_()[:8] == b"\x89PNG\r\n\x1a\n"
    drawn = fig.draw()
    drawn.canvas.draw()
    labels = [t.get_text() for t in drawn.axes[0].get_xticklabels() if t.get_text()]
    assert labels and all("Q" in t for t in labels)  # quarterly periods, labelled as such
    with pytest.raises(ValueError, match="at least one panel"):
        Figure(())


def test_series_figures_of_every_kind(data):
    assert len(data.plot(["c", "i", "g", "y"], layout="each").panels) == 4
    assert data.plot(["c", "i", "g", "y"], layout="each").columns == 2
    assert _layers(data.plot("i", kind="bar", span=("2020Q1", None)), Bars)[0].y.index[0] == pd.Period("2020Q1", "Q")
    assert _layers(data.plot("i", kind="area"), Band)
    scatter = data.plot(["y", "c"], kind="scatter")
    assert _layers(scatter, Scatter)[0].x.name == "y" and _layers(scatter, Scatter)[0].fit
    assert _layers(data.plot("i", kind="histogram"), Histogram)
    with pytest.raises(ValueError, match="x first"):
        data.plot("y", kind="scatter")


def test_panel_figures():
    rows = [{"id": k, "t": p, "y": float(i + 10 * j), "x": float(i)} for j, k in enumerate("abc") for i, p in enumerate(pd.period_range("2000Q1", periods=8, freq="Q"))]
    panel = Dataset.from_frame(pd.DataFrame(rows), time="t", entity="id")
    assert [l.label for l in _layers(panel.plot("y"), Line)] == ["y (a)", "y (b)", "y (c)"]
    assert [l.label for l in _layers(panel.plot(["y", "x"], entity="b"), Line)] == ["y", "x"]
    with pytest.raises(ValueError, match="choose an entity"):
        panel.plot(["y", "x"])


def test_result_figures(data):
    fit = Equation("c ~ y + c[-1]").fit(data)
    fig = fit.plot()
    assert [p.title for p in fig.panels] == ["Actual and fitted", "Residuals"]
    assert np.allclose(_layers(fig, Line)[0].y, fit.fitted + fit.resid)
    se = fit.stats["se_regression"]
    assert np.allclose(_layers(fig, Band)[0].upper, 2 * se)
    fc = Equation("c ~ y").fit(data, span=(None, "2022Q4")).forecast(data, ("2023Q1", "2024Q4"))
    assert {l.label for l in _layers(fc.plot(), Line)} == {"actual", "forecast"} and _layers(fc.plot(), Band)
    var = VAR(["lc", "ly"], lags=2).fit(data)
    vfc = var.forecast(data, ("2025Q1", "2026Q4"))
    from pymetrix.figures import forecast_figure

    with_history = forecast_figure(vfc, data, history=8)
    assert len(with_history.panels) == 2 and len(_layers(with_history, Line)[0].y) == 8  # the last 8 actual periods first
    irf = var.irf(8)
    grid = irf.plot()
    assert len(grid.panels) == 4 and grid.columns == 2 and grid.panels[1].title == "lc, shock to ly"
    assert len(_layers(grid, Band)) == 4 and grid.panels[-1].xlabel == "horizon"
    assert len(irf.plot(response="ly").panels) == 2
    lp = LocalProjection("lc", "ly", horizons=4, lags=1).fit(data).plot()
    assert [b.label for b in _layers(lp, Band)] == ["90%", "68%"]
    uc = UnobservedComponents("ly", trend="local linear trend", cycle=0).fit(data)
    assert [p.title for p in uc.plot().panels] == ["trend (smoothed)", "slope (smoothed)"]
    assert uc.plot("filtered").panels[0].title == "trend (filtered)"
    filtered = data.filter("ly", HP())
    assert [p.title for p in filter_figure(filtered, "ly").panels] == ["ly and its trend", "Cycle"]
    corr = correlogram_figure(data.correlogram("ly", 8))
    rules = [layer.y for layer in _layers(corr, Rule)]
    assert len(corr.panels) == 2 and any(r > 0 for r in rules) and any(r < 0 for r in rules)
    rolling = Equation("c ~ y").fit_rolling(data, 40)
    assert [p.title for p in rolling.plot().panels] == ["const", "y"]


def test_plot_dialog_records_and_replays(data, tmp_path):
    from pymetrix.commands import PlotSeries, Session, load_project, save_project

    s = Session()
    s.objects["data"] = data
    s.run(PlotSeries("c, y", kind="bar", layout="each", start="2020Q1", title="Spending"))
    assert "figure = data.plot(['c', 'y'], kind='bar', layout='each', span=('2020Q1', None), title='Spending')" in s.script()
    assert isinstance(s["figure"], Figure) and s["figure"].title == "Spending"
    assert PlotSeries("c", kind="scatter").validate(s) == ["a scatter plot needs two series: the x series first, then y"]
    reopened = load_project(save_project(s, tmp_path / "fig.pymx"))
    assert len(reopened["figure"].panels) == 2


def test_copy_data_ends_lines_with_newline_on_every_platform(data, monkeypatch):
    """The figure window's Copy data ends lines with \\n even where pandas would use the platform's line ending
    (\\r\\n on Windows, simulated here), as a matrix's and a table's copies do."""
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import figure_view

    figure = data.plot(["c", "y"])
    monkeypatch.setattr(os, "linesep", "\r\n")
    page = figure_view(figure)
    next(b for b in page.findChildren(QtWidgets.QPushButton) if b.text() == "Copy data").click()
    copied = app.clipboard().text()
    assert "\r" not in copied and copied == figure.data().to_csv(sep="\t", lineterminator="\n")


def test_figures_in_windows_and_notebooks(data):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import object_view
    from pymetrix.ui.render import to_html

    fit = Equation("c ~ y").fit(data)
    tabs = object_view(fit)
    assert tabs.tabText(1) == "Graph" and "Graph" in tabs.figures
    page = tabs.widget(1)
    assert page.canvas.figure.axes
    copy = next(b for b in page.findChildren(QtWidgets.QPushButton) if b.text() == "Copy image")
    copy.click()
    assert not app.clipboard().image().isNull()
    shown = object_view(data.plot(["c", "y"]))
    assert shown.tabText(0) == "Figure"
    assert to_html(data.plot("c")).startswith('<img alt="&lt;Figure: 1 panel&gt;" src="data:image/png;base64,')


# -- more kinds of series plot: bars side by side and stacked, across, steps, areas, boxes, a right axis, transforms --


def _drawn(fig):
    drawn = fig.draw()
    drawn.canvas.draw()
    return drawn


def test_bars_stand_side_by_side(data):
    fig = data.plot(["c", "i", "g"], kind="bar", span=("2023Q1", None))
    ax = _drawn(fig).axes[0]
    bars = ax.patches
    assert len(bars) == 3 * 8  # three series, eight quarters, none hidden behind another
    for n, name in enumerate(["c", "i", "g"]):
        mine = bars[8 * n: 8 * (n + 1)]
        expected = data.frame[name].loc["2023Q1":].to_numpy(float)
        assert np.allclose([b.get_height() for b in mine], expected)
    first = [bars[8 * n].get_x() + bars[8 * n].get_width() / 2 for n in range(3)]  # the centres in 2023Q1
    assert first[0] < first[1] < first[2] and np.isclose(first[1] - first[0], first[2] - first[1])
    assert all(bars[8 * n].get_x() + bars[8 * n].get_width() <= bars[8 * (n + 1)].get_x() + 1e-6 for n in range(2))  # no overlap


def test_stacked_bars_put_negatives_below_zero(data):
    d = data.with_series(gc="400*dlog(c)", gi="400*dlog(i)")
    fig = d.plot(["gc", "gi"], kind="bar", stacked=True, span=("2021Q1", "2022Q4"))
    assert _layers(fig, Rule)  # contributions that change sign get a zero line
    bars = _drawn(fig).axes[0].patches
    gc, gi = (d.frame[n].loc["2021Q1":"2022Q4"].to_numpy(float) for n in ("gc", "gi"))
    for t in range(8):
        base = bars[8 + t].get_y()  # gi's bar starts where gc's ended, on its own side of zero
        expected = (gc[t] if gc[t] >= 0 else 0.0) if gi[t] >= 0 else (gc[t] if gc[t] < 0 else 0.0)
        assert np.isclose(base, expected) and np.isclose(bars[8 + t].get_height(), gi[t])


def test_horizontal_bars_label_their_periods(data):
    d = data.with_series(gc="400*dlog(c)", gi="400*dlog(i)")
    ax = _drawn(d.plot(["gc", "gi"], kind="barh", stacked=True, span=("2024Q1", None))).axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]
    widths = [p.get_width() for p in ax.patches]
    assert np.allclose(widths[:4], d.frame["gc"].loc["2024Q1":].to_numpy(float))
    assert ax.yaxis_inverted()  # the first period at the top


def test_a_cross_section_labels_its_bars():
    frame = pd.DataFrame({"country": ["Fiji", "Samoa", "Tonga"], "debt": [80.0, 45.0, 40.0], "deficit": [-3.0, -1.0, 2.0]})
    xs = Dataset.from_frame(frame, entity="country")
    ax = _drawn(xs.plot(["debt", "deficit"], kind="bar")).axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Fiji", "Samoa", "Tonga"]
    assert len(ax.patches) == 6


def test_steps_areas_and_boxes(data):
    step = _drawn(data.plot("g", kind="step")).axes[0].lines[0]
    assert step.get_drawstyle() == "steps-post"
    area = data.plot(["c", "i", "g"], kind="area", stacked=True)
    top = _layers(area, Band)[-1].upper
    assert np.allclose(top.to_numpy(float), data.frame[["c", "i", "g"]].sum(axis=1).to_numpy(float))
    d = data.with_series(gc="400*dlog(c)", gi="400*dlog(i)")
    box = _drawn(d.plot(["gc", "gi"], kind="box")).axes[0]
    medians = [line.get_ydata()[0] for line in box.lines if line.get_color() == "#222222"]
    assert np.allclose(medians, [d.frame["gc"].median(), d.frame["gi"].median()])
    assert [t.get_text() for t in box.get_xticklabels()] == ["gc", "gi"]
    assert list(d.plot(["gc", "gi"], kind="box").data().columns) == ["gc", "gi"]


def test_a_right_hand_axis(data):
    fig = data.plot(["y", "i"], right="i")
    assert [l.axis for l in _layers(fig, Line)] == ["left", "right"] and fig.panels[0].ylabel_right == "i"
    drawn = _drawn(fig)
    assert len(drawn.axes) == 2  # the twin
    legend = [t.get_text() for t in drawn.axes[0].get_legend().get_texts()]
    assert legend == ["y", "i (right)"]
    with pytest.raises(ValueError, match="some, not all"):
        data.plot(["y", "i"], right=["y", "i"])
    with pytest.raises(ValueError, match="right-hand axis"):
        data.plot(["y", "i"], kind="bar", stacked=True, right="i")


def test_transforms_by_hand(data):
    c = data.frame["c"]
    yoy = _layers(data.plot("c", transform="yoy", span=("2010Q1", None)), Line)[0].y
    assert np.isclose(yoy.loc["2010Q1"], 100 * (c.loc["2010Q1"] / c.loc["2009Q1"] - 1))  # uses the year before the span
    growth = _layers(data.plot("c", transform="growth"), Line)[0].y
    assert np.isclose(growth.iloc[5], 100 * (c.iloc[5] / c.iloc[4] - 1)) and np.isnan(growth.iloc[0])
    index = _layers(data.plot(["c", "i"], transform="index", span=("2015Q1", None)), Line)
    assert all(np.isclose(l.y.iloc[0], 100) for l in index)
    assert np.isclose(index[1].y.iloc[-1], 100 * data.frame["i"].iloc[-1] / data.frame["i"].loc["2015Q1"])
    change = _layers(data.plot("c", transform="change"), Line)[0].y
    assert np.isclose(change.iloc[3], c.iloc[3] - c.iloc[2])
    assert np.allclose(_layers(data.plot("c", transform="log"), Line)[0].y, np.log(c))
    assert data.plot("c", transform="yoy").panels[0].ylabel == "% change on a year earlier"
    with pytest.raises(ValueError, match="transform must be"):
        data.plot("c", transform="pct")


def test_kinds_that_cannot_stack_or_do_not_exist(data):
    with pytest.raises(ValueError, match="only bars and areas stack"):
        data.plot(["c", "i"], stacked=True)
    with pytest.raises(ValueError, match="one panel"):
        data.plot(["c", "i"], kind="bar", stacked=True, layout="each")
    with pytest.raises(ValueError, match="kind must be"):
        data.plot("c", kind="pie")


def test_the_plot_dialog_takes_the_new_settings(data, tmp_path):
    from pymetrix.commands import PlotSeries, Session, form_spec, load_project, save_project

    s = Session()
    s.objects["data"] = data
    s.run(PlotSeries("c, i", kind="bar", stacked=True, start="2020Q1", transform="growth"))
    s.run(PlotSeries("y, i", right="i", name="twin"))
    script = s.script()
    assert "figure = data.plot(['c', 'i'], kind='bar', span=('2020Q1', None), stacked=True, transform='growth')" in script
    assert "twin = data.plot(['y', 'i'], right=['i'])" in script
    ns = {"data": data}
    exec(script, ns)  # the dialog steps, replayed as plain Python
    assert ns["twin"].panels[0].ylabel_right == "i"
    assert PlotSeries("c, i", kind="line", stacked=True).validate(s) == ["only bars and areas stack"]
    assert PlotSeries("c", right="i").validate(s) == ["the right-axis series must be among the series plotted"]
    stacked = next(f for f in form_spec(PlotSeries) if f.name == "stacked")
    assert stacked.visible({"kind": "bar"}) and not stacked.visible({"kind": "line"})
    kind = next(f for f in form_spec(PlotSeries) if f.name == "kind")
    assert kind.display("barh") == "Horizontal bar"
    assert len(load_project(save_project(s, tmp_path / "plots.pymx"))["twin"].panels) == 1


def test_shading_periods(data):
    from matplotlib import dates

    from pymetrix.figures import shade_spans

    assert shade_spans("2008Q3:2009Q2=GFC, 2020Q1") == (("2008Q3", "2009Q2", "GFC"), ("2020Q1", "2020Q1", ""))
    assert shade_spans([("2008Q3", "2009Q2")]) == (("2008Q3", "2009Q2", ""),)
    fig = data.plot(["c", "i"], shade="2008Q3:2009Q2=GFC, 2020Q1:2020Q2=COVID")
    ax = _drawn(fig).axes[0]
    spans = [p for p in ax.patches if p.get_alpha() == 0.18]
    def extent(p):  # a Rectangle (matplotlib 3.9 and later) or a Polygon (earlier)
        xs = [p.get_x(), p.get_x() + p.get_width()] if hasattr(p, "get_width") else list(np.asarray(p.get_xy())[:, 0])
        return [dates.num2date(x).date().isoformat() for x in (min(xs), max(xs))]

    starts, ends = (sorted(extent(p)[k] for p in spans) for k in (0, 1))
    assert starts == ["2008-07-01", "2020-01-01"] and ends == ["2009-07-01", "2020-07-01"]  # whole periods, start to end
    assert {"GFC", "COVID"} <= {t.get_text() for t in ax.get_legend().get_texts()}
    with pytest.raises(ValueError, match="start and an end"):
        shade_spans([("2008Q3", "")])
    with pytest.raises(ValueError, match="cannot shade"):
        data.plot("c", shade="2008-13:2009-02").draw()


def test_shading_from_a_dummy_series_and_on_any_figure(data):
    d = data.with_series(gy="400*dlog(y)")
    d = d.with_series(recession=(d["gy"] < 0).astype(float))
    fig = d.plot("gy", shade="recession")
    flags = d["recession"].to_numpy(float) != 0
    runs = sum(1 for t in range(len(flags)) if flags[t] and (t == 0 or not flags[t - 1]))  # each run of ones, counted by hand
    assert len(fig.shade) == runs and all(label == "recession" for _, _, label in fig.shade)
    for start, end, _ in fig.shade:
        assert d["recession"].loc[start:end].eq(1).all()
    fit = Equation("c ~ y").fit(data)
    shaded = fit.plot().with_options(shade="2008Q3:2009Q2")
    drawn = _drawn(shaded)
    assert all(any(p.get_alpha() == 0.18 for p in ax.patches) for ax in drawn.axes)  # every dated panel
    hist = _drawn(data.plot("c", kind="histogram", shade="2008Q3:2009Q2")).axes[0]
    assert not any(p.get_alpha() == 0.18 for p in hist.patches)  # nothing dated to shade


def test_the_plot_dialog_shades(data):
    from pymetrix.commands import PlotSeries, Session

    s = Session()
    s.objects["data"] = data
    s.run(PlotSeries("c", shade="2008Q3:2009Q2=GFC"))
    assert "figure = data.plot('c', shade='2008Q3:2009Q2=GFC')" in s.script()
    assert s["figure"].shade == (("2008Q3", "2009Q2", "GFC"),)


def test_event_and_reference_lines(data):
    from matplotlib import dates

    from pymetrix.figures import event_marks, reference_lines

    assert event_marks("2016Q2=Brexit vote, 2020Q1") == (("2016Q2", "Brexit vote"), ("2020Q1", ""))
    assert reference_lines("2=inflation target, 60") == ((2.0, "inflation target"), (60.0, ""))
    assert reference_lines(2) == ((2.0, ""),)
    ax = _drawn(data.plot(["c", "i"], transform="yoy", events="2016Q2=Brexit vote, 2020Q1", reference="2=target")).axes[0]
    def day(x):  # a date, or matplotlib's number for one
        return (pd.Timestamp(x) if not isinstance(x, (int, float, np.floating)) else dates.num2date(x)).date().isoformat()

    vertical = sorted(day(l.get_xdata()[0]) for l in ax.lines if l.get_linestyle() == "--")
    assert vertical == ["2016-04-01", "2020-01-01"]  # at the start of each event's period
    assert any(l.get_linestyle() == ":" and l.get_ydata()[0] == 2.0 for l in ax.lines)
    assert {"Brexit vote ", "target "} <= {t.get_text() for t in ax.texts}
    across = _drawn(data.plot(["c", "i"], kind="barh", span=("2024Q1", None), reference="100")).axes[0]
    assert any(l.get_linestyle() == ":" and l.get_xdata()[0] == 100 for l in across.lines)  # across horizontal bars: upright
    with pytest.raises(ValueError, match="a number"):
        reference_lines("two=target")
    with pytest.raises(ValueError, match="not a period"):
        data.plot("c", events="2016-13=bad").draw()
    fit = Equation("c ~ y").fit(data)
    drawn = _drawn(fit.plot().with_options(events="2020Q1=COVID"))
    assert all(any(l.get_linestyle() == "--" for l in ax.lines) for ax in drawn.axes)  # any figure, every dated panel


def test_scales_limits_and_one_scale_for_all_panels(data):
    from pymetrix.figures import value_limits

    log = _drawn(data.plot(["c", "i"], log=True)).axes[0]
    log.figure.canvas.draw()
    assert log.get_yscale() == "log"
    labels = [t.get_text() for t in log.get_yticklabels() if t.get_text()]
    assert labels and all("10^" not in t and "\times" not in t for t in labels)  # plain numbers
    assert np.allclose(_drawn(data.plot("c", ylim=(0, 500))).axes[0].get_ylim(), (0, 500))
    low_only = _drawn(data.plot("c", ylim=(0, None))).axes[0].get_ylim()
    assert low_only[0] == 0 and low_only[1] > data.frame["c"].max()  # the other end stays automatic
    assert value_limits("0:10") == (0.0, 10.0) and value_limits(":10") == (None, 10.0) and value_limits(None) is None
    with pytest.raises(ValueError, match="below the highest"):
        value_limits((10, 0))
    shared = _drawn(data.plot(["c", "i"], layout="each", sharey=True)).axes
    assert shared[0].get_ylim() == shared[1].get_ylim()
    apart = _drawn(data.plot(["c", "i"], layout="each")).axes
    assert apart[0].get_ylim() != apart[1].get_ylim()


def test_values_written_on_bars(data):
    ax = _drawn(data.plot(["c", "i"], kind="bar", span=("2024Q1", None), bar_labels=True)).axes[0]
    written = [t.get_text() for t in ax.texts]
    expected = [f"{v:,.0f}" for n in ("c", "i") for v in data.frame[n].loc["2024Q1":]]  # one format: the largest is over 100
    assert written == expected
    d = data.with_series(gc="400*dlog(c)", gi="400*dlog(i)")
    stacked = _drawn(d.plot(["gc", "gi"], kind="bar", stacked=True, span=("2024Q1", None), bar_labels=True)).axes[0]
    assert [t.get_text() for t in stacked.texts] == [f"{v:,.1f}" for n in ("gc", "gi") for v in d.frame[n].loc["2024Q1":]]
    assert all(t.get_color() == "white" for t in stacked.texts)


def test_the_plot_dialog_takes_marks_and_scales(data):
    from pymetrix.commands import PlotSeries, Session

    s = Session()
    s.objects["data"] = data
    s.run(PlotSeries("c, i", kind="bar", events="2020Q1=COVID", reference="300=line", bar_labels=True, log_scale=True,
                     lowest="10", start="2019Q1"))
    s.run(PlotSeries("c, i", layout="each", same_scale=True, name="pair"))
    script = s.script()
    assert ("figure = data.plot(['c', 'i'], kind='bar', span=('2019Q1', None), events='2020Q1=COVID', reference='300=line', "
            "bar_labels=True, log=True, ylim=(10.0, None))") in script
    assert "pair = data.plot(['c', 'i'], layout='each', sharey=True)" in script
    ns = {"data": data}
    exec(script, ns)
    assert ns["figure"].events == (("2020Q1", "COVID"),) and ns["figure"].ylim == (10.0, None) and ns["pair"].sharey
    assert PlotSeries("c", reference="two").validate(s) == ["a reference line is a number, optionally =label; got 'two'"]
    assert PlotSeries("c", lowest="5", highest="1").validate(s) == ["the lowest value shown (5) must be below the highest (1)"]


# -- editing a figure: Figure options, and the Edit… button --


def test_figure_options_edit_a_result_figure_and_replay(data):
    from pymetrix import figure_of
    from pymetrix.commands import FigureOptions, Session

    s = Session()
    s.objects["data"] = data.with_series(recession=(data["c"].diff() < 0).astype(float))
    fit = Equation("c ~ y").fit(data)
    s.objects["fit"] = fit
    form = FigureOptions.for_figure("fit", figure_of(fit), "Graph")
    assert form.name == "fit_graph" and form.title == figure_of(fit).title  # filled in with how it is drawn now
    s.run(FigureOptions("fit", title="Consumption", footnote="Source: sample data", shade="recession", events="2020Q1=COVID",
                        reference="0", lowest="", highest="", name="fit_graph"))
    edited = s["fit_graph"]
    assert edited.title == "Consumption" and edited.events == (("2020Q1", "COVID"),) and edited.reference == ((0.0, ""),)
    assert edited.shade and all(label == "recession" for _, _, label in edited.shade)
    assert s["fit"] is fit and figure_of(fit).title != "Consumption"  # the result is untouched
    code = s.history[-1].code
    assert code.startswith("fit_graph = figure_of(fit).with_options(title='Consumption'") and "shade=data['recession']" in code
    ns = {"data": s["data"], "fit": fit}
    exec("from pymetrix import *\n" + code, ns)
    assert ns["fit_graph"].shade == edited.shade and ns["fit_graph"].footnote == "Source: sample data"


def test_figure_options_round_trip_and_in_place(data):
    from pymetrix.commands import FigureOptions, PlotSeries, Session

    s = Session()
    s.objects["data"] = data
    s.run(PlotSeries("c, i", shade="2008Q3:2009Q2=GFC, 2020Q1", events="2016Q2=vote", reference="300=line", log_scale=True,
                     lowest="100", title="Spending", subtitle="Quarterly", footnote="Source: sample"))
    before = s["figure"]
    assert before.subtitle == "Quarterly" and before.footnote == "Source: sample"
    unchanged = FigureOptions.for_figure("figure", before, is_figure=True)
    assert unchanged.name == "figure" and unchanged.shade == "2008Q3:2009Q2=GFC, 2020Q1" and unchanged.lowest == "100"
    s.run(unchanged)  # opened and accepted as it is: the same figure
    after = s["figure"]
    for option in ("title", "subtitle", "footnote", "shade", "events", "reference", "log", "ylim"):
        assert getattr(after, option) == getattr(before, option), option
    assert s.history[-1].code.startswith("figure = figure_of(figure).with_options(")


def test_figure_options_say_what_is_wrong(data):
    from pymetrix.commands import FigureOptions, Session

    s = Session()
    s.objects["data"] = data
    s.objects["fit"] = Equation("c ~ y").fit(data)
    assert FigureOptions("nothing").validate(s) == ["no object named 'nothing'"]
    assert FigureOptions("fit", shade="recession").validate(s) == [
        "no series 'recession' in data to shade from (or write nber or cepr for recession dates)"]
    assert "no figure on a tab 'Coefficients'" in FigureOptions("fit", tab="Coefficients").validate(s)[0]
    assert FigureOptions("fit", reference="high").validate(s) == ["a reference line is a number, optionally =label; got 'high'"]
    assert FigureOptions("fit", lowest="5", highest="1").validate(s) == ["the lowest value shown (5) must be below the highest (1)"]


def test_figures_have_an_edit_button(data, monkeypatch):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("qtconsole")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.commands import FitEquation, PlotSeries, Session
    from pymetrix.ui import qt
    from pymetrix.ui.qt import main_window, object_view

    fit = Equation("c ~ y").fit(data)
    calls = []
    tabs = object_view(fit, edit_figure=lambda tab, figure, kind: calls.append((tab, figure, kind)))
    page = tabs.widget(1)
    assert page.edit.text() == "Edit…" and set(page.edit_actions) == {"options", "style", "properties", "note", "method", "styles", "copy", "house", "arrow", "textbox", "change"}
    page.edit_actions["options"].trigger()
    page.edit_actions["style"].trigger()
    assert [(c[0], c[2]) for c in calls] == [("Graph", "options"), ("Graph", "style")] and calls[0][1] is tabs.figures["Graph"]
    assert not any(b.text() == "Edit…" for b in object_view(fit).widget(1).findChildren(QtWidgets.QPushButton))

    opened = []

    class Accept:  # the dialog, accepted as filled in
        def __init__(self, cls, session, parent=None, initial=None):
            opened.append((cls, initial))
            self.session, self.initial = session, initial

        def exec(self):
            from dataclasses import replace

            self.command = replace(self.initial, title="Edited")
            self.result_value = self.session.run(self.command)
            return True

    monkeypatch.setattr(qt, "command_dialog", Accept)
    session = Session()
    session.objects["data"] = data
    session.run(FitEquation("c ~ y", name="fit"))
    session.run(PlotSeries("c", name="figure"))
    window = main_window(session)
    try:
        from pymetrix import figure_of

        window.edit_figure("fit", "Graph", figure_of(session["fit"]))
        assert opened[-1][0].__name__ == "FigureOptions" and session["fit_graph"].title == "Edited"
        assert any(sub.windowTitle().startswith("fit_graph") for sub in window.mdi.subWindowList())  # the edited figure is shown
        window.edit_figure("figure", "Figure", session["figure"])
        assert opened[-1][0] is PlotSeries and opened[-1][1] is session.history[1].command  # Plot series reopens its dialog
        assert session["figure"].title == "Edited"
    finally:
        session.dirty = False
        window.close()


def test_figure_windows_zoom_and_pan(data):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import figure_view

    page = figure_view(data.plot(["c", "y"]))
    toolbar = page.toolbar
    names = [a.text() for a in toolbar.actions() if a.text()]
    assert names[:5] == ["Home", "Back", "Forward", "Pan", "Zoom"] and "Save" not in names and "Customize" not in names
    ax = page.canvas.figure.axes[0]
    drawn = ax.get_xlim()
    toolbar.push_current()
    ax.set_xlim(drawn[0], (drawn[0] + drawn[1]) / 2)  # as a zoom leaves it
    toolbar.push_current()
    toolbar.home()  # Home: back to the figure as drawn
    assert np.allclose(ax.get_xlim(), drawn)
    app.processEvents()


# -- recession dates, and labels at the ends of lines --


def test_recession_dates_match_the_published_indicators():
    from pymetrix import recession_series, recessions
    from pymetrix.recessions import turning_points

    quarterly = {s: e for s, e, _ in recessions("nber", "Q")}
    # FRED's USRECQ (1 from the quarter after the NBER peak through the trough), written out by hand
    assert quarterly["2008Q1"] == "2009Q2" and quarterly["2020Q1"] == "2020Q2" and quarterly["1980Q2"] == "1980Q3"
    assert quarterly["1981Q4"] == "1982Q4" and quarterly["2001Q2"] == "2001Q4" and len(quarterly) == 12
    monthly = {s: e for s, e, _ in recessions("nber", "M")}  # FRED's USREC
    assert monthly["2008-01"] == "2009-06" and monthly["2020-03"] == "2020-04" and monthly["1980-02"] == "1980-07"
    cepr = {s: e for s, e, _ in recessions("cepr", "Q")}
    assert cepr["2008Q2"] == "2009Q2" and cepr["2011Q4"] == "2013Q1" and cepr["2020Q1"] == "2020Q2"
    assert {s: e for s, e, _ in recessions("cepr", "M")}["2008-04"] == "2009-06"  # the same months, from quarters
    assert {s: e for s, e, _ in recessions("nber", "Y")}["2008"] == "2009"
    assert list(turning_points("nber", "Q").iloc[-1]) == ["2019Q4", "2020Q2"]
    q = recession_series(pd.period_range("2007Q1", "2010Q4", freq="Q"))
    assert q.sum() == 6 and q.name == "nber_recession" and q.loc["2008Q1"] == 1 and q.loc["2009Q3"] == 0
    m = recession_series(pd.Series(0.0, index=pd.period_range("2008-01", "2008-12", freq="M")))
    assert m.sum() == 12
    with pytest.raises(ValueError, match="no business-cycle dates"):
        recessions("imf")
    with pytest.raises(ValueError, match="dated"):
        recession_series(pd.Series([1.0, 2.0]))


def test_shading_recessions_keeps_to_the_data(data):
    from matplotlib import dates

    from pymetrix.figures import shade_spans, shade_text

    plain = _drawn(data.plot("c")).axes[0].get_xlim()
    ax = _drawn(data.plot("c", shade="nber")).axes[0]
    assert np.allclose(ax.get_xlim(), plain)  # recessions before 2000 do not widen the axis
    shaded = [p for p in ax.patches if p.get_alpha() == 0.18]
    def left(p):  # a Rectangle (matplotlib 3.9 and later) or a Polygon (3.8)
        return p.get_x() if hasattr(p, "get_x") else float(np.asarray(p.get_xy())[:, 0].min())

    starts = sorted(dates.num2date(left(p)).date().isoformat() for p in shaded)
    assert starts == ["2001-04-01", "2008-01-01", "2020-01-01"]  # the sample's three NBER recessions, quarterly
    assert "NBER recessions" in [t.get_text() for t in ax.get_legend().get_texts()]
    assert shade_text(shade_spans("nber, 2001Q1=x")) == "nber, 2001Q1=x"
    named = data.with_series(nber=data["c"] * 0)  # a series of that name comes first
    assert named.plot("c", shade="nber").shade == ()


def test_lines_named_at_their_ends(data):
    fig = data.plot(["c", "i", "g", "y"], line_labels=True)
    ax = _drawn(fig).axes[0]
    notes = {t.get_text(): t for t in ax.texts}
    assert set(notes) == {"c", "i", "g", "y"} and ax.get_legend() is None  # names at the lines instead of a legend
    lines = [l for l in ax.lines if len(l.get_xdata()) == len(data)]
    for line, name in zip(lines, ["c", "i", "g", "y"]):
        assert notes[name].get_color() == line.get_color()
    heights = sorted(t.xy[1] for t in notes.values())
    gap = 0.055 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    assert all(b - a >= gap - 1e-9 for a, b in zip(heights, heights[1:]))  # i and g end close together: moved apart
    labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert labels and max(labels) <= "2024Q4"  # no dates in the room made for the names


def test_the_dialogs_take_recessions_and_line_names(data):
    from pymetrix.commands import FigureOptions, PlotSeries, Session

    s = Session()
    s.objects["data"] = data
    s.run(PlotSeries("c, i", shade="nber", line_labels=True))
    assert "figure = data.plot(['c', 'i'], shade='nber', line_labels=True)" in s.script()
    s.objects["fit"] = Equation("c ~ y").fit(data)
    options = FigureOptions("fit", shade="cepr", line_labels=True, name="fit_graph")
    assert options.validate(s) == []
    s.run(options)
    assert "shade='cepr'" in s.history[-1].code and "line_labels=True" in s.history[-1].code
    assert s["fit_graph"].line_labels and s["fit_graph"].shade[0][0] == "@cepr"
    assert FigureOptions.for_figure("fit_graph", s["fit_graph"], is_figure=True).shade == "cepr"


# -- styling in matplotlib's terms (docs/notes/figure-styling.md, phase A) --


def test_style_calls_are_recorded_in_matplotlibs_words(data):
    from pymetrix.figures import StyleCall

    base = data.plot(["c", "i"])
    fig = base.line("c", lw=2.5, ls="--", c="C3").set(yscale="log", ylabel="levels").legend(loc="upper left", ncols=2)
    assert fig.style[0] == StyleCall("line", (None, "c"), {"linewidth": 2.5, "linestyle": "--", "color": "C3"})  # aliases resolved
    assert fig.style[1] == StyleCall("set", (0, "left"), {"yscale": "log", "ylabel": "levels"})
    assert base.style == () and fig.data().equals(base.data())  # a new figure; the numbers drawn are unchanged
    with pytest.raises(ValueError, match="'dottted' is not a valid value"):
        base.line("c", linestyle="dottted")  # checked by matplotlib when recorded, not when drawn
    with pytest.raises(ValueError, match="color"):
        base.line("c", color="notacolour")
    with pytest.raises(ValueError, match="plain|numbers, text"):
        base.line("c", color=object())
    with pytest.raises(ValueError, match="layout takes"):
        base.layout(figsize=(6, 3), papersize="A4")
    with pytest.raises(ValueError, match="no line 'z' in this figure; its lines are c, i"):
        base.line("z", color="red").draw()


def test_styles_reach_the_drawn_figure(data):
    from matplotlib import dates
    from matplotlib.colors import to_hex

    fig = (data.plot(["c", "i"])
           .line("c", color="#aa0000", linestyle="dashed", linewidth=3)
           .line("i", marker="o", markersize=4)
           .set(yscale="log", ylim=(10, 1000), ylabel="levels", xlim=("2010Q1", "2020Q4"))
           .ticks(axis="y", format="{x:,.0f}", labelsize=6)
           .grid(visible=False)
           .spines(which="left", visible=False)
           .legend(loc="lower right", ncols=2, frameon=True)
           .layout(figsize=(8, 4), suptitle="Styled"))
    drawn = _drawn(fig)
    ax = drawn.axes[0]
    c, i = (l for l in ax.lines if l.get_label() in ("c", "i"))
    assert to_hex(c.get_color()) == "#aa0000" and c.get_linestyle() == "--" and c.get_linewidth() == 3
    assert i.get_marker() == "o" and i.get_markersize() == 4
    assert ax.get_yscale() == "log" and ax.get_ylim() == (10, 1000) and ax.get_ylabel() == "levels"
    lo, hi = (dates.num2date(x).date().isoformat() for x in ax.get_xlim())
    assert (lo, hi) == ("2010-01-01", "2020-10-01")
    assert ax.yaxis.get_major_formatter()(1234.0, 0) == "1,234"
    assert not ax.spines["left"].get_visible()
    legend = ax.get_legend()
    assert legend._loc == 4 and legend._ncols == 2 and legend.get_frame_on()  # lower right, two columns, framed
    assert to_hex(legend.legend_handles[0].get_color()) == "#aa0000"  # the legend shows the styled line
    assert tuple(drawn.get_size_inches()) == (8, 4) and drawn.get_suptitle() == "Styled"


def test_panels_are_addressed_by_position_title_or_all(data):
    fig = data.plot(["c", "i"], layout="each")
    titled = fig.set(ax="i", ylabel="investment")
    assert [a.get_ylabel() for a in _drawn(titled).axes] == ["", "investment"]
    everywhere = fig.grid(ax="all", visible=False)
    assert all(not any(l.get_visible() for l in a.get_ygridlines()) for a in _drawn(everywhere).axes)
    assert [l.get_color() for l in _drawn(fig.line("i", color="red")).axes[1].lines][0] == "red"  # found in its panel
    with pytest.raises(ValueError, match="no panel 5"):
        fig.set(ax=5, ylabel="x").draw()
    with pytest.raises(ValueError, match="no panel titled 'z'"):
        fig.set(ax="z", ylabel="x").draw()
    with pytest.raises(ValueError, match="no right-hand axis"):
        fig.set(side="right", ylabel="x").draw()
    twin = data.plot(["y", "i"], right="i").set(side="right", ylabel="investment")
    assert _drawn(twin).axes[1].get_ylabel() == "investment"
    assert len(_drawn(twin.line("i", color="red")).axes[1].lines) == 1  # a right-axis series by its own name


def test_a_styled_figure_replays_and_saves_exactly(data, tmp_path):
    import json

    from pymetrix.commands import Session, StyleFigure
    from pymetrix.saved import _decode_view, _encode_view

    s = Session()
    s.objects["data"] = data
    s.objects["fit"] = Equation("c ~ y").fit(data)
    form = StyleFigure("fit", series="actual (line)", color="C3", linestyle="dashed", linewidth="2", marker="o",
                       yscale="log", ylabel="levels", tick_format="{x:,.0f}", grid="off", hide_spines="top, right",
                       legend_loc="lower right", legend_columns="2", legend_frame="off", width="8", height="4", name="fit_graph")
    assert form.validate(s) == []
    s.run(form)
    code = s.history[-1].code
    assert code == ("fit_graph = figure_of(fit).line('actual', color='C3', linestyle='dashed', linewidth=2, marker='o')"
                    ".set(yscale='log', ylabel='levels').ticks(axis='y', format='{x:,.0f}')"
                    ".grid(visible=False).spines(which='top', visible=False).spines(which='right', visible=False)"
                    ".legend(loc='lower right', ncols=2, frameon=False).layout(figsize=(8, 4))")
    ns = {"fit": s["fit"]}
    exec("from pymetrix import *\n" + code, ns)  # the code a user reads replays to the same figure, pixel for pixel
    assert ns["fit_graph"].style == s["fit_graph"].style and ns["fit_graph"].png() == s["fit_graph"].png()
    saved = _decode_view(json.loads(json.dumps(_encode_view(s["fit_graph"]))))  # and a saved result keeps it
    assert saved.style == s["fit_graph"].style and saved.png() == s["fit_graph"].png()
    assert s["fit"].plot().style == ()  # the result itself is untouched


def test_the_style_dialog_offers_the_figures_series_and_checks_values(data):
    from pymetrix.commands import PlotSeries, Session, StyleFigure, form_spec

    s = Session()
    s.objects["data"] = data
    s.objects["fit"] = Equation("c ~ y").fit(data)
    s.run(PlotSeries("c, i", kind="bar", start="2023Q1"))
    spec = next(f for f in form_spec(StyleFigure) if f.name == "series")
    assert spec.dynamic and s.choices(spec, {"source": "fit"}) == ["", "actual (line)", "fitted (line)", "±2 s.e. (band)", "residuals (line)"]
    assert s.choices(spec, {"source": "figure"}) == ["", "c (bars)", "i (bars)"]
    bars = StyleFigure.for_figure("figure", s["figure"], "Figure", is_figure=True)
    assert bars.name == "figure"
    s.run(StyleFigure("figure", series="c (bars)", color="navy", marker="o", name="figure"))  # no markers on bars: left out
    assert s.history[-1].code == "figure = figure_of(figure).bars('c', color='navy')"
    assert StyleFigure("fit", series="actual (line)", color="notacolour").validate(s)[0].startswith("line: 'notacolour' is not a valid value")
    assert StyleFigure("fit", tick_format="{x:,.0q}").validate(s)  # a format matplotlib cannot use: said in the dialog
    assert StyleFigure("nothing").validate(s) == ["no object named 'nothing'"]


def test_the_figure_dialogs_open_with_their_titles(data):
    """Figure options and Figure style have no menu item (Edit… opens them): their dialogs must still build."""
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix import figure_of
    from pymetrix.commands import FigureOptions, Session, StyleFigure
    from pymetrix.ui.qt import command_dialog

    s = Session()
    s.objects["data"] = data
    s.objects["fit"] = Equation("c ~ y").fit(data)
    figure = figure_of(s["fit"])
    options = command_dialog(FigureOptions, s, initial=FigureOptions.for_figure("fit", figure))
    style = command_dialog(StyleFigure, s, initial=StyleFigure.for_figure("fit", figure))
    assert options.windowTitle() == "Figure options" and style.windowTitle() == "Figure style"
    assert style.editors["name"].get() == "fit_graph"


# -- every property matplotlib has (phase B): the catalogue, the property editor, tabs --


def test_the_catalogue_is_matplotlibs_own():
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    from pymetrix.figure_properties import properties, property_table

    classes = {"line": Line2D, "bars": Rectangle, "band": PolyCollection}
    for part, cls in classes.items():
        names = {p.name for p in properties(part)}
        assert all(hasattr(cls, f"set_{n}") for n in names)  # each is a real setter of that artist
        assert not names & {"xdata", "ydata", "data", "xy", "x", "y", "width", "height", "paths", "verts", "transform", "picker"}
    line = {p.name: p for p in properties("line")}
    assert line["drawstyle"].choices == ("", "default", "steps", "steps-pre", "steps-mid", "steps-post")
    assert line["linewidth"].kind == "number" and line["visible"].kind == "bool" and line["color"].help == "color"
    assert "\\" in {p.name: p for p in properties("bars")}["hatch"].choices
    axes = {p.name for p in properties("axes")}
    assert {"facecolor", "ylim", "yscale", "aspect", "frame_on"} <= axes and "prop_cycle" not in axes
    assert len(property_table("line").rows) == len(properties("line"))


def test_every_catalogued_property_can_be_applied(data):
    """The editor never offers what matplotlib cannot use: each property with a fixed kind of value, applied and drawn."""
    from pymetrix.figure_properties import properties

    base = data.plot(["c"], kind="bar", span=("2023Q1", None))
    examples = {"bool": "True", "number": 1.0}
    figures = {"line": data.plot("c"), "bars": base, "band": data.plot("c", kind="area"), "axes": data.plot("c")}
    for part, figure in figures.items():
        for p in properties(part):
            value = True if p.kind == "bool" else p.choices[1] if p.kind == "choice" else examples.get(p.kind)
            if value is None or p.name in ("alpha",) and part == "axes":
                continue
            styled = figure.set(**{p.name: value}) if part == "axes" else getattr(figure, part)("c", **{p.name: value})
            styled.draw()


def test_the_property_editor_records_matplotlibs_call(data):
    from pymetrix.commands import Session, SetProperties, form_spec, option_string

    s = Session()
    s.objects["data"] = data
    s.objects["fit"] = Equation("c ~ y").fit(data)
    part = next(f for f in form_spec(SetProperties) if f.name == "part")
    assert s.choices(part, {"source": "fit"}) == ["", "actual (line)", "fitted (line)", "±2 s.e. (band)", "residuals (line)",
                                                  "panel 0: Actual and fitted (axes)", "panel 1: Residuals (axes)"]
    s.objects["twin"] = data.plot(["y", "i"], right="i")
    assert "panel 0 right (axes)" in s.choices(part, {"source": "twin"})
    props = next(f for f in form_spec(SetProperties) if f.name == "properties")
    sub = s.options_form(props, {"part": "actual (line)"})
    text = option_string(sub, {"drawstyle": "steps-post", "markevery": "4", "color": "#aa0000", "linewidth": "2.5",
                               "antialiased": "False", "dashes": "(4, 2)"})
    assert text == "antialiased=False, color='#aa0000', dashes=((4, 2)), drawstyle='steps-post', linewidth=2.5, markevery=4"
    s.run(SetProperties("fit", part="actual (line)", properties=text, name="fit_graph"))
    code = s.history[-1].code
    assert code == ("fit_graph = figure_of(fit).line('actual', antialiased=False, color='#aa0000', dashes=(4, 2), "
                    "drawstyle='steps-post', linewidth=2.5, markevery=4)")
    ns = {"fit": s["fit"]}
    exec("from pymetrix import *\n" + code, ns)
    assert ns["fit_graph"].png() == s["fit_graph"].png()  # replayed, pixel for pixel
    actual = next(l for l in _drawn(s["fit_graph"]).axes[0].lines if l.get_label() == "actual")
    assert actual.get_drawstyle() == "steps-post" and actual.get_markevery() == 4 and not actual.get_antialiased()
    axes_text = option_string(s.options_form(props, {"part": "panel 1 (axes)"}), {"facecolor": "0.95", "ylim": "(-3, 3)"})
    s.run(SetProperties("fit", part="panel 1: Residuals (axes)", properties=axes_text, name="g2"))
    assert s.history[-1].code == "g2 = figure_of(fit).set(ax=1, facecolor='0.95', ylim=(-3, 3))"
    assert _drawn(s["g2"]).axes[1].get_ylim() == (-3, 3)
    assert "is not a valid value" in SetProperties("fit", part="actual (line)", properties="drawstyle='zigzag'").validate(s)[0]
    assert "no figure part" in SetProperties("fit", part="nothing").validate(s)[0]


def test_the_dialogs_have_tabs_and_the_editor_lists_properties(data):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix import figure_of
    from pymetrix.commands import Session, SetProperties, StyleFigure
    from pymetrix.figure_properties import properties
    from pymetrix.ui.qt import command_dialog, figure_view

    s = Session()
    s.objects["data"] = data
    s.objects["fit"] = Equation("c ~ y").fit(data)
    style = command_dialog(StyleFigure, s, initial=StyleFigure.for_figure("fit", figure_of(s["fit"])))
    assert [style.tabs.tabText(i) for i in range(style.tabs.count())] == ["General", "Series", "Axes", "Legend", "Layout"]
    assert style.advanced_toggle is not None  # its advanced settings: the one collapsed section, not a tab
    editor = command_dialog(SetProperties, s, initial=SetProperties("fit", part="actual (line)", name="g"))
    assert editor.windowTitle() == "Figure properties"
    assert [f.name for f in editor.editors["properties"].sub_specs] == [p.name for p in properties("line")]
    assert editor.editors["properties"].sub["antialiased"].spec.display("") == "(unchanged)"
    page = figure_view(figure_of(s["fit"]), on_edit=lambda kind: None)
    assert set(page.edit_actions) == {"options", "style", "properties", "note", "method", "styles", "copy", "house", "arrow", "textbox", "change"}


# -- things added to a figure, and click-to-place (phase C) --


def test_notes_lines_and_bands_go_where_the_data_say(data):
    from matplotlib import dates

    def num(period):
        return dates.date2num(pd.Period(period, freq="Q").to_timestamp())

    fig = (data.plot("c")
           .annotate("crisis", xy=("2008Q4", 300), xytext=("2005Q1", 400), arrowprops={"arrowstyle": "->"})
           .annotate("peak", xy=("2024Q4", 450), xytext=(10, 5), textcoords="offset points")
           .text("2010Q1", 350, "calm", style="italic")
           .axhline(250, linestyle=":")
           .axvline("2020Q1", color="C3")
           .axvspan("2012Q1", "2013Q4", alpha=0.2)
           .axhspan(200, 220, alpha=0.1))
    ax = _drawn(fig).axes[0]
    crisis, peak = (t for t in ax.texts if t.get_text() in ("crisis", "peak"))
    assert np.allclose(crisis.xy, (num("2008Q4"), 300)) and np.allclose(crisis.xyann, (num("2005Q1"), 400))
    assert crisis.arrow_patch is not None and peak.arrow_patch is None and peak.xyann == (10, 5)
    calm = next(t for t in ax.texts if t.get_text() == "calm")
    assert np.allclose(calm.get_position(), (num("2010Q1"), 350)) and calm.get_style() == "italic"
    assert any(l.get_ydata()[0] == 250 and l.get_linestyle() == ":" for l in ax.lines)
    assert any(np.isclose(dates.date2num(pd.Timestamp(l.get_xdata()[0])) if not isinstance(l.get_xdata()[0], (int, float, np.floating)) else l.get_xdata()[0], num("2020Q1")) for l in ax.lines)
    spans = [p for p in ax.patches if np.isclose(p.get_alpha() or 0, 0.2)]

    def extent(p):
        xs = [p.get_x(), p.get_x() + p.get_width()] if hasattr(p, "get_width") else list(np.asarray(p.get_xy())[:, 0])
        return min(xs), max(xs)

    assert np.allclose(extent(spans[0]), (num("2012Q1"), num("2014Q1")))  # through the end of the last quarter
    with pytest.raises(ValueError, match="Unknown style"):
        data.plot("c").annotate("x", xy=("2008Q4", 1), arrowprops={"arrowstyle": "-->>"})
    with pytest.raises(ValueError, match="not a period"):
        data.plot("c").axvline("2008Q5").draw()


def test_the_note_dialog_records_matplotlibs_call(data):
    import json

    from pymetrix.commands import AnnotateFigure, Session
    from pymetrix.saved import _decode_view, _encode_view

    s = Session()
    s.objects["data"] = data
    s.objects["fit"] = Equation("c ~ y").fit(data)
    form = AnnotateFigure.at("fit", s["fit"].plot(), panel=1, x="2008Q4", y=-1.23456)
    assert (form.panel, form.x, form.y, form.name) == ("1", "2008Q4", "-1.235", "fit_graph")  # as clicked
    from dataclasses import replace

    s.run(replace(form, text="crisis", color="C3"))
    code = s.history[-1].code
    assert code == ("fit_graph = figure_of(fit).annotate('crisis', ('2008Q4', -1.235), xytext=(15, 20), ax=1, "
                    "textcoords='offset points', arrowprops={'arrowstyle': '->', 'color': 'C3'}, color='C3')")
    ns = {"fit": s["fit"]}
    exec("from pymetrix import *\n" + code, ns)
    assert ns["fit_graph"].png() == s["fit_graph"].png()  # replays, at any size: the note is in data terms
    saved = _decode_view(json.loads(json.dumps(_encode_view(s["fit_graph"]))))
    assert saved.png() == s["fit_graph"].png()
    cases = {"axvspan": dict(x="2012Q1", x2="2013Q4", alpha="0.2"), "axhline": dict(y="0", linestyle="dotted"),
             "text": dict(x="2010Q1", y="300", text="calm"), "axvline": dict(x="2020Q1"), "axhspan": dict(y="-1", y2="1")}
    expected = {"axvspan": "figure_of(fit).axvspan('2012Q1', '2013Q4', alpha=0.2)",
                "axhline": "figure_of(fit).axhline(linestyle='dotted')",  # y=0 is the default
                "text": "figure_of(fit).text('2010Q1', 300, 'calm')", "axvline": "figure_of(fit).axvline('2020Q1')",
                "axhspan": "figure_of(fit).axhspan(-1, 1)"}
    for what, fields in cases.items():
        command = AnnotateFigure("fit", what=what, name="g", **fields)
        assert command.validate(s) == [] and command.to_code() == f"g = {expected[what]}"
    assert AnnotateFigure("fit", what="annotate", text="x").validate(s) == ["give the x, y (click on the figure, or type them)"]
    assert AnnotateFigure("fit", what="axvspan", x="2012Q1").validate(s) == ["give the x2 (click on the figure, or type it)"]
    assert AnnotateFigure("fit", what="text", x="2010Q1", y="3").validate(s) == ["give the words"]


def test_clicking_on_a_figure_places_a_note(data):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from types import SimpleNamespace

    from matplotlib import dates

    from pymetrix.ui.qt import figure_view

    placed = []
    page = figure_view(data.plot(["y", "i"], right="i"), on_edit=lambda kind: None, on_place=lambda *a: placed.append(a))
    left, right = page.canvas.figure.axes[:2]
    middle = dates.date2num(pd.Timestamp("2008-11-15"))
    page.clicked(SimpleNamespace(inaxes=left, xdata=middle, ydata=3.21))  # not in Add a note mode: nothing
    assert placed == []
    page.place.setChecked(True)
    page.clicked(SimpleNamespace(inaxes=right, xdata=middle, ydata=40.5))  # on the right-hand axis: the same panel
    assert placed == [(0, "2008Q4", 40.5)] and not page.place.isChecked()  # the quarter clicked; the mode ends
    assert "note" in page.edit_actions


# -- the rest of matplotlib (phase D): any Axes method as data, style sheets, standalone code --


def test_any_styling_axes_method_is_recorded_as_data(data):
    from pymetrix.figure_properties import axes_methods

    methods = axes_methods()
    assert {"set_aspect", "ticklabel_format", "minorticks_on", "axline", "hlines", "invert_yaxis"} <= set(methods)
    assert not {"plot", "bar", "scatter", "clear", "cla", "remove", "set_prop_cycle", "set_figure"} & set(methods)
    fig = data.plot("c").call("minorticks_on").call("ticklabel_format", axis="y", style="sci", scilimits=(0, 0))
    assert fig.style[1].props == {"method": "ticklabel_format", "args": (), "kwargs": {"axis": "y", "style": "sci", "scilimits": (0, 0)}}
    ax = _drawn(fig).axes[0]
    assert len(ax.xaxis.get_minorticklocs()) > 0 and ax.yaxis.get_major_formatter().get_scientific() if hasattr(ax.yaxis.get_major_formatter(), "get_scientific") else True
    with pytest.raises(ValueError, match="not an Axes method fig.call records"):
        data.plot("c").call("plot", [1, 2], [3, 4])
    with pytest.raises(ValueError, match="set_aspect"):
        data.plot("c").call("set_aspect", "equal", nonsense=1)  # checked against the method's signature
    twin = data.plot(["y", "i"], right="i").call("set_ylabel", "investment", side="right")
    assert _drawn(twin).axes[1].get_ylabel() == "investment"


def test_style_sheets_become_themes(data, tmp_path):
    from matplotlib.colors import to_hex

    from pymetrix import Theme, get_theme

    sheet = tmp_path / "house.mplstyle"
    sheet.write_text("axes.prop_cycle: cycler('color', ['112233', 'aa5500'])\nfont.size: 9\nlines.linewidth: 2.5\n"
                     "axes.grid: False\naxes.facecolor: fafafa\naxes.spines.top: False\n", encoding="utf-8")
    from pymetrix.themes import _THEMES

    theme = Theme.from_mplstyle(str(sheet))
    try:
        assert theme.name == "house" and get_theme("house") is theme
        assert theme.palette == ("#112233", "#aa5500") and theme.line_width == 2.5 and theme.font_size == 9 and not theme.grid
        ax = _drawn(data.plot(["c", "i"], theme="house")).axes[0]
        assert [to_hex(l.get_color()) for l in ax.lines[:2]] == ["#112233", "#aa5500"]
        assert to_hex(ax.get_facecolor()) == "#fafafa"  # the sheet's other settings apply too
    finally:
        _THEMES.pop("house", None)  # the registry is shared with other tests
    with pytest.raises(LookupError, match="no style sheet"):
        Theme.from_mplstyle(str(tmp_path / "missing.mplstyle"))


def _run_standalone(code):
    import re
    import sys

    import matplotlib

    assert not re.search(r"^\s*(import|from) pymetrix", code, re.M)  # matplotlib, NumPy and pandas only
    ns: dict = {"__name__": "standalone"}
    # run as a plain script would: matplotlib 3.8's pyplot hooks into an IPython shell left by other tests, through a
    # function newer IPython no longer has
    import matplotlib.pyplot as plt

    plt.close("all")  # the scripts' earlier figures
    ipython = sys.modules.pop("IPython", None)
    try:
        matplotlib.use("Agg")  # switching a loaded pyplot reaches the same hook
        exec(compile(code, "standalone.py", "exec"), ns)
        ns["fig"].canvas.draw()
    finally:
        if ipython is not None:
            sys.modules["IPython"] = ipython
    return ns["fig"]


def test_standalone_matplotlib_code_draws_the_same_chart(data):
    from matplotlib.colors import to_hex

    d = data.with_series(gc="400*dlog(c)", gi="400*dlog(i)")
    fig = (d.plot(["gc", "gi"], shade="nber", events="2020Q1=COVID", reference="2=target", title="Growth")
           .line("gc", color="C3", linestyle="--").set(ylabel="% a year", ylim=(-15, 25))
           .ticks(axis="y", format="{x:.0f}%").legend(loc="upper left")
           .annotate("dip", xy=("2020Q2", -10), xytext=(15, -20), textcoords="offset points", arrowprops={"arrowstyle": "->"})
           .call("minorticks_on"))
    theirs = _run_standalone(fig.matplotlib_code()).axes[0]
    ours = _drawn(fig).axes[0]
    for name in ("gc", "gi"):
        a = next(l for l in ours.lines if l.get_label() == name)
        b = next(l for l in theirs.lines if l.get_label() == name)
        assert np.allclose(np.asarray(a.get_ydata(), float), np.asarray(b.get_ydata(), float), equal_nan=True)
        assert to_hex(a.get_color()) == to_hex(b.get_color()) and a.get_linestyle() == b.get_linestyle()
    assert theirs.get_ylim() == ours.get_ylim() == (-15, 25) and theirs.get_ylabel() == "% a year"
    assert theirs.yaxis.get_major_formatter()(12, 0) == "12%"
    assert {"dip", "COVID ", "target "} <= {t.get_text() for t in theirs.texts}
    shaded = lambda ax: sum(1 for p in ax.patches if np.isclose(p.get_alpha() or 0, 0.18))  # noqa: E731
    assert shaded(theirs) == shaded(ours) == 3
    assert len(theirs.xaxis.get_minorticklocs()) > 0


@pytest.mark.parametrize("make", [
    lambda d: d.plot(["c", "i"], kind="bar", stacked=True, span=("2023Q1", None), bar_labels=True),
    lambda d: d.plot(["c", "i", "g"], layout="each", sharey=True),
    lambda d: d.plot(["y", "i"], right="i", line_labels=True),
    lambda d: d.plot(["c", "i"], kind="box"),
    lambda d: d.plot(["y", "c"], kind="scatter"),
    lambda d: d.plot("c", kind="histogram"),
], ids=["stacked bars", "panels", "right axis", "box", "scatter", "histogram"])
def test_standalone_code_for_every_kind_runs(data, make):
    fig = make(data)
    theirs = _run_standalone(fig.matplotlib_code())
    ours = _drawn(fig)
    assert len([a for a in theirs.axes if a.get_visible()]) == len([a for a in ours.axes if a.get_visible()])
    for a, b in zip(ours.axes, theirs.axes):
        assert len(a.patches) == len(b.patches)  # bars, boxes, histogram bins
        mine = [np.asarray(l.get_ydata(), float) for l in a.lines]
        other = [np.asarray(l.get_ydata(), float) for l in b.lines]
        assert len(mine) == len(other) and all(np.allclose(x, y, equal_nan=True) for x, y in zip(mine, other))
        assert sorted(t.get_text() for t in a.texts) == sorted(t.get_text() for t in b.texts)  # line names, bar values


def test_the_method_dialog_and_copy_code_button(data):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix import figure_of
    from pymetrix.commands import CallAxesMethod, Session, form_spec
    from pymetrix.ui.qt import figure_view

    s = Session()
    s.objects["data"] = data
    s.objects["fit"] = Equation("c ~ y").fit(data)
    method = next(f for f in form_spec(CallAxesMethod) if f.name == "method")
    assert "ticklabel_format" in s.choices(method, {})
    command = CallAxesMethod("fit", method="ticklabel_format", keywords="axis='y', style='sci', scilimits=(0, 0)", name="g")
    assert command.validate(s) == []
    s.run(command)
    assert s.history[-1].code == "g = figure_of(fit).call('ticklabel_format', axis='y', style='sci', scilimits=(0, 0))"
    assert "set_aspect" in CallAxesMethod("fit", method="set_aspect", arguments="'equal'", keywords="nonsense=1").validate(s)[0]
    page = figure_view(figure_of(s["fit"]), on_edit=lambda kind: None)
    assert "method" in page.edit_actions
    copy = next(b for b in page.findChildren(QtWidgets.QPushButton) if b.text() == "Copy plotting code")
    copy.click()
    assert app.clipboard().text().startswith('"""c ~ y, drawn with matplotlib alone')


# -- managing recorded styles: list, remove, reorder, change, copy to another figure --


def _styled_twin(data):
    return (data.plot(["y", "i"], right="i").line("y", color="C3", ls="--").set(ylabel="%").set(side="right", ylabel="inv")
            .ticks(axis="y", format="{x:.0f}").grid(visible=False).spines(which="top", visible=False).legend(loc="upper left")
            .layout(figsize=(6, 3)).annotate("dip", xy=("2020Q2", 100), xytext=(15, 20), textcoords="offset points")
            .axvline("2020Q1").call("minorticks_on").call("set_ylabel", "z", side="right"))


def test_each_recorded_style_reads_as_the_call_that_records_it(data):
    from dataclasses import replace

    fig = _styled_twin(data)
    steps = fig.style_steps()
    assert steps[:3] == ["line('y', color='C3', linestyle='--')", "set(ax=0, ylabel='%')", "set(ax=0, side='right', ylabel='inv')"]
    assert steps[-1] == "call('set_ylabel', 'z', ax=0, side='right')"
    rebuilt = replace(fig, style=())
    for code in steps:  # each step, run as written, records the same call: the list is the code
        rebuilt = eval(f"rebuilt.{code}", {"rebuilt": rebuilt})
    assert rebuilt.style == fig.style and rebuilt.png() == fig.png()


def test_styles_are_removed_reordered_and_changed(data):
    from matplotlib.colors import to_hex

    from pymetrix.errors import SpecificationError

    fig = _styled_twin(data)
    n = len(fig.style)
    assert fig.without_style(0, -1).style == fig.style[1:-1]
    assert fig.reorder_style([1, 0]).style == (fig.style[1], fig.style[0])  # those not listed are removed
    with pytest.raises(SpecificationError, match=f"no style step 99: this figure has {n}"):
        fig.without_style(99)
    with pytest.raises(SpecificationError, match="listed twice"):
        fig.reorder_style([0, 0])
    later_wins = data.plot("c").line("c", color="C1").line("c", color="C2")
    line = lambda f: next(l for l in _drawn(f).axes[0].lines if l.get_label() == "c")  # noqa: E731
    assert to_hex(line(later_wins).get_color()) == to_hex("C2")
    assert to_hex(line(later_wins.reorder_style([1, 0])).get_color()) == to_hex("C1")
    changed = fig.restyle(0, color="C2", linestyle=None)  # change one, remove one, keep its place
    assert changed.style[0].props == {"color": "C2"} and changed.style[1:] == fig.style[1:]
    assert fig.restyle(0, replace_all=True, lw=3).style[0].props == {"linewidth": 3}
    assert fig.restyle(-1, fontsize=7).style[-1].props["kwargs"] == {"fontsize": 7}
    moved = fig.restyle(8, xy=("2021Q1", 90))
    assert moved.style[8].props["xy"] == ("2021Q1", 90) and moved.style[8].props["text"] == "dip"
    with pytest.raises(SpecificationError, match="annotate needs text"):
        fig.restyle(8, replace_all=True, xy=("2021Q1", 90))
    with pytest.raises(ValueError, match="dottted"):
        fig.restyle(0, linestyle="dottted")  # matplotlib checks the new value


def test_a_style_is_copied_where_it_applies(data):
    house = _styled_twin(data)
    plain = data.plot("y")  # draws y, one panel, no right-hand axis
    copied = plain.with_style_of(house)
    assert copied.style_steps() == ["line('y', color='C3', linestyle='--')", "set(ax=0, ylabel='%')",
                                   "ticks(ax=0, axis='y', format='{x:.0f}')", "grid(ax=0, visible=False)",
                                   "spines(ax=0, which='top', visible=False)", "legend(ax=0, loc='upper left')",
                                   "layout(figsize=(6, 3))", "call('minorticks_on', ax=0)"]
    assert plain.style_skipped(house) == [  # the right-hand axis, and the notes on the other chart's data
        "set(ax=0, side='right', ylabel='inv')",
        "annotate(text='dip', xy=('2020Q2', 100), xytext=(15, 20), textcoords='offset points', ax=0)",
        "axvline(x='2020Q1', ax=0)", "call('set_ylabel', 'z', ax=0, side='right')"]
    assert len(plain.with_style_of(house, notes=True).style) == len(copied.style) + 2
    assert _drawn(copied).axes[0].get_ylabel() == "%" and tuple(copied.draw().get_size_inches()) == (6, 3)
    assert data.plot("c").with_style_of(house).style[0].kind == "set"  # no line 'y' here: its call is left out
    own = data.plot("y").line("y", lw=4).with_style_of(house)
    assert own.style[0].props == {"linewidth": 4} and own.style[1] == house.style[0]  # added after its own
    assert plain.with_style_of(house.style[:2]).style == house.style[:2]  # or from the calls themselves
    titled = Figure((Panel((Line(data["y"], label="y"),), title="Output"),))
    assert titled.with_style_of(data.plot("y").set(ax=0, ylabel="t")).style[0].props == {"ylabel": "t"}
    assert titled.with_style_of(Figure((Panel((Line(data["y"], label="y"),), title="Prices"),)).set(ax="Prices", ylabel="t")).style == ()


def test_the_style_commands_record_and_replay(data):
    from dataclasses import replace

    from pymetrix.commands import ArrangeStyles, ChangeStyle, CopyStyle, Session, form_spec

    s = Session()
    s.objects["data"] = data
    s.objects["house"] = _styled_twin(data)
    s.objects["fit"] = Equation("c ~ y").fit(data)
    s.objects["g"] = data.plot("y")
    n = len(s["house"].style)
    arrange = ArrangeStyles.for_figure("house", s["house"], is_figure=True)
    assert arrange.keep == ", ".join(map(str, range(n))) and arrange.steps == n
    s.run(replace(arrange, keep=", ".join(str(k) for k in range(n) if k not in (2, 9))))
    assert s.history[-1].code == "house = figure_of(house).without_style(2, 9)"
    assert len(s["house"].style) == n - 2
    s.run(ArrangeStyles("house", keep="1, 0", steps=n - 2, name="two"))
    assert s.history[-1].code == "two = figure_of(house).reorder_style([1, 0])"
    assert ArrangeStyles("house", keep="0, x").validate(s) == ["'x' is not a style step number"]
    step = next(f for f in form_spec(ChangeStyle) if f.name == "step")
    assert s.choices(step, {"source": "two"}) == ["", "0: set(ax=0, ylabel='%')", "1: line('y', color='C3', linestyle='--')"]
    change = ChangeStyle.for_step("two", s["two"], 1, is_figure=True)
    assert change.step == "1: line('y', color='C3', linestyle='--')" and change.properties == "color='C3', linestyle='--'"
    s.run(replace(change, properties="color='C3', linewidth=2"))
    assert s.history[-1].code == "two = figure_of(two).restyle(1, replace_all=True, color='C3', linewidth=2)"
    assert s["two"].style[1].props == {"color": "C3", "linewidth": 2}
    assert "dottted" in ChangeStyle("two", step="1", properties="ls='dottted'").validate(s)[0]
    s.run(CopyStyle("fit", style_from="two", name="fit_styled"))
    assert s.history[-1].code == "fit_styled = figure_of(fit).with_style_of(figure_of(two))"
    assert s["fit_styled"].style_steps() == ["set(ax=0, ylabel='%')"]  # fit draws no series 'y'
    assert CopyStyle("g", style_from="g").validate(s) == ["g has no recorded styles to copy"]
    assert CopyStyle("g", style_from="").validate(s) == ["name the figure whose style to copy"]
    nothing = Session()
    nothing.objects.update(a=data.plot("c"), b=data.plot("y").line("y", color="C1"))
    assert CopyStyle("a", style_from="b").validate(nothing) == ["none of b's styles apply here: line('y', color='C1')"]
    ns = {"house": _styled_twin(data), "fit": s["fit"]}
    exec("from pymetrix import *\n" + "\n".join(r.code for r in s.history), ns)  # the steps replay as plain Python
    for name in ("house", "two", "fit_styled"):
        assert ns[name].style == s[name].style


def test_the_recorded_styles_dialog(data):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import style_steps_dialog

    fig = data.plot("c").line("c", color="C1").set(ylabel="a").grid(visible=False)
    dialog = style_steps_dialog(fig)
    assert [dialog.steps.item(i).text() for i in range(3)] == fig.style_steps()
    dialog.steps.setCurrentRow(2)
    dialog.buttons["Up"].click()  # the grid call now applies before the label
    assert dialog.order() == [0, 2, 1]
    dialog.steps.setCurrentRow(0)
    dialog.buttons["Remove"].click()
    assert dialog.order() == [2, 1]
    dialog.steps.setCurrentRow(1)
    dialog.buttons["Change…"].click()
    assert dialog.change == 1 and dialog.result() == QtWidgets.QDialog.DialogCode.Accepted


def test_the_window_rearranges_changes_and_copies_styles(data, monkeypatch):
    pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.commands import FitEquation, Session
    from pymetrix.ui import qt
    from pymetrix.ui.qt import main_window

    opened = []

    class Accept:  # a command dialog, accepted as filled in
        def __init__(self, cls, session, parent=None, initial=None):
            opened.append(initial)
            self.session, self.initial = session, initial

        def exec(self):
            if type(self.initial).__name__ == "CopyStyle":
                return False  # opened on the figure, then cancelled
            self.command = self.initial
            self.result_value = self.session.run(self.command)
            return True

    class Arranged:  # the Recorded styles list: the last two swapped, the one now first picked to change
        def __init__(self, figure, parent=None):
            self.change = 2

        def exec(self):
            return True

        def order(self):
            return [0, 2, 1]

    monkeypatch.setattr(qt, "command_dialog", Accept)
    monkeypatch.setattr(qt, "style_steps_dialog", Arranged)
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: opened.append("none recorded"))
    session = Session()
    session.objects["data"] = data
    session.run(FitEquation("c ~ y", name="fit"))
    session.objects["chart"] = data.plot("y").line("y", color="C1").set(ylabel="a").grid(visible=False)
    window = main_window(session)
    try:
        from pymetrix import figure_of

        window.edit_figure("fit", "Graph", figure_of(session["fit"]), "styles")
        assert opened == ["none recorded"]
        window.edit_figure("chart", "Figure", session["chart"], "styles")
        assert [r.code for r in session.history[-2:]] == [
            "chart = figure_of(chart, tab='Figure').reorder_style([0, 2, 1])",
            "chart = figure_of(chart).restyle(1, replace_all=True, visible=False)"]  # the grid call, now step 1
        window.edit_figure("fit", "Graph", figure_of(session["fit"]), "copy")
        assert type(opened[-1]).__name__ == "CopyStyle" and opened[-1].name == "fit_graph"
    finally:
        session.dirty = False
        window.close()


# -- macro charts: contributions, fans, vintages --------------------------------------------------------------------


def _bar_sums(ax):
    """The stacked bars' ends at each x: positive segments up, negative down, as drawn."""
    from collections import defaultdict

    sums = defaultdict(float)
    for p in ax.patches:
        if p.get_label() != "_nolegend_" or p.get_height():
            sums[round(p.get_x() + p.get_width() / 2, 6)] += p.get_height()
    return sums


def test_contributions_add_up_to_the_total_line(data):
    from pymetrix.figures import contributions_figure

    d = data.with_series(dc="d(c)", di="d(i)", dg="d(g)", dy="d(y)")
    parts = d.frame[["dc", "di", "dg"]].dropna().iloc[-12:]
    exact = contributions_figure(parts)  # no total: the sum is the total, and nothing is left over
    assert [type(l).__name__ + ":" + getattr(l, "label", "") for l in exact.panels[0].layers] == [
        "Bars:dc", "Bars:di", "Bars:dg", "Line:total", "Rule:"]
    identity = contributions_figure(parts, d.frame["dy"].reindex(parts.index), total_label="dy")
    assert "other" not in [getattr(l, "label", "") for l in identity.panels[0].layers]  # GDP is C + I + G here
    total = 1.1 * d.frame["dy"].reindex(parts.index)
    fig = contributions_figure(parts, total, total_label="dy")
    assert [getattr(l, "label", "") for l in fig.panels[0].layers][:5] == ["dc", "di", "dg", "other", "dy"]
    other = next(l for l in fig.panels[0].layers if getattr(l, "label", "") == "other")
    assert np.allclose(other.y, total - parts.sum(axis=1))  # what the parts leave unexplained
    ax = _drawn(fig).axes[0]
    line = next(l for l in ax.lines if l.get_label() == "dy")
    heights = _bar_sums(ax)
    assert len(heights) == len(parts)
    assert np.allclose(sorted(heights.values()), sorted(np.asarray(line.get_ydata(), float)))  # the bars add up to the line
    same = d.with_series(dy2="1.1*dy").plot(["dc", "di", "dg"], kind="contributions", total="dy2", span=(str(parts.index[0]), None))
    assert same.panels[0].layers[3].label == "other" and np.allclose(same.panels[0].layers[3].y, other.y)
    with pytest.raises(SpecificationError, match="not one of its own parts"):
        d.plot(["dc", "dy"], kind="contributions", total="dy")
    with pytest.raises(SpecificationError, match="total is for kind='contributions'"):
        d.plot(["dc"], total="dy")


def test_fans_are_bands_at_each_coverage(data):
    from scipy import stats

    from pymetrix import Dataset
    from pymetrix.figures import Fan
    from pymetrix.saved import _decode_view, _encode_view

    frame = data.frame[["y"]].copy()
    frame["y_se"] = np.nan
    frame.loc[frame.index[-6:], "y_se"] = np.linspace(1, 6, 6)
    fan = Fan.from_frame(frame, "y")
    assert fan.coverages == (0.9, 0.6, 0.3)
    z90 = stats.norm.ppf(0.95)  # an independent check of the band
    assert np.allclose((fan.upper[0] - frame["y"]).dropna(), z90 * frame["y_se"].dropna())
    parts = fan.parts()
    assert [type(p).__name__ + ":" + getattr(p, "label", "") for p in parts] == ["Band:90%", "Band:60%", "Band:30%", "Line:y", "Rule:"]
    assert parts[-1].y == str(frame.index[-6]) and parts[-1].vertical  # the forecast's start
    frame["y_lo80"], frame["y_hi80"] = frame["y"] - 1, frame["y"] + 1
    frame["y_lo50"], frame["y_hi50"] = frame["y"] - 0.5, frame["y"] + 0.5
    given = Fan.from_frame(frame, "y")  # the bands given win over the standard error
    assert given.coverages == (0.8, 0.5) and np.allclose(given.upper[1] - frame["y"], 0.5)
    assert Fan.from_frame(frame, "y", coverages=(0.5,)).coverages == (0.5,)
    assert Fan.from_frame(frame, "y", coverages=(0.95,)).coverages == (0.95,)  # not given: normal, from the standard error
    with pytest.raises(SpecificationError, match="no bands for 'c'"):
        Fan.from_frame(data.frame, "c")
    with pytest.raises(SpecificationError, match="between 0 and 1"):
        Fan.normal(frame["y"], frame["y_se"], coverages=(90,))
    fig = Dataset.from_frame(frame[["y", "y_se"]]).plot("y", kind="fan", span=("2020Q1", None))
    ax = _drawn(fig).axes[0]
    assert len(ax.collections) == 3 and [l.get_label() for l in ax.lines if not l.get_label().startswith("_")] == ["y"]
    from matplotlib import dates

    start = [l for l in ax.lines if l.get_linestyle() == "--"]
    assert len(start) == 1 and np.isclose(start[0].get_xdata()[0], dates.date2num(frame.index[-6].start_time))
    styled = fig.band("90%", alpha=0.05)  # each part is styled as a band or a line
    assert np.isclose(_drawn(styled).axes[0].collections[0].get_alpha(), 0.05)
    assert {"y", "90% lower", "90% upper"} <= set(fig.data().columns)
    saved = _decode_view(json.loads(json.dumps(_encode_view(fig))))  # a saved result keeps the fan
    assert type(saved.panels[0].layers[0]).__name__ == "Fan" and saved.png() == fig.png()
    theirs = _run_standalone(fig.matplotlib_code()).axes[0]  # and matplotlib alone draws the same
    assert len(theirs.collections) == 3 and len([l for l in theirs.lines if l.get_linestyle() == "--"]) == 1
    at = [l for l in theirs.lines if l.get_linestyle() == "--"][0].get_xdata()[0]
    assert np.isclose(dates.date2num(at), start[0].get_xdata()[0])  # a Timestamp there, a date number here


def test_vintages_darken_to_the_latest(data):
    from matplotlib.colors import to_rgb

    from pymetrix.figures import vintages_figure

    releases = {f"release {k}": data["y"].iloc[-12:] * (1 + 0.002 * k) for k in range(4)}
    fig = vintages_figure(releases, title="Revisions")
    lines = [l for l in _drawn(fig).axes[0].lines if l.get_label().startswith("release")]
    lightness = [sum(to_rgb(l.get_color())) for l in lines]
    assert lightness == sorted(lightness, reverse=True)  # each release darker than the one before
    assert [l.get_linewidth() for l in lines][-1] > lines[0].get_linewidth()
    assert np.allclose(lines[-1].get_ydata(), releases["release 3"].to_numpy())
    via_data = data.with_series(y2="1.01*y").plot(["y", "y2"], kind="vintages")
    assert [l.label for l in via_data.panels[0].layers] == ["y", "y2"]


def test_the_plot_dialog_draws_macro_charts(data):
    from pymetrix.commands import PlotSeries, Session, form_spec

    s = Session()
    s.objects["data"] = data.with_series(dc="d(c)", di="d(i)", dy="d(y)")
    s.run(PlotSeries("dc, di", kind="contributions", total="dy", name="parts"))
    assert s.history[-1].code == "parts = data.plot(['dc', 'di'], kind='contributions', total='dy')"
    assert PlotSeries("dc, dy", kind="contributions", total="dy").validate(s) == [
        "the total is not one of its own parts: untick it among the series"]
    assert PlotSeries("y", kind="fan", coverages="30, 150").validate(s) == ["coverages are percentages between 0 and 100, e.g. 30, 60, 90"]
    assert PlotSeries("y", kind="fan", coverages="30, 6o").validate(s) == ["coverages are numbers, e.g. 30, 60, 90"]
    assert PlotSeries("y", kind="fan", coverages="50%, 90%")._options()["coverages"] == (0.5, 0.9)
    total = next(f for f in form_spec(PlotSeries) if f.name == "total")
    assert total.visible({"kind": "contributions"}) and not total.visible({"kind": "line"})
    s.run(PlotSeries("y, c", kind="vintages", layout="each", name="v"))
    assert s.history[-1].code == "v = data.plot(['y', 'c'], kind='vintages')"  # one panel: the releases overlap


# -- saving for publication -------------------------------------------------------------------------------------------


def test_saved_files_have_the_size_asked_for_and_say_what_made_them(data, tmp_path):
    import re

    from PIL import Image

    from pymetrix import __version__
    from pymetrix.figures import SIZES, _size_inches

    fig = data.plot(["c", "i"], title="Spending").line("c", color="C3")
    png = fig.save(str(tmp_path / "f.png"), dpi=200, size="column")
    with Image.open(png) as image:
        width, height = image.size
        info = dict(image.info)
    assert abs(width - 89 / 25.4 * 200) < 1  # 89 mm at 200 dots an inch
    assert abs(height / width - 3.6 / 7.5) < 0.01  # the figure's shape kept
    assert info["Title"] == "Spending" and f"pymetrix {__version__}" in info["Software"]
    assert "Style: line('c', color='C3')" in info["Description"] and "Panel 0: c, i" in info["Description"]
    pdf = fig.save(str(tmp_path / "f.pdf"), size=("120mm", "80mm"))
    raw = pdf.read_bytes()
    box = [float(v) for v in re.search(rb"/MediaBox \[\s*([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)", raw).groups()]
    assert np.allclose(box[2:], [120 / 25.4 * 72, 80 / 25.4 * 72], atol=0.1)  # points
    assert b"/FontFile2" in raw and b"/Type3" not in raw  # TrueType fonts embedded, no Type 3
    assert fig.save(str(tmp_path / "g.pdf"), size=("120mm", "80mm")).read_bytes() == raw  # the same figure, the same file
    svg = fig.save(str(tmp_path / "f.svg"), size="slide").read_text(encoding="utf-8")
    assert 'width="720pt"' in svg and "<text" not in svg and "Spending" in svg  # text drawn as outlines; described
    assert "<text" in fig.save(str(tmp_path / "e.svg"), editable_text=True).read_text(encoding="utf-8")
    clear = fig.save(str(tmp_path / "t.png"), dpi=60, transparent=True)
    with Image.open(clear) as image:
        assert image.convert("RGBA").getpixel((1, 1))[3] == 0
    assert set(SIZES) == {"column", "page", "slide", "half-a4"}
    assert _size_inches(("4in", None), (8.0, 4.0)) == (4.0, 2.0) and _size_inches((5, 3), (8, 4)) == (5.0, 3.0)
    for bad, message in (("poster", "no figure size 'poster'"), (("12 furlongs", None), "not a length"), ((None, 3), "the figure's width"),
                         (("-3cm", None), "positive")):
        with pytest.raises(SpecificationError, match=message):
            _size_inches(bad, (7.5, 3.6))


def test_the_save_dialog_records_the_file_written(data, tmp_path):
    from pymetrix.commands import SaveFigure, Session

    s = Session()
    s.objects["fit"] = Equation("c ~ y").fit(data)
    initial = SaveFigure.for_figure("fit", None)
    assert initial.path == "fit_graph.png" and initial.dpi == 300
    path = str(tmp_path / "fit.pdf")
    command = SaveFigure("fit", path=path, size="custom", width="120mm")
    assert command.validate(s) == []
    written = s.run(command)
    assert written == Path(path) and written.exists()
    assert s.history[-1].code == f"figure_of(fit).save({path!r}, dpi=300, size=('120mm', None))"
    assert SaveFigure("fit", path=str(tmp_path / "f.doc")).validate(s) == ["the file's extension sets the format: .png, .pdf, .svg or .eps"]
    assert SaveFigure("fit", path=path, size="custom").validate(s) == ["give the width, e.g. 120mm"]
    assert SaveFigure("fit", path="").validate(s) == ["choose the file to write"]
    ns = {"fit": s["fit"]}
    exec("from pymetrix import *\n" + s.history[-1].code, ns)  # the step replays to the same file
    assert written.read_bytes() == Path(path).read_bytes()


def test_the_save_button_opens_the_save_dialog(data):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import figure_view

    asked = []
    page = figure_view(data.plot("c"), on_edit=asked.append)
    next(b for b in page.findChildren(QtWidgets.QPushButton) if b.text() == "Save…").click()
    assert asked == ["save"]


# -- coefficient plots and heatmaps ---------------------------------------------------------------------------------


def _two_fits(data):
    short = Equation("dlog(c) ~ dlog(y) + dlog(c)[-1]").fit(data)
    long = Equation("dlog(c) ~ dlog(y) + dlog(c)[-1] + dlog(i)").fit(data)
    return short, long


def test_coefficient_plots_show_each_estimate_and_interval(data):
    from scipy import stats

    from pymetrix import coefficient_figure
    from pymetrix.figures import Estimates

    short, long = _two_fits(data)
    fig = coefficient_figure({"short": short, "long": long}, level=0.9)
    layers = fig.panels[0].layers
    assert [type(l).__name__ + ":" + l.label for l in layers] == ["Estimates:short", "Estimates:long"]
    se = np.sqrt(np.diag(short.params.cov))  # an independent check: estimate ± z(0.95) × standard error
    z = stats.norm.ppf(0.95)
    assert np.allclose(layers[0].upper.to_numpy(), short.params.values + z * se)
    assert np.allclose(layers[0].lower.to_numpy(), short.params.values - z * se)
    ax = _drawn(fig).axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["const", "dlog(y)", "dlog(c)[-1]", "dlog(i)"]
    assert ax.get_ylim()[0] > ax.get_ylim()[1]  # the first coefficient at the top
    markers = {c.get_label(): c.lines[0] for c in ax.containers}  # errorbar labels its container
    assert np.allclose(markers["short"].get_xdata(), short.params.values)
    rows = markers["long"].get_ydata()
    assert np.allclose(rows - np.round(rows), 0.15) and np.allclose(markers["short"].get_ydata() - [0, 1, 2], -0.15)  # side by side
    assert any(l.get_linestyle() == "--" and np.allclose(l.get_xdata(), 0) for l in ax.lines)  # zero marked
    assert ax.get_xlabel() == "estimate, with its 90% interval"
    picked = coefficient_figure(long, terms=["dlog(i)", "dlog(y)"])
    assert list(picked.panels[0].layers[0].estimate.index) == ["dlog(i)", "dlog(y)"]
    with pytest.raises(SpecificationError, match="none of x among"):
        coefficient_figure(long, terms=["x"])
    with pytest.raises(SpecificationError, match="level is between 0 and 1"):
        coefficient_figure(long, level=95)
    with pytest.raises(SpecificationError, match="has no estimates with intervals"):
        coefficient_figure(data)
    styled = fig.line("long", marker="s", color="C4")  # the markers style as a line
    assert {c.get_label(): c.lines[0] for c in _drawn(styled).axes[0].containers}["long"].get_marker() == "s"
    over_time = Figure((Panel((Estimates(data["c"].iloc[-8:] * 0, data["c"].iloc[-8:] * 0 - 1, data["c"].iloc[-8:] * 0 + 1, "b"),)),))
    assert _drawn(over_time).axes[0].pymetrix_freq is not None  # dated estimates run along the dates


def test_heatmaps_colour_each_cell_on_a_scale_that_fits(data):
    from matplotlib.colors import to_hex

    from pymetrix import heatmap_figure
    from pymetrix.figures import Heatmap, _heatmap_scale

    d = data.with_series(gy="dlog(y)", gc="dlog(c)", gi="dlog(i)")
    fig = d.plot(["gy", "gc", "gi"], kind="correlation")
    corr = fig.panels[0].layers[0].values
    frame = d.frame[["gy", "gc", "gi"]].astype(float)
    assert np.allclose(corr.to_numpy(), np.corrcoef(frame.dropna().to_numpy().T))  # independent: NumPy on the complete rows
    fig_ax = _drawn(fig)
    ax = fig_ax.axes[0]
    image = ax.get_images()[0]
    assert image.get_cmap().name == "RdBu_r" and image.get_clim() == (-1.0, 1.0)
    assert len(fig_ax.axes) == 2 and fig_ax.axes[1].get_ylabel() == "correlation"  # the colour bar
    texts = [t.get_text() for t in ax.texts]
    assert len(texts) == 9 and texts[0] == "1.00" and texts[1] == f"{corr.iloc[0, 1]:.2f}"
    assert to_hex(ax.texts[0].get_color()) == "#ffffff"  # dark cell, white text
    assert [t.get_text() for t in ax.get_yticklabels()] == ["gy", "gc", "gi"]
    positive = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["a", "b"], columns=["x", "y"])
    assert _heatmap_scale(Heatmap(positive)) == ("Blues", 1.0, 4.0)
    assert _heatmap_scale(Heatmap(positive - 2)) == ("RdBu_r", -2.0, 2.0)  # both signs: symmetric about zero
    assert _heatmap_scale(Heatmap(positive, center=3.0)) == ("RdBu_r", 1.0, 5.0)
    assert _heatmap_scale(Heatmap(positive, cmap="viridis", vmin=0)) == ("viridis", 0.0, 4.0)
    quiet = _drawn(heatmap_figure(positive, annotate=False)).axes[0]
    assert len(quiet.texts) == 0
    with pytest.raises(SpecificationError, match="at least two series"):
        d.plot(["gy"], kind="correlation")


def test_new_layers_save_and_draw_in_matplotlib_alone(data):
    from pymetrix import coefficient_figure, heatmap_figure
    from pymetrix.saved import _decode_view, _encode_view

    short, long = _two_fits(data)
    table = pd.DataFrame([[0.5, np.nan], [-0.25, 1.0]], index=["shock 1", "shock 2"], columns=["h0", "h1"])
    for fig in (coefficient_figure({"short": short, "long": long}), heatmap_figure(table, label="response")):
        saved = _decode_view(json.loads(json.dumps(_encode_view(fig))))
        assert saved.png() == fig.png()  # a saved result draws the same
        theirs, ours = _run_standalone(fig.matplotlib_code()), _drawn(fig)
        assert len(theirs.axes) == len(ours.axes)  # with the colour bar
        a, b = ours.axes[0], theirs.axes[0]
        assert sorted(t.get_text() for t in a.texts) == sorted(t.get_text() for t in b.texts)
        assert [t.get_text() for t in a.get_yticklabels()] == [t.get_text() for t in b.get_yticklabels()]
        assert [c.get_label() for c in a.containers] == [c.get_label() for c in b.containers]
        for ca, cb in zip(a.containers, b.containers):  # the estimates, where they are drawn
            assert np.allclose(ca.lines[0].get_xdata(), cb.lines[0].get_xdata()) and np.allclose(ca.lines[0].get_ydata(), cb.lines[0].get_ydata())
        if a.get_images():
            assert np.allclose(np.ma.filled(a.get_images()[0].get_array(), np.nan), np.ma.filled(b.get_images()[0].get_array(), np.nan),
                               equal_nan=True)
            assert a.get_images()[0].get_clim() == b.get_images()[0].get_clim()
    assert {"short", "short lower", "short upper"} <= set(coefficient_figure({"short": short}).data().columns)


def test_the_coefficient_plot_dialog(data):
    from pymetrix.commands import CoefficientPlot, PlotSeries, Session, form_spec

    s = Session()
    s.objects["data"] = data.with_series(gy="dlog(y)", gc="dlog(c)")
    s.objects["short"], s.objects["long"] = _two_fits(data)
    field = next(f for f in form_spec(CoefficientPlot) if f.name == "results")
    assert s.choices(field, {}) == ["long", "short"]
    command = CoefficientPlot("short, long", terms="dlog(y), dlog(c)[-1]", level="90", title="Consumption")
    assert command.validate(s) == []
    s.run(command)
    assert s.history[-1].code == ("coefficients = coefficient_figure({'short': short, 'long': long}, "
                                  "terms=['dlog(y)', 'dlog(c)[-1]'], level=0.9, title='Consumption')")
    ns = {"short": s["short"], "long": s["long"]}
    exec("from pymetrix import *\n" + s.history[-1].code, ns)  # replays as plain Python
    assert ns["coefficients"].png() == s["coefficients"].png()
    assert CoefficientPlot("").validate(s) == ["tick the estimated equations to plot"]
    assert "none of x among" in CoefficientPlot("short", terms="x").validate(s)[0]
    s.run(PlotSeries("gy, gc", kind="correlation", name="corr"))
    assert s.history[-1].code == "corr = data.plot(['gy', 'gc'], kind='correlation')"


# -- distributions: histograms with bins, empirical distributions, two-dimensional histograms -------------------------


def test_empirical_distributions_step_through_the_shares(data):
    d = data.with_series(gy="400*dlog(y)", gi="400*dlog(i)")
    fig = d.plot(["gy", "gi"], kind="ecdf")
    ax = _drawn(fig).axes[0]
    lines = {l.get_label(): l for l in ax.lines if l.get_label() in ("gy", "gi")}
    assert set(lines) == {"gy", "gi"} and ax.get_ylabel() == "share at or below"
    values = np.sort(d["gy"].dropna().to_numpy())
    xs, ys = np.asarray(lines["gy"].get_xdata(), float), np.asarray(lines["gy"].get_ydata(), float)
    inside = np.isfinite(xs)
    for x, y in zip(xs[inside], ys[inside]):  # independent: the share of observations at or below each step's value
        assert np.isclose(y, np.searchsorted(values, x, side="right") / len(values)) or np.isclose(
            y, np.searchsorted(values, x, side="left") / len(values))
    assert ys[inside].max() == 1.0
    above = _drawn(Figure((Panel((Ecdf(d["gy"], "gy", complementary=True),)),))).axes[0]
    assert above.get_ylabel() == "share above"
    styled = fig.line("gy", linestyle="--")  # each distribution styles as a line
    assert next(l for l in _drawn(styled).axes[0].lines if l.get_label() == "gy").get_linestyle() == "--"


def test_two_dimensional_histograms_count_each_cell(data):
    d = data.with_series(gy="400*dlog(y)", gc="400*dlog(c)")
    fig = d.plot(["gy", "gc"], kind="hist2d", bins=10)
    drawn = _drawn(fig)
    ax = drawn.axes[0]
    counts = np.ma.filled(ax.collections[0].get_array(), 0).reshape(10, 10)  # rows of y, columns of x
    x, y = d["gy"].to_numpy(float), d["gc"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    expected, _, _ = np.histogram2d(x[ok], y[ok], bins=10)  # independent: NumPy's own counts
    assert np.array_equal(counts, expected.T)
    assert len(drawn.axes) == 2 and drawn.axes[1].get_ylabel() == "observations"  # the colour bar
    assert ax.get_xlabel() == "gy" and ax.get_ylabel() == "gc" and ax.pymetrix_freq is None
    with pytest.raises(SpecificationError, match="needs two series"):
        d.plot(["gy"], kind="hist2d")


def test_histograms_take_bins_and_scatter_plots_have_number_axes(data):
    d = data.with_series(gy="400*dlog(y)", gc="400*dlog(c)")
    fig = d.plot("gy", kind="histogram", bins=12)
    assert fig.panels[0].layers[0].bins == 12
    bars = [p for p in _drawn(fig).axes[0].patches if p.get_height() > 0 or p.get_width() > 0]
    heights, _ = np.histogram(d["gy"].dropna(), bins=12, density=True)
    assert len(bars) == 12 and np.allclose([p.get_height() for p in bars], heights)
    with pytest.raises(SpecificationError, match="bins is for kind='histogram', 'hist2d' or 'hexbin'"):
        d.plot("gy", bins=12)
    with pytest.raises(SpecificationError, match="whole number of at least 1"):
        d.plot("gy", kind="histogram", bins=0)
    scatter = _drawn(d.plot(["gy", "gc"], kind="scatter"))
    scatter.canvas.draw()
    ax = scatter.axes[0]
    assert ax.pymetrix_freq is None and all("Q" not in t.get_text() for t in ax.get_xticklabels())  # numbers, not dates


def test_distribution_kinds_save_draw_alone_and_are_in_the_dialog(data):
    from pymetrix.commands import PlotSeries, Session
    from pymetrix.saved import _decode_view, _encode_view

    d = data.with_series(gy="400*dlog(y)", gc="400*dlog(c)")
    for fig in (d.plot(["gy", "gc"], kind="ecdf"), d.plot(["gy", "gc"], kind="hist2d", bins=8)):
        saved = _decode_view(json.loads(json.dumps(_encode_view(fig))))
        assert saved.png() == fig.png()
        theirs, ours = _run_standalone(fig.matplotlib_code()), _drawn(fig)
        assert len(theirs.axes) == len(ours.axes)
        a, b = ours.axes[0], theirs.axes[0]
        for la, lb in zip([l for l in a.lines if l.get_label() in ("gy", "gc")], [l for l in b.lines if l.get_label() in ("gy", "gc")]):
            assert np.allclose(la.get_xdata(), lb.get_xdata(), equal_nan=True) and np.allclose(la.get_ydata(), lb.get_ydata())
        if a.collections:
            assert np.array_equal(np.ma.filled(a.collections[0].get_array(), 0), np.ma.filled(b.collections[0].get_array(), 0))
    s = Session()
    s.objects["data"] = d
    s.run(PlotSeries("gy, gc", kind="hist2d", bins="15", name="h"))
    assert s.history[-1].code == "h = data.plot(['gy', 'gc'], kind='hist2d', bins=15)"
    s.run(PlotSeries("gy, gc", kind="ecdf", name="e"))
    assert s.history[-1].code == "e = data.plot(['gy', 'gc'], kind='ecdf')"
    assert PlotSeries("gy", kind="hist2d").validate(s) == ["a two-dimensional histogram needs two series: the x series first, then y"]
    assert PlotSeries("gy", kind="histogram", bins="x").validate(s) == ["bins is a whole number of at least 1, e.g. 20"]


# -- the new charts in results and reports ---------------------------------------------------------------------------


def test_equations_have_a_q_q_plot_of_their_residuals(data):
    from scipy import stats

    from pymetrix.figures import QQ, qq_points
    from pymetrix.views import view

    fit = Equation("dlog(c) ~ dlog(y)").fit(data)
    fig = view(fit, "Q-Q plot")
    assert isinstance(fig.panels[0].layers[0], QQ)
    theory, z = qq_points(fit.resid)
    resid = fit.resid.dropna().to_numpy()
    expected = np.sort((resid - resid.mean()) / resid.std(ddof=1))
    (osm, osr), _ = stats.probplot(expected, dist="norm")  # independent: SciPy's probability plot (Filliben positions)
    assert np.allclose(z, expected) and np.allclose(osr, z)
    assert np.allclose(theory, stats.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z)))
    assert np.corrcoef(theory, osm)[0, 1] > 0.999  # the two plotting positions agree on the shape
    ax = _drawn(fig).axes[0]
    points = ax.collections[0].get_offsets()
    assert np.allclose(points[:, 0], theory) and np.allclose(points[:, 1], z)
    assert ax.get_xlabel() == "normal quantiles" and ax.get_ylabel() == "standardised values"
    theirs = _run_standalone(fig.matplotlib_code()).axes[0]
    assert np.allclose(theirs.collections[0].get_offsets(), points)
    with pytest.raises(SpecificationError, match="at least three values"):
        qq_points(pd.Series([1.0, 2.0]))


def test_responses_and_variance_shares_have_heatmaps(data):
    from pymetrix import VAR
    from pymetrix.views import view

    d = data.with_series(gy="400*dlog(y)", gc="400*dlog(c)")
    var = VAR(["gy", "gc"], lags=1).fit(d)
    irf = var.irf(8, draws=0)
    cells = view(irf, "Heatmap").panels[0].layers[0].values
    assert list(cells.index) == ["gy <- gy", "gc <- gy", "gy <- gc", "gc <- gc"]
    assert np.isclose(cells.loc["gc <- gy", 1], irf.values[1, 1, 0]) and np.isclose(cells.loc["gy <- gc", 0], irf.values[0, 0, 1])
    fevd = var.fevd(8) if hasattr(var, "fevd") else None
    if fevd is not None:
        shares = fevd.heatmap().panels[0].layers[0].values
        assert np.allclose(shares.sum(axis=1), 100) and shares.shape == (2, 2)  # each variable's shares add up to 100
        assert np.allclose(fevd.heatmap(1).panels[0].layers[0].values.to_numpy(), 100 * fevd.values[0])
        with pytest.raises(SpecificationError, match="horizon is 1 to"):
            fevd.heatmap(99)
    from pymetrix.figures import _cell_text

    assert _cell_text("{:.2f}", -0.001) == "0.00" and _cell_text("{:.2f}", -0.5) == "-0.50" and _cell_text("{:.0%}", -0.001) == "0%"


def test_any_table_of_numbers_becomes_a_heatmap(data, tmp_path):
    from pymetrix import heatmap_figure
    from pymetrix.commands import Session, TableHeatmap
    from pymetrix.output import Table

    table = Table(["a", "b", "note"], [[1.0, -2.0, "x"], [0.5, 3.0, "y"]], index=["r1", "r2"], title="Numbers")
    fig = heatmap_figure(table)
    cells = fig.panels[0].layers[0].values
    assert list(cells.columns) == ["a", "b"] and list(cells.index) == ["r1", "r2"] and fig.title == "Numbers"  # text columns left out
    with pytest.raises(SpecificationError, match="a table of numbers"):
        heatmap_figure(Table(["note"], [["x"], ["y"]]))
    s = Session()
    s.objects["fit"] = Equation("c ~ y").fit(data)
    s.run(TableHeatmap("fit", "Covariance", label="covariance"))
    assert s.history[-1].code == "fit_heatmap = heatmap_figure(fit, label='covariance', view='Covariance', scale='auto', diagnostics=False)"
    assert s["fit_heatmap"].panels[0].layers[0].values.shape == (2, 2)
    assert TableHeatmap("fit", "Nothing").validate(s)[0].startswith("no view 'Nothing'")
    ns = {"fit": s["fit"]}
    exec("from pymetrix import *\n" + s.history[-1].code, ns)
    assert ns["fit_heatmap"].png() == s["fit_heatmap"].png()


def test_reports_draw_figures_at_their_size_with_alt_text(data, tmp_path):
    import re
    import zipfile

    from pymetrix.report import Report

    fig = data.plot(["c", "i"], title="Spending", footnote="Source: sample data.")
    assert fig.describe(technical=False) == "Spending. c, i. Source: sample data."
    rep = Report("Test", figure_size="column").add(fig, caption="Consumption and investment")
    html = (rep.save(str(tmp_path / "r.html")) or tmp_path / "r.html").read_text(encoding="utf-8")
    assert 'alt="Spending. c, i. Source: sample data."' in html
    md = tmp_path / "r.md"
    rep.save(str(md))
    assert "![Spending. c, i. Source: sample data.](" in md.read_text(encoding="utf-8")
    pytest.importorskip("docx")  # Word reports need python-docx (pymetrix[report])
    rep.save(str(tmp_path / "r.docx"))
    with zipfile.ZipFile(tmp_path / "r.docx") as z:
        document = z.read("word/document.xml").decode("utf-8")
        image = next(n for n in z.namelist() if n.startswith("word/media/"))
        from PIL import Image
        import io

        with Image.open(io.BytesIO(z.read(image))) as png:
            width = png.size[0]
    assert 'descr="Spending. c, i. Source: sample data."' in document
    assert abs(width - 89 / 25.4 * 200) < 2  # drawn 89 mm wide at 200 dots an inch, the size it appears
    cx = int(re.search(r'<wp:extent cx="(\d+)"', document).group(1))
    assert abs(cx / 914400 - 89 / 25.4) < 0.01  # and placed at that width (EMUs)


# -- the house style -------------------------------------------------------------------------------------------------


def test_a_house_style_starts_every_figure_where_it_applies(data):
    from matplotlib.colors import to_hex

    from pymetrix import house_style, set_house_style
    from pymetrix.saved import _decode_view, _encode_view

    house = data.plot(["c", "i"]).line("c", color="C3").set(ylabel="house units").grid(visible=False).annotate("note", xy=("2020Q1", 1.0))
    previous = set_house_style(house)
    try:
        assert previous == () and [c.kind for c in house_style()] == ["line", "set", "grid"]  # the note stays with its chart
        plain = data.plot(["c", "y"])
        assert plain.style == () and plain.style_steps() == []  # the figure's own steps are its own
        ax = _drawn(plain).axes[0]
        assert ax.get_ylabel() == "house units" and to_hex(next(l for l in ax.lines if l.get_label() == "c").get_color()) == to_hex("C3")
        own = data.plot("c").line("c", color="C1").set(ylabel="mine")
        mine = _drawn(own).axes[0]
        assert mine.get_ylabel() == "mine" and to_hex(mine.lines[0].get_color()) == to_hex("C1")  # its own calls come after, and win
        assert _drawn(data.plot("y")).axes[0].get_ylabel() == "house units"  # no series c: the line call is left out, the rest applies
        assert _drawn(plain.with_options(house=False)).axes[0].get_ylabel() == ""  # opted out
        assert "House style: line('c', color='C3')" in plain.describe()
        code = plain.matplotlib_code()
        assert "set_ylabel('house units')" in code or "ylabel='house units'" in code  # the script draws it too
        saved = _decode_view(json.loads(json.dumps(_encode_view(plain))))
        assert len(saved.style) == 3 and saved.house is False  # saved as drawn: it looks the same anywhere
        set_house_style(house, notes=True)
        assert [c.kind for c in house_style()][-1] == "annotate"
    finally:
        set_house_style(None)
    assert house_style() == () and _drawn(data.plot(["c", "y"])).axes[0].get_ylabel() == ""


def test_the_house_style_step_records_replays_and_reopens(data, tmp_path):
    from pymetrix import house_style, set_house_style
    from pymetrix.commands import PlotSeries, Session, StyleFigure, UseHouseStyle, load_project, save_project

    s = Session()
    s.objects["data"] = data
    try:
        s.run(PlotSeries("c", name="chart"))
        s.run(StyleFigure("chart", ylabel="house", name="chart"))
        assert UseHouseStyle("nothing").validate(s) == ["no object named 'nothing'"]
        s.objects["bare"] = data.plot("c")
        assert UseHouseStyle("bare").validate(s) == ["bare has no recorded style to use: style it first (Edit… → Style)"]
        s.run(UseHouseStyle("chart"))
        assert s.history[-1].code == "set_house_style(figure_of(chart))" and house_style() == s["chart"].style
        set_house_style(None)
        ns = {"chart": s["chart"]}
        exec("from pymetrix import *\n" + s.history[-1].code, ns)  # replays as plain Python
        assert house_style() == s["chart"].style
        path = save_project(s, tmp_path / "house.pymx")
        set_house_style(None)
        load_project(path)  # opening the project sets its house style again
        assert [c.props for c in house_style()] == [{"ylabel": "house"}]
        assert _drawn(data.plot("y")).axes[0].get_ylabel() == "house"
        s.run(UseHouseStyle(stop=True))
        assert s.history[-1].code == "set_house_style(None)" and house_style() == ()
        other = Session()
        other.objects["data"] = data
        set_house_style(s["chart"])
        load_project(save_project(other, tmp_path / "plain.pymx"))  # a project without one clears it
        assert house_style() == ()
    finally:
        set_house_style(None)


# -- the hover readout -----------------------------------------------------------------------------------------------


def _pixel(ax, x, y):
    fig = ax.figure
    fig.canvas.draw()
    return ax.transData.transform((x, y))


def test_the_readout_names_what_is_under_the_mouse(data):
    from matplotlib import dates

    from pymetrix import heatmap_figure
    from pymetrix.figures import readout

    d = data.with_series(gy="400*dlog(y)")
    twin = _drawn(d.plot(["gy", "i"], right="i", span=("2020Q1", None)))
    left, right = twin.axes[0], twin.axes[1]
    period = d["gy"].loc["2020Q3":].index[0]
    x = dates.date2num(period.start_time)
    px, py = _pixel(left, x, d["gy"].loc[period])
    assert readout(right, px, py) == f"gy  2020Q3: {d['gy'].loc[period]:.4g}"  # the pointer is over the twin; the line is on the left
    px, py = _pixel(right, x, d["i"].loc[period])
    assert readout(right, px, py) == f"i (right)  2020Q3: {d['i'].loc[period]:.4g}"
    far = readout(left, *_pixel(left, x, left.get_ylim()[1]))
    assert far.startswith("2020Q") and "  " not in far  # no line near: the position, as a period and a value
    bars = _drawn(d.plot(["c"], kind="bar", span=("2024Q1", None))).axes[0]
    patch = bars.containers[0].patches[1]
    centre = (patch.get_x() + patch.get_width() / 2, patch.get_height() / 2)
    assert readout(bars, *_pixel(bars, *centre)) == f"c  2024Q2: {d['c'].loc['2024Q2']:.4g}"
    across = _drawn(d.plot(["c", "i"], kind="barh", span=("2024Q3", None))).axes[0]
    patch = across.containers[0].patches[0]
    text = readout(across, *_pixel(across, patch.get_width() / 2, patch.get_y() + patch.get_height() / 2))
    assert text.endswith(f"{d['c'].loc['2024Q3']:.4g}") and "2024Q3" in text
    heat = _drawn(heatmap_figure(pd.DataFrame([[1.0, 2.5], [3.0, np.nan]], index=["a", "b"], columns=["x", "y"]))).axes[0]
    assert readout(heat, *_pixel(heat, 1, 0)) == "a, y: 2.5"
    assert readout(heat, *_pixel(heat, 1, 1)) == "b, y: no value"


def test_the_figure_window_shows_the_readout(data):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import figure_view

    page = figure_view(data.plot("c", span=("2024Q1", None)))
    canvas = page.toolbar.canvas
    canvas.draw()
    ax = canvas.figure.axes[0]
    line = next(l for l in ax.lines if l.get_label() == "c")
    px, py = ax.transData.transform((line.get_xdata(orig=False)[2], line.get_ydata()[2]))

    class Move:  # a mouse move, as matplotlib reports it
        inaxes, x, y = ax, px, py

    page.hover(Move)
    assert page.readout.text() == f"c  2024Q3: {data['c'].loc['2024Q3']:.4g}"
    Move.inaxes = None
    page.hover(Move)
    assert page.readout.text() == " "


# -- mixed frequencies in one panel ---------------------------------------------------------------------------------


def _mixed():
    monthly = pd.Series(np.sin(np.arange(36) / 4), index=pd.period_range("2022-01", periods=36, freq="M"), name="monthly")
    quarterly = pd.Series(np.arange(12) * 0.1 - 0.5, index=pd.period_range("2022Q1", periods=12, freq="Q"), name="quarterly")
    return monthly, quarterly


def test_a_panel_mixing_frequencies_puts_each_value_over_its_period():
    from matplotlib import dates

    from pymetrix.figures import _clip

    monthly, quarterly = _mixed()
    for layers in ((Bars(quarterly, "quarterly"), Line(monthly, "monthly")), (Line(monthly, "monthly"), Bars(quarterly, "quarterly"))):
        fig = Figure((Panel(layers),))
        ax = _drawn(fig).axes[0]
        assert "Month" in type(ax.pymetrix_freq).__name__  # the finest frequency, whichever layer comes first
        bar = ax.containers[0].patches[0]
        centre = dates.num2date(bar.get_x() + bar.get_width() / 2).replace(tzinfo=None)
        q1 = pd.Period("2022Q1")
        assert abs(centre - (q1.start_time + (q1.end_time - q1.start_time) / 2)) < pd.Timedelta(hours=1)  # the middle of 2022Q1
        mine = next(l for l in ax.lines if l.get_label() == "monthly")
        assert dates.num2date(mine.get_xdata(orig=False)[0]).date() == pd.Timestamp("2022-01-16").date()  # the middle of January
        theirs = _run_standalone(fig.matplotlib_code()).axes[0]  # the script places them the same way
        other = next(l for l in theirs.lines if l.get_label() == "monthly")
        assert np.allclose(mine.get_xdata(orig=False), other.get_xdata(orig=False))
        assert np.isclose(theirs.containers[0].patches[0].get_x(), bar.get_x())
    single = _drawn(Figure((Panel((Line(monthly, "monthly"),)),))).axes[0]
    assert dates.num2date(single.lines[0].get_xdata(orig=False)[0]).date() == pd.Timestamp("2022-01-01").date()  # one frequency: the starts
    clipped = _clip(monthly, ("2023Q1", "2023Q4"))  # quarters as a monthly series' span: January to December
    assert (str(clipped.index[0]), str(clipped.index[-1]), len(clipped)) == ("2023-01", "2023-12", 12)
    assert len(_clip(quarterly, ("2023-02", "2023-11"))) == 4  # months as a quarterly series' span: the quarters they fall in
    data = Figure((Panel((Bars(quarterly, "quarterly"), Line(monthly, "monthly"))),), span=("2023Q1", "2023Q4")).data()
    assert data["monthly"].notna().sum() == 12 and data["quarterly"].notna().sum() == 4


# -- from the review of the charting code: each bug found, kept fixed ------------------------------------------------


def _two_panels(data):
    return data.plot(["c", "i"], layout="each", span=("2023Q1", None))


def test_the_script_matches_pymetrix_for_calls_on_every_panel(data):
    two = _two_panels(data)
    fig = two.ticks(ax="all", axis="y", format="{x:.1f}").annotate("x", xy=("2023Q3", 300), ax="all").legend(ax="all", visible=False)
    ours, theirs = _drawn(fig), _run_standalone(fig.matplotlib_code())
    for a, b in zip(ours.axes, theirs.axes):  # each panel gets the call, not only the first
        assert type(a.yaxis.get_major_formatter()).__name__ == type(b.yaxis.get_major_formatter()).__name__ == "StrMethodFormatter"
        assert [t.get_text() for t in a.texts if t.get_text() == "x"] == [t.get_text() for t in b.texts if t.get_text() == "x"] == ["x"]
        assert (a.get_legend() is None) == (b.get_legend() is None)


def test_the_script_matches_pymetrix_for_legends(data):
    twin = data.plot(["c", "i"], right="i", span=("2023Q1", None)).line("c", color="red").legend(ax=0, frameon=True, loc="upper left")
    ours, theirs = _drawn(twin).axes[0].get_legend(), _run_standalone(twin.matplotlib_code()).axes[0].get_legend()
    assert [t.get_text() for t in ours.get_texts()] == [t.get_text() for t in theirs.get_texts()] == ["c", "i (right)"]
    from matplotlib.colors import to_hex

    assert to_hex(ours.legend_handles[0].get_color()) == to_hex(theirs.legend_handles[0].get_color()) == "#ff0000"  # styled first
    assert ours.get_frame_on() and theirs.get_frame_on()


def test_the_script_matches_pymetrix_for_horizontal_panels():
    values = pd.Series([1.0, 20.0, 300.0], index=["x", "y", "z"])
    fig = Figure((Panel((Bars(values, "u", horizontal=True), Bars(values * 2, "v", horizontal=True))),), log=True, ylim=(0.5, 1000),
                 reference="10")
    ours, theirs = _drawn(fig).axes[0], _run_standalone(fig.matplotlib_code()).axes[0]
    assert ours.yaxis_inverted() and theirs.yaxis_inverted()  # flipped once, with two series
    assert ours.get_xscale() == theirs.get_xscale() == "log" and ours.get_yscale() == theirs.get_yscale() == "linear"
    assert ours.get_xlim() == theirs.get_xlim() == (0.5, 1000)
    assert [t.get_text() for t in ours.get_yticklabels()] == [t.get_text() for t in theirs.get_yticklabels()] == ["x", "y", "z"]
    assert any(np.allclose(l.get_xdata(), 10) for l in theirs.lines)  # the reference value across, as pymetrix draws it


def test_the_script_matches_pymetrix_at_the_edges(data):
    s = data["c"].iloc[-8:]
    shaded = Figure((Panel((Bars(s, "a"),)),), shade=f"{s.index[2]}:{s.index[3]}")  # shading without a line
    count = lambda ax: sum(1 for p in ax.patches if np.isclose(p.get_alpha() or 0, 0.18))  # noqa: E731
    assert count(_drawn(shaded).axes[0]) == count(_run_standalone(shaded.matplotlib_code()).axes[0]) == 1
    shared = Figure((Panel((Band(s - 1, s + 1, "a"), Line(s, "a"))),))  # a band and a line with one name
    assert len(_run_standalone(shared.matplotlib_code()).axes[0].lines) == 1
    empty = pd.Series([np.nan, np.nan], index=s.index[:2])
    for layer in (Histogram(empty, label="h"), Ecdf(empty, "e")):  # nothing to draw: drawn as nothing, by both
        fig = Figure((Panel((layer,)),))
        _drawn(fig)
        _run_standalone(fig.matplotlib_code())
    from pymetrix.figures import QQ

    flat = Figure((Panel((QQ(pd.Series([3.0] * 5), "q"),)),))
    assert np.allclose(_run_standalone(flat.matplotlib_code()).axes[0].collections[0].get_offsets()[:, 1], 0)
    scatter = Figure((Panel((Scatter(s, s * 2, label="obs"),)),))
    assert _run_standalone(scatter.matplotlib_code()).axes[0].get_legend_handles_labels()[1] == ["obs"]


def test_marks_outside_the_dates_shown_are_left_out(data):
    from matplotlib import dates

    s = data["c"].loc["2000Q1":"2001Q4"]
    fig = Figure((Panel((Line(s, "c"),)),), events="1990Q1=old")  # long before the data
    lo, _ = _drawn(fig).axes[0].get_xlim()
    assert dates.num2date(lo).year == 1999  # not widened back to 1990
    assert not [t for t in _drawn(fig).axes[0].texts if "old" in t.get_text()]
    se = pd.Series(1.0, index=s.index)
    fan = Fan.normal(s, se.where(s.index >= pd.Period("2000Q3")))
    clipped = Figure((Panel((fan,)),), span=("2001Q2", None))
    assert dates.num2date(_drawn(clipped).axes[0].get_xlim()[0]).year == 2001  # the fan's start line, outside the span
    assert not [l for l in _drawn(clipped).axes[0].lines if l.get_linestyle() == "--"]


def test_notes_on_annual_and_number_axes(data):
    from matplotlib import dates

    from pymetrix.commands import AnnotateFigure, Session

    annual = data.frame["c"].groupby(data.frame.index.year).mean()
    annual.index = pd.PeriodIndex(annual.index.astype(str), freq="Y")
    s = Session()
    s.objects["afig"] = Figure((Panel((Line(annual, "c"),)),))
    s.run(AnnotateFigure("afig", what="axvline", x="2007"))
    assert s.history[-1].code.endswith(".axvline('2007')")
    line = next(l for l in _drawn(s["afig_figure"]).axes[0].lines if l.get_label() != "c")
    assert dates.num2date(line.get_xdata()[0]).year == 2007  # the year, not day 2007 of 1970
    horizons = Figure((Panel((Line(pd.Series([0.0, 1.0, 0.5, 0.2]), "r"),)),))
    marked = _drawn(horizons.axvline("2"))  # text on a number axis is the number
    assert np.allclose([l for l in marked.axes[0].lines if l.get_label() != "r"][0].get_xdata(), 2)
    band = _drawn(Figure((Panel((Line(pd.Series(1.0, index=pd.period_range("2013-01", periods=24, freq="M")), "m"),)),))
                  .axvspan("2013Q4", "2013Q4")).axes[0]
    span = band.patches[0]  # a Rectangle (or, on matplotlib 3.8, a Polygon)
    right = span.get_x() + span.get_width() if hasattr(span, "get_width") else np.asarray(span.get_xy())[:, 0].max()
    assert dates.num2date(right).date() == pd.Timestamp("2014-01-01").date()  # through the end of December, on a monthly axis


def test_dialogs_keep_values_and_refuse_bad_steps(data):
    from pymetrix.commands import ArrangeStyles, FigureOptions, Session, StyleFigure

    s = Session()
    s.objects["fig"] = Figure((Panel((Line(data["c"], "c"),)), Panel((Line(data["i"], "i"),))),
                              reference="2.123456=t", ylim=(0.0012345, 9.87654321)).line("c", color="red").set(ylabel="a").grid(visible=False)
    assert ArrangeStyles("fig", keep="0, 5", steps=3).validate(s) == ["no style step 5: this figure has 3 (0 to 2)"]
    options = FigureOptions.for_figure("fig", s["fig"], is_figure=True)
    assert options.reference == "2.123456=t" and options.lowest == "0.0012345" and options.highest == "9.87654321"  # as they are
    natural = s["fig"].draw().get_size_inches()
    s.run(StyleFigure("fig", width="9", name="wide"))
    assert s.history[-1].code.endswith(".layout(figwidth=9)")
    assert tuple(s["wide"].draw().get_size_inches()) == (9.0, natural[1])  # the height as it was
    assert tuple(_run_standalone(s["wide"].matplotlib_code()).get_size_inches()) == (9.0, natural[1])


def test_copied_styles_reach_series_on_the_right_axis_and_heatmaps_keep_their_columns(data):
    from pymetrix import heatmap_figure
    from pymetrix.saved import _decode_view, _encode_view

    twin = data.plot(["c", "i"], right="i")
    assert len(twin.with_style_of(data.plot("i").line("i", color="red")).style) == 1  # 'i' is drawn, on the right
    fig = heatmap_figure(pd.DataFrame([[1.0, 2.0]], index=["a"], columns=[1, 2]))
    assert list(_decode_view(json.loads(json.dumps(_encode_view(fig)))).panels[0].layers[0].values.columns) == [1, 2]
    histogram = Figure((Panel((Histogram(data["c"].iloc[-8:], label="h"),)),), span=(str(data["c"].index[-4]), None))
    assert len(histogram.data()) == 4  # a span limits distributions too
