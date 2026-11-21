"""Arrows, text boxes and change markers (Figure.change, the Add a note dialog's arrow and text box, Mark a change),
and editing what a figure's recorded steps added by dragging it on the figure (off-screen Qt, matplotlib's own
mouse and key events): each edit is the recorded step's new arguments, replays identically, and leaves no handles in
exports."""

import os

import numpy as np
import pandas as pd
import pytest

from pymetrix.errors import SpecificationError

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def data():
    from pymetrix import Dataset

    idx = pd.period_range("2015Q1", periods=24, freq="Q", name="time")
    level = 100 + np.arange(24) * 2.0
    rate = 5 + np.sin(np.arange(24) / 3)
    return Dataset.from_frame(pd.DataFrame({"c": level, "u": rate}, index=idx))


# -- change markers ---------------------------------------------------------------------------------------------------


def test_change_label_values(data):
    from pymetrix.figures import change_between

    fig = data.plot(["c"])
    c = change_between(fig.panels[0], "c", "2015Q1", "2020Q1", annualise=True)
    assert c["y0"] == 100 and c["y1"] == 140 and c["change"] == 40 and c["percent"] == pytest.approx(40.0)
    assert c["label"].splitlines()[0] == "+40.0 (+40.0%)"
    assert c["label"].splitlines()[1] == f"{(1.4 ** (1 / 5) - 1) * 100:+.1f}% a year, +8.00 a year"
    assert change_between(fig.panels[0], "c", "2015Q1", "2016Q1", show="percent")["label"] == "+8.0%"
    assert change_between(fig.panels[0], "c", "2015Q1", "2016Q1", show="change", units="pp")["label"] == "+8.00 pp"
    rates = data.plot(["u"]).with_options()
    from dataclasses import replace

    rate_panel = replace(rates.panels[0], ylabel="% a year")  # a rate: the change in percentage points
    c = change_between(rate_panel, "u", "2015Q1", "2016Q1", show="change")
    assert c["label"] == f"{data['u'].loc['2016Q1'] - data['u'].loc['2015Q1']:+.2f} pp"
    with pytest.raises(SpecificationError):
        change_between(fig.panels[0], "c", "2015Q1", "2015Q1x")


def test_change_is_recomputed_from_the_data_and_written_as_matplotlib(data):
    fig = data.plot(["c"]).change("c", "2015Q1", "2020Q1")
    assert fig.style_steps() == ["change(series='c', start='2015Q1', end='2020Q1', show='both', units='auto', ax=None)"]
    drawn = fig.draw()
    assert "+40.0 (+40.0%)" in [t.get_text() for t in drawn.axes[0].texts]
    doubled = data.with_columns(c=data["c"] * 2) if hasattr(data, "with_columns") else None
    from pymetrix import Dataset

    doubled = Dataset.from_frame(data.frame.assign(c=data.frame["c"] * 2))
    again = doubled.plot(["c"]).with_style_of(fig, notes=True)
    assert "+80.0 (+40.0%)" in [t.get_text() for t in again.draw().axes[0].texts]  # the same step, the new data
    code = fig.matplotlib_code()
    assert "arrowstyle': '<->'" in code and "'+40.0 (+40.0%)'" in code
    import matplotlib

    matplotlib.use("Agg")
    namespace = {"__name__": "script"}
    exec(code, namespace)  # noqa: S102 - the generated script runs on its own
    assert "+40.0 (+40.0%)" in [t.get_text() for t in namespace["ax0"].texts]
    with pytest.raises(SpecificationError):
        data.plot(["c"]).change("nope", "2015Q1", "2016Q1").draw()


def test_arrow_and_text_box_dialogs_record_steps_that_replay(data):
    from pymetrix.commands import AnnotateFigure, MarkChange, PlotSeries, Session

    session = Session()
    session.objects["data"] = data
    session.run(PlotSeries(series="c, u", name="fig"))
    session.run(AnnotateFigure("fig", what="arrow", text="peak", x="2019Q1", y="130", from_x="2016Q1", from_y="150",
                               arrowstyle="-|>", connection="curved", curvature="-0.2", width="2", head="15", color="C3", name="fig"))
    session.run(AnnotateFigure("fig", what="textbox", text="Note", x="2018Q1", y="120", boxstyle="sawtooth", facecolor="#ffeecc",
                               pad="0.5", ha="center", va="top", fontsize="9", name="fig"))
    session.run(MarkChange("fig", series="c", start="2015Q1", end="2020Q1", show="percent", name="fig"))
    fig = session["fig"]
    kinds = [c.kind for c in fig.style]
    assert kinds == ["annotate", "text", "change"]
    arrow, box, change = fig.style
    assert arrow.props["arrowprops"] == {"arrowstyle": "-|>", "connectionstyle": "arc3,rad=-0.2", "color": "C3",
                                         "linewidth": 2, "mutation_scale": 15}
    assert arrow.props["xy"] == ("2019Q1", 130) and arrow.props["xytext"] == ("2016Q1", 150)
    assert box.props["bbox"] == {"boxstyle": "sawtooth,pad=0.5", "facecolor": "#ffeecc", "edgecolor": "#888888"}
    assert box.props["ha"] == "center" and box.props["va"] == "top" and box.props["fontsize"] == 9
    replayed = Session()
    replayed.objects["data"] = data
    for record in session.history:
        replayed.run(record.command)
    assert replayed["fig"].style == fig.style and replayed["fig"].png() == fig.png()
    code = fig.matplotlib_code()
    assert "arc3,rad=-0.2" in code and "sawtooth,pad=0.5" in code
    script = session.script()
    assert ".change('c', '2015Q1', '2020Q1', show='percent')" in script


# -- direct editing ------------------------------------------------------------------------------------------------


def _press(canvas, x, y, dbl=False):
    from matplotlib.backend_bases import MouseEvent

    MouseEvent("button_press_event", canvas, x, y, button=1, dblclick=dbl)._process()


def _move(canvas, x, y, key=None):
    from matplotlib.backend_bases import MouseEvent

    MouseEvent("motion_notify_event", canvas, x, y, button=1, key=key)._process()


def _release(canvas, x, y):
    from matplotlib.backend_bases import MouseEvent

    MouseEvent("button_release_event", canvas, x, y, button=1)._process()


def _key(canvas, key):
    from matplotlib.backend_bases import KeyEvent

    KeyEvent("key_press_event", canvas, key)._process()


class _Recorder:
    """What a figure window hands back: each edit applied as the app's steps apply it (restyle, without_style)."""

    def __init__(self, figure):
        self.figure, self.edits = figure, []

    def restyle(self, step, props):
        self.edits.append(("restyle", step, props))
        self.figure = self.figure.restyle(step, **props)

    def remove(self, step):
        self.edits.append(("remove", step))
        self.figure = self.figure.without_style(step)


def _window(figure, recorder, texts=None):
    _app()
    from pymetrix.ui.qt import figure_view

    page = figure_view(figure, on_restyle=recorder.restyle, on_remove=recorder.remove)
    if texts is not None:
        page.editor.on_text = lambda step, key, old: recorder.restyle(step, {key: texts})
    page.resize(800, 600)
    page.show()
    _app().processEvents()
    page.canvas.draw()
    return page


def _pixel(ax, x, y, freq=True):
    from pymetrix.figures import _x_number

    return ax.transData.transform((_x_number(x, "Q" if freq else None), y))


def _styled(data):
    return (data.plot(["c"]).annotate("note", xy=("2017Q1", 110), xytext=("2016Q1", 135))
            .text("2019Q1", 115, "words", bbox={"boxstyle": "round"})
            .axvline("2020Q1").axvspan("2016Q3", "2017Q2", alpha=0.2).change("c", "2015Q1", "2018Q1"))


def test_dragging_a_note_moves_it_and_records_the_step(data):
    figure = _styled(data)
    rec = _Recorder(figure)
    page = _window(figure, rec)
    ax = page.canvas.figure.axes[0]
    tx, ty = _pixel(ax, "2019Q1", 115)
    _press(page.canvas, tx + 2, ty + 2)
    assert page.editor.selected is not None and page.editor.selected.call.kind == "text"
    assert page.editor.markers and page.identify.text() == ""  # handles shown; the identify box left alone
    target = _pixel(ax, "2020Q3", 125)
    _move(page.canvas, (tx + target[0]) / 2, (ty + target[1]) / 2)
    _move(page.canvas, target[0] + 2, target[1] + 2)
    _release(page.canvas, target[0] + 2, target[1] + 2)
    (kind, step, props), = rec.edits
    assert kind == "restyle" and step == 1 and props["x"] == "2020Q3" and props["y"] == pytest.approx(125, abs=1)
    replayed = rec.figure
    assert replayed.style[1].props["x"] == "2020Q3" and replayed.style[1].props["s"] == "words"
    assert "pymetrix-edit" not in str(replayed.png())  # nothing of the handles in an export
    gids = [a.get_gid() for a in replayed.draw().axes[0].get_children()]
    assert "pymetrix-edit" not in gids


def test_dragging_an_arrow_head_and_shift_constrains(data):
    figure = _styled(data)
    rec = _Recorder(figure)
    page = _window(figure, rec)
    ax = page.canvas.figure.axes[0]
    tx, ty = _pixel(ax, "2016Q1", 135)  # the note's words: select it
    _press(page.canvas, tx + 3, ty + 3)
    _release(page.canvas, tx + 3, ty + 3)
    assert page.editor.selected.call.kind == "annotate" and set(page.editor.selected.points) == {"head", "tail"}
    hx, hy = _pixel(ax, "2017Q1", 110)
    _press(page.canvas, hx, hy)  # the head's handle
    to = _pixel(ax, "2018Q1", 100)
    _move(page.canvas, to[0], to[1] + 3, key="shift")  # mostly across: kept level
    _release(page.canvas, to[0], to[1] + 3)
    (kind, step, props), = rec.edits
    assert step == 0 and props["xy"][0] == "2018Q1" and props["xy"][1] == pytest.approx(110, abs=0.01)
    assert props["xytext"] == ("2016Q1", 135)


def test_escape_cancels_a_drag_and_delete_removes(data):
    figure = _styled(data)
    rec = _Recorder(figure)
    page = _window(figure, rec)
    ax = page.canvas.figure.axes[0]
    tx, ty = _pixel(ax, "2019Q1", 115)
    _press(page.canvas, tx + 2, ty + 2)
    _move(page.canvas, tx + 80, ty + 40)
    _key(page.canvas, "escape")
    _release(page.canvas, tx + 80, ty + 40)
    assert rec.edits == [] and page.editor.selected.points["at"] == page.editor.selected.start["at"]
    _key(page.canvas, "delete")
    assert rec.edits == [("remove", 1)] and [c.kind for c in rec.figure.style] == ["annotate", "axvline", "axvspan", "change"]


def test_dragging_lines_bands_and_change_ends(data):
    figure = _styled(data)
    rec = _Recorder(figure)
    page = _window(figure, rec)
    ax = page.canvas.figure.axes[0]
    y_mid = sum(ax.get_ylim()) / 2
    lx, ly = _pixel(ax, "2020Q1", y_mid)
    _press(page.canvas, lx, ly)
    to = _pixel(ax, "2021Q2", y_mid)
    _move(page.canvas, *to)
    _release(page.canvas, *to)
    assert rec.edits[-1] == ("restyle", 2, {"x": "2021Q2"})
    page = _window(rec.figure, rec)  # the window redrawn with the new step, as the app does
    ax = page.canvas.figure.axes[0]
    y_mid = sum(ax.get_ylim()) / 2
    ex, ey = _pixel(ax, "2016Q3", y_mid)  # the band's left edge: select the band, then drag its edge
    _press(page.canvas, ex + 6, ey)
    _release(page.canvas, ex + 6, ey)
    assert page.editor.selected.call.kind == "axvspan"
    _press(page.canvas, ex, ey)
    to = _pixel(ax, "2016Q1", y_mid)
    _move(page.canvas, *to)
    _release(page.canvas, *to)
    assert rec.edits[-1] == ("restyle", 3, {"xmin": "2016Q1", "xmax": "2017Q2"})
    page = _window(rec.figure, rec)
    ax = page.canvas.figure.axes[0]
    sx, sy = _pixel(ax, "2018Q1", 124)  # the change marker's end point: select, then drag it along the series
    _press(page.canvas, sx, sy)
    _release(page.canvas, sx, sy)
    assert page.editor.selected.call.kind == "change"
    _press(page.canvas, sx, sy)
    to = _pixel(ax, "2019Q2", 140)  # snaps to the data point at 2019Q2, whatever the height
    _move(page.canvas, *to)
    _release(page.canvas, *to)
    assert rec.edits[-1] == ("restyle", 4, {"start": "2015Q1", "end": "2019Q2"})
    texts = [t.get_text() for t in rec.figure.draw().axes[0].texts]
    assert "+34.0 (+34.0%)" in texts  # the label recomputed from the data


def test_double_click_edits_the_words(data):
    figure = _styled(data)
    rec = _Recorder(figure)
    page = _window(figure, rec, texts="new words")
    ax = page.canvas.figure.axes[0]
    tx, ty = _pixel(ax, "2019Q1", 115)
    _press(page.canvas, tx + 2, ty + 2, dbl=True)
    assert rec.edits == [("restyle", 1, {"s": "new words"})]
    assert rec.figure.style[1].props["s"] == "new words"


# -- drawing arrows as in a spreadsheet's chart -----------------------------------------------------------------------


def _arrow_only(data):
    return data.plot(["c"]).annotate("", xy=("2019Q1", 120), xytext=("2016Q1", 135), arrowprops={"arrowstyle": "->"})


def test_an_arrow_without_words_is_picked_by_its_line_dragged_and_formatted(data):
    figure = _arrow_only(data)
    rec = _Recorder(figure)
    page = _window(figure, rec)
    formats = []
    page.editor.on_format = formats.append
    ax = page.canvas.figure.axes[0]
    (hx, hy), (tx, ty) = _pixel(ax, "2019Q1", 120), _pixel(ax, "2016Q1", 135)
    mx, my = (hx + tx) / 2, (hy + ty) / 2
    _press(page.canvas, mx, my + 2)  # on the line, not on any words
    assert page.editor.selected is not None and set(page.editor.selected.points) == {"head", "tail"}
    to = _pixel(ax, "2018Q1", 130)
    _move(page.canvas, (mx + to[0]) / 2, my)
    _move(page.canvas, to[0], my)  # the whole arrow moves along
    _release(page.canvas, to[0], my)
    (kind, step, props), = rec.edits
    assert kind == "restyle" and step == 0 and props["xy"][1] == pytest.approx(120, abs=0.5)
    assert props["xy"][0] > "2019Q1" and props["xytext"][0] > "2016Q1"
    page = _window(rec.figure, rec)
    page.editor.on_format = formats.append
    ax = page.canvas.figure.axes[0]
    art = next(a for a in ax.texts if getattr(a, "arrow_patch", None) is not None)
    (hx, hy), (tx, ty) = ax.transData.transform(art.xy), ax.transData.transform(art.xyann)
    _press(page.canvas, (hx + tx) / 2, (hy + ty) / 2, dbl=True)
    assert formats == [0]  # a double-click on an arrow opens its format
    far = ax.transData.transform((ax.get_xlim()[0], ax.get_ylim()[0]))
    page.editor.select(None)
    _press(page.canvas, far[0] + 3, far[1] + 3)
    assert page.editor.selected is None  # away from the line: nothing picked


def test_the_arrow_tool_draws_by_dragging_with_a_preview(data):
    _app()
    from pymetrix.ui.qt import figure_view

    figure = data.plot(["c"])
    picks = []
    page = figure_view(figure, on_pick=lambda *a: picks.append(a), on_restyle=lambda *a: None, on_remove=lambda *a: None)
    page.resize(800, 600)
    page.show()
    _app().processEvents()
    page.canvas.draw()
    ax = page.canvas.figure.axes[0]
    page.arrow_action.trigger()
    assert page.arrow_action.isChecked() and page.picking["kind"] == "arrow"
    a, b = _pixel(ax, "2016Q1", 110), _pixel(ax, "2019Q1", 130)
    _press(page.canvas, *a)
    _move(page.canvas, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    rubber = page.picking["rubber"]
    assert rubber is not None and rubber.get_gid() == "pymetrix-edit"  # drawn as it is dragged, never recorded
    _move(page.canvas, *b)
    _release(page.canvas, *b)
    (kind, panel, first, second, _), = picks
    assert kind == "arrow" and panel == 0 and first[0] == "2016Q1" and second[0] == "2019Q1"
    assert first[1] == pytest.approx(110, abs=0.5) and second[1] == pytest.approx(130, abs=0.5)
    assert page.picking is None and not page.arrow_action.isChecked() and rubber not in ax.texts
    page.arrow_action.trigger()  # two clicks still work
    _press(page.canvas, *a)
    _release(page.canvas, *a)
    _move(page.canvas, *b)
    _press(page.canvas, *b)
    assert len(picks) == 2 and picks[1][3][0] == "2019Q1"
    page.arrow_action.trigger()
    _press(page.canvas, *a)
    _key(page.canvas, "escape")  # Esc stops, the preview gone
    assert page.picking is None and not page.arrow_action.isChecked()


def test_arrows_drawn_in_the_app_are_added_at_once_and_kept(data, monkeypatch):
    _app()
    from pymetrix.commands import FitEquation, PlotSeries, Session
    from pymetrix.ui import plugins_qt
    from pymetrix.ui.qt import main_window
    from pymetrix.views import figure_of

    monkeypatch.setattr(plugins_qt, "ask_new_plugins", lambda *a, **k: None)
    session = Session()
    session.objects["data"] = data
    window = main_window(session)
    session.run(PlotSeries(series="c", name="fig"))
    for first, second in ((("2016Q1", 110), ("2018Q1", 130)), (("2019Q1", 120), ("2020Q1", 140))):
        window.figure_edit("fig", "Figure", session["fig"], "pick", "arrow", 0, first, second, "")
    calls = session["fig"].style
    assert [c.kind for c in calls] == ["annotate", "annotate"] and calls[1].props["xy"] == ("2020Q1", 140.0)
    assert type(session.history[-1].command).__name__ == "AnnotateFigure"  # added at once, no dialog
    session.run(FitEquation("c ~ u", name="fit"))
    graph = figure_of(session["fit"], "Graph")  # on a result's graph, a second arrow joins the first
    window.figure_edit("fit", "Graph", graph, "pick", "arrow", 0, ("2016Q1", 1), ("2018Q1", 2), "")
    window.figure_edit("fit", "Graph", graph, "pick", "arrow", 0, ("2017Q1", 1), ("2019Q1", 2), "")
    assert [c.kind for c in session["fit_graph"].style] == ["annotate", "annotate"]
    session.dirty = False
    window.close()
