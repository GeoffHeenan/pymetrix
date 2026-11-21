"""The Format… form for what a figure's recorded steps added (arrows, notes, text boxes, lines, bands, change markers)
and the figure's right-click menu: the form's fields come from the step, its changes are the step's new arguments
(drawn as matplotlib draws them), and the menu offers what is under the mouse."""

import os

import numpy as np
import pandas as pd
import pytest

from pymetrix.ui.chart_format import fields, props

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def data():
    from pymetrix import Dataset

    idx = pd.period_range("2015Q1", periods=24, freq="Q", name="time")
    return Dataset.from_frame(pd.DataFrame({"c": 100 + np.arange(24) * 2.0, "u": 5 + np.sin(np.arange(24) / 3)}, index=idx))


def _arrow(data):
    return data.plot(["c"]).annotate("", xy=("2019Q1", 120), xytext=("2016Q1", 135), arrowprops={"arrowstyle": "->"})


def test_an_arrows_form_and_its_changes(data):
    fig = _arrow(data)
    call = fig.style[0]
    form = {f.key: f for f in fields(call)}
    assert set(form) == {"text", "color", "fontsize", "arrowstyle", "linewidth", "mutation_scale", "curvature"}
    assert form["arrowstyle"].value == "->" and form["curvature"].value == 0.0
    assert props(call, {k: f.value if f.value is not None else "" for k, f in form.items()}) == {}  # unchanged: nothing
    changed = props(call, {"text": "", "color": "#cc0000", "fontsize": "", "arrowstyle": "-|>", "linewidth": "2.5",
                           "mutation_scale": "20", "curvature": "0.3"})
    assert changed == {"color": "#cc0000", "arrowprops": {"arrowstyle": "-|>", "linewidth": 2.5, "mutation_scale": 20,
                                                         "connectionstyle": "arc3,rad=0.3", "color": "#cc0000"}}
    drawn = fig.restyle(0, **changed).draw().axes[0]
    patch = next(a for a in drawn.texts if getattr(a, "arrow_patch", None) is not None).arrow_patch
    assert patch.get_linewidth() == pytest.approx(2.5) and patch.get_mutation_scale() == pytest.approx(20)
    curved = fig.restyle(0, **changed)
    assert {f.key: f.value for f in fields(curved.style[0])}["curvature"] == pytest.approx(0.3)  # read back
    straight = props(curved.style[0], {"curvature": "0", "arrowstyle": "-|>"})
    assert "connectionstyle" not in straight["arrowprops"]
    with pytest.raises(ValueError):
        props(call, {"linewidth": "thick", "arrowstyle": "->"})


def test_text_box_line_band_and_change_forms(data):
    fig = (data.plot(["c"]).text("2019Q1", 115, "words", bbox={"boxstyle": "round,pad=0.3", "facecolor": "white"})
           .axvline("2020Q1").axvspan("2016Q3", "2017Q2", alpha=0.2).change("c", "2015Q1", "2018Q1"))
    text, line, band, change = fig.style
    assert props(text, {"s": "new", "box": "square", "facecolor": "#ffffcc"}) == {
        "s": "new", "bbox": {"boxstyle": "square,pad=0.3", "facecolor": "#ffffcc", "edgecolor": "#888888"}}
    assert props(text, {"box": "none"}) == {"bbox": None}
    assert props(line, {"color": "red", "linewidth": "2", "linestyle": "dashed"}) == {"color": "red", "linewidth": 2,
                                                                                      "linestyle": "dashed"}
    assert props(band, {"alpha": "0.5"}) == {"alpha": 0.5}
    with pytest.raises(ValueError, match="opacity"):
        props(band, {"alpha": "2"})
    assert props(change, {"style": "bracket", "show": "percent"}) == {"style": "bracket", "show": "percent"}
    restyled = fig.restyle(0, **props(text, {"box": "none"})).restyle(1, linestyle="dashed")
    restyled.draw()  # each change is something matplotlib draws


def _window(figure, **callbacks):
    _app()
    from pymetrix.ui.qt import figure_view

    page = figure_view(figure, on_edit=lambda *a: None, **callbacks)
    page.resize(800, 600)
    page.show()
    _app().processEvents()
    page.canvas.draw()
    return page


def _texts(menu):
    return [a.text() for a in menu.actions() if not a.isSeparator()]


def test_the_right_click_menu_offers_what_is_under_the_mouse(data, monkeypatch):
    from pymetrix.figures import _x_number

    edits, styles = [], []
    figure = _arrow(data)
    page = _window(figure, on_restyle=lambda *a: edits.append(("restyle", *a)), on_remove=lambda *a: edits.append(("remove", *a)),
                   on_format=lambda *a: edits.append(("properties", *a)), on_style=lambda *a: styles.append(a),
                   on_pick=lambda *a: None, on_place=lambda *a: edits.append(("note", *a)))
    ax = page.canvas.figure.axes[0]
    (hx, hy), (tx, ty) = ax.transData.transform((_x_number("2019Q1", "Q"), 120)), ax.transData.transform((_x_number("2016Q1", "Q"), 135))
    menu = page.context_menu((hx + tx) / 2, (hy + ty) / 2)  # on the arrow's line
    assert _texts(menu)[:6] == ["Format…", "Edit the words…", "Bring to front", "Send to back", "Edit as code (every property)…", "Delete"]
    assert page.editor.selected is not None  # selected, its handles shown, as Excel does
    next(a for a in menu.actions() if a.text() == "Bring to front").trigger()
    assert edits[-1] == ("restyle", 0, {"zorder": 5})
    next(a for a in menu.actions() if a.text() == "Delete").trigger()
    assert edits[-1] == ("remove", 0)
    lx, ly = ax.transData.transform(((_x_number("2017Q1", "Q") + _x_number("2017Q2", "Q")) / 2, 117))  # on c, between points
    menu = page.context_menu(lx, ly)
    assert _texts(menu)[0] == "Format series “c”…"
    menu.actions()[0].trigger()
    assert styles[-1] == ("c", 0)
    ex, ey = ax.transData.transform((_x_number("2020Q1", "Q"), 110))  # empty space in the panel
    texts = _texts(page.context_menu(ex, ey))
    assert texts[:3] == ["Format axes…", "Draw an arrow", "Add a note here…"] and "Copy image" in texts
    next(a for a in page.context_menu(ex, ey).actions() if a.text() == "Add a note here…").trigger()
    assert edits[-1][:3] == ("note", 0, "2020Q1")
    assert _texts(page.context_menu(2, 2))[0] == "Titles, shading and marks…"  # outside every panel: the figure's own


def test_format_opens_the_form_and_runs_its_changes(data, monkeypatch):
    from pymetrix.ui import chart_format

    edits = []
    page = _window(_arrow(data), on_restyle=lambda *a: edits.append(a), on_remove=lambda *a: None)
    opened, real = [], chart_format.format_dialog

    def answer(parent, call):  # the form, filled in and OK'd
        dialog = real(parent, call)
        dialog.widgets["linewidth"].setText("3")
        dialog.widgets["arrowstyle"].setCurrentIndex(dialog.widgets["arrowstyle"].findData("<->"))
        dialog.exec = lambda: (dialog.findChild(QtWidgets.QDialogButtonBox).accepted.emit(), True)[1]
        opened.append(dialog)
        return dialog

    monkeypatch.setattr(chart_format, "format_dialog", answer)
    page.format_item(0)
    assert opened and opened[0].windowTitle() == "Format the arrow or note"
    (step, changed), = edits
    assert step == 0 and changed["arrowprops"]["linewidth"] == 3 and changed["arrowprops"]["arrowstyle"] == "<->"


def test_the_form_shows_its_change_as_code_and_runs_what_is_typed_there(data):
    _app()
    from pymetrix.ui.chart_format import format_dialog, keywords

    call = _arrow(data).style[0]
    dialog = format_dialog(None, call)
    dialog.code_toggle.setChecked(True)
    assert dialog.code_box.toPlainText() == ""  # no change yet
    dialog.widgets["linewidth"].setText("2")
    assert dialog.code_box.toPlainText() == keywords(props(call, dialog.values()))  # follows the form
    assert "'linewidth': 2" in dialog.code_box.toPlainText()
    dialog.code_box.setPlainText("arrowprops={'arrowstyle': 'fancy', 'color': 'navy'}, zorder=4")  # an expert types
    dialog.widgets["linewidth"].setText("3")  # the typed code is kept, not overwritten
    assert "fancy" in dialog.code_box.toPlainText()
    dialog.findChild(QtWidgets.QDialogButtonBox).accepted.emit()
    assert dialog.result_props() == {"arrowprops": {"arrowstyle": "fancy", "color": "navy"}, "zorder": 4}


def test_every_dialog_can_show_the_code_it_records():
    _app()
    from pymetrix import Dataset
    from pymetrix.commands import FitEquation, PlotSeries, Session
    from pymetrix.ui.qt import command_dialog

    session = Session()
    session.objects["data"] = Dataset.example("sample_macro")
    dialog = command_dialog(PlotSeries, session)
    try:
        dialog.code_toggle.setChecked(True)
        dialog.editors["series"].set("c, i")
        _app().processEvents()
        assert dialog.code_box.toPlainText() == "figure = data.plot(['c', 'i'])"  # as the fields are filled in
        assert command_dialog(FitEquation, session).code_text() == "# the code appears once the fields it needs are filled in"
    finally:
        dialog.code_toggle.setChecked(False)  # the choice is remembered: put it back for the other tests
