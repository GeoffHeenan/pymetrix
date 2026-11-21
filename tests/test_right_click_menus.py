"""Right-click menus on data and results: the data window's cells and series headings (copy, paste, fill down,
clear; plot, describe, new series from it, keep as a dataset, rename, remove, freeze) and a result's tables (copy,
sort the rows shown, show as a heatmap, add to the report)."""

import os

import numpy as np
import pandas as pd
import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _texts(menu):
    return [a.text() for a in menu.actions() if not a.isSeparator()]


def _action(menu, text):
    return next(a for a in menu.actions() if a.text() == text)


@pytest.fixture
def sheet():
    _app()
    from pymetrix import Dataset
    from pymetrix.ui.sheet_qt import SheetWindow

    idx = pd.period_range("2020Q1", periods=6, freq="Q", name="time")
    data = Dataset.from_frame(pd.DataFrame({"c": np.arange(6.0), "u": np.arange(6.0) * 10}, index=idx))
    asked = []
    window = SheetWindow("data", lambda: data, lambda command: True,
                         series_actions={k: (lambda names, k=k: asked.append((k, names))) for k in ("plot", "describe", "generate", "select")})
    window.asked = asked
    window.show()
    return window


def test_a_right_click_on_cells(sheet):
    model, table = sheet.model, sheet.table
    table.setCurrentIndex(model.index(1, 0))
    table.selectionModel().select(model.index(1, 0), table.selectionModel().SelectionFlag.Select)
    for row in (2, 3):
        table.selectionModel().select(model.index(row, 0), table.selectionModel().SelectionFlag.Select)
    menu = sheet.cell_menu(model.index(2, 0))
    texts = _texts(menu)
    assert texts[:4] == ["Copy", "Paste", "Fill down", "Clear (make missing)"]
    assert texts[4:8] == ["Plot “c”…", "Describe “c”…", "New series from it…", "Keep “c” as a new dataset…"]
    assert "Rename…" in texts and "Remove" in texts and "Series info" in texts
    _action(menu, "Fill down").trigger()  # 1.0 into rows 2 and 3, one undoable change, pending until Apply
    edit = sheet.model.data
    from PySide6.QtCore import Qt

    assert [float(edit(model.index(r, 0), Qt.ItemDataRole.EditRole)) for r in (1, 2, 3)] == [1.0, 1.0, 1.0]
    assert len(model.pending) == 2 and sheet.undo.count() == 1
    sheet.undo.undo()
    assert not model.pending
    _action(sheet.cell_menu(model.index(2, 0)), "Clear (make missing)").trigger()
    assert all(v is None for v in model.pending.values()) and len(model.pending) == 3
    _action(sheet.cell_menu(model.index(2, 0)), "Plot “c”…").trigger()
    assert sheet.asked[-1] == ("plot", ["c"])
    menu = sheet.cell_menu(model.index(4, 1))  # outside the selection: that cell is picked, as Excel does
    assert "Plot “u”…" in _texts(menu) and not _action(menu, "Fill down").isEnabled()


def test_a_right_click_on_a_series_heading(sheet):
    menu = sheet.column_menu(1)
    texts = _texts(menu)
    assert texts[0] == "Plot “u”…" and texts[-1] == "Copy the column"
    _action(menu, "Describe “u”…").trigger()
    assert sheet.asked[-1] == ("describe", ["u"])
    sheet.table.selectionModel().clearSelection()
    sheet.table.selectColumn(0)
    from PySide6.QtCore import QItemSelectionModel

    sheet.table.selectionModel().select(sheet.model.index(0, 1),
                                        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Columns)
    assert _texts(sheet.column_menu(1))[0] == "Plot 2 series…" and "Rename…" not in _texts(sheet.column_menu(1))


def test_a_right_click_on_a_result_table():
    _app()
    from pymetrix.output import Table
    from pymetrix.ui.qt import _table_menu

    table = Table(["coef", "p"], [[0.5, 0.01], [-1.5, 0.3], [2.5, float("nan")]], index=["a", "b", "c"], title="Estimates")
    grid = QtWidgets.QTableWidget(3, 2)
    shown = []
    _table_menu(grid, table, report=lambda: shown.append("report"), heatmap=lambda: shown.append("heatmap"))
    from pymetrix.ui.qt import _fill_grid, _qt

    _fill_grid(_qt(), grid, table.to_frame(), table)
    grid.resize(400, 300)
    grid.show()
    _app().processEvents()
    pos = grid.visualItemRect(grid.item(0, 0)).center()
    grid.menu_at(pos)
    assert grid.last_menu[:3] == ["Copy table, for Excel", "Copy table as Markdown", "Copy table as CSV"]
    assert "Sort by “coef”, largest first" in grid.last_menu and grid.last_menu[-2:] == ["Show as a heatmap", "Add to report"]
    grid.last_menu_actions["Sort by “coef”, largest first"].trigger()
    assert [grid.verticalHeaderItem(r).text() for r in range(3)] == ["c", "a", "b"] and grid.item(0, 0).text().startswith("2.5")
    grid.menu_at(grid.visualItemRect(grid.item(0, 1)).center())
    grid.last_menu_actions["Sort by “p”, smallest first"].trigger()  # missing values last
    assert [grid.verticalHeaderItem(r).text() for r in range(3)] == ["a", "b", "c"]
    grid.menu_at(pos)
    grid.last_menu_actions["Back to the table's order"].trigger()
    assert [grid.verticalHeaderItem(r).text() for r in range(3)] == ["a", "b", "c"] and grid.sorted_by is None
    grid.last_menu_actions["Show as a heatmap"].trigger()
    assert shown == ["heatmap"]


@pytest.fixture
def window(monkeypatch):
    _app()
    from pymetrix import Dataset
    from pymetrix.commands import Session
    from pymetrix.ui import plugins_qt
    from pymetrix.ui.qt import main_window

    monkeypatch.setattr(plugins_qt, "ask_new_plugins", lambda *a, **k: None)
    session = Session()
    session.objects["data"] = Dataset.example("sample_macro")
    w = main_window(session)
    yield w, session
    session.dirty = False
    w.close()


def test_a_right_click_on_problems_acts_on_what_each_is_about(window):
    from pymetrix.commands import FitEquation

    w, session = window
    from pymetrix.errors import PyMetrixError

    with pytest.raises(PyMetrixError):  # a series not in the data: the step fails, and Problems lists it
        session.run(FitEquation("nope ~ c", name="bad"))
    w.refresh()  # Problems is rebuilt on the window's refresh, as after any step in the app
    view = w.problems_view
    from PySide6.QtCore import Qt

    items = [view.topLevelItem(i) for i in range(view.topLevelItemCount())]
    failed = next(i for i in items if i.data(0, Qt.ItemDataRole.UserRole) and i.data(0, Qt.ItemDataRole.UserRole)[0] == "step")
    texts = _texts(w.problems_menu(failed))
    assert texts[:3] == ["Show the step in History", "Open its dialog, filled in…", "Copy its code"] and texts[-1] == "Copy the problem"
    _action(w.problems_menu(failed), "Copy its code").trigger()
    assert "nope" in QtWidgets.QApplication.clipboard().text()


def test_a_right_click_on_the_report_and_the_script(window):
    w, session = window
    from pymetrix.commands import PlotSeries

    session.run(PlotSeries(series="c", name="fig"))
    session.add_to_report("fig", None)
    session.add_report_heading("Results")
    from pymetrix.ui.report_qt import report_panel

    opened = []
    panel = report_panel(lambda: session, lambda: None, open_result=opened.append)
    texts = _texts(panel.item_menu(0))
    assert texts[:2] == ["Edit the caption…", "Open “fig”"] and "Move to the end" in texts and texts[-1] == "Remove from the report"
    _action(panel.item_menu(0), "Open “fig”").trigger()
    assert opened == ["fig"]
    _action(panel.item_menu(0), "Move to the end").trigger()
    assert [i["kind"] for i in session.report_outline] == ["heading", "item"]
    assert _texts(panel.item_menu(1))[0] == "Edit the caption…" and not _action(panel.item_menu(1), "Move down").isEnabled()
    assert _texts(panel.item_menu(-1)) == ["Add a heading…", "Add a paragraph…"]  # on empty space
    menu = w.script_menu(QtCore_point(0, 0))
    assert _texts(menu)[-2:] == ["Copy the whole script", "Open in a Code window"]


def QtCore_point(x, y):  # a QPoint, as a right-click gives
    from PySide6.QtCore import QPoint

    return QPoint(x, y)


def test_a_right_click_in_a_code_window_runs_and_gives_help(window):
    _, session = window
    from pymetrix.ui.editor_qt import EditorWindow

    ran = []
    editor = EditorWindow(lambda code, first, e: ran.append(code), lambda: session.namespace, text="x = data.plot('c')\n")
    cursor = editor.editor.textCursor()
    cursor.setPosition(len("x = data.pl"))  # on plot
    menu = editor.editor.context_menu(editor.editor.cursorRect(cursor).center())
    texts = _texts(menu)
    assert {"Run this line", "Run the whole script", "Comment or uncomment", "Find…"} <= set(texts)
    help_item = next(t for t in texts if t.startswith("Help on"))
    assert "plot" in help_item
    name, doc = editor.editor.help_at(cursor)
    assert name.endswith("plot") and doc.strip()
    _action(menu, "Run the whole script").trigger()
    assert ran == ["x = data.plot('c')\n"]
    box = editor.show_help(name, doc)
    from pymetrix.ui.messages import TITLES_SHOWN

    # a Mac shows no title on a message box (Qt drops it there); the name heads the text on every platform
    assert box.windowTitle() == (f"Help: {name}" if TITLES_SHOWN else "") and box.text() == f"<b>{name}</b>"
    assert box.detailedText() == doc
    box.close()


def test_a_window_s_own_menu(window):
    w, session = window
    from pymetrix.commands import PlotSeries

    session.run(PlotSeries(series="c", name="fig"))
    w.show_object("fig")
    sub = w.mdi.activeSubWindow() or w.mdi.subWindowList()[-1]
    texts = _texts(w.window_menu(sub))
    assert texts[:2] == ["Copy the window as an image", "Add “fig” to the report"]
    assert {"Close", "Close the other windows", "Tile windows", "Cascade windows"} <= set(texts)
    _action(w.window_menu(sub), "Add “fig” to the report").trigger()
    assert session.report_outline[-1]["source"] == "fig"
