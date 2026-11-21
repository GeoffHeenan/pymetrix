"""The reviewer-demo polish: active-window zoom, Fields entries and table copying, dataset header, window titles,
status bar, Problems panel, producer actions, empty states and demo reloading. Off-screen Qt."""

import os

import pytest

from pymetrix import demo

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("qtconsole")


@pytest.fixture
def window():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import main_window

    w = main_window(demo("consumption"))
    w.show()
    yield w
    w.state["session"].dirty = False
    w.close()


def _status(w):
    return {k: label.text() for k, label in w.status_labels.items()}


def test_zoom_acts_on_the_active_view_only(window):
    window.show_object("dynamic")
    view = window.mdi.activeSubWindow().widget()
    base = view.font().pointSizeF()
    menu_font = window.menuBar().font().pointSizeF()
    window.zoom(1)
    assert view.zoom_factor() == pytest.approx(1.15) and view.font().pointSizeF() == pytest.approx(base * 1.15)
    assert window.menuBar().font().pointSizeF() == menu_font  # the application itself keeps its size
    window.zoom(0)
    assert view.font().pointSizeF() == pytest.approx(base)
    window.mdi.closeAllSubWindows()
    window.zoom(1)
    assert "No window to zoom" in window.statusBar().currentMessage()


def test_zoom_in_the_code_editor_and_the_spreadsheet():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.editor_qt import EditorWindow
    from pymetrix.ui.sheet_qt import SheetWindow

    editor = EditorWindow(lambda code, line, ed: None, lambda: {})
    size = editor.editor.font().pointSizeF()
    editor.zoom_in()
    assert editor.editor.font().pointSizeF() == pytest.approx(size * 1.15)
    data = demo("consumption")["data"]
    sheet = SheetWindow("data", lambda: data, lambda cmd: True)
    sheet.zoom_to(5)
    assert sheet.table.verticalHeader().defaultSectionSize() == round(22 * 1.3)


def test_fields_entries_and_copying_table_cells(window):
    from pymetrix.ui.qt import _selected_text

    window.show_object("dynamic")
    view = window.mdi.activeSubWindow().widget()
    fields = view.grids["Python access"]
    rows = [fields.verticalHeaderItem(r).text() for r in range(fields.rowCount())]
    assert 'dynamic.stats["nobs"]' in rows and "dynamic.params" in rows
    row = rows.index('dynamic.stats["nobs"]')
    fields.menu_at(fields.visualItemRect(fields.item(row, 0)).center())
    assert fields.last_menu[0] == 'Copy Python: dynamic.stats["nobs"]' and "Copy table as Markdown" in fields.last_menu
    coefficients = view.grids["Coefficients"]
    coefficients.setRangeSelected(QtWidgets.QTableWidgetSelectionRange(0, 0, 0, 1), True)
    text = _selected_text(coefficients)
    assert text.splitlines()[0].split("\t")[1:] == ["coef", "std err"] and text.splitlines()[1].startswith("const\t")
    assert view.made_by.text() == "Copy the step's code"


def test_dataset_header_and_window_titles(window):
    window.show_object("data")
    sub = window.mdi.activeSubWindow()
    assert sub.windowTitle().startswith("data — Dataset")
    header = sub.widget().header.text()
    assert "series" in header and "quarterly" in header and "missing" in header
    from pymetrix.ui.qt import kind_of
    from pymetrix.models.estimation import FittedModel

    assert kind_of(FittedModel.__new__(FittedModel)) == "Estimated model"
    assert kind_of(type("ShockDecompositionPlot", (), {})()) == "Shock decomposition plot"


def test_status_bar_problems_and_producer_actions(window):
    from pymetrix.commands.data_edit import GenerateSeries
    from pymetrix.commands.samples import SetSample

    status = _status(window)
    assert status["dataset"].startswith("data:") and "series" in status["dataset"] and status["last"].startswith("Last step ✓")
    assert window.problem_list == []
    session = window.state["session"]
    session.run(SetSample("data", start="2010Q1", end="2024Q4"))
    window.switch(session)
    assert any("out of date" in what for _, what in window.problem_list)
    assert _status(window)["sample"].startswith("sample")
    item = next(window.objects.topLevelItem(i) for i in range(window.objects.topLevelItemCount())
                if window.objects.topLevelItem(i).text(0) == "dynamic")
    window.objects_menu(window.objects.visualItemRect(item).center())
    assert {"Show the step in History", "Copy the step's code"} <= set(window.last_objects_menu)
    window.history.select_record(session.producer("dynamic"))
    assert window.history.selected() == [session.producer("dynamic")]
    with pytest.raises(Exception):
        session.run(GenerateSeries("c", "c * 2"))  # refused: c exists
    window.switch(session)


def test_empty_states_and_demo_reloading(window):
    from pymetrix.commands import Session

    assert window.windowTitle() == "PyMetrix — demo: consumption"  # opened, so nothing unsaved yet
    window.state["session"].dirty = False  # nothing to discard, so no question
    window.reload_demo()
    assert window.state["session"].demo == "consumption"
    window.state["session"].dirty = False
    window.switch(Session())
    assert window.objects_empty.isVisible() and "No objects yet" in window.objects_empty.text()
    assert window.history.topLevelItem(0).text(0).startswith("No steps yet")  # across the row
    assert window.problem_list == [] and _status(window)["dataset"] == "No dataset"
    window.reload_demo()
    assert "not a demo project" in window.statusBar().currentMessage()


def test_building_a_demo_shows_a_busy_cursor_and_restores_it(window, monkeypatch):
    """nk_estimation takes about twenty seconds to build: the window says so rather than freezing silently."""
    import importlib

    module = importlib.import_module("pymetrix.demos")  # the module the window takes demo from
    seen = {}
    real = module.demo

    def watched(name):
        seen["cursor"] = QtWidgets.QApplication.overrideCursor() is not None
        seen["message"] = window.statusBar().currentMessage()
        return real(name)

    monkeypatch.setattr(module, "demo", watched)
    window.state["session"].dirty = False
    window.reload_demo()
    assert seen["cursor"] and seen["message"].startswith("Building demo consumption")
    assert QtWidgets.QApplication.overrideCursor() is None  # restored afterwards


def test_tables_read_cleanly_and_copy_exact_numbers(window):
    """Numbers right-aligned, a tiny p-value shown as <0.0001 but copied as the number, provenance in words, row numbers
    hidden where the first column names the rows, and short tables no taller than their rows."""
    from PySide6.QtCore import Qt as QtNS

    from pymetrix.output import bounds_text
    from pymetrix.ui.qt import _selected_text

    window.show_object("dynamic")
    tabs = window.mdi.activeSubWindow().widget()
    summary = tabs.grids["Summary"]
    p_col = [summary.horizontalHeaderItem(c).text() for c in range(summary.columnCount())].index("p>|t|")
    assert summary.item(0, p_col).text() == "<0.0001"
    assert summary.item(0, 0).textAlignment() & QtNS.AlignmentFlag.AlignRight
    summary.clearSelection()
    summary.item(0, p_col).setSelected(True)
    copied = _selected_text(summary).splitlines()[1].split("\t")[1]
    assert float(copied) == tabs.tables["Summary"].iat[0, p_col] and float(copied) < 1e-4  # the number, in full

    provenance = tabs.grids["Provenance"]
    values = {provenance.item(r, 0).text(): provenance.item(r, 1).text() for r in range(provenance.rowCount())}
    assert " to " in values["sample"] and "{" not in values["method"] and len(values["created"]) == 16
    assert not provenance.verticalHeader().isVisible()  # the first column names each row

    tabs.setCurrentIndex(0)  # Summary
    window.mdi.activeSubWindow().showMaximized()
    QtWidgets.QApplication.processEvents()
    assert summary.maximumHeight() < 300  # three coefficients: the notes follow the table

    assert bounds_text(0, 1e9) == "[0, ∞)" and bounds_text(0, 1) == "[0, 1]" and bounds_text(float("-inf"), 2) == "(-∞, 2]"
    assert bounds_text(float("-inf"), float("inf")) == ""


@pytest.mark.parametrize("project", ["empty", "consumption", "policy_and_nonlinear"])
def test_every_menu_dialog_opens(project):
    """Each dialog reachable from a menu builds, with data and without: Data > Seasonal adjustment crashed in every
    project because its yes / no / automatic setting had no choices."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.commands import COMMANDS, Session
    from pymetrix.ui.qt import command_dialog

    session = Session() if project == "empty" else demo(project)
    failed = []
    for info in COMMANDS.values():
        if not info.menu:  # recorded by other windows (code, spreadsheet edits), never opened as a dialog
            continue
        try:
            command_dialog(info.cls, session).deleteLater()
        except Exception as exc:  # noqa: BLE001 - the test lists every failure
            failed.append(f"{info.menu}: {type(exc).__name__}: {exc}")
    assert not failed, failed


def test_dialog_fields_say_what_defaults_mean(window):
    """A 0 that means something says it, a yes/no/automatic setting offers all three, a blank choice reads (none)."""
    from pymetrix.commands import COMMANDS
    from pymetrix.forms import form_spec
    from pymetrix.seasonal import STL
    from pymetrix.ui.qt import command_dialog

    session = window.state["session"]
    mcmc = command_dialog(next(i.cls for i in COMMANDS.values() if i.cls.__name__ == "RunMCMC"), session)
    assert mcmc.editors["workers"].widget.specialValueText() == "automatic" and mcmc.editors["workers"].get() == 0
    assert mcmc.editors["burn"].widget.specialValueText() == "as many as kept"
    log = next(s for s in form_spec(STL) if s.name == "log")
    assert log.choices == (None, True, False) and [log.display(c) for c in log.choices] == ["Automatic", "Yes", "No"]
    sample = command_dialog(next(i.cls for i in COMMANDS.values() if i.cls.__name__ == "SetSample"), session)
    assert sample.editors["sample"].widget.currentText() == "(none)" and sample.editors["sample"].get() == ""


def test_figures_and_help_read_consistently():
    """Whole-number axes for horizons; one wording for a response to a shock; tests named in the help contents."""
    from pymetrix import templates
    from pymetrix.help import contents

    nk = templates.new_keynesian()
    fig = nk.model.linearize(nk.model.steady_state(), shocks=nk.shocks).solve().irf(20).plot().draw()
    ticks = fig.axes[0].get_xticks()
    title = fig._suptitle.get_text()  # no parentheses inside parentheses: "Impulse responses: Cholesky (VAR order)"
    assert all(float(t).is_integer() for t in ticks) and "(" not in title.split("(", 1)[-1] and title.startswith("Impulse responses")
    tests = dict(contents())["Tests"]
    assert ("Augmented Dickey-Fuller test (adf)" in [t for t, _ in tests]) and all("(" in t for t, _ in tests)


def test_panes_and_editors_read_consistently(window):
    """Python in a fixed-width font in the Script pane; spreadsheet numbers as in the result tables; empty states
    across the row; spreadsheets and code editors open inside the workspace."""
    from PySide6.QtGui import QFontDatabase

    from pymetrix.commands import Session

    script = next(d for d in window.findChildren(QtWidgets.QDockWidget) if d.windowTitle() == "Script").widget()

    assert script.font().family() == QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()
    from pymetrix.ui.sheet_qt import _fmt

    assert _fmt(195.1455123) == "195.1455" and _fmt(1234567.8) == "1,234,568"
    window.resize(1000, 700)
    window.open_sheet("data")
    window.open_editor(text="x = 1\n")
    QtWidgets.QApplication.processEvents()
    area = window.mdi.viewport().rect()
    assert all(area.contains(sub.geometry()) for sub in window.mdi.subWindowList())
    window.state["session"].dirty = False
    for sheet in list(window.sheets.values()):
        sheet.model.pending.clear()
    for editor in window.editors():
        editor.editor.document().setModified(False)
    window.switch(Session())
    QtWidgets.QApplication.processEvents()
    empty = window.history.topLevelItem(0)
    assert empty.text(0).startswith("No steps yet") and empty.isFirstColumnSpanned()
    problem = window.problems_view.topLevelItem(0)
    assert problem.text(0).startswith("No problems:") and problem.isFirstColumnSpanned()


def test_recent_projects_in_the_file_menu(window, tmp_path, monkeypatch):
    """Projects saved or opened are listed under File > Open recent, newest first; a missing file drops out."""
    from pymetrix.commands import Session

    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(tmp_path / "first.pymx"), "")))
    window.save_project(as_new=True)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(tmp_path / "second.pymx"), "")))
    window.save_project(as_new=True)
    assert [p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in window.recent_projects()][:2] == ["second.pymx", "first.pymx"]
    window.list_recent()
    labels = [a.text().replace("&", "") for a in window.recent_menu.actions() if a.text()]
    assert labels[:2] == ["1 second.pymx", "2 first.pymx"] and labels[-1] == "Clear recent"
    window.state["session"].dirty = False
    window.switch(Session())
    first = next(a for a in window.recent_menu.actions() if a.text().endswith("first.pymx"))
    first.trigger()
    QtWidgets.QApplication.processEvents()
    assert "data" in window.state["session"].objects and window.state["session"].path.name == "first.pymx"
    (tmp_path / "second.pymx").unlink()
    assert all(not p.endswith("second.pymx") for p in window.recent_projects())  # gone from disk: gone from the list
    window.list_recent()
    next(a for a in window.recent_menu.actions() if a.text().replace("&", "") == "Clear recent").trigger()
    assert window.recent_projects() == []


def test_copy_summary_on_a_result_window(window):
    window.show_object("dynamic")
    tabs = window.mdi.activeSubWindow().widget()
    QtWidgets.QApplication.clipboard().clear()
    tabs.copy_summary.click()
    text = QtWidgets.QApplication.clipboard().text()
    assert "c ~ y + c[-1]" in text and "R-squared" in text and tabs.made_by.text() == "Copy the step's code"


def test_save_result_and_open_it_again(window, tmp_path, monkeypatch):
    """File > Save result... writes the active window's result; File > Open result... reads it back as a saved result."""
    from pymetrix.commands import OpenResult
    from pymetrix.ui.qt import kind_of

    from PySide6.QtGui import QAction

    labels = [a.text().replace("&", "") for a in window.findChildren(QAction)]  # not through menu(): PySide6 6.6 hands back deleted menus
    assert "Save result…" in labels and "Open result…" in labels
    window.show_object("dynamic")
    window.save_result(str(tmp_path / "dynamic"))
    assert (tmp_path / "dynamic.pymxr").is_file() and "dynamic saved to dynamic.pymxr" in window.statusBar().currentMessage()
    session = window.state["session"]
    session.run(OpenResult(str(tmp_path / "dynamic.pymxr"), "theirs"))
    window.refresh()
    window.show_object("theirs")
    tabs = window.mdi.activeSubWindow().widget()
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    assert titles[0] == "Summary" and "Saved" in titles and "Coefficients" in titles
    assert kind_of(session.objects["theirs"]) == "Saved fitted equation"
    told = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: told.append(a[2])))
    window.mdi.closeAllSubWindows()
    window.objects.setCurrentItem(next(window.objects.topLevelItem(i) for i in range(window.objects.topLevelItemCount())
                                       if window.objects.topLevelItem(i).text(0) == "data"))
    window.save_result(str(tmp_path / "data"))  # a dataset is not a result: nothing written, and it says where data go
    assert not (tmp_path / "data.pymxr").exists() and "is a dataset" in told[-1]
