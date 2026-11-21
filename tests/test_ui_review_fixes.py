"""Fixes from a review of the background steps, the start page, the stage bar and the report outline."""

import json
import os

import pytest

from pymetrix import Dataset
from pymetrix.commands import ExportReport, FitEquation, FitVAR, GenerateSeries, ImpulseResponse, Session, save_project
from pymetrix.progress import Task, task


@pytest.fixture(scope="module")
def data():
    return Dataset.example("sample_macro").with_series(gy="400*dlog(y)", gc="400*dlog(c)")


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _settings():
    from PySide6.QtCore import QSettings

    return QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope, "pymetrix", "desktop")


def test_cancelling_the_recalculation_keeps_the_step(data):
    s = Session()
    s.objects["data"] = data
    s.run(FitVAR("gy, gc", lags="1", name="var"))
    s.run(ImpulseResponse("var", periods=4, draws=50, name="irf"))
    s.set_mode("automatic")
    steps = len(s.history)
    with task() as t:
        t.cancel()  # the revision itself never checks; the bootstrap redone after it does
        s.run(GenerateSeries("gy", "gy * 1.01", replace=True))
    assert s.recalculation_stopped and type(s.history[steps].command).__name__ == "GenerateSeries"  # recorded
    assert "irf" in s.stale()  # left out of date, for Recalculate
    s.run(GenerateSeries("z", "gy + 1"))
    assert not s.recalculation_stopped and "irf" not in s.stale()  # the next step recalculates in full


def test_the_outline_wins_over_a_template_in_the_script():
    outline = json.dumps([{"kind": "heading", "text": "Results", "level": 2}, {"kind": "item", "source": "fit"}])
    cmd = ExportReport(objects="", path="r.docx", template="estimation_summary", use_outline=True, outline=outline)
    code = cmd.to_code()
    assert "from_template" not in code and "report.heading('Results', 2)" in code and "report.add(fit)" in code


def test_a_dialog_ignores_ok_and_cancels_on_esc_while_its_step_runs(app, data, monkeypatch):
    from PySide6.QtWidgets import QDialogButtonBox, QMessageBox

    from pymetrix.ui import qt
    from pymetrix.ui.qt import command_dialog

    s = Session()
    s.objects["data"] = data
    said = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: said.append(a[1]))
    dialog = command_dialog(FitEquation, s, initial=FitEquation("gc ~ gy", name="fit"))
    buttons = dialog.findChild(QDialogButtonBox)
    ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
    seen = {}

    def running(fn, label=""):
        qt._BUSY["task"] = t = Task()
        seen["ok enabled"] = ok.isEnabled()
        buttons.accepted.emit()  # OK again: ignored, not a second run
        dialog.reject()  # Esc, Cancel or close: asks the step to stop, the dialog stays
        seen["cancelled"], seen["visible result"] = t.cancelled, dialog.result()
        try:
            return fn()
        finally:
            qt._BUSY.pop("task", None)

    monkeypatch.setattr(qt, "run_blocking", running)
    buttons.accepted.emit()
    assert seen == {"ok enabled": False, "cancelled": True, "visible result": 0}
    assert len(s.history) == 1 and ok.isEnabled() and dialog.result() == 1 and not said  # one run, then closed
    dialog = command_dialog(FitEquation, s, initial=FitEquation("gc ~ gy", name="fit2"))
    dialog.reject()  # not running: an ordinary Cancel
    assert dialog.result() == 0 and len(s.history) == 1

    def stopped(fn, label=""):
        value = fn()
        s.recalculation_stopped = True  # as if Cancel stopped the recalculation after the step
        return value

    monkeypatch.setattr(qt, "run_blocking", stopped)
    dialog = command_dialog(FitEquation, s, initial=FitEquation("gc ~ gy", name="fit3"))
    dialog.findChild(QDialogButtonBox).accepted.emit()
    assert said == ["Recalculation stopped"] and "fit3" in s.objects and dialog.result() == 1


def test_results_not_computed_count_in_their_stage(app, data):
    from pymetrix.commands import COMMANDS
    from pymetrix.ui.workspaces import stage_of_command

    by_name = {i.cls.__name__: i.cls for i in COMMANDS.values()}
    unmade = lambda name: by_name[name].__new__(by_name[name])  # only the step's kind matters
    assert stage_of_command(unmade("MakeRound")) == "forecast" and stage_of_command(unmade("SolveModel")) == "model"
    assert stage_of_command(GenerateSeries("z", "y")) == "explore" and stage_of_command(FitEquation("c ~ y")) == "estimate"
    assert stage_of_command(unmade("LoadData")) == "explore" and stage_of_command(unmade("OpenResult")) == "estimate"
    assert stage_of_command(object()) == "estimate"


def test_the_start_page_goes_when_another_session_opens(app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMdiSubWindow, QMessageBox

    from pymetrix.ui.qt import main_window

    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    project = save_project(s, tmp_path / "p.pymx")
    window = main_window(Session())
    try:
        starts = lambda: [w for w in window.mdi.subWindowList() if isinstance(w, QMdiSubWindow) and w.windowTitle() == "Start"]
        assert window.start_page is not None and len(starts()) == 1
        new = next(a for m in window.menuBar().findChildren(type(window.menuBar().actions()[0].menu())) for a in m.actions()
                   if a.text().replace("&", "") == "New session")
        new.trigger()
        new.trigger()  # never two Start windows
        app.processEvents()
        assert len(starts()) == 1
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
        window.open_dropped([str(project)])  # opening a project: the Start window goes
        app.processEvents()
        assert window.start_page is None and not starts() and "data" in window.state["session"].objects
    finally:
        window.state["session"].dirty = False
        window.close()


def test_the_code_toggle_hides_the_code_panels_in_all(app):
    from PySide6.QtWidgets import QDockWidget

    from pymetrix.ui.qt import main_window

    _settings().remove("workspace")
    window = main_window(Session())
    try:
        window.switch_workspace("all")
        script = next(d for d in window.findChildren(QDockWidget) if d.windowTitle() == "Script")
        code = next(a for a in window.findChildren(type(window.workspace_actions["all"])) if a.text().replace("&", "") == "Code")
        immediate = next(d for d in window.findChildren(QDockWidget) if d.windowTitle() == "Immediate")
        script.setVisible(False)
        immediate.setVisible(True)  # as a first run shows All
        code.setChecked(True)
        assert script.isVisibleTo(window) and immediate.isVisibleTo(window)
        code.setChecked(False)  # off, in All: the code panels as they were before it was on (not all hidden)
        assert not script.isVisibleTo(window) and immediate.isVisibleTo(window)
        script.setVisible(True)
        window.switch_workspace("all")  # switching to All keeps them as they were
        assert script.isVisibleTo(window)
        view = next(m for m in window.menuBar().findChildren(type(window.menuBar().actions()[0].menu())) if m.title() == "&View"
                    or m.title() == "View")
        assert any(a.text().replace("&", "") == "Report" for a in view.actions())  # the Report panel has its View toggle
    finally:
        window.close()
        _settings().remove("workspace")


def test_dropping_several_files_says_what_was_left_and_finds_a_free_name(app, tmp_path, monkeypatch):
    from pymetrix.ui import qt
    from pymetrix.ui.qt import main_window

    s = Session()
    s.objects["prices"] = s.objects["prices_2"] = Dataset.example("sample_macro")
    window = main_window(s)
    try:
        opened = []
        monkeypatch.setattr(qt, "command_dialog", lambda cls, session, parent=None, initial=None: opened.append(initial) or
                            type("Shut", (), {"exec": lambda self: False})())
        first, second = tmp_path / "prices.csv", tmp_path / "more.csv"
        window.open_dropped([str(first), str(second)], dropped=3)
        assert opened[-1].name == "prices_3"  # prices and prices_2 are taken
        assert "2 other files not opened" in window.statusBar().currentMessage()
    finally:
        s.dirty = False
        window.close()


def test_typing_in_the_edit_line_then_moving_keeps_the_value(app, data):
    from pymetrix.ui.sheet_qt import SheetWindow

    sheet = SheetWindow("data", lambda: data, lambda cmd: True)
    m, line = sheet.model, sheet.edit_line
    c = m.columns.index("c")
    sheet.table.setCurrentIndex(m.index(2, c))
    line.line.setText("512.25")
    line.line.textEdited.emit("512.25")
    sheet.table.setCurrentIndex(m.index(5, c))  # another cell: the typed value is accepted first
    assert m.pending[("c", m.label(2))] == 512.25
    assert line.line.text() == repr(float(data["c"].iloc[5]))
    m.pending.clear()


def test_help_lists_every_shortcut_and_no_two_commands_share_one(app):
    from collections import defaultdict

    from pymetrix.ui.qt import OTHER_SHORTCUTS, main_window, shortcut_list, shortcuts_dialog

    window = main_window(Session())
    try:
        from PySide6.QtGui import QKeySequence

        def native(keys):  # as the list shows them: the platform's own words (⌘K for Ctrl+K on a Mac)
            return QKeySequence(keys).toString(QKeySequence.SequenceFormat.NativeText)

        rows = shortcut_list(window)
        menus = rows[:len(rows) - len(OTHER_SHORTCUTS)]
        assert ("Calculate", "Recalculate", "F9") in menus and ("View", "Find anything", native("Ctrl+K")) in menus
        assert ("Code windows", "Find text", native("Ctrl+F")) in rows[len(menus):]  # the other keys in those words too
        assert any(where == "View → Workspace" and label == "Estimate" for where, label, _ in menus)
        by_key = defaultdict(set)
        for _, label, keys in menus:
            for key in keys.split(", "):
                by_key[key].add(label)
        assert {k: v for k, v in by_key.items() if len(v) > 1} == {}  # a shared key would fire neither
        dialog = shortcuts_dialog(window)
        assert dialog.table.rowCount() == len(rows)
        dialog.search.setText("recalc")  # the filter narrows the table
        shown = [r for r in range(dialog.table.rowCount()) if not dialog.table.isRowHidden(r)]
        assert [dialog.table.item(r, 1).text() for r in shown] == ["Recalculate"]
        assert any(a.text().replace("&", "") == "Keyboard shortcuts" for a in window.help_menu.actions())
    finally:
        window.close()


def test_symbol_buttons_have_names_for_screen_readers(app, data):
    from pymetrix.ui.sheet_qt import SheetWindow

    line = SheetWindow("data", lambda: data, lambda cmd: True).edit_line
    assert (line.accept_button.accessibleName(), line.cancel_button.accessibleName()) == ("Accept value", "Put value back")
    assert line.line.accessibleName() == "Cell value"


def test_a_step_cancelled_before_its_progress_shows_says_cancelling(app):
    from PySide6.QtWidgets import QPushButton

    from pymetrix.ui import qt
    from pymetrix.ui.qt import main_window

    window = main_window(Session())
    try:
        t = Task()
        t.cancel()  # from its dialog, in the first moments
        qt._PROGRESS["show"](t, "Slow")
        button = window.progress_box.findChild(QPushButton)
        assert button.text() == "Cancelling…" and not button.isEnabled()
        qt._PROGRESS["hide"]()
    finally:
        window.close()


def test_a_step_that_ignores_cancel_says_it_was_recorded(app, data, monkeypatch):
    from PySide6.QtWidgets import QDialogButtonBox, QMessageBox

    from pymetrix.ui import qt
    from pymetrix.ui.qt import command_dialog, run_blocking

    s = Session()
    s.objects["data"] = data
    said = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: said.append(a[1]))

    def cancelled_but_finished(fn, label=""):
        def step():
            qt._BUSY["task"].cancel()  # Cancel pressed; the step has no check, so it finishes
            return fn()
        return run_blocking(step, label)

    monkeypatch.setattr(qt, "run_blocking", cancelled_but_finished)
    dialog = command_dialog(FitEquation, s, initial=FitEquation("gc ~ gy", name="fit"))
    dialog.findChild(QDialogButtonBox).accepted.emit()
    assert said == ["Finished before stopping"] and "fit" in s.objects and dialog.result() == 1
    said.clear()
    monkeypatch.setattr(qt, "run_blocking", run_blocking)
    dialog = command_dialog(FitEquation, s, initial=FitEquation("gc ~ gy", name="fit2"))
    dialog.findChild(QDialogButtonBox).accepted.emit()
    assert said == [] and "fit2" in s.objects  # an ordinary step: nothing said
