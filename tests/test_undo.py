"""Undo: taking back the last dialog steps, and refusing when that would lose other work."""

import os

import pytest

from pymetrix import Dataset
from pymetrix.commands import FitEquation, GenerateSeries, Session, load_project, save_project
from pymetrix.commands.session import UNDO_STEPS, CommandError


@pytest.fixture
def session():
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    return s


def test_undo_puts_back_the_objects_and_the_history(session):
    s = session
    data = s.objects["data"]
    s.run(GenerateSeries("z", "y * 2"))
    s.run(FitEquation("c ~ z", name="fit"))
    assert s.undo_label() == "fit = Equation('c ~ z').fit(data, method=OLS())"
    undone = s.undo()
    assert type(undone.command).__name__ == "FitEquation" and "fit" not in s.objects and len(s.history) == 1
    assert "fit" not in s.namespace  # typed code sees the same objects
    s.undo()
    assert s.objects["data"] is data and s.history == [] and not s.stale()  # the very object from before
    assert s.undo_blocked() is not None and s.undo_label() == ""
    with pytest.raises(CommandError, match="no step to undo"):
        s.undo()


def test_undo_takes_back_a_replaced_result_and_its_recalculation(session):
    s = session
    s.run(FitEquation("c ~ y", name="fit"))
    first = s.objects["fit"]
    s.set_mode("automatic")
    s.run(GenerateSeries("y", "y * 1.01", replace=True))  # the fit is recomputed after it
    assert s.objects["fit"] is not first and len(s.history) == 3
    s.undo()  # the revision and the recalculation it caused
    assert s.objects["fit"] is first and len(s.history) == 1 and not s.stale()


def test_undo_refuses_when_objects_changed_since(session):
    s = session
    s.run(FitEquation("c ~ y", name="fit"))
    s.objects["note"] = 3  # by hand
    assert "changed since" in s.undo_blocked()
    del s.objects["note"]
    assert s.undo_blocked() is None  # as the step left them again
    before = s.begin_code()
    s.namespace["x"] = 1
    s.end_code(before, "x = 1", True)  # typed code: the steps before it cannot be undone
    assert s.undo_blocked() is not None and "fit" in s.objects


def test_a_failed_step_is_not_undone_and_the_stack_is_bounded(session):
    s = session
    s.run(GenerateSeries("z", "y * 2"))
    with pytest.raises(Exception):
        s.run(FitEquation("c ~ nonexistent", name="bad"))
    assert type(s.history[len(s._undo[-1]["state"]["history"])].command).__name__ == "GenerateSeries"
    s.undo()  # the last successful step, with the failed record after it
    assert s.history == []
    for k in range(UNDO_STEPS + 5):
        s.run(GenerateSeries(f"z{k}", "y"))
    assert len(s._undo) == UNDO_STEPS


def test_an_opened_project_starts_with_nothing_to_undo(session, tmp_path):
    session.run(FitEquation("c ~ y", name="fit"))
    opened = load_project(save_project(session, tmp_path / "p.pymx"))
    assert opened.undo_blocked() is not None and "fit" in opened.objects


@pytest.fixture
def app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_edit_undo_in_the_window(app, session):
    from PySide6.QtWidgets import QLineEdit

    from pymetrix.ui.qt import main_window

    window = main_window(session)
    try:
        titles = [a.text().replace("&", "") for a in window.menuBar().actions()]
        assert titles[:2] == ["File", "Edit"]
        assert not window.undo_action.isEnabled() and window.undo_action.text() == "Undo"
        session.run(FitEquation("c ~ y", name="fit"))
        window.refresh()
        assert window.undo_action.isEnabled() and window.undo_action.text().startswith("Undo fit = ")
        window.undo_step()
        assert "fit" not in session.objects and window.statusBar().currentMessage().startswith("Undid fit = ")
        assert not window.undo_action.isEnabled()
        window.undo_step()  # nothing left: says why, changes nothing
        assert window.statusBar().currentMessage().startswith("Cannot undo")
        field = QLineEdit(window)
        window.follow_focus(None, field)  # a field has its own Undo: Ctrl+Z is its
        assert window.undo_action.shortcut().isEmpty()
        window.follow_focus(field, window.mdi)
        assert window.undo_action.shortcut().toString() == "Ctrl+Z"
    finally:
        session.dirty = False
        window.close()


def test_undo_refuses_after_the_history_is_cleaned_up_or_a_freeze(session):
    s = session
    s.run(FitEquation("c ~ y", name="fit"))
    s.run(FitEquation("c ~ y", name="fit"))  # the first fit is now a dead end
    s.run(GenerateSeries("z", "y * 2"))
    assert s.prune()  # Clean up history: the records Undo saved no longer match
    assert "history has changed" in s.undo_blocked() and s.undo_label() == "" and s.undo_changes() == set()
    s.run(GenerateSeries("w", "y"))
    s.freeze("fit")  # after the step: Undo would quietly drop it
    assert "frozen or unfrozen" in s.undo_blocked()
    s.unfreeze("fit")
    assert s.undo_blocked() is None and s.undo_changes() == {"data"}  # the dataset Undo would replace


def test_ctrl_z_belongs_to_a_field_only_while_it_can_undo(app, session, monkeypatch):
    from PySide6.QtWidgets import QMessageBox, QPlainTextEdit

    from pymetrix.ui.qt import main_window

    window = main_window(session)
    try:
        read_only = QPlainTextEdit(window)
        read_only.setReadOnly(True)  # as the Script panel and help are
        window.follow_focus(None, read_only)
        assert window.undo_action.shortcut().toString() == "Ctrl+Z"
        session.run(GenerateSeries("z", "y * 2"))
        window.refresh()
        sheet = window.open_sheet("data")
        window.follow_focus(None, sheet.table)  # nothing typed: Ctrl+Z undoes the step
        assert window.undo_action.shortcut().toString() == "Ctrl+Z"
        assert not sheet.set_cells({(0, sheet.model.columns.index("c")): "123"}, "Edit cell")  # the sheet's own Undo has it
        assert sheet.undo.canUndo()
        window.follow_focus(None, sheet.table)
        assert window.undo_action.shortcut().isEmpty()
        asked = []
        with monkeypatch.context() as patch:  # puts the original back (del would delete QMessageBox.question for good)
            patch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[1]) or QMessageBox.StandardButton.No)
            window.undo_step()  # the sheet's edits would be lost: asked, and No keeps everything
        assert asked == ["Unapplied edits"] and "z" in session.objects["data"].frame.columns and sheet.model.pending
        sheet.model.pending.clear()
    finally:
        session.dirty = False
        window.close()
