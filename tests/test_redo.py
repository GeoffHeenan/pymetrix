"""Redo: running again the steps Undo took back, in order, and refusing when that would lose other work."""

import os

import pytest

from pymetrix import Dataset
from pymetrix.commands import FitEquation, GenerateSeries, Session, load_project, save_project
from pymetrix.commands.session import CommandError


@pytest.fixture
def session():
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    return s


def test_undo_then_redo_restores_the_objects_and_the_script(session):
    s = session
    s.run(GenerateSeries("z", "y * 2"))
    s.run(FitEquation("c ~ z", name="fit"))
    script, params = s.script(), s.objects["fit"].params.as_dict()
    s.undo()
    s.undo()
    assert "fit" not in s.objects and "z" not in s.objects["data"].names
    assert s.redo_label() == "data = data.generate('z', 'y * 2')" or s.redo_label().startswith("data")
    first = s.redo()
    assert type(first.command).__name__ == "GenerateSeries" and "z" in s.objects["data"].names
    assert s.redo_label().startswith("fit = ")
    s.redo()
    assert s.script() == script
    assert s.objects["fit"].params.as_dict() == pytest.approx(params)
    assert s.redo_blocked() is not None and s.redo_label() == ""
    with pytest.raises(CommandError, match="no step to redo"):
        s.redo()
    s.undo()  # and undo again, after a redo
    assert "fit" not in s.objects


def test_a_new_step_clears_what_can_be_redone(session):
    s = session
    s.run(FitEquation("c ~ y", name="fit"))
    s.undo()
    assert s.redo_label()
    s.run(FitEquation("c ~ i", name="other"))
    assert s.redo_label() == "" and "no step to redo" in s.redo_blocked()


def test_redo_refuses_once_objects_have_changed_since_the_undo(session):
    s = session
    s.run(FitEquation("c ~ y", name="fit"))
    s.undo()
    s.objects["by_hand"] = 1  # changed outside a step
    assert "changed since the undo" in s.redo_blocked()
    with pytest.raises(CommandError, match="changed since the undo"):
        s.redo()


def test_observers_hear_a_redo_as_a_step(session, monkeypatch):
    from pymetrix.commands import events

    heard = []
    original = events.emit
    monkeypatch.setattr(events, "emit", lambda session_, kind, **fields: (heard.append(kind), original(session_, kind, **fields))[1])
    s = session
    s.run(FitEquation("c ~ y", name="fit"))
    s.undo()
    heard.clear()
    s.redo()
    assert heard == ["step"]  # as any step run: the extension API is unchanged


def test_a_project_saved_after_redo_replays_identically(session, tmp_path):
    s = session
    s.run(GenerateSeries("z", "y * 2"))
    s.run(FitEquation("c ~ z", name="fit"))
    s.undo()
    s.redo()
    opened = load_project(save_project(s, tmp_path / "p.pymx"))
    assert opened.script() == s.script()
    assert opened.objects["fit"].params.as_dict() == pytest.approx(s.objects["fit"].params.as_dict())


def test_redo_is_in_the_edit_menu_and_the_shortcuts():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from PySide6 import QtGui

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import main_window, shortcut_list

    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    window = main_window(s)
    keys = [k.toString() for k in window.redo_action.shortcuts()]
    assert "Ctrl+Y" in keys and "Ctrl+Shift+Z" in keys
    rows = shortcut_list(window)
    ctrl_y = QtGui.QKeySequence("Ctrl+Y").toString(QtGui.QKeySequence.SequenceFormat.NativeText)  # ⌘Y on a Mac
    assert any(what == "Redo" and ctrl_y in keys_.split(", ") for _, what, keys_ in rows)
    s.run(FitEquation("c ~ y", name="fit"))
    window.update_undo()
    window.undo_step()
    window.update_undo()
    assert window.redo_action.isEnabled() and window.redo_action.text().startswith("Redo fit = ")
    window.redo_step()
    assert "fit" in s.objects
    window.update_undo()
    assert not window.redo_action.isEnabled() and window.redo_action.text() == "Redo"
    s.dirty = False
    window.close()


def test_figure_edits_in_the_app_undo_and_redo():
    """In the app a drag is a Change a style step: Undo puts the figure back, Redo makes the edit again."""
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.commands import Session
    from pymetrix.ui.qt import main_window

    session = Session()
    session.objects["data"] = Dataset.example("sample_macro")
    window = main_window(session)
    from pymetrix.commands import PlotSeries

    session.run(PlotSeries(series="c", name="fig"))
    from pymetrix.commands import AnnotateFigure

    session.run(AnnotateFigure("fig", what="text", text="words", x="2019Q1", y="115", name="fig"))
    before = session["fig"]
    window.figure_edit("fig", "Figure", before, "restyle", 0, {"x": "2020Q1", "y": 120.0})
    assert session["fig"].style[0].props["x"] == "2020Q1"
    session.undo()
    assert session["fig"].style == before.style
    session.redo()
    assert session["fig"].style[0].props["x"] == "2020Q1"
    window.figure_edit("fig", "Figure", session["fig"], "remove", 0)
    assert session["fig"].style == ()
    session.dirty = False  # closing asks to save otherwise
    window.close()
