"""A failed dialog step that the same dialog then runs successfully on the same target is resolved: it stays in the
history (greyed in the History window) but is no longer counted in Problems. Saving and replay are unchanged (failed
steps are never saved), and undoing the successful step makes the failure count again."""

import os

import pytest

from pymetrix import Dataset
from pymetrix.commands import FitEquation, GenerateSeries, Session, load_project, save_project


@pytest.fixture
def session():
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    return s


def _fail(s, cmd):
    with pytest.raises(Exception):
        s.run(cmd)
    assert not s.history[-1].ok


def test_a_successful_retry_resolves_the_failure(session, tmp_path):
    s = session
    _fail(s, FitEquation("c ~ log(-y)", name="fit"))  # fails when run: recorded as failed
    _fail(s, FitEquation("c ~ log(-y)", name="other"))  # another target: not resolved by a fit of "fit"
    _fail(s, GenerateSeries("z", "nope * 2"))  # another dialog
    assert s.resolved_failures() == {}
    s.run(FitEquation("c ~ y", name="fit"))
    assert s.resolved_failures() == {0: 3}
    assert [r.ok for r in s.history] == [False, False, False, True]  # all kept

    again = load_project(save_project(s, tmp_path / "p.pymx"))
    assert [r.ok for r in again.history] == [True] and again.resolved_failures() == {}  # failed steps not saved

    s.undo()  # the retry undone: the failure counts again
    assert s.resolved_failures() == {}


def test_typed_code_is_never_resolved(session):
    s = session
    with pytest.raises(Exception):
        s.run_code("x = undefined_name")
    assert not s.history[-1].ok
    s.run_code("x = undefined_name if False else 1")
    assert s.history[-1].ok and s.resolved_failures() == {}


QtWidgets = pytest.importorskip("PySide6.QtWidgets")


def test_problems_and_history_show_the_resolution(session):
    pytest.importorskip("qtconsole")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import main_window

    s = session
    w = main_window(s)
    try:
        _fail(s, FitEquation("c ~ log(-y)", name="fit"))
        w.switch(s)
        assert any(what.startswith("failed:") for _, what in w.problem_list)
        assert w.history.topLevelItem(0).text(3) == "failed"
        s.run(FitEquation("c ~ y", name="fit"))
        w.switch(s)
        assert not any(what.startswith("failed:") for _, what in w.problem_list)
        first = w.history.topLevelItem(0)
        assert first.text(3) == "failed, resolved" and "step 2" in first.toolTip(3)
        from pymetrix.ui.theme import muted

        assert first.foreground(0).color() == muted()  # greyed: the theme's muted text
    finally:
        s.dirty = False
        w.close()
