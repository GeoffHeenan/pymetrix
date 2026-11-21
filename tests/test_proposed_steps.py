"""A step an assistant proposed and the user ran from its dialog (extension API 1.9): marked as proposed, saved and
reopened with the project, noted in the script; a recalculation after it, and other steps, are not marked."""

import pytest

from pymetrix.commands import (
    FitEquation,
    GenerateSeries,
    LoadExample,
    Session,
    load_project,
    save_project,
)


@pytest.fixture
def session():
    s = Session()
    s.run(LoadExample())
    return s


def test_the_proposed_step_is_marked_and_nothing_else(session):
    with session.proposing(FitEquation, "Claude"):
        session.run(GenerateSeries("z", "y * 2", data="data"))  # another dialog: not the one proposed
        session.run(FitEquation("c ~ y", data="data", name="fit"))
        session.run(FitEquation("c ~ y + i", data="data", name="fit2"))  # only the first one run
    marks = [r.proposed_by for r in session.history]
    assert marks == ["", "", "Claude", ""]
    session.run(FitEquation("c ~ i", data="data", name="fit3"))
    assert session.history[-1].proposed_by == ""  # outside the block


def test_the_mark_is_saved_and_reopened_and_in_the_script(session, tmp_path):
    with session.proposing(FitEquation, "Claude"):
        session.run(FitEquation("c ~ y", data="data", name="fit"))
    script = session.script()
    assert "# proposed by Claude (an assistant); run from its dialog\n" in script
    assert script.index("# proposed by Claude") < script.index("fit = ")
    path = save_project(session, tmp_path / "p.pymx")
    opened = load_project(path)
    assert [r.proposed_by for r in opened.history] == ["", "Claude"]
    again = load_project(save_project(opened, tmp_path / "q.pymx"))
    assert again.history[-1].proposed_by == "Claude"


def test_who_proposed_must_be_said(session):
    from pymetrix.errors import SpecificationError

    with pytest.raises(SpecificationError), session.proposing(FitEquation, "  "):
        pass


def test_history_says_proposed(session):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.history_qt import HistoryPanel

    with session.proposing(FitEquation, "Claude"):
        session.run(FitEquation("c ~ y", data="data", name="fit"))
    panel = HistoryPanel(print, print, print, print)
    panel.show_session(session)
    last = panel.topLevelItem(1)
    assert last.text(1).endswith("· proposed") and "Proposed by Claude" in last.toolTip(1)
    assert not panel.topLevelItem(0).text(1).endswith("proposed")
