"""The start page (recent projects, demos), opening dropped files, and the stage bar's badges."""

import os

import pytest

from pymetrix import Dataset
from pymetrix.commands import FitEquation, GenerateSeries, Session, save_project
from pymetrix.ui.workspaces import stage_of


def test_results_belong_to_stages():
    from pymetrix import Equation
    from pymetrix.figures import Figure, Line, Panel

    data = Dataset.example("sample_macro")
    assert stage_of(data) == "explore" and stage_of(Equation("c ~ y").fit(data)) == "estimate"
    assert stage_of(Figure((Panel((Line(data["c"], "c"),)),))) == "report" and stage_of(object()) == "estimate"


@pytest.fixture
def app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _settings():
    from PySide6.QtCore import QSettings

    return QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope, "pymetrix", "desktop")


def test_the_start_page_lists_recent_projects_and_demos(app, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from pymetrix.ui.qt import main_window

    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    project = save_project(s, tmp_path / "earlier.pymx")
    _settings().setValue("recent_projects", [str(project)])
    window = main_window(Session())
    try:
        page = window.start_page
        recent = page.lists["Recent projects"]
        assert recent.entries[0].text() == "earlier.pymx" and recent.entries[0].toolTip() == str(project)
        demos = page.lists["Demo projects"]
        assert len(demos.entries) >= 1 and ":" in demos.entries[0].text()
        QTest.mouseClick(recent.entries[0], Qt.MouseButton.LeftButton)  # a click opens it
        assert "data" in window.state["session"].objects
    finally:
        window.close()
        _settings().remove("recent_projects")


def test_the_start_page_lists_open_with_the_keyboard(app):
    """Tab reaches the entries, Up and Down move between them, Enter opens one."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from pymetrix.demos import demos as all_demos
    from pymetrix.ui.qt import main_window

    window = main_window(Session())
    opened = []
    try:
        window.show()
        page = window.start_page
        demos = page.lists["Demo projects"]
        assert all(e.focusPolicy() & Qt.FocusPolicy.TabFocus for e in demos.entries)
        demos.activated.disconnect()  # the row opened, not the demo itself
        demos.activated.connect(opened.append)
        demos.entries[0].setFocus()
        QTest.keyClick(demos.entries[0], Qt.Key.Key_Down)
        assert window.focusWidget() is demos.entries[1]  # the window's focus (it may not be the active one)
        QTest.keyClick(demos.entries[1], Qt.Key.Key_Up)
        QTest.keyClick(demos.entries[0], Qt.Key.Key_Up)  # the first stays the first
        assert window.focusWidget() is demos.entries[0]
        QTest.keyClick(demos.entries[0], Qt.Key.Key_Down)
        QTest.keyClick(demos.entries[1], Qt.Key.Key_Return)
        assert opened == [1]
        names = list(all_demos())
        for entry, name in zip(demos.entries, names):
            assert entry.toolTip().startswith(f"{name}: ")
    finally:
        window.close()
        _settings().remove("recent_projects")


def test_dropped_files_open(app, tmp_path, monkeypatch):
    from PySide6.QtCore import QMimeData, QUrl

    from pymetrix.ui import qt
    from pymetrix.ui.qt import main_window

    csv = tmp_path / "2024 prices.csv"
    csv.write_text("date,x\n2020Q1,1\n", encoding="utf-8")
    window = main_window(Session())
    try:
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(csv)), QUrl.fromLocalFile(str(tmp_path / "notes.docx"))])
        assert window.dropped_files(mime) == [str(csv).replace("\\", "/")] or window.dropped_files(mime) == [str(csv)]
        opened = []
        monkeypatch.setattr(qt, "command_dialog", lambda cls, session, parent=None, initial=None: opened.append(initial) or
                            type("Shut", (), {"exec": lambda self: False})())
        window.open_dropped(window.dropped_files(mime))  # a data file: Load data, filled in
        assert type(opened[-1]).__name__ == "LoadData" and opened[-1].name == "data_2024_prices"
        assert opened[-1].path.replace("\\", "/").endswith("2024 prices.csv")
        empty = QMimeData()
        assert window.dropped_files(empty) == []
    finally:
        window.close()


def test_the_stage_bar_counts_what_needs_attention(app):
    from PySide6.QtWidgets import QDockWidget

    from pymetrix.ui.qt import main_window

    _settings().remove("workspace")
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    s.run(FitEquation("dlog(c) ~ dlog(y)", name="fit"))
    window = main_window(s)
    try:
        actions = window.workspace_actions
        assert actions["estimate"].text() == "Estimate" and actions["all"].text() == "All"
        s.run(GenerateSeries("y", "y * 1.01", replace=True))  # a revision: the fit is out of date
        window.refresh()  # the window counts what each stage needs
        assert "fit" in s.stale()
        assert actions["estimate"].text() == "Estimate · 1" and actions["all"].text() == "All · 1"
        assert "fit" in actions["estimate"].toolTip() and actions["forecast"].text() == "Forecast"
        problems = next(d for d in window.findChildren(QDockWidget) if d.windowTitle().startswith("Problems"))  # with its count
        problems.setVisible(False)
        actions["estimate"].trigger()  # a stage with a badge: Problems says which
        assert window.workspace == "estimate" and problems.isVisibleTo(window)
        window.update_badges({})
        assert actions["estimate"].text() == "Estimate"
        window.switch_workspace("all")
    finally:
        s.dirty = False
        window.close()
        _settings().remove("workspace")
