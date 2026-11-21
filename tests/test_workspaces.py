"""Workspaces: the app arranged for one stage of the work (Explore, Estimate, Model, Forecast, Report), or All."""

import os

import pytest

from pymetrix.commands import COMMANDS, Session
from pymetrix.ui.workspaces import ALWAYS_MENUS, CODE_DOCKS, WORKSPACES

DOCKS = {"Objects", "History", "Script", "Immediate", "Problems", "Report"}


def test_each_workspace_names_menus_panels_and_commands_that_exist():
    classes = {i.cls.__name__ for i in COMMANDS.values() if i.menu}
    tops = {i.menu.partition("/")[0] for i in COMMANDS.values() if i.menu}
    assert [w.key for w in WORKSPACES] == ["all", "explore", "estimate", "model", "forecast", "report"]  # the order of the work
    for ws in WORKSPACES:
        assert set(ws.menus or ()) <= tops | set(ALWAYS_MENUS), ws.key
        assert set(ws.docks or ()) <= DOCKS, ws.key
        assert {t for t in ws.tools if not t.startswith("@")} <= classes, ws.key
    assert set(CODE_DOCKS) <= DOCKS


@pytest.fixture
def window():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from PySide6.QtCore import QSettings

    from pymetrix.ui.qt import main_window

    _settings().remove("workspace")  # a fresh start, and nothing left for other tests
    s = Session()
    win = main_window(s)
    yield win
    s.dirty = False
    win.close()
    _settings().remove("workspace")


def _settings():
    from PySide6.QtCore import QSettings

    return QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope, "pymetrix", "desktop")  # as the window opens them


def _shown_menus(win):
    return [a.text().replace("&", "") for a in win.menuBar().actions() if a.menu() is not None and a.isVisible()]


def _docks(win):
    from PySide6.QtWidgets import QDockWidget

    return {d.windowTitle(): d for d in win.findChildren(QDockWidget)}


def test_switching_workspaces_shows_their_menus_panels_and_tools(window):
    assert window.workspace == "all" and window.workspace_actions["all"].isChecked()
    everything = _shown_menus(window)
    assert {"File", "Data", "Estimate", "Test", "Model", "Forecast"} <= set(everything)
    window.switch_workspace("model")
    assert _shown_menus(window) == ["File", "Edit", "Data", "Model", "View", "Calculate", "Help"]
    docks = _docks(window)
    assert docks["Problems"].isVisibleTo(window) and not docks["Script"].isVisibleTo(window)
    tools = [a.text() for a in window.workspace_tools.actions()]
    assert tools[:3] == ["Define equation", "Template model", "Check model"]
    window.switch_workspace("forecast")
    assert "Forecast" in _shown_menus(window) and "Model" not in _shown_menus(window)
    assert [a.text() for a in window.workspace_tools.actions()][0] == "Make round"
    window.switch_workspace("report")
    assert [a.text() for a in window.workspace_tools.actions()] == ["Export report", "Plot series", "Save result", "Export script"]
    window.switch_workspace("explore")
    assert "Spreadsheet" in [a.text() for a in window.workspace_tools.actions()] and "Test" in _shown_menus(window)
    entries = {e.title for e in window.palette_entries()}
    assert "Model → Solution → Solve model…" in entries  # Find anything still reaches the menus a workspace hides
    window.switch_workspace("all")
    assert _shown_menus(window) == everything and not window.workspace_tools.isVisibleTo(window)


def test_the_code_toggle_and_each_workspace_remembers_its_panels(window):
    docks = _docks(window)
    window.switch_workspace("estimate")
    window.code_action.setChecked(True)  # code alongside, in any workspace
    assert all(docks[t].isVisibleTo(window) for t in CODE_DOCKS)
    window.code_action.setChecked(False)
    assert not docks["Script"].isVisibleTo(window) and docks["Problems"].isVisibleTo(window)  # Estimate keeps its Problems
    docks["History"].setVisible(False)  # arranged by hand in Estimate
    window.switch_workspace("model")
    assert docks["History"].isVisibleTo(window)  # Model as it first shows
    window.switch_workspace("estimate")
    assert not docks["History"].isVisibleTo(window)  # Estimate as it was left
    window.reset_workspace()
    assert docks["History"].isVisibleTo(window)  # Reset layout: as it first shows
    from PySide6.QtCore import QSettings

    assert _settings().value("workspace/current") == "estimate"  # where the next session starts
    window.workspace_actions["model"].trigger()  # the stage bar's button
    assert window.workspace == "model" and window.workspace_actions["model"].shortcut().toString() == "Ctrl+4"
