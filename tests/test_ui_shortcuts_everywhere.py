"""Undo, Redo and the other menu shortcuts answer in every workspace, whichever menus it shows."""

import os

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from pymetrix import Dataset  # noqa: E402
from pymetrix.commands import GenerateSeries, Session  # noqa: E402


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app):
    from pymetrix.ui.qt import main_window

    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    w = main_window(s)
    w.show()
    w.activateWindow()
    QtWidgets.QApplication.processEvents()
    yield w
    s.dirty = False
    w.close()


def _press(window, key, modifiers):
    window.activateWindow()
    window.mdi.setFocus()
    QtWidgets.QApplication.processEvents()
    QTest.keyClick(window.mdi, key, modifiers)
    QtWidgets.QApplication.processEvents()


def test_ctrl_z_and_ctrl_y_change_the_history_in_every_workspace(window):
    s = window.state["session"]
    for key in window.workspace_actions:
        window.switch_workspace(key)
        n = len(s.history)
        s.run(GenerateSeries(f"z_{key}", "y * 2"))
        window.refresh()
        assert len(s.history) == n + 1
        _press(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        assert len(s.history) == n, key  # Ctrl+Z took the step back
        _press(window, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
        assert len(s.history) == n + 1, key  # and Ctrl+Y ran it again


def test_undo_works_even_with_the_edit_menu_hidden(window):
    s = window.state["session"]
    window.switch_workspace("model")
    window.edit_menu.menuAction().setVisible(False)  # as before Edit joined every workspace
    s.run(GenerateSeries("z", "y * 2"))
    window.refresh()
    _press(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert s.history == []


def test_every_menu_shortcut_is_on_the_window(window):
    from pymetrix.ui.workspaces import ALWAYS_MENUS

    assert "Edit" in ALWAYS_MENUS
    window.follow_focus(None, window.mdi)  # no field with an Undo of its own has focus: Ctrl+Z is the step's
    on_window = set(map(id, window.actions()))
    keyed = []

    def walk(menu):
        for a in menu.actions():
            if a.menu() is not None and a.menu() is not menu:
                walk(a.menu())
            elif any(not k.isEmpty() for k in a.shortcuts()):
                keyed.append(a)

    for top in window.menuBar().actions():
        if top.menu() is not None:
            walk(top.menu())
    texts = {}
    for a in keyed:
        for k in a.shortcuts():
            texts.setdefault(k.toString(), set()).add(id(a))
    for a in keyed:  # Save, Open, Find anything, Recalculate...: each unshared key works from the window itself
        if all(len(texts[k.toString()]) == 1 for k in a.shortcuts() if not k.isEmpty()):
            assert id(a) in on_window, a.text()
    names = {a.text().replace("&", "") for a in keyed if id(a) in on_window}
    assert {"Save project", "Open project…", "Find anything…", "Undo", "Redo"} <= names
