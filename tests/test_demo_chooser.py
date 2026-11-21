"""File → Open demo project…: a dialog listing the demos (the Start window's list, not a dropdown), each with what it
shows and the windows it opens; Up and Down move, Enter or a click opens, Open opens the demo with the focus, Esc
cancels. pymetrix-gui --demo NAME and Reload demo open a demo without it."""

import os

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from pymetrix.commands import Session  # noqa: E402
from pymetrix.demos import demos, headline  # noqa: E402
from pymetrix.ui import demo_chooser  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_each_demo_is_listed_with_what_it_shows_and_opens(app):
    dialog = demo_chooser.DemoChooser()
    assert dialog.windowTitle() == "Open demo project" and dialog.names == list(demos())
    assert isinstance(dialog.entries, __import__("pymetrix.ui.fit", fromlist=["EntryList"]).EntryList)
    for name, entry in zip(dialog.names, dialog.entries.entries):
        lines = entry.text().split("\n")
        assert lines[0] == name and lines[1] == demos()[name] and entry.accessibleName() == name
        windows = headline(name)[0]
        assert lines[2].startswith(f"Opens {windows[-1][0]} ({windows[-1][1]})") and windows[0][0] in lines[2]
    assert not dialog.findChildren(QtWidgets.QComboBox)
    dialog.deleteLater()


def test_keys_move_and_enter_opens(app):
    dialog = demo_chooser.DemoChooser()
    dialog.show()
    dialog.activateWindow()
    app.processEvents()
    entries = dialog.entries.entries
    assert app.focusWidget() is entries[0]
    QTest.keyClick(entries[0], Qt.Key.Key_Down)
    QTest.keyClick(entries[1], Qt.Key.Key_Down)
    assert app.focusWidget() is entries[2]
    QTest.keyClick(entries[2], Qt.Key.Key_Up)
    QTest.keyClick(entries[1], Qt.Key.Key_Return)
    assert dialog.chosen == dialog.names[1] and dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
    dialog.deleteLater()


def test_a_click_or_open_opens_and_escape_cancels(app):
    dialog = demo_chooser.DemoChooser()
    dialog.show()
    app.processEvents()
    QTest.mouseClick(dialog.entries.entries[2], Qt.MouseButton.LeftButton)
    assert dialog.chosen == dialog.names[2]
    dialog.deleteLater()
    dialog = demo_chooser.DemoChooser()
    dialog.show()
    app.processEvents()
    dialog.entries.entries[3].setFocus()
    dialog.open_button.click()  # Open: the demo with the focus
    assert dialog.chosen == dialog.names[3]
    dialog.deleteLater()
    dialog = demo_chooser.DemoChooser()
    dialog.show()
    app.processEvents()
    QTest.keyClick(dialog, Qt.Key.Key_Escape)
    assert dialog.chosen is None and dialog.result() == QtWidgets.QDialog.DialogCode.Rejected
    dialog.deleteLater()


def test_choose_demo_returns_the_name_or_none(app, monkeypatch):
    monkeypatch.setattr(QtWidgets.QDialog, "exec", lambda d: d.open_row(1) or 1)
    assert demo_chooser.choose_demo() == list(demos())[1]
    monkeypatch.setattr(QtWidgets.QDialog, "exec", lambda d: d.reject() or 0)
    assert demo_chooser.choose_demo() is None


def test_it_fits_a_laptop_screen_without_sideways_scrolling(app, monkeypatch):
    import _fit_check as fc

    from pymetrix.ui import fit

    monkeypatch.setattr(fit, "_available", lambda widget: QRect(0, 0, 1366, 768))
    for look in fc.LOOKS:
        with fc.look(look):
            dialog = demo_chooser.DemoChooser()
            dialog.show()
            app.processEvents()
            scroll = dialog.findChild(QtWidgets.QScrollArea)
            assert dialog.width() <= 1366 and dialog.height() <= 768 and not scroll.horizontalScrollBar().isVisible(), look
            assert fc.check_sizes(dialog, ((560, 460), (400, 300))) == [], look
            dialog.close()
            dialog.deleteLater()


def test_the_menu_opens_the_chosen_demo_and_cancel_opens_nothing(app, monkeypatch):
    from pymetrix.ui.qt import main_window

    window = main_window(Session())
    try:
        monkeypatch.setattr(demo_chooser, "choose_demo", lambda parent=None: None)
        window.open_demo()
        assert window.state["session"].objects == {}
        monkeypatch.setattr(demo_chooser, "choose_demo", lambda parent=None: "consumption")
        window.open_demo()
        assert "dynamic" in window.state["session"].objects and window.state["session"].demo == "consumption"
        window.state["session"].objects.pop("dynamic")
        window.state["session"].dirty = False
        window.reload_demo()  # Reload demo: as it first was, without asking which
        assert "dynamic" in window.state["session"].objects
    finally:
        window.state["session"].dirty = False
        window.close()


def test_the_command_line_still_names_a_demo():
    from pymetrix.ui.qt import _arguments

    assert _arguments(["--demo", "consumption"]).demo == "consumption"
