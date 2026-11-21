"""A dialog gives the keyboard to its first field; a demo is built in the background, with progress and Cancel."""

import os
import threading

import pytest

from pymetrix import Dataset
from pymetrix.commands import COMMANDS, Session


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.parametrize("name", ["FitEquation", "GenerateSeries", "SetSample", "Forecast"])
def test_a_dialog_opens_with_the_keyboard_on_its_first_field(app, name):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QScrollArea

    from pymetrix.ui.qt import command_dialog

    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    cls = next(i.cls for i in COMMANDS.values() if i.cls.__name__ == name)
    dialog = command_dialog(cls, s)
    try:
        dialog.show()
        app.processEvents()
        focus = app.focusWidget()
        assert focus is not None and not isinstance(focus, QScrollArea)
        first = next(e for e in dialog.editors.values() if e.outer.isVisibleTo(dialog))
        assert focus is first.outer or first.outer.isAncestorOf(focus)
        from pymetrix.ui.fit import FitScrollArea

        form = [a for a in dialog.findChildren(FitScrollArea) if a.isAncestorOf(focus)]
        assert form and all(a.focusPolicy() == Qt.FocusPolicy.NoFocus for a in form)  # not a Tab stop before the field
    finally:
        dialog.close()


def test_a_demo_is_built_in_the_background_and_cancel_leaves_the_session(app, monkeypatch):
    import importlib

    from pymetrix.progress import Cancelled
    from pymetrix.ui.qt import main_window

    module = importlib.import_module("pymetrix.demos")
    seen = {}
    real = module.demo

    def watched(name):
        seen["thread"] = threading.current_thread() is not threading.main_thread()
        return real(name)

    monkeypatch.setattr(module, "demo", watched)
    window = main_window(Session())
    try:
        window.open_demo("consumption")
        assert seen["thread"]  # a worker: the window stays live, with progress and Cancel after a moment
        assert "dynamic" in window.state["session"].objects
        before = window.state["session"]
        before.dirty = False

        def cancelled(name):
            raise Cancelled("stopped")

        monkeypatch.setattr(module, "demo", cancelled)
        window.open_demo("forecast_round")
        assert window.state["session"] is before and "not opened: cancelled" in window.statusBar().currentMessage()
    finally:
        window.state["session"].dirty = False
        window.close()
