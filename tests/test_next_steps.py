"""Next steps: a result window's Next menu opens the next stage's dialog with the result filled in."""

import os
from dataclasses import fields

import pytest

from pymetrix import Dataset
from pymetrix.commands import COMMANDS, FitEquation, Session
from pymetrix.ui.next_steps import NEXT_STEPS, prefilled, steps_for
from pymetrix.ui.workspaces import WORKSPACES


def test_every_next_step_names_a_dialog_a_field_and_a_workspace():
    by_class = {i.cls.__name__: i.cls for i in COMMANDS.values()}
    stages = {w.key for w in WORKSPACES}
    for kind, steps in NEXT_STEPS.items():
        for label, cls, field, stage in steps:
            assert cls in by_class and field in {f.name for f in fields(by_class[cls])}, (kind, cls, field)
            assert stage in stages and label.endswith("…")
            command = prefilled(by_class[cls], **{field: "x"})
            assert command is not None and getattr(command, field) == "x"  # the dialog opens filled in


def test_steps_follow_the_type_of_the_result():
    data = Dataset.example("sample_macro")
    s = Session()
    s.objects["data"] = data
    s.run(FitEquation("dlog(c) ~ dlog(y)", name="fit"))
    assert [label for label, *_ in steps_for(data)][:2] == ["Plot series…", "Check data…"]
    assert [label for label, *_ in steps_for(s["fit"])] == ["Test…", "Coefficient plot…", "Forecast…", "Backtest…", "Solve as a model…", "As a matrix…"]
    assert steps_for(object()) == []


@pytest.fixture
def window():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from PySide6.QtCore import QSettings

    from pymetrix.ui.qt import main_window

    settings = QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope, "pymetrix", "desktop")
    settings.remove("workspace")
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    s.run(FitEquation("dlog(c) ~ dlog(y)", name="fit"))
    win = main_window(s)
    yield win, s
    s.dirty = False
    win.close()
    settings.remove("workspace")


def test_the_next_menu_opens_the_step_filled_in_and_in_its_workspace(window, monkeypatch):
    from pymetrix.ui import qt

    win, s = window
    opened = []
    monkeypatch.setattr(qt, "command_dialog", lambda cls, session, parent=None, initial=None: opened.append(initial) or
                        type("Shut", (), {"exec": lambda self: False})())
    win.show_object("fit")
    menu = win.mdi.subWindowList()[-1].widget().next_menu
    labels = [a.text() for a in menu.actions()]
    assert labels == ["Test…", "Coefficient plot…", "Forecast…", "Backtest…", "Solve as a model…", "As a matrix…"]
    menu.actions()[labels.index("Forecast…")].trigger()  # in All: stays in All
    assert type(opened[-1]).__name__ == "Forecast" and opened[-1].fit == "fit" and win.workspace == "all"
    win.switch_workspace("estimate")
    menu.actions()[labels.index("Backtest…")].trigger()  # from a stage: to the step's own
    assert type(opened[-1]).__name__ == "RunBacktest" and opened[-1].fit == "fit" and win.workspace == "forecast"
    win.show_object("data")
    data_menu = win.mdi.subWindowList()[-1].widget().next_menu  # the data's window: its header and Next
    data_labels = [a.text() for a in data_menu.actions()]
    data_menu.actions()[data_labels.index("Estimate an equation…")].trigger()
    assert type(opened[-1]).__name__ == "FitEquation" and opened[-1].data == "data" and opened[-1].formula == "" and win.workspace == "estimate"
    win.switch_workspace("all")
