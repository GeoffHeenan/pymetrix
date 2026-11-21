"""Command dialogs before and after OK: OK waits for the fields marked * (and says which), a step whose call cannot be
built is never run or recorded, an object field with nothing to pick says so in words with a way to make one, and a
failed step's message is titled after the dialog, in plain words, with the technical text under Details."""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pymetrix import Dataset  # noqa: E402
from pymetrix.commands import COMMANDS, Forecast, Session  # noqa: E402
from pymetrix.forms import form_spec, missing_fields, needs_value  # noqa: E402
from pymetrix.ui import messages  # noqa: E402


@pytest.fixture(scope="module")
def app():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import main_window

    host = main_window(Session())  # plugins' commands join the registry as the window loads them
    host.hide()
    yield app
    host.state["session"].dirty = False
    host.close()


def _ok(dialog):
    return dialog.findChild(QtWidgets.QDialogButtonBox).button(QtWidgets.QDialogButtonBox.StandardButton.Ok)


def _dialog(cls, session):
    from pymetrix.ui.qt import command_dialog

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return command_dialog(cls, session)


def _fill(dialog, spec, data, path):
    """A plausible value for an empty field, as a user would give it; False when the form offers nothing to pick."""
    editor = dialog.editors[spec.name]
    w = editor.widget
    if spec.kind in ("file", "savefile"):
        editor.set(str(path))
    elif spec.kind == "period":
        editor.set(str(data.frame.index[4]))
    elif spec.multiple:
        if not w.count():
            return False
        editor.set(w.item(0).text())
    elif isinstance(w, QtWidgets.QComboBox):
        options = [w.itemText(i) for i in range(w.count()) if w.itemText(i) and not w.itemText(i).startswith("(")]
        if not options:
            return False
        editor.set(options[0])
    else:
        editor.set("y" if spec.kind in ("text", "multiline") else "c ~ y")
    QtWidgets.QApplication.processEvents()
    return True


def test_every_dialog_holds_ok_until_its_marked_fields_are_filled(app, tmp_path):
    """Across every registered dialog: the * marks exactly the fields the step needs; with one of them empty OK is
    disabled and the line above the buttons names it; filled, the hold is gone."""
    session = Session()
    data = Dataset.example("sample_macro")
    session.objects["data"] = data
    path = tmp_path / "file.csv"
    path.write_text("date,x\n2000Q1,1\n", encoding="utf-8")
    held = released = 0
    for key, info in COMMANDS.items():
        dialog = _dialog(info.cls, session)
        try:
            specs = {s.name: s for s in form_spec(info.cls)}
            for name, spec in specs.items():
                label = dialog.row_label(name)
                if label is not None and hasattr(label, "text"):
                    assert label.text().endswith(" *") == needs_value(spec), (key, name)
            values = {k: e.get() for k, e in dialog.editors.items()}
            missing = missing_fields(list(specs.values()), values)
            if not missing:
                assert "required" not in dialog.ok_gate.reasons, key
                continue
            held += 1
            assert not _ok(dialog).isEnabled(), key
            note = dialog.needed_note.text()
            assert all(s.label in note for s in missing), (key, note)
            assert _ok(dialog).toolTip(), key  # the reason, on OK too
            if all(_fill(dialog, s, data, path) for s in missing):
                values = {k: e.get() for k, e in dialog.editors.items()}
                if not missing_fields(list(specs.values()), values):
                    assert "required" not in dialog.ok_gate.reasons, key
                    released += 1
        finally:
            dialog.close()
            dialog.deleteLater()
    assert held >= 100 and released >= 60, (held, released)  # most dialogs have such fields, and most can be filled here


def test_ok_with_a_marked_field_empty_runs_and_records_nothing(app):
    session = Session()
    session.objects["data"] = Dataset.example("sample_macro")
    dialog = _dialog(Forecast, session)
    assert not _ok(dialog).isEnabled()
    dialog.findChild(QtWidgets.QDialogButtonBox).accepted.emit()  # OK forced (Enter in a field, a script)
    assert session.history == [] and dialog.result_value is None and getattr(dialog, "last_error", None) is None
    assert "Estimated model" in dialog.needed_note.text() and dialog.needed_note.isVisibleTo(dialog)  # its display label
    dialog.close()


def test_a_step_that_cannot_be_written_as_python_is_not_recorded():
    from pymetrix.commands.session import CommandError

    session = Session()
    session.objects["data"] = Dataset.example("sample_macro")
    with pytest.raises(CommandError, match="could not be written as Python"):
        session.run(Forecast("", "", ""))  # forecast = .forecast(data, ...): not Python
    assert session.history == []  # no failed record either: nothing in History or Problems


def test_a_field_whose_empty_means_something_is_not_marked(app):
    from pymetrix.commands.builtin import SolveModel

    specs = {s.name: s for s in form_spec(SolveModel)}
    assert not needs_value(specs["start"]) and not needs_value(specs["end"])  # empty: the sample's first and last periods
    assert needs_value(specs["equations"])


# --- nothing to pick ---------------------------------------------------------------------------------------------------

def test_a_forecast_with_no_estimated_model_says_so_in_words_at_a_normal_width(app):
    session = Session()
    session.objects["data"] = Dataset.example("sample_macro")
    dialog = _dialog(Forecast, session)
    dialog.show()
    QtWidgets.QApplication.processEvents()
    picker = dialog.pickers["fit"]
    assert picker.widget.isVisibleTo(dialog)
    text = picker.label.text()
    assert text.startswith("No estimated models yet") and "Estimate → Equation…" in text and "Fitted" not in text
    assert "Fitted" not in dialog.editors["fit"].widget.placeholderText()
    assert dialog.width() < 900  # it opened 1,278 px wide, naming every class it takes
    assert not _ok(dialog).isEnabled()
    dialog.close()


def test_every_object_picker_with_nothing_to_pick_has_an_empty_state(app):
    """An empty session: every object field says what is missing in words (no class names), and offers the dialog
    that makes one where pymetrix knows it."""
    session = Session()
    seen = 0
    for key, info in COMMANDS.items():
        dialog = _dialog(info.cls, session)
        dialog.show_advanced(True)  # a picker in the Advanced settings section shows with it
        try:
            for name, picker in dialog.pickers.items():
                if not dialog.row_visible(name):
                    continue
                text = picker.label.text()
                if picker.spec.empty:  # empty means something there: no line
                    assert picker.widget.isHidden(), (key, name)
                    continue
                seen += 1
                assert not picker.widget.isHidden() and text.startswith("No "), (key, name)
                assert "Fitted" not in text and "," not in text.split(":")[0].replace(", ", " "), (key, name, text)
        finally:
            dialog.close()
            dialog.deleteLater()
    assert seen >= 150


def test_the_pickers_button_opens_the_dialog_that_makes_one(app, monkeypatch):
    from pymetrix.ui import qt

    window = qt.main_window(Session())
    window.state["session"].objects["data"] = Dataset.example("sample_macro")
    opened = []
    real = qt.command_dialog

    def recording(cls, session, parent=None, initial=None):
        dialog = real(cls, session, parent, initial)
        opened.append(cls.__name__)
        if cls is Forecast:
            dialog.exec = lambda: (dialog.pickers["fit"].button.click(), 0)[1]
        else:
            dialog.exec = lambda: 0
        return dialog

    monkeypatch.setattr(qt, "command_dialog", recording)
    try:
        window.app_interface.open_dialog(Forecast)  # as the menu opens it
        assert opened == ["Forecast", "FitEquation"]
    finally:
        window.state["session"].dirty = False
        window.close()


# --- a failed step, in plain words ------------------------------------------------------------------------------------

def _raised(fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        return exc
    raise AssertionError("no error")


@pytest.mark.parametrize("make, says, hint", [
    (lambda: open(".", encoding="utf-8"), "Choose a file, not a folder", "Browse"),
    (lambda: np.max(np.array([])), "No data left after removing missing values in the chosen sample (2020Q1–2019Q4)", "widen"),
    (lambda: np.concatenate([]), "No data left after removing missing values", "widen"),
    (lambda: pd.Series(range(100))[np.zeros(0, bool)], "The series don't cover the sample (2020Q1–2019Q4)", "Set sample"),
    (lambda: 1 / 0, "divided by zero", "Check the series"),
])
def test_low_level_errors_become_plain_messages(make, says, hint):
    from pymetrix.ui.messages import explain

    exc = _raised(make)
    found = explain(exc, {"start": "2020Q1", "end": "2019Q4", "path": "."})
    assert found.text.startswith(says) or says in found.text, found.text
    assert hint in found.hint
    assert str(exc) in found.details and type(exc).__name__ in found.details  # the technical text, under Details


def test_a_folder_refused_as_a_file_says_choose_a_file():
    """On Windows, opening a folder as a file is refused (PermissionError, errno 13) rather than IsADirectoryError:
    a folder is still named as one; a file that is refused is still said to be refused."""
    from pymetrix.ui.messages import explain

    found = explain(PermissionError(13, "Permission denied", "."), {"path": "."})
    assert found.text.startswith("Choose a file, not a folder") and "Browse" in found.hint, found.text
    assert "PermissionError" in found.details
    found = explain(PermissionError(13, "Permission denied", __file__), {"path": __file__})
    assert found.text == "The file could not be opened: permission was refused.", found.text


def test_pymetrixs_own_messages_are_kept():
    from pymetrix.errors import NotFoundError, SpecificationError
    from pymetrix.ui.messages import explain

    assert explain(SpecificationError("give at least one shock")).text == "Give at least one shock"
    assert explain(NotFoundError("no object named 'x'")).text == "No object named 'x'"  # without the KeyError's quotes


def test_a_failed_step_is_titled_after_its_dialog_with_details(app, tmp_path):
    from pymetrix.commands.builtin import FitEquation as Fit

    session = Session()
    data = Dataset.example("sample_macro")
    data.frame.loc[:, "y"] = np.nan  # nothing left once missing values go
    session.objects["data"] = data
    dialog = _dialog(Fit, session)
    dialog.editors["formula"].set("c ~ y")
    QtWidgets.QApplication.processEvents()
    assert _ok(dialog).isEnabled()
    _ok(dialog).click()
    QtWidgets.QApplication.processEvents()
    box = dialog.last_error
    title = f"{dialog.windowTitle()}: could not run"
    if messages.TITLES_SHOWN:
        assert box.windowTitle() == title and box.windowTitle() != "Error" and box.text() == box.explained.text
    else:  # a Mac shows no title on a message box: the title is the text's first line instead
        assert box.windowTitle() == "" and box.text() == f"{title}\n\n{box.explained.text}"
    assert box.text()[0].isupper() and box.explained.text[0].isupper() and box.detailedText()
    assert dialog.result_value is None and dialog.result() != QtWidgets.QDialog.DialogCode.Accepted  # it stays, to correct
    box.close()
    dialog.close()


def test_a_folder_for_a_file_says_choose_a_file(app):
    info = next(i for i in COMMANDS.values() if i.cls.__name__ == "OpenResult")
    session = Session()
    dialog = _dialog(info.cls, session)
    dialog.editors["path"].set(".")
    QtWidgets.QApplication.processEvents()
    _ok(dialog).click()
    QtWidgets.QApplication.processEvents()
    box = dialog.last_error
    assert box.explained.text.startswith("Choose a file, not a folder") and "Browse" in box.informativeText()
    assert box.text().endswith(box.explained.text)  # after the title on a Mac (see the test above), alone elsewhere
    # IsADirectoryError ([Errno 21] Is a directory) on Linux and macOS; on Windows, PermissionError ([Errno 13])
    assert "IsADirectoryError" in box.detailedText() or "PermissionError" in box.detailedText(), box.detailedText()
    shown = box.windowTitle() if messages.TITLES_SHOWN else box.text().partition("\n\n")[0]
    assert shown.endswith(": could not run")
    box.close()
    dialog.close()


def test_message_box_titles_are_said_where_the_platform_shows_none(app, monkeypatch):
    """Qt's QMessageBox drops its title on macOS (as Apple's guidelines ask); messages.TITLES_SHOWN says so on each
    platform, and where it is False the title leads the text, so that "Could not save the project" is not lost."""
    box = QtWidgets.QMessageBox()
    box.setWindowTitle("Could not save the project")
    assert (box.windowTitle() == "Could not save the project") == messages.TITLES_SHOWN
    assert messages.TITLES_SHOWN == (sys.platform != "darwin")
    assert messages.titled("A", "b") == ("A", "b" if messages.TITLES_SHOWN else "A\n\nb")
    monkeypatch.setattr(messages, "TITLES_SHOWN", False)  # the Mac's way, wherever the tests run
    assert messages.titled("Could not save the project", "Permission denied") == (
        "Could not save the project", "Could not save the project\n\nPermission denied")
    from pymetrix.ui.dialog_checks import error_box

    failed = error_box(None, "Forecast", ValueError("no data"), show=False)
    assert failed.text() == f"Forecast: could not run\n\n{failed.explained.text}" and failed.explained.text
    monkeypatch.setattr(messages, "TITLES_SHOWN", True)
    failed = error_box(None, "Forecast", ValueError("no data"), show=False)
    assert failed.text() == failed.explained.text


def test_a_marked_field_in_the_closed_advanced_section_is_named_so_and_ok_opens_it(app):
    """The two batches together: the missing line names fields by their display labels, says when one is in the
    closed Advanced settings section, and OK opens the section to show it."""
    from dataclasses import dataclass
    from typing import Annotated

    from pymetrix.commands.registry import Command
    from pymetrix.forms import Widget

    @dataclass
    class _Probe(Command):
        lag_rule: Annotated[str, Widget("text", label="Lag rule", advanced=True)]
        note: str = ""

        def run(self, session):  # pragma: no cover - never reached: OK is held
            raise AssertionError("ran")

    session = Session()
    dialog = _dialog(_Probe, session)
    assert not _ok(dialog).isEnabled()
    assert dialog.needed_note.text() == "Fill in Lag rule (in Advanced settings) to run this step."
    assert not dialog.advanced_toggle.isChecked()
    dialog.findChild(QtWidgets.QDialogButtonBox).accepted.emit()
    assert dialog.advanced_toggle.isChecked() and session.history == []
    assert dialog.needed_note.text() == "Fill in Lag rule to run this step."
    dialog.editors["lag_rule"].set("aic")
    assert _ok(dialog).isEnabled() and not dialog.needed_note.text()
    dialog.close()


def test_end_keeps_its_python_hint_after_the_span_check(app):
    from pymetrix.commands.builtin import SolveModel

    session = Session()
    session.objects["data"] = Dataset.example("sample_macro")
    dialog = _dialog(SolveModel, session)
    end = dialog.editors["end"]
    assert "Python: end=" in end.widget.toolTip()
    dialog.editors["start"].set("2010Q1")
    end.set("2005Q1")  # before the start: the problem, and still the hint
    assert "Python: end=" in end.widget.toolTip() and "Python: end=" != end.widget.toolTip().splitlines()[0]
    end.set("2015Q1")
    assert "Python: end=" in end.widget.toolTip()
    dialog.close()


def _slots(widget):
    from PySide6.QtCore import SIGNAL

    return widget.receivers(SIGNAL("itemChanged(QListWidgetItem*)"))


def test_a_ticked_list_has_one_slot_that_runs_every_handler(app):
    """A list's tick handlers run from its one Python slot. A handler can refill the list (the dialog's refresh does),
    deleting its items, and PySide converts the changed item for each slot it calls: with a second slot after the
    refresh, that slot was handed a deleted item, which crashed with PySide6 6.6 and on macOS (Estimate >
    Cointegrating regression: its row condition brought the formula check's slot; valgrind showed the read of the
    freed item). Here the tick runs the filter's count, the refresh and the formula check from the one slot."""
    from PySide6.QtCore import Qt

    from pymetrix.commands.builtin import FitCointegratingRegression

    session = Session()
    session.objects["data"] = Dataset.example("sample_macro")
    dialog = _dialog(FitCointegratingRegression, session)
    try:
        editor = dialog.editors["x"]
        w = editor.widget
        assert w.count() and _slots(w) == 1
        dialog.formula_timer.stop()
        w.item(1).setCheckState(Qt.CheckState.Checked)  # a tick, as a click makes it
        assert editor.get() == w.item(1).text() and dialog.formula_timer.isActive()  # the refresh, then the check
        assert "1" in editor.count.text()  # the filter's count of ticks
        dialog.formula_timer.stop()
        editor.set(w.item(0).text())  # set: the handlers once, with no item handed on
        assert editor.get() == w.item(0).text() and dialog.formula_timer.isActive() and _slots(w) == 1
    finally:
        dialog.close()
        dialog.deleteLater()


def test_a_refresh_keeps_a_lists_items_when_its_choices_are_the_same(app):
    """set_options keeps the items of a list whose choices have not changed (only the ticks follow), so a refresh
    from a tick deletes nothing; new choices still refill it."""
    import shiboken6

    from pymetrix.commands.builtin import FitCointegratingRegression

    session = Session()
    session.objects["data"] = Dataset.example("sample_macro")
    dialog = _dialog(FitCointegratingRegression, session)
    try:
        editor = dialog.editors["x"]
        w = editor.widget
        names = [w.item(i).text() for i in range(w.count())]
        before = [shiboken6.getCppPointer(w.item(i))[0] for i in range(w.count())]
        editor.set(names[0])
        assert [shiboken6.getCppPointer(w.item(i))[0] for i in range(w.count())] == before  # the same items
        assert editor.get() == names[0]
        editor.set_options(names[:2])  # new choices: refilled, the tick kept
        assert [w.item(i).text() for i in range(w.count())] == names[:2] and editor.get() == names[0]
    finally:
        dialog.close()
        dialog.deleteLater()


def test_every_dialogs_lists_have_one_slot_for_their_ticks(app):
    session = Session()
    session.objects["data"] = Dataset.example("sample_macro")
    lists = 0
    for key, info in COMMANDS.items():
        dialog = _dialog(info.cls, session)
        try:
            for name, editor in dialog.editors.items():
                if editor.spec.multiple:
                    assert _slots(editor.widget) == 1, (key, name)
                    lists += 1
        finally:
            dialog.close()
            dialog.deleteLater()
    assert lists > 20
