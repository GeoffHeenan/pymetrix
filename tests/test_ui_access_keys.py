"""Access keys on dialog field labels (pymetrix.ui.form_keys): in every dialog each field label's key is unique, none
is a button's (OK, Cancel, Help), each label's buddy is its field's input, and Alt+key goes to that field."""

import os
import sys
import warnings

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pymetrix import Dataset  # noqa: E402
from pymetrix.commands import COMMANDS, Session  # noqa: E402
from pymetrix.forms import access_key, access_keys  # noqa: E402


@pytest.fixture(scope="module")
def app():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import main_window

    host = main_window(Session())  # plugins' commands join the registry as the window loads them
    host.hide()
    yield app
    host.state["session"].dirty = False
    host.close()


def test_keys_from_the_labels_words():
    assert access_keys(["Standard errors", "Sample", "Start", "Series"]) == ["&Standard errors", "S&ample", "S&tart", "S&eries"]
    assert access_keys(["Data", "Output name"], taken={"o", "c", "h"}) == ["&Data", "Output &name"]  # word starts first
    assert access_keys(["R&D share"]) == ["&R&&D share"]  # an ampersand of its own is shown, not a key
    assert access_keys(["a", "a"]) == ["&a", "a"]  # no letter left: none
    # too few letters: the labels first in the list keep theirs (a field in view before a hidden one), moving keys
    # along where they can; "b" had taken the only letter "a" could have had before "a" was given one
    assert access_keys(["ab", "a", "b"]) == ["a&b", "&a", "b"]
    assert [access_key(t) for t in ("&Data", "R&&D", "S&&P &x", "plain")] == ["d", "", "x", ""]


def _inputs(editor):
    """What may take the keys for a field: its widget, a file field's path, a radio field's buttons."""
    if editor.spec.kind in ("file", "savefile"):
        return [editor.path]
    if editor.spec.radio:
        return list(editor.group.buttons())
    return [editor.widget]


def _all_editors(editors):
    for name, editor in editors.items():
        yield name, editor
        yield from _all_editors(getattr(editor, "sub", {}) if editor.spec.kind == "options" else {})


def test_every_dialog_has_unique_keys_on_its_field_labels_with_their_inputs_as_buddies(app):
    from pymetrix.ui.qt import command_dialog

    session = Session()
    session.objects["data"] = Dataset.example("sample_macro")
    keyed = dialogs = 0
    for info in list(COMMANDS.values()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dialog = command_dialog(info.cls, session)
        try:
            name = info.cls.__name__
            buttons = dialog.findChild(QtWidgets.QDialogButtonBox)
            button_keys = {access_key(b.text()) for b in buttons.buttons()} | {"o", "c", "h"}
            keys = [access_key(label.text()) for label, _ in dialog.access_keys]
            keys = [k for k in keys if k]
            assert len(keys) == len(set(keys)), (name, keys)
            assert not set(keys) & button_keys, (name, keys)
            inputs = {id(w): n for n, e in _all_editors(dialog.editors) for w in _inputs(e)}
            for label, target in dialog.access_keys:
                assert label.buddy() is target and id(target) in inputs, (name, label.text())
            shown = [label for label, _ in dialog.access_keys if label.isVisibleTo(dialog)]
            assert all(access_key(label.text()) for label in shown[:20]), name  # the fields in view first
            keyed += len(keys)
            dialogs += 1
        finally:
            dialog.close()
            dialog.deleteLater()
    assert dialogs > 150 and keyed > 600


def test_the_fields_in_view_have_keys_whichever_settings_are_shown(app):
    """A method's or solver's settings (a sub-form) change the fields in view: with each choice, the fields in view
    have keys before the hidden ones (Estimate > Equation with Robust regression shows Weighting (norm), which had
    lost its letter to a hidden HAC setting)."""
    from pymetrix.ui.qt import command_dialog

    session = Session()
    session.objects["data"] = Dataset.example("sample_macro")
    checked = 0
    for info in list(COMMANDS.values()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dialog = command_dialog(info.cls, session)
        try:
            drivers = {e.spec.depends_on for e in dialog.editors.values() if e.spec.kind == "options" and e.spec.depends_on}
            for driver in sorted(d for d in drivers if d in dialog.editors):
                combo = dialog.editors[driver].widget
                if not isinstance(combo, QtWidgets.QComboBox):
                    continue
                for i in range(combo.count()):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        combo.setCurrentIndex(i)
                    QtWidgets.QApplication.processEvents()
                    shown = [label for label, _ in dialog.access_keys if label.isVisibleTo(dialog)]
                    keys = [access_key(label.text()) for label, _ in dialog.access_keys]
                    keys = [k for k in keys if k]
                    assert len(keys) == len(set(keys)), (info.cls.__name__, combo.itemText(i), keys)
                    missing = [label.text() for label in shown[:20] if not access_key(label.text())]
                    assert not missing, (info.cls.__name__, combo.itemText(i), missing)
                    checked += 1
        finally:
            dialog.close()
            dialog.deleteLater()
    assert checked > 50


def test_alt_and_the_key_goes_to_the_field(app):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from pymetrix.commands.builtin import FitEquation
    from pymetrix.ui.qt import command_dialog

    session = Session()
    session.objects["data"] = Dataset.example("sample_macro")
    dialog = command_dialog(FitEquation, session)
    dialog.show()
    dialog.activateWindow()
    QtWidgets.QApplication.processEvents()
    label = dialog.row_label("method")
    key = access_key(label.text())
    assert label.text().replace("&", "") == "Method" and label.buddy() is dialog.editors["method"].widget
    from PySide6.QtGui import QKeySequence

    mnemonics = not QKeySequence.mnemonic(label.text()).isEmpty()
    assert mnemonics == (sys.platform != "darwin")  # Qt turns access keys off on macOS, by design: Mac apps have none
    if mnemonics:
        QTest.keyClick(dialog, getattr(Qt.Key, f"Key_{key.upper()}"), Qt.KeyboardModifier.AltModifier)
        QtWidgets.QApplication.processEvents()
        assert QtWidgets.QApplication.focusWidget() is dialog.editors["method"].widget
    else:  # the Mac way: Tab from the first field reaches it
        dialog.editors["name"].widget.setFocus(Qt.FocusReason.TabFocusReason)
        for _ in range(12):
            QTest.keyClick(QtWidgets.QApplication.focusWidget(), Qt.Key.Key_Tab)
            QtWidgets.QApplication.processEvents()
            if QtWidgets.QApplication.focusWidget() is dialog.editors["method"].widget:
                break
        assert QtWidgets.QApplication.focusWidget() is dialog.editors["method"].widget
    dialog.close()
