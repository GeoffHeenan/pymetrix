"""The Advanced settings section of command dialogs (pymetrix.ui.advanced_section): what goes in it, when it opens by
itself, what its header says, that its state is remembered per command, and its keys."""

import os
import sys
import warnings

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from pymetrix import Dataset  # noqa: E402
from pymetrix.commands import COMMANDS, FitEquation, Session, form_spec  # noqa: E402
from pymetrix.forms import access_key, needs_value  # noqa: E402


@pytest.fixture(scope="module")
def app():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import main_window

    host = main_window(Session())  # plugins' commands join the registry as the window loads them
    host.hide()
    yield app
    host.state["session"].dirty = False
    host.close()


@pytest.fixture
def session():
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    return s


@pytest.fixture
def forget():
    """Each test starts from nothing remembered, and leaves nothing behind for the other tests' dialogs."""
    from pymetrix.ui.advanced_section import _settings

    def clear():
        settings = _settings()
        settings.remove("dialogs")
        settings.sync()

    clear()
    yield
    clear()


def _key(app, widget, letter):
    QTest.keyClick(widget, getattr(Qt.Key, f"Key_{letter.upper()}"), Qt.KeyboardModifier.AltModifier)
    QTest.qWait(250)  # a button's key clicks it as a mouse would, after a moment (Qt's animateClick)
    app.processEvents()


def _mnemonics() -> bool:
    """Whether Alt+letter access keys work: everywhere but macOS, where Qt turns them off by design (Mac apps have
    none; QKeySequence.mnemonic gives no key), so that the Mac way is tested there instead: Tab and Space."""
    from PySide6.QtGui import QKeySequence

    on = not QKeySequence.mnemonic("&w").isEmpty()
    assert on == (sys.platform != "darwin")
    return on


def _space(app, button):
    """The Mac way to work the header without its access key: Tab to it (a Tab stop, see the tab-order test), Space."""
    button.setFocus(Qt.FocusReason.TabFocusReason)
    app.processEvents()
    QTest.keyClick(button, Qt.Key.Key_Space)
    app.processEvents()


def _tab_reaches(app, start, wanted, tabs=12) -> bool:
    start.setFocus(Qt.FocusReason.TabFocusReason)
    app.processEvents()
    for _ in range(tabs):
        QTest.keyClick(QtWidgets.QApplication.focusWidget(), Qt.Key.Key_Tab)
        app.processEvents()
        if QtWidgets.QApplication.focusWidget() is wanted:
            return True
    return False


def test_no_required_field_is_advanced(app):
    """A field the step cannot run without (no default) is never folded away, in any command, plugins' included."""
    hidden = [(key, s.name) for key, info in COMMANDS.items() for s in form_spec(info.cls)
              if s.advanced and (s.required or needs_value(s))]
    assert hidden == []
    from importlib.metadata import entry_points

    from pymetrix.commands.registry import COMMAND_IDS
    from pymetrix.plugins import CORE

    providers = {info.plugin for info in COMMAND_IDS.values()}
    installed = {ep.dist.name for ep in entry_points(group="pymetrix.commands") if ep.dist is not None}
    assert installed <= providers  # every installed plugin's dialogs were among those checked (none in a bare install)
    assert sum(info.plugin == CORE.name for info in COMMAND_IDS.values()) > 170  # and pymetrix's own, every one


def test_tuning_settings_are_advanced_everywhere(app):
    """Seeds, tolerances and iteration limits sit in the Advanced section in every dialog that has them."""
    tuning = ("seed", "tol", "max_iter", "sign_seed", "workers")
    shown = [(key, s.name) for key, info in COMMANDS.items() for s in form_spec(info.cls) if s.name in tuning and not s.advanced]
    assert shown == []


def test_every_dialog_opens_collapsed_with_nothing_changed(app, session, forget):
    """A fresh dialog: the section closed and its header plain; opening and closing it works in every dialog."""
    from pymetrix.ui.qt import command_dialog

    with_section = 0
    for key, info in list(COMMANDS.items()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dialog = command_dialog(info.cls, session)
        try:
            if any(s.advanced for s in form_spec(info.cls)):
                assert dialog.advanced_section is not None, key
            if dialog.advanced_section is None:
                continue
            with_section += 1
            assert not dialog.advanced_toggle.isChecked(), key
            assert dialog.advanced_changed() == (0, 0), (key, dialog.advanced_toggle.text())
            assert "changed" not in dialog.advanced_toggle.text(), key
            dialog.show_advanced(True)
            assert dialog.advanced_toggle.isChecked()
            dialog.show_advanced(False)
        finally:
            dialog.close()
            dialog.deleteLater()
    assert with_section > 100


def test_a_changed_advanced_setting_opens_the_section_and_the_header_counts_it(app, session, forget):
    from pymetrix.ui.qt import command_dialog

    reopened = command_dialog(FitEquation, session, initial=FitEquation("c ~ y", missing="raise", cov="hac", hac_lags=3))
    assert reopened.advanced_toggle.isChecked()
    assert reopened.advanced_changed() == (2, 0)
    assert reopened.advanced_toggle.text().replace("&", "") == "Advanced settings (2 changed)"
    reopened.editors["hac_lags"].set(0)  # back to its default: one left
    assert reopened.advanced_toggle.text().replace("&", "") == "Advanced settings (1 changed)"
    reopened.editors["cov"].set("classic")  # HAC lags no longer applies, and is no longer counted
    reopened.editors["missing"].set("drop")
    assert reopened.advanced_toggle.text().replace("&", "") == "Advanced settings"
    plain = command_dialog(FitEquation, session, initial=FitEquation("c ~ y", cov="hac"))
    assert not plain.advanced_toggle.isChecked() and plain.advanced_changed() == (0, 0)
    plain.editors["missing"].set("raise")  # changed while closed: the header says so
    assert not plain.advanced_toggle.isChecked()
    assert plain.advanced_toggle.text().replace("&", "") == "Advanced settings (1 changed)"
    for d in (reopened, plain):
        d.close()


def test_a_sub_forms_advanced_setting_counts_too(app, session, forget):
    """The advanced settings of the method chosen (Method settings) are in the section, and counted there."""
    from pymetrix.ui.qt import command_dialog

    dialog = command_dialog(FitEquation, session)
    part = dialog.editors["method_settings"]
    advanced = [n for n, e in part.sub.items() if e.spec.advanced]
    if not advanced:  # pragma: no cover - OLS's settings changed: pick a method that has advanced ones
        pytest.skip("OLS has no advanced settings")
    before = dialog.advanced_changed()[0]
    editor = part.sub[advanced[0]]
    spec = editor.spec
    if spec.kind == "checkbox":
        editor.set(not spec.default)
    elif spec.kind == "number" and not spec.scientific:
        editor.widget.setValue(editor.widget.value() + editor.widget.singleStep())
    else:
        editor.widget.setText("7") if hasattr(editor.widget, "setText") else pytest.skip(spec.kind)
    dialog.editors["method"].widget.currentIndexChanged.emit(dialog.editors["method"].widget.currentIndex())
    assert dialog.advanced_changed()[0] == before + 1
    dialog.close()


def test_an_invalid_advanced_setting_opens_the_section(app, session, forget):
    """A value that cannot be used as it stands, in a closed section: at open (a reopened step) the section opens, and
    OK opens it rather than leave the reason out of sight."""
    from pymetrix.commands import OptimalControlCommand
    from pymetrix.ui.qt import command_dialog

    bad = command_dialog(FitEquation, session, initial=FitEquation("c ~ y", where="y >"))
    bad.check_formulas()
    assert bad.formula_checks["where"].error
    assert bad.advanced_toggle.isChecked() and bad.advanced_changed()[1] == 1
    bad.close()

    control = command_dialog(OptimalControlCommand, session)
    assert not control.advanced_toggle.isChecked()
    control.editors["tol"].widget.setText("1e-")  # typed half way: not a number yet
    assert control.advanced_changed() == (1, 1)
    control.show_advanced(False)
    control.findChild(QtWidgets.QDialogButtonBox).accepted.emit()
    assert control.advanced_toggle.isChecked()
    control.close()


def test_the_section_state_is_remembered_per_command(app, session, forget):
    from pymetrix.commands import FitVAR
    from pymetrix.ui.advanced_section import SETTINGS_KEY, _settings
    from pymetrix.ui.qt import command_dialog

    first = command_dialog(FitEquation, session)
    assert not first.advanced_toggle.isChecked()  # collapsed by default
    first.advanced_toggle.click()  # opened by hand: kept
    assert first.advanced_toggle.isChecked()
    assert str(_settings().value(SETTINGS_KEY.format(command="FitEquation"))).lower() == "true"
    first.close()
    again = command_dialog(FitEquation, session)
    other = command_dialog(FitVAR, session)
    assert again.advanced_toggle.isChecked() and not other.advanced_toggle.isChecked()  # per command
    again.advanced_toggle.click()  # closed by hand: kept closed
    third = command_dialog(FitEquation, session)
    assert not third.advanced_toggle.isChecked()
    third.show_advanced(True)  # opened by the program (a reopened step, a field's key): not remembered
    fourth = command_dialog(FitEquation, session)
    assert not fourth.advanced_toggle.isChecked()
    for d in (again, other, third, fourth):
        d.close()


def test_settings_that_cannot_be_read_or_written_leave_it_collapsed(app, session, forget, monkeypatch):
    from pymetrix.ui import advanced_section
    from pymetrix.ui.qt import command_dialog

    def broken():
        raise OSError("no settings here")

    monkeypatch.setattr(advanced_section, "_settings", broken)
    dialog = command_dialog(FitEquation, session)
    assert not dialog.advanced_toggle.isChecked()
    dialog.advanced_toggle.click()  # writing fails quietly
    assert dialog.advanced_toggle.isChecked()
    dialog.close()


def test_the_header_has_an_access_key_and_a_hidden_fields_key_opens_the_section(app, session, forget):
    from pymetrix.ui.qt import command_dialog

    dialog = command_dialog(FitEquation, session)
    dialog.show()
    dialog.activateWindow()
    app.processEvents()
    header = access_key(dialog.advanced_toggle.text())
    labels = [access_key(label.text()) for label, _ in dialog.access_keys]
    assert header and header not in labels
    if not _mnemonics():  # a Mac: no access keys (Qt's choice there); Space on the header opens and closes it
        assert dialog.advanced_toggle.shortcut().isEmpty()
        _space(app, dialog.advanced_toggle)
        assert dialog.advanced_toggle.isChecked() and dialog.row_label("weights").isVisibleTo(dialog)
        assert _tab_reaches(app, dialog.advanced_toggle, dialog.editors["weights"].widget)  # and Tab goes on into it
        _space(app, dialog.advanced_toggle)
        assert not dialog.advanced_toggle.isChecked() and not dialog.row_label("weights").isVisibleTo(dialog)
        dialog.close()
        return
    _key(app, dialog, header)  # the header's key opens and closes the section
    assert dialog.advanced_toggle.isChecked()
    _key(app, dialog, header)
    assert not dialog.advanced_toggle.isChecked()

    label = dialog.row_label("weights")  # in the closed section: its label hidden, its key kept
    key = access_key(label.text())
    assert key and not label.isVisibleTo(dialog)
    _key(app, dialog, key)
    assert dialog.advanced_toggle.isChecked() and label.isVisibleTo(dialog)
    assert QtWidgets.QApplication.focusWidget() is dialog.editors["weights"].widget
    assert access_key(dialog.row_label("weights").text()) == key  # the same key, open or closed
    _key(app, dialog, key)  # open now: the label's own key, not the section's shortcut (no ambiguity)
    assert QtWidgets.QApplication.focusWidget() is dialog.editors["weights"].widget
    dialog.close()


def test_tab_order_runs_through_the_header_into_the_open_section(app, session, forget):
    from pymetrix.ui.qt import command_dialog

    dialog = command_dialog(FitEquation, session)
    dialog.show()
    dialog.activateWindow()
    app.processEvents()
    dialog.editors["name"].widget.setFocus()
    seen = []
    for _ in range(12):
        QTest.keyClick(QtWidgets.QApplication.focusWidget(), Qt.Key.Key_Tab)
        app.processEvents()
        seen.append(QtWidgets.QApplication.focusWidget())
    assert dialog.advanced_toggle in seen  # the header is a Tab stop
    assert not any(w is dialog.editors["weights"].widget for w in seen)  # closed: its fields are skipped
    dialog.show_advanced(True)
    app.processEvents()
    dialog.advanced_toggle.setFocus()
    seen = []
    for _ in range(12):
        QTest.keyClick(QtWidgets.QApplication.focusWidget(), Qt.Key.Key_Tab)
        app.processEvents()
        seen.append(QtWidgets.QApplication.focusWidget())
    assert any(w is dialog.editors["weights"].widget for w in seen)  # open: Tab goes on into its fields
    dialog.close()


def test_help_still_lists_every_field(app):
    """The dialog's help page names the advanced settings too, marked as such."""
    from pymetrix.help import help_page

    page = help_page(FitEquation).to_markdown()
    for spec in form_spec(FitEquation):
        if spec.advanced:
            assert f"{spec.label} (advanced)" in page or f"{spec.name} (advanced)" in page, spec.name


@pytest.fixture
def code_folded():
    """Show the code folded, as dialogs first open; the remembered choice as it was afterwards."""
    from PySide6.QtCore import QSettings

    from pymetrix.ui.qt import SHOW_CODE_KEY

    settings = QSettings()
    kept = settings.value(SHOW_CODE_KEY) if settings.contains(SHOW_CODE_KEY) else None
    settings.remove(SHOW_CODE_KEY)
    yield
    if kept is None:
        settings.remove(SHOW_CODE_KEY)
    else:
        settings.setValue(SHOW_CODE_KEY, kept)


def test_the_section_and_show_the_code_live_together(app, session, forget, code_folded):
    """The Advanced section is the end of the form, Show the code is under the form (above the buttons): neither
    overlaps the other, open or folded; each opens and folds on its own; the code follows an advanced setting; and the
    dialog's access keys stay unique (Show the code takes none, so no field's or the header's can clash with it)."""
    from pymetrix.ui.qt import command_dialog

    dialog = command_dialog(FitEquation, session)
    dialog.show()
    app.processEvents()
    section, code = dialog.advanced_toggle, dialog.code_toggle
    assert dialog.form_scroll.widget().isAncestorOf(section) and not dialog.form_scroll.isAncestorOf(code)
    buttons = dialog.findChild(QtWidgets.QDialogButtonBox)

    def top(widget):
        return widget.mapTo(dialog, widget.rect().topLeft()).y()

    for section_open, code_open in ((False, False), (True, False), (False, True), (True, True)):
        dialog.show_advanced(section_open)
        code.setChecked(code_open)
        app.processEvents()
        assert section.isChecked() == section_open and code.isChecked() == code_open  # one does not move the other
        assert dialog.code_box.isVisibleTo(dialog) == code_open
        assert dialog.row_label("weights").isVisibleTo(dialog) == section_open
        assert top(dialog.form_scroll) + dialog.form_scroll.height() <= top(code) < top(buttons)
        if code_open:
            assert top(code) + code.height() <= top(dialog.code_box) < top(buttons)
        keyed = [access_key(label.text()) for label, _ in dialog.access_keys] + [access_key(section.text())]
        keyed += [access_key(b.text()) for b in buttons.buttons()]
        keyed = [k for k in keyed if k]
        assert len(keyed) == len(set(keyed)), keyed
        assert access_key(code.text()) == ""

    dialog.editors["formula"].set("c ~ y")
    dialog.editors["missing"].set("raise")  # an advanced setting: the code shown says so at once
    app.processEvents()
    assert "missing='raise'" in dialog.code_text().replace('"', "'") and dialog.code_box.toPlainText() == dialog.code_text()
    if _mnemonics():
        _key(app, dialog, access_key(section.text()))  # the header's key folds the section and leaves the code open
    else:
        _space(app, section)  # a Mac: no access keys; Space on the header does the same
    assert not section.isChecked() and code.isChecked()
    code.setChecked(False)
    dialog.close()


def test_toggling_the_section_in_an_open_dialog_refits_it_without_an_error(app, session, forget, monkeypatch):
    """In a visible dialog, opening or closing the section refits the window a moment later, and a hidden field's key
    puts the cursor there again once the rows are laid out: both through ``ui.timers.single_shot``. With PySide6 6.6.0
    the three-argument ``QTimer.singleShot(0, dialog, callable)`` used before raised TypeError in the slot, which Qt
    only prints, so the dialog was never refitted; slot errors are collected here from ``sys.excepthook``."""
    from pymetrix.ui.qt import command_dialog

    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda kind, value, tb: errors.append(f"{kind.__name__}: {value}"))
    dialog = command_dialog(FitEquation, session)
    real_adjust = dialog.adjustSize
    fits = []

    def adjust():
        fits.append(dialog.advanced_toggle.isChecked())
        real_adjust()

    dialog.adjustSize = adjust  # looked up when the section toggles
    dialog.show()
    app.processEvents()
    for on in (True, False):
        dialog.show_advanced(on)
        QTest.qWait(50)
        app.processEvents()
        assert not errors
        assert fits and fits[-1] is on  # refitted, after this toggle
    fits.clear()
    weights = dialog.editors["weights"].widget
    dialog.advanced_section._reveal(weights)  # what a hidden field's access key does
    QTest.qWait(50)
    app.processEvents()
    assert not errors
    assert dialog.advanced_toggle.isChecked() and weights.isVisibleTo(dialog) and fits
    dialog.adjustSize = real_adjust
    dialog.close()


def _native_single_shot() -> bool:
    """Whether this PySide6 takes QTimer.singleShot(msec, context, callable) (6.6.0 raises TypeError)."""
    from PySide6.QtCore import QObject, QTimer

    probe = QObject()
    try:
        QTimer.singleShot(0, probe, lambda: None)
    except TypeError:
        return False
    finally:
        probe.deleteLater()
    return True


@pytest.mark.parametrize("native", [True, False], ids=["context-overload", "own-timer"])
def test_opening_the_section_by_hand_fits_the_dialog_with_either_form(app, session, forget, monkeypatch, native):
    """Clicking the header in a shown dialog fits the dialog to its form once the layout has settled (taller open,
    back closed), and a hidden field's key puts the cursor in it again then, with either form ``ui.timers.single_shot``
    may use: Qt's context-object overload (where PySide6 has it) and its own timer (forced here on every version).
    Calls run once each, and not once their context is deleted. (Cases from the PySide6 6.6 fix made in parallel,
    PR #47, kept with the one helper.)"""
    import shiboken6

    from pymetrix.ui import timers
    from pymetrix.ui.qt import command_dialog

    if native and not _native_single_shot():
        pytest.skip("this PySide6 has no context-object overload of QTimer.singleShot")
    monkeypatch.setitem(timers._STATE, "native", None if native else False)
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: errors.append(exc))  # an exception in a slot comes here
    dialog = command_dialog(FitEquation, session)
    dialog.show()
    app.processEvents()
    section = dialog.advanced_section
    closed = dialog.height()
    for on in (True, False, True):
        dialog.advanced_toggle.click()  # by hand, as a user does
        assert dialog.advanced_toggle.isChecked() == on
        for _ in range(3):
            app.processEvents()
        assert errors == []
        assert (dialog.height() > closed) == on  # the room for the section, and back
    assert timers._STATE["native"] is native  # the form asked for was the one used
    section._flip(False)  # called directly, an error would reach the test itself
    section._reveal(dialog.editors["weights"].widget)
    for _ in range(3):
        app.processEvents()
    assert errors == []

    ran = []
    timers.single_shot(0, dialog, lambda: ran.append(True))
    timers.single_shot(0, dialog, lambda: ran.append(True))
    app.processEvents()
    assert ran == [True, True]
    gone = QtWidgets.QWidget()
    timers.single_shot(0, gone, lambda: ran.append(gone))
    shiboken6.delete(gone)  # deleted before the event loop comes back: the call is dropped
    app.processEvents()
    assert ran == [True, True] and errors == []
    dialog.close()
