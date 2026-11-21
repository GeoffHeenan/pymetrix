"""No command dialog scrolls sideways at 1366 by 768 (a common laptop screen), in any look (this machine's, a wider
font, Windows' style): a form that opens shorter than its content scrolls up and down, and its least width leaves
room for that scroll bar, so the bar no longer pushes a sideways one into view (System of equations and Equation for
each selected series did, by one scroll bar's width). With the Advanced settings section closed, as dialogs open, and
open."""

import os
import warnings

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect  # noqa: E402

import _fit_check as fc  # noqa: E402
from pymetrix import Dataset  # noqa: E402
from pymetrix.commands import COMMANDS, Session  # noqa: E402


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
def laptop(monkeypatch):
    from pymetrix.ui import fit

    monkeypatch.setattr(fit, "_available", lambda widget: QRect(0, 0, 1366, 768))


@pytest.mark.parametrize("look", fc.LOOKS)
def test_no_dialog_scrolls_sideways_at_1366_by_768(app, laptop, look):
    from pymetrix.ui.qt import command_dialog

    session = Session()
    session.objects["data"] = Dataset.example("sample_macro")
    sideways, scrolled = [], []
    with fc.look(look):
        for info in list(COMMANDS.values()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    dialog = command_dialog(info.cls, session)
                except Exception:  # noqa: BLE001 - a dialog that needs more than the sample data; its own tests cover it
                    continue
            try:
                dialog.show()
                for section_open in (False, True):  # as it opens, and with its Advanced settings section open
                    if section_open:
                        if dialog.advanced_section is None:
                            break
                        dialog.show_advanced(True)
                    for _ in range(4):
                        app.processEvents()
                    scroll = dialog.form_scroll
                    name = info.cls.__name__
                    if scroll.horizontalScrollBar().isVisible():
                        sideways.append((name, section_open, dialog.size().toTuple(), scroll.widget().minimumSizeHint().width(),
                                         scroll.viewport().width()))
                    if scroll.verticalScrollBar().isVisible():
                        if not section_open:
                            scrolled.append(name)
                        assert scroll.viewport().width() >= scroll.widget().minimumSizeHint().width(), (name, section_open)
            finally:
                dialog.close()
                dialog.deleteLater()
    assert sideways == []
    if look == "wide":  # the case this guards: forms that open shorter than they would be (the two that scrolled sideways)
        assert {"FitSystem", "FitEach"} <= set(scrolled)


def test_a_field_goes_under_its_label_where_the_two_do_not_fit_side_by_side(app):
    """A form's least width is its widest label or field, not the two together: a field too wide to sit beside the
    labels at the form's width goes under its own (Reconcile's long label and its method's long radio choice, side by
    side, scrolled sideways at 1366 pixels where every character is a 16-pixel box: the look test above, on Windows)."""
    from PySide6.QtWidgets import QFormLayout

    from pymetrix.commands import Reconcile
    from pymetrix.ui.qt import command_dialog

    session = Session()
    session.objects["data"] = Dataset.example("sample_macro")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dialog = command_dialog(Reconcile, session)
    try:
        form = next(f for f in dialog.form_scroll.widget().findChildren(QFormLayout) if f.indexOf(dialog.editors["method"].widget) >= 0)
        assert form.rowWrapPolicy() == QFormLayout.RowWrapPolicy.WrapLongRows
        roles = QFormLayout.ItemRole
        labels = [form.itemAt(r, roles.LabelRole) for r in range(form.rowCount())]
        fields = [form.itemAt(r, roles.FieldRole) for r in range(form.rowCount())]
        widest_label = max(i.minimumSize().width() for i in labels if i is not None)
        widest_field = max(i.minimumSize().width() for i in fields if i is not None)
        assert form.minimumSize().width() < widest_label + form.horizontalSpacing() + widest_field
    finally:
        dialog.close()
        dialog.deleteLater()
