"""The Lab book panel in the app: the brief written and amended through its form, notes, ticked objectives linking
the steps run, and the checks kept up to date."""

import os

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pymetrix_labbook import current_brief, lab_book, register
from pymetrix_labbook.app import (
    brief_dialog,
    create_panel,
    format_pairs,
    parse_lines,
    parse_pairs,
)
from PySide6.QtCore import Qt

from pymetrix import Dataset
from pymetrix.commands import FitEquation, Session
from pymetrix.extension import AppInterface

register()


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def panel(app):
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    messages = []
    interface = AppInterface(lambda: s, lambda: None, lambda cls, initial: None, s.run, messages.append)
    widget = create_panel(interface)
    widget.session, widget.interface, widget.messages = s, interface, messages
    return widget


def test_the_forms_text_parses_and_round_trips():
    assert parse_pairs("O1: estimate it\n\nO2: test: twice") == {"O1": "estimate it", "O2": "test: twice"}
    with pytest.raises(ValueError, match="id: text"):
        parse_pairs("O1 estimate it")
    assert parse_lines(" a \n\n b") == ["a", "b"]
    assert parse_pairs(format_pairs({"H1": {"statement": "below one", "supports": "x"}, "O1": "do it"})) == {
        "H1": "below one", "O1": "do it"}


def _fill(dialog, **values):
    for key, text in values.items():
        widget, kind = dialog.widgets[key]
        widget.setText(text) if kind == "line" else widget.setPlainText(text)


def test_the_panel_prompts_for_a_brief_then_shows_it(panel):
    status = panel.findChild(QtWidgets.QLabel)
    assert "No brief yet" in status.text()
    from pymetrix_labbook import brief

    dialog = brief_dialog(panel, None)
    _fill(dialog, title="Consumption and income", question="How much of a rise in income is consumed?",
          objectives="O1: Estimate the MPC", hypotheses="H1: The MPC is below one", primary_outcomes="the coefficient on y")
    brief(panel.session, **dialog.values())
    panel.refresh()
    assert "Consumption and income" in status.text()
    ticks = panel.findChild(QtWidgets.QListWidget, "labbook-objectives")
    assert [ticks.item(i).text() for i in range(ticks.count())] == ["O1: Estimate the MPC", "H1: The MPC is below one"]
    checks = panel.findChild(QtWidgets.QListWidget, "labbook-checks")
    assert any("no analysis linked" in checks.item(i).text() for i in range(checks.count()))
    entries = panel.findChild(QtWidgets.QListWidget, "labbook-entries")
    assert "[" not in entries.item(entries.count() - 1).text()  # a brief's objectives are not tags


def test_ticked_objectives_link_the_steps_run(panel):
    from pymetrix_labbook import brief

    brief(panel.session, "C", "Q?", objectives={"O1": "first", "O2": "second"})
    panel.refresh()
    ticks = panel.findChild(QtWidgets.QListWidget, "labbook-objectives")
    ticks.item(1).setCheckState(Qt.CheckState.Checked)  # O2
    panel.interface.run(FitEquation("c ~ y", name="fit"))
    ticks.item(1).setCheckState(Qt.CheckState.Unchecked)
    panel.session.run(FitEquation("c ~ y[-1]", name="fit2"))
    steps = [e["data"].get("objectives") for e in lab_book(panel.session).entries if e["event"] == "step"]
    assert steps == [["O2"], None] and "O2" in panel.messages[0]


def test_an_amendment_keeps_what_the_form_does_not_show_and_needs_a_reason(panel):
    from pymetrix_labbook import brief

    brief(panel.session, "C", "Q?", hypotheses={"H1": {"statement": "below one", "supports": "a coefficient under 1"}})
    dialog = brief_dialog(panel, current_brief(panel.session))
    assert "reason" in dialog.widgets and dialog.widgets["hypotheses"][0].toPlainText() == "H1: below one"
    with pytest.raises(ValueError, match="reason"):
        brief(panel.session, **dialog.values())
    _fill(dialog, objectives="O1: added later", reason="the referee asked")
    brief(panel.session, **dialog.values())
    now = current_brief(panel.session)
    assert now["hypotheses"]["H1"]["supports"] == "a coefficient under 1" and now["objectives"] == {"O1": "added later"}
    assert now["reason"] == "the referee asked"


def test_the_app_docks_the_panel(app):
    from pymetrix.ui.qt import main_window

    s = Session()
    window = main_window(s)
    try:
        assert "Lab book" in window.plugin_docks
    finally:
        s.dirty = False
        window.close()


def test_the_plan_form_offers_the_history_and_locks_the_plan(panel):
    from pymetrix_labbook import lock_plan
    from pymetrix_labbook.app import plan_dialog

    panel.session.run(FitEquation("c ~ y", name="fit"))
    dialog = plan_dialog(panel, None, [r.code for r in panel.session.history])
    dialog.findChild(QtWidgets.QComboBox, "plan-history").setCurrentIndex(1)  # copy the step into the plan
    dialog.fields["data"].setText("data")
    values = dialog.values()
    assert values["specifications"] == ["fit = Equation('c ~ y').fit(data, method=OLS())"] and values["data"] == ["data"]
    lock_plan(panel.session, **values)
    panel.refresh()
    assert "Plan locked: 1 specification(s)" in panel.findChild(QtWidgets.QLabel).text()
    changed = plan_dialog(panel, current_plan_of(panel), [])
    assert "reason" in changed.fields


def current_plan_of(panel):
    from pymetrix_labbook import current_plan

    return current_plan(panel.session)


def test_the_brief_form_offers_a_private_brief_and_asks_a_reason_after_one(panel, tmp_path):
    from pymetrix_labbook import commit_brief
    from pymetrix_labbook.app import brief_dialog

    assert not brief_dialog(panel, None).private.isChecked()
    commit_brief(panel.session, tmp_path / "brief.json", "T", "Q?")
    panel.refresh()
    assert "Private brief committed" in panel.findChild(QtWidgets.QLabel).text()
    from pymetrix_labbook.brief import _has_brief

    assert "reason" in brief_dialog(panel, None, _has_brief(panel.session)).widgets


def test_the_curve_window_and_the_counts_in_the_checks(panel):
    from pymetrix_labbook.app import curve_dialog

    for formula, name in (("c ~ y", "a"), ("c ~ y + y[-1]", "b")):
        panel.session.run(FitEquation(formula, name=name))
    panel.refresh()
    checks = panel.findChild(QtWidgets.QListWidget, "labbook-checks")
    assert any("c: 2 specification(s) fitted (2 fit(s))" in checks.item(i).text() for i in range(checks.count()))
    dialog = curve_dialog(panel, panel.session, ["c"], "b")
    assert dialog.findChild(QtWidgets.QComboBox, "curve-term").currentText() == "y"
    assert dialog.figure is not None and "b highlighted" in dialog.figure.axes[0].get_title()


def test_the_panel_says_what_its_lists_are_empty_of(panel):
    panel.refresh()
    ticks = panel.findChild(QtWidgets.QListWidget, "labbook-objectives")
    empty = panel.findChild(QtWidgets.QLabel, "labbook-objectives-empty")
    assert ticks.isHidden() and not empty.isHidden() and empty.text().startswith("No objectives yet")
    checks = panel.findChild(QtWidgets.QListWidget, "labbook-checks")
    assert checks.wordWrap() and checks.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    entries = panel.findChild(QtWidgets.QListWidget, "labbook-entries")
    first = entries.item(0).text()
    assert "lab book started" in first and "not recorded" not in first  # was "0  start  · not recorded"
    buttons = [b.text() for b in panel.findChildren(QtWidgets.QPushButton)]
    assert "Specification curve…" in buttons and all(b.toolTip() for b in panel.findChildren(QtWidgets.QPushButton))


def test_a_deleted_panel_is_not_refreshed(panel):
    """The app refreshing a Lab book panel Qt has deleted (its refresh is a plain function, kept by the app): nothing
    touches its widgets."""
    import shiboken6

    refresh, interface = panel.refresh, panel.interface
    shiboken6.delete(panel)
    refresh()  # no crash, no RuntimeError
    interface._changed()


def test_a_right_click_on_an_entry_notes_it_or_copies_it(panel, monkeypatch):
    from pymetrix.commands import FitEquation
    from pymetrix_labbook import app as labbook_app
    from pymetrix_labbook import lab_book

    panel.session.run(FitEquation("c ~ y", name="fit"))
    panel.refresh()
    entries = panel.findChild(QtWidgets.QListWidget, "labbook-entries")
    item = entries.item(entries.count() - 1)
    seq = item.data(Qt.ItemDataRole.UserRole)
    menu = panel.entry_menu(item)
    texts = [a.text() for a in menu.actions()]
    assert texts == [f"Add a note on entry {seq}…", "Copy the entry", "Copy its full record (JSON)"]
    next(a for a in menu.actions() if a.text() == "Copy its full record (JSON)").trigger()
    assert f'"seq": {seq}' in QtWidgets.QApplication.clipboard().text()

    class Answer:  # the note form, filled in and OK'd
        def __init__(self, *a):
            self.title = ""

        def setWindowTitle(self, title):  # noqa: N802
            self.title = title

        def exec(self):
            return True

        def values(self):
            return "the fit uses the full sample", ""

    monkeypatch.setattr(labbook_app, "note_dialog", Answer)
    next(a for a in panel.entry_menu(item).actions() if a.text().startswith("Add a note")).trigger()
    last = lab_book(panel.session).entries[-1]
    assert last["event"] == "note" and last["data"] == {"text": "the fit uses the full sample", "step": seq}
