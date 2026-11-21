"""Small things a first-time reviewer meets: Find anything's order, Immediate under a dark palette, an unknown
function's message, short field hints, End marked without losing its arrows, the series browser's width, the demo
picker's width."""

import os

import pytest

from pymetrix import Dataset
from pymetrix.commands import COMMANDS, Session


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_find_anything_puts_commands_before_help_and_skips_a_dialogs_own_page():
    from pymetrix.ui.palette import PaletteEntry, rank

    entries = [PaletteEntry("Estimate → VARs and local projections → VAR…", "Menu", lambda: None),
               PaletteEntry("Estimate → VARs and local projections → Variance decomposition…", "Menu", lambda: None),
               PaletteEntry("VAR", "Help", lambda: None, "Commands"),
               PaletteEntry("VAR", "Help", lambda: None, "Models"),
               PaletteEntry("VARMA", "Help", lambda: None, "Commands")]
    found = [(e.title, e.kind, e.detail) for e in rank(entries, "var")]
    assert ("VAR", "Help", "Commands") not in found  # the menu item stands for its dialog's page
    assert found[0][:2] == ("Estimate → VARs and local projections → VAR…", "Menu")
    assert found.index(("VARMA", "Help", "Commands")) > found.index(("Estimate → VARs and local projections → Variance "
                                                                     "decomposition…", "Menu", ""))  # a menu item first


def test_an_unknown_function_names_the_likely_one_without_listing_all():
    from pymetrix.expr import ExpressionError, parse_expression
    from pymetrix.ui.messages import explain

    with pytest.raises(ExpressionError) as close:
        parse_expression("logg(y)")
    assert "did you mean log()?" in str(close.value) and "known:" not in str(close.value)
    with pytest.raises(ExpressionError) as far:
        parse_expression("foo(y)")
    box = explain(far.value)
    assert "known:" not in box.text and "abs, betacdf" in box.details and "Help → Help contents" in box.hint


@pytest.mark.parametrize("help_text, hint", [
    ("a condition on the rows, e.g. recession == 0 or year >= 1990; empty = all rows", "e.g. recession == 0"),
    ("the parameters' bounds, e.g. rho=(0, 1); beta=(0, 0.999)", "e.g. rho=(0, 1)"),
    ("the lower limit (empty: none)", "the lower limit"),
    ("comma-separated names of the equations the shocks enter and their order in the system", "comma-separated names of the equations…"),
])
def test_a_text_fields_hint_is_short(help_text, hint):
    from pymetrix.ui.qt import short_placeholder

    assert short_placeholder(help_text) == hint and len(hint) <= 45


def test_start_after_end_marks_end_without_a_style_sheet(app):
    from pymetrix.ui.qt import command_dialog

    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    cls = next(i.cls for i in COMMANDS.values() if i.cls.__name__ == "FitEquation")
    dialog = command_dialog(cls, s)
    try:
        dialog.editors["start"].set("2010Q1")
        dialog.editors["end"].set("2005Q1")
        app.processEvents()
        end = dialog.editors["end"].widget
        assert end.styleSheet() == "" and end.property("invalid") and end.accessibleDescription()  # native arrows kept
        assert not dialog.ok_gate.ok.isEnabled()
        dialog.editors["end"].set("2015Q1")
        app.processEvents()
        assert not end.property("invalid") and end.accessibleDescription() == ""
    finally:
        dialog.close()


def test_the_immediate_window_follows_a_dark_palette(app):
    from pymetrix.ui import console_qt

    if not console_qt.available():
        pytest.skip("qtconsole is not installed")
    from PySide6.QtGui import QColor, QPalette

    old = app.palette()
    dark = QPalette(old)
    dark.setColor(QPalette.ColorRole.Window, QColor("#2b2b2b"))
    app.setPalette(dark)
    try:
        assert console_qt.palette_is_dark()
        immediate = console_qt.Immediate(Session(), lambda record: None)
        try:
            assert immediate.widget.syntax_style == "monokai"  # qtconsole's dark ("linux") colours
        finally:
            immediate.shutdown()
    finally:
        app.setPalette(old)
    assert not console_qt.palette_is_dark()


def test_the_demo_picker_lists_short_lines(app, monkeypatch):
    """File → Open demo project…: a list of the demos (not a dropdown of cut lines), each with its description, and
    the dialog no wider than a laptop's screen leaves room for."""
    from PySide6 import QtWidgets

    from pymetrix.demos import demos
    from pymetrix.ui.qt import main_window

    seen = {}

    def chooser_exec(dialog):
        seen["names"], seen["width"] = list(dialog.names), dialog.sizeHint().width()
        seen["texts"] = [e.text() for e in dialog.entries.entries]
        dialog.open_row(1)
        return 1

    monkeypatch.setattr(QtWidgets.QDialog, "exec", chooser_exec)
    window = main_window(Session())
    try:
        window.open_demo()
        assert seen["names"] == list(demos()) and seen["names"][1] == "consumption" and seen["width"] <= 700
        assert all(text.split("\n")[1] == demos()[name] for name, text in zip(seen["names"], seen["texts"]))
        assert "dynamic" in window.state["session"].objects
    finally:
        window.state["session"].dirty = False
        window.close()


def test_the_series_browser_gives_fred_the_width(app):
    from pymetrix.ui.series_browser_qt import series_browser

    browser = series_browser(Session())
    keys = [browser.provider.itemData(i) for i in range(browser.provider.count())]
    if "fred" not in keys:
        pytest.skip("no FRED connector")
    browser.provider.setCurrentIndex(keys.index("fred"))
    assert not browser.side.isVisibleTo(browser.dialog) and browser.note.text()[:1].isupper()
    if "sdmx" in keys:
        browser.provider.setCurrentIndex(keys.index("sdmx"))
        assert browser.side.isVisibleTo(browser.dialog)
    browser.dialog.close()


def test_window_titles_follow_one_pattern():
    from pymetrix import Equation
    from pymetrix.ui.qt import kind_of, window_title
    from pymetrix.var import VAR
    from pymetrix.views import dataset_overview

    data = Dataset.example("sample_macro")
    assert window_title("var", VAR(["c", "y"], lags=1).fit(data)) == "var — Fitted VAR"  # as "bvar — Fitted BVAR"
    assert kind_of(Equation("c ~ y").fit(data)) == "Fitted equation"
    assert dataset_overview(data, "data").title.startswith("Dataset ")  # its kind, then its title
