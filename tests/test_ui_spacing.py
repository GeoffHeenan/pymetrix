"""The app's spacing comes from a few tokens (pymetrix.ui.metrics): no margin or spacing in pymetrix.ui is a bare number
except the deliberate exceptions listed here; every dialog has the same margins; a section heading, a section's fold
toggle and a group box have the same room above them; and dialogs' buttons sit at the right, as a QDialogButtonBox
puts them."""

import ast
import os
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "src" / "pymetrix" / "ui"
GUIDE = ROOT / "packages" / "pymetrix-guide" / "src" / "pymetrix_guide" / "app.py"  # the Guide panel, a bundled plugin
CALLS = ("setContentsMargins", "setSpacing", "setHorizontalSpacing", "setVerticalSpacing", "addSpacing", "setMargin")
# file, the call's source: numbers kept on purpose, each with its reason where it is set
DELIBERATE = {
    ("qt.py", "bar.setSpacing(3)"),  # a result's seven figure buttons side by side at the window's opening width
    ("qt.py", "objects_empty.setMargin(6)"),  # the Objects panel's empty-state line (that panel's own decoration)
    ("fit.py", "self.setContentsMargins(6, 3, 6, 3)"),  # a list entry's padding, as a list view's item has
}


def _literal_calls():
    """(file, source) of each call in CALLS with a whole-number argument other than 0, in pymetrix.ui and the Guide."""
    found = []
    for path in [*sorted(UI.glob("*.py")), *([GUIDE] if GUIDE.is_file() else [])]:
        text = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in CALLS:
                if any(isinstance(a, ast.Constant) and isinstance(a.value, (int, float)) and a.value != 0 for a in node.args):
                    found.append((path.name, ast.get_source_segment(text, node)))
    return found


def test_no_margin_or_spacing_is_a_bare_number_but_the_deliberate_ones():
    found = set(_literal_calls())
    assert found - DELIBERATE == set()
    assert DELIBERATE - found == set(), "an exception no longer there: take it off the list"


def test_the_deliberate_numbers_say_why_where_they_are_set():
    for name, call in DELIBERATE:
        line = next(line for line in (UI / name).read_text(encoding="utf-8").splitlines() if call in line)
        assert "#" in line or "objects_empty" in call, (name, call)  # the Objects panel's line: that panel's own


def test_the_tokens_sit_on_the_grid_and_scale_with_the_text():
    pytest.importorskip("PySide6.QtWidgets")
    from pymetrix.ui import metrics, text_size

    for name, size in metrics.TOKENS.items():
        assert size % metrics.GRID == 0 or name == "hairline", name  # hairline: half a step, on purpose
    assert metrics.TOKENS["section"] > metrics.TOKENS["normal"] > metrics.TOKENS["tight"] > metrics.TOKENS["hairline"]
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    try:
        assert metrics.px("section") == 12
        text_size.apply("Largest")
        assert metrics.px("section") == round(12 * 1.3) and metrics.px("none") == 0
    finally:
        text_size.apply("Normal")


QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def session():
    from pymetrix import Dataset
    from pymetrix.commands import Session

    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    return s


def _dialog(session, name):
    from pymetrix.commands import COMMANDS
    from pymetrix.ui.qt import command_dialog

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return command_dialog(next(i.cls for i in COMMANDS.values() if i.cls.__name__ == name), session)


def _margins(layout):
    m = layout.contentsMargins()
    return m.left(), m.top(), m.right(), m.bottom()


def test_every_dialog_has_the_dialog_margins(app, session):
    from pymetrix.ui.metrics import px
    from pymetrix.ui.plugins_qt import plugin_manager
    from pymetrix.ui.qt import main_window, shortcuts_dialog

    want = (px("dialog_margin"),) * 4
    dialog = _dialog(session, "FitSARIMA")
    assert _margins(dialog.layout()) == want
    window = main_window(session)
    try:
        for other in (plugin_manager(window), shortcuts_dialog(window)):
            assert _margins(other.layout()) == want, type(other).__name__
            other.deleteLater()
    finally:
        session.dirty = False
        window.close()
    dialog.deleteLater()


def test_headings_toggles_and_group_boxes_have_the_same_room_above(app, session):
    """A form's section heading ("Sample"), the Advanced section's toggle and Show the code's: a section's room above
    each; the first thing on a page has none."""
    from PySide6.QtWidgets import QFormLayout, QLabel

    from pymetrix.ui.metrics import px

    room = px("section")
    dialog = _dialog(session, "FitSARIMA")
    dialog.show()
    app.processEvents()
    heading = next(w for w in dialog.findChildren(QLabel) if w.text() == "<b>Sample</b>")
    assert heading.contentsMargins().top() == room
    for toggle in (dialog.advanced_toggle, dialog.code_toggle):
        holder = toggle.parentWidget()
        assert holder.layout().contentsMargins().top() == room
        assert toggle.geometry().top() == room  # the room is there, above the button
    form = next(f for f in dialog.form_scroll.widget().findChildren(QFormLayout) if f.indexOf(dialog.editors["series"].outer) >= 0)
    first = form.itemAt(0, QFormLayout.ItemRole.LabelRole).widget()
    assert first.contentsMargins().top() == 0  # the form's first row: no room above it
    dialog.close()
    dialog.deleteLater()


def test_the_start_windows_headings_and_the_model_editors_group_box(app):
    from pymetrix import demo
    from pymetrix.commands import Session
    from pymetrix.ui.metrics import px
    from pymetrix.ui.qt import main_window

    room = px("section")
    window = main_window(Session())
    try:
        window.show()
        app.processEvents()
        start = window.start_page
        headings = [w for w in start.widget().widget().findChildren(QtWidgets.QLabel) if w.text().startswith("<b>")]
        assert headings and all(h.contentsMargins().top() == room for h in headings)
        assert _margins(start.widget().widget().layout()) == (px("normal"),) * 4  # a page in the workspace
    finally:
        window.state["session"].dirty = False
        window.close()
    window = main_window(demo("consumption"))
    try:
        window.show()
        window.open_model_editor()
        app.processEvents()
        box = next(b for s in window.mdi.subWindowList() for b in s.widget().findChildren(QtWidgets.QGroupBox)
                   if b.title() == "Impulse responses")
        layout = box.parentWidget().layout()
        index = layout.indexOf(box)
        assert layout.itemAt(index - 1).spacerItem() is not None
        assert layout.itemAt(index - 1).sizeHint().height() == room
    finally:
        window.state["session"].dirty = False
        window.close()


def test_dialog_buttons_sit_at_the_right(app, session):
    """OK, Cancel and Help (a button box) and Close in the dialogs that lay their own buttons out: their right edge
    at the dialog's margin."""
    from PySide6.QtWidgets import QPushButton

    from pymetrix.ui.metrics import px
    from pymetrix.ui.plugins_qt import plugin_manager
    from pymetrix.ui.qt import main_window

    def right_edge(dialog, text):
        button = next(b for b in dialog.findChildren(QPushButton) if b.text().replace("&", "") == text and b.isVisible())
        return button.mapTo(dialog, button.rect().topRight()).x() + 1

    dialog = _dialog(session, "FitSARIMA")
    dialog.show()
    app.processEvents()
    assert right_edge(dialog, "Help") == dialog.width() - px("dialog_margin")
    dialog.close()
    window = main_window(session)
    try:
        plugins = plugin_manager(window)
        plugins.show()
        app.processEvents()
        assert right_edge(plugins, "Close") == plugins.width() - px("dialog_margin")
        plugins.close()
    finally:
        session.dirty = False
        window.close()


def test_spacing_follows_a_text_size_chosen_while_open(app, session):
    from pymetrix.ui import text_size
    from pymetrix.ui.metrics import px

    dialog = _dialog(session, "FitSARIMA")
    try:
        text_size.apply("Largest")
        assert _margins(dialog.layout()) == (px("dialog_margin"),) * 4 == (round(12 * 1.3),) * 4
        assert dialog.code_toggle.parentWidget().layout().contentsMargins().top() == round(12 * 1.3)
    finally:
        text_size.apply("Normal")
        dialog.deleteLater()
    assert _margins(dialog.layout()) == (12,) * 4


def test_the_literal_check_finds_a_bare_number(tmp_path):
    """The scan above would see a stray number (it parses calls, whatever their spacing)."""
    source = "layout.setContentsMargins(4, 0, 0, 0)\nlayout.setSpacing( 0 )\nrow.addSpacing(6)\n"
    tree = ast.parse(source)
    hits = [ast.get_source_segment(source, n) for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in CALLS and any(isinstance(a, ast.Constant) and a.value != 0 for a in n.args)]
    assert hits == ["layout.setContentsMargins(4, 0, 0, 0)", "row.addSpacing(6)"]


def test_the_guides_sizes_are_the_apps_tokens():
    """The Guide panel (a plugin: pymetrix's public surface only) writes the tokens' sizes out: the same numbers."""
    guide = pytest.importorskip("pymetrix_guide.app")
    from pymetrix.ui.metrics import TOKENS

    assert {k: TOKENS[k] for k in guide._SIZES} == guide._SIZES
