"""Keys named in the app's own words on each platform: "Ctrl+K" on Linux and Windows, "⌘K" on a Mac.

Shortcuts are set as "Ctrl+K", which Qt makes the Command key on a Mac; text that names a key (tooltips, the Start
window's tour, the Keyboard shortcuts list) goes through ``pymetrix.ui.keys.key_text`` so it says what the key is
there. The Mac's text is checked on every system by setting ``keys.MAC`` (as tests/test_ui_dialog_checks.py does for
message box titles), and on a Mac the simulated text is checked against Qt's own.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtGui import QAction, QKeySequence  # noqa: E402

from pymetrix.commands import Session  # noqa: E402
from pymetrix.ui import keys  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NATIVE = QKeySequence.SequenceFormat.NativeText

# every key sequence the app names in text, and a few more of the kinds Qt writes with symbols on a Mac
SEQUENCES = ("Ctrl+K", "Ctrl+S", "Ctrl+F", "Ctrl+/", "Ctrl+`", "Ctrl+1", "Ctrl+9", "Ctrl+Return", "Ctrl+PgDown",
             "Ctrl+PgUp", "Ctrl+Shift+Z", "Ctrl+Y", "Ctrl+=", "Ctrl++", "Ctrl+-", "Ctrl+0", "Ctrl+Shift+N",
             "Ctrl+Alt+Del", "Meta+Tab", "Alt+Shift+Backspace", "F5", "Esc")
MAC_EXPECTED = {"Ctrl+K": "⌘K", "Ctrl+Shift+Z": "⇧⌘Z", "Ctrl++": "⌘+", "Ctrl+-": "⌘-", "Ctrl+Return": "⌘↵",
                "Ctrl+PgDown": "⌘⇟", "Ctrl+`": "⌘`", "Meta+Tab": "⌃⇥", "Ctrl+Alt+Del": "⌥⌘⌦", "F5": "F5"}

# "Ctrl" written into the app's own text on purpose, each with its reason
ALLOWED = {
    # the Mac's secondary click is Control-click, the physical Control key, which is not Qt's Ctrl (⌘) there
    ("src/pymetrix/ui/qt.py", "Right-click (Ctrl+click on a Mac)"),
}
ALLOWED_FILES = {
    # a help topic written as prose for every platform (help topics are not made per platform)
    "src/pymetrix/coming_from.py",
}
KEY_SPEC = re.compile(r"(?:(?:Ctrl|Shift|Alt|Meta)\+)+\S*")  # "Ctrl+Shift+N": a key given to Qt, not text shown


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def mac(monkeypatch):
    """The Mac's words on any system."""
    monkeypatch.setattr(keys, "MAC", True)


def test_ctrl_k_is_ctrl_k_on_linux_and_windows_and_command_k_on_a_mac(app, monkeypatch):
    monkeypatch.setattr(keys, "MAC", sys.platform == "darwin")
    expected = "⌘K" if sys.platform == "darwin" else "Ctrl+K"
    assert keys.key_text("Ctrl+K") == expected == QKeySequence("Ctrl+K").toString(NATIVE)
    assert keys.key_text(QKeySequence("Ctrl+K")) == expected
    if sys.platform != "darwin":
        assert keys.key_text("Ctrl+Shift+Z") == "Ctrl+Shift+Z" and keys.key_text("F5") == "F5"
    assert keys.key_text(QKeySequence()) == ""


def test_the_mac_path_on_any_system(app, mac):
    assert keys.key_text("Ctrl+K") == "⌘K"
    for sequence, text in MAC_EXPECTED.items():
        assert keys.key_text(sequence) == text, sequence
    assert keys.key_text("Ctrl+Y, Ctrl+Shift+Z") == "⌘Y, ⇧⌘Z"
    assert keys.key_text(QKeySequence()) == ""


@pytest.mark.skipif(sys.platform != "darwin", reason="Qt writes a Mac's symbols only on a Mac")
def test_the_simulated_mac_text_is_qts_own_on_a_mac(app):
    for sequence in SEQUENCES:
        assert keys.mac_text(sequence) == QKeySequence(sequence).toString(NATIVE), sequence
    for standard in (QKeySequence.StandardKey.Save, QKeySequence.StandardKey.Find, QKeySequence.StandardKey.Replace):
        assert keys.mac_text(standard) == QKeySequence(standard).toString(NATIVE), standard


def _sources():
    yield from sorted((ROOT / "src" / "pymetrix").rglob("*.py"))
    yield from sorted(ROOT.glob("packages/*/src/**/*.py"))


def _docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                yield id(first.value)


def test_no_ctrl_written_into_the_apps_text():
    """Every "Ctrl+" or "Ctrl-" in a string of the app's code (not its docstrings or comments) is a key given to Qt
    ("Ctrl+K", which Qt maps per platform) or on the list above: anything else is text that would say Ctrl on a Mac."""
    found = []
    for path in _sources():
        rel = path.relative_to(ROOT).as_posix()
        if rel in ALLOWED_FILES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docs = set(_docstrings(tree))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs
                    and re.search(r"Ctrl[+-]", node.value) and not KEY_SPEC.fullmatch(node.value)
                    and (rel, node.value) not in ALLOWED):
                found.append(f"{rel}:{node.lineno}: {node.value!r}")
    assert not found, "\n".join(found)


def _texts(*widgets):
    for top in widgets:
        for item in [top, *top.findChildren(QtWidgets.QWidget), *top.findChildren(QAction)]:
            for getter in ("toolTip", "statusTip", "whatsThis", "text", "placeholderText"):
                value = getattr(item, getter, None)
                if callable(value):
                    try:
                        text = value()
                    except TypeError:
                        continue
                    if isinstance(text, str) and text:
                        yield f"{type(item).__name__}.{getter}", text


def test_the_windows_say_the_macs_keys_on_a_mac(app, mac):
    """The main window (with its Start page), a Code window, a spreadsheet and a result's views, built as on a Mac:
    no tooltip or text says Ctrl+, and the keys they name are ⌘ ones."""
    from pymetrix.ui.editor_qt import EditorWindow
    from pymetrix.ui.qt import main_window, shortcut_list
    from pymetrix.ui.sheet_qt import SheetWindow
    from pymetrix.ui.view_switcher import ViewSwitcher
    from pymetrix import Dataset

    import pandas as pd

    window = main_window(Session())
    editor = EditorWindow(lambda code, line, w: None, lambda: {})
    idx = pd.period_range("2020Q1", periods=3, freq="Q", name="time")
    data = Dataset.from_frame(pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=idx))
    sheet = SheetWindow("data", lambda: data, lambda command: True)
    views = ViewSwitcher(kind="test")
    try:
        texts = list(_texts(window, editor, sheet, views))
        said = [(where, text) for where, text in texts if re.search(r"Ctrl[+-]", text)
                and "Ctrl+click on a Mac" not in text]
        assert not said, said
        joined = "\n".join(text for _, text in texts)
        for expected in ("⌘K", "⌘1", "⌘`", "⌘F", "⌘/", "⌘↵", "⌘⇟"):
            assert expected in joined, expected
        assert ("Code windows", "Find text", "⌘F") in shortcut_list(window)
        tips = {a.text(): a.toolTip() for a in editor.actions_by_name.values()}
        assert tips["Find"] == "find (⌘F)" and tips["Comment"] == "comment or uncomment the selected lines (⌘/)"
    finally:
        window.state["session"].dirty = False
        for w in (window, editor, sheet, views):
            w.close()


def test_the_windows_name_the_keys_as_the_platform_does(app, monkeypatch):
    """The same tooltips, built as this system builds them: Qt's own words for the keys (Ctrl+F on Linux and
    Windows, ⌘F on a Mac)."""
    monkeypatch.setattr(keys, "MAC", sys.platform == "darwin")
    from pymetrix.ui.editor_qt import EditorWindow

    editor = EditorWindow(lambda code, line, w: None, lambda: {})
    try:
        tips = {a.text(): a.toolTip() for a in editor.actions_by_name.values()}
        find = QKeySequence(QKeySequence.StandardKey.Find).toString(NATIVE)
        assert tips["Find"] == f"find ({find})" and tips["Comment"].endswith(f"({QKeySequence('Ctrl+/').toString(NATIVE)})")
        save = QKeySequence(QKeySequence.StandardKey.Save).toString(NATIVE)
        assert tips["Save"] == f"save the script ({save} while editing)"
        if sys.platform != "darwin":
            assert find == "Ctrl+F" and save == "Ctrl+S"
    finally:
        editor.close()
