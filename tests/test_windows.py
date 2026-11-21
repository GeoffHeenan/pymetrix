"""The desktop windows: History, Immediate, Code editor, and out-of-date results."""

import os
from pathlib import Path

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from pymetrix import Dataset  # noqa: E402
from pymetrix.commands import FitEquation, Session  # noqa: E402

DATA = Path(__file__).parents[1] / "examples" / "sample_macro.parquet"


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app):
    pytest.importorskip("qtconsole")
    from pymetrix.ui.qt import main_window

    session = Session()
    session.objects["data"] = Dataset.read(str(DATA))
    w = main_window(session)
    w.show()
    yield w
    for editor in w.editors():
        editor.editor.document().setModified(False)
    w.state["session"].dirty = False
    w.close()


def test_immediate_window_records_and_flags_out_of_date(window, app):
    s = window.state["session"]
    s.run(FitEquation("c ~ y", name="fit"))
    window.refresh()
    window.send_immediate("ly = data['y'].mean()")
    app.processEvents()
    assert s["ly"] == pytest.approx(float(s["data"]["y"].mean()))
    assert window.history.topLevelItemCount() == 2 and window.history.topLevelItem(1).text(1) == "Immediate"
    window.send_immediate("data = data.with_series(y=data['y'] * 2)")
    app.processEvents()
    status = {window.objects.topLevelItem(i).text(0): window.objects.topLevelItem(i).text(2) for i in range(window.objects.topLevelItemCount())}
    assert status == {"data": "", "fit": "out of date", "ly": "out of date"}
    window.recompute()
    assert s.stale() == {} and s["ly"] == pytest.approx(float(s["data"]["y"].mean()))
    window.send_immediate("1/0")
    app.processEvents()
    last = window.history.topLevelItem(window.history.topLevelItemCount() - 1)
    assert last.text(3) == "failed" and "ZeroDivisionError" in last.toolTip(2)


def test_history_copy_send_and_reopen(window, app):
    from pymetrix.ui.qt import command_dialog

    s = window.state["session"]
    s.run(FitEquation("y ~ c", method="ML", variance="GJR-GARCH", variance_settings="garch=2", dist="t", name="vol"))
    window.send_immediate("x = 1")
    app.processEvents()
    window.refresh()
    window.history.selectAll()
    copied = window.history.copy()
    assert copied.startswith("from pymetrix import *") and "x = 1" in copied and "garch=2" in copied
    assert window.history.mimeData([window.history.topLevelItem(1)]).text() == "x = 1"
    window.send_editor("x = 2")
    editor = window.editors()[0]
    assert editor.editor.toPlainText().startswith("from pymetrix import *") and "x = 2" in editor.editor.toPlainText()
    dialog = command_dialog(FitEquation, s, initial=s.history[0].command)  # Open dialog: filled in as it was
    assert dialog.editors["method"].get() == "ML" and dialog.editors["variance"].get() == "GJR-GARCH"
    assert dialog.editors["dist"].get() == "t" and dialog.editors["variance_settings"].sub["garch"].get() == 2
    assert dialog.editors["variance_settings"].get() == "garch=2"


def test_code_window_runs_and_marks_errors(window, app, tmp_path):
    editor = window.open_editor(text="a = 1\nb = a + missing_name\n")
    editor.run_all()
    app.processEvents()
    assert editor.editor.error_line == 2 and not window.state["session"].history[-1].ok
    assert window.state["session"].history[-1].command.source == "editor"
    editor.editor.setPlainText("a = 1\nb = a + 1\n")
    cursor = editor.editor.textCursor()
    cursor.movePosition(cursor.MoveOperation.Start)
    editor.editor.setTextCursor(cursor)
    editor.run_selection()  # runs line 1 and moves to line 2
    app.processEvents()
    assert window.state["session"]["a"] == 1 and editor.editor.textCursor().blockNumber() == 1
    path = tmp_path / "script.py"
    editor.path = path
    editor.save()
    assert path.read_text(encoding="utf-8") == "a = 1\nb = a + 1\n" and not editor.editor.document().isModified()


def _editor(app, text=""):
    from pymetrix.ui.editor_qt import CodeEditor

    session = Session()
    session.objects["data"] = Dataset.read(str(DATA))
    session.run(FitEquation("c ~ y", name="fit"))
    editor = CodeEditor(lambda: session.namespace)
    editor.setPlainText(text)
    editor.show()
    return editor


def _type(editor, keys):
    for key in keys:
        if isinstance(key, str):
            QTest.keyClicks(editor, key)
        else:
            QTest.keyClick(editor, key)


def test_editor_line_numbers_quiet_through_construction_and_teardown(app):
    import subprocess
    import sys

    from pymetrix.ui.editor_qt import _LineNumbers

    assert _LineNumbers.__new__(_LineNumbers).sizeHint().width() == 0  # no editor yet: no AttributeError
    script = (
        "import gc\n"
        "from PySide6 import QtWidgets\n"
        "from pymetrix.ui.editor_qt import CodeEditor\n"
        "app = QtWidgets.QApplication([])\n"
        "for _ in range(3):\n"
        "    e = CodeEditor()\n"
        "    e.setPlainText('x = 1\\n' * 50)\n"
        "    e.show(); app.processEvents()\n"
        "    e.close(); e.deleteLater(); app.processEvents()\n"
        "    del e; gc.collect()\n"
    )
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=120, check=False)
    assert run.returncode == 0, run.stderr
    assert "_LineNumbers" not in run.stderr


def test_editor_indents_comments_and_matches(app):
    editor = _editor(app)
    _type(editor, ["def f(x):", Qt.Key.Key_Return, "if x:", Qt.Key.Key_Return, "return 1", Qt.Key.Key_Return, "y = 2"])
    assert editor.toPlainText() == "def f(x):\n    if x:\n        return 1\n    y = 2"
    editor.selectAll()
    editor.toggle_comment()
    assert editor.toPlainText().splitlines()[0] == "# def f(x):" and editor.toPlainText().splitlines()[1] == "#     if x:"
    editor.selectAll()
    editor.toggle_comment()
    assert editor.toPlainText() == "def f(x):\n    if x:\n        return 1\n    y = 2"
    editor.selectAll()
    QTest.keyClick(editor, Qt.Key.Key_Tab)
    assert editor.toPlainText().splitlines()[0] == "    def f(x):"
    QTest.keyClick(editor, Qt.Key.Key_Backtab)
    assert editor.toPlainText().splitlines()[0] == "def f(x):"
    editor.setPlainText("z = (1 + [2, 3])")
    cursor = editor.textCursor()
    cursor.setPosition(4)  # just before "("
    editor.setTextCursor(cursor)
    marked = [s.cursor.selectionStart() for s in editor.extraSelections() if s.cursor.hasSelection()]
    assert 4 in marked and len(editor.toPlainText()) - 1 in marked
    editor.setPlainText('s = """one\n# not a comment\n"""\nx = 1  # comment')
    app.processEvents()  # the highlighter copes with strings across lines
    assert editor.document().findBlockByNumber(1).userState() in (-1, 0, 1)


def test_editor_completion_signatures_and_checks(app):
    editor = _editor(app, "fit.")
    cursor = editor.textCursor()
    cursor.movePosition(cursor.MoveOperation.End)
    editor.setTextCursor(cursor)
    names = editor.completions()
    assert {"params", "resid", "summary"} <= set(names)  # from the live object in the session
    editor.setPlainText("data.with_series(")
    c = editor.textCursor()
    c.movePosition(c.MoveOperation.End)
    editor.setTextCursor(c)
    assert "with_series" in editor.signature()
    editor.setPlainText("import os\ny = fit.params\nz = not_defined + 1\n")
    problems = editor.check()
    assert list(problems) == [3] and "not_defined" in problems[3]  # session names count as defined


def test_find_and_replace(app):
    from pymetrix.ui.editor_qt import FindBar

    editor = _editor(app, "alpha = 1\nAlpha = 2\nalpha += 3\n")
    bar = FindBar(editor)
    bar.find.setText("alpha")
    bar.replace.setText("beta")
    assert bar.replace_all() == 3
    assert editor.toPlainText() == "beta = 1\nbeta = 2\nbeta += 3\n"
    bar.find.setText("beta")
    bar.case.setChecked(True)
    bar.replace.setText("b")
    assert bar.replace_all() == 3


def test_period_spinners_follow_the_dataset_and_check_the_order(app):
    from pymetrix.commands import FitVAR
    from pymetrix.ui.qt import command_dialog

    session = Session()
    session.objects["data"] = Dataset.read(str(DATA))
    session.objects["annual"] = Dataset.read(str(DATA)).to_freq("Y", how="mean")
    dialog = command_dialog(FitVAR, session)
    start, end = dialog.editors["start"].widget, dialog.editors["end"].widget
    assert start.periods[0] == "2000Q1" and start.periods[-1] == "2024Q4"
    # empty: the whole sample, shown as the default it is, in the text's colour (grey looked disabled)
    assert start.lineEdit().placeholderText() == "2000Q1 (first period)" and end.lineEdit().placeholderText() == "2024Q4 (last period)"
    from PySide6.QtGui import QPalette

    palette = start.lineEdit().palette()
    assert palette.color(QPalette.ColorRole.PlaceholderText) == palette.color(QPalette.ColorRole.Text)
    start.stepBy(-1)
    end.stepBy(1)
    assert dialog.editors["start"].get() == "2000Q1" and dialog.editors["end"].get() == "2024Q4"  # either arrow, from the ends
    start.lineEdit().setText("")
    end.lineEdit().setText("")
    start.stepBy(1)
    end.stepBy(-1)
    assert dialog.editors["start"].get() == "2000Q1" and dialog.editors["end"].get() == "2024Q4"  # from empty: first and last
    start.stepBy(8)
    assert dialog.editors["start"].get() == "2002Q1"
    end.stepBy(-100)  # clamps at the first period, before the start
    app.processEvents()
    ok = dialog.findChild(QtWidgets.QDialogButtonBox).button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    assert not ok.isEnabled() and "after the end" in end.toolTip() and "span" in dialog.ok_gate.reasons
    end.lineEdit().setText("")
    assert "span" not in dialog.ok_gate.reasons  # the order is fixed (OK may still wait for a required field)
    dialog.editors["data"].widget.setCurrentText("annual")  # another dataset: its periods
    app.processEvents()
    assert start.periods[0] == "2000" and len(start.periods) == 25
    dialog.close()

    session.objects["data"] = session["data"].with_sample(("2005Q1", "2009Q4"))  # an active sample: empty means it
    dialog = command_dialog(FitVAR, session)
    start, end = dialog.editors["start"].widget, dialog.editors["end"].widget
    assert start.lineEdit().placeholderText() == "2005Q1 (first period)" and end.lineEdit().placeholderText() == "2009Q4 (last period)"
    start.stepBy(1)
    end.stepBy(1)
    assert dialog.editors["start"].get() == "2005Q1" and dialog.editors["end"].get() == "2009Q4"
    dialog.close()
