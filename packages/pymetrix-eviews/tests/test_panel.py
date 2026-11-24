"""File → Translate EViews program… and the EViews translation panel: the dialog's step equals translate_program and
its script replays; the panel (built offscreen with an AppInterface, as the app builds it) shows a row per line with
its status in words and colour, filters each status with its count, follows the selected row in the program and the
script (an included program's text for its rows), copies the rows and the script, and runs the steps through the
interface's runner, recorded and undoable, stopping at the first failure and marking its row."""

import os

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor

from pymetrix.commands import Session
from pymetrix.commands.registry import COMMAND_IDS
from pymetrix.extension import AppInterface
from pymetrix_eviews import translate_program
from pymetrix_eviews.app import TranslateEViewsProgram
from pymetrix_eviews.panel import COLOURS, create_panel

LIB = "' the library\nsubroutine mark(string %name)\n  series {%name} = 1\nendsub\n"
MAIN = """wfcreate a 2001 2003
include lib
series x = 1
for !i = 1 to 2
  series y{!i} = x * !i
next
call mark(%0)
graph g.line x
"""


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def program(tmp_path):
    (tmp_path / "lib.prg").write_text(LIB)
    path = tmp_path / "main.prg"
    path.write_text(MAIN)
    return path


@pytest.fixture
def panel(app, program):
    s = Session()
    messages, dialogs = [], []
    interface = AppInterface(lambda: s, lambda: None, lambda cls, initial: dialogs.append(cls), s.run, messages.append)
    widget = create_panel(interface)
    widget.session, widget.interface, widget.messages, widget.dialogs = s, interface, messages, dialogs
    return widget


def _translate(panel, program, **fields):
    panel.session.run(TranslateEViewsProgram(path=str(program), args="first", **fields))
    panel.interface._changed()  # what the app does after a step
    return panel.current


def test_the_dialog_is_the_api_call_and_its_script_replays(program):
    from pymetrix.help import topic

    assert COMMAND_IDS["eviews:translate_program"].menu == "File/Translate EViews program..."
    assert TranslateEViewsProgram.topics == ("eviews-programs",)
    text = topic("eviews-programs").text
    assert "File > Translate EViews program" in text and "Run as steps" in text and "program's arguments" in text
    s = Session()
    t = s.run(TranslateEViewsProgram(path=str(program), args=" first  "))
    direct = translate_program(program, args="first")
    assert t.script == direct.script and t.report.rows == direct.report.rows
    assert s.objects == {}  # the step translates; it changes no data
    files = s.attachments["pymetrix-eviews"]
    assert files["main.py"].decode() == t.script and files["main/lib.prg"].decode() == LIB
    code = s.script()
    assert "main_translation = translate_program(" in code and "args='first'" in code
    namespace: dict = {}
    exec(code, namespace)
    assert namespace["main_translation"].script == t.script
    assert TranslateEViewsProgram(path=str(program)).validate(s) == []
    assert "choose the EViews program" in TranslateEViewsProgram().validate(s)[0]
    assert "there is no file" in TranslateEViewsProgram(path=str(program), data="nothere.csv").validate(s)[0]


def test_the_panel_shows_the_report_with_status_words_colours_and_counts(panel, program):
    assert panel.table.rowCount() == 0 and not panel.message.isHidden() and not panel.run_button.isEnabled()
    t = _translate(panel, program)
    assert panel.message.isHidden() and panel.table.rowCount() == len(t.report) == 8 + 4
    for i, row in enumerate(t.report):
        status = panel.table.item(i, 1)
        assert status.text() == row.status  # the status in words, not only in colour
        assert status.background().color() == QColor(*COLOURS[row.status])
    assert panel.table.item(2, 0).text() == "lib.prg:1"  # the included program's rows, under its name
    counts = t.report.counts()
    assert [b.text() for b in panel.filters.values()] == [f"{s} ({n})" for s, n in counts.items()]
    assert counts["left out"] == 1  # the graph
    panel.filters["translated"].setChecked(False)
    hidden = [panel.table.isRowHidden(i) for i in range(panel.table.rowCount())]
    assert hidden == [r.status == "translated" for r in t.report]
    panel.filters["translated"].setChecked(True)
    assert not any(panel.table.isRowHidden(i) for i in range(panel.table.rowCount()))


def test_selecting_a_row_follows_it_in_the_program_and_the_script(panel, program):
    t = _translate(panel, program)
    panel.table.selectRow(panel.row_of(5))  # series y{!i} = x * !i, inside the loop
    assert panel.source.toPlainText() == MAIN and panel.source_line() == 5
    assert panel.script.toPlainText().split("\n")[panel.script_at() - 1].strip() == "# 5: series y{!i} = x * !i"
    panel.table.selectRow(panel.row_of(3, "lib.prg"))  # the included program's line: its own text
    assert panel.source.toPlainText() == LIB and panel.source_line() == 3
    assert panel.script.toPlainText().split("\n")[panel.script_at() - 1].strip() == "# 3: series {%name} = 1"
    assert t.script_line(3, "lib.prg") == panel.script_at()


def test_copy_and_copy_script(panel, program):
    t = _translate(panel, program)
    clipboard = QtWidgets.QApplication.clipboard()
    assert panel.copy() == t.report.to_text() and clipboard.text() == t.report.to_text()  # nothing selected: all
    panel.table.selectRow(panel.row_of(8))
    assert panel.copy() == t.report.to_text([t.report.line(8)]) and "graph g.line x" in clipboard.text()
    assert panel.copy_script() == t.script == clipboard.text()
    panel.translate_button.click()
    assert panel.dialogs == [TranslateEViewsProgram]


def test_run_as_steps_through_the_interface_recorded_and_undoable(panel, program):
    t = _translate(panel, program)
    ran = []
    panel.interface._run = lambda c: (ran.append(c), panel.session.run(c))[1]
    assert panel.run_steps()
    s = panel.session
    frame = s["untitled"].frame
    assert list(frame.columns) == ["x", "y1", "y2", "first"] and frame["y2"].tolist() == [2.0] * 3
    assert len(ran) == 5  # the page, x, y1, y2 and the subroutine's series: each a recorded step, none a loop
    assert [r.command for r in s.history[1:]] == ran
    s.undo()  # the last step (the subroutine's series) undone
    assert list(s["untitled"].frame.columns) == ["x", "y1", "y2"]
    assert "Ran main.prg as steps" in panel.messages[-1]
    assert t is panel.current


def test_run_as_steps_stops_at_the_first_failure_and_marks_its_row(panel, tmp_path):
    path = tmp_path / "bad.prg"
    path.write_text("wfcreate a 2001 2003\nfor !i = 1 to 2\n  series s{!i} = q{!i} * 2\nnext\nseries after = 1\n")
    panel.session.run(TranslateEViewsProgram(path=str(path)))
    panel.interface._changed()
    assert not panel.run_steps()
    i = panel.row_of(3)
    assert panel.table.item(i, 1).text() == "failed" and "q1" in panel.table.item(i, 4).text()
    assert panel.table.item(i, 1).background().color() == QColor(*COLOURS["failed"])
    assert panel.table.selectedItems() and panel.table.selectedItems()[0].row() == i
    assert "after" not in panel.session["untitled"].frame and "Run as steps stopped" in panel.messages[-1]


def test_undoing_the_translate_step_and_a_second_program(panel, program, tmp_path):
    _translate(panel, program)
    other = tmp_path / "other.prg"
    other.write_text("wfcreate q 2000q1 2000q4\nseries z = 1\n")
    panel.session.run(TranslateEViewsProgram(path=str(other)))
    panel.interface._changed()
    assert panel.program.count() == 2 and panel.program.currentIndex() == 1 and panel.current.program == "other.prg"
    panel.program.setCurrentIndex(0)
    assert panel.current.program == "main.prg"
    panel.session.undo()
    panel.interface._changed()
    assert panel.program.count() == 1 and panel.current.program == "main.prg"
