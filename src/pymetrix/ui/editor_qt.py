"""The Code window: a Python editor for scripts that run in the session.

Built on Qt's plain-text editor: syntax colouring, line numbers, auto-indent,
block indent and comment, bracket matching, find and replace, completion
from Jedi (which sees the session's live objects, so ``fit.`` offers
``params``), signature hints, and pyflakes checks as you type. Running code
sends it to the Immediate window, so it is recorded like anything typed there.
"""

from __future__ import annotations

import keyword
import re
from pathlib import Path
from typing import Any, Callable

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from . import theme
from .keys import key_text
from .metrics import set_margins

INDENT = "    "
DEDENT_AFTER = re.compile(r"^\s*(return|pass|break|continue|raise)\b")
BRACKETS = {"(": ")", "[": "]", "{": "}"}


def _dark(widget: QtWidgets.QWidget) -> bool:
    return widget.palette().color(QtGui.QPalette.ColorRole.Base).lightness() < 128


def _format(color: str, bold: bool = False, italic: bool = False) -> QtGui.QTextCharFormat:
    fmt = QtGui.QTextCharFormat()
    fmt.setForeground(QtGui.QColor(color))
    if bold:
        fmt.setFontWeight(QtGui.QFont.Weight.Bold)
    fmt.setFontItalic(italic)
    return fmt


class PythonHighlighter(QtGui.QSyntaxHighlighter):
    """Keywords, built-ins, numbers, strings (including triple-quoted across lines), comments, decorators."""

    def __init__(self, document: QtGui.QTextDocument, dark: bool = False) -> None:
        super().__init__(document)
        c = theme.PYTHON_SYNTAX["dark" if dark else "light"]
        import builtins

        names = [b for b in dir(builtins) if not b.startswith("_")]
        self.rules = [  # (pattern, format, group to colour)
            (re.compile(r"\b(" + "|".join(keyword.kwlist + keyword.softkwlist) + r")\b"), _format(c["kw"], bold=True), 0),
            (re.compile(r"\b(" + "|".join(names) + r")\b"), _format(c["builtin"]), 0),
            (re.compile(r"\b(?:def|class)\s+(\w+)"), _format(c["def"], bold=True), 1),
            (re.compile(r"\b\d+(\.\d*)?([eE][-+]?\d+)?\b"), _format(c["num"]), 0),
            (re.compile(r"^\s*@[\w.]+"), _format(c["deco"]), 0),
        ]
        self.string = _format(c["str"])
        self.comment = _format(c["comment"], italic=True)
        self.single = re.compile(r"""[rbfu]{0,2}("([^"\\\n]|\\.)*"|'([^'\\\n]|\\.)*')""", re.IGNORECASE)

    def highlightBlock(self, text: str) -> None:  # noqa: N802 (Qt override)
        for pattern, fmt, group in self.rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(group), m.end(group) - m.start(group), fmt)
        # strings and comments, scanning left to right so '#' inside strings is not a comment
        self.setCurrentBlockState(0)
        i = 0
        state = self.previousBlockState()
        if state in (1, 2):  # inside a triple-quoted string from an earlier line
            quote = '"""' if state == 1 else "'''"
            end = text.find(quote)
            if end < 0:
                self.setFormat(0, len(text), self.string)
                self.setCurrentBlockState(state)
                return
            self.setFormat(0, end + 3, self.string)
            i = end + 3
        while i < len(text):
            ch = text[i]
            if ch == "#":
                self.setFormat(i, len(text) - i, self.comment)
                return
            if text.startswith(('"""', "'''"), i):
                quote = text[i:i + 3]
                end = text.find(quote, i + 3)
                if end < 0:
                    self.setFormat(i, len(text) - i, self.string)
                    self.setCurrentBlockState(1 if quote == '"""' else 2)
                    return
                self.setFormat(i, end + 3 - i, self.string)
                i = end + 3
                continue
            if ch in "\"'":
                m = self.single.match(text, i)
                if m:
                    self.setFormat(i, m.end() - i, self.string)
                    i = m.end()
                    continue
            i += 1


class _LineNumbers(QtWidgets.QWidget):
    def __init__(self, editor: CodeEditor) -> None:
        self.editor = editor  # before the base initialiser, which can already deliver events
        super().__init__(editor)

    # Qt can also call these while the widget is torn down, after the attribute is gone.
    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        editor = getattr(self, "editor", None)
        return QtCore.QSize(editor.number_width() if editor is not None else 0, 0)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        editor = getattr(self, "editor", None)
        if editor is not None:
            editor.paint_numbers(event)


class CodeEditor(QtWidgets.QPlainTextEdit):
    """A Python editor. ``namespace`` (a callable returning a dict) supplies live objects for completion."""

    def __init__(self, namespace: Callable[[], dict[str, Any]] | None = None, parent: Any = None) -> None:
        super().__init__(parent)
        self.namespace = namespace or (lambda: {})
        from .text_size import fixed_font

        font = fixed_font(least=10)  # at the app's text size (View → Text size)
        self.setFont(font)
        self.setTabStopDistance(QtGui.QFontMetricsF(font).horizontalAdvance(" ") * 4)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        # The editor does not own its document. A document the editor owns is deleted inside the
        # editor's destructor and emits signals on the way out, which reach the Python slots below
        # on a half-destroyed widget and crash the process (when a window closes, or at exit).
        self._document = QtGui.QTextDocument()
        self._document.setDocumentLayout(QtWidgets.QPlainTextDocumentLayout(self._document))
        self.setDocument(self._document)
        self.highlighter = PythonHighlighter(self.document(), _dark(self))
        self.numbers = _LineNumbers(self)
        self.blockCountChanged.connect(self._set_margins)
        self.updateRequest.connect(self._update_numbers)
        self.cursorPositionChanged.connect(self._decorate)
        self.setViewportMargins(self.number_width(), 0, 0, 0)
        self.problems: dict[int, str] = {}  # line (1-based) -> pyflakes message
        self.error_line: int | None = None  # from the last run
        self._check_timer = QtCore.QTimer(self, singleShot=True, interval=600)
        self._check_timer.timeout.connect(self.check)
        self.textChanged.connect(self._check_timer.start)
        self.textChanged.connect(self._clear_error)
        self.completer = QtWidgets.QCompleter(self)
        self.completer.setWidget(self)
        self.completer.setCompletionMode(QtWidgets.QCompleter.CompletionMode.PopupCompletion)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.activated.connect(self._insert_completion)
        self._prefix = ""
        self._decorate()

    # -- line numbers -----------------------------------------------------------------

    def _set_margins(self, _: int = 0) -> None:
        self.setViewportMargins(self.number_width(), 0, 0, 0)

    def number_width(self) -> int:
        digits = max(3, len(str(self.blockCount())))
        return 12 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_numbers(self, rect: QtCore.QRect, dy: int) -> None:
        if dy:
            self.numbers.scroll(0, dy)
        else:
            self.numbers.update(0, rect.y(), self.numbers.width(), rect.height())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.numbers.setGeometry(QtCore.QRect(cr.left(), cr.top(), self.number_width(), cr.height()))

    def paint_numbers(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self.numbers)
        pal = self.palette()
        painter.fillRect(event.rect(), pal.color(QtGui.QPalette.ColorRole.AlternateBase))
        block = self.firstVisibleBlock()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        height = round(self.blockBoundingRect(block).height())
        current = self.textCursor().blockNumber()
        while block.isValid() and top <= event.rect().bottom():
            line = block.blockNumber() + 1
            if block.isVisible() and top + height >= event.rect().top():
                color = theme.colour("error", pal) if line == self.error_line or line in self.problems else None
                painter.setPen(color if color else pal.color(QtGui.QPalette.ColorRole.Text if block.blockNumber() == current else QtGui.QPalette.ColorRole.PlaceholderText))
                painter.drawText(0, top, self.numbers.width() - 6, height, Qt.AlignmentFlag.AlignRight, str(line))
            block = block.next()
            top += height
            height = round(self.blockBoundingRect(block).height())

    # -- highlights: current line, matching bracket, problems, run error ---------------

    def _decorate(self) -> None:
        selections = []
        dark = _dark(self)
        line = QtWidgets.QTextEdit.ExtraSelection()
        line.format.setBackground(QtGui.QColor(theme.EDITOR_BACKGROUNDS["current line"][0 if dark else 1]))
        line.format.setProperty(QtGui.QTextFormat.Property.FullWidthSelection, True)
        line.cursor = self.textCursor()
        line.cursor.clearSelection()
        selections.append(line)
        if self.error_line:
            block = self.document().findBlockByNumber(self.error_line - 1)
            err = QtWidgets.QTextEdit.ExtraSelection()
            err.format.setBackground(QtGui.QColor(theme.EDITOR_BACKGROUNDS["error line"][0 if dark else 1]))
            err.format.setProperty(QtGui.QTextFormat.Property.FullWidthSelection, True)
            err.cursor = QtGui.QTextCursor(block)
            selections.append(err)
        for number in self.problems:
            block = self.document().findBlockByNumber(number - 1)
            if not block.isValid():
                continue
            mark = QtWidgets.QTextEdit.ExtraSelection()
            mark.format.setUnderlineStyle(QtGui.QTextCharFormat.UnderlineStyle.WaveUnderline)
            mark.format.setUnderlineColor(theme.colour("warning"))
            cursor = QtGui.QTextCursor(block)
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.EndOfBlock, QtGui.QTextCursor.MoveMode.KeepAnchor)
            mark.cursor = cursor
            selections.append(mark)
        selections += self._bracket_match()
        self.setExtraSelections(selections)

    def _bracket_match(self) -> list[Any]:
        text = self.toPlainText()
        pos = self.textCursor().position()
        for at in (pos, pos - 1):
            if 0 <= at < len(text) and text[at] in "()[]{}":
                other = self._partner(text, at)
                if other is None:
                    return []
                out = []
                for p in (at, other):
                    sel = QtWidgets.QTextEdit.ExtraSelection()
                    sel.format.setBackground(QtGui.QColor(theme.EDITOR_BACKGROUNDS["same word"][0 if _dark(self) else 1]))
                    cursor = self.textCursor()
                    cursor.setPosition(p)
                    cursor.setPosition(p + 1, QtGui.QTextCursor.MoveMode.KeepAnchor)
                    sel.cursor = cursor
                    out.append(sel)
                return out
        return []

    @staticmethod
    def _partner(text: str, at: int) -> int | None:
        ch = text[at]
        if ch in BRACKETS:
            close, step, opening = BRACKETS[ch], 1, ch
        else:
            opening = next(k for k, v in BRACKETS.items() if v == ch)
            close, step = opening, -1
        depth, i = 0, at
        while 0 <= i < len(text):
            if text[i] == ch:
                depth += 1
            elif text[i] == close:
                depth -= 1
                if depth == 0:
                    return i
            i += step
        return None

    def mark_error(self, line: int | None) -> None:
        """Highlight the line where the last run failed (None clears it) and move there."""
        self.error_line = line
        if line:
            cursor = QtGui.QTextCursor(self.document().findBlockByNumber(line - 1))
            self.setTextCursor(cursor)
            self.centerCursor()
        self._decorate()
        self.numbers.update()

    def _clear_error(self) -> None:
        if self.error_line is not None:
            self.error_line = None
            self._decorate()

    # -- checking ------------------------------------------------------------------------

    def check(self) -> dict[int, str]:
        """pyflakes messages by line (names from the session count as defined)."""
        try:
            from pyflakes import api, reporter
        except ImportError:
            return {}
        import io

        out, err = io.StringIO(), io.StringIO()
        api.check(self.toPlainText(), "<editor>", reporter.Reporter(out, err))
        known = set(self.namespace())
        problems: dict[int, str] = {}
        for line in (out.getvalue() + err.getvalue()).splitlines():
            m = re.match(r"<editor>:(\d+):(?:\d+:?)?\s*(.*)", line)
            if not m:
                continue
            message = m.group(2)
            undefined = re.match(r"undefined name '(\w+)'", message)
            if undefined and undefined.group(1) in known:
                continue  # defined in the session
            if "imported but unused" in message or "unable to detect undefined names" in message:
                continue
            problems.setdefault(int(m.group(1)), message)
        self.problems = problems
        self._decorate()
        self.numbers.update()
        return problems

    def changeEvent(self, e: QtCore.QEvent) -> None:  # noqa: N802 - Qt's name
        super().changeEvent(e)
        if e.type() == QtCore.QEvent.Type.FontChange:  # a text size or a zoom: tabs of four spaces in the new font
            self.setTabStopDistance(QtGui.QFontMetricsF(self.font()).horizontalAdvance(" ") * 4)
            self._set_margins()

    def event(self, e: QtCore.QEvent) -> bool:
        if e.type() == QtCore.QEvent.Type.ToolTip:
            cursor = self.cursorForPosition(self.viewport().mapFromGlobal(e.globalPos()))
            message = self.problems.get(cursor.blockNumber() + 1)
            if message:
                QtWidgets.QToolTip.showText(e.globalPos(), message, self)
            else:
                QtWidgets.QToolTip.hideText()
            return True
        return super().event(e)

    # -- editing keys ------------------------------------------------------------------------

    def _selected_blocks(self) -> tuple[int, int]:
        cursor = self.textCursor()
        doc = self.document()
        first = doc.findBlock(cursor.selectionStart()).blockNumber()
        last_pos = cursor.selectionEnd()
        last = doc.findBlock(last_pos).blockNumber()
        if cursor.hasSelection() and doc.findBlock(last_pos).position() == last_pos and last > first:
            last -= 1  # selection ends at the start of a line: that line is not included
        return first, last

    def _edit_lines(self, change: Callable[[str], str]) -> None:
        first, last = self._selected_blocks()
        cursor = self.textCursor()
        cursor.beginEditBlock()
        for number in range(first, last + 1):
            block = self.document().findBlockByNumber(number)
            new = change(block.text())
            if new != block.text():
                c = QtGui.QTextCursor(block)
                c.movePosition(QtGui.QTextCursor.MoveOperation.EndOfBlock, QtGui.QTextCursor.MoveMode.KeepAnchor)
                c.insertText(new)
        cursor.endEditBlock()
        start = self.document().findBlockByNumber(first).position()
        end_block = self.document().findBlockByNumber(last)
        cursor.setPosition(start)
        cursor.setPosition(end_block.position() + end_block.length() - 1, QtGui.QTextCursor.MoveMode.KeepAnchor)
        self.setTextCursor(cursor)

    def indent(self) -> None:
        self._edit_lines(lambda t: INDENT + t if t.strip() else t)

    def dedent(self) -> None:
        self._edit_lines(lambda t: t[min(len(t) - len(t.lstrip(" ")), 4):] if t.startswith(" ") else t.lstrip("\t"))

    def toggle_comment(self) -> None:
        first, last = self._selected_blocks()
        lines = [self.document().findBlockByNumber(n).text() for n in range(first, last + 1)]
        code = [t for t in lines if t.strip()]
        uncomment = bool(code) and all(t.lstrip().startswith("#") for t in code)
        margin = min((len(t) - len(t.lstrip()) for t in code), default=0)
        if uncomment:
            self._edit_lines(lambda t: re.sub(r"^(\s*)# ?", r"\1", t, count=1))
        else:
            self._edit_lines(lambda t: t[:margin] + "# " + t[margin:] if t.strip() else t)

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:  # noqa: N802
        popup = self.completer.popup()
        if popup.isVisible() and e.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return, Qt.Key.Key_Tab, Qt.Key.Key_Escape, Qt.Key.Key_Backtab):
            e.ignore()  # the completer handles these
            return
        key, mods = e.key(), e.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        cursor = self.textCursor()
        if key == Qt.Key.Key_Tab and cursor.hasSelection() and " " in cursor.selectedText():
            self.indent()
            return
        if key == Qt.Key.Key_Tab and not ctrl:
            self.insertPlainText(INDENT[: 4 - cursor.positionInBlock() % 4])
            return
        if key == Qt.Key.Key_Backtab:
            self.dedent()
            return
        if ctrl and key == Qt.Key.Key_Slash:
            self.toggle_comment()
            return
        if ctrl and key == Qt.Key.Key_Space:
            self.complete(force=True)
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not ctrl:
            line = cursor.block().text()[: cursor.positionInBlock()]
            indent = line[: len(line) - len(line.lstrip())]
            stripped = line.split("#")[0].rstrip()
            if stripped.endswith(":"):
                indent += INDENT
            elif DEDENT_AFTER.match(line):
                indent = indent[:-4] if indent.endswith(INDENT) else indent
            elif stripped and stripped[-1] in BRACKETS:
                indent += INDENT
            cursor.insertText("\n" + indent)
            self.ensureCursorVisible()
            return
        if key == Qt.Key.Key_Backspace and not cursor.hasSelection():
            before = cursor.block().text()[: cursor.positionInBlock()]
            if before and not before.strip() and len(before) % 4 == 0:
                for _ in range(4):
                    cursor.deletePreviousChar()
                return
        super().keyPressEvent(e)
        text = e.text()
        if text == ".":
            self.complete()
        elif text == "(":
            self.show_signature()
        elif popup.isVisible():
            self.complete()

    # -- completion and signatures --------------------------------------------------------------

    def _jedi(self) -> Any:
        import jedi

        return jedi.Interpreter(self.toPlainText(), [self.namespace()])

    def _line_col(self) -> tuple[int, int]:
        cursor = self.textCursor()
        return cursor.blockNumber() + 1, cursor.positionInBlock()

    def completions(self) -> list[str]:
        """Completions at the cursor, from the code and the session's live objects."""
        try:
            found = self._jedi().complete(*self._line_col())
        except Exception:  # completion is a convenience; never let it break typing
            return []
        return [c.name for c in found if not c.name.startswith("__")]

    def complete(self, force: bool = False) -> None:
        line = self.textCursor().block().text()[: self.textCursor().positionInBlock()]
        self._prefix = re.search(r"\w*$", line).group(0)
        if not force and not self._prefix and not line.endswith("."):
            self.completer.popup().hide()
            return
        names = self.completions()
        if not names:
            self.completer.popup().hide()
            return
        self.completer.setModel(QtCore.QStringListModel(names, self.completer))
        self.completer.setCompletionPrefix(self._prefix)
        rect = self.cursorRect()
        rect.setWidth(self.completer.popup().sizeHintForColumn(0) + self.completer.popup().verticalScrollBar().sizeHint().width() + 8)
        self.completer.complete(rect)

    def _insert_completion(self, name: str) -> None:
        cursor = self.textCursor()
        cursor.insertText(name[len(self.completer.completionPrefix()):])
        self.setTextCursor(cursor)

    def help_at(self, cursor: QtGui.QTextCursor | None = None) -> tuple[str, str] | None:
        """The name at ``cursor`` (the text cursor's place by default) and its documentation, from the code and the
        session's live objects: (``fit.forecast``, its docstring); None where there is no name with help."""
        cursor = cursor or self.textCursor()
        try:
            found = self._jedi().help(cursor.blockNumber() + 1, cursor.positionInBlock())
        except Exception:  # noqa: BLE001 - help is a convenience; never let it break the menu
            return None
        for name in found:
            doc = name.docstring()
            if doc.strip():
                return name.full_name or name.name, doc
        return None

    def context_menu(self, pos: QtCore.QPoint) -> QtWidgets.QMenu:
        """The right-click menu at ``pos``: the standard editing actions, then ``context_actions(menu, cursor)``'s
        (a Code window's run, comment and find, and Help on the name under the mouse)."""
        menu = self.createStandardContextMenu(pos)
        extra = getattr(self, "context_actions", None)
        if extra is not None:
            extra(menu, self.cursorForPosition(pos))
        return menu

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        self.context_menu(event.pos()).exec(event.globalPos())

    def signature(self) -> str:
        try:
            sigs = self._jedi().get_signatures(*self._line_col())
        except Exception:
            return ""
        return sigs[0].to_string() if sigs else ""

    def show_signature(self) -> None:
        text = self.signature()
        if text:
            QtWidgets.QToolTip.showText(self.viewport().mapToGlobal(self.cursorRect().bottomLeft()), text, self)

    # -- what to run ------------------------------------------------------------------------------

    def selection_or_line(self) -> tuple[str, int]:
        """The selected lines (whole lines), or the current line: (code, first line number)."""
        cursor = self.textCursor()
        if not cursor.hasSelection():
            return cursor.block().text(), cursor.blockNumber() + 1
        first, last = self._selected_blocks()
        lines = [self.document().findBlockByNumber(n).text() for n in range(first, last + 1)]
        import textwrap

        return textwrap.dedent("\n".join(lines)), first + 1


class FindBar(QtWidgets.QWidget):
    """Find and replace for a CodeEditor (Ctrl+F, Ctrl+H; Enter = next, Shift+Enter = previous)."""

    def __init__(self, editor: CodeEditor) -> None:
        super().__init__()
        self.editor = editor
        row = QtWidgets.QHBoxLayout(self)
        set_margins(row, "tight", "hairline")  # a compact bar, as the spreadsheet's edit line
        self.find = QtWidgets.QLineEdit(placeholderText="Find")
        self.replace = QtWidgets.QLineEdit(placeholderText="Replace with")
        self.case = QtWidgets.QCheckBox("Match case")
        buttons = [("Next", lambda: self.next()), ("Previous", lambda: self.next(back=True)),
                   ("Replace", self.replace_one), ("Replace all", self.replace_all), ("✕", self.hide)]
        row.addWidget(self.find, 2)
        row.addWidget(self.replace, 2)
        row.addWidget(self.case)
        for label, handler in buttons:
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(handler)
            row.addWidget(b)
        self.find.returnPressed.connect(lambda: self.next(back=bool(QtWidgets.QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)))
        self.hide()

    def _flags(self, back: bool = False) -> QtGui.QTextDocument.FindFlag:
        flags = QtGui.QTextDocument.FindFlag(0)
        if back:
            flags |= QtGui.QTextDocument.FindFlag.FindBackward
        if self.case.isChecked():
            flags |= QtGui.QTextDocument.FindFlag.FindCaseSensitively
        return flags

    def open(self, replace: bool = False) -> None:
        self.show()
        self.replace.setVisible(True)
        selected = self.editor.textCursor().selectedText()
        if selected and " " not in selected:
            self.find.setText(selected)
        (self.replace if replace and self.find.text() else self.find).setFocus()
        self.find.selectAll()

    def next(self, back: bool = False) -> bool:
        text = self.find.text()
        if not text:
            return False
        if self.editor.find(text, self._flags(back)):
            return True
        cursor = self.editor.textCursor()  # wrap around
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End if back else QtGui.QTextCursor.MoveOperation.Start)
        self.editor.setTextCursor(cursor)
        return self.editor.find(text, self._flags(back))

    def replace_one(self) -> None:
        cursor = self.editor.textCursor()
        same = cursor.selectedText() == self.find.text() if self.case.isChecked() else cursor.selectedText().lower() == self.find.text().lower()
        if cursor.hasSelection() and same:
            cursor.insertText(self.replace.text())
        self.next()

    def replace_all(self) -> int:
        count = 0
        cursor = self.editor.textCursor()
        cursor.beginEditBlock()
        start = QtGui.QTextCursor(self.editor.document())
        self.editor.setTextCursor(start)
        while self.editor.find(self.find.text(), self._flags()):
            self.editor.textCursor().insertText(self.replace.text())
            count += 1
        cursor.endEditBlock()
        return count


class EditorWindow(QtWidgets.QWidget):
    """A script: editor, find bar and toolbar (Run F5, Run selection F9, Save Ctrl+S).

    ``run(code, first_line, editor)`` is supplied by the main window and sends the code to the Immediate window.
    """

    def __init__(self, run: Callable[[str, int, EditorWindow], None], namespace: Callable[[], dict[str, Any]],
                 path: str | None = None, text: str = "", parent: Any = None) -> None:
        super().__init__(parent)
        self.path = Path(path) if path else None
        self.run_code = run
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QToolBar()
        self.editor = CodeEditor(namespace)
        self.find_bar = FindBar(self.editor)
        from .zoom import make_zoomable, scale_font

        def show_at(factor: float) -> None:
            scale_font(self.editor, factor)
            self.editor.setTabStopDistance(QtGui.QFontMetricsF(self.editor.font()).horizontalAdvance(" ") * 4)

        make_zoomable(self, show_at)
        layout.addWidget(bar)
        layout.addWidget(self.editor, 1)
        layout.addWidget(self.find_bar)
        self.actions_by_name: dict[str, QtGui.QAction] = {}
        for name, label, keys, handler, tip in (
            ("run", "▶ Run", "F5", self.run_all, "run the whole script in the Immediate window (F5)"),
            ("run_selection", "Run selection", "F9", self.run_selection, "run the selected lines, or the current line (F9)"),
            # the main window routes its Save key (Ctrl+S, ⌘S on a Mac) here
            ("save", "Save", None, self.save, f"save the script ({key_text(QtGui.QKeySequence.StandardKey.Save)} while editing)"),
            ("save_as", "Save as…", None, lambda: self.save(as_new=True), "save under a new name"),
            ("find", "Find", QtGui.QKeySequence.StandardKey.Find, lambda: self.find_bar.open(), "find"),
            ("replace", "Replace", QtGui.QKeySequence.StandardKey.Replace, lambda: self.find_bar.open(replace=True), "find and replace"),
            ("comment", "Comment", "Ctrl+/", self.editor.toggle_comment, "comment or uncomment the selected lines"),
        ):
            action = QtGui.QAction(label, self)
            if keys is not None:
                action.setShortcut(QtGui.QKeySequence(keys))
                action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            # the action's own key, in the platform's words (⌘F on a Mac; none where the platform binds no Replace)
            shown = key_text(action.shortcut()) if name in ("find", "replace", "comment") else ""
            action.setToolTip(f"{tip} ({shown})" if shown else tip)
            action.triggered.connect(lambda _=False, h=handler: h())
            self.addAction(action)
            if name not in ("replace", "comment"):
                bar.addAction(action)
            self.actions_by_name[name] = action
        self.editor.context_actions = self._context_actions
        if self.path and not text:
            text = self.path.read_text(encoding="utf-8")
        self.editor.setPlainText(text)
        self.editor.document().setModified(False)
        self.editor.document().modificationChanged.connect(lambda _: self.update_title())

    @property
    def title(self) -> str:
        name = self.path.name if self.path else "untitled.py"
        return f"{name}{' *' if self.editor.document().isModified() else ''} — Code"

    def update_title(self) -> None:
        parent = self.parentWidget()
        if parent is not None:
            parent.setWindowTitle(self.title)

    def _context_actions(self, menu: QtWidgets.QMenu, cursor: QtGui.QTextCursor) -> None:
        """The Code window's part of the right-click menu: run, comment, find; help on the name under the mouse."""
        menu.addSeparator()
        selected = self.editor.textCursor().hasSelection()
        menu.addAction("Run the selection" if selected else "Run this line", self.run_selection)
        menu.addAction("Run the whole script", self.run_all)
        menu.addAction("Comment or uncomment", self.editor.toggle_comment)
        menu.addAction("Find…", lambda: self.find_bar.open())
        found = self.editor.help_at(cursor)
        if found is not None:
            name, doc = found
            menu.addSeparator()
            menu.addAction(f"Help on “{name}”", lambda: self.show_help(name, doc))

    def show_help(self, name: str, doc: str) -> QtWidgets.QMessageBox:
        """A name's documentation, in a box beside the code."""
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle(f"Help: {name}")
        box.setText(f"<b>{name}</b>")
        box.setInformativeText(doc.split("\n\n")[0][:600])
        box.setDetailedText(doc)
        box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Close)
        box.setModal(False)
        box.show()
        self.help_box = box
        return box

    def run_all(self) -> None:
        self.run_code(self.editor.toPlainText(), 1, self)

    def run_selection(self) -> None:
        code, first = self.editor.selection_or_line()
        if code.strip():
            self.run_code(code, first, self)
            if not self.editor.textCursor().hasSelection():  # move on, as in MATLAB and RStudio
                self.editor.moveCursor(QtGui.QTextCursor.MoveOperation.NextBlock)

    def save(self, as_new: bool = False) -> bool:
        path = self.path
        if as_new or path is None:
            name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save script", str(path or "script.py"), "Python (*.py)")
            if not name:
                return False
            path = Path(name)
        path.write_text(self.editor.toPlainText(), encoding="utf-8")
        self.path = path
        self.editor.document().setModified(False)
        self.update_title()
        return True

    def confirm_close(self) -> bool:
        if not self.editor.document().isModified():
            return True
        answer = QtWidgets.QMessageBox.question(self, "Unsaved script", f"Save {self.path.name if self.path else 'the script'} first?",
                                                QtWidgets.QMessageBox.StandardButton.Save | QtWidgets.QMessageBox.StandardButton.Discard
                                                | QtWidgets.QMessageBox.StandardButton.Cancel)
        if answer == QtWidgets.QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QtWidgets.QMessageBox.StandardButton.Discard


def error_line(error: BaseException | None) -> int | None:
    """The line, within the code that was run, where ``error`` arose (None if it was elsewhere)."""
    if error is None:
        return None
    if isinstance(error, SyntaxError) and error.lineno:
        return error.lineno
    import traceback

    lines = [f.lineno for f in traceback.extract_tb(error.__traceback__)
             if f.filename.startswith("<") or "ipykernel_" in f.filename or "ipython-input" in f.filename]
    return lines[-1] if lines else None


def statements(code: str, first: int = 1) -> list[tuple[str, int]]:
    """Top-level statements of a script, each with the editor line it starts on, so a run is
    recorded one step per statement (a loop or function definition is one statement). Code that
    does not parse is one piece, so the error is reported as Python reports it."""
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [(code, first)] if code.strip() else []
    lines = code.splitlines()
    out = []
    for node in tree.body:
        start = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
        text = "\n".join(lines[start - 1 : node.end_lineno])
        out.append((text, first + start - 1))
    return out
