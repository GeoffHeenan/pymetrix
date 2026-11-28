"""A translation panel: the report of a program translated, for any front end. Each registers its own, made with
``translation_panel(PanelSpec(...))`` (the EViews plugin's is View → EViews translation):

- a table, a row per line: its number (and file, for an included program), its status in words and in colour, the
  program's text, the Python and the reason;
- the program and the script side by side; selecting a row scrolls both to that line;
- a filter for each status, with its count;
- Copy (the selected rows, or the whole report, as text) and Copy script;
- Run as steps: the translation's steps run one at a time through ``AppInterface.run``, so each is recorded and can be
  undone; the run stops at the first step that fails, and that row is marked.

The panel uses only ``AppInterface``: the session (``PanelSpec.translations`` finds the translations in it), its
runner, the status bar and the front end's translate dialog. When a new translation is made, the panel shows itself
(its dock). Qt is imported only when a panel is made.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["COLOURS", "PanelSpec", "panel_class", "translation_panel"]

# a status's colour, behind its text (never the colour alone): translucent, so it reads on light and dark themes
COLOURS = {"translated": (46, 160, 67, 55), "translated with a note": (31, 111, 235, 60), "left out": (191, 135, 0, 75),
           "error": (207, 34, 46, 85), "failed": (207, 34, 46, 150)}


@dataclass(frozen=True)
class PanelSpec:
    """What a front end's panel shows: ``language`` (``"EViews"``: the column of the program's text, and the words
    of the tooltips), ``translations(session)`` (the session's translations, ``[(step, Translation)]``, oldest first:
    from its History's translate steps, say), ``dialog`` (the translate dialog's command class, opened by
    Translate…), ``menu`` (where that dialog is, as the help writes it: ``"File → Translate EViews program…"``),
    ``what`` (``"an EViews program"``), ``noun`` (``"program"``) and ``prefix``, the start of its widgets' object
    names (``"eviews"``: ``eviews-report``, ``eviews-source``)."""

    language: str
    translations: Callable[[Any], list[tuple[Any, Any]]]
    dialog: Any
    menu: str
    what: str
    noun: str = "program"
    prefix: str = "translation"

    @property
    def columns(self) -> tuple[str, ...]:
        return ("line", "status", self.language, "Python", "reason")


def translation_panel(spec: PanelSpec) -> Callable[[Any], Any]:
    """The ``create(app)`` of an ``AppPanel`` that shows ``spec``'s translations: ``register("panels",
    AppPanel("EViews translation", translation_panel(spec), area="bottom"))``."""

    def create_panel(app: Any) -> Any:
        """The panel's widget (``AppPanel.create``): ``app`` is pymetrix's ``AppInterface``."""
        return panel_class()(app, spec)

    return create_panel


_CLASS: Any = None


def panel_class() -> Any:
    """The panel's Qt class, made on first use (Qt is not imported with pymetrix); an instance is made with
    ``(app, spec)``."""
    global _CLASS
    if _CLASS is None:
        _CLASS = _panel_class()
    return _CLASS


def _panel_class() -> Any:
    from PySide6 import QtCore, QtGui, QtWidgets

    from .report import STATUSES

    class TranslationPanel(QtWidgets.QWidget):
        def __init__(self, app: Any, spec: PanelSpec) -> None:
            super().__init__()
            self.app = app
            self.spec = spec
            noun, language = spec.noun, spec.language
            self.setObjectName(f"{spec.prefix}-translation-panel")
            self.items: list[tuple[Any, Any]] = []  # (the translate step, its translation)
            self.current: Any = None
            self.failed: tuple[str, int, str] | None = None  # (file, line, message) of the step that failed
            self._known = 0
            self._running = False  # Run as steps under way: the History changes at each step, the report does not
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            top = QtWidgets.QHBoxLayout()
            self.program = QtWidgets.QComboBox()
            self.program.setObjectName(f"{spec.prefix}-{noun}")
            self.program.setToolTip(f"The {noun}s translated in this session ({spec.menu})")
            self.program.currentIndexChanged.connect(self._chosen)
            top.addWidget(self.program, 1)
            self.translate_button = QtWidgets.QPushButton("Translate…")
            self.translate_button.setToolTip(f"Translate {spec.what} ({spec.menu})")
            self.translate_button.clicked.connect(self.translate)
            self.copy_button = QtWidgets.QPushButton("Copy")
            self.copy_button.setToolTip("Copy the selected rows, or the whole report when none is selected, as text")
            self.copy_button.clicked.connect(self.copy)
            self.copy_script_button = QtWidgets.QPushButton("Copy script")
            self.copy_script_button.setToolTip("Copy the readable script")
            self.copy_script_button.clicked.connect(self.copy_script)
            self.run_button = QtWidgets.QPushButton("Run as steps")
            self.run_button.setToolTip(f"Run the {noun}'s work in this session, one recorded step at a time (each can be "
                                       "undone); stops at the first step that fails and marks its row")
            self.run_button.clicked.connect(self.run_steps)
            for b in (self.translate_button, self.copy_button, self.copy_script_button, self.run_button):
                top.addWidget(b)
            layout.addLayout(top)
            filters = QtWidgets.QHBoxLayout()
            self.filters: dict[str, Any] = {}
            for status in STATUSES:
                box = QtWidgets.QCheckBox(status)
                box.setChecked(True)
                box.setToolTip(f"Show the lines {status}")
                box.toggled.connect(self._filter)
                self.filters[status] = box
                filters.addWidget(box)
            filters.addStretch(1)
            layout.addLayout(filters)
            self.message = QtWidgets.QLabel(f"No {noun} translated yet: {spec.menu}.")
            self.message.setWordWrap(True)
            self.message.setObjectName(f"{spec.prefix}-translation-message")
            layout.addWidget(self.message)
            split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            self.table = QtWidgets.QTableWidget(0, len(spec.columns))
            self.table.setObjectName(f"{spec.prefix}-report")
            self.table.setHorizontalHeaderLabels(list(spec.columns))
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.verticalHeader().setVisible(False)
            self.table.horizontalHeader().setStretchLastSection(True)
            self.table.itemSelectionChanged.connect(self._selected)
            split.addWidget(self.table)
            texts = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
            fixed = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
            self.source = QtWidgets.QPlainTextEdit()
            self.script = QtWidgets.QPlainTextEdit()
            for name, edit in ((f"{spec.prefix}-source", self.source), (f"{spec.prefix}-script", self.script)):
                edit.setObjectName(name)
                edit.setReadOnly(True)
                edit.setFont(fixed)
                edit.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
                texts.addWidget(edit)
            self.source.setToolTip(f"The {language} {noun} (an included {noun}'s text when its row is selected)")
            self.script.setToolTip(f"The readable script: each {language} line as a comment above what it became")
            split.addWidget(texts)
            layout.addWidget(split, 1)
            self.shown_file = ""
            self.refresh()
            app.on_history_changed(self.refresh)

        # -- the translations --------------------------------------------------------------------------------------

        def refresh(self) -> None:
            """The session's translations again (after a step, an undo or another project)."""
            if self._running:
                return
            items = self.spec.translations(self.app.session)
            if [id(t) for _, t in items] == [id(t) for _, t in self.items] and self.items:
                return  # the same translations: a step of another kind, nothing to show again
            before = self.program.currentIndex()
            newer = len(items) > self._known
            self.items = items
            self.program.blockSignals(True)
            self.program.clear()
            for cmd, t in items:
                counts = t.report.counts()
                self.program.addItem(f"{t.program or getattr(cmd, 'path', '')} — {sum(counts.values())} lines, {counts['left out']} left out, "
                                     f"{counts['error']} errors")
            self.program.blockSignals(False)
            if items:
                index = len(items) - 1 if newer or before < 0 or before >= len(items) else before
                self.program.setCurrentIndex(index)
                self._show(items[index][1], reset=newer or self.current is not items[index][1])
                if newer and self._known:
                    self._reveal()
            else:
                self._show(None, reset=True)
            self._known = len(items)

        def _reveal(self) -> None:
            """A new translation: show the panel's own dock, if it is hidden."""
            dock = self.parentWidget()
            while dock is not None and not isinstance(dock, QtWidgets.QDockWidget):
                dock = dock.parentWidget()
            if dock is not None:
                dock.setVisible(True)
                dock.raise_()

        def _chosen(self, index: int) -> None:
            if 0 <= index < len(self.items):
                self._show(self.items[index][1], reset=True)

        def _show(self, t: Any, reset: bool) -> None:
            if reset:
                self.failed = None
            self.current = t
            enabled = t is not None
            for b in (self.copy_button, self.copy_script_button, self.run_button):
                b.setEnabled(enabled)
            self.message.setVisible(not enabled)
            if t is None:
                self.table.setRowCount(0)
                self.source.setPlainText("")
                self.script.setPlainText("")
                for status, box in self.filters.items():
                    box.setText(f"{status} (0)")
                return
            counts = t.report.counts()
            for status, box in self.filters.items():
                box.setText(f"{status} ({counts[status]})")
            self.script.setPlainText(t.script)
            self.shown_file = t.program
            self.source.setPlainText(t.sources.get(t.program, ""))
            self.table.setRowCount(len(t.report))
            for i, row in enumerate(t.report):
                where = f"{row.line}" if row.file == t.program else f"{row.file}:{row.line}"
                cells = (where, row.status, row.source.strip(), row.python, row.reason)
                for j, text in enumerate(cells):
                    item = QtWidgets.QTableWidgetItem(text)
                    if j in (2, 3, 4):
                        item.setToolTip(text)
                    self.table.setItem(i, j, item)
                self._colour(i, row.status)
            if self.failed is not None:
                self._mark_failed()
            self._filter()

        def _colour(self, i: int, status: str) -> None:
            item = self.table.item(i, 1)
            item.setBackground(QtGui.QBrush(QtGui.QColor(*COLOURS[status])))

        def _filter(self, *_: Any) -> None:
            if self.current is None:
                return
            shown = {s for s, box in self.filters.items() if box.isChecked()}
            for i, row in enumerate(self.current.report):
                failed = self.failed is not None and (row.file, row.line) == self.failed[:2]
                self.table.setRowHidden(i, row.status not in shown and not failed)

        def row_of(self, line: int, file: str | None = None) -> int:
            """The table's row of a program's line (an included program's, with ``file``), or -1."""
            if self.current is None:
                return -1
            file = self.current.program if file is None else file
            for i, row in enumerate(self.current.report):
                if row.line == line and row.file == file:
                    return i
            return -1

        # -- following the selected row ----------------------------------------------------------------------------

        def _selected(self) -> None:
            rows = sorted({i.row() for i in self.table.selectedItems()})
            if not rows or self.current is None:
                return
            row = self.current.report[rows[0]]
            if row.file != self.shown_file:
                self.shown_file = row.file
                self.source.setPlainText(self.current.sources.get(row.file, ""))
            self._go(self.source, row.line)
            at = self.current.script_line(row.line, row.file)
            if at is not None:
                self._go(self.script, at)

        @staticmethod
        def _go(edit: Any, line: int) -> None:
            """Put line ``line`` (from 1) of ``edit`` at the top of its view, selected."""
            block = edit.document().findBlockByLineNumber(max(0, line - 1))
            cursor = QtGui.QTextCursor(block)
            cursor.select(QtGui.QTextCursor.SelectionType.LineUnderCursor)
            edit.setTextCursor(cursor)
            bar = edit.verticalScrollBar()
            bar.setValue(min(bar.maximum(), max(0, line - 1)))

        def source_line(self) -> int:
            """The line (from 1) of the program at the top of the program's view, selected."""
            return self.source.textCursor().blockNumber() + 1

        def script_at(self) -> int:
            return self.script.textCursor().blockNumber() + 1

        # -- actions -----------------------------------------------------------------------------------------------

        def translate(self) -> None:
            self.app.open_dialog(self.spec.dialog, None)

        def selected_rows(self) -> list[Any]:
            if self.current is None:
                return []
            rows = sorted({i.row() for i in self.table.selectedItems()})
            return [self.current.report[i] for i in rows]

        def copy(self) -> str:
            """The selected rows, or the whole report, onto the clipboard as text."""
            if self.current is None:
                return ""
            rows = self.selected_rows()
            text = self.current.report.to_text(rows or None)
            QtWidgets.QApplication.clipboard().setText(text)
            self.app.status(f"Copied {len(rows) if rows else 'every'} row{'s' if len(rows) != 1 else ''} of the translation")
            return text

        def copy_script(self) -> str:
            if self.current is None:
                return ""
            QtWidgets.QApplication.clipboard().setText(self.current.script)
            self.app.status("Copied the translated script")
            return self.current.script

        def run_steps(self) -> bool:
            """The translation's steps, one at a time through the app's runner (each recorded and undoable); at the
            first that fails, its row is marked and selected. True when every step ran."""
            t = self.current
            if t is None:
                return False
            self.failed = None
            self._show(t, reset=False)
            self._running = True
            try:
                t.run(self.app.session, run=self.app.run)
            except Exception as exc:  # noqa: BLE001 - the failing step is shown on its row, not raised into the app
                self._running = False
                line, file = getattr(exc, "line", 0), getattr(exc, "file", "") or t.program
                self.failed = (file, line, str(exc))
                self._mark_failed()
                self.app.status(f"Run as steps stopped: {exc}")
                return False
            self._running = False
            self.app.status(f"Ran {t.program or 'the ' + self.spec.noun} as steps: see History")
            return True

        def _mark_failed(self) -> None:
            file, line, message = self.failed
            i = self.row_of(line, file)
            if i < 0:
                return
            self.table.setRowHidden(i, False)
            item = self.table.item(i, 1)
            item.setText("failed")
            item.setToolTip(message)
            item.setBackground(QtGui.QBrush(QtGui.QColor(*COLOURS["failed"])))
            self.table.item(i, 4).setText(f"failed when run: {message}")
            self.table.selectRow(i)
            self.table.scrollToItem(item)

    return TranslationPanel

