"""The spreadsheet editor: a dataset as a table you can edit.

Periods are rows and series are columns (for panels, pick an entity or show
all rows). The edit line above the table shows the selected cell's value in full,
to change it there: ✓ (or Enter) accepts it, ✗ (or Esc) puts the value back.
Typed and pasted values are pending until Apply, which records them
as one step (``data = data.edit({...})``); until then Undo and Redo work cell by
cell, and derived series already show what the edits will do. Renaming,
removing, freezing and describing series are steps too. Shading shows pending
edits, cells edited earlier, derived series (read-only) and flagged
observations.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable

import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from ..commands import Command, DropSeries, EditData, FreezeSeries, GenerateSeries, RenameSeries, SeriesInfo
from ..data import Dataset
from ..metadata import Adjustment, Measure, Prices, Status
from . import theme
from .keys import key_text
from .messages import titled
from .metrics import set_margins, set_spacing

ALL = "(all rows)"
_NA = {"", "na", "n/a", "nan", ".", "#n/a", "-"}


def parse_cell(text: str) -> float | None:
    """A typed or pasted value: a number (thousands separators allowed), or missing."""
    t = text.strip()
    if t.lower() in _NA:
        return None
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d*)?", t):
        t = t.replace(",", "")
    return float(t)


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float):
        from ..output import _fmt as table_format

        return table_format(value) if abs(value) < 1e6 else f"{value:,.0f}"  # as in the result tables
    return str(value)


def _dark(widget: QtWidgets.QWidget) -> bool:
    return widget.palette().color(QtGui.QPalette.ColorRole.Base).lightness() < 128


class DataModel(QtCore.QAbstractTableModel):
    """Rows of the dataset (all, or one entity's) by series, over the original plus pending edits."""

    def __init__(self, data: Dataset, dark: bool = False) -> None:
        super().__init__()
        self.dark = dark
        self.entity: Any = ALL
        self.scope: set[str] | None = None  # the series shown (a working set); None: all
        self.load(data)

    # -- state ----------------------------------------------------------------------

    def load(self, data: Dataset) -> None:
        self.beginResetModel()
        self.original = data
        self.pending: dict[tuple[str, Any], float | None] = {}
        self.working = data
        self._select_rows()
        self.endResetModel()

    def _select_rows(self) -> None:
        frame = self.working.frame
        entities = self.working.entity_axes
        if self.entity == ALL or not entities:
            self.rows = list(range(len(frame)))
        else:
            level = frame.index.get_level_values(entities[0].name)
            self.rows = [i for i, v in enumerate(level) if str(v) == str(self.entity)]
        self.columns = self._shown(self.working.names)

    def _shown(self, names: Any) -> list[str]:
        return [n for n in names if self.scope is None or n in self.scope]

    def set_scope(self, names: Any) -> None:
        """Show only these series (None: all); pending edits to the others are kept."""
        self.beginResetModel()
        self.scope = None if names is None else set(names)
        self._select_rows()
        self.endResetModel()

    def set_entity(self, entity: Any) -> None:
        self.beginResetModel()
        self.entity = entity
        self._select_rows()
        self.endResetModel()

    def entities(self) -> list[str]:
        axes = self.original.entity_axes
        if not axes:
            return []
        return [ALL] + [str(v) for v in dict.fromkeys(self.original.frame.index.get_level_values(axes[0].name))]

    def label(self, row: int) -> Any:
        return self.working.frame.index[self.rows[row]]

    def is_derived(self, column: int) -> bool:
        return self.working.variables[self.columns[column]].recipe is not None

    def apply_pending(self, pending: dict[tuple[str, Any], float | None]) -> None:
        """Show the dataset with ``pending`` edits (raises ValueError if derived series cannot follow)."""
        working = self.original
        if pending:
            cells: dict[str, dict[Any, Any]] = {}
            for (name, label), value in pending.items():
                cells.setdefault(name, {})[_code_key(label)] = value
            working = self.original.edit(cells)
        self.pending = dict(pending)
        reshaped = self._shown(working.names) != self.columns or len(working.frame) != len(self.working.frame)
        if reshaped:  # series or rows were added or removed elsewhere
            self.beginResetModel()
            self.working = working
            self._select_rows()
            self.endResetModel()
            return
        self.working = working
        self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1))

    def edits_json(self) -> str:
        return json.dumps([[name, _parts(label), value] for (name, label), value in self.pending.items()])

    # -- Qt model ------------------------------------------------------------------------

    def rowCount(self, parent: Any = QtCore.QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent: Any = QtCore.QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.columns)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:  # noqa: N802
        if orientation == Qt.Orientation.Horizontal and section < len(self.columns):
            name = self.columns[section]
            var = self.working.variables[name]
            if role == Qt.ItemDataRole.DisplayRole:
                return name
            if role == Qt.ItemDataRole.ToolTipRole:
                parts = [var.label or name]
                if var.units:
                    parts.append(f"units: {var.units}")
                if var.recipe is not None:
                    parts.append(f"derived: {var.derived_from} (read-only; freeze it to edit)")
                if var.vintage is not None:
                    parts.append(f"released {var.vintage.released}")
                return "\n".join(parts)
            if role == Qt.ItemDataRole.FontRole and var.recipe is not None:
                font = QtGui.QFont()
                font.setItalic(True)
                return font
        if orientation == Qt.Orientation.Vertical and role == Qt.ItemDataRole.DisplayRole and section < len(self.rows):
            label = self.label(section)
            if self.entity != ALL and isinstance(label, tuple):  # one entity: show the period only
                label = label[-1]
            return " ".join(map(str, label)) if isinstance(label, tuple) else str(label)
        return None

    def data(self, index: QtCore.QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        name, row = self.columns[index.column()], self.rows[index.row()]
        label = self.working.frame.index[row]
        value = self.working.frame.iat[row, self.working.frame.columns.get_loc(name)]
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return _fmt(value) if role == Qt.ItemDataRole.DisplayRole else ("" if pd.isna(value) else repr(float(value)) if isinstance(value, float) else str(value))
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        status = self.working.flag(name).iat[row] if self.working.flags is not None and name in self.working.flags.columns else ""
        pending = (name, label) in self.pending
        if role == Qt.ItemDataRole.BackgroundRole:
            d = self.dark
            if pending:
                return QtGui.QColor(theme.SHEET_CELLS["edited"][0 if d else 1])
            if self.is_derived(index.column()):
                return QtGui.QColor(theme.SHEET_CELLS["outside sample"][0 if d else 1])
            if status == Status.EDITED.value:
                return QtGui.QColor(theme.SHEET_CELLS["formula"][0 if d else 1])
            if status:
                return QtGui.QColor(theme.SHEET_CELLS["selected series"][0 if d else 1])
            return None
        if role == Qt.ItemDataRole.FontRole and self.is_derived(index.column()):
            font = QtGui.QFont()
            font.setItalic(True)
            return font
        if role == Qt.ItemDataRole.ToolTipRole:
            notes = []
            if pending:
                notes.append("pending edit: Apply to record it")
            if status:
                notes.append(f"status: {status}")
            if self.is_derived(index.column()):
                notes.append(f"derived: {self.working.variables[name].derived_from}")
            return "\n".join(notes) or None
        return None

    def flags(self, index: QtCore.QModelIndex) -> Qt.ItemFlag:
        base = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        if index.isValid() and not self.is_derived(index.column()):
            base |= Qt.ItemFlag.ItemIsEditable
        return base


def _parts(label: Any) -> list[str]:
    return [str(p) for p in (label if isinstance(label, tuple) else (label,))]


def _code_key(label: Any) -> Any:
    parts = _parts(label)
    return parts[0] if len(parts) == 1 else tuple(parts)


class _CellsCommand(QtGui.QUndoCommand):
    """Typing or pasting values (undoable until Apply)."""

    def __init__(self, sheet: SheetWindow, before: dict[tuple[str, Any], Any], after: dict[tuple[str, Any], Any], text: str) -> None:
        super().__init__(text)
        self.sheet, self.before, self.after = sheet, before, after

    def redo(self) -> None:
        self.sheet.model.apply_pending(self.after)
        self.sheet.update_status()

    def undo(self) -> None:
        self.sheet.model.apply_pending(self.before)
        self.sheet.update_status()


class _Delegate(QtWidgets.QStyledItemDelegate):
    def setModelData(self, editor: QtWidgets.QWidget, model: Any, index: QtCore.QModelIndex) -> None:  # noqa: N802
        self.parent().set_cells({(index.row(), index.column()): editor.text()}, "Edit cell")


class _CellName(QtWidgets.QLabel):
    """The edit line's cell name: 140 pixels wide or more when there is room, so the value beside it does not jump
    as the selection moves; narrower (cut short) in a narrow window rather than over the ✓ and ✗ buttons."""

    WIDTH = 140

    def sizeHint(self) -> QtCore.QSize:
        hint = super().sizeHint()
        return QtCore.QSize(max(hint.width(), self.WIDTH), hint.height())

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(min(super().minimumSizeHint().width(), 40), super().minimumSizeHint().height())


class _EditLine(QtWidgets.QWidget):
    """The edit line above the table, as in a spreadsheet: the selected cell's name and its value in full precision,
    to change there. ✓ or Enter accepts the new value (a pending edit, as typing in the cell makes, so Undo, Revert
    and Apply work as ever) and moves down a row on Enter; ✗ or Esc puts the cell's value back. A derived series'
    cell is shown but not editable."""

    def __init__(self, sheet: SheetWindow) -> None:
        super().__init__()
        self.sheet = sheet
        row = QtWidgets.QHBoxLayout(self)
        set_margins(row, "tight", "hairline")  # a compact bar, as the code editor's find bar
        set_spacing(row, "tight")
        self.where = _CellName()  # the cell: series and period
        self.where.setToolTip("the selected cell: its series and period")
        self.accept_button = QtWidgets.QToolButton()
        self.accept_button.setText("✓")
        self.accept_button.setAccessibleName("Accept value")  # screen readers say this, not the symbol
        self.accept_button.setToolTip("accept the new value (Enter)")
        self.accept_button.setStyleSheet(f"QToolButton:enabled {{ {theme.css('success')} font-weight: bold; }}")
        self.cancel_button = QtWidgets.QToolButton()
        self.cancel_button.setText("✗")
        self.cancel_button.setAccessibleName("Put value back")
        self.cancel_button.setToolTip("put the cell's value back (Esc)")
        self.cancel_button.setStyleSheet(f"QToolButton:enabled {{ {theme.css('error')} font-weight: bold; }}")
        self.line = QtWidgets.QLineEdit()
        self.line.setAccessibleName("Cell value")
        self.line.setToolTip("the selected cell's value, in full: type a new one (a number, or empty for missing)")
        for widget in (self.where, self.accept_button, self.cancel_button):
            row.addWidget(widget)
        row.addWidget(self.line, 1)
        self.accept_button.clicked.connect(lambda: self.accept(move=False))
        self.cancel_button.clicked.connect(self.cancel)
        self.line.returnPressed.connect(lambda: self.accept(move=True))
        self.line.textEdited.connect(lambda _text: self._changed())
        self.line.installEventFilter(self)
        self.shown = ""  # the cell's value as last shown
        self.cell: tuple[int, int] | None = None
        self.show_cell(QtCore.QModelIndex())

    def eventFilter(self, watched: Any, event: Any) -> bool:  # noqa: N802
        if watched is self.line and event.type() == QtCore.QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
            self.cancel()
            return True
        return super().eventFilter(watched, event)

    def editing(self) -> bool:
        """Whether the line holds a value not yet accepted."""
        return self.cell is not None and self.line.text() != self.shown

    def _changed(self) -> None:
        dirty = self.editing()
        self.accept_button.setEnabled(dirty)
        self.cancel_button.setEnabled(dirty)
        self.line.setStyleSheet("")

    def show_cell(self, index: QtCore.QModelIndex) -> None:
        """Fill the line with the cell's value (full precision); empty and disabled with no cell selected."""
        model = self.sheet.model
        if not index.isValid():
            self.cell, self.shown = None, ""
            self.where.setText("")
            self.line.setText("")
            self.line.setEnabled(False)
            self._changed()
            return
        self.cell = (index.row(), index.column())
        label = model.headerData(index.row(), Qt.Orientation.Vertical)
        self.where.setText(f"{model.columns[index.column()]} · {label}")
        self.shown = model.data(index, Qt.ItemDataRole.EditRole) or ""
        self.line.setText(self.shown)
        derived = model.is_derived(index.column())
        self.line.setEnabled(True)
        self.line.setReadOnly(derived)
        self.line.setPlaceholderText("")
        if derived:
            self.line.setToolTip(f"derived: {model.working.variables[model.columns[index.column()]].derived_from} (freeze it to edit)")
        else:
            self.line.setToolTip("the selected cell's value, in full: type a new one (a number, or empty for missing)")
        self._changed()

    def move_to(self, index: QtCore.QModelIndex) -> None:
        """Another cell was selected: a value typed for the last one is accepted first (as in a spreadsheet)."""
        if self.editing() and self.cell != (index.row(), index.column()):
            self.accept(move=False)  # a value that is not a number is dropped (the cell keeps its own)
        self.show_cell(index)

    def refresh(self) -> None:
        """After the data change (an edit, Undo, a paste): show the cell's value again, unless one is being typed."""
        if not self.editing():
            self.show_cell(self.sheet.table.currentIndex())

    def accept(self, move: bool = False) -> bool:
        """Put the typed value in the cell as a pending edit; on Enter, move down a row. False if it is not a number."""
        if self.cell is None:
            return False
        if not self.editing():
            if move:
                self._move_down()
            return True
        row, col = self.cell
        problems = self.sheet.set_cells({(row, col): self.line.text()}, "Edit cell")
        if problems:
            self.line.setStyleSheet(f"QLineEdit {{ border: 1px solid {theme.name('error')}; }}")  # left for correcting
            self.line.setToolTip("; ".join(problems))
            return False
        self.shown = self.line.text()
        self.show_cell(self.sheet.model.index(row, col))
        if move:
            self._move_down()
        return True

    def cancel(self) -> None:
        """Put the cell's value back in the line."""
        self.show_cell(self.sheet.table.currentIndex())

    def _move_down(self) -> None:
        current = self.sheet.table.currentIndex()
        if current.isValid() and current.row() + 1 < self.sheet.model.rowCount():
            self.sheet.table.setCurrentIndex(self.sheet.model.index(current.row() + 1, current.column()))
            self.line.setFocus()
            self.line.selectAll()


class SheetWindow(QtWidgets.QWidget):
    """``run(command) -> bool`` records a step in the session (the main window supplies it);
    ``dataset()`` returns the current dataset object from the session."""

    def __init__(self, name: str, dataset: Callable[[], Dataset], run: Callable[[Command], bool], parent: Any = None,
                 working_set: Callable[[], Any] | None = None, series_actions: dict[str, Callable[[list[str]], Any]] | None = None) -> None:
        super().__init__(parent)
        self.name, self.dataset, self.run = name, dataset, run
        self.series_actions = series_actions or {}  # plot, describe, generate, select: the app's dialogs for the series
        self.working_set = working_set or (lambda: None)  # (series shown, how it was chosen), or None
        self.picked: list[str] | None = None  # series picked in the series browser, shown for now instead
        self.model = DataModel(dataset(), _dark(self))
        self._applying = False
        self.undo = QtGui.QUndoStack(self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QToolBar()
        layout.addWidget(bar)
        self.banner = QtWidgets.QLabel()
        self.banner.setWordWrap(True)
        self.banner.setVisible(False)
        self.banner.setStyleSheet(f"padding: 4px; background: {theme.SHEET_BANNER[0]}; color: {theme.SHEET_BANNER[1]};")
        layout.addWidget(self.banner)
        self.edit_line = _EditLine(self)  # placed under the toolbar, once the table exists
        layout.insertWidget(1, self.edit_line)
        split = QtWidgets.QSplitter()
        layout.addWidget(split, 1)
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.model)
        self.table.setItemDelegate(_Delegate(self))
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.horizontalHeader().setDefaultSectionSize(90)
        self.table.verticalHeader().setDefaultSectionSize(22)
        from .zoom import make_zoomable, scale_font

        def show_at(factor: float) -> None:  # text, column widths and row heights together
            scale_font(self, factor)
            self.table.horizontalHeader().setDefaultSectionSize(round(90 * factor))
            self.table.verticalHeader().setDefaultSectionSize(round(22 * factor))

        make_zoomable(self, show_at)
        split.addWidget(self.table)
        self.info = _InfoPanel(self)
        split.addWidget(self.info)
        split.setSizes([700, 260])
        self.info.setVisible(False)
        self.status = QtWidgets.QLabel()
        layout.addWidget(self.status)

        def act(label: str, handler: Callable[[], Any], keys: Any = None, tip: str = "") -> QtGui.QAction:
            action = QtGui.QAction(label, self)
            if keys is not None:
                action.setShortcut(QtGui.QKeySequence(keys))
                action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            action.setToolTip(tip or label)
            action.triggered.connect(lambda _=False: handler())
            self.addAction(action)
            bar.addAction(action)
            return action

        self.apply_action = act("Apply", self.apply, "Ctrl+Return", f"record the pending edits as one step ({key_text('Ctrl+Return')})")
        act("Revert", self.revert, None, "discard the pending edits")
        bar.addSeparator()
        undo, redo = self.undo.createUndoAction(self, "Undo"), self.undo.createRedoAction(self, "Redo")
        for action, keys in ((undo, QtGui.QKeySequence.StandardKey.Undo), (redo, QtGui.QKeySequence.StandardKey.Redo)):
            action.setShortcut(QtGui.QKeySequence(keys))
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            self.addAction(action)
            bar.addAction(action)
        act("Copy", self.copy, QtGui.QKeySequence.StandardKey.Copy, "copy the selected cells (tab-separated, for Excel)")
        act("Paste", self.paste, QtGui.QKeySequence.StandardKey.Paste, "paste values from Excel or other tables at the current cell")
        bar.addSeparator()
        act("New series…", self.new_series, None, "add a series from a formula")
        act("Rename…", self.rename, None, "rename the selected series")
        act("Remove", self.remove, None, "remove the selected series")
        act("Freeze", self.freeze, None, "keep a derived series' values but drop its formula, so it can be edited")
        info = act("Series info", lambda: self.info.setVisible(not self.info.isVisible()), None, "show or edit the selected series' description")
        info.setCheckable(True)
        bar.addSeparator()
        self.scope_action = QtGui.QAction("Working set only", self)
        self.scope_action.setCheckable(True)
        self.scope_action.setChecked(True)
        self.scope_action.setToolTip("show only the series of the dataset's working set (right-click the dataset in the workspace: Working set…)")
        self.scope_action.toggled.connect(lambda _on: self.apply_scope())
        bar.addAction(self.scope_action)
        self.entity_box: QtWidgets.QComboBox | None = None
        if self.model.entities():
            bar.addSeparator()
            bar.addWidget(QtWidgets.QLabel(" Entity: "))
            self.entity_box = QtWidgets.QComboBox()
            self.entity_box.addItems(self.model.entities())
            self.entity_box.currentTextChanged.connect(self.model.set_entity)
            bar.addWidget(self.entity_box)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(
            lambda pos: self.cell_menu(self.table.indexAt(pos)).popup(self.table.viewport().mapToGlobal(pos)))
        header = self.table.horizontalHeader()
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(
            lambda pos: self.column_menu(header.logicalIndexAt(pos)).popup(header.mapToGlobal(pos)))
        self.table.selectionModel().currentColumnChanged.connect(lambda cur, _prev: self.info.show_series(self.current_series()))
        self.table.selectionModel().currentChanged.connect(lambda cur, _prev: self.edit_line.move_to(cur))
        self.model.dataChanged.connect(lambda *_: self.edit_line.refresh())
        self.model.modelReset.connect(self._after_reset)
        self.apply_scope()
        self.update_status()

    def show_picked(self, names: list[str]) -> None:
        """Show only these series for now (picked in the series browser); the switch in the toolbar shows all."""
        self.picked = list(names)
        self.scope_action.setChecked(True)
        self.apply_scope()

    def _scope(self) -> Any:
        if self.picked:
            return self.picked, f"{len(self.picked)} picked"
        return self.working_set()

    def apply_scope(self) -> None:
        """Show the series picked, or the working set's (if the dataset has one), while the switch is on; else all."""
        found = self._scope()
        self.scope_action.setText("Picked series only" if self.picked else "Working set only")
        self.scope_action.setVisible(found is not None)
        names = found[0] if found is not None and self.scope_action.isChecked() else None
        if (None if names is None else set(names)) != self.model.scope:
            self.model.set_scope(names)
        self.update_status()

    def _after_reset(self) -> None:
        """The table was rebuilt (another entity, series added or removed): the edit line follows the current cell."""
        self.edit_line.show_cell(self.table.currentIndex())

    # -- state -------------------------------------------------------------------------

    def current_series(self) -> str | None:
        index = self.table.currentIndex()
        return self.model.columns[index.column()] if index.isValid() else None

    def selected_series(self) -> list[str]:
        cols = sorted({i.column() for i in self.table.selectionModel().selectedIndexes()} or ({self.table.currentIndex().column()} if self.table.currentIndex().isValid() else set()))
        return [self.model.columns[c] for c in cols]

    def update_status(self) -> None:
        n = len(self.model.pending)
        self.apply_action.setEnabled(bool(n))
        found = self._scope() if hasattr(self, "scope_action") else None
        total = len(self.model.working.names)
        scope = ((f" of {total:,} ({found[1]})" if self.picked else f" of {total:,} (working set: {found[1]})")
                 if found is not None and self.model.scope is not None else "")
        self.status.setText(f"  {n} pending edit{'s' if n != 1 else ''}: Apply records {'them' if n != 1 else 'it'} as one step" if n
                            else f"  {len(self.model.rows)} {self._row_word()} × {len(self.model.columns):,} series{scope}")
        parent = self.parentWidget()
        if parent is not None:
            parent.setWindowTitle(f"{self.name}{' *' if n else ''} — Spreadsheet")

    def _row_word(self) -> str:
        """What a row is: a period of a time series (or of one entity's), else a row."""
        data = self.model.working
        one_series = data.time_axis is not None and (not data.entity_axes or self.model.entity != ALL)
        return "periods" if one_series else "rows"

    def has_pending(self) -> bool:
        return bool(self.model.pending)

    def sync(self) -> bool:
        """Follow the session's dataset after other steps; False if it no longer exists."""
        try:
            current = self.dataset()
        except KeyError:
            return False
        if current is self.model.original:
            return True
        if self.model.pending and not self._applying:
            self.banner.setText(f"{self.name} has changed since you started editing. Apply to put your edits on the new version, or Revert to discard them.")
            self.banner.setVisible(True)
            self.model.original = current
            try:
                self.model.apply_pending(self.model.pending)
            except (KeyError, ValueError) as exc:
                self.banner.setText(f"{self.name} has changed and your edits no longer fit it ({exc}). Revert to continue.")
            return True
        self.model.load(current)
        self.undo.clear()
        self.banner.setVisible(False)
        self.apply_scope()
        self.info.show_series(self.current_series())
        return True

    # -- editing --------------------------------------------------------------------------

    def set_cells(self, typed: dict[tuple[int, int], str], label: str) -> list[str]:
        """Set cells (row, column) -> text as one undoable change; returns problems (skipped cells)."""
        before = dict(self.model.pending)
        after = dict(before)
        problems = []
        for (row, col), text in typed.items():
            if row >= self.model.rowCount() or col >= self.model.columnCount():
                continue
            name = self.model.columns[col]
            if self.model.is_derived(col):
                problems.append(f"{name} is derived; freeze it to edit")
                continue
            try:
                value = parse_cell(text)
            except ValueError:
                problems.append(f"{text!r} is not a number")
                continue
            label_ = self.model.label(row)
            original = self.model.original.frame.loc[label_, name]
            if (value is None and pd.isna(original)) or (value is not None and not pd.isna(original) and value == float(original)):
                after.pop((name, label_), None)  # back to the original value
            else:
                after[(name, label_)] = value
        if after != before:
            try:
                self.model.apply_pending(after)
            except (KeyError, ValueError) as exc:
                self.model.apply_pending(before)
                QtWidgets.QMessageBox.warning(self, *titled("Edit not possible", str(exc)))
                return problems + [str(exc)]
            self.model.apply_pending(before)
            self.undo.push(_CellsCommand(self, before, after, label))
        if problems:
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), "Skipped: " + "; ".join(dict.fromkeys(problems)), self)
        return problems

    def fill_down(self) -> list[str]:
        """Each selected column's first selected value, copied into the cells selected below it: one undoable change."""
        indexes = self.table.selectionModel().selectedIndexes()
        edit = Qt.ItemDataRole.EditRole
        typed: dict[tuple[int, int], str] = {}
        for col in sorted({i.column() for i in indexes}):
            rows = sorted(i.row() for i in indexes if i.column() == col)
            first = self.model.data(self.model.index(rows[0], col), edit)
            typed.update({(r, col): str(first) for r in rows[1:]})
        return self.set_cells(typed, "Fill down") if typed else []

    def clear_cells(self) -> list[str]:
        """The selected cells made missing: one undoable change."""
        typed = {(i.row(), i.column()): "" for i in self.table.selectionModel().selectedIndexes()}
        return self.set_cells(typed, "Clear") if typed else []

    # -- right-click menus ------------------------------------------------------------

    def _series_section(self, menu: QtWidgets.QMenu, names: list[str]) -> None:
        """What the menus offer for series: the app's dialogs (plot, describe, generate, select) and the sheet's own."""
        if not names:
            return
        which = f"“{names[0]}”" if len(names) == 1 else f"{len(names)} series"
        for key, words in (("plot", f"Plot {which}…"), ("describe", f"Describe {which}…"),
                           ("generate", "New series from " + ("it" if len(names) == 1 else "each") + "…"),
                           ("select", f"Keep {which} as a new dataset…")):
            if key in self.series_actions:
                menu.addAction(words, lambda k=key: self.series_actions[k](list(names)))
        menu.addSeparator()
        if len(names) == 1:
            menu.addAction("Rename…", self.rename)
        menu.addAction("Remove", self.remove)
        if any(self.model.is_derived(self.model.columns.index(n)) for n in names if n in self.model.columns):
            menu.addAction("Freeze (keep the values, drop the formula)", self.freeze)
        menu.addAction("Series info", lambda: self.info.setVisible(True))

    def cell_menu(self, index: QtCore.QModelIndex) -> QtWidgets.QMenu:
        """The menu for a right-click on the cells: copy, paste, fill down, clear, then the selected series'."""
        if index.isValid() and not self.table.selectionModel().isSelected(index):
            self.table.setCurrentIndex(index)  # a right-click on a cell outside the selection picks it, as Excel does
        menu = QtWidgets.QMenu(self)
        menu.addAction("Copy", self.copy)
        menu.addAction("Paste", self.paste)
        many = len({i.row() for i in self.table.selectionModel().selectedIndexes()}) > 1
        fill = menu.addAction("Fill down", self.fill_down)
        fill.setEnabled(many)
        fill.setToolTip("copy each column's first selected value into the cells selected below it")
        menu.addAction("Clear (make missing)", self.clear_cells)
        menu.addSeparator()
        self._series_section(menu, self.selected_series())
        menu.setToolTipsVisible(True)
        return menu

    def column_menu(self, column: int) -> QtWidgets.QMenu:
        """The menu for a right-click on a series' heading: the series (or the columns selected with it)."""
        if column >= 0 and column not in {i.column() for i in self.table.selectionModel().selectedIndexes()}:
            self.table.selectColumn(column)
        menu = QtWidgets.QMenu(self)
        self._series_section(menu, self.selected_series())
        menu.addSeparator()
        menu.addAction("Copy the column", lambda: (self.table.selectColumn(column), self.copy()))
        return menu

    def copy(self) -> str:
        indexes = self.table.selectionModel().selectedIndexes()
        if not indexes:
            return ""
        rows = sorted({i.row() for i in indexes})
        cols = sorted({i.column() for i in indexes})
        chosen = {(i.row(), i.column()) for i in indexes}
        edit = Qt.ItemDataRole.EditRole  # the values at full precision, not the four decimals shown
        lines = ["\t".join(self.model.data(self.model.index(r, c), edit) if (r, c) in chosen else "" for c in cols) for r in rows]
        text = "\n".join(lines)
        QtWidgets.QApplication.clipboard().setText(text)
        return text

    def paste(self, text: str | None = None) -> list[str]:
        text = QtWidgets.QApplication.clipboard().text() if text is None else text
        start = self.table.currentIndex()
        if not text or not start.isValid():
            return []
        lines = text.replace("\r\n", "\n").rstrip("\n").split("\n")
        typed = {(start.row() + r, start.column() + c): cell for r, line in enumerate(lines) for c, cell in enumerate(line.split("\t"))}
        return self.set_cells(typed, "Paste")

    def apply(self) -> bool:
        if self.edit_line.editing() and not self.edit_line.accept():  # a value typed in the edit line goes in too
            return False
        if not self.model.pending:
            return True
        self._applying = True  # the session's refresh then loads the edited dataset
        try:
            ok = self.run(EditData(self.name, self.model.edits_json()))
        finally:
            self._applying = False
        if ok:
            self.banner.setVisible(False)
            self.sync()
        return ok

    def revert(self) -> None:
        self.model.load(self.dataset())
        self.undo.clear()
        self.banner.setVisible(False)
        self.update_status()

    def _structural(self, command: Command) -> bool:
        """Rename, remove and the like: apply pending edits first, so the steps stay in order."""
        if self.model.pending and not self.apply():
            return False
        return self.run(command)

    def new_series(self) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("New series")
        form = QtWidgets.QFormLayout(dialog)
        from .metrics import set_margins

        set_margins(form, "dialog_margin")
        name, formula = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        formula.setPlaceholderText("e.g. log(y), c / y * 100, y - y[-1]")
        keep = QtWidgets.QCheckBox("Keep updated (a derived series that follows its inputs)")
        form.addRow("Name", name)
        form.addRow("Formula", formula)
        form.addRow(keep)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() and name.text().strip() and formula.text().strip():
            self._structural(GenerateSeries(name.text().strip(), formula.text().strip(), self.name, keep.isChecked()))

    def rename(self) -> None:
        series = self.current_series()
        if series is None:
            return
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename series", f"New name for {series}:", text=series)
        if ok and new.strip() and new.strip() != series:
            self._structural(RenameSeries(self.name, series, new.strip()))

    def remove(self) -> None:
        names = self.selected_series()
        if names and QtWidgets.QMessageBox.question(self, "Remove series", f"Remove {', '.join(names)} from {self.name}?") == QtWidgets.QMessageBox.StandardButton.Yes:
            self._structural(DropSeries(self.name, ", ".join(names)))

    def freeze(self) -> None:
        names = [n for n in self.selected_series() if self.model.working.variables[n].recipe is not None]
        if names:
            self._structural(FreezeSeries(self.name, ", ".join(names)))

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:  # noqa: N802
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and not self.table.state() == QtWidgets.QAbstractItemView.State.EditingState:
            cells = {(i.row(), i.column()): "" for i in self.table.selectionModel().selectedIndexes()}
            if cells:
                self.set_cells(cells, "Clear cells")
                return
        super().keyPressEvent(e)

    def confirm_close(self) -> bool:
        if not self.model.pending:
            return True
        answer = QtWidgets.QMessageBox.question(self, "Pending edits", f"Apply the pending edits to {self.name}?",
                                                QtWidgets.QMessageBox.StandardButton.Apply | QtWidgets.QMessageBox.StandardButton.Discard
                                                | QtWidgets.QMessageBox.StandardButton.Cancel)
        if answer == QtWidgets.QMessageBox.StandardButton.Apply:
            return self.apply()
        return answer == QtWidgets.QMessageBox.StandardButton.Discard


def _add_described(box: Any, name: str, codes: list[str]) -> None:
    """A series' coded description as words in the list ("not seasonally adjusted"), the code kept as the item's data."""
    from ..wording import choice_label, python_hint, sentence

    for code in codes:
        box.addItem(sentence(choice_label("Variable", name, code) or code), code)
        box.setItemData(box.count() - 1, f"{python_hint(name)}{code!r}", Qt.ItemDataRole.ToolTipRole)


class _InfoPanel(QtWidgets.QWidget):
    """The selected series' description; Save records it as a step."""

    TEXT = ("label", "units", "description", "source", "symbol")
    CHOICES = {"measure": [""] + [m.value for m in Measure], "adjustment": [a.value for a in Adjustment], "prices": [p.value for p in Prices]}

    def __init__(self, sheet: SheetWindow) -> None:
        super().__init__()
        self.sheet = sheet
        form = QtWidgets.QFormLayout(self)
        self.title = QtWidgets.QLabel()
        form.addRow(self.title)
        self.editors: dict[str, Any] = {}
        for name in self.TEXT:
            self.editors[name] = QtWidgets.QLineEdit()
            form.addRow(name.capitalize(), self.editors[name])
        for name, values in self.CHOICES.items():
            box = QtWidgets.QComboBox()
            _add_described(box, name, values)
            self.editors[name] = box
            form.addRow(name.capitalize(), box)
        self.scale = QtWidgets.QSpinBox()
        self.scale.setRange(-12, 12)
        self.scale.setToolTip("power of ten the values are in, e.g. 9 for billions")
        form.addRow("Scale", self.scale)
        self.recipe = QtWidgets.QLabel()
        self.recipe.setWordWrap(True)
        self.vintage = QtWidgets.QLabel()
        form.addRow("Derived", self.recipe)
        form.addRow("Released", self.vintage)
        save = QtWidgets.QPushButton("Save description")
        save.clicked.connect(self.save)
        form.addRow(save)
        self.series: str | None = None

    def show_series(self, name: str | None) -> None:
        self.series = name
        self.setEnabled(name is not None)
        if name is None:
            return
        var = self.sheet.model.working.variables[name]
        self.title.setText(f"<b>{name}</b>")
        for field in self.TEXT:
            self.editors[field].setText(getattr(var, field) or "")
        for field in self.CHOICES:
            value = getattr(var, field)
            box = self.editors[field]
            box.setCurrentIndex(max(0, box.findData("" if value is None else getattr(value, "value", str(value)))))
        self.scale.setValue(var.scale or 0)
        self.recipe.setText(var.derived_from if var.recipe is not None else "no (plain values)")
        self.vintage.setText(str(var.vintage.released) if var.vintage else "not recorded")

    def changes(self) -> dict[str, Any]:
        if self.series is None:
            return {}
        var = self.sheet.model.working.variables[self.series]
        out: dict[str, Any] = {}
        for field in self.TEXT:
            text = self.editors[field].text()
            if text != (getattr(var, field) or ""):
                out[field] = text or (None if field == "symbol" else "")
        for field in self.CHOICES:
            value, old = self.editors[field].currentData() or "", getattr(var, field)
            if value != ("" if old is None else getattr(old, "value", str(old))):
                out[field] = value or None
        if self.scale.value() != (var.scale or 0):
            out["scale"] = self.scale.value()
        return out

    def save(self) -> None:
        changes = self.changes()
        if changes and self.series is not None:
            self.sheet._structural(SeriesInfo(self.sheet.name, self.series, json.dumps(changes)))
