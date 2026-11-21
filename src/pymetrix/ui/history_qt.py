"""The History panel: every step of the session, dialogs and typed code alike.

Select steps (Shift- and Ctrl-click) to copy their code (Ctrl+C), send them to
the Immediate or Code window, re-run them, or reopen a step's dialog filled in
as it was. Rows can also be dragged into either window. Failed steps are shown
in red with their error. A step an assistant proposed, and the user ran from its
dialog, says "proposed" after its dialog's name.
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from ..commands import CommandRecord, Session
from ..commands.session import SCRIPT_HEADER, _plugin_imports
from . import theme

SOURCES = {"immediate": "Immediate", "editor": "Code window", "script": "Script"}


def describe(record: CommandRecord) -> str:
    """Where a step came from: the dialog's name, or the window the code was typed in."""
    if record.code_step:
        return SOURCES.get(getattr(record.command, "source", ""), "Code")
    menu = getattr(type(record.command), "label", "") or getattr(type(record.command), "menu", "") or type(record.command).__name__
    from ..wording import menu_label

    return menu_label(menu).rstrip(".… ")


def menu_path(record: CommandRecord) -> str:
    """Where a dialog step's dialog is in the menus ("Estimate → Equation…"), or "" for typed code and steps whose
    dialog has no menu item."""
    if record.code_step:
        return ""
    menu = getattr(type(record.command), "menu", "") or ""
    if not menu:
        return ""
    from ..commands.menus import shown
    from ..help import menu_path as laid_out
    from ..wording import menu_label

    return laid_out(type(record.command)) or f"{menu.partition('/')[0]} → {shown(menu_label(menu))}"


def code_of(records: list[CommandRecord], header: bool = False, include_failed: bool = False) -> str:
    """The code of the steps, in order; with ``header``, the imports a standalone script needs."""
    body = "\n".join(r.code for r in records if (r.ok or include_failed) and r.code.strip())
    return (SCRIPT_HEADER + _plugin_imports(body) + "\n" + body + "\n") if header and body else body


class HistoryPanel(QtWidgets.QTreeWidget):
    """Callbacks: ``send_immediate(code)``, ``send_editor(code)``, ``rerun(records)``, ``open_dialog(record)``."""

    def __init__(self, send_immediate: Callable[[str], None], send_editor: Callable[[str], None],
                 rerun: Callable[[list[CommandRecord]], None], open_dialog: Callable[[CommandRecord], None]) -> None:
        super().__init__()
        self.send_immediate, self.send_editor, self.rerun, self.open_dialog = send_immediate, send_editor, rerun, open_dialog
        self.setHeaderLabels(["#", "From", "Code", "Made", "Time"])
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)  # Code gets the spare width
        self.number_shown = False  # # and Time hidden at first: in a 270-pixel panel they left Code three letters ("dat…");
        for col in (0, 4):  # the header's menu shows them, and each row's tooltip has its time
            self.setColumnHidden(col, True)
        self.header().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.header().customContextMenuRequested.connect(self._columns_menu)
        self.setRootIsDecorated(False)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragOnly)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.itemDoubleClicked.connect(lambda item, _c: self._open(item))
        copy = QtGui.QAction("Copy code", self)
        copy.setShortcut(QtGui.QKeySequence.StandardKey.Copy)
        copy.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        copy.triggered.connect(self.copy)
        self.addAction(copy)
        self.records: list[CommandRecord] = []

    def show_session(self, session: Session) -> None:
        self.clear()
        self.records = list(session.history)
        red = QtGui.QBrush(theme.colour("error"))
        grey = QtGui.QBrush(theme.colour("muted"))
        resolved = session.resolved_failures() if hasattr(session, "resolved_failures") else {}
        for i, record in enumerate(self.records, 1):
            first = record.code.strip().splitlines()[0] if record.code.strip() else ""
            more = " …" if record.code.strip().count("\n") else ""
            item = QtWidgets.QTreeWidgetItem([str(i), describe(record), first + more, ", ".join(sorted(record.writes)),
                                              record.created.strftime("%H:%M:%S")])
            item.setData(0, Qt.ItemDataRole.UserRole, i - 1)
            where = menu_path(record)
            item.setToolTip(1, f"{where}, step {i} at {record.created.strftime('%H:%M:%S')}" if where
                            else f"Step {i} at {record.created.strftime('%H:%M:%S')}")
            from ..plugins import plugin_of_command

            origin = plugin_of_command(type(record.command))
            item.setToolTip(2, record.code + (f"\n\nError: {record.error}" if record.error else "")
                            + (f"\n\nFrom the plugin {origin.name} {origin.version}" if origin is not None else ""))
            if origin is not None:
                item.setText(1, f"{item.text(1)} · {origin.prefix}")
            if record.proposed_by:  # an assistant proposed it; the user ran it from its dialog
                item.setText(1, f"{item.text(1)} · proposed")
                item.setToolTip(1, f"{item.toolTip(1)}\nProposed by {record.proposed_by} (an assistant); you ran it from its dialog")
            if record.skipped:
                for col in range(5):
                    item.setForeground(col, grey)
                item.setText(3, "not run")
            elif record.deferred:  # recorded, not run yet: Compute (or Recalculate) runs it in its place
                for col in range(5):
                    item.setForeground(col, grey)
                item.setText(3, "not computed")
                item.setToolTip(3, f"Not computed yet (deferred): {', '.join(record.outputs)}. Compute it, or Recalculate."
                                + (f"\nIt may not compute: {record.note}" if record.note else ""))
            elif not record.ok and i - 1 in resolved:  # run again successfully later: no longer a problem
                for col in range(5):
                    item.setForeground(col, grey)
                item.setText(3, "failed, resolved")
                item.setToolTip(3, f"Failed; step {resolved[i - 1] + 1} then ran the same dialog on the same target successfully")
            elif not record.ok:
                for col in range(5):
                    item.setForeground(col, red)
                item.setText(3, "failed")
            elif not record.replayable:
                item.setToolTip(2, record.code + "\n\n(IPython syntax: recorded, not replayed)")
            self.addTopLevelItem(item)
        if not self.records:  # an empty state, not a blank pane; it cannot be selected
            from .messages import EMPTY_STATES

            empty = QtWidgets.QTreeWidgetItem([EMPTY_STATES["history"]])
            empty.setToolTip(0, EMPTY_STATES["history_tip"])
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self.addTopLevelItem(empty)
            empty.setFirstColumnSpanned(True)  # the whole row, not the narrow Code column
        self.setColumnHidden(0, not self.number_shown and bool(self.records))  # the empty line spans from column 0
        for col in (0, 1, 3, 4):
            self.resizeColumnToContents(col)
        if self.records:
            self.scrollToBottom()

    def _show_column(self, col: int, on: bool) -> None:
        if col == 0:
            self.number_shown = on
        self.setColumnHidden(col, not on)
        self.resizeColumnToContents(col)

    def _columns_menu(self, pos: QtCore.QPoint) -> None:
        """The header's menu: show or hide each column but Code."""
        menu = QtWidgets.QMenu(self)
        for col in (0, 1, 3, 4):
            action = menu.addAction(self.headerItem().text(col) if col else "# (step number)")
            action.setCheckable(True)
            action.setChecked(not self.isColumnHidden(col))
            action.toggled.connect(lambda on, c=col: self._show_column(c, on))
        menu.exec(self.header().mapToGlobal(pos))

    def select_record(self, record: CommandRecord | None) -> None:
        """Show and select one step (e.g. the one that made an object)."""
        if record is None or record not in self.records:
            return
        row = self.records.index(record)
        item = self.topLevelItem(row)
        self.clearSelection()
        item.setSelected(True)
        self.setCurrentItem(item)
        self.scrollToItem(item)
        window = self.window()
        for dock in window.findChildren(QtWidgets.QDockWidget) if window is not None else []:
            if dock.widget() is self:
                dock.show()
                dock.raise_()

    def selected(self) -> list[CommandRecord]:
        rows = sorted(item.data(0, Qt.ItemDataRole.UserRole) for item in self.selectedItems())
        return [self.records[r] for r in rows]

    def copy(self) -> str:
        """Copy the selected steps' code, with imports, so it runs anywhere."""
        text = code_of(self.selected(), header=True)
        QtWidgets.QApplication.clipboard().setText(text)
        return text

    def mimeData(self, items: list[QtWidgets.QTreeWidgetItem]) -> QtCore.QMimeData:  # noqa: N802 (drag)
        data = QtCore.QMimeData()
        data.setText(code_of([self.records[i.data(0, Qt.ItemDataRole.UserRole)] for i in items]))
        return data

    def _open(self, item: QtWidgets.QTreeWidgetItem) -> None:
        record = self.records[item.data(0, Qt.ItemDataRole.UserRole)]
        if record.code_step:
            self.send_editor(record.code)
        else:
            self.open_dialog(record)

    def _menu(self, pos: QtCore.QPoint) -> None:
        records = self.selected()
        if not records:
            return
        menu = QtWidgets.QMenu(self)
        ok = [r for r in records if r.ok]
        menu.addAction("Copy code", self.copy)
        menu.addAction("Send to Immediate", lambda: self.send_immediate(code_of(ok))).setEnabled(bool(ok))
        menu.addAction("Send to Code window", lambda: self.send_editor(code_of(ok))).setEnabled(bool(ok))
        menu.addSeparator()
        menu.addAction("Re-run", lambda: self.rerun(ok)).setEnabled(bool(ok))
        single = records[0] if len(records) == 1 and not records[0].code_step else None
        menu.addAction("Open dialog…", lambda: self.open_dialog(single)).setEnabled(single is not None)
        menu.exec(self.viewport().mapToGlobal(pos))
