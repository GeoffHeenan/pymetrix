"""File → Browse series…: search and browse a data service's series, then load them (or fill Load from connector).

The browser only finds codes. What it loads is the step File → Load from connector… records, ``name =
Dataset.read("provider://codes")``, so the script replays without the browser (or the cache). SDMX: choose a
dataflow, then codes for its dimensions; the key is previewed as it is built. An SDMX provider's dataflows are in its own
category folders where it publishes useful ones, else grouped by family (``pymetrix.ext.dataflow_groups``) or ID prefix
(``pymetrix.ext.browse_folders`` has the rule; the tree's header says which), and the search box's words filter them
as typed, at any depth. A long World Bank topic or source has a folder per indicator-code prefix (NY., SL., ...). The
table's columns sort by clicking their headings; a third click returns to the provider's order (a search's relevance).
With a dataflow chosen, the search box can search the names and descriptions of its dimensions' codes instead
("In WEO's codes"); a code found is chosen by double-clicking it (or Enter).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from ..errors import PyMetrixError, SpecificationError
from . import theme
from .metrics import add_space, set_margins

NAME_ROLE, NOTES_ROLE = 0x0102, 0x0103  # a dataflow row's full name and notes (UserRole holds its code)
COLUMNS = ("Title", "Code", "Frequency", "Units", "Start", "End", "Updated")
CODE_COLUMNS = ("Name", "Code", "Dimension", "Description")  # the table's columns when it lists a dataflow's codes
QUERY_HINT = "words in the title or code, e.g. unemployment rate"
KEY_HINT = "Choose a dataflow on the left; its dimensions appear here to pick codes from."
WB_SOURCES = (("2", "World Development Indicators"), ("all", "All indicators (large)"))
TREE_HEADER = "Browse"
SORT_TIP = "Click to sort by this column, again for the reverse; a third click returns to the provider's order"


def _say_error(exc: BaseException) -> str:
    return str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)


def _EnterChooses(browser: Any) -> Any:  # noqa: N802 - made as a class is, built when Qt is loaded
    """An event filter for the results table: Enter (or Return) on a code found chooses it. The table passes Enter on
    to the dialog otherwise, where it presses the default button and searches again."""
    from PySide6.QtCore import QEvent, QObject, Qt

    class EnterChooses(QObject):
        def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802 - Qt's name
            if (event.type() == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                    and browser.found_codes):
                browser.choose_found(browser.table.currentRow())
                return True
            return False

    return EnterChooses(browser.dialog)


class SeriesBrowser:
    """The dialog, wrapped so tests can drive it: ``search``, ``select_rows``, ``choose`` (SDMX codes), ``command``."""

    def __init__(self, session: Any, parent: Any = None, on_load: Callable[[str], None] | None = None,
                 on_edit: Callable[[Any], None] | None = None) -> None:
        from PySide6 import QtWidgets
        from PySide6.QtCore import Qt

        from .. import connectors as api
        from ..io import connectors
        from .fit import ActionRow, fit_to_screen, scrolled_page

        self.session, self.on_load, self.on_edit = session, on_load, on_edit
        self.api = api
        self.results: list[Any] = []
        self.found_codes: list[tuple[str, str, str, str]] = []  # the codes the table lists, when it lists codes
        self.structure: Any = None
        self.pickers: dict[str, Any] = {}
        self.countries_loaded = False
        self.last_error = ""
        self.dialog = d = QtWidgets.QDialog(parent)
        d.setWindowTitle("Browse series")
        outer = QtWidgets.QVBoxLayout(d)
        outer.setContentsMargins(0, 0, 0, 0)
        self.page, body = scrolled_page()  # a small window scrolls rather than squashes the lists and boxes
        outer.addWidget(self.page)
        root = QtWidgets.QVBoxLayout(body)
        set_margins(root, "dialog_margin")

        top = QtWidgets.QHBoxLayout()
        self.provider = QtWidgets.QComboBox()
        self.descriptions = {k: c for k, c in connectors().items() if c.search}
        for key, conn in self.descriptions.items():
            self.provider.addItem(conn.label, key)
        top.addWidget(QtWidgets.QLabel("Provider"))
        top.addWidget(self.provider, 1)
        self.source_label = QtWidgets.QLabel("Source")  # a World Bank source or an SDMX provider (ECB, IMF...): source=
        self.source = QtWidgets.QComboBox()
        self.source.setEditable(True)
        self.source.setMinimumWidth(200)
        top.addWidget(self.source_label)
        top.addWidget(self.source)
        root.addLayout(top)

        keyrow = QtWidgets.QHBoxLayout()
        self.key_label = QtWidgets.QLabel("FRED API key")
        self.key = QtWidgets.QLineEdit()
        self.key.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.key.setPlaceholderText("kept for this session only, never saved (or set FRED_API_KEY)")
        keyrow.addWidget(self.key_label)
        keyrow.addWidget(self.key, 1)
        root.addLayout(keyrow)
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        root.addWidget(self.note)

        search_row = QtWidgets.QHBoxLayout()
        self.query = QtWidgets.QLineEdit()
        self.query.setPlaceholderText(QUERY_HINT)
        self.search_button = QtWidgets.QPushButton("Search")
        self.in_flow = QtWidgets.QCheckBox()  # shown once an SDMX dataflow is chosen: search its codes, not dataflows
        self.in_flow.setToolTip("search the names and descriptions of the chosen dataflow's codes (double-click a code "
                                "found to choose it); untick to search the provider's dataflows")
        self.in_flow.setVisible(False)
        self.refresh = QtWidgets.QCheckBox("Ask again (skip the cache)")
        self.refresh.setToolTip("searches are kept for a day and lists of categories, indicators and dataflows for a week; "
                                "tick to ask the provider afresh")
        search_row.addWidget(self.query, 1)
        search_row.addWidget(self.search_button)
        search_row.addWidget(self.in_flow)
        search_row.addWidget(self.refresh)
        root.addLayout(search_row)

        split = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels([TREE_HEADER])
        self.grouping = ""  # how the SDMX dataflows are grouped: "categories", "names", "prefixes", "flat"
        self._unsorted: list[Any] = []  # the table's rows in the provider's order (a search's relevance)
        self._sorted = (-1, True)  # (column, ascending); -1: the provider's order
        split.addWidget(self.tree)
        middle = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionsClickable(True)
        self.table.horizontalHeader().sectionClicked.connect(self.sort_by)
        middle.addWidget(self.table)
        self.details = QtWidgets.QTextBrowser()
        middle.addWidget(self.details)
        split.addWidget(middle)

        side = self.side = QtWidgets.QWidget()  # shown for the providers with a series key or countries to pick
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        self.key_box = QtWidgets.QGroupBox("Series key")
        key_layout = QtWidgets.QVBoxLayout(self.key_box)
        self.pick_area = QtWidgets.QScrollArea()
        self.pick_area.setWidgetResizable(True)
        self.pick_host = QtWidgets.QWidget()
        self.pick_layout = QtWidgets.QFormLayout(self.pick_host)
        self.pick_area.setWidget(self.pick_host)
        key_layout.addWidget(self.pick_area, 1)
        self.key_preview = QtWidgets.QLineEdit()
        self.key_preview.setReadOnly(True)
        key_layout.addWidget(QtWidgets.QLabel("Key (a dimension with nothing chosen is any code)"))
        key_layout.addWidget(self.key_preview)
        side_layout.addWidget(self.key_box, 1)
        add_space(side_layout)  # a section's room above a group box, as above a section's heading
        self.country_box = QtWidgets.QGroupBox("Countries (none: all)")
        country_layout = QtWidgets.QVBoxLayout(self.country_box)
        self.countries = QtWidgets.QListWidget()
        self.countries.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        country_layout.addWidget(self.countries)
        side_layout.addWidget(self.country_box, 1)
        split.addWidget(side)
        split.setSizes([300, 520, 280])  # the tree's folders nest (a provider's categories): names need the room
        root.addWidget(split, 1)

        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(QtWidgets.QLabel("Name"))
        self.name = QtWidgets.QLineEdit("data")
        self.name.setMaximumWidth(160)
        bottom.addWidget(self.name)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        bottom.addWidget(self.status, 1)
        self.actions_row = buttons = ActionRow()  # what does not fit goes in its » menu
        self.edit_button = QtWidgets.QPushButton("Add to Load from connector…")
        self.edit_button.setToolTip("open File → Load from connector… with these codes, to add periods, a vintage or options")
        self.load_button = QtWidgets.QPushButton("Load")
        self.load_button.setToolTip("read the series now, recorded as Dataset.read(address)")
        close = QtWidgets.QPushButton("Close")
        for b in (self.edit_button, self.load_button, close):
            buttons.addWidget(b)
        bottom.addWidget(buttons)
        root.addLayout(bottom)
        fit_to_screen(d, 1100, 680)

        self.provider.currentIndexChanged.connect(lambda *_: self._provider_changed())
        self.source.currentTextChanged.connect(lambda *_: self._source_changed())
        self.in_flow.toggled.connect(lambda *_: self._search_mode_changed())
        self.table.itemActivated.connect(lambda item: self.choose_found(item.row()))  # a double-click
        self._enter = _EnterChooses(self)  # Enter on a code found chooses it (and is not the dialog's Search)
        self.table.installEventFilter(self._enter)
        self.key.textChanged.connect(lambda text: self.api.set_api_key("fred", text) if text else None)
        self.search_button.clicked.connect(lambda: self.search())
        self.query.returnPressed.connect(lambda: self.search())
        self.query.textChanged.connect(lambda *_: self.filter_tree())
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.tree.itemExpanded.connect(self._expand)
        self.tree.itemClicked.connect(lambda item, _col: self._open_item(item))
        self.edit_button.clicked.connect(self.add_to_load)
        self.load_button.clicked.connect(self.load)
        close.clicked.connect(d.reject)
        self._provider_changed()

    # -- state ----------------------------------------------------------------------------------------------------------

    def provider_key(self) -> str:
        return str(self.provider.currentData() or "")

    def set_provider(self, key: str) -> None:
        index = self.provider.findData(key)
        if index < 0:
            raise SpecificationError(f"no searchable connector {key!r}")
        self.provider.setCurrentIndex(index)
        self._provider_changed()

    def source_value(self) -> str:
        text = self.source.currentText().strip()
        index = self.source.findText(text)
        data = self.source.itemData(index) if index >= 0 else None
        return str(data) if data else text.split(" ", 1)[0]

    def _error(self, exc: BaseException | str) -> None:
        self.last_error = exc if isinstance(exc, str) else _say_error(exc)
        self.status.setText(self.last_error)
        self.status.setStyleSheet(theme.css("error"))

    def _ok(self, text: str) -> None:
        self.last_error = ""
        self.status.setText(text)
        self.status.setStyleSheet("")

    def _provider_changed(self) -> None:
        from ..ext.connector_search import SDMX_STRUCTURES

        key = self.provider_key()
        conn = self.descriptions.get(key)
        note = conn.search_note if conn is not None else ""
        self.note.setText(note[:1].upper() + note[1:] + ("." if note and not note.endswith((".", ")")) else ""))
        self.key_label.setVisible(key == "fred")
        self.key.setVisible(key == "fred")
        self.source.blockSignals(True)
        self.source.clear()
        if key == "sdmx":
            for name in SDMX_STRUCTURES:
                self.source.addItem(name, name)
        elif key == "worldbank":
            for code, label in WB_SOURCES:
                self.source.addItem(f"{code} {label}", code)
        self.source.blockSignals(False)
        has_source = key in ("sdmx", "worldbank")
        self.source_label.setVisible(has_source)
        self.source.setVisible(has_source)
        self.key_box.setVisible(key == "sdmx")
        self.country_box.setVisible(key == "worldbank")
        self.side.setVisible(key in ("sdmx", "worldbank"))  # FRED's results take the width (a third was left empty)
        self._forget_structure()
        self._show([])
        self._fill_tree()
        if key == "worldbank" and not self.countries_loaded:
            self._load_countries()

    def _source_changed(self) -> None:
        self._forget_structure()  # the dataflow chosen was the last provider's
        self._fill_tree()

    def _forget_structure(self) -> None:
        self.structure = None
        self._clear_pickers()
        self.in_flow.setChecked(False)
        self.in_flow.setVisible(False)
        self._search_mode_changed()

    def codes_mode(self) -> bool:
        """Whether the search box searches the chosen dataflow's codes (else the provider's series or dataflows)."""
        return self.structure is not None and self.in_flow.isChecked()  # unticked whenever the dataflow is forgotten

    def _search_mode_changed(self) -> None:
        if self.codes_mode():
            self.query.setPlaceholderText(f"words in the names or descriptions of {self._flow_name()}'s codes, "
                                          "e.g. current account")
        else:
            self.query.setPlaceholderText(QUERY_HINT)
        self.filter_tree()

    def _flow_name(self) -> str:
        """The chosen dataflow's id: WEO for IMF.RES,WEO (or IMF.RES,WEO,1.0), EXR for EXR."""
        if self.structure is None:
            return ""
        parts = self.structure.flow.split(",")
        return parts[1] if len(parts) > 1 else parts[0]

    def _fill_tree(self) -> None:
        from PySide6 import QtWidgets
        from PySide6.QtCore import Qt

        self.tree.clear()
        self._set_grouping("", "")
        key = self.provider_key()
        if not key:
            return
        root_path = self.source_value() if key == "sdmx" else ""
        try:
            listing = self.api.browse(key, root_path, refresh=False) if key != "fred" or self.api.api_key("fred") else None
        except (PyMetrixError, ConnectionError) as exc:
            self._error(exc)
            return
        if listing is None:
            text = "Enter a FRED API key above to browse FRED's categories (searching needs one too)."
            item = QtWidgets.QTreeWidgetItem([text])
            item.setToolTip(0, text)
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.tree.addTopLevelItem(item)
            return
        if key == "sdmx" and root_path:
            self._fill_dataflows(root_path, listing.categories)
            self.filter_tree()
            return
        for cat in listing.categories:
            self.tree.addTopLevelItem(self._tree_item(cat))

    def _fill_dataflows(self, provider: str, categories: Any) -> None:
        """An SDMX provider's dataflows in folders: its categories, else by family or ID prefix, else a sorted list
        (``browse_folders.choose_tree``); the header says which, its tooltip why."""
        from PySide6 import QtWidgets

        from ..ext.browse_folders import Folder
        from ..ext.connector_search import dataflow_tree

        by_code = {cat.path: cat for cat in categories}
        flows = [(cat.path, cat.title or cat.path) for cat in categories]
        try:
            tree = dataflow_tree(provider, flows)
        except (PyMetrixError, ConnectionError) as exc:  # the dataflows were listed: show them, by name
            from ..ext.browse_folders import choose_tree

            tree = choose_tree(flows, None, why=_say_error(exc))

        def add(parent: Any, node: Any) -> None:
            if isinstance(node, Folder):
                folder = QtWidgets.QTreeWidgetItem([node.title])
                count = len(node.codes())
                folder.setToolTip(0, f"{node.title}\n{count} dataflow{'s' if count != 1 else ''}"
                                  + (f"\n\n{node.notes[:1].upper()}{node.notes[1:]}" if node.notes else ""))
                for child in node.children:
                    add(folder, child)
                item = folder
            else:
                item = self._tree_item(by_code[node.code], node.label)
            if parent is None:
                self.tree.addTopLevelItem(item)
            else:
                parent.addChild(item)

        for node in tree.root.children:
            add(None, node)
        self._set_grouping(tree.grouping, tree.note + (f".\nNot by the provider's categories: {tree.why}" if tree.why and tree.grouping != "categories" else "."))

    def _set_grouping(self, grouping: str, note: str) -> None:
        """Say how the dataflows are grouped, quietly: in the tree's header (``Browse · by category``) and its tooltip."""
        from ..ext.browse_folders import HEADERS

        self.grouping = grouping
        header = self.tree.headerItem()
        header.setText(0, f"{TREE_HEADER} · {HEADERS[grouping]}" if grouping else TREE_HEADER)
        header.setToolTip(0, note)

    def _tree_item(self, cat: Any, label: str = "") -> Any:
        from PySide6 import QtWidgets
        from PySide6.QtCore import Qt

        item = QtWidgets.QTreeWidgetItem([label or cat.title or cat.path])
        item.setData(0, Qt.ItemDataRole.UserRole, cat.path)
        item.setToolTip(0, cat.notes or cat.path)
        if cat.provider == "sdmx" and "/" in cat.path:  # a dataflow: names are long and elided, so the whole of it
            name = cat.title or cat.path
            item.setData(0, NAME_ROLE, name)
            item.setData(0, NOTES_ROLE, cat.notes)
            item.setToolTip(0, f"{name}\nDataflow {cat.path.partition('/')[2]}" + (f"\n\n{cat.notes}" if cat.notes else ""))
        if not cat.leaf:
            item.setChildIndicatorPolicy(QtWidgets.QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
        return item

    def _expand(self, item: Any) -> None:
        if item.childCount() or item.data(0, 0x0101) == "done":
            return
        self._open_item(item, show=False)

    def _open_item(self, item: Any, show: bool = True) -> None:
        from PySide6.QtCore import Qt

        path = item.data(0, Qt.ItemDataRole.UserRole)
        if not path:
            return
        key = self.provider_key()
        if key == "sdmx" and "/" in path:  # a dataflow
            self.show_structure(path)
            return
        if key == "worldbank" and path.startswith("country/"):
            self.choose_countries([path.partition("/")[2]], add=True)
            return
        try:
            listing = self.api.browse(key, path, refresh=self.refresh.isChecked())
        except (PyMetrixError, ConnectionError) as exc:
            self._error(exc)
            return
        if not item.childCount():
            for cat in listing.categories:
                item.addChild(self._tree_item(cat))
            item.setData(0, 0x0101, "done")
        if show and listing.series:
            self._show(list(listing.series))
            self._ok(f"{len(listing.series)} series in {item.text(0)}")

    def _items(self) -> tuple[list[Any], list[Any]]:
        """(dataflow rows, folders) of the tree, at any depth; folders innermost first."""
        rows, folders = [], []

        def walk(item: Any) -> None:
            if item.childCount():
                for j in range(item.childCount()):
                    walk(item.child(j))
                folders.append(item)
            elif item.data(0, NAME_ROLE) is not None:
                rows.append(item)

        for i in range(self.tree.topLevelItemCount()):
            walk(self.tree.topLevelItem(i))
        return rows, folders

    def filter_tree(self, text: str | None = None) -> int:
        """Show only the dataflows whose name or code (else notes) hold every word of ``text`` (else the search box's),
        at any depth, opening the folders with any; nothing typed shows them all. The number shown (SDMX's list only:
        other providers' trees are places to browse into, and are left alone)."""
        if self.provider_key() != "sdmx":
            return 0
        words = (self.query.text() if text is None else text).strip()
        if text is None and self.codes_mode():  # the words are for the dataflow's codes, not the list of dataflows
            words = ""
        rows, folders = self._items()
        shown = set(map(id, self.api.rank(rows, words, lambda r: (f"{r.data(0, NAME_ROLE)} {r.data(0, 0x0100)}",
                                                                     r.data(0, NOTES_ROLE) or ""), len(rows)))) if words else None
        for row in rows:
            row.setHidden(shown is not None and id(row) not in shown)
        for folder in folders:  # innermost first, so a folder sees whether its sub-folders are shown
            any_shown = any(not folder.child(j).isHidden() for j in range(folder.childCount()))
            folder.setHidden(not any_shown)
            if shown is not None:
                if any_shown and not folder.isExpanded():
                    folder.setExpanded(True)
                    folder.setData(0, 0x0101, "filtered")
            elif folder.data(0, 0x0101) == "filtered":  # opened by the filter: closed again when it is cleared
                folder.setExpanded(False)
                folder.setData(0, 0x0101, None)
        found = {r.data(0, 0x0100) for r in rows if shown is None or id(r) in shown}
        return len(found)  # a dataflow in two folders counts once

    # -- searching -------------------------------------------------------------------------------------------------------

    def search(self, text: str | None = None) -> list[Any]:
        """Search the provider for ``text`` (else the search box's words) and list what is found."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        if text is not None:
            self.query.setText(text)
        words = self.query.text().strip()
        if not words:
            self._error("type words to search for")
            return []
        if self.codes_mode():
            return self.search_codes(words)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            found = self.api.search(self.provider_key(), words, source=self.source_value() if self.provider_key() in ("sdmx", "worldbank") else "",
                                    refresh=self.refresh.isChecked())
        except (PyMetrixError, ConnectionError) as exc:
            self._error(exc)
            return []
        finally:
            QApplication.restoreOverrideCursor()
        self._show(found)
        self._ok(f"{len(found)} found" + ("" if found else ": try other words"))
        return found

    def search_codes(self, text: str) -> list[tuple[str, str, str, str]]:
        """The chosen dataflow's codes whose name or code (else description) hold every word of ``text``, as
        (dimension, code, name, description), listed in the table to choose from. An indicator dimension's first."""
        assert self.structure is not None
        dims = sorted(self.structure.dimensions, key=lambda d: "INDICATOR" not in d.id.upper())
        codes = [(d.id, c, n, dict(d.descriptions).get(c, "")) for d in dims for c, n in d.codes]
        found = self.api.rank(codes, text, lambda m: (f"{m[2]} {m[1]}", m[3]), len(codes))
        self.results = []
        self._unsorted, self._sorted = list(found), (-1, True)
        self._render_codes(found)
        self.table.resizeColumnsToContents()
        flow = self._flow_name()
        self._ok(f"{len(found)} of {flow}'s codes found: double-click one (or press Enter) to choose it" if found
                 else f"none of {flow}'s codes hold those words: try others, or untick “In {flow}'s codes” to search dataflows")
        return found

    def _render_codes(self, found: list[tuple[str, str, str, str]]) -> None:
        from PySide6 import QtWidgets

        self.found_codes = list(found)
        self.table.clearContents()
        self.table.setColumnCount(len(CODE_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(CODE_COLUMNS))
        self.table.setRowCount(len(found))
        for row, (dim, code, name, notes) in enumerate(found):
            for col, value in enumerate((name, code, dim, notes)):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(f"{name}\n{dim} {code}" + (f"\n\n{notes}" if notes else "") + "\n\nDouble-click (or Enter) to choose it")
                self.table.setItem(row, col, item)
        self._header_state()
        self.details.clear()

    def choose_found(self, row: int) -> None:
        """Choose the code found in ``row`` of the table (as double-clicking it does), adding it to its dimension."""
        if not self.found_codes or not 0 <= row < len(self.found_codes):
            return
        dim, code, _name, _notes = self.found_codes[row]
        self.choose(dim, sorted(set(self.choices().get(dim, [])) | {code}))
        self._ok(f"{dim} {code} chosen (the key is below, under Series key)")

    def _show(self, found: list[Any]) -> None:
        """List series (or dataflows) in the table, in the order given: the provider's (a search's relevance)."""
        self._unsorted, self._sorted = list(found), (-1, True)
        self._render_series(found)
        self.table.resizeColumnsToContents()

    @staticmethod
    def _cells(s: Any) -> tuple[str, ...]:
        return (s.title, s.code, s.frequency, s.units, s.start, s.end, s.last_updated[:10])

    def _render_series(self, found: list[Any]) -> None:
        from PySide6 import QtWidgets

        self.found_codes = []
        self.table.clearContents()
        self.table.setColumnCount(len(COLUMNS))
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.results = list(found)
        self.table.setRowCount(len(found))
        for row, s in enumerate(found):
            for col, value in enumerate(self._cells(s)):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(value))
        self._header_state()
        self.details.clear()

    def _header_state(self) -> None:
        """The sort arrow on the column sorted by (none in the provider's order), and each heading's tooltip."""
        from PySide6.QtCore import Qt

        header = self.table.horizontalHeader()
        column, ascending = self._sorted
        header.setSortIndicatorShown(column >= 0)
        if column >= 0:
            header.setSortIndicator(column, Qt.SortOrder.AscendingOrder if ascending else Qt.SortOrder.DescendingOrder)
        for i in range(self.table.columnCount()):
            heading = self.table.horizontalHeaderItem(i)
            if heading is not None:
                heading.setToolTip(SORT_TIP)

    def sort_by(self, column: int) -> None:
        """Sort the table by ``column`` (as clicking its heading does): ascending, then descending, then the provider's
        order again. Ties keep the provider's order, and empty cells go last; the rows selected stay selected."""
        if not self._unsorted or column < 0:
            return
        current, ascending = self._sorted
        self._sorted = (column, True) if column != current else (column, False) if ascending else (-1, True)
        codes = bool(self.found_codes)
        cell = (lambda m: (m[2], m[1], m[0], m[3])[column]) if codes else (lambda s: self._cells(s)[column])
        rows = list(self._unsorted)
        if self._sorted[0] >= 0:
            filled = sorted((r for r in rows if cell(r)), key=lambda r: cell(r).casefold(), reverse=not self._sorted[1])
            rows = filled + [r for r in rows if not cell(r)]
        was = [self.found_codes[i] if codes else self.results[i] for i in
               sorted({i.row() for i in self.table.selectionModel().selectedRows()})
               if i < len(self.found_codes if codes else self.results)]
        self.table.blockSignals(True)  # the same rows: their details are shown again, nothing is asked for
        try:
            if codes:
                self._render_codes(rows)
            else:
                self._render_series(rows)
            self.select_rows([i for i, r in enumerate(rows) if any(r is w for w in was)])
        finally:
            self.table.blockSignals(False)
        self._selection_changed(structure=False)

    def select_rows(self, rows: list[int]) -> None:
        from PySide6.QtCore import QItemSelectionModel

        model = self.table.selectionModel()
        model.clearSelection()
        flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
        for r in rows:
            model.select(self.table.model().index(r, 0), flags)

    def selected(self) -> list[Any]:
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        return [self.results[r] for r in rows if r < len(self.results)]

    def _selection_changed(self, structure: bool = True) -> None:
        import html

        if self.found_codes:
            rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
            if rows and rows[-1] < len(self.found_codes):
                dim, code, name, notes = self.found_codes[rows[-1]]
                self.details.setHtml(f"<h3>{html.escape(name or code)}</h3><p><b>{html.escape(dim)}</b> {html.escape(code)}</p>"
                                     + (f"<p>{html.escape(notes)}</p>" if notes else "") + "<p>Double-click to choose it.</p>")
            return
        chosen = self.selected()
        if not chosen:
            self.details.clear()
            return
        s = chosen[-1]
        facts = [("Code", s.code), ("Frequency", s.frequency), ("Units", s.units), ("Seasonal adjustment", s.seasonal_adjustment),
                 ("Observations", f"{s.start} to {s.end}" if s.start or s.end else ""), ("Last updated", s.last_updated),
                 ("Source", s.source), ("Address", s.address if s.kind == "series" else "")]
        rows = "".join(f"<tr><td><b>{k}</b></td><td>{html.escape(v)}</td></tr>" for k, v in facts if v)
        notes = "".join(f"<p>{html.escape(p)}</p>" for p in s.notes.split("\n\n") if p.strip())
        self.details.setHtml(f"<h3>{html.escape(s.title or s.code)}</h3><table>{rows}</table>{notes}")
        if structure and s.kind == "dataflow" and len(chosen) == 1:
            self.show_structure(s.code)

    # -- SDMX keys ----------------------------------------------------------------------------------------------------

    def _clear_pickers(self) -> None:
        from PySide6 import QtWidgets

        while self.pick_layout.rowCount():
            self.pick_layout.removeRow(0)
        self.pickers = {}
        self.unloaded: dict[str, Any] = {}  # dimension -> the label saying its codes could not be loaded
        self.key_preview.clear()
        self.key_box.setTitle("Series key")
        self.key_hint = None
        if self.structure is None:  # never an empty box: say what goes here
            self.key_hint = QtWidgets.QLabel(KEY_HINT)
            self.key_hint.setWordWrap(True)
            self.pick_layout.addRow(self.key_hint)

    def show_structure(self, code: str) -> Any:
        """Ask for a dataflow's structure (``PROVIDER/FLOW``) and give a picker per dimension."""
        from PySide6 import QtWidgets

        try:
            structure = self.api.structure("sdmx", code, refresh=self.refresh.isChecked())
        except (PyMetrixError, ConnectionError) as exc:
            self._error(exc)
            return None
        self.structure = structure
        self._clear_pickers()
        missing = []
        for dim in structure.dimensions:
            if not dim.codes:  # codes that could not be loaded: said in words, and typed instead
                missing.append(dim.id)
                said = QtWidgets.QLabel(f"{dim.id}: its codes could not be loaded ({dim.note or 'none were given'}). "
                                        "Type codes, with + between them.")
                said.setWordWrap(True)
                said.setStyleSheet(theme.css("error"))
                typed = QtWidgets.QLineEdit()
                typed.setPlaceholderText("codes, e.g. A+B (empty: any code)")
                typed.setToolTip(f"{dim.name or dim.id}: {dim.note}" if dim.note else dim.name or dim.id)
                typed.textChanged.connect(self._update_key)
                self.pick_layout.addRow(said)  # the whole width, so the reason is read in full
                self.pick_layout.addRow(dim.id, typed)
                self.unloaded[dim.id] = said
                self.pickers[dim.id] = typed
                continue
            picker = QtWidgets.QListWidget()
            picker.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
            picker.setMaximumHeight(96)
            notes = dict(dim.descriptions)
            for c, label in dim.codes:
                item = QtWidgets.QListWidgetItem(f"{c}  {label}".rstrip())
                item.setData(0x0100, c)
                if notes.get(c):
                    item.setToolTip(f"{label or c}\n\n{notes[c]}")
                picker.addItem(item)
            picker.itemSelectionChanged.connect(self._update_key)
            picker.setToolTip((dim.name or dim.id) + (f"\n{dim.note[:1].upper()}{dim.note[1:]}" if dim.note else ""))
            self.pick_layout.addRow(dim.id, picker)
            self.pickers[dim.id] = picker
        self.key_box.setTitle(f"Series key: {structure.title or structure.flow}")
        self._update_key()
        self.in_flow.setText(f"In {self._flow_name()}'s codes")
        self.in_flow.setVisible(True)
        self.in_flow.setChecked(True)  # the next search is in this dataflow's codes (untick for dataflows)
        self._search_mode_changed()
        if missing:
            self._error(f"{structure.flow}: the codes of {', '.join(missing)} could not be loaded; type them")
        else:
            self._ok(f"{structure.flow}: {len(structure.dimensions)} dimensions; choose codes for the key, or search its codes")
        return structure

    def choose(self, dimension: str, codes: list[str]) -> None:
        """Choose ``codes`` for an SDMX dimension (as clicking them does)."""
        from PySide6 import QtWidgets

        picker = self.pickers.get(dimension)
        if picker is None:
            raise SpecificationError(f"no dimension {dimension!r}; dimensions: {', '.join(self.pickers)}")
        if isinstance(picker, QtWidgets.QLineEdit):  # codes typed
            picker.setText("+".join(codes))
            return
        for i in range(picker.count()):
            picker.item(i).setSelected(picker.item(i).data(0x0100) in codes)
        self._update_key()

    def choices(self) -> dict[str, list[str]]:
        from PySide6 import QtWidgets

        return {d: [c for c in re.split(r"[+,\s]+", p.text()) if c] if isinstance(p, QtWidgets.QLineEdit)
                else [i.data(0x0100) for i in p.selectedItems()] for d, p in self.pickers.items()}

    def _update_key(self) -> None:
        if self.structure is not None:
            try:
                self.key_preview.setText(self.structure.address(self.choices()))
            except PyMetrixError as exc:
                self._error(exc)

    # -- World Bank countries -----------------------------------------------------------------------------------------

    def _load_countries(self) -> None:
        from PySide6 import QtWidgets

        try:
            listing = self.api.browse("worldbank", "countries")
        except (PyMetrixError, ConnectionError) as exc:
            self._error(exc)
            return
        self.countries.clear()
        for cat in listing.categories:
            code = cat.path.partition("/")[2]
            item = QtWidgets.QListWidgetItem(f"{code}  {cat.title}")
            item.setData(0x0100, code)
            item.setToolTip(cat.notes)
            self.countries.addItem(item)
        self.countries_loaded = True

    def choose_countries(self, codes: list[str], add: bool = False) -> None:
        for i in range(self.countries.count()):
            item = self.countries.item(i)
            if item.data(0x0100) in codes:
                item.setSelected(True)
            elif not add:
                item.setSelected(False)

    def chosen_countries(self) -> list[str]:
        return [self.countries.item(i).data(0x0100) for i in range(self.countries.count()) if self.countries.item(i).isSelected()]

    # -- the step --------------------------------------------------------------------------------------------------------

    def command(self) -> Any:
        """The Load from connector step for what is chosen (a SpecificationError says what is missing)."""
        from ..commands import LoadFromConnector

        key = self.provider_key()
        name = self.name.text().strip() or "data"
        if key == "sdmx":
            if self.structure is None:
                raise SpecificationError("choose a dataflow, then codes for its dimensions")
            return LoadFromConnector("sdmx", self.key_preview.text().partition("://")[2], name=name)
        chosen = [s for s in self.selected() if s.kind == "series"]
        if not chosen:
            raise SpecificationError("select one or more series in the list")
        if key == "fred":
            freqs = sorted({s.frequency for s in chosen if s.frequency})
            if len(freqs) > 1:
                raise SpecificationError(f"the series are {' and '.join(freqs)}: load series of one frequency at a time")
        options = ""
        if key == "worldbank" and self.chosen_countries():
            options = "country=" + ";".join(self.chosen_countries())
        return LoadFromConnector(key, ",".join(s.code for s in chosen), name=name, options=options)

    def add_to_load(self) -> Any:
        """Open File → Load from connector… filled with the codes chosen."""
        try:
            cmd = self.command()
        except PyMetrixError as exc:
            self._error(exc)
            return None
        if self.on_edit is not None:
            self.on_edit(cmd)
        return cmd

    def load(self) -> Any:
        """Read the series chosen now, as a recorded Load from connector step."""
        try:
            cmd = self.command()
            errors = cmd.validate(self.session)
            if errors:
                raise SpecificationError("; ".join(errors))
            self.session.run(cmd)
        except (PyMetrixError, ConnectionError) as exc:
            self._error(exc)
            return None
        self._ok(f"loaded {cmd.name}: {cmd.address()}")
        if self.on_load is not None:
            self.on_load(cmd.name)
        return cmd


def series_browser(session: Any, parent: Any = None, on_load: Callable[[str], None] | None = None,
                   on_edit: Callable[[Any], None] | None = None) -> SeriesBrowser:
    """File → Browse series…: the browser (``.dialog`` is the QDialog)."""
    return SeriesBrowser(session, parent, on_load=on_load, on_edit=on_edit)
