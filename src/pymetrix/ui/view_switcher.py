"""The view switcher of a result window: a list of its views down the left side, grouped, beside the view shown.

A fitted equation has a dozen views (Summary, Graph, Coefficients, ..., Provenance, Python access) and a VAR nine: as tabs
they ran off the window's edge behind scroll arrows, at 125% scaling only three or four in view. Down the side every
view is in view (the list scrolls, in a short window, rather than hiding its end), each name in full, grouped as
Summary, Tables, Graphs, Diagnostics and Provenance. Up and Down move through the views, Ctrl+PgDown and Ctrl+PgUp too
(as between tabs), and the view a reader last picked for a kind of result is the one its next window opens on.

``ViewSwitcher`` keeps ``QTabWidget``'s calls (``addTab``, ``count``, ``tabText``, ``widget``, ``currentIndex``,
``setCurrentIndex``, ``currentWidget``, ``indexOf``, ``currentChanged``, ``setCornerWidget``, ``cornerWidget``), so code
and tests written for the tabs work unchanged; ``titles()``, ``select(title)`` and ``current_title()`` pick views by
name. Indexes follow the order views were added, whatever group the list shows them in.
"""

from __future__ import annotations

from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets
from .keys import key_text
from .metrics import set_spacing

__all__ = ["GROUPS", "ViewSwitcher", "view_group"]

GROUPS = ("Summary", "Tables", "Graphs", "Diagnostics", "Provenance")
_DIAGNOSTIC_WORDS = ("diagnostic", "correlogram", "roots", "q-q", "identification", "convergence", "check", "influence",
                     "leverage", "stability")
_DIAGNOSTIC_TITLES = ("residuals", "residual graph", "residual plot")  # the residuals themselves (their covariance is a table)
_SUMMARY_TITLES = ("summary", "overview")
_PROVENANCE_TITLES = ("provenance", "python access", "fields")
HEADINGS_FROM = 6  # views: fewer are a short list, shown without group headings
_LEAST_SIDE, _SIDE_SHARE = 72, 0.3  # the list's narrowest, and at most this share of the window


def view_group(title: str, figure: bool = False, first: bool = False) -> str:
    """The group a view is listed under: Summary (the summary or overview, or the first table), Provenance (how it
    was made, and its fields), Diagnostics (residuals, correlograms, roots, checks: tables and their graphs), Graphs
    (the other figures) or Tables (the rest)."""
    words = title.strip().lower()
    if words in _PROVENANCE_TITLES:
        return "Provenance"
    if words in _SUMMARY_TITLES or (first and not figure):
        return "Summary"
    if any(w in words for w in _DIAGNOSTIC_WORDS) or words in _DIAGNOSTIC_TITLES:
        return "Diagnostics"
    return "Graphs" if figure else "Tables"


def _settings() -> Any:
    return QtCore.QSettings(QtCore.QSettings.defaultFormat(), QtCore.QSettings.Scope.UserScope, "pymetrix", "desktop")


class ViewSwitcher(QtWidgets.QWidget):
    """Views listed down the left (grouped when there are many), the chosen one beside them; the corner widget (Copy
    summary, Next ▸, Add to report...) above the view. ``kind`` (a result's type name) is what the view picked last is
    remembered under."""

    currentChanged = QtCore.Signal(int)

    def __init__(self, parent: Any = None, kind: str = "") -> None:
        super().__init__(parent)
        self.kind = kind
        self._titles: list[str] = []
        self._figures: list[bool] = []
        self._corner: Any = None
        self._filling = False
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        set_spacing(outer, "tight")  # the list of views and the view beside it
        self.side = QtWidgets.QListWidget()
        self.side.setObjectName("result_views")
        self.side.setAccessibleName("Views")
        self.side.setToolTip(f"the views of this result: Up and Down (or {key_text('Ctrl+PgDown')}, {key_text('Ctrl+PgUp')}) "
                             "move between them")
        self.side.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.side.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self.side.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)
        self.side.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.side.currentRowChanged.connect(self._row_changed)
        outer.addWidget(self.side)
        right = QtWidgets.QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        set_spacing(right, "hairline")  # the corner row close over its view
        self._corner_row = QtWidgets.QHBoxLayout()
        self._corner_row.setContentsMargins(0, 0, 0, 0)
        self._corner_row.addStretch(1)
        right.addLayout(self._corner_row)
        self.stack = QtWidgets.QStackedWidget()
        right.addWidget(self.stack, 1)
        outer.addLayout(right, 1)
        bound: set[str] = set()  # the platform's next and previous tab, and Ctrl+PgDown, Ctrl+PgUp: each key once
        for keys, by in ((QtGui.QKeySequence("Ctrl+PgDown"), 1), (QtGui.QKeySequence("Ctrl+PgUp"), -1),
                         *((k, 1) for k in QtGui.QKeySequence.keyBindings(QtGui.QKeySequence.StandardKey.NextChild)),
                         *((k, -1) for k in QtGui.QKeySequence.keyBindings(QtGui.QKeySequence.StandardKey.PreviousChild))):
            if keys.isEmpty() or keys.toString() in bound:
                continue
            bound.add(keys.toString())
            shortcut = QtGui.QShortcut(keys, self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda by=by: self.step(by))
        self.side.installEventFilter(self)  # the list itself would take Ctrl+PgDown as PgDown
        self.side.setVisible(False)

    def eventFilter(self, watched: Any, event: Any) -> bool:
        if watched is self.side and event.type() == QtCore.QEvent.Type.KeyPress \
                and event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier \
                and event.key() in (QtCore.Qt.Key.Key_PageDown, QtCore.Qt.Key.Key_PageUp):
            self.step(1 if event.key() == QtCore.Qt.Key.Key_PageDown else -1)
            return True
        return super().eventFilter(watched, event)

    # --- QTabWidget's calls
    def addTab(self, widget: Any, title: str, figure: bool = False) -> int:
        """Add a view (``figure``: a graph, listed under Graphs unless it is a diagnostic one)."""
        self.stack.addWidget(widget)
        self._titles.append(title)
        self._figures.append(bool(figure))
        self._fill()
        return len(self._titles) - 1

    def count(self) -> int:
        return len(self._titles)

    def tabText(self, index: int) -> str:
        return self._titles[index] if 0 <= index < len(self._titles) else ""

    def widget(self, index: int) -> Any:
        return self.stack.widget(index)

    def indexOf(self, widget: Any) -> int:
        return self.stack.indexOf(widget)

    def currentIndex(self) -> int:
        return self.stack.currentIndex()

    def currentWidget(self) -> Any:
        return self.stack.currentWidget()

    def setCurrentWidget(self, widget: Any) -> None:
        self.setCurrentIndex(self.indexOf(widget))

    def setCurrentIndex(self, index: int) -> None:
        if not 0 <= index < self.count():
            return
        row = self._row_of(index)
        if row is not None and self.side.currentRow() != row:
            self._filling = True  # not the reader's pick: not remembered
            try:
                self.side.setCurrentRow(row)
            finally:
                self._filling = False
        if self.stack.currentIndex() != index:
            self.stack.setCurrentIndex(index)
            self.currentChanged.emit(index)

    def setCornerWidget(self, widget: Any, *_: Any) -> None:
        if self._corner is not None:
            self._corner_row.removeWidget(self._corner)
            self._corner.setParent(None)
        self._corner = widget
        if widget is not None:  # at the right, after the stretch; a button row folds into » what does not fit
            self._corner_row.addWidget(widget, 0, QtCore.Qt.AlignmentFlag.AlignRight)

    def cornerWidget(self, *_: Any) -> Any:
        return self._corner

    # --- by name
    def titles(self) -> list[str]:
        """The views' names, in the order they were added (their indexes)."""
        return list(self._titles)

    def current_title(self) -> str:
        return self.tabText(self.currentIndex())

    def select(self, title: str) -> bool:
        """Show the view named ``title``; False when there is none."""
        if title not in self._titles:
            return False
        self.setCurrentIndex(self._titles.index(title))
        return True

    def step(self, by: int) -> None:
        """The next (``by`` 1) or previous (-1) view, in the order the list shows them, round from the end."""
        order = self.listed()
        if not order:
            return
        here = order.index(self.currentIndex()) if self.currentIndex() in order else 0
        self.setCurrentIndex(order[(here + by) % len(order)])

    def listed(self) -> list[int]:
        """The views' indexes in the order the list shows them (by group)."""
        return [self.side.item(r).data(QtCore.Qt.ItemDataRole.UserRole) for r in range(self.side.count())
                if self.side.item(r).data(QtCore.Qt.ItemDataRole.UserRole) is not None]

    def groups(self) -> dict[str, list[str]]:
        """Each group's views, as listed."""
        out: dict[str, list[str]] = {}
        for i, title in enumerate(self._titles):
            out.setdefault(view_group(title, self._figures[i], i == 0), []).append(title)
        return {g: out[g] for g in GROUPS if g in out}

    def headings_shown(self) -> bool:
        return any(self.side.item(r).data(QtCore.Qt.ItemDataRole.UserRole) is None for r in range(self.side.count()))

    # --- remembering the view picked, per kind of result
    def restore(self) -> None:
        """Open on the view last picked for this kind of result, when it has one by that name."""
        if not self.kind:
            return
        try:
            title = _settings().value(f"result_views/{self.kind}", "")
        except Exception:  # noqa: BLE001 - no settings: the first view
            return
        if title:
            self.select(str(title))

    def _remember(self) -> None:
        if self.kind:
            _settings().setValue(f"result_views/{self.kind}", self.current_title())

    # --- the list
    def _row_of(self, index: int) -> int | None:
        for r in range(self.side.count()):
            if self.side.item(r).data(QtCore.Qt.ItemDataRole.UserRole) == index:
                return r
        return None

    def _fill(self) -> None:
        current = self.currentIndex()
        self._filling = True
        try:
            self.side.clear()
            groups: dict[str, list[int]] = {}
            for i, title in enumerate(self._titles):
                groups.setdefault(view_group(title, self._figures[i], i == 0), []).append(i)
            headings = len(self._titles) >= HEADINGS_FROM and len(groups) > 1
            for group in GROUPS:
                if group not in groups:
                    continue
                if headings:
                    head = QtWidgets.QListWidgetItem(group.upper())
                    head.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
                    font = head.font()
                    font.setBold(True)
                    font.setPointSizeF(max(font.pointSizeF() * 0.85, 6))
                    head.setFont(font)
                    head.setForeground(self.palette().color(QtGui.QPalette.ColorRole.PlaceholderText))
                    self.side.addItem(head)
                for i in groups[group]:
                    item = QtWidgets.QListWidgetItem(("  " if headings else "") + self._titles[i])
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, i)
                    item.setToolTip(self._titles[i])
                    self.side.addItem(item)
            row = self._row_of(current if current >= 0 else 0)
            if row is not None:
                self.side.setCurrentRow(row)
        finally:
            self._filling = False
        self.side.setVisible(len(self._titles) > 1)
        self._fit_side()

    def _row_changed(self, row: int) -> None:
        if row < 0:
            return
        index = self.side.item(row).data(QtCore.Qt.ItemDataRole.UserRole)
        if index is None:
            return
        if self.stack.currentIndex() != index:
            self.stack.setCurrentIndex(index)
            self.currentChanged.emit(index)
        if not self._filling:
            self._remember()

    def _fit_side(self) -> None:
        """The list as wide as its longest name, within the narrowest and a share of the window."""
        metrics = self.side.fontMetrics()
        wanted = max([metrics.horizontalAdvance(self.side.item(r).text()) for r in range(self.side.count())] or [0]) + 28
        if self.side.verticalScrollBar().isVisible():
            wanted += self.side.verticalScrollBar().sizeHint().width()
        room = int(self.width() * _SIDE_SHARE) if self.width() > 0 else wanted
        self.side.setFixedWidth(max(_LEAST_SIDE, min(wanted, max(room, _LEAST_SIDE))))

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._fit_side()

    def changeEvent(self, event: Any) -> None:
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.FontChange:  # a zoom: names wider or narrower
            self._fit_side()

    def minimumSizeHint(self) -> QtCore.QSize:
        least = self.stack.minimumSizeHint()
        side = _LEAST_SIDE + 4 if self.side.isVisibleTo(self) else 0
        corner = self._corner.minimumSizeHint().height() + 2 if self._corner is not None and self._corner.isVisibleTo(self) else 0
        return QtCore.QSize(least.width() + side, least.height() + corner)
