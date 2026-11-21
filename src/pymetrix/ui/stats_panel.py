"""A result table's notes in its window: the statistics grouped under small headings, side by side where the window
is wide (up to three columns) and stacked where it is narrow, then the notes that are sentences.

The split and the grouping are ``pymetrix.summary_layout.split_notes`` (the same as the notebook HTML); this draws
them with widgets, so the columns follow the window's width and the colours follow the palette."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from ..summary_layout import StatGroup, is_number_text, split_notes
from . import theme
from .metrics import px, set_margins, set_spacing

LABEL_GAP = 18  # pixels between a statistic's label and its value
COLUMN_GAP = 36  # pixels between groups side by side
MAX_COLUMNS = 3


def _tabular(font: QtGui.QFont) -> QtGui.QFont:
    """``font`` with tabular figures, so the decimals of a column of values line up (Qt 6.7 or later; before, the
    font's own figures, which are tabular in most UI fonts)."""
    tag = getattr(QtGui.QFont, "Tag", None)
    if tag is not None and hasattr(font, "setFeature"):
        try:
            font.setFeature(tag("tnum"), 1)
        except (TypeError, ValueError):  # a binding without the feature API
            pass
    return font


class _Wrapping(QtWidgets.QLabel):
    """A label on one line where there is room (its size hint), wrapped where there is not (a narrow window)."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setWordWrap(True)
        self.setTextFormat(QtCore.Qt.TextFormat.PlainText)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt's name
        width = min(self.fontMetrics().horizontalAdvance(self.text()) + 2, 420)
        return QtCore.QSize(width, self.heightForWidth(width))

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt's name
        """Its longest word, one line high: the layout asks ``heightForWidth`` for the height at the width it gives,
        so the minimum is not the height of the text wrapped at its narrowest (which would make a scroll area think
        the page squashed)."""
        fm = self.fontMetrics()
        word = max((fm.horizontalAdvance(w) for w in self.text().split()), default=0)
        return QtCore.QSize(word + 2, fm.height())


class _Group(QtWidgets.QWidget):
    """One group: a muted heading over rows of a muted label, left, and its value, right-aligned."""

    def __init__(self, group: StatGroup, headed: bool, parent: Any = None) -> None:
        super().__init__(parent)
        self.setObjectName("statGroup")
        self.title = group.title
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(LABEL_GAP)
        grid.setVerticalSpacing(px("hairline"))  # deliberately close: rows of a table of figures
        row = 0
        muted = theme.muted_css(self.palette())  # labels and headings: grey on a light theme, light grey on a dark one
        self.heading = None
        if headed:
            self.heading = QtWidgets.QLabel(group.title)
            self.heading.setStyleSheet(muted + " padding-bottom: 2px;")
            self.heading.setObjectName("statHeading")
            grid.addWidget(self.heading, 0, 0, 1, 2)
            row = 1
        right = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop
        left = QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
        self.rows = []
        for label, value in group.items:
            name = _Wrapping(label)  # a very narrow window wraps rather than scrolls sideways
            name.setStyleSheet(muted)
            shown = _Wrapping(value)
            shown.setAlignment(right if is_number_text(value) else left)  # numbers line up on the right, words read from the left
            shown.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            shown.setToolTip(f"{label}: {value}")
            grid.addWidget(name, row, 0, left)
            grid.addWidget(shown, row, 1)
            self.rows.append((name, shown))
            row += 1
        grid.setColumnStretch(1, 1)  # the values' column takes any extra room, the numbers staying at its right
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
        self.refresh()

    def refresh(self) -> None:
        """The heading's and values' fonts from the group's own (which follows the window's zoom), and the width the
        group takes with each statistic on one line."""
        base = self.font()
        natural = 0
        if self.heading is not None:
            font = QtGui.QFont(base)
            font.setPointSizeF(base.pointSizeF() * 0.88)
            font.setWeight(QtGui.QFont.Weight.DemiBold)
            self.heading.setFont(font)
            natural = QtGui.QFontMetrics(font).horizontalAdvance(self.title)
        figures = _tabular(QtGui.QFont(base))
        plain, numbers = QtGui.QFontMetrics(base), QtGui.QFontMetrics(figures)
        labels = values = 0
        for name, shown in self.rows:
            shown.setFont(figures)
            labels = max(labels, plain.horizontalAdvance(name.text()))
            values = max(values, numbers.horizontalAdvance(shown.text()))
        natural = max(natural, labels + LABEL_GAP + values + 4)
        self.natural = min(natural, 420)  # a long Method or Sample wraps past 420 pixels
        self.updateGeometry()

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt's name
        hint = super().sizeHint()
        return QtCore.QSize(self.natural, hint.height())


class StatsPanel(QtWidgets.QWidget):
    """A table's notes: grouped statistics in columns that follow the width, then the prose, word-wrapped."""

    def __init__(self, notes: Sequence[Any], parent: Any = None) -> None:
        super().__init__(parent)
        self.setObjectName("statsPanel")
        split = split_notes(notes)
        outer = QtWidgets.QVBoxLayout(self)
        set_margins(outer, "none", "normal", "none", "none")  # below the table above it
        set_spacing(outer, "normal")
        self.groups = [_Group(g, split.headed) for g in split.groups]
        self.host = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(COLUMN_GAP)  # side by side: the gap columns_for() counts on
        self.grid.setVerticalSpacing(px("section"))  # stacked: a section's room between groups
        if self.groups:
            outer.addWidget(self.host)
        else:
            self.host.setParent(self)
            self.host.hide()
        self.prose = []
        for text in split.prose:
            label = QtWidgets.QLabel(text)
            label.setWordWrap(True)
            label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
            label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            outer.addWidget(label)
            self.prose.append(label)
        self.columns = 0
        self._place(1)

    def text(self) -> str:
        """Every note's words, as shown: for tests and accessibility."""
        lines = [f"{name.text()}: {shown.text()}" for group in self.groups for name, shown in group.rows]
        return "\n".join(lines + [p.text() for p in self.prose])

    def columns_for(self, width: int) -> int:
        """The most columns (at most three) whose groups, each column as wide as its widest, fit in ``width``."""
        n = len(self.groups)
        for cols in range(min(MAX_COLUMNS, n), 1, -1):
            widths = [max(g.natural for g in self.groups[c::cols]) for c in range(cols)]
            if sum(widths) + COLUMN_GAP * (cols - 1) <= width:
                return cols
        return 1

    def _place(self, cols: int) -> None:
        if cols == self.columns:
            return
        self.columns = cols
        for g in self.groups:
            self.grid.removeWidget(g)
        for c in range(MAX_COLUMNS + 1):
            self.grid.setColumnStretch(c, 0)
        flags = QtCore.Qt.AlignmentFlag
        # side by side, each group its own width (a short one beside a long Sample keeps its values near its labels);
        # stacked, each fills the column, so the values line up from group to group
        where = flags.AlignTop | flags.AlignLeft if cols > 1 else flags.AlignTop
        for i, g in enumerate(self.groups):
            self.grid.addWidget(g, i // cols, i % cols, where)
        self.grid.setColumnStretch(cols, 1)  # each column as wide as its widest group, packed to the left

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt's name
        """The groups stacked, whatever the columns now: as narrow as the widest group, and as tall as all of them
        one under another. Given less room than side by side needs, it stacks them (``resizeEvent``); with the
        minimum the same either way, the page around it never has to catch up with a change of columns."""
        hint = super().minimumSizeHint()
        if not self.groups:
            return hint
        least = max(g.minimumSizeHint().width() for g in self.groups)
        margins = self.layout().contentsMargins()
        stacked = (margins.top() + margins.bottom() + sum(g.minimumSizeHint().height() for g in self.groups)
                   + self.grid.verticalSpacing() * (len(self.groups) - 1)
                   + sum(self.layout().spacing() + p.minimumSizeHint().height() for p in self.prose))
        return QtCore.QSize(min(hint.width(), least), max(hint.height(), stacked))

    def changeEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.FontChange and self.groups:  # zoomed: measure again, and the columns
            for g in self.groups:
                g.setFont(self.font())
                g.refresh()
            self.columns = 0
            self._place(self.columns_for(self.width()))

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
        super().resizeEvent(event)
        if self.groups:
            self._place(self.columns_for(event.size().width()))
