"""Checks that a window fits the room it is given (off-screen Qt): its widgets never drawn over one another, nothing
cut off without a way to scroll to it, and content bigger than the window inside a scroll area whose scroll bars
show. Used by tests/test_window_fit.py on every kind of window and dialog."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from PySide6 import QtCore, QtWidgets

SMALL = ((400, 300), (300, 220), (230, 150))  # a window made small: narrow and short together
NARROW, SHORT = (230, 700), (900, 150)  # one direction at a time


def _layout_widgets(layout: Any) -> list[Any]:
    """The widgets a layout places, through its nested layouts: its parent's child widgets that it or one of its nested
    layouts holds. Not through ``itemAt``: with PySide6 6.6 each item it hands back stays referenced by the layout's
    wrapper, and when Qt deletes that item (its widget deleted) the wrapper stays mapped to the freed address: a
    widget made there later comes back to Python as a ``QWidgetItem`` (no ``installEventFilter`` on a console's
    viewport, in a later test of the same process)."""
    parent = layout.parentWidget()
    if parent is None:
        return []
    layouts = [layout, *layout.findChildren(QtWidgets.QLayout)]
    children = parent.findChildren(QtWidgets.QWidget, options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly)
    return [w for w in children if any(inner.indexOf(w) >= 0 for inner in layouts)]


def layout_widget(layout: Any, index: int) -> Any:
    """The widget at ``index`` in ``layout`` (None where that is a layout or spacing), without ``itemAt`` (see
    ``_layout_widgets``)."""
    parent = layout.parentWidget()
    children = [] if parent is None else parent.findChildren(
        QtWidgets.QWidget, options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly)
    return next((w for w in children if layout.indexOf(w) == index), None)


def _name(w: Any) -> str:
    text = w.text() if hasattr(w, "text") and callable(w.text) else ""
    return f"{type(w).__name__}({text!r})" if text else type(w).__name__


def problems(root: Any) -> list[str]:
    """What does not fit in ``root`` as it is laid out now: overlapping siblings, widgets cut off by their parent,
    scroll areas whose content is bigger than them with no scroll bar showing, a tab widget's corner over its tabs."""
    QtWidgets.QApplication.processEvents()
    found = []
    widgets = [root, *root.findChildren(QtWidgets.QWidget)]
    for parent in widgets:
        if not parent.isVisible():
            continue
        if isinstance(parent, QtWidgets.QAbstractScrollArea) and not isinstance(parent, QtWidgets.QScrollArea):
            continue  # tables, lists, text: they scroll their own content
        layout = parent.layout()
        if layout is not None:
            placed = [w for w in _layout_widgets(layout) if w.isVisible() and w.parentWidget() is parent]
            area = parent.rect()
            for i, a in enumerate(placed):
                g = a.geometry()
                if g.width() > 0 and g.height() > 0 and not area.contains(g):
                    found.append(f"{_name(a)} cut off in {_name(parent)}: {g.getRect()} outside {area.getRect()}")
                for b in placed[i + 1:]:
                    both = g.intersected(b.geometry())
                    if both.width() > 0 and both.height() > 0:
                        found.append(f"{_name(a)} over {_name(b)} in {_name(parent)}")
        if isinstance(parent, QtWidgets.QTabWidget):
            bar, corner = parent.tabBar(), parent.cornerWidget()
            if corner is not None and corner.isVisible() and bar.isVisible():
                both = bar.geometry().intersected(corner.geometry())
                if both.width() > 0 and both.height() > 0:
                    found.append(f"corner {_name(corner)} over the tabs of {_name(parent)}")
                if corner.geometry().left() < 0 or corner.geometry().right() > parent.width():
                    found.append(f"corner {_name(corner)} cut off in {_name(parent)}")
        if isinstance(parent, QtWidgets.QScrollArea) and parent.widget() is not None:
            content, view = parent.widget(), parent.viewport()
            least = content.minimumSizeHint().expandedTo(content.minimumSize())
            if least.width() > view.width() and not parent.horizontalScrollBar().isVisible():
                found.append(f"{_name(content)} wider ({least.width()}) than its scroll area ({view.width()}), no scroll bar")
            if least.height() > view.height() and not parent.verticalScrollBar().isVisible():
                found.append(f"{_name(content)} taller ({least.height()}) than its scroll area ({view.height()}), no scroll bar")
            if content.width() < least.width() or content.height() < least.height():
                found.append(f"{_name(content)} squashed below its minimum in its scroll area")
    return found


def _switches_views(w: Any) -> bool:
    """A tab widget, or a result window's list of views (pymetrix.ui.view_switcher), which takes the same calls."""
    return isinstance(w, QtWidgets.QTabWidget) or type(w).__name__ == "ViewSwitcher"


def tab_widgets(root: Any) -> list[Any]:
    return [t for t in [root, *root.findChildren(QtWidgets.QWidget)] if _switches_views(t)]


def check_sizes(window: Any, sizes: Any = SMALL, resize: Any = None, every_tab: bool = True) -> list[str]:
    """``problems`` at each size (``resize(w, h)``, default ``window.resize``), on every tab of every tab widget."""
    resize = resize or window.resize
    found = []
    for w, h in sizes:
        resize(w, h)
        QtWidgets.QApplication.processEvents()
        tabs = tab_widgets(window) if every_tab else []
        pages = [(t, i) for t in tabs for i in range(t.count())] or [(None, None)]
        for t, i in pages:
            if t is not None:
                if not t.isVisible():
                    continue
                t.setCurrentIndex(i)
            found += [f"{w}x{h}{'' if t is None else ' ' + repr(t.tabText(i))}: {p}" for p in problems(window)]
    return found


LOOKS = ("default", "wide", "windows")
WIDE = 1.35  # a font a third bigger: text as wide as with Windows' Segoe UI or CI's DejaVu Sans
WIDE_FAMILY = "DejaVu Sans"  # its family where installed (Linux CI's own): the same text widths on every such machine
WINDOWS_BUTTONS = "QPushButton { min-width: 96px; }"  # push buttons as wide as Windows 11's style draws them


@contextmanager
def look(name: str) -> Any:
    """The app as it looks on another machine, for the length of a test, put back afterwards: "default" (as it is),
    "wide" (the font a third bigger, DejaVu Sans where installed) or "windows" (Qt's Windows style and push buttons as wide as Windows 11 draws
    them: layouts that fit with this machine's font and style but not with Windows' failed only on Windows CI)."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    font, style, palette, sheet = app.font(), app.style().name(), app.palette(), app.styleSheet()
    try:
        if name == "wide":  # DejaVu Sans where there is one, not this machine's own sans serif (narrower on some:
            # a Start window that fitted here with Inter scrolled sideways on CI with DejaVu Sans)
            from PySide6.QtGui import QFontDatabase

            wide = app.font()
            if WIDE_FAMILY in QFontDatabase.families():
                wide.setFamily(WIDE_FAMILY)
            wide.setPointSizeF(font.pointSizeF() * WIDE if font.pointSizeF() > 0 else 13)
            app.setFont(wide)
        if name == "windows":
            app.setStyle("Windows")
            app.setStyleSheet(WINDOWS_BUTTONS)
        yield name
    finally:
        if name == "windows":
            app.setStyleSheet(sheet)
            app.setStyle(style)
            app.setPalette(palette)
        app.setFont(font)
