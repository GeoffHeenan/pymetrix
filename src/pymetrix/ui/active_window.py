"""The window in front, made obvious: its title bar in the theme's accent colour, the others' titles muted.

Every window of the workspace (a QMdiSubWindow: a result, the dataset, the spreadsheet, the code and model editors)
is watched here. The one in front (``QMdiArea.currentSubWindow``) gets a title bar in the palette's Highlight colour
with its title in HighlightedText, and a thin frame in the same colour; the others a title bar in the window colour,
their titles in the muted colour (theme.muted). The colours are the palette's, so a dark theme gets its own; the bar is
darkened (or lightened) just enough for the title to read at 4.5:1 where the palette's own pair does not.

Why paint, not only set palettes: what each style does with a title bar's palette (Qt 6, CC_TitleBar):

* Fusion fills the front bar with Highlight but writes every title in fixed white or black (a background window's
  title in Text: not muted); the Windows style (QCommonStyle) follows Highlight and HighlightedText;
* windows11 fills both bars with Button and writes the title in WindowText;
* windowsvista draws the system's caption picture and the system's caption text colours: no palette at all.

And a palette set on a QMdiSubWindow reaches the window's contents too (its tables' selection colours, its text). So
the palette roles are applied to the title bar alone, by painting it over the style's own (an event filter: Qt paints
the window, frame and buttons first, then the bar's background and title are painted again), on every style the same.
``QMdiArea.subWindowActivated`` repaints the windows' bars when another comes to the front.

The panel (QDockWidget) that has the keyboard focus gets a thin accent line along its top.
"""

from __future__ import annotations

from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from . import theme
from .icons import view_size

__all__ = ["follow_active_window", "follow_panel_focus", "title_colours", "title_roles"]

FRAME = 2  # pixels: the front window's accent frame, and the focused panel's line
_BUTTONS = ("SC_TitleBarMinButton", "SC_TitleBarMaxButton", "SC_TitleBarCloseButton", "SC_TitleBarNormalButton",
            "SC_TitleBarShadeButton", "SC_TitleBarUnshadeButton", "SC_TitleBarContextHelpButton")


def _is_front(sub: Any) -> bool:
    area = sub.mdiArea()
    return area is not None and area.currentSubWindow() is sub


def title_roles(sub: Any) -> tuple[Any, Any]:
    """The palette roles of ``sub``'s title bar: (background, title). The window in front: (Highlight,
    HighlightedText); any other: (Window, PlaceholderText): muted, as theme.muted makes it readable."""
    R = QtGui.QPalette.ColorRole
    return (R.Highlight, R.HighlightedText) if _is_front(sub) else (R.Window, R.PlaceholderText)


def title_colours(sub: Any) -> tuple[Any, Any]:
    """The colours ``sub``'s title bar is painted in: (background, title), from its palette's roles (title_roles)."""
    pal = sub.palette()
    G, R = QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole
    background_role, text_role = title_roles(sub)
    if background_role == R.Highlight:
        text = pal.color(G, text_role)
        return theme.readable(pal.color(G, background_role), [text]), text  # the bar, not the title, moves to 4.5:1
    return pal.color(G, background_role), theme.muted(pal)


def _int(flag: Any) -> int:
    return int(getattr(flag, "value", flag))


def _title_option(sub: Any) -> tuple[Any, int]:
    """The style option Qt paints ``sub``'s title bar with (QMdiSubWindowPrivate::titleBarOptions), and the bar's
    height: 0 where no bar is drawn."""
    style = sub.style()
    S = QtWidgets.QStyle
    opt = QtWidgets.QStyleOptionTitleBar()
    opt.initFrom(sub)
    opt.subControls = S.SubControl.SC_All
    opt.titleBarFlags = sub.windowFlags()
    opt.icon = sub.windowIcon()
    state = _int(sub.windowState())
    if _is_front(sub):
        opt.state |= S.StateFlag.State_Active
        state |= _int(S.StateFlag.State_Active)
    else:
        opt.state &= ~S.StateFlag.State_Active
    opt.titleBarState = state
    border = 0 if style.styleHint(S.StyleHint.SH_TitleBar_NoBorder, opt, sub) else 4
    height = style.pixelMetric(S.PixelMetric.PM_TitleBarHeight, opt, sub)
    if border:
        height += 8 if sub.isMinimized() else 4
    height -= 2 * border if sub.isMinimized() else border
    opt.rect = QtCore.QRect(border, border, sub.width() - 2 * border, height)
    return opt, height


def _paint_title(sub: Any) -> None:
    """Paint ``sub``'s title bar background and title in title_colours, leaving the style's buttons as they are;
    and the accent frame round the window in front."""
    if (sub.windowFlags() & QtCore.Qt.WindowType.FramelessWindowHint) or sub.isMaximized() or sub.mdiArea() is None:
        return  # no title bar of its own (maximized: its buttons are in the menu bar)
    if sub.mdiArea().viewMode() == QtWidgets.QMdiArea.ViewMode.TabbedView:
        return
    style = sub.style()
    S = QtWidgets.QStyle
    opt, height = _title_option(sub)
    if height <= 0:
        return
    background, text = title_colours(sub)
    bar = opt.rect.adjusted(1, 1, -1, -1)
    keep = QtGui.QRegion()
    for name in _BUTTONS:
        control = getattr(S.SubControl, name)
        rect = style.subControlRect(S.ComplexControl.CC_TitleBar, opt, control, sub)
        if rect.isValid() and rect.intersects(bar):
            keep = keep.united(rect)
    painter = QtGui.QPainter(sub)
    try:
        painter.setClipRegion(QtGui.QRegion(bar).subtracted(keep))
        painter.fillRect(bar, background)
        icon_rect = style.subControlRect(S.ComplexControl.CC_TitleBar, opt, S.SubControl.SC_TitleBarSysMenu, sub)
        if icon_rect.isValid() and not opt.icon.isNull() and opt.titleBarFlags & QtCore.Qt.WindowType.WindowSystemMenuHint:
            size = min(icon_rect.width(), icon_rect.height(), view_size(sub))  # with the text size, as far as the bar has room
            mode = QtGui.QIcon.Mode.Selected if _is_front(sub) else QtGui.QIcon.Mode.Normal  # on Highlight: as a selected row
            opt.icon.paint(painter, QtCore.QRect(0, 0, size, size).translated(icon_rect.center() - QtCore.QPoint(size // 2, size // 2)),
                           QtCore.Qt.AlignmentFlag.AlignCenter, mode)
        label = style.subControlRect(S.ComplexControl.CC_TitleBar, opt, S.SubControl.SC_TitleBarLabel, sub)
        font = QtWidgets.QApplication.font("QMdiSubWindowTitleBar")
        painter.setFont(font)
        painter.setPen(text)
        centred = style.name().lower() in ("fusion", "windows11")
        title = QtGui.QFontMetrics(font).elidedText(sub.windowTitle(), QtCore.Qt.TextElideMode.ElideRight, max(label.width() - 8, 0))
        align = (QtCore.Qt.AlignmentFlag.AlignHCenter if centred else QtCore.Qt.AlignmentFlag.AlignLeft) | QtCore.Qt.AlignmentFlag.AlignVCenter
        painter.drawText(label.adjusted(4, 0, -4, 0), int(align), title)
        if _is_front(sub) and not sub.isMinimized():
            painter.setClipping(False)
            pen = QtGui.QPen(background, FRAME)
            pen.setJoinStyle(QtCore.Qt.PenJoinStyle.MiterJoin)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            half = FRAME / 2
            painter.drawRect(QtCore.QRectF(sub.rect()).adjusted(half, half, -half, -half))
    finally:
        painter.end()


class _Watcher(QtCore.QObject):
    """Paints over the title bars of the area's windows (eventFilter on each) and repaints them on activation."""

    def __init__(self, area: Any) -> None:
        super().__init__(area)
        self.area = area
        self.watched: set[int] = set()
        area.viewport().installEventFilter(self)
        area.subWindowActivated.connect(self.activated)
        for sub in area.subWindowList():
            self.watch(sub)

    def watch(self, sub: Any) -> None:
        if id(sub) not in self.watched:
            self.watched.add(id(sub))
            sub.installEventFilter(self)
            sub.destroyed.connect(lambda *_, key=id(sub): self.watched.discard(key))

    def activated(self, _sub: Any) -> None:
        try:
            for sub in self.area.subWindowList():
                self.watch(sub)
                sub.update()
        except RuntimeError:  # the area is being deleted
            pass

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802
        kind = event.type()
        if kind == QtCore.QEvent.Type.Paint and isinstance(obj, QtWidgets.QMdiSubWindow):
            obj.event(event)  # Qt's own painting first (filters are not called again for this direct call)
            _paint_title(obj)
            return True
        if kind == QtCore.QEvent.Type.ChildPolished and isinstance(event.child(), QtWidgets.QMdiSubWindow):
            self.watch(event.child())  # a window added: watched from its first paint, before it is ever activated
        return False


def follow_active_window(area: Any) -> Any:
    """Make the window in front of ``area`` (a QMdiArea) obvious, and keep it so as windows come and go."""
    watcher = getattr(area, "_pymetrix_front_window", None)
    if watcher is None:
        watcher = area._pymetrix_front_window = _Watcher(area)
    return watcher


class _PanelWatcher(QtCore.QObject):
    """A thin accent line along the top of the panel (QDockWidget) that has the keyboard focus."""

    def __init__(self, window: Any) -> None:
        super().__init__(window)
        self.window = window
        self.focused: Any = None
        self.watched: set[int] = set()
        app = QtWidgets.QApplication.instance()
        app.focusChanged.connect(self.focus_changed)
        window.destroyed.connect(lambda *_: self._stop(app))

    def _stop(self, app: Any) -> None:
        try:
            app.focusChanged.disconnect(self.focus_changed)
        except (RuntimeError, TypeError):
            pass

    def focus_changed(self, _old: Any, new: Any) -> None:
        dock = new
        try:
            while dock is not None and not isinstance(dock, QtWidgets.QDockWidget):
                dock = dock.parentWidget()
            if dock is not None and dock.window() is not self.window:
                dock = None
            if dock is self.focused:
                return
            previous, self.focused = self.focused, dock
            if dock is not None and id(dock) not in self.watched:
                self.watched.add(id(dock))
                dock.installEventFilter(self)
                dock.destroyed.connect(lambda *_, key=id(dock): self.watched.discard(key))
            for each in (previous, dock):
                if each is not None:
                    each.update()
        except RuntimeError:  # a widget is gone
            self.focused = None

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802
        if event.type() == QtCore.QEvent.Type.Paint and obj is self.focused and obj.titleBarWidget() is None:
            obj.event(event)
            painter = QtGui.QPainter(obj)
            try:
                painter.fillRect(QtCore.QRect(0, 0, obj.width(), FRAME), obj.palette().color(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.Highlight))
            finally:
                painter.end()
            return True
        return False


def follow_panel_focus(window: Any) -> Any:
    """Mark the panel of ``window`` (a QMainWindow) that has the keyboard focus with an accent line."""
    watcher = getattr(window, "_pymetrix_focused_panel", None)
    if watcher is None:
        watcher = window._pymetrix_focused_panel = _PanelWatcher(window)
    return watcher
