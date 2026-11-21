"""Windows that fit whatever room they get: a row of buttons that never overlaps what is above it, and content that
scrolls rather than squashes.

A Qt layout given less room than its minimum lays its widgets out over one another; a window inside the workspace (a
QMdiSubWindow) can be made that small, so its buttons ended up drawn on top of the figure. Here:

- ``ActionRow``: a row of buttons whose minimum width is one button. Buttons that do not fit move into a "»" menu at
  the row's end, in their order, and come back when there is room.
- ``scroll_area(content)``: ``content`` at no less than its own minimum size, with a horizontal scroll bar when the
  window is narrower than that and a vertical one when it is shorter, each on its own. Above the minimum the content
  fills the window as before. Its size hint is the content's, kept to the screen, so a window opens at the same
  size as without it on a normal screen and no bigger than a small one.
- ``scrolled_page(parent)``: a page (outer) holding a scroll area around a body to lay out in.
- ``fit_to_screen(window, width, height)``: resize, but never beyond the screen's available area.
- ``fit_in_viewport(sub, natural, least)`` and ``KeepInViewport``: a workspace window sized and centred in the
  workspace's visible area (never under a panel), and kept so as the area changes until the user moves it.
- ``WrapButton``: a push button whose text wraps where it is narrower than the text on one line, so a large font
  in a narrow window wraps its buttons rather than holding the window wider than its room.
- ``FlowRow``: a row of widgets (radio buttons) that wraps onto more lines where it is narrower than them side by
  side, so its least width is its widest widget's.
- ``WrapLabel``: a word-wrapped label that breaks a word only where the word alone is wider than the label, so its
  least width is a few characters rather than its longest word.
- ``EntryList``: a list of entries (``Entry``) that open on a click or Enter, each wrapped to the list's width, in any
  style: a column of wrapped labels rather than a list view, whose items can keep a width wider than the list.
"""

from __future__ import annotations

import math
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

MORE = "»"
TAB_ROOM, TAB_SHARE = 160, 0.35  # a corner row beside tabs leaves them this much (pixels, share of the width) at least
SCREEN_SHARE = 0.92  # of the screen's free area, at most, for a window's or dialog's opening size


def _available(widget: Any) -> QtCore.QRect | None:
    screen = widget.screen() if widget is not None else None
    screen = screen or QtGui.QGuiApplication.primaryScreen()
    return screen.availableGeometry() if screen is not None else None


def fit_to_screen(window: Any, width: int, height: int) -> None:
    """``window.resize(width, height)``, kept within the screen's free area (a laptop at 1024 by 600, say)."""
    area = _available(window)
    if area is not None:
        width = min(width, int(area.width() * SCREEN_SHARE))
        height = min(height, int(area.height() * SCREEN_SHARE))
    window.resize(max(width, window.minimumWidth()), max(height, window.minimumHeight()))


class FitScrollArea(QtWidgets.QScrollArea):
    """A frameless scroll area whose content fills it down to the content's minimum size and scrolls below it, in
    each direction separately. Its size hint is the content's, so a window opens as it would without it, kept to the
    screen. ``sideways`` False: it never scrolls sideways, the content always as wide as the viewport (its text
    wraps; a word wider than that is cut at the edge), for a page of wrapped text and lists only."""

    def __init__(self, content: Any, parent: Any = None, frame: bool = False, share: float = SCREEN_SHARE,
                 keep_minimum: bool = False, sideways: bool = True) -> None:
        super().__init__(parent)
        self.share, self.keep_minimum, self.sideways = share, keep_minimum, sideways
        self.setWidgetResizable(True)  # fills the viewport, but never below the content's minimum size (Qt keeps that)
        if not frame:
            self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        if not sideways:  # the content's least width is ignored: Qt's scroll area then sizes it to the viewport's width
            policy = content.sizePolicy()
            policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Ignored)
            content.setSizePolicy(policy)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded if sideways
                                          else QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setWidget(content)
        content.setAutoFillBackground(False)  # the window's own background, as before
        self.viewport().setAutoFillBackground(False)
        self._set_minimum: QtCore.QSize | None = None  # the content's minimum size as this area last set it
        self._floor = content.minimumSize()  # as its maker set it (a form's 420 pixels for its hints)
        self._hold_minimum()
        content.installEventFilter(self)
        self.verticalScrollBar().installEventFilter(self)  # shown or hidden: room for it beside the content, or not

    def _hold_minimum(self) -> None:
        """The content's minimum size: its layout's, and no less than its maker's. Qt's scroll area takes an explicit
        minimum size over the layout's (a form given 420 pixels whose fields need 526 with another font or style is
        squashed to 420 rather than scrolled), so the layout's goes in the explicit one too, kept up to date."""
        content = self.widget()
        if content is None:
            return
        if self._set_minimum is not None and content.minimumSize() != self._set_minimum:
            self._floor = content.minimumSize()  # its maker changed it since
        layout = content.layout()
        least = self._floor if layout is None else self._floor.expandedTo(layout.totalMinimumSize())
        if not self.sideways:  # its height only: an explicit least width would hold it wider than the viewport
            least.setWidth(self._floor.width())
        if least != content.minimumSize():
            content.setMinimumSize(least)
        self._set_minimum = content.minimumSize()

    def eventFilter(self, watched: Any, event: Any) -> bool:
        """The content's layout changed (a field shown, a longer choice, another font, a style's polish on a tab's
        first showing): its minimum size may have grown. Qt's scroll area looks again only when it is resized, so it
        would keep the content at the old width with no scroll bar, squashed, and the window would keep its old least
        size; here the content's minimum, the scroll bars and this area's own size hints follow at once."""
        if watched is self.verticalScrollBar() and event.type() in (QtCore.QEvent.Type.Show, QtCore.QEvent.Type.Hide):
            if self.keep_minimum and self.sideways:  # a dialog's form: its least width with or without the bar
                self.updateGeometry()
            return super().eventFilter(watched, event)
        if watched is self.widget() and event.type() == QtCore.QEvent.Type.LayoutRequest:
            self._hold_minimum()
            self.updateGeometry()  # the window's layout asks again: a dialog's least size grows with its form
            QtCore.QCoreApplication.sendEvent(self, QtCore.QEvent(QtCore.QEvent.Type.LayoutRequest))  # the scroll bars
        return super().eventFilter(watched, event)

    def sizeHint(self) -> QtCore.QSize:
        content = self.widget()
        if content is None:
            return super().sizeHint()
        hint = content.sizeHint().expandedTo(content.minimumSizeHint()).expandedTo(content.minimumSize())
        frame = 2 * self.frameWidth()
        size = QtCore.QSize(hint.width() + frame, hint.height() + frame)
        limit = self._limit()
        if limit is not None:  # a long form opens at the screen's height, and scrolls
            if size.height() > limit.height() or self._bar_room():  # it opens at the limit, shorter than it would be, so it scrolls: room for
                # the bar (only when it scrolls at any height, it had none: an Equations list with its note and button
                # opened FitSystem a little too tall, and the bar then pushed a sideways scroll bar into view)
                size.setWidth(size.width() + self.verticalScrollBar().sizeHint().width())
            size = size.boundedTo(limit)
        return size

    def _bar_room(self) -> int:
        """The vertical scroll bar's width while it shows beside a form that may not scroll sideways for it: the form
        opened shorter than its content (the screen's share, or its window's own limit on a top-level window's height),
        and without this room its content, at its least width, was a bar's width wider than the viewport, so a sideways
        bar showed too (System of equations, Equation for each selected series at 1366 by 768)."""
        bar = self.verticalScrollBar()
        if not self.sideways or not bar.isVisibleTo(self):
            return 0
        return bar.sizeHint().width()

    def _limit(self) -> QtCore.QSize | None:
        area = _available(self)
        return None if area is None else QtCore.QSize(int(area.width() * SCREEN_SHARE), int(area.height() * self.share))

    def minimumSizeHint(self) -> QtCore.QSize:
        bar = self.style().pixelMetric(QtWidgets.QStyle.PixelMetric.PM_ScrollBarExtent)
        least = QtCore.QSize(4 * bar, 4 * bar)
        content = self.widget()
        if self.keep_minimum and content is not None:  # a dialog keeps its least size where the screen has room
            frame = 2 * self.frameWidth()
            size = content.minimumSizeHint().expandedTo(content.minimumSize())
            size = QtCore.QSize(size.width() + frame + self._bar_room(), size.height() + frame)
            limit = self._limit()
            least = least.expandedTo(size if limit is None else size.boundedTo(limit))
        return least


def scroll_area(content: Any, parent: Any = None, frame: bool = False, share: float = SCREEN_SHARE,
                keep_minimum: bool = False, sideways: bool = True) -> FitScrollArea:
    """``content`` in a ``FitScrollArea``. ``share``: of the screen's height, at most, for its size hint;
    ``keep_minimum``: no smaller than the content's minimum while the screen has room for it (a dialog's form: its
    least size as without the scroll area on a normal screen, scrolling on a small one); ``sideways`` False: never
    a sideways scroll bar, the content as wide as the viewport (see ``FitScrollArea``)."""
    return FitScrollArea(content, parent, frame, share, keep_minimum, sideways)


def scrolled_page(parent: Any = None) -> tuple[Any, Any]:
    """(page, body): ``page`` holds a scroll area around ``body``; lay the content out in ``body`` and hand ``page``
    on. The body's descendants are the page's, so ``page.findChildren`` still finds its buttons."""
    page = QtWidgets.QWidget(parent)
    outer = QtWidgets.QVBoxLayout(page)
    outer.setContentsMargins(0, 0, 0, 0)
    body = QtWidgets.QWidget()
    page.scroll = scroll_area(body)
    outer.addWidget(page.scroll)
    page.body = body
    return page, body


VIEW_MARGIN = 12  # pixels between a window fitted to the workspace and the workspace's edges, where there is room


def fit_in_viewport(sub: Any, natural: QtCore.QSize, least: QtCore.QSize, margin: int = VIEW_MARGIN) -> bool:
    """Size and place a workspace window (a QMdiSubWindow) inside its area's visible viewport, centred: at its
    ``natural`` size where there is room, shrunk to the room (less a margin) where there is not, but no smaller than
    ``least`` unless the viewport itself is smaller, when it takes the whole viewport (its content then scrolls).
    It never extends past the viewport, so never under a panel. False if the area is not laid out yet."""
    mdi = sub.mdiArea()
    view = mdi.viewport().rect() if mdi is not None else QtCore.QRect()
    if view.width() < 50 or view.height() < 50:
        return False
    sizes = []
    for nat, low, room in ((natural.width(), least.width(), view.width()), (natural.height(), least.height(), view.height())):
        size = min(nat, room - 2 * margin)
        size = max(size, min(low, room))  # below its least size only when the viewport is smaller still
        sizes.append(max(min(size, room), 1))
    width, height = sizes
    sub.setGeometry((view.width() - width) // 2, (view.height() - height) // 2, width, height)
    return True


class KeepInViewport(QtCore.QObject):
    """Keeps a workspace window fitted to its area's viewport (``fit_in_viewport``) as the area is laid out and
    resized (panels shown, docked or dragged; the main window resized), until the user takes it over: a click on its
    title bar or an edge (to move or resize it), maximizing or minimizing it. ``natural()`` gives its natural size,
    asked each time (it may depend on the font)."""

    def __init__(self, sub: Any, natural: Any, least: QtCore.QSize, margin: int = VIEW_MARGIN) -> None:
        super().__init__(sub)
        self.sub, self.natural, self.least, self.margin = sub, natural, least, margin
        self.touched = False
        self.viewport = sub.mdiArea().viewport()
        sub.installEventFilter(self)
        self.viewport.installEventFilter(self)
        self.fit()

    def fit(self) -> bool:
        """Fit it now, unless the user has taken it over; False if not done."""
        if self.touched:
            return False
        try:
            return fit_in_viewport(self.sub, self.natural(), self.least, self.margin)
        except RuntimeError:  # closed
            return False

    def eventFilter(self, watched: Any, event: Any) -> bool:
        kind = event.type()
        if watched is self.sub and kind in (QtCore.QEvent.Type.MouseButtonPress, QtCore.QEvent.Type.MouseButtonDblClick,
                                            QtCore.QEvent.Type.NonClientAreaMouseButtonPress):
            self.touched = True  # its title bar or an edge (clicks on its content go to the content)
        elif watched is self.sub and kind == QtCore.QEvent.Type.WindowStateChange:
            # maximized or minimized (the event, not windowStateChanged: PySide6 6.6 cannot deliver that signal)
            if self.sub.windowState() & (QtCore.Qt.WindowState.WindowMaximized | QtCore.Qt.WindowState.WindowMinimized):
                self.touched = True
        elif not self.touched and ((watched is self.viewport and kind == QtCore.QEvent.Type.Resize)
                                   or (watched is self.sub and kind == QtCore.QEvent.Type.Show)):
            # after the area has handled the event itself: on a resize it may move its windows
            # (no context-object overload: PySide6 before 6.7 has none; fit() is safe once the window is closed)
            QtCore.QTimer.singleShot(0, self.fit)
        return super().eventFilter(watched, event)


class ActionRow(QtWidgets.QWidget):
    """A row of buttons (or labels) that never asks for more width than one button: those that do not fit move,
    in order, into a "»" menu at the row's end. Use it where a QHBoxLayout of buttons was (``addWidget``,
    ``addStretch`` and ``addSpacing`` as there); widgets added after ``addStretch`` sit at the right, as before."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._layout = QtWidgets.QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.items: list[Any] = []
        self._stretch_at: int | None = None  # how many items come before the stretch
        self._folded: list[Any] = []  # hidden by this row for want of room (not by their owner)
        self.more = QtWidgets.QToolButton(self)
        self.more.setText(MORE)
        self.more.setToolTip("more: the buttons there is no room for")
        self.more.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.more.setAutoRaise(False)
        self.more_menu = QtWidgets.QMenu(self.more)
        self.more_menu.aboutToShow.connect(self._fill_menu)
        self.more.setMenu(self.more_menu)
        self.more.setVisible(False)
        self._layout.addWidget(self.more)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)

    # the QHBoxLayout calls a button row used
    def addWidget(self, widget: Any, stretch: int = 0) -> None:
        self._layout.insertWidget(self._layout.indexOf(self.more), widget, stretch)
        self.items.append(widget)
        self._relayout()

    def addStretch(self, stretch: int = 1) -> None:
        if self._stretch_at is None:
            self._layout.insertStretch(self._layout.indexOf(self.more), max(stretch, 1))
            self._stretch_at = len(self.items)

    def addSpacing(self, size: int) -> None:
        self._layout.insertSpacing(self._layout.indexOf(self.more), size)

    def setSpacing(self, spacing: int) -> None:
        self._layout.setSpacing(spacing)

    def setContentsMargins(self, *margins: Any) -> None:
        self._layout.setContentsMargins(*margins)

    def folded(self) -> list[Any]:
        """The widgets in the "»" menu now."""
        return list(self._folded)

    def _wanted(self) -> list[Any]:
        return [w for w in self.items if w in self._folded or not w.isHidden()]

    def _widths(self, widgets: list[Any], least: bool = False) -> int:
        """Their width side by side: at their preferred sizes, or (``least``) squeezed to their minimum sizes."""
        spacing = max(self._layout.spacing(), 0)
        width = sum((w.minimumSizeHint() if least else w.sizeHint()).expandedTo(w.minimumSize()).width() for w in widgets)
        return width + spacing * max(len(widgets) - 1, 0)

    def _margins(self) -> int:
        m = self._layout.contentsMargins()
        return m.left() + m.right()

    def _height(self) -> int:
        heights = [w.sizeHint().height() for w in self._wanted()] + [self.more.sizeHint().height()]
        return max(heights) + self._vertical_margins()

    def sizeHint(self) -> QtCore.QSize:
        width = self._widths(self._wanted()) + self._margins()
        tabs = self.parentWidget()
        if isinstance(tabs, QtWidgets.QTabWidget) and tabs.width() > 0:  # a tab widget's corner: room left for the tabs
            room = max(TAB_ROOM, int(tabs.width() * TAB_SHARE))
            width = max(min(width, tabs.width() - room), self.minimumSizeHint().width())
        return QtCore.QSize(width, self._height())

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self.more.sizeHint().width() + self._margins(), self._height())

    def _vertical_margins(self) -> int:
        m = self._layout.contentsMargins()
        return m.top() + m.bottom()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._relayout(event.size().width())

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self._relayout()

    def event(self, event: Any) -> bool:
        """A button's size hint changed (its text, the font, a zoom, the style): fold again from the sizes as they
        are now, not as they were when the row was last resized."""
        done = super().event(event)
        if event.type() == QtCore.QEvent.Type.LayoutRequest and self.isVisible():
            self._relayout()
        return done

    def _relayout(self, width: int | None = None) -> None:
        width = self.width() if width is None else width
        wanted = self._wanted()
        room = width - self._margins()
        if not self.isVisible() and width <= 0:  # not laid out yet: show everything
            keep = wanted
        elif self._widths(wanted, least=True) <= room:
            keep = wanted
        else:
            spacing = max(self._layout.spacing(), 0)
            room -= self.more.sizeHint().width() + spacing
            keep = []
            for w in wanted:  # in order: the first buttons stay, the rest fold
                if self._widths(keep + [w], least=True) > room:
                    break
                keep.append(w)
        folded = [w for w in wanted if w not in keep]
        for w in wanted:
            w.setVisible(w in keep)
        self._folded = folded
        self.more.setVisible(bool(folded))
        self.updateGeometry()

    def _fill_menu(self) -> None:
        """The folded buttons as menu entries: a button with a menu as a submenu, a toggle as a checkable entry, a
        label as its text."""
        menu = self.more_menu
        menu.clear()
        for w in self._folded:
            if isinstance(w, QtWidgets.QLabel):
                action = menu.addAction(w.text().strip())
                action.setEnabled(False)
                continue
            if not isinstance(w, QtWidgets.QAbstractButton):
                continue
            text = w.text()
            sub = w.menu() if isinstance(w, (QtWidgets.QPushButton, QtWidgets.QToolButton)) else None
            if sub is not None:
                action = menu.addMenu(sub)
                action.setText(text)
            else:
                action = menu.addAction(text)
                if w.isCheckable():
                    action.setCheckable(True)
                    action.setChecked(w.isChecked())
                action.triggered.connect(lambda _=False, b=w: b.click())
            action.setToolTip(w.toolTip())
            action.setEnabled(w.isEnabled())
        menu.setToolTipsVisible(True)


class WrapButton(QtWidgets.QPushButton):
    """A push button whose text wraps onto more lines where it is given less width than the text takes on one line
    (a large font in a narrow workspace), instead of holding its parent at that width: its minimum width is its
    longest word's, and it takes the height its wrapped text needs. Where it has the room it is a plain push button
    (the same size hint and drawing)."""

    def __init__(self, text: str = "", parent: Any = None) -> None:
        super().__init__(text, parent)
        policy = self.sizePolicy()
        policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Preferred)  # may be narrower than its one-line text
        policy.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Minimum)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    _FLAGS = QtCore.Qt.TextFlag.TextShowMnemonic.value | QtCore.Qt.TextFlag.TextWordWrap.value

    def _around(self, text: QtCore.QSize) -> QtCore.QSize:
        """The button's size around a block of text of size ``text``, as its style draws it."""
        option = QtWidgets.QStyleOptionButton()
        self.initStyleOption(option)
        size = self.style().sizeFromContents(QtWidgets.QStyle.ContentsType.CT_PushButton, option, text, self)
        return size.expandedTo(self.minimumSize())

    def _padding(self) -> int:
        """The pixels across the style adds around the text (wide text, so no least width of the style's counts)."""
        wide = 10000
        return self._around(QtCore.QSize(wide, self.fontMetrics().height())).width() - wide

    def _text_block(self, width: int) -> QtCore.QSize:
        """The text's size wrapped to ``width`` pixels."""
        metrics = self.fontMetrics()
        return metrics.boundingRect(QtCore.QRect(0, 0, max(width, 1), 100000), self._FLAGS, self.text()).size()

    def minimumSizeHint(self) -> QtCore.QSize:
        metrics = self.fontMetrics()
        word = max((metrics.horizontalAdvance(w) for w in self.text().replace("&", "").split()), default=0)
        width = min(self._around(QtCore.QSize(word, metrics.height())).width(), self.sizeHint().width())
        return QtCore.QSize(width, self.heightForWidth(width))

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        if width >= self.sizeHint().width():
            return self.sizeHint().height()
        block = self._text_block(width - self._padding())
        return max(self._around(QtCore.QSize(block.width(), block.height())).height(), self.sizeHint().height())

    def paintEvent(self, event: Any) -> None:
        if self.width() >= self.sizeHint().width():
            super().paintEvent(event)
            return
        painter = QtWidgets.QStylePainter(self)
        option = QtWidgets.QStyleOptionButton()
        self.initStyleOption(option)
        text, option.text = option.text, ""
        painter.drawControl(QtWidgets.QStyle.ControlElement.CE_PushButton, option)  # its frame, without the text
        room = self.style().subElementRect(QtWidgets.QStyle.SubElement.SE_PushButtonContents, option, self)
        flags = self._FLAGS | QtCore.Qt.AlignmentFlag.AlignCenter.value
        painter.drawItemText(room, flags, option.palette, self.isEnabled(), text, QtGui.QPalette.ColorRole.ButtonText)
        painter.end()


class FlowRow(QtWidgets.QWidget):
    """A row of widgets (a field's radio buttons) that wraps onto more lines, in order and to the left, where it is
    narrower than they are side by side: its least width is its widest widget's, not the whole row's. Where it has the
    room it is one line, laid out as a QHBoxLayout with a stretch at its end lays it out. A row of five long choices
    held a dialog's form wider than a laptop's screen with a large font (FitSystem's standard errors and Fit
    cointegrating regression's methods, where every character is a 16-pixel box: Windows CI with the font a third
    bigger), so the form scrolled sideways."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.items: list[Any] = []
        policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def addWidget(self, widget: Any) -> None:
        hidden = widget.testAttribute(QtCore.Qt.WidgetAttribute.WA_WState_ExplicitShowHide) and widget.isHidden()
        widget.setParent(self)
        widget.setVisible(not hidden)  # shown with the row, as a layout shows the widgets added to it
        self.items.append(widget)
        self.updateGeometry()
        self._place()

    def _shown(self) -> list[Any]:
        return [w for w in self.items if not w.isHidden()]

    @staticmethod
    def _hint(widget: Any) -> QtCore.QSize:
        return widget.sizeHint().expandedTo(widget.minimumSize()).expandedTo(widget.minimumSizeHint())

    def _spacing(self, orientation: QtCore.Qt.Orientation) -> int:
        """The style's spacing between two such widgets, as a layout with no spacing of its own uses."""
        horizontal = orientation == QtCore.Qt.Orientation.Horizontal
        metric = (QtWidgets.QStyle.PixelMetric.PM_LayoutHorizontalSpacing if horizontal
                  else QtWidgets.QStyle.PixelMetric.PM_LayoutVerticalSpacing)
        spacing = self.style().pixelMetric(metric, None, self)
        if spacing < 0:
            shown = self._shown()
            kind = shown[0].sizePolicy().controlType() if shown else QtWidgets.QSizePolicy.ControlType.DefaultType
            spacing = self.style().layoutSpacing(kind, kind, orientation, None, self)
        return max(spacing, 0)

    def _lines(self, width: int) -> list[list[Any]]:
        """The shown widgets in lines no wider than ``width`` (a widget wider than that on a line of its own)."""
        across = self._spacing(QtCore.Qt.Orientation.Horizontal)
        lines: list[list[Any]] = []
        used = 0
        for widget in self._shown():
            w = self._hint(widget).width()
            if lines and lines[-1] and used + across + w <= width:
                lines[-1].append(widget)
                used += across + w
            else:
                lines.append([widget])
                used = w
        return lines

    def _margins(self) -> QtCore.QMargins:
        return self.contentsMargins()

    def _height(self, lines: list[list[Any]]) -> int:
        down = self._spacing(QtCore.Qt.Orientation.Vertical)
        heights = [max(self._hint(w).height() for w in line) for line in lines]
        m = self._margins()
        return sum(heights) + down * max(len(heights) - 1, 0) + m.top() + m.bottom()

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        m = self._margins()
        return self._height(self._lines(width - m.left() - m.right()))

    def sizeHint(self) -> QtCore.QSize:
        shown = self._shown()
        m = self._margins()
        across = self._spacing(QtCore.Qt.Orientation.Horizontal)
        width = sum(self._hint(w).width() for w in shown) + across * max(len(shown) - 1, 0) + m.left() + m.right()
        return QtCore.QSize(width, self._height([shown]) if shown else m.top() + m.bottom())

    def minimumSizeHint(self) -> QtCore.QSize:
        shown = self._shown()
        m = self._margins()
        width = max((self._hint(w).width() for w in shown), default=0) + m.left() + m.right()
        return QtCore.QSize(width, self._height([shown]) if shown else m.top() + m.bottom())

    def _place(self) -> None:
        """Each shown widget at its size hint, in lines as wide as the row, each line's widgets centred down it."""
        m = self._margins()
        room = self.rect().adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        across, down = self._spacing(QtCore.Qt.Orientation.Horizontal), self._spacing(QtCore.Qt.Orientation.Vertical)
        y = room.top()
        for line in self._lines(room.width()):
            height = max(self._hint(w).height() for w in line)
            x = room.left()
            for widget in line:
                hint = self._hint(widget)
                place = QtCore.QRect(x, y + (height - hint.height()) // 2, hint.width(), hint.height())
                widget.setGeometry(QtWidgets.QStyle.visualRect(self.layoutDirection(), self.rect(), place))
                x += hint.width() + across
            y += height + down

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._place()

    def event(self, event: Any) -> bool:
        """A widget shown, hidden or resized (its text, the font, the style): Qt tells a parent with no layout so."""
        done = super().event(event)
        if event.type() == QtCore.QEvent.Type.LayoutRequest:
            self.updateGeometry()
            self._place()
        return done


class WrapLabel(QtWidgets.QLabel):
    """A word-wrapped label (plain or rich text) that breaks a word only where that word alone is wider than the label:
    its least width is a few characters (``LEAST_CHARS``), not its longest word. Where every word fits it is a plain
    QLabel (the same height for width and drawing, links and all); narrower, its text is laid out and drawn with words
    broken where they must be. A label's own least width is its longest word, so a large font, a long name or a title
    in a narrow window held the window wider than its room: on Windows CI, where every character is a 12-pixel box (no
    fonts installed), ``identification`` in a demo's description and the Start window's title ``PyMetrix`` (18-pixel
    boxes in a heading) held the Start window wider than a narrow workspace, which never scrolls sideways."""

    LEAST_CHARS = 6  # its least width, in average characters (less where its longest word is narrower)

    def __init__(self, text: str = "", parent: Any = None) -> None:
        super().__init__(text, parent)
        self.setWordWrap(True)

    def _margins(self) -> QtCore.QSize:
        """What the label adds around its text: its contents margins and ``margin()`` on each side."""
        m = self.contentsMargins()
        return QtCore.QSize(m.left() + m.right() + 2 * self.margin(), m.top() + m.bottom() + 2 * self.margin())

    def _fits(self, width: int) -> bool:
        """Every word fits in ``width``: the label's own wrapping and drawing."""
        return width >= super().minimumSizeHint().width()

    def _document(self, width: int) -> QtGui.QTextDocument:
        """Its text laid out ``width`` pixels wide (its text, less the margins), a word broken only where it must be."""
        document = QtGui.QTextDocument()
        document.setDefaultFont(self.font())
        document.setDocumentMargin(0)
        option = document.defaultTextOption()
        option.setWrapMode(QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        document.setDefaultTextOption(option)
        rich = self.textFormat() == QtCore.Qt.TextFormat.RichText or (
            self.textFormat() == QtCore.Qt.TextFormat.AutoText and QtCore.Qt.mightBeRichText(self.text()))
        if rich:
            document.setHtml(self.text())
        else:
            document.setPlainText(self.text())
        document.setTextWidth(max(width, 1))
        return document

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        if self._fits(width):
            return super().heightForWidth(width)
        margins = self._margins()
        return math.ceil(self._document(width - margins.width()).size().height()) + margins.height()

    def minimumSizeHint(self) -> QtCore.QSize:
        own = super().minimumSizeHint()  # its longest word across, a line down
        least = self.LEAST_CHARS * self.fontMetrics().averageCharWidth() + self._margins().width()
        return QtCore.QSize(min(own.width(), least), own.height())

    def paintEvent(self, event: Any) -> None:
        if self._fits(self.width()):
            super().paintEvent(event)
            return
        room = self.contentsRect().adjusted(self.margin(), self.margin(), -self.margin(), -self.margin())
        document = self._document(room.width())
        painter = QtGui.QPainter(self)
        painter.setClipRect(room)
        painter.translate(room.left(), room.top() + max((room.height() - math.ceil(document.size().height())) // 2, 0))
        context = QtGui.QAbstractTextDocumentLayout.PaintContext()
        palette = QtGui.QPalette(self.palette())
        palette.setColor(QtGui.QPalette.ColorRole.Text, palette.color(self.foregroundRole()))
        context.palette = palette
        document.documentLayout().draw(painter, context)
        painter.end()


class Entry(WrapLabel):
    """One entry of an ``EntryList``: its text wrapped to the width it is given (``WrapLabel``: the same in every style,
    a word broken only where it alone is wider than the entry), with its full text in the tooltip. A click, Enter or
    Space opens it (``activated``); Up and Down move to the entry above and below."""

    activated = QtCore.Signal()

    def __init__(self, text: str, tip: str, shaded: bool = False, parent: Any = None) -> None:
        super().__init__(text, parent)
        self.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.setToolTip(tip)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover)
        self.setContentsMargins(6, 3, 6, 3)  # deliberate, off metrics' grid: a list item's padding, as a list view draws it
        self.setAccessibleDescription(tip)
        self.shaded = shaded  # every other entry: wrapped entries told apart
        self._pressed = False

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._pressed = True
            self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        pressed, self._pressed = self._pressed, False
        if pressed and event.button() == QtCore.Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            event.accept()
            self.activated.emit()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: Any) -> None:
        key = event.key()
        if key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter, QtCore.Qt.Key.Key_Space):
            event.accept()
            self.activated.emit()
            return
        owner = self.parentWidget()
        if key in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down) and isinstance(owner, EntryList):
            event.accept()
            owner.step(self, -1 if key == QtCore.Qt.Key.Key_Up else 1)
            return
        super().keyPressEvent(event)

    def event(self, event: Any) -> bool:
        if event.type() in (QtCore.QEvent.Type.HoverEnter, QtCore.QEvent.Type.HoverLeave):
            self.update()
        return super().event(event)

    def focusInEvent(self, event: Any) -> None:
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event: Any) -> None:
        super().focusOutEvent(event)
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QtGui.QPainter(self)
        palette = self.palette()
        if self.hasFocus() or self.underMouse():
            shade = QtGui.QColor(palette.color(QtGui.QPalette.ColorRole.Highlight))
            shade.setAlpha(70 if self.hasFocus() else 35)
            painter.fillRect(self.rect(), shade)
        elif self.shaded:
            painter.fillRect(self.rect(), palette.color(QtGui.QPalette.ColorRole.AlternateBase))
        if self.hasFocus():
            option = QtWidgets.QStyleOptionFocusRect()
            option.initFrom(self)
            option.backgroundColor = palette.color(QtGui.QPalette.ColorRole.Base)
            self.style().drawPrimitive(QtWidgets.QStyle.PrimitiveElement.PE_FrameFocusRect, option, painter, self)
        painter.end()
        super().paintEvent(event)


class EntryList(QtWidgets.QFrame):
    """A list of entries that wrap their text to the list's width: a plain column of ``Entry`` labels, laid out by
    height for width like any wrapped label, so no style's item margins or cached item sizes can make an entry wider
    than the list (a list view's items kept a width wider than the list on Windows once the window narrowed).
    ``entries`` in order; ``activated(row)`` when one is opened."""

    activated = QtCore.Signal(int)

    def __init__(self, items: list[tuple[str, str]], parent: Any = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setBackgroundRole(QtGui.QPalette.ColorRole.Base)
        self.setAutoFillBackground(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.entries: list[Entry] = []
        for row, (text, tip) in enumerate(items):
            entry = Entry(text, tip, shaded=row % 2 == 1, parent=self)
            entry.activated.connect(lambda r=row: self.activated.emit(r))
            layout.addWidget(entry)
            self.entries.append(entry)

    def step(self, entry: Entry, by: int) -> None:
        """Focus the entry ``by`` rows from ``entry`` (if there is one), scrolled into view."""
        row = self.entries.index(entry) + by
        if not 0 <= row < len(self.entries):
            return
        target = self.entries[row]
        target.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
        owner = self.parentWidget()
        while owner is not None and not isinstance(owner, QtWidgets.QScrollArea):
            owner = owner.parentWidget()
        if owner is not None:
            owner.ensureWidgetVisible(target)
