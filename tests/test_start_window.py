"""The Start window fits the workspace it opens in (off-screen Qt): at 1366 by 768, 1280 by 720 and 1920 by 1080, in
every workspace, it lies wholly inside the workspace's visible area (never under the Objects, Report or code
panels), centred, its text and buttons in full, and the demo list wraps its entries to its width, with no sideways
scroll bar.
It follows the area as panels come and go until the user moves it. The chart gallery link: shown, and opening the
notebook, where the file is there (a checkout); left out where it is not (an installed copy)."""

import os

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("qtconsole")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import _fit_check as fc  # noqa: E402

from pymetrix.commands import Session  # noqa: E402
from pymetrix.ui.qt import BREAK  # noqa: E402
from pymetrix.ui.workspaces import WORKSPACES  # noqa: E402

SCREENS = ((1366, 768), (1280, 720), (1920, 1080))


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _settings():
    from PySide6.QtCore import QSettings

    return QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope, "pymetrix", "desktop")


def _settle():
    for _ in range(5):
        _app().processEvents()


@pytest.fixture
def opened():
    """open(workspace, width, height): the app's window as a first run opens it in that workspace, shown at that
    size; closed and its settings forgotten afterwards."""
    from pymetrix.ui.qt import main_window

    _app()
    windows = []
    settings = _settings()
    kept = {key: settings.value(key) for key in settings.allKeys()}

    def open_in(workspace, width, height):
        for key in ("workspace", "geometry", "state"):  # a first run: no arrangement saved by an earlier window
            _settings().remove(key)
        _settings().setValue("workspace/current", workspace)
        window = main_window(Session())
        window.resize(width, height)
        window.show()
        _settle()
        windows.append(window)
        return window

    yield open_in
    for window in windows:
        window.close()
    settings.clear()  # as before the test (closing a window saves its arrangement)
    for key, value in kept.items():
        settings.setValue(key, value)


def _shown_in_full(widget, sub, workspace):
    """``widget``'s rectangle, in the Start window's scroll area's viewport, lies inside it: nothing of it cut off.
    Across always (everything on the page wraps to its width); down unless the workspace (``workspace``, its viewport's
    rectangle) has not the height for it: the tallest the Start window can be there, its content scrolled to the
    top, ends above the widget's bottom (a large font in a short workspace: the page then scrolls)."""
    from PySide6.QtCore import QPoint, QRect

    from pymetrix.ui.fit import VIEW_MARGIN
    from pymetrix.ui.qt import START_LEAST

    scroll = sub.widget()
    view = scroll.viewport()
    shown = QRect(widget.mapTo(view, QPoint(0, 0)), widget.size())
    if shown.left() < 0 or shown.right() > view.rect().right():
        return False
    if view.rect().contains(shown):
        return True
    tallest = max(workspace.height() - 2 * VIEW_MARGIN, min(START_LEAST[1], workspace.height()))
    bottom = widget.mapTo(scroll.widget(), QPoint(0, widget.height())).y()  # on the page
    return bottom > tallest - (sub.height() - view.height())


def _what_holds_it_wide(window, top=5):
    """Why the Start window's page is wider than its viewport: the sizes, the page layout's margins and spacing, the
    widgets with the largest least widths, and the style, font and pixel ratio (a sideways scroll on Windows CI only
    is told from its log)."""
    app = QtWidgets.QApplication.instance()
    sub = window.start_page
    scroll = sub.widget()
    page = scroll.widget()
    layout = page.layout()
    m = layout.contentsMargins()
    widgets = sorted(page.findChildren(QtWidgets.QWidget), key=lambda w: -w.minimumSizeHint().width())
    tops = []
    for widget in [w for w in widgets if w.isVisibleTo(page)][:top]:
        name = widget.objectName() or (widget.text()[:40] if hasattr(widget, "text") else "")
        tops.append(f"{type(widget).__name__} {name!r} min {widget.minimumSizeHint().width()}"
                    f" minW {widget.minimumWidth()} hint {widget.sizeHint().width()} at {widget.width()}")
    font = app.font()
    return (f"workspace {window.mdi.viewport().width()}, window {sub.width()}, viewport {scroll.viewport().width()},"
            f" vbar {scroll.verticalScrollBar().isVisible()} {scroll.verticalScrollBar().width()},"
            f" page {page.width()} min {page.minimumWidth()} minHint {page.minimumSizeHint().width()}"
            f" hint {page.sizeHint().width()} layout min {layout.totalMinimumSize().width()},"
            f" margins {m.left()},{m.top()},{m.right()},{m.bottom()} spacing {layout.spacing()};"
            f" widest: {'; '.join(tops)};"
            f" style {app.style().name()}, font {font.family()} {font.pointSizeF()}pt,"
            f" ratio {window.devicePixelRatioF()}")


def _check_fits(window, in_full=True):
    """What is wrong with the Start window in ``window`` as it is laid out now; ``in_full``: its text and buttons
    are to be seen without scrolling (the workspace has room for them)."""
    found = []
    sub = window.start_page
    view = window.mdi.viewport().rect()
    g = sub.geometry()
    if not view.contains(g):
        found.append(f"Start window {g.getRect()} outside the workspace's viewport {view.getRect()}")
    if abs((g.left() - view.left()) - (view.right() - g.right())) > 1:
        found.append(f"not centred across: {g.getRect()} in {view.getRect()}")
    found += fc.problems(sub)
    scroll = sub.widget()
    if scroll.horizontalScrollBar().isVisible():
        found.append("the Start window scrolls sideways: " + _what_holds_it_wide(window))
    page = scroll.widget()
    text = fc.layout_widget(page.layout(), 0)  # not itemAt: see _fit_check._layout_widgets
    for widget in [text, *sub.buttons.values(), *([sub.gallery] if sub.gallery is not None else [])]:
        if in_full and not _shown_in_full(widget, sub, view):
            found.append(f"{fc._name(widget)} not shown in full")
    demos = sub.lists["Demo projects"]
    inside = demos.contentsRect()
    for row, entry in enumerate(demos.entries):
        rect = entry.geometry()
        if rect.left() < inside.left() or rect.right() > inside.right():
            found.append(f"demo {row} wider than the list: {rect.getRect()} in {inside.getRect()}")
        if entry.height() < entry.heightForWidth(entry.width()):
            found.append(f"demo {row}: its wrapped text cut off ({entry.height()} < {entry.heightForWidth(entry.width())})")
        if entry.toolTip() != entry.text().replace(BREAK, ""):  # less its places to break
            found.append(f"demo {row}: the tooltip does not hold the full text")
    return found


@pytest.mark.parametrize("width,height", SCREENS)
@pytest.mark.parametrize("workspace", [w.key for w in WORKSPACES])
def test_the_start_window_fits_the_workspace_it_opens_in(opened, workspace, width, height):
    window = opened(workspace, width, height)
    assert window.workspace == workspace and window.start_page is not None
    assert _check_fits(window) == []


@pytest.fixture(params=fc.LOOKS)
def look(request):
    """This machine's font and style, a wider font, and Windows' style with it (see _fit_check.look)."""
    _app()
    with fc.look(request.param) as name:
        yield name


@pytest.mark.parametrize("width,height", SCREENS)
def test_the_start_window_follows_the_workspace_as_panels_change(look, opened, width, height):
    """Opened in Report (the narrowest: Objects and History on the left, Report on the right), then every other
    workspace, the Code panels, and the window made smaller and larger: it fits each time, with a wider font and
    Windows' style too (a demo's name, policy_and_nonlinear, wider than the list at 1024 by 600 with all the panels;
    the ways to start in two columns where the room was a few pixels short of them; at 1024 by 600, where the
    workspace is about 200 pixels wide, a button's or the "About this window" toggle's text on one line, wider than
    that with DejaVu Sans a third bigger or Windows' Segoe UI, made the window scroll sideways: they wrap now)."""
    window = opened("report", width, height)
    assert _check_fits(window) == []
    for ws in WORKSPACES:
        window.switch_workspace(ws.key)
        _settle()
        assert _check_fits(window) == [], ws.key
    window.code_action.setChecked(True)
    _settle()
    assert _check_fits(window) == []
    for size in ((1024, 600), (width, height)):  # 1024 by 600 with every panel: little room, so it scrolls
        window.resize(*size)
        _settle()
        assert _check_fits(window, in_full=size != (1024, 600)) == [], size


def test_its_natural_size_is_its_largest_and_it_scrolls_when_the_workspace_is_small(opened):
    from pymetrix.ui.qt import START_LEAST, START_SIZE

    window = opened("all", 1920, 1080)
    sub = window.start_page
    assert sub.width() == START_SIZE[0] and sub.height() <= START_SIZE[1]  # room to spare: no bigger than it needs
    window.mdi.setMinimumSize(1, 1)
    # a workspace smaller than its least size, one way or both, and narrower than the page's longest word: it never
    # scrolls sideways, its content as wide as the workspace (a sideways scroll bar on Windows CI only, all panels open)
    for size in ((330, 240), (500, 200), (300, 500), (150, 500)):
        window.mdi.setFixedSize(*size)
        _settle()
        view = window.mdi.viewport().rect()
        assert view.width() < START_LEAST[0] or view.height() < START_LEAST[1], view
        assert view.contains(sub.geometry()), size  # the whole viewport at most, never beyond it
        assert sub.width() >= min(START_LEAST[0], view.width()) and sub.height() >= min(START_LEAST[1], view.height())
        assert fc.problems(sub) == [], size
        assert sub.widget().verticalScrollBar().isVisible(), size  # what does not fit scrolls
        scroll = sub.widget()
        assert not scroll.horizontalScrollBar().isVisible(), _what_holds_it_wide(window)
        assert scroll.widget().width() == scroll.viewport().width(), size


def test_once_the_user_moves_it_it_stays_where_they_put_it(opened):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    window = opened("explore", 1366, 768)
    sub = window.start_page
    fitted = sub.geometry()
    window.resize(1280, 720)  # untouched: it follows
    _settle()
    assert sub.geometry() != fitted and _check_fits(window) == []
    title = QPointF(sub.width() / 2, 8)  # a press on its title bar: the user is moving it
    for kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease):
        QtWidgets.QApplication.sendEvent(sub, QMouseEvent(kind, title, sub.mapToGlobal(title), Qt.MouseButton.LeftButton,
                                                          Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    sub.move(5, 7)
    window.resize(1920, 1080)
    _settle()
    assert sub.pos().x() == 5 and sub.pos().y() == 7  # left where the user put it


def test_maximized_it_is_left_alone(opened):
    window = opened("all", 1366, 768)
    sub = window.start_page
    sub.showMaximized()
    window.resize(1280, 720)
    _settle()
    assert sub.isMaximized() and sub.keep_fitted.touched


def test_the_chart_gallery_link_opens_the_notebook_where_it_is_there(opened, monkeypatch, tmp_path):
    from pymetrix.ui import qt

    notebook = tmp_path / "chart_gallery.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(qt, "chart_gallery_file", lambda: notebook)
    opening = []
    monkeypatch.setattr(qt, "open_file", lambda path: opening.append(path) or True)
    window = opened("report", 1366, 768)
    link = window.start_page.gallery
    assert link is not None and link.isVisible() and "Chart gallery" in link.text() and link.toolTip() == str(notebook)
    assert "examples/gallery" not in link.text()  # no repository path to read
    link.linkActivated.emit("gallery")
    assert opening == [notebook]
    assert window.start_page is not None and window.start_page.isVisible()  # the Start window stays
    for size in SCREENS:  # shown whatever the window's size (under the buttons, not at the foot of the window)
        window.resize(*size)
        _settle()
        assert _shown_in_full(link, window.start_page, window.mdi.viewport().rect()), size


def test_without_the_notebook_there_is_no_gallery_line(opened, monkeypatch):
    from pymetrix.ui import qt

    monkeypatch.setattr(qt, "chart_gallery_file", lambda: None)
    window = opened("all", 1366, 768)
    sub = window.start_page
    assert sub.gallery is None
    texts = [w.text() for w in sub.widget().widget().findChildren(QtWidgets.QLabel)]
    assert not any("gallery" in t.lower() for t in texts)
    assert _check_fits(window) == []


def test_the_gallery_file_is_found_in_a_checkout_only(tmp_path):
    from pathlib import Path

    from pymetrix.ui import qt

    root = Path(__file__).resolve().parents[1]
    here = root / "examples" / "gallery" / "chart_gallery.ipynb"
    assert here.is_file()  # this is a checkout
    # the checkout's own module (the tests may run against an installed copy, as in CI's wheel job)
    assert qt.chart_gallery_file(root / "src" / "pymetrix" / "ui" / "qt.py") == here
    # an installed copy: the module in site-packages, with no examples beside it (the wheel holds only the packages)
    installed = tmp_path / "lib" / "python3" / "site-packages" / "pymetrix" / "ui" / "qt.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("", encoding="utf-8")
    assert qt.chart_gallery_file(installed) is None
    import tomllib

    wheel = tomllib.loads((here.parents[2] / "pyproject.toml").read_text(encoding="utf-8"))
    assert wheel["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["src/pymetrix", "src/pymetrix_app"]


def test_the_first_run_opens_in_all(opened):
    """A first run (nothing saved) opens in All, which has the Objects and History panels and the code panels but
    leaves the workspace more room than Report; a saved choice is kept."""
    window = opened("all", 1366, 768)
    window.close()
    for key in ("workspace", "geometry", "state"):
        _settings().remove(key)
    from pymetrix.ui.qt import main_window

    fresh = main_window(Session())
    try:
        assert fresh.workspace == "all"
    finally:
        fresh.close()
    _settings().setValue("workspace/current", "report")
    again = main_window(Session())
    try:
        assert again.workspace == "report"
    finally:
        again.close()
