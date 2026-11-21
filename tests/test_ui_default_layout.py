"""The default layout on a laptop screen (off-screen Qt): at 1366 by 768 and 1280 by 720 the result windows keep most
of the height in every workspace (the Script, Immediate and Problems panels about a fifth), and the Objects panel
shows each object's type in full, a long name giving way. A layout the user arranged is kept as it is; Reset layout
goes back to the defaults."""

import os

import pytest

from pymetrix import demo

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("qtconsole")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

WORKSPACES = ("all", "explore", "estimate", "model", "forecast", "report")


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _settle(n=20):
    for _ in range(n):
        _app().processEvents()


@pytest.fixture
def fresh_window():
    """The main window as on a first run: no saved geometry, layout or workspace arrangement."""
    _app()
    from PySide6.QtCore import QSettings

    settings = QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope, "pymetrix", "desktop")
    for key in ("state", "geometry", "workspace"):
        settings.remove(key)
    from pymetrix.ui.qt import main_window

    windows = []

    def make(width, height):
        w = main_window(demo("consumption"))
        w.resize(width, height)
        w.show()
        _settle()
        windows.append(w)
        return w

    yield make
    for w in windows:
        w.state["session"].dirty = False
        w.close()
    for key in ("state", "geometry", "workspace"):
        settings.remove(key)


def _type_column_in_view(w):
    """The Type column wholly inside the Objects panel and as wide as its longest entry."""
    tree, header = w.objects, w.objects.header()
    left = header.sectionViewportPosition(1)
    return left >= 0 and left + header.sectionSize(1) <= tree.viewport().width() and header.sectionSize(1) >= tree.sizeHintForColumn(1)


@pytest.fixture(params=("default", "wide"))
def font(request):
    """The app's font for the test: as it is, or a third bigger (text as wide as with Windows' Segoe UI or CI's DejaVu
    Sans), put back afterwards."""
    app = _app()
    before = app.font()
    if request.param == "wide":
        wide = app.font()
        wide.setPointSizeF(before.pointSizeF() * 1.35 if before.pointSizeF() > 0 else 13)
        app.setFont(wide)
    yield request.param
    app.setFont(before)


@pytest.mark.parametrize("size", [(1366, 768), (1280, 720)])
def test_every_workspace_gives_the_results_most_of_the_height(fresh_window, size, font):
    w = fresh_window(*size)
    for key in WORKSPACES:
        w.switch_workspace(key)
        _settle()
        share = w.mdi.height() / w.height()
        assert share >= 0.55, f"{key}: the result windows get {share:.0%} of the height"
        assert _type_column_in_view(w), key


def test_a_workspace_left_before_its_panels_were_fitted_is_fitted_when_back(fresh_window, font):
    """A workspace left while its panels were still in their default arrangement (here, before the window had fitted
    them: on a slow machine the fitting waits for the event loop) is fitted again when it comes back, keeping which
    panels show; an arrangement the user made is kept as it is."""
    from pymetrix.ui.qt import main_window

    w = fresh_window(1280, 720)
    late = main_window(demo("consumption"))
    try:
        late.resize(1280, 720)
        late.show()
        late.switch_workspace("explore")  # at once: All's panels not yet fitted to the new size
        late.switch_workspace("all")
        _settle()
        assert late.mdi.height() / late.height() >= 0.55 and _type_column_in_view(late)
        assert late.layout_fit["pending"]
        assert abs(late.objects.width() - w.objects.width()) <= 2  # as in a window fitted in good time
    finally:
        late.state["session"].dirty = False
        late.close()


def test_a_long_name_gives_way_to_the_type(fresh_window):
    w = fresh_window(1366, 768)
    session = w.state["session"]
    session.objects["a_very_long_result_name_that_does_not_fit_the_panel"] = session.objects["dynamic"]
    w.switch(session)
    _settle()
    assert _type_column_in_view(w)
    item = next(w.objects.topLevelItem(i) for i in range(w.objects.topLevelItemCount())
                if w.objects.topLevelItem(i).text(0).startswith("a_very_long"))
    assert item.toolTip(0) == item.text(0)  # in full where it is shortened


def test_a_users_layout_is_kept_and_reset_layout_restores_the_default(fresh_window):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    w = fresh_window(1366, 768)
    w.switch_workspace("estimate")
    _settle()
    problems = next(d for d in w.findChildren(QtWidgets.QDockWidget) if d.objectName() == "problems")
    press = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(5, 5), QPointF(5, 5), Qt.MouseButton.LeftButton,
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QtWidgets.QApplication.sendEvent(w, press)  # the user takes a panel's edge ...
    w.resizeDocks([problems], [420], Qt.Orientation.Vertical)  # ... and makes Problems tall
    _settle()
    tall = problems.height()
    assert tall > 350 and not w.layout_fit["pending"]
    w.resize(1280, 720)
    _settle()
    w.switch_workspace("explore")
    _settle()
    w.switch_workspace("estimate")  # its arrangement as the user left it
    _settle()
    assert problems.height() > 300 and w.mdi.height() / w.height() < 0.55
    reset = next(a for a in w.view_menu.actions() if a.text().replace("&", "") == "Reset layout")
    reset.trigger()
    _settle()
    assert w.mdi.height() / w.height() >= 0.55 and w.layout_fit["pending"]
