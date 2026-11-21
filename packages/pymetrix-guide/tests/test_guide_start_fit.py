"""The Guide panel docked beside the app's Start window: the Start window moves and shrinks to stay inside the
workspace's visible area, never under the panel, its buttons in full (tests/test_start_window.py has the rest)."""

import os

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pymetrix_guide import register  # noqa: E402

from pymetrix.commands import Session  # noqa: E402

register()


@pytest.mark.parametrize("workspace", ["all", "report"])
@pytest.mark.parametrize("size", [(1366, 768), (1280, 720), (1920, 1080)])
def test_the_start_window_fits_beside_the_guide_panel(workspace, size):
    from PySide6.QtCore import QRect, QSettings

    from pymetrix.ui.qt import main_window

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope, "pymetrix", "desktop")
    for key in ("workspace", "geometry", "state"):
        settings.remove(key)
    settings.setValue("workspace/current", workspace)
    window = main_window(Session())
    try:
        window.resize(*size)
        window.show()
        guide = window.plugin_docks["Guide"]
        guide.setVisible(True)
        for _ in range(5):
            app.processEvents()
        sub = window.start_page
        view = window.mdi.viewport()
        assert view.rect().contains(sub.geometry())
        panel = QRect(guide.mapTo(window, guide.rect().topLeft()), guide.size())
        placed = QRect(sub.mapTo(window, sub.rect().topLeft()), sub.size())
        assert not placed.intersects(panel)  # not under the Guide panel
        scroll = sub.widget()
        for button in sub.buttons.values():
            corner = button.mapTo(scroll.viewport(), button.rect().topLeft())
            assert scroll.viewport().rect().contains(QRect(corner, button.size())), button.text()
        demos = sub.lists["Demo projects"]
        assert all(demos.contentsRect().contains(e.geometry()) for e in demos.entries)  # wrapped to the list's width
        assert not scroll.horizontalScrollBar().isVisible()
    finally:
        window.close()
        for key in ("workspace", "geometry", "state"):
            settings.remove(key)
