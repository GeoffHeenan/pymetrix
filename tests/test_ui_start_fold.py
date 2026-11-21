"""The Start window puts the demos above the fold on a small, scaled screen (1366×768 at 125%: 1093×614), with the
ways to start beside them and the tour of the window folded."""

import os

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect  # noqa: E402

from pymetrix.commands import Session  # noqa: E402


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _settle(app, sub):
    """Until the Start window has its place: it is fitted to the workspace on a later turn of the event loop (after
    the window's resize reaches the workspace), so a busy machine can need more turns than an idle one."""
    import time

    last, steady = None, 0
    for _ in range(200):
        app.processEvents()
        now = (sub.geometry().getRect(), sub.widget().viewport().size().toTuple())
        steady = steady + 1 if now == last else 0
        if steady >= 4:
            return
        last = now
        time.sleep(0.005)


def test_the_start_window_shows_the_demos_above_the_fold_on_a_scaled_laptop(app, first_run):
    """As a first run opens on that screen (first_run, tests/conftest.py): with the panels' sizes restored from a larger
    window closed earlier in the same test process, the workspace was shorter and the first demo below the fold."""
    from pymetrix.ui.qt import main_window

    window = main_window(Session())
    try:
        window.resize(1093, 614)  # 1366×768 at 125%
        window.show()
        sub = window.start_page
        _settle(app, sub)
        scroll = sub.widget()
        demos_list = sub.lists["Demo projects"]
        view = scroll.viewport().rect()
        first = demos_list.entries[0]
        top_left = first.mapTo(scroll.viewport(), first.rect().topLeft())
        assert view.contains(QRect(top_left, first.size())), "the first demo is in view without scrolling"
        for button in sub.buttons.values():  # and the ways to start
            corner = button.mapTo(scroll.viewport(), button.rect().topLeft())
            assert view.contains(QRect(corner, button.size())), button.text()
        assert not sub.tour.isVisible() and "workspace" in sub.tour.text()  # the tour of the window, folded
        sub.about.click()
        assert sub.tour.isVisibleTo(sub)
    finally:
        window.close()
