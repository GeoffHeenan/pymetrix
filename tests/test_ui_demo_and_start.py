"""A demo project opens with its headline results in view, each on the view that shows the work, selected in Objects,
with a note of what is open."""

import os

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pymetrix.commands import Session  # noqa: E402
from pymetrix.demos import demos, headline  # noqa: E402


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_every_demo_says_what_it_shows_first():
    from pymetrix.views import view_sections

    for name in demos():
        windows, note = headline(name)
        assert 1 <= len(windows) <= 3 and note, name
    from pymetrix import demo

    s = demo("consumption")  # the objects and views named are there (the fast demos; nk_estimation takes seconds)
    for obj, view in headline("consumption")[0]:
        assert view in [t for t, _ in view_sections(s[obj])]
    s = demo("forecast_round")
    for obj, view in headline("forecast_round")[0]:
        assert view in [t for t, _ in view_sections(s[obj])]


@pytest.mark.parametrize("name", ["forecast_round", "consumption", "policy_and_nonlinear"])
def test_opening_a_demo_shows_its_headline_results(app, name):
    from pymetrix.ui.qt import main_window

    window = main_window(Session())
    try:
        window.resize(1366, 768)
        window.show()
        window.open_demo(name)
        app.processEvents()
        windows, note = headline(name)
        titles = [s.windowTitle() for s in window.mdi.subWindowList()]
        assert len(titles) == len(windows)  # the workspace is not empty
        top = window.mdi.activeSubWindow()
        obj, view = windows[-1]
        assert top.windowTitle().startswith(f"{obj} — ") and top.widget().current_title() == view  # on the view that shows it
        assert window.objects.currentItem().text(0) == obj  # and selected in Objects
        assert window.statusBar().currentMessage().startswith(f"Demo {name}: ")
    finally:
        window.state["session"].dirty = False
        window.close()
