"""The Guide's buttons stay in view however short its dock: Open…, Show me, Back, Next and Start over sit under the
part that scrolls (the step's text and the list of steps), and the reload button beside a long case-study title."""

import os

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pymetrix_guide import register  # noqa: E402
from pymetrix_guide.app import create_panel  # noqa: E402
from PySide6.QtCore import QRect  # noqa: E402

from pymetrix.commands import Session  # noqa: E402
from pymetrix.extension import AppInterface  # noqa: E402

register()


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _panel():
    session = Session()
    interface = AppInterface(lambda: session, lambda: None, lambda cls, initial: None, session.run, lambda *_: None)
    return create_panel(interface)


@pytest.mark.parametrize("size", [(300, 260), (260, 420), (420, 600)])
def test_the_step_buttons_are_always_in_view(app, look, size):
    """In view whatever the font and style: where Back, Next and Start over are too wide to sit side by side (Windows'
    style), Start over goes on a row of its own rather than the buttons scrolling sideways in a dock 300 wide."""
    panel = _panel()
    panel.resize(*size)
    panel.show()
    app.processEvents()
    choice = panel.choice
    if choice.count():
        choice.setCurrentIndex(choice.count() - 1)
        app.processEvents()
    assert not panel.scroll.isAncestorOf(panel.next_button)  # outside the part that scrolls
    shown = panel.rect()
    strip = panel.button_strip
    sideways = strip.horizontalScrollBar().isVisible()  # a dock narrower than the buttons: they scroll sideways
    for button in (panel.open_button, panel.show_button, panel.back_button, panel.next_button, panel.restart_button,
                   panel.findChild(QtWidgets.QToolButton, "guide-rescan")):
        box = QRect(button.mapTo(panel, button.rect().topLeft()), button.size())
        assert button.isVisible() and shown.top() <= box.top() and box.bottom() <= shown.bottom(), (button.text(), box.getRect())
        assert sideways or shown.contains(box), (button.text(), box.getRect(), shown.getRect())
    if size[0] >= 300:
        assert not sideways
    assert panel.scroll.isAncestorOf(panel.text) and panel.scroll.isAncestorOf(panel.steps)  # these scroll
    panel.close()
