"""The window in front is obvious, and the app's colours come from one place (pymetrix.ui.theme).

The front window's title bar takes the palette's accent (Highlight, the title in HighlightedText) and the others'
titles are muted, following activation; checked on the roles and on the pixels painted. The theme's colour roles read
at 4.5:1 or more on a light and a dark palette (the ratio itself checked against published values), and no colour is
written in ``pymetrix.ui`` outside the theme module."""

import os
import re
from pathlib import Path

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtCore, QtGui  # noqa: E402

UI = Path(__file__).resolve().parents[1] / "src" / "pymetrix" / "ui"
HEX = re.compile(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3}(?:[0-9a-fA-F]{2})?)?\b")
ALLOWED: dict[str, set[str]] = {}  # file: the colour literals it may keep (none: chart marks are theme.CHART too)


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _palette(window, text, base, placeholder):
    pal = QtGui.QPalette()
    R = QtGui.QPalette.ColorRole
    for role, colour in {R.Window: window, R.WindowText: text, R.Text: text, R.Base: base, R.Button: window,
                         R.ButtonText: text, R.Highlight: "#2a82da", R.HighlightedText: "#ffffff"}.items():
        pal.setColor(role, QtGui.QColor(colour))
    pal.setColor(R.PlaceholderText, placeholder)
    return pal


LIGHT = ("#efefef", "#000000", "#ffffff", QtGui.QColor(0, 0, 0, 128))  # Qt's own default light palette (Fusion)
DARK = ("#2d2d2d", "#e6e6e6", "#1e1e1e", QtGui.QColor(230, 230, 230, 128))  # a dark theme, half-opaque placeholder


def _area(app, kinds=(QtWidgets.QMdiSubWindow, QtWidgets.QMdiSubWindow)):
    from pymetrix.ui.active_window import follow_active_window

    area = QtWidgets.QMdiArea()
    area.resize(900, 600)
    follow_active_window(area)
    subs = []
    for i, kind in enumerate(kinds):
        sub = kind()
        sub.setWidget(QtWidgets.QLabel(f"window {i}"))
        sub.setWindowTitle(f"w{i} — Fitted equation")
        area.addSubWindow(sub)
        sub.setGeometry(20 + 120 * i, 20 + 60 * i, 420, 260)
        sub.show()
        subs.append(sub)
    area.show()
    app.processEvents()
    return area, subs


def _bar_pixel(sub):
    """The colour painted at the left end of the title's label area (the bar's background, clear of the text)."""
    from pymetrix.ui.active_window import _title_option

    opt, _height = _title_option(sub)
    label = sub.style().subControlRect(QtWidgets.QStyle.ComplexControl.CC_TitleBar, opt,
                                       QtWidgets.QStyle.SubControl.SC_TitleBarLabel, sub)
    image = sub.grab().toImage()
    ratio = image.devicePixelRatio()
    return QtGui.QColor(image.pixel(int((label.left() + 2) * ratio), int(label.center().y() * ratio)))


@pytest.mark.parametrize("style", ["Fusion", "Windows"])
def test_the_front_windows_title_differs_and_follows_activation(app, style):
    from pymetrix.ui.active_window import title_colours, title_roles

    R = QtGui.QPalette.ColorRole

    class SheetSub(QtWidgets.QMdiSubWindow):  # as the app's spreadsheet and editor windows: a subclass
        def closeEvent(self, event):  # noqa: N802
            super().closeEvent(event)

    old = app.style().name()
    app.setStyle(style)
    try:
        area, (first, second) = _area(app, (QtWidgets.QMdiSubWindow, SheetSub))
        try:
            area.setActiveSubWindow(first)
            app.processEvents()
            assert title_roles(first) == (R.Highlight, R.HighlightedText)
            assert title_roles(second) == (R.Window, R.PlaceholderText)
            front_bg, front_text = title_colours(first)
            back_bg, back_text = title_colours(second)
            assert front_text != back_text and front_bg != back_bg
            assert front_text == first.palette().color(R.HighlightedText)
            assert _bar_pixel(first) == front_bg and _bar_pixel(second) == back_bg  # what is painted, in this style
            area.setActiveSubWindow(second)  # the other comes to the front: the roles follow
            app.processEvents()
            assert title_roles(second) == (R.Highlight, R.HighlightedText)
            assert title_roles(first) == (R.Window, R.PlaceholderText)
            assert _bar_pixel(second) == front_bg and _bar_pixel(first) == back_bg
            from pymetrix.ui.theme import contrast

            assert contrast(*title_colours(second)) >= 4.5 and contrast(*title_colours(first)) >= 4.5
        finally:
            area.close()
    finally:
        app.setStyle(old)


def test_a_window_added_without_activation_has_a_muted_title(app):
    from pymetrix.ui.active_window import title_colours
    from pymetrix.ui.theme import muted

    area, (first,) = _area(app, (QtWidgets.QMdiSubWindow,))
    try:
        later = area.addSubWindow(QtWidgets.QLabel("later"))  # Qt makes the window: watched as it arrives
        later.setWindowTitle("later — Dataset")
        later.setGeometry(300, 200, 300, 200)
        later.show()
        area.setActiveSubWindow(first)
        app.processEvents()
        assert title_colours(later)[1] == muted(later.palette()) and _bar_pixel(later) == title_colours(later)[0]
    finally:
        area.close()


def test_the_focused_panel_is_marked(app):
    from pymetrix.ui.active_window import follow_panel_focus

    window = QtWidgets.QMainWindow()
    window.setCentralWidget(QtWidgets.QTextEdit())
    docks = []
    for name in ("Objects", "History"):
        dock = QtWidgets.QDockWidget(name, window)
        dock.setWidget(QtWidgets.QLineEdit())
        window.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        docks.append(dock)
    watcher = follow_panel_focus(window)
    window.show()
    window.activateWindow()
    try:
        watcher.focus_changed(None, docks[1].widget())
        assert watcher.focused is docks[1]
        image = docks[1].grab().toImage()
        highlight = docks[1].palette().color(QtGui.QPalette.ColorRole.Highlight)
        assert QtGui.QColor(image.pixel(int(image.width() / 2), 0)) == highlight
        watcher.focus_changed(docks[1].widget(), window.centralWidget())  # the workspace: no panel marked
        assert watcher.focused is None
        assert QtGui.QColor(docks[1].grab().toImage().pixel(int(image.width() / 2), 0)) != highlight
    finally:
        window.close()


def test_contrast_matches_published_ratios(app):
    from pymetrix.ui.theme import contrast

    assert contrast("#000000", "#ffffff") == pytest.approx(21.0)
    assert contrast("#777777", "#ffffff") == pytest.approx(4.48, abs=0.01)  # WebAIM's checker: #777 on white
    assert contrast("#767676", "#ffffff") == pytest.approx(4.54, abs=0.01)  # the lightest grey that passes on white


@pytest.mark.parametrize("colours", [LIGHT, DARK], ids=["light", "dark"])
def test_theme_roles_read_on_light_and_dark_palettes(app, colours):
    from pymetrix.ui import theme

    pal = _palette(*colours)
    assert theme.is_dark(pal) == (colours is DARK)
    window, base = pal.color(QtGui.QPalette.ColorRole.Window), pal.color(QtGui.QPalette.ColorRole.Base)
    for role in ("error", "warning", "success", "link", "muted"):
        colour = theme.colour(role, pal)
        assert theme.contrast(colour, window) >= 4.5 and theme.contrast(colour, base) >= 4.5, role
        assert theme.css(role, pal) == f"color: {colour.name()};"
    assert theme.colour("error", pal).name() == theme.ROLES["error"][1 if colours is DARK else 0]  # no adjusting needed
    # the palette's own placeholder text, as seen on the window, is too faint: the muted colour is derived instead
    seen = theme._over(pal.color(QtGui.QPalette.ColorRole.PlaceholderText), window)
    assert theme.contrast(seen, window) < 4.5
    assert theme.muted_css(pal) == f"color: {theme.muted(pal).name()};"
    muted = theme.muted(pal)  # still secondary: between the text and the window colour
    assert theme.contrast(muted, window) < theme.contrast(pal.color(QtGui.QPalette.ColorRole.WindowText), window)


def test_readable_moves_a_colour_only_as_far_as_needed(app):
    from pymetrix.ui import theme

    faint = theme.readable("#e5a50a", ["#ffffff"])  # the amber once used for warnings: 2.1:1 on white
    assert theme.contrast("#e5a50a", "#ffffff") < 2.5 and 4.5 <= theme.contrast(faint, "#ffffff") < 5.2
    assert theme.readable("#1864ab", ["#ffffff"]).name() == "#1864ab"  # already readable: unchanged


def test_no_colour_literals_in_the_ui_outside_the_theme():
    found = {}
    for path in sorted(UI.glob("*.py")):
        if path.name == "theme.py":
            continue
        text = path.read_text(encoding="utf-8")
        literals = set(HEX.findall(text)) - ALLOWED.get(path.name, set())
        if literals:
            found[path.name] = sorted(literals)
        if "palette(placeholder-text)" in text:  # muted text: theme.muted_css(), readable on every palette
            found.setdefault(path.name, []).append("palette(placeholder-text)")
    assert found == {}


def test_the_app_marks_its_front_window(app):
    pytest.importorskip("qtconsole")
    from pymetrix.demos import demo
    from pymetrix.ui.active_window import title_roles
    from pymetrix.ui.qt import main_window

    w = main_window(demo("consumption"))
    w.show()
    try:
        w.show_object("data")
        w.show_object("dynamic")
        app.processEvents()
        data_sub, fit_sub = (next(s for s in w.mdi.subWindowList() if s.windowTitle().startswith(n)) for n in ("data", "dynamic"))
        R = QtGui.QPalette.ColorRole
        assert title_roles(fit_sub)[1] == R.HighlightedText and title_roles(data_sub)[1] == R.PlaceholderText
        w.mdi.setActiveSubWindow(data_sub)
        app.processEvents()
        assert title_roles(data_sub)[1] == R.HighlightedText and title_roles(fit_sub)[1] == R.PlaceholderText
        assert w.objects.currentItem().text(0) == "data"  # the Objects panel already follows the front window
    finally:
        w.state["session"].dirty = False
        w.close()
