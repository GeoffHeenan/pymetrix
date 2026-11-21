"""View → Text size (pymetrix.ui.text_size): one setting that scales the whole app's text from the system's font, kept in
the settings, applied at start-up and live; the per-window zoom multiplies it; charts follow on screen while exported
files keep their sizes; and windows and dialogs still fit their screens at every step."""

import os
import warnings

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect  # noqa: E402

import _fit_check as fc  # noqa: E402
from pymetrix import Dataset, demo  # noqa: E402
from pymetrix.commands import COMMANDS, Session  # noqa: E402
from pymetrix.ui import text_size  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def normal_after(app):
    """Each test starts at Normal, with Normal kept, and leaves the app so (a later test's font is the system's)."""
    text_size.apply("Normal")
    text_size.save("Normal")
    normal = app.font().pointSizeF()
    yield normal
    text_size.apply("Normal")
    text_size.save("Normal")
    assert app.font().pointSizeF() == pytest.approx(normal)


@pytest.fixture
def window(app, normal_after):
    from pymetrix.ui.qt import main_window

    win = main_window(demo("consumption"))
    win.resize(1366, 768)
    win.show()
    app.processEvents()
    yield win
    win.state["session"].dirty = False
    win.close()


def _settle(app, n=4):
    for _ in range(n):
        app.processEvents()


def _result(window):
    name = next(n for n, o in window.state["session"].objects.items() if type(o).__name__ != "Dataset")
    window.show_object(name)
    return window.mdi.activeSubWindow().widget()


def test_the_steps_and_their_scales():
    assert text_size.STEPS == ("Smaller", "Normal", "Larger", "Largest")
    assert [text_size.FACTORS[s] for s in text_size.STEPS] == sorted(text_size.FACTORS.values())
    assert text_size.FACTORS["Normal"] == 1
    from pymetrix.errors import SpecificationError

    with pytest.raises(SpecificationError):
        text_size.apply("Huge")


@pytest.mark.parametrize("step", text_size.STEPS)
def test_each_step_reaches_a_dialog_a_panel_and_a_result_window(app, window, normal_after, step):
    """The app's font at the step's scale of the system's; a dialog made now, the Objects panel and an open result
    window (its text, and its chart's surface) all at it."""
    from pymetrix.ui.chart_look import look
    from pymetrix.ui.qt import command_dialog

    view = _result(window)
    window.set_text_size(step)
    _settle(app)
    want = normal_after * text_size.FACTORS[step]
    assert app.font().pointSizeF() == pytest.approx(want)
    assert window.text_size_actions[step].isChecked()
    dialog = command_dialog(next(i.cls for i in COMMANDS.values() if i.cls.__name__ == "FitSARIMA"), window.state["session"])
    try:
        assert dialog.font().pointSizeF() == pytest.approx(want)
        assert dialog.editors["order"].widget.font().pointSizeF() == pytest.approx(want)
    finally:
        dialog.deleteLater()
    assert window.objects.font().pointSizeF() == pytest.approx(want)  # a panel
    view = window.mdi.activeSubWindow().widget()  # drawn again at the new size
    assert view.font().pointSizeF() == pytest.approx(want)
    assert look(view)[1].text_scale == pytest.approx(text_size.FACTORS[step])


def test_the_setting_is_kept_and_applied_as_the_app_starts(app, normal_after):
    from pymetrix.ui.qt import main_window

    first = main_window(Session())
    try:
        first.set_text_size("Larger")
        assert text_size.saved() == "Larger"
    finally:
        first.state["session"].dirty = False
        first.close()
    text_size.apply("Normal")  # as a new process starts: the system's font, the setting still Larger
    assert app.font().pointSizeF() == pytest.approx(normal_after)
    second = main_window(Session())
    try:
        assert text_size.current() == "Larger"
        assert app.font().pointSizeF() == pytest.approx(normal_after * 1.15)
        assert second.text_size_actions["Larger"].isChecked()
    finally:
        second.state["session"].dirty = False
        second.close()


def test_a_setting_that_is_not_a_step_reads_as_normal(app, normal_after):
    from PySide6.QtCore import QSettings

    QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope, "pymetrix", "desktop").setValue(text_size.KEY, "Gigantic")
    assert text_size.saved() == "Normal"


def test_a_live_change_reflows_every_open_window_without_errors(app, window, normal_after):
    """Result, spreadsheet, code editor, model editor and Start windows open, then every step in turn: no error, the
    code editors' fixed-width font follows, and each window still fits (its content scrolls rather than overlaps)."""
    from pymetrix.ui.qt import start_page

    _result(window)
    window.open_sheet("data")
    window.open_editor()
    window.open_model_editor()
    _settle(app)
    editor = next(s.widget().editor for s in window.mdi.subWindowList() if type(s.widget()).__name__ == "EditorWindow")
    code_normal = editor.font().pointSizeF()
    for step in ("Largest", "Smaller", "Larger", "Normal", "Largest"):
        window.set_text_size(step)
        _settle(app)
        assert editor.font().pointSizeF() == pytest.approx(code_normal * text_size.FACTORS[step])
        for sub in window.mdi.subWindowList():
            assert fc.problems(sub.widget()) == [], (step, type(sub.widget()).__name__)
    start = start_page(window, lambda *a: None, lambda *a: None, lambda *a: None)
    assert fc.check_sizes(start.widget(), ((620, 640), (400, 300)), resize=start.resize) == []
    start.close()


def test_the_zoom_multiplies_the_text_size(app, window, normal_after):
    view = _result(window)
    view.zoom_in()  # 115%
    assert view.font().pointSizeF() == pytest.approx(normal_after * 1.15)
    window.set_text_size("Largest")
    _settle(app)
    view = window.mdi.activeSubWindow().widget()  # the window drawn again keeps its zoom
    assert view.zoom_factor() == pytest.approx(1.15)
    assert view.font().pointSizeF() == pytest.approx(normal_after * 1.3 * 1.15)
    view.zoom_reset()
    assert view.font().pointSizeF() == pytest.approx(normal_after * 1.3)


def test_charts_follow_on_screen_and_files_keep_their_sizes(app, normal_after):
    import _gallery

    from pymetrix.ui.chart_look import look

    figure = _gallery.build("Figure")

    def sizes(fig):
        """Every text's size: titles, labels, tick labels, legends."""
        from matplotlib.text import Text

        fig.draw_without_rendering()  # the ticks as drawn
        return sorted(t.get_fontsize() for t in fig.findobj(Text) if t.get_text() and t.get_visible())

    exported = figure.draw()
    on_screen_normal = figure.draw(*look())
    assert sizes(on_screen_normal) == pytest.approx(sizes(exported))  # at Normal: the theme's own
    text_size.apply("Largest")
    on_screen = figure.draw(*look())
    assert look()[1].text_scale == pytest.approx(1.3)
    # every text by the same scale: the theme's two sizes are scaled, and the half or one point a legend or a bar label
    # takes off them stays as it is (a legend at 10.4 - 0.5 rather than 1.3 x 7.5)
    assert sizes(on_screen) == pytest.approx([s * 1.3 for s in sizes(exported)], abs=0.5 * 0.3 + 1e-9)
    assert sum(sizes(on_screen)) > 1.25 * sum(sizes(exported))
    assert sizes(figure.draw()) == pytest.approx(sizes(exported))  # a file, a copy: the theme's own sizes
    assert on_screen.get_size_inches().tolist() == exported.get_size_inches().tolist()  # the size in inches kept


def test_text_size_has_no_keys_of_its_own_and_the_zoom_keeps_its_keys(window):
    from pymetrix.ui.qt import shortcut_list

    assert all(a.shortcut().isEmpty() for a in window.text_size_actions.values())
    from PySide6.QtGui import QKeySequence

    def native(*keys):  # as the list shows them: the platform's own words (⌘ for Ctrl on a Mac)
        return ", ".join(QKeySequence(k).toString(QKeySequence.SequenceFormat.NativeText) for k in keys)

    rows = {label: keys for _, label, keys in shortcut_list(window)}
    assert rows["Zoom in"] == native("Ctrl+=", "Ctrl++") and rows["Zoom out"] == native("Ctrl+-")
    assert rows["Actual size"] == native("Ctrl+0")
    menu = window.text_size_actions["Normal"].parent()
    assert menu.title().replace("&", "") == "Text size" and [a.text().replace("&", "") for a in menu.actions()] == list(text_size.STEPS)


SCREENS = {"Largest": (1366, 768), "Smaller": (1024, 600), "Normal": (1024, 600)}


@pytest.fixture(params=list(SCREENS))
def sized(request, monkeypatch, app, normal_after):
    """A text size and the screen it is checked on: Largest on a 1366 by 768 laptop, Smaller and Normal on 1024 by
    600; put back to Normal afterwards."""
    from pymetrix.ui import fit

    width, height = SCREENS[request.param]
    monkeypatch.setattr(fit, "_available", lambda widget: QRect(0, 0, width, height))
    text_size.apply(request.param)
    return request.param, QRect(0, 0, width, height)


def test_every_command_dialog_fits_its_screen_at_each_text_size(app, sized):
    """Each dialog's least size and opening size fit the screen; opened, its form scrolls up and down if it must,
    never sideways."""
    from pymetrix.ui.qt import command_dialog

    step, screen = sized
    session = Session()
    session.objects["data"] = Dataset.example("sample_macro")
    too_big, sideways, checked = [], [], 0
    for info in list(COMMANDS.values()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                dialog = command_dialog(info.cls, session)
            except Exception:  # noqa: BLE001 - a dialog that needs more than the sample data; its own tests cover it
                continue
        try:
            name = info.cls.__name__
            least, opens = dialog.minimumSizeHint(), dialog.sizeHint()
            if least.width() > screen.width() or least.height() > screen.height() or opens.height() > screen.height():
                too_big.append((name, least.toTuple(), opens.toTuple()))
            dialog.resize(opens.boundedTo(screen.size()))
            dialog.show()
            _settle(app, 3)
            if dialog.form_scroll.horizontalScrollBar().isVisible():
                sideways.append((name, dialog.size().toTuple()))
            checked += 1
        finally:
            dialog.close()
            dialog.deleteLater()
    assert checked > 100
    assert too_big == [] and sideways == [], step


def test_the_main_window_and_start_window_fit_at_each_text_size(app, sized):
    from pymetrix.ui.qt import main_window

    step, screen = sized
    win = main_window(Session())
    try:
        win.resize(screen.width(), screen.height())
        win.show()
        _settle(app)
        least = win.minimumSizeHint()
        assert least.width() <= screen.width() and least.height() <= screen.height(), (step, least.toTuple())
        start = win.start_page
        assert fc.check_sizes(start.widget(), ((620, 640), (400, 300), (300, 220)), resize=start.resize) == []
        _settle(app)
        assert not start.widget().horizontalScrollBar().isVisible()  # the Start window never scrolls sideways
        assert fc.problems(win.report_view) == []
    finally:
        win.state["session"].dirty = False
        win.close()


def test_the_objects_icons_follow_the_text_size_and_the_title_bar_still_paints_its_icon(app, window):
    """The kinds' icons (pymetrix.ui.icons) scale with the text size: the Objects panel's rows, Find anything's list and
    the tabs follow a change at once, and the front window's title bar still paints its icon, at the new size as far
    as the bar has room. The expected size is the style's own small icon size times the step's factor."""
    from PySide6 import QtGui

    from pymetrix.ui import icons
    from pymetrix.ui.active_window import _title_option, title_colours
    from pymetrix.ui.palette import palette_dialog

    S = QtWidgets.QStyle
    small = window.objects.style().pixelMetric(S.PixelMetric.PM_SmallIconSize, None, window.objects)
    window.show_object("dynamic")
    _settle(app)
    sub = window.mdi.activeSubWindow()
    assert not sub.windowIcon().isNull()

    def title_icon_extent():
        """The width of what the title bar paints in its icon's place (pixels not the bar's background)."""
        opt, _ = _title_option(sub)
        rect = sub.style().subControlRect(S.ComplexControl.CC_TitleBar, opt, S.SubControl.SC_TitleBarSysMenu, sub)
        image = sub.grab(rect).toImage()
        background = QtGui.QColor(title_colours(sub)[0]).rgb() & 0xFFFFFF
        xs = [x for y in range(image.height()) for x in range(image.width()) if image.pixel(x, y) & 0xFFFFFF != background]
        return (max(xs) - min(xs) + 1) if xs else 0, min(rect.width(), rect.height())

    extents = {}
    for step in ("Smaller", "Normal", "Largest"):
        window.set_text_size(step)
        _settle(app)
        sub = next(x for x in window.mdi.subWindowList() if x.windowTitle().startswith("dynamic"))
        window.mdi.setActiveSubWindow(sub)
        _settle(app)
        want = 2 * round(small * text_size.FACTORS[step] / 2)
        assert icons.view_size(window.objects) == want
        assert window.objects.iconSize().width() == window.objects.iconSize().height() == want
        assert window.objects.visualItemRect(window.objects.topLevelItem(0)).height() >= want
        dialog = palette_dialog(window.palette_entries, window)
        try:
            assert dialog.findChild(QtWidgets.QListWidget).iconSize().width() == want
        finally:
            dialog.deleteLater()
        extent, room = title_icon_extent()
        assert extent > 0, f"no icon painted on the title bar at {step}"
        assert extent <= min(want, room)
        extents[step] = extent
    assert extents["Smaller"] < extents["Normal"] <= extents["Largest"]
    tabbed = next(a for a in window.findChildren(QtGui.QAction) if a.text().replace("&", "") == "Tabbed windows")
    tabbed.setChecked(True)  # View → Tabbed windows: the tabs show the icons, at the text size too
    try:
        _settle(app)
        bar = window.mdi.findChild(QtWidgets.QTabBar)
        assert bar.iconSize().width() == 2 * round(small * text_size.FACTORS["Largest"] / 2)
        window.set_text_size("Normal")
        _settle(app)
        assert bar.iconSize().width() == small
    finally:
        tabbed.setChecked(False)
        _settle(app)
