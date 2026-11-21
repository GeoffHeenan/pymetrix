"""Windows that fit the room they get (off-screen Qt): made small, a window's buttons are never drawn over its figure
or table, and what does not fit scrolls, sideways when it is too narrow and up and down when it is too short, each
on its own. Every kind of result window (tests/_gallery.py), the dataset window, the spreadsheet, the model editor,
the Start window, the code editor, the Report panel, the dialogs generated from commands (on a 1024 by 600 screen)
and the app's other dialogs. The checks themselves are in tests/_fit_check.py."""

import os
import warnings

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtCore  # noqa: E402
pytest.importorskip("qtconsole")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import _fit_check as fc  # noqa: E402
import _gallery  # noqa: E402
from test_series_browser import net  # noqa: E402,F401 - every request answered from a fixture

SIZES = ((680, 460), (400, 300), (230, 150), fc.NARROW, fc.SHORT)  # the opening size, small, and one way at a time


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _settle():
    """Posted layout requests, and the ones they post in turn (a form's, then its window's)."""
    for _ in range(3):
        _app().processEvents()


def _in_workspace(widget):
    """``widget`` in a window of the workspace, as the app shows it; the window's resize."""
    mdi = QtWidgets.QMdiArea()
    mdi.resize(1000, 800)
    sub = mdi.addSubWindow(widget)
    mdi.show()
    mdi.keep = sub
    return mdi, sub


def _result_view(obj, name="x"):
    from pymetrix.ui.qt import object_view

    return object_view(obj, name=name, code=f"{name} = ...", edit_figure=lambda *a: None, place_note=lambda *a: None,
                       heatmap=lambda t: None, add_to_report=lambda t: None, next_steps=[("Forecast…", lambda: None)],
                       series_actions={"plot": lambda names: None, "sheet": lambda names: None})


@pytest.fixture
def small_screen(monkeypatch):
    """A laptop's 1024 by 600 screen, for the sizes windows and dialogs open at."""
    from PySide6.QtCore import QRect

    from pymetrix.ui import fit

    monkeypatch.setattr(fit, "_available", lambda widget: QRect(0, 0, 1024, 600))
    return QRect(0, 0, 1024, 600)


FONTS = ("default", "wide")  # this machine's font, and one as wide as CI's and Windows' (DejaVu Sans 11, Segoe UI)


@pytest.fixture(params=FONTS)
def font(request):
    """The app's font for the test: as it is, or a third bigger (text wider than with the font the sizes were tried
    with), put back afterwards. Folding and scrolling go by the sizes as they are, whatever the font."""
    app = _app()
    before = app.font()
    if request.param == "wide":
        wide = app.font()
        wide.setPointSizeF(before.pointSizeF() * 1.35 if before.pointSizeF() > 0 else 13)
        app.setFont(wide)
    yield request.param
    app.setFont(before)


@pytest.mark.parametrize("kind", sorted(_gallery.BUILDERS))
def test_every_result_window_fits_when_small(kind):
    """Every tab of every kind of result window, at its opening size, small, narrow and short."""
    _app()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = _gallery.build(kind)
    view = _result_view(obj)
    mdi, sub = _in_workspace(view)
    assert fc.check_sizes(view, SIZES, resize=sub.resize) == []
    mdi.deleteLater()


def test_dataset_window_fits_when_small():
    """The dataset window: its status line and Next ▸ beside the tabs, and the series browser's buttons."""
    _app()
    from pymetrix import demo

    session = demo("consumption")
    data = next(o for o in session.objects.values() if type(o).__name__ == "Dataset")
    view = _result_view(data)
    mdi, sub = _in_workspace(view)
    assert fc.check_sizes(view, SIZES, resize=sub.resize) == []
    sub.resize(300, 250)
    _app().processEvents()
    assert view.corner.folded(), "the status line and Next ▸ fold into » beside the tabs"
    mdi.deleteLater()


def _figure_window():
    from pymetrix.ui.qt import figure_view

    figure = _gallery.build("Figure")
    page = figure_view(figure, on_edit=lambda kind: None, on_place=lambda *a: None, on_report=lambda: None)
    page.show()
    return page


@pytest.mark.parametrize("size, horizontal, vertical", [((900, 700), False, False), ((230, 700), True, False),
                                                        ((900, 200), False, True), ((230, 200), True, True)])
def test_a_figure_scrolls_in_each_direction_on_its_own(size, horizontal, vertical):
    """Below the figure's readable size the window scrolls: sideways when narrow, up and down when short, both when
    both; above it the figure follows the window and no scroll bar shows."""
    _app()
    page = _figure_window()
    page.resize(*size)
    _app().processEvents()
    scroll = page.scroll
    assert scroll.horizontalScrollBar().isVisible() is horizontal
    assert scroll.verticalScrollBar().isVisible() is vertical
    canvas = page.canvas
    assert canvas.width() >= canvas.minimumWidth() and canvas.height() >= canvas.minimumHeight()  # never squashed
    if not horizontal and not vertical:
        assert canvas.width() > 600 and canvas.height() > 400  # the figure fills a big window, as before
    assert fc.problems(page) == []
    page.close()


def _row_room(row, beside):
    """The page's width at which every button of ``row`` fits side by side, from the buttons' sizes now (the font's);
    ``beside``: the page's width less the row's."""
    return beside + row._widths(row.items, least=True) + row._margins()


def test_buttons_under_a_figure_fold_into_a_menu_rather_than_overlap(font):
    _app()
    page = _figure_window()
    row = page.actions_row
    page.resize(680 - 12, 460)  # a result window at its opening size: all seven with the usual fonts, some in » with wider
    _app().processEvents()
    assert fc.problems(page) == []
    beside = page.width() - row.width()
    page.resize(max(_row_room(row, beside), page.width()), 460)  # room for every button: none folds
    _app().processEvents()
    assert row.folded() == [] and not row.more.isVisible()
    page.resize(300, 400)
    _app().processEvents()
    folded = row.folded()
    assert folded and row.more.isVisible()
    assert fc.problems(page) == []
    canvas = page.canvas.geometry()
    for button in [b for b in row.items if b.isVisible()] + [row.more]:
        place = button.geometry().translated(row.pos())
        assert not place.intersects(canvas), button.text()
        assert row.rect().contains(button.geometry()), f"{button.text()} cut off"
        assert button.width() >= button.minimumSizeHint().width(), f"{button.text()} squashed"
    row._fill_menu()  # the » menu: the folded buttons, in order, a button with a menu as a submenu
    entries = [a.text() for a in row.more_menu.actions()]
    assert entries == [b.text() for b in folded]
    copy_data = next((a for a in row.more_menu.actions() if a.text() == "Copy data"), None)
    if copy_data is not None:  # an entry does what its button does
        _app().clipboard().setText("")
        copy_data.trigger()
        assert "\t" in _app().clipboard().text()
    page.resize(max(_row_room(row, beside), 900), 600)
    _app().processEvents()
    assert row.folded() == [] and all(b.isVisible() for b in row.items)  # back when there is room
    page.close()


def test_the_corner_buttons_leave_the_views_in_view():
    """The views are listed down the side, every one of them: the corner buttons sit above the view, never over the
    list, and fold into » when the window is narrow."""
    _app()
    view = _result_view(_gallery.build("FittedVAR"))
    mdi, sub = _in_workspace(view)
    for width in (680, 400, 260):
        sub.resize(width, 300)
        _app().processEvents()
        side, corner = view.side, view.cornerWidget()
        corner_box = QtCore.QRect(corner.mapTo(view, QtCore.QPoint(0, 0)), corner.size())
        side_box = QtCore.QRect(side.mapTo(view, QtCore.QPoint(0, 0)), side.size())
        assert not side_box.intersects(corner_box)
        assert side.width() >= 72, "room for the views' names beside the view"
        assert side.count() >= view.count()  # every view listed, none behind scroll arrows
    assert view.corner.folded()
    mdi.deleteLater()


@pytest.fixture(params=fc.LOOKS)
def look(request):
    """The app as it looks here, with a font a third wider, and in Windows' style (tests/_fit_check.py)."""
    with fc.look(request.param) as name:
        yield name


@pytest.fixture
def code_folded():
    """Dialogs open with Show the code folded, as at first; the choice (remembered in the settings) as it was after."""
    from PySide6.QtCore import QSettings

    from pymetrix.ui.qt import SHOW_CODE_KEY

    settings = QSettings()
    kept = settings.value(SHOW_CODE_KEY) if settings.contains(SHOW_CODE_KEY) else None
    settings.remove(SHOW_CODE_KEY)
    yield
    if kept is None:
        settings.remove(SHOW_CODE_KEY)
    else:
        settings.setValue(SHOW_CODE_KEY, kept)


def test_command_dialogs_fit_a_small_screen(small_screen, look, code_folded):
    """Each dialog generated from a command fits 1024 by 600: its least size, and the size it opens at; a long form
    scrolls, and a dialog made small scrolls rather than squashes. All of it with the Advanced settings section
    closed (as dialogs open) and open, and with Show the code (under the form) folded and open beside it."""
    _app()
    from pymetrix import demo
    from pymetrix.commands import COMMANDS
    from pymetrix.ui.qt import command_dialog

    session = demo("consumption")
    too_big, long_forms, opened = [], [], 0
    for key, info in COMMANDS.items():
        try:
            dialog = command_dialog(info.cls, session)
        except Exception:  # noqa: BLE001 - a dialog that needs more than the demo has; its own tests cover it
            continue
        if dialog.form_scroll.widget().minimumSizeHint().height() > 0.7 * small_screen.height():  # as it opens
            long_forms.append(key)
        for section_open, code_open in ((False, False), (False, True), (True, False), (True, True)):
            if section_open and dialog.advanced_section is None:
                break
            if section_open and not dialog.advanced_toggle.isChecked():
                dialog.show_advanced(True)
                opened += 1
            dialog.code_toggle.setChecked(code_open)
            least, opens = dialog.minimumSizeHint(), dialog.sizeHint()
            if least.width() > small_screen.width() or least.height() > small_screen.height() or opens.height() > small_screen.height():
                too_big.append((key, section_open, code_open, least.toTuple(), opens.toTuple()))
        dialog.code_toggle.setChecked(False)
        dialog.deleteLater()
    assert too_big == []
    assert opened > 100
    assert long_forms, "some forms are longer than a small screen: the case this guards"
    for key in long_forms[:3] + ["PlotSeries"]:
        info = COMMANDS.get(key) or next(i for i in COMMANDS.values() if i.cls.__name__ == key)
        for section_open, code_open in ((False, False), (True, False), (True, True)):
            dialog = command_dialog(info.cls, session)
            dialog.show_advanced(section_open)
            dialog.code_toggle.setChecked(code_open)
            dialog.show()
            assert fc.check_sizes(dialog, ((500, 400), (300, 250), (200, 500), (600, 200))) == [], (key, section_open, code_open)
            dialog.resize(300, 250)
            _app().processEvents()
            scroll = dialog.form_scroll
            # up and down; sideways only where the form is wider than its viewport (the dialog keeps its least width,
            # which leaves room for the vertical bar: a sideways bar there was one bar's width of nothing to scroll to)
            assert scroll.verticalScrollBar().isVisible()
            assert scroll.horizontalScrollBar().isVisible() == (scroll.widget().width() > scroll.viewport().width())
            assert scroll.widget().width() >= scroll.widget().minimumSizeHint().width()
            dialog.code_toggle.setChecked(False)
            dialog.close()


def test_a_form_that_grows_after_it_opens_scrolls_rather_than_squashes(small_screen, font):
    """A field wider than the form as it opened (another font, a style's polish on a tab's first showing, a field
    shown later): the dialog's least size grows with it where the screen has room, and below that the form scrolls
    sideways, never drawn narrower than its fields. The widths go by the dialog's own least width as it opened, which
    depends on the font and style (586 pixels with Windows' Segoe UI, less here)."""
    _app()
    from pymetrix import demo
    from pymetrix.commands import COMMANDS
    from pymetrix.ui.qt import command_dialog

    info = next(i for i in COMMANDS.values() if i.cls.__name__ == "SolveModel")
    dialog = command_dialog(info.cls, demo("consumption"))
    dialog.show()
    _settle()
    scroll = dialog.form_scroll
    own = dialog.minimumSizeHint().width()  # the form's own least width, at least its 420 pixels, with this font
    wider = own + 140
    assert 420 < own and wider < small_screen.width()
    field = dialog.editors["equations"].outer  # a field grown wider than the form
    field.setMinimumWidth(wider)
    _settle()
    assert scroll.widget().width() >= wider and dialog.minimumSizeHint().width() > wider
    assert fc.check_sizes(dialog, ((500, 400), (300, 250), (200, 500), (600, 200))) == []
    field.setMinimumWidth(1400)  # wider than the screen: the dialog stays within it and the form scrolls sideways
    _settle()
    assert dialog.minimumSizeHint().width() <= small_screen.width()
    dialog.resize(600, 400)
    assert fc.problems(dialog) == [] and scroll.horizontalScrollBar().isVisible()
    field.setMinimumWidth(0)  # and back: the least size follows the form down again, to its own
    _settle()
    assert dialog.minimumSizeHint().width() == own
    dialog.close()


def test_a_short_dialog_opens_as_before():
    """A form that fits the screen opens at its full size, no scroll bars: the scroll area changes nothing there."""
    _app()
    from pymetrix import demo
    from pymetrix.commands import COMMANDS
    from pymetrix.ui.qt import command_dialog

    info = next(i for i in COMMANDS.values() if i.cls.__name__ == "LoadExample")
    dialog = command_dialog(info.cls, demo("consumption"))
    dialog.show()
    _app().processEvents()
    form = dialog.form_scroll
    assert not form.horizontalScrollBar().isVisible() and not form.verticalScrollBar().isVisible()
    assert dialog.width() >= 420
    dialog.close()


@pytest.fixture(scope="module")
def window():
    _app()
    from pymetrix import demo
    from pymetrix.ui.qt import main_window

    win = main_window(demo("consumption"))
    win.resize(1100, 700)
    win.show()
    yield win
    win.close()


def _active(window):
    _app().processEvents()
    return window.mdi.activeSubWindow()


def test_spreadsheet_model_editor_and_code_editor_fit_when_small(window):
    window.open_sheet("data")
    sub = _active(window)
    assert fc.check_sizes(sub.widget(), SIZES[1:], resize=sub.resize) == []
    window.open_model_editor()
    sub = _active(window)
    editor = sub.widget()
    editor.sidecars.setCurrentWidget(editor.solution_page)  # the verdict, steady state, roots and responses
    assert fc.check_sizes(editor, SIZES[1:], resize=sub.resize) == []
    sub.resize(400, 300)
    _app().processEvents()
    assert editor.solution_page.scroll.verticalScrollBar().isVisible()  # scrolls rather than squashes the figure
    window.open_editor()
    sub = _active(window)
    assert fc.check_sizes(sub.widget(), SIZES[1:], resize=sub.resize) == []


def test_start_window_and_report_panel_fit_when_small():
    _app()
    from pymetrix.commands import Session
    from pymetrix.ui.qt import main_window

    win = main_window(Session())
    win.show()
    start = win.start_page
    assert fc.check_sizes(start.widget(), SIZES[1:], resize=start.resize) == []
    start.resize(300, 200)
    _app().processEvents()
    assert start.widget().verticalScrollBar().isVisible()
    assert fc.check_sizes(win.report_view, SIZES[1:]) == []
    win.close()


def test_other_windows_open_within_a_small_screen(small_screen, window):
    """Help, Plugins, Keyboard shortcuts, the command palette and the app's own window open no bigger than the
    screen, and their least sizes fit it."""
    from pymetrix.ui import fit
    from pymetrix.ui.plugins_qt import plugin_manager
    from pymetrix.ui.qt import shortcuts_dialog

    help_window = window.open_help()
    for widget in (help_window, plugin_manager(window), shortcuts_dialog(window)):
        assert widget.width() <= small_screen.width() and widget.height() <= small_screen.height(), type(widget).__name__
        least = widget.minimumSizeHint()
        assert least.width() <= small_screen.width() and least.height() <= small_screen.height()
    other = QtWidgets.QMainWindow()
    fit.fit_to_screen(other, 1100, 700)
    assert other.width() <= small_screen.width() and other.height() <= small_screen.height()


@pytest.mark.parametrize("provider", ["fred", "worldbank", "sdmx"])
def test_the_connector_series_browser_fits_when_small(small_screen, net, provider):
    """File → Browse series…: it opens within a small screen; made small, its lists and boxes scroll and its buttons
    fold into » rather than overlap, for each provider's layout (SDMX with a dataflow's dimension pickers)."""
    _app()
    from pymetrix.commands import Session
    from pymetrix.ui.series_browser_qt import series_browser

    b = series_browser(Session())
    b.set_provider(provider)
    if provider == "sdmx":
        b.search("exchange rates")
        b.select_rows([0])  # a dataflow: its dimensions become pickers
        assert b.pickers
    d = b.dialog
    d.show()
    assert d.width() <= small_screen.width() and d.height() <= small_screen.height()
    assert fc.check_sizes(d, SIZES[1:]) == []
    d.resize(300, 250)
    _app().processEvents()
    scroll = b.page.scroll
    assert scroll.verticalScrollBar().isVisible()
    # Load, Add to Load from connector… and Close fold into » unless the row is wider than this, scrolled sideways
    assert b.actions_row.folded() or scroll.horizontalScrollBar().isVisible()
    d.close()


def test_the_checks_find_a_squashed_window():
    """The checks themselves: a layout given less room than it needs is caught (buttons drawn over the widget above)."""
    _app()
    page = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(page)
    big = QtWidgets.QLabel("figure")
    big.setMinimumSize(320, 220)
    layout.addWidget(big)
    row = QtWidgets.QHBoxLayout()
    for text in ("Edit…", "Copy image", "Save…", "Copy data"):
        row.addWidget(QtWidgets.QPushButton(text))
    layout.addLayout(row)
    mdi, sub = _in_workspace(page)
    sub.resize(250, 150)
    _app().processEvents()
    assert fc.problems(page)
    mdi.deleteLater()


def test_the_fit_check_leaves_no_layout_item_behind_to_be_mistaken_for_a_widget():
    """Checking a window holds no Python wrapper of its layouts' items afterwards: with PySide6 6.6 one handed out by
    ``itemAt`` stayed referenced by its layout, so once Qt deleted the item (its widget deleted) the wrapper stayed
    mapped to the freed address, and a widget made there later (a console's viewport, a later test in the same
    process) came back to Python as a ``QWidgetItem``."""
    import gc

    import shiboken6

    def items():
        gc.collect()
        return sum(isinstance(o, QtWidgets.QLayoutItem) and not isinstance(o, QtWidgets.QLayout)
                   for o in shiboken6.getAllValidWrappers())

    page = QtWidgets.QWidget()
    outer = QtWidgets.QVBoxLayout(page)
    inner = QtWidgets.QHBoxLayout()
    outer.addLayout(inner)
    widgets = [QtWidgets.QLabel("a"), QtWidgets.QPushButton("b")]
    outer.addWidget(widgets[0])
    inner.addWidget(widgets[1])
    page.resize(300, 200)
    page.show()
    before = items()
    assert set(fc._layout_widgets(outer)) == set(widgets)  # the nested layout's widget too
    assert fc.layout_widget(outer, 1) is widgets[0] and fc.layout_widget(outer, 0) is None  # 0: the nested layout
    fc.problems(page)
    assert items() == before
    page.deleteLater()
