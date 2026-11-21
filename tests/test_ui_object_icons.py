"""Objects carry an icon of their kind: in the Objects panel, on their windows and in Find anything (Ctrl+K).

The kind comes from one table (pymetrix.ui.icons.CLASS_KINDS); every kind the app names in words (ui.qt._KINDS) has an
icon, and anything unknown the generic one. Each icon draws: not empty at device pixel ratios 1 and 2, the ratio-2
drawing made at twice the pixels (not the 16-pixel one enlarged), on a light and a dark palette, and the shapes of
the kinds differ from one another (not only their colours). Icons are decoration: the rows' text is unchanged."""

import os

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtCore, QtGui  # noqa: E402

from pymetrix.ui import icons  # noqa: E402

# kinds the app names in words that are plain values, with no kind of their own
GENERIC = {"float", "int", "float64", "str", "list", "dict", "Sample"}


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _palette(window, text, base):
    pal = QtGui.QPalette()
    R = QtGui.QPalette.ColorRole
    for role, colour in {R.Window: window, R.WindowText: text, R.Text: text, R.Base: base, R.Highlight: "#2a82da",
                         R.HighlightedText: "#ffffff"}.items():
        pal.setColor(role, QtGui.QColor(colour))
    return pal


LIGHT = _palette("#efefef", "#000000", "#ffffff")
DARK = _palette("#2d2d2d", "#e6e6e6", "#1e1e1e")


def _opaque(image):
    """The pixels drawn (alpha over a quarter), as a set of (x, y)."""
    image = image.convertToFormat(QtGui.QImage.Format.Format_ARGB32)
    return {(x, y) for y in range(image.height()) for x in range(image.width()) if QtGui.qAlpha(image.pixel(x, y)) > 64}


def test_every_kind_the_app_names_has_an_icon_and_unknowns_the_generic_one(app):
    from pymetrix.ui.qt import _KINDS

    for name in _KINDS:
        kind = icons.kind_of_name(name)
        assert kind in icons.KINDS, name
        assert (kind == "generic") == (name in GENERIC), (name, kind)
        assert not icons.icon(kind).isNull()
    assert set(icons.CLASS_KINDS.values()) | {"generic"} == set(icons.KINDS)  # no kind without a class
    assert icons.kind_of_name("SomethingNew", core=False) == "generic"
    assert icons.kind_of(object()) == "generic" and not icons.icon("no such kind").isNull()

    class PluginResult:  # a plugin's class (its module is not pymetrix's): generic, whatever its name says
        pass

    PluginResult.__name__ = "FittedThing"
    assert icons.kind_of(PluginResult()) == "generic"
    assert icons.kind_of_name("FittedThing") == "equation"  # pymetrix's own estimates of one equation


def test_objects_map_to_their_kinds(app, ts):
    import numpy as np
    import pandas as pd

    from pymetrix import Equation, Table
    from pymetrix.saved import SavedResult
    from pymetrix.var import FittedVAR

    fit = Equation("y ~ x").fit(ts)
    assert icons.kind_of(ts) == "dataset" and icons.kind_of(ts["y"]) == "series"
    assert icons.kind_of(fit) == "equation" and icons.kind_of(Equation("y ~ x")) == "equation"
    assert icons.kind_of(fit.test("reset")) == "test"
    assert icons.kind_of(pd.DataFrame({"a": [1]})) == "table" and icons.kind_of(np.eye(2)) == "table"
    assert icons.kind_of(fit.plot()) == "figure" and icons.kind_of(Table.from_frame(pd.DataFrame({"a": [1]}))) == "table"
    assert icons.kind_of(3.0) == "generic"

    class MyVAR(FittedVAR):  # a subclass takes its base's kind
        def __init__(self):
            pass

    assert icons.kind_of(MyVAR()) == "system"
    saved = SavedResult(kind="FittedVAR", name="v", saved="", package_version="", views=())
    assert icons.kind_of(saved) == "system"
    for name, kind in {"ForecastRound": "forecast", "Backtest": "forecast", "Template": "model", "LinearSolution": "model",
                       "ScenarioSet": "model", "Report": "report", "IRF": "system", "FittedVECM": "system",
                       "Matrix": "table", "JohansenResult": "test"}.items():
        assert icons.kind_of_name(name) == kind, name


@pytest.mark.parametrize("palette", [LIGHT, DARK], ids=["light", "dark"])
def test_icons_draw_at_ratios_1_and_2_and_differ_in_shape(app, palette):
    shapes = {}
    for kind in icons.KINDS:
        one, two = icons.pixmap(kind, 16, 1.0, palette), icons.pixmap(kind, 16, 2.0, palette)
        assert (one.width(), one.height(), one.devicePixelRatio()) == (16, 16, 1.0)
        assert (two.width(), two.height(), two.devicePixelRatio()) == (32, 32, 2.0)
        drawn_one, drawn_two = _opaque(one.toImage()), _opaque(two.toImage())
        assert 12 <= len(drawn_one) <= 220, kind  # something, not a filled square
        assert len(drawn_two) > 2.5 * len(drawn_one), kind  # drawn at 32 pixels, not 16 blown up later
        shapes[kind] = frozenset(drawn_one)
        # the icon Qt picks for a 16-pixel view on a screen at ratio 2 is the 32-pixel drawing
        picked = icons.icon(kind, palette).pixmap(QtCore.QSize(16, 16), 2.0)
        assert picked.width() == 32 and _opaque(picked.toImage())
        selected = icons.icon(kind, palette).pixmap(QtCore.QSize(16, 16), 1.0, QtGui.QIcon.Mode.Selected)
        assert _opaque(selected.toImage()), kind
    for a in icons.KINDS:  # the shapes differ, so a kind tells without its colour
        for b in icons.KINDS:
            if a < b:
                overlap = len(shapes[a] & shapes[b]) / len(shapes[a] | shapes[b])
                assert overlap < 0.6, (a, b, overlap)


def test_icons_read_on_light_and_dark(app):
    """The outline is the palette's text colour and the accent a theme role: each reads against the list background."""
    from pymetrix.ui import theme

    for pal in (LIGHT, DARK):
        base = pal.color(QtGui.QPalette.ColorRole.Base)
        image = icons.pixmap("table", 16, 2.0, pal).toImage()  # its grid in ink, its header in the accent
        colours = {QtGui.QColor(image.pixel(x, y)).name() for x, y in _opaque(image) if QtGui.qAlpha(image.pixel(x, y)) == 255}
        assert pal.color(QtGui.QPalette.ColorRole.Text).name() in colours
        assert theme.name("link", pal) in colours
        assert max(theme.contrast(c, base) for c in colours) >= 4.5
    assert icons.icon("dataset", LIGHT).cacheKey() != icons.icon("dataset", DARK).cacheKey()  # one per palette


def test_the_app_shows_icons_on_rows_windows_and_palette_entries(app):
    pytest.importorskip("qtconsole")
    from pymetrix.demos import demo
    from pymetrix.ui.qt import kind_of, main_window

    s = demo("consumption")
    w = main_window(s)
    w.show()
    try:
        app.processEvents()
        rows = {w.objects.topLevelItem(i).text(0): w.objects.topLevelItem(i) for i in range(w.objects.topLevelItemCount())}
        assert set(rows) == set(s.objects)
        for name, item in rows.items():
            assert item.text(1) == kind_of(s.objects[name])  # the text is as it was
            assert not item.icon(0).isNull()
            assert item.icon(0).cacheKey() == icons.object_icon(s.objects[name]).cacheKey()
        assert rows["data"].icon(0).cacheKey() != rows["dynamic"].icon(0).cacheKey()
        w.show_object("dynamic")
        app.processEvents()
        sub = next(x for x in w.mdi.subWindowList() if x.windowTitle().startswith("dynamic"))
        assert sub.windowIcon().cacheKey() == icons.icon("equation").cacheKey()
        objects = [e for e in w.palette_entries() if e.kind == "Object"]
        assert objects and all(e.icon is not None and not e.icon.isNull() for e in objects)
    finally:
        w.state["session"].dirty = False
        w.close()


def test_the_front_title_bar_paints_the_icon(app):
    """The active-window painter draws the window's icon on the title bar (in the selection's colours in front)."""
    from pymetrix.ui.active_window import _title_option, follow_active_window

    area = QtWidgets.QMdiArea()
    area.resize(600, 400)
    follow_active_window(area)
    sub = QtWidgets.QMdiSubWindow()
    sub.setWidget(QtWidgets.QLabel("x"))
    sub.setWindowTitle("x — Figure")
    area.addSubWindow(sub)
    sub.setGeometry(10, 10, 400, 250)
    sub.show()
    area.show()
    area.setActiveSubWindow(sub)
    app.processEvents()

    def icon_pixels():
        opt, _ = _title_option(sub)
        rect = sub.style().subControlRect(QtWidgets.QStyle.ComplexControl.CC_TitleBar, opt,
                                          QtWidgets.QStyle.SubControl.SC_TitleBarSysMenu, sub)
        image = sub.grab(rect).toImage()
        return {image.pixel(x, y) for y in range(image.height()) for x in range(image.width())}

    plain = icon_pixels()
    sub.setWindowIcon(icons.icon("figure"))
    app.processEvents()
    assert sub.windowIcon().cacheKey() == icons.icon("figure").cacheKey()
    assert icon_pixels() != plain
    area.close()
