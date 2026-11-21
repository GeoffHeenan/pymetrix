"""The charts' house look (``pymetrix.figure_style``): one palette, grid, ticks and type for every pymetrix figure,
applied for that figure only (a user's own matplotlib plots and the global rcParams are untouched); the size in inches
unchanged by the look, the screen's device pixel ratio or the app's dark palette; exported files light."""

import io
import logging
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Okabe & Ito (2008), "Color Universal Design", the published RGB values, in the order the default theme cycles them:
# blue, vermillion, bluish green, reddish purple, orange, sky blue, yellow, black
OKABE_ITO_RGB = [(0, 114, 178), (213, 94, 0), (0, 158, 115), (204, 121, 167), (230, 159, 0), (86, 180, 233), (240, 228, 66), (0, 0, 0)]


def _series(n=3, periods=24):
    rng = np.random.default_rng(7)
    idx = pd.period_range("2015Q1", periods=periods, freq="Q")
    return [pd.Series(np.cumsum(rng.normal(0, 1, periods)) + k, index=idx) for k in range(n)]


def _lines(n=3, **options):
    from pymetrix.figures import Figure, Line, Panel

    return Figure((Panel(tuple(Line(s, f"s{k}") for k, s in enumerate(_series(n))), title="lines"),), title="t", **options)


def _rgb255(color):
    from matplotlib.colors import to_rgb

    return tuple(int(round(255 * v)) for v in to_rgb(color))


def test_house_palette_is_okabe_ito_in_its_published_order():
    from pymetrix.figure_style import LIGHT

    fig = _lines(5).draw()
    ax = fig.axes[0]
    drawn = [_rgb255(line.get_color()) for line in ax.get_lines() if not line.get_label().startswith("_")]
    assert drawn == OKABE_ITO_RGB[:5]
    # the grid: major lines only, light grey, behind the data; no top or right spine; ticks outwards
    assert ax.get_axisbelow() is True
    assert all(g.get_visible() for g in ax.yaxis.get_gridlines())
    assert {_rgb255(g.get_color()) for g in ax.yaxis.get_gridlines()} == {_rgb255(LIGHT.grid)}
    assert not any(t.gridline.get_visible() for t in ax.yaxis.get_minor_ticks())
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
    assert {t._tickdir for t in ax.yaxis.get_major_ticks()} == {"out"}
    # light grey: closer to the white background than to the text
    grid = np.array(_rgb255(LIGHT.grid))
    assert np.abs(grid - 255).max() < 40 and np.ptp(grid) == 0


def test_bars_have_lines_across_the_values_only():
    from pymetrix.figures import Bars, Figure, Panel

    s = pd.Series([1.0, -2.0, 3.0], index=pd.period_range("2020", periods=3, freq="Y"))
    ax = Figure((Panel((Bars(s, "a"),)),)).draw().axes[0]
    assert all(g.get_visible() for g in ax.yaxis.get_gridlines())
    assert not any(g.get_visible() for g in ax.xaxis.get_gridlines())


def test_global_rcparams_and_a_users_own_plot_are_untouched(tmp_path):
    import matplotlib
    from matplotlib.figure import Figure as MplFigure

    from pymetrix.figure_style import DARK

    before = dict(matplotlib.rcParams)
    fig = _lines(3, footnote="source")
    fig.draw()
    fig.draw(font="DejaVu Sans", surface=DARK)
    fig.png()
    fig.save(str(tmp_path / "c.svg"))
    assert dict(matplotlib.rcParams) == before
    mine = MplFigure().subplots()
    (line,) = mine.plot([0, 1], [1, 2])
    assert line.get_color() == matplotlib.rcParams["axes.prop_cycle"].by_key()["color"][0]  # the user's cycle, not ours
    assert mine.spines["top"].get_visible() == matplotlib.rcParams["axes.spines.top"]


def test_ticks_made_later_keep_the_house_look_whatever_the_global_settings():
    from matplotlib import rc_context

    ax = _lines().draw().axes[0]
    with rc_context({"xtick.direction": "in", "ytick.direction": "in", "grid.color": "red"}):
        ticks = ax.yaxis.get_major_ticks(numticks=len(ax.yaxis.majorTicks) + 5)  # new ticks, made now
    assert {t._tickdir for t in ticks} == {"out"}
    assert {_rgb255(t.gridline.get_color()) for t in ticks} != {(255, 0, 0)}


def test_size_in_inches_is_the_figures_whatever_the_look():
    from pymetrix.figure_style import DARK

    fig = _lines(width=6.4, height=3.2)
    for drawn in (fig.draw(), fig.draw(font="DejaVu Sans", surface=DARK)):
        assert tuple(drawn.get_size_inches()) == (6.4, 3.2)
        assert drawn.dpi == 100
    from PIL import Image

    assert Image.open(io.BytesIO(fig.png(dpi=150))).size == (960, 480)


def test_font_the_apps_when_known_else_dejavu_sans_without_warnings(caplog):
    from pymetrix.figure_style import FALLBACK_FONT, known_font

    assert known_font("DejaVu Sans") == "DejaVu Sans"
    assert known_font("No Such Font 123") is None
    with caplog.at_level(logging.WARNING, logger="matplotlib.font_manager"):
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig = _lines().draw(font="No Such Font 123")
        FigureCanvasAgg(fig).draw()
    assert not [r for r in caplog.records if "not found" in r.getMessage()]
    assert fig.axes[0].title.get_fontfamily()[0] == FALLBACK_FONT
    published = _lines(theme="publication").draw()
    assert published.axes[0].title.get_fontfamily()[0] == "serif"  # a theme's own font wins


def test_dark_surface_gives_a_dark_chart_with_readable_greys():
    from matplotlib.colors import to_rgb

    from pymetrix.figure_style import DARK, LIGHT, surface_from
    from pymetrix.figures import Figure, Line, Panel

    s = _series(1)[0]
    fig = Figure((Panel((Line(s, "actual", color="#444444"), Line(s + 1, "forecast")), title="p"),), title="t", reference=((0.0, "zero"),))
    dark = fig.draw(surface=DARK)
    light = fig.draw()
    assert to_rgb(dark.get_facecolor()) == to_rgb(DARK.background) and sum(to_rgb(dark.get_facecolor())) < 0.6
    assert to_rgb(light.get_facecolor()) == (1.0, 1.0, 1.0)
    ax = dark.axes[0]
    assert sum(to_rgb(ax.title.get_color())) > 2.0  # light text
    actual = next(line for line in ax.get_lines() if line.get_label() == "actual")
    assert sum(to_rgb(actual.get_color())) > 2.0  # the dark grey "actual" line turned light
    forecast = next(line for line in ax.get_lines() if line.get_label() == "forecast")
    assert _rgb255(forecast.get_color()) == OKABE_ITO_RGB[0]  # colours kept
    assert {_rgb255(g.get_color()) for g in ax.yaxis.get_gridlines()} == {_rgb255(DARK.grid)}  # the grid stays subdued
    assert surface_from("#202124", "#e8eaed").dark and not surface_from("#f0f0f0", "#000000").dark
    assert LIGHT.dark is False


def test_fan_legend_keys_shade_as_the_overlapping_bands_do():
    from pymetrix.figures import Fan, Figure, Panel

    s = _series(1)[0]
    se = pd.Series(np.linspace(0.1, 1.0, len(s)), index=s.index)
    ax = Figure((Panel((Fan.normal(s, se, label="c"),)),)).draw().axes[0]
    legend = ax.get_legend()
    keys = {t.get_text(): h for t, h in zip(legend.get_texts(), legend.legend_handles)}
    alphas = [float(keys[name].get_facecolor()[3]) for name in ("90%", "60%", "30%")]  # as drawn in the key
    assert alphas[0] == pytest.approx(0.22)
    assert alphas[1] == pytest.approx(1 - 0.78 ** 2) and alphas[2] == pytest.approx(1 - 0.78 ** 3)


def test_split_axes_render_in_the_house_look():
    from pymetrix.figures import Figure, Line, Panel

    idx = pd.period_range("2015Q1", periods=20, freq="Q")
    s = pd.Series(np.r_[np.linspace(0, 5, 10), np.linspace(95, 100, 10)], index=idx)
    fig = Figure((Panel((Line(s, "jump"),)),), ybreak=(10, 90)).draw()
    assert len(fig.axes) == 2
    for ax in fig.axes:
        assert {t._tickdir for t in ax.yaxis.get_major_ticks()} == {"out"}
        assert ax.get_axisbelow() is True


def test_matplotlib_code_sets_the_house_look_for_itself():
    code = _lines().matplotlib_code()
    assert "'axes.spines.top': False" in code and "'axes.axisbelow': True" in code


def _app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_qt_canvas_draws_at_the_device_pixel_ratio():
    _app()
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    from pymetrix.ui.chart_look import sharpen

    fig = _lines(width=6.0, height=3.0).draw()
    canvas = FigureCanvasQTAgg(fig)
    canvas.devicePixelRatioF = lambda: 2.0  # a high-resolution screen
    assert sharpen(canvas) == 2.0
    assert fig.dpi == 200 and tuple(fig.get_size_inches()) == (6.0, 3.0)
    canvas.draw()
    assert canvas.get_renderer().width == 1200  # twice the pixels, the same inches


def test_app_charts_use_the_font_the_app_really_draws_in():
    app = _app()
    from PySide6.QtGui import QFontInfo

    from pymetrix.figure_style import FALLBACK_FONT, known_font
    from pymetrix.ui.chart_look import look
    from pymetrix.ui.qt import figure_view

    family = look()[0]
    # the family Qt resolves the app's font to (not an alias such as "Sans Serif"), when matplotlib has that font; else
    # matplotlib's DejaVu Sans (where Qt has no fonts at all, as offscreen on Windows CI, it resolves to nothing)
    assert family == (known_font(QFontInfo(app.font()).family()) or known_font(app.font().family()) or FALLBACK_FONT)
    assert known_font(family) == family  # one matplotlib finds: no "font not found" for each chart
    page = figure_view(_lines(), "f")
    assert page.canvas.figure.axes[0].title.get_fontfamily()[0] == family
    page.close()


@pytest.mark.parametrize("resolved", ["", "Sans Serif", "A Font No Machine Has"])
def test_app_charts_fall_back_to_dejavu_sans_when_matplotlib_cannot_find_the_apps_font(monkeypatch, resolved):
    """Where Qt resolves the app's font to nothing (no fonts installed), an alias or a font matplotlib has not got, and
    the family asked for is no better, charts use matplotlib's own DejaVu Sans."""
    app = _app()
    import PySide6.QtGui as QtGui

    from pymetrix.figure_style import FALLBACK_FONT
    from pymetrix.ui.chart_look import look

    class Resolved:
        def __init__(self, font):
            pass

        def family(self):
            return resolved

    monkeypatch.setattr(QtGui, "QFontInfo", Resolved)
    old = app.font()
    try:
        asked = QtGui.QFont(old)
        asked.setFamily("Another Font No Machine Has")
        app.setFont(asked)
        assert look()[0] == FALLBACK_FONT
    finally:
        app.setFont(old)


def test_app_figure_view_follows_a_dark_palette_and_copies_stay_light():
    app = _app()
    from matplotlib.colors import to_rgb
    from PySide6.QtGui import QColor, QPalette
    from PIL import Image

    from pymetrix.ui.chart_look import look
    from pymetrix.ui.qt import figure_view

    fig = _lines()
    old = app.palette()
    try:
        palette = QPalette(old)
        palette.setColor(QPalette.ColorRole.Window, QColor("#202124"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#e8eaed"))
        app.setPalette(palette)
        assert look()[1].dark
        page = figure_view(fig, "f")
        drawn = page.canvas.figure
        assert sum(to_rgb(drawn.get_facecolor())) < 0.6  # not a white box on a dark window
        assert tuple(drawn.get_size_inches()) == tuple(fig.draw().get_size_inches())
        page.close()
    finally:
        app.setPalette(old)
    assert not look()[1].dark
    corner = Image.open(io.BytesIO(fig.png())).convert("RGB").getpixel((2, 2))
    assert corner == (255, 255, 255)  # what is copied and saved is light


def test_bands_read_on_a_dark_surface_and_their_keys_keep_in_step():
    from pymetrix.figure_style import DARK
    from pymetrix.figures import Fan, Figure, Panel

    s = _series(1)[0]
    se = pd.Series(np.linspace(0.1, 1.0, len(s)), index=s.index)
    fig = Figure((Panel((Fan.normal(s, se, label="c"),)),))
    light, dark = fig.draw().axes[0], fig.draw(surface=DARK).axes[0]
    assert light.collections[0].get_alpha() == pytest.approx(0.22)
    assert dark.collections[0].get_alpha() == pytest.approx(1 - 0.78 ** 1.6)  # each layer covers more on dark
    legend = dark.get_legend()
    keys = {t.get_text(): float(h.get_facecolor()[3]) for t, h in zip(legend.get_texts(), legend.legend_handles)
            if t.get_text().endswith("%")}
    assert keys["90%"] == pytest.approx(1 - 0.78 ** 1.6) and keys["30%"] == pytest.approx(1 - 0.78 ** (3 * 1.6))
