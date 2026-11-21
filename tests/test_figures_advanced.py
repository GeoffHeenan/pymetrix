"""The advanced chart kinds: 3-D surfaces, wireframes, lines and points; contours; violins; hexagonal bins. Each is
checked against what matplotlib drew, computed independently with NumPy or SciPy, and against what the standalone
matplotlib script draws, after saving and reopening."""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from pymetrix import Dataset, surface_figure  # noqa: E402
from pymetrix.errors import SpecificationError  # noqa: E402
from pymetrix.figures import Contour, Figure, Hexbin, Line, Line3d, Panel, Scatter3d, Surface, Violin  # noqa: E402
from pymetrix.saved import _decode_view, _encode_view  # noqa: E402

from test_figures import _drawn, _run_standalone  # noqa: E402

DATA = Path(__file__).parents[1] / "examples" / "sample_macro.parquet"


@pytest.fixture(scope="module")
def data():
    return Dataset.read(str(DATA)).with_series(gy="400*dlog(y)", gc="400*dlog(c)", gi="400*dlog(i)")


@pytest.fixture(scope="module")
def yields():
    """A yield curve over time: dates down the rows, maturities (years) across."""
    dates = pd.period_range("2015Q1", periods=24, freq="Q", name="date")
    maturities = pd.Index([0.25, 1.0, 2.0, 5.0, 10.0, 30.0], name="maturity")
    level = 1 + 0.08 * np.arange(24)
    values = level[:, None] + 1.5 * (1 - np.exp(-maturities.to_numpy() / 3))[None, :] + 0.2 * np.sin(np.arange(24))[:, None]
    return pd.DataFrame(values, index=dates, columns=maturities)


def _saved(fig):
    return _decode_view(json.loads(json.dumps(_encode_view(fig), default=str)))


def _three_d(drawn):
    return next(a for a in drawn.axes if a.name == "3d")


# -- surfaces and wireframes ----------------------------------------------------------------------------------------


def test_a_surface_draws_every_cell_with_dates_along_y(yields):
    fig = surface_figure(yields, label="yield", title="Yield curve")
    drawn = _drawn(fig)
    ax = _three_d(drawn)
    (surface,) = ax.collections
    Z = yields.to_numpy()
    # independent: matplotlib colours each face by the mean height of its four corners (every cell drawn, no thinning)
    corners = np.array([Z[r:r + 2, c:c + 2].mean() for r in range(Z.shape[0] - 1) for c in range(Z.shape[1] - 1)])
    assert np.allclose(np.asarray(surface.get_array()), corners)
    assert np.isclose(surface.norm.vmin, corners.min()) and np.isclose(surface.norm.vmax, corners.max())
    bar = next(a for a in drawn.axes if a is not ax)
    assert bar.get_ylabel() == "yield" and ax.get_zlabel() == "yield"
    assert ax.get_xlabel() == "maturity" and ax.get_ylabel() == "date"
    dates = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
    assert dates and all(re.fullmatch(r"\d{4}Q\d", t) for t in dates)  # the data's own periods
    lo, hi = ax.get_ylim()
    from matplotlib import dates as mdates

    assert mdates.num2date(lo).year <= 2015 and mdates.num2date(hi).year >= 2020
    assert fig.drawn_height() > Figure((Panel((Line(yields[1.0]),)),)).drawn_height()  # room for the depth
    assert list(fig.data().columns) == [str(c) for c in yields.columns]


def test_a_wireframe_draws_each_row_and_column(yields):
    fig = surface_figure(yields, kind="wireframe", label="yield", color="#D55E00")
    ax = _three_d(_drawn(fig))
    (wires,) = ax.collections
    segments = [np.asarray(s) for s in wires._segments3d]
    rows = [s for s in segments if len(s) == yields.shape[1]]
    assert len(rows) == yields.shape[0]
    for r, line in enumerate(rows):  # each row's heights, as the table has them
        assert np.allclose(line[:, 2], yields.to_numpy()[r])
        assert np.allclose(line[:, 0], yields.columns.to_numpy())
    from matplotlib.colors import to_hex

    assert to_hex(wires.get_color()[0]) == "#d55e00"


def test_the_view_angle_is_recorded_replayed_and_checked(yields):
    fig = surface_figure(yields).view_init(elev=15, azim=-130)
    assert fig.style_steps() == ["view_init(ax=0, elev=15, azim=-130)"]
    ax = _three_d(_drawn(fig))
    assert (ax.elev, ax.azim) == (15, -130)
    theirs = _three_d(_run_standalone(fig.matplotlib_code()))
    assert (theirs.elev, theirs.azim) == (15, -130)
    assert _three_d(_drawn(_saved(fig))).azim == -130
    turned = fig.restyle(0, azim=45)
    assert _three_d(_drawn(turned)).azim == 45 and _three_d(_drawn(turned)).elev == 15
    ns = {"fig": surface_figure(yields)}
    exec("fig = fig." + fig.style[0].code(), ns)  # the step replays as written
    assert ns["fig"].style == fig.style
    flat = Figure((Panel((Line(yields[1.0], "1y"),)),))
    with pytest.raises(SpecificationError, match="panel 0 is flat"):
        flat.view_init(elev=10).draw()
    assert flat.style_skipped(fig) == ["view_init(ax=0, elev=15, azim=-130)"]  # copied only to 3-D panels
    mixed = Figure((Panel((Line(yields[1.0], "1y"),)), Panel((Surface(yields),))), columns=2).view_init(azim=10, ax="all")
    assert _three_d(_drawn(mixed)).azim == 10  # every 3-D panel; the flat one left as it is
    with pytest.raises(SpecificationError, match="view_init"):
        fig.view_init(elev="high")


def test_3d_panels_hold_3d_layers_only_and_take_the_options_that_apply(yields):
    with pytest.raises(SpecificationError, match="a 3-D panel holds surfaces"):
        Panel((Surface(yields), Line(yields[1.0])))
    with pytest.raises(SpecificationError, match="projection is '3d' or ''"):
        Panel((Line(yields[1.0]),), projection="polar")
    assert Panel((Surface(yields),)).projection == "3d" and Panel((Line(yields[1.0]),)).projection == ""
    fig = surface_figure(yields, label="yield")
    with pytest.raises(SpecificationError, match="flat panel"):
        fig.annotate("here", xy=(1, 2)).draw()
    with pytest.raises(SpecificationError, match="log scale is for flat panels"):
        fig.with_options(log=True).draw()
    ax = _three_d(_drawn(fig.with_options(ylim=(0, 6), shade="2016Q1:2017Q1", reference="2=target")))
    assert ax.get_zlim() == (0, 6)  # the values shown are the heights; shading and lines across are for flat panels
    styled = fig.set(zlabel="per cent", zlim=(1, 5)).ticks(axis="z", labelsize=6).grid(visible=False)
    ax = _three_d(_drawn(styled))
    assert ax.get_zlabel() == "per cent" and ax.get_zlim() == (1, 5)
    theirs = _three_d(_run_standalone(styled.matplotlib_code()))
    assert theirs.get_zlabel() == "per cent" and theirs.get_zlim() == (1, 5)
    with pytest.raises(SpecificationError, match="no z axis"):
        Figure((Panel((Line(yields[1.0]),)),)).ticks(axis="z", labelsize=6).draw()


def test_3d_lines_and_points_take_dates_on_any_axis(yields):
    horizon = pd.Series(np.arange(12.0), name="horizon")
    when = pd.Series(pd.PeriodIndex([pd.Period("2016Q3")] * 12), name="date")
    response = pd.Series(np.exp(-0.3 * np.arange(12)), name="response")
    fig = Figure((Panel((Line3d(horizon, when, response, "2016Q3"), Scatter3d(horizon, when + 4, -response, "2017Q3")),),))
    fig = fig.line("2016Q3", color="#CC79A7", linewidth=3)
    drawn = _drawn(fig)
    ax = _three_d(drawn)
    (line,) = ax.lines
    x, y, z = line.get_data_3d()
    from matplotlib import dates as mdates

    assert np.allclose(x, horizon) and np.allclose(z, response)
    assert np.allclose(y, mdates.date2num(pd.Period("2016Q3").to_timestamp()))
    assert line.get_linewidth() == 3 and line.get_color() == "#CC79A7"
    points = ax.collections[0]
    assert np.allclose(np.asarray(points._offsets3d[2]), -response)
    assert np.allclose(np.asarray(points._offsets3d[1]), mdates.date2num(pd.Period("2017Q3").to_timestamp()))
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["2016Q3", "2017Q3"]
    for other in (_saved(fig), ):  # the dates survive saving as periods
        assert isinstance(other.panels[0].layers[0].y.dtype, pd.PeriodDtype)
        assert other.png() == fig.png()
    theirs = _three_d(_run_standalone(fig.matplotlib_code()))
    tx, ty, tz = theirs.lines[0].get_data_3d()
    assert np.allclose(tx, x) and np.allclose(ty, y) and np.allclose(tz, z) and theirs.lines[0].get_color() == "#CC79A7"
    assert np.allclose(np.asarray(theirs.collections[0]._offsets3d[2]), -response)
    assert {"2016Q3: horizon", "2016Q3: date", "2016Q3: response"} <= set(fig.data().columns)
    with pytest.raises(SpecificationError, match="no line 'nothing'"):
        fig.line("nothing", color="red").draw()


# -- contours ---------------------------------------------------------------------------------------------------------


def _bowl():
    grid = np.linspace(-2, 2, 81)
    return pd.DataFrame(np.add.outer(grid**2, grid**2), index=pd.Index(grid, name="y"), columns=pd.Index(grid, name="x"))


def test_contour_lines_lie_on_their_levels():
    bowl = _bowl()  # z = x^2 + y^2: the contour at level c is the circle of radius sqrt(c)
    fig = surface_figure(bowl, kind="contour", levels=(1, 2.25), clabel=True, label="height")
    drawn = _drawn(fig)
    ax = drawn.axes[0]
    (contours,) = ax.collections
    assert list(contours.levels) == [1, 2.25]
    for level, segments in zip(contours.levels, contours.allsegs):
        points = np.vstack(segments)
        assert np.allclose(np.hypot(points[:, 0], points[:, 1]), np.sqrt(level), atol=0.01)  # independent: the circle
    assert sorted({t.get_text() for t in ax.texts}) == ["1", "2.25"]  # each line's level written on it
    assert drawn.axes[1].get_ylabel() == "height"  # the colour bar
    assert ax.get_xlabel() == "x" and ax.get_ylabel() == "y"


def test_filled_contours_cover_the_data_and_take_dated_rows(yields):
    fig = surface_figure(yields, kind="contourf", levels=8, label="yield")
    drawn = _drawn(fig)
    ax = drawn.axes[0]
    (filled,) = ax.collections
    levels = np.asarray(filled.levels)
    assert levels[0] <= yields.to_numpy().min() and levels[-1] >= yields.to_numpy().max()  # every value in a band
    assert 5 <= len(levels) <= 10 and np.allclose(np.diff(levels), np.diff(levels)[0])  # about 8, evenly spaced
    from matplotlib.ticker import MaxNLocator

    assert np.allclose(levels, MaxNLocator(8 + 1).tick_values(yields.to_numpy().min(), yields.to_numpy().max()))
    dates = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
    assert dates and all(re.fullmatch(r"\d{4}Q\d", t) for t in dates)
    assert drawn.axes[1].get_ylabel() == "yield"
    over = Figure((Panel((Contour(yields, levels=8), Contour(yields, levels=8, filled=False, color="k", clabel=True, colorbar=False))),))
    assert len(_drawn(over).axes) == 2  # one colour bar
    spanned = fig.with_options(span=("2016Q1", "2017Q4"))
    assert spanned._panels()[0].layers[0].values.shape == (8, 6)


# -- violins and hexagonal bins ---------------------------------------------------------------------------------------


def test_violins_trace_each_series_density(data):
    from scipy import stats

    fig = data.plot(["gy", "gc", "gi"], kind="violin")
    assert isinstance(fig.panels[0].layers[0], Violin)
    drawn = _drawn(fig)
    ax = drawn.axes[0]
    bodies = [c for c in ax.collections if type(c).__name__ in ("PolyCollection", "FillBetweenPolyCollection")]
    assert len(bodies) == 3
    medians = next(c for c in ax.collections if type(c).__name__ == "LineCollection" and np.isclose(c.get_linewidth()[0], 1.6))
    for k, (name, body) in enumerate(zip(["gy", "gc", "gi"], bodies)):
        values = data[name].dropna().to_numpy()
        assert np.isclose(medians.get_segments()[k][0][1], np.median(values))
        vertices = body.get_paths()[0].vertices
        assert np.isclose(vertices[:, 1].min(), values.min()) and np.isclose(vertices[:, 1].max(), values.max())
        right = vertices[vertices[:, 0] > k + 1]
        kde = stats.gaussian_kde(values, bw_method="scott")  # independent: SciPy's kernel density, Scott's bandwidth
        grid = np.linspace(values.min(), values.max(), 100)
        assert np.allclose(right[:, 0] - (k + 1), 0.35 * kde(right[:, 1]) / kde(grid).max(), atol=1e-6)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["gy", "gc", "gi"]
    each = data.plot(["gy", "gc"], kind="violin", layout="each")
    assert len(each.panels) == 2 and each.panels[1].layers[0].labels == ("gc",)


def test_hexagonal_bins_count_each_observation_once(data):
    fig = data.plot(["gy", "gc"], kind="hexbin", bins=12)
    assert isinstance(fig.panels[0].layers[0], Hexbin) and fig.panels[0].layers[0].gridsize == 12
    drawn = _drawn(fig)
    ax = drawn.axes[0]
    (cells,) = ax.collections
    counts, centres = np.asarray(cells.get_array()), np.asarray(cells.get_offsets())
    x, y = data["gy"].to_numpy(float), data["gc"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    assert counts.sum() == len(x) and counts.min() >= 1  # every observation once; empty hexagons left out
    # independent: each point to its nearest centre, on the hexagons' own scale (the centres' spacing)
    sx = 2 * np.min(np.diff(np.unique(np.round(centres[:, 0], 9))))
    sy = 2 * np.min(np.diff(np.unique(np.round(centres[:, 1], 9))))
    distance = ((x[:, None] - centres[None, :, 0]) / sx) ** 2 + 3 * ((y[:, None] - centres[None, :, 1]) / sy) ** 2
    expected = np.bincount(distance.argmin(axis=1), minlength=len(centres))
    assert np.array_equal(expected, counts)
    assert drawn.axes[1].get_ylabel() == "observations" and ax.get_xlabel() == "gy" and ax.get_ylabel() == "gc"
    with pytest.raises(SpecificationError, match="hexagonal bins need two series"):
        data.plot(["gy"], kind="hexbin")


# -- saving, standalone code, the Plot series dialog ---------------------------------------------------------------------


def _kinds(yields, data):
    return {
        "surface": surface_figure(yields, label="yield").view_init(elev=25, azim=-50),
        "wireframe": surface_figure(yields, kind="wireframe"),
        "contourf": surface_figure(yields, kind="contourf", clabel=True, label="yield"),
        "contour": surface_figure(yields.rename(columns=lambda m: f"{m:g}y"), kind="contour", levels=(2, 3), color="k", clabel=True),
        "violin": data.plot(["gy", "gc"], kind="violin"),
        "hexbin": data.plot(["gy", "gc"], kind="hexbin", bins=15),
        "series surface": data.plot(["gy", "gc", "gi"], kind="surface", span=("2015Q1", None)),
    }


def test_every_new_kind_saves_and_draws_the_same_alone(yields, data):
    for name, fig in _kinds(yields, data).items():
        saved = _saved(fig)
        assert saved.png() == fig.png(), name
        ours, theirs = _drawn(fig), _run_standalone(fig.matplotlib_code())
        assert len(ours.axes) == len(theirs.axes), name
        for a, b in zip(ours.axes, theirs.axes):
            assert a.name == b.name, name
            for ca, cb in zip(a.collections, b.collections):
                assert type(ca) is type(cb), name
                if ca.get_array() is not None:
                    assert np.allclose(np.ma.filled(ca.get_array(), np.nan), np.ma.filled(cb.get_array(), np.nan), equal_nan=True), name
                if hasattr(ca, "levels"):
                    assert np.allclose(ca.levels, cb.levels), name
                if hasattr(ca, "_segments3d"):
                    assert len(ca._segments3d) == len(cb._segments3d), name
                    assert all(np.allclose(np.asarray(sa, float), np.asarray(sb, float)) for sa, sb in zip(ca._segments3d, cb._segments3d)), name
            if a.name == "3d":
                assert (a.elev, a.azim) == (b.elev, b.azim), name
                assert a.get_xlabel() == b.get_xlabel() and a.get_zlabel() == b.get_zlabel(), name
            assert sorted(t.get_text() for t in a.texts) == sorted(t.get_text() for t in b.texts), name  # contour labels


def test_the_plot_series_dialog_offers_the_new_kinds(data):
    from pymetrix.commands import PlotSeries, Session

    s = Session()
    s.objects["data"] = data
    s.run(PlotSeries("gy, gc", kind="hexbin", bins="20", name="h"))
    assert s.history[-1].code == "h = data.plot(['gy', 'gc'], kind='hexbin', bins=20)"
    s.run(PlotSeries("gy, gc, gi", kind="violin", layout="each", name="v"))
    assert s.history[-1].code == "v = data.plot(['gy', 'gc', 'gi'], kind='violin', layout='each')"
    s.run(PlotSeries("gy, gc, gi", kind="surface", start="2010Q1", name="surf"))
    assert s.history[-1].code == "surf = data.plot(['gy', 'gc', 'gi'], kind='surface', span=('2010Q1', None))"
    surface = s["surf"].panels[0].layers[0]
    assert isinstance(surface, Surface) and list(surface.values.columns) == ["gy", "gc", "gi"]
    assert surface.values.index[0] == pd.Period("2010Q1", "Q")
    assert PlotSeries("gy", kind="hexbin").validate(s) == ["hexagonal bins need two series: the x series first, then y"]
    assert PlotSeries("gy", kind="surface").validate(s) == ["a surface needs two or more series, side by side along x (tick them in order)"]
    with pytest.raises(SpecificationError, match="dated"):
        Dataset.from_frame(pd.DataFrame({"id": ["p", "q"], "a": [1.0, 2.0], "b": [3.0, 4.0]}), entity="id").plot(["a", "b"], kind="surface")
    with pytest.raises(SpecificationError, match="kind is surface, wireframe"):
        surface_figure(pd.DataFrame(np.ones((3, 3))), kind="bars")


# -- the app: windows, readouts, notes, the Edit… dialogs -------------------------------------------------------------


def test_new_kinds_open_in_a_window_and_the_edit_dialogs_open(yields, data):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.commands import FigureOptions, Session, StyleFigure
    from pymetrix.figures import readout
    from pymetrix.ui.qt import figure_view

    placed = []
    for name, fig in _kinds(yields, data).items():
        page = figure_view(fig, on_place=lambda *a: placed.append(a))
        canvas = page.canvas
        canvas.draw()
        ax = canvas.figure.axes[0]
        middle = ax.transAxes.transform((0.5, 0.5))
        assert isinstance(readout(ax, *middle), str), name
        page.place.setChecked(True)
        event = type("Click", (), {"inaxes": ax, "xdata": 0.0, "ydata": 0.0})()
        page.clicked(event)
        if ax.name == "3d":
            assert not placed and "flat panels" in page.readout.text()
        else:
            assert placed and placed.pop()[0] == 0
        page.deleteLater()
    s = Session()
    s.objects["fig"] = _kinds(yields, data)["surface"]
    for dialog in (FigureOptions.for_figure("fig", s["fig"], is_figure=True), StyleFigure.for_figure("fig", s["fig"], is_figure=True)):
        assert dialog.validate(s) == []
    s.run(FigureOptions("fig", title="Yields", name="fig"))
    assert s["fig"].title == "Yields" and s["fig"].style_steps() == ["view_init(ax=0, elev=25, azim=-50)"]
    s.run(StyleFigure("fig", panel="0", ylabel="when", name="fig"))
    assert _three_d(_drawn(s["fig"])).get_ylabel() == "when"
