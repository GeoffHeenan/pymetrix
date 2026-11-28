"""Figures: charts as data, drawn with matplotlib.

A ``Figure`` is panels of layers (lines, bands, bars, scatter, histograms,
reference lines). Results build their own (``fit.plot()``, ``irf.plot()``,
``lp.plot()``...), ``data.plot(...)`` charts series, and the desktop app shows
them in object windows, notebooks display them inline, and ``save`` writes
PNG, SVG or PDF. Building a figure needs no plotting library; drawing it
uses matplotlib (a core dependency).

Options change how a figure is drawn, not what it shows:

    fig.with_options(theme="policy", span=("2020Q1", None), subtitle="Percent, year on year",
                     footnote="Source: national statistics office; staff projections.")

``span`` limits the dates drawn (and ``data()``); ``theme`` names a theme (see ``pymetrix.themes``), or
the session's default is used.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from ._api import api, renamed_arguments
from .errors import SpecificationError
from .results import deprecated_alias
from .themes import OKABE_ITO, Theme, get_theme

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["SIZES", "QQ", "Band", "Bars", "Contour", "Ecdf", "Estimates", "Fan", "Figure", "Heatmap", "Hexbin", "Hist2d", "Histogram",
           "Line", "Line3d", "Panel", "Rule", "Scatter", "Scatter3d", "Surface", "Violin", "house_style", "set_house_style",
           "coefficient_figure", "contributions_figure", "heatmap_figure", "surface_figure", "vintages_figure"]

PALETTE = OKABE_ITO  # the colours results give their layers; a theme maps them to its own palette


@dataclass(frozen=True, eq=False)
class Line:
    y: pd.Series
    label: str = ""
    color: str | None = None
    style: str = "-"
    width: float = 1.6
    step: bool = False  # a step line: each value held until the next (policy rates, regimes)
    axis: str = "left"  # "right": against a second y-axis on the right


class _Columns(dict):
    """A figure's data columns by name: a name given twice keeps both, the later as "name (2)", "name (3)"..."""

    def __setitem__(self, key: str, value: Any) -> None:
        name, n = key, 2
        while name in self:
            name, n = f"{key} ({n})", n + 1
        super().__setitem__(name, value)


@dataclass(frozen=True, eq=False)
class Band:
    """A shaded interval, e.g. confidence bands."""

    lower: pd.Series
    upper: pd.Series
    label: str = ""
    color: str | None = None
    alpha: float = 0.18
    axis: str = "left"
    step: bool = False  # a step band: each bound held until the next point, as a step line (survival curves)


@dataclass(frozen=True, eq=False)
class Bars:
    """Bars of a series. Unstacked bars in the same panel stand side by side; ``horizontal`` draws them across."""

    y: pd.Series
    label: str = ""
    color: str | None = None
    stacked: bool = False  # on top of the panel's other stacked bars (positive above, negative below), e.g. contributions
    horizontal: bool = False
    axis: str = "left"


@dataclass(frozen=True, eq=False)
class Box:
    """Box plots: for each series, the median, the quartiles (the box), whiskers to 1.5 interquartile ranges, and the
    observations beyond them."""

    series: tuple[pd.Series, ...]
    labels: tuple[str, ...] = ()


@dataclass(frozen=True, eq=False)
class Scatter:
    x: pd.Series
    y: pd.Series
    label: str = ""
    fit: bool = False  # add the least-squares line


@dataclass(frozen=True, eq=False)
class Histogram:
    values: pd.Series
    bins: int = 30
    normal: bool = True  # overlay the normal density with the same mean and standard deviation
    label: str = ""


@dataclass(frozen=True, eq=False)
class Ecdf:
    """The empirical cumulative distribution of a series (matplotlib's ``ecdf``): at each value, the share of
    observations at or below it, as a step. ``complementary`` draws the share above it instead. Several in one panel
    compare distributions without choosing bins."""

    values: pd.Series
    label: str = ""
    color: str | None = None
    complementary: bool = False


@dataclass(frozen=True, eq=False)
class QQ:
    """A normal Q-Q plot: the values, standardised and sorted, against the normal distribution's quantiles at the same
    probabilities, (i - 0.5) / n, with the 45-degree line they lie along if they are normal. Points above it on the
    right and below it on the left mean fat tails; a bow means skew."""

    values: pd.Series
    label: str = ""


def qq_points(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """The Q-Q plot's points: (normal quantiles, standardised values sorted)."""
    from scipy import stats

    v = values.dropna().to_numpy(float)
    if len(v) < 3:
        raise SpecificationError("a Q-Q plot needs at least three values")
    sd = v.std(ddof=1)
    z = np.sort((v - v.mean()) / (sd if sd > 0 else 1.0))
    return stats.norm.ppf((np.arange(1, len(v) + 1) - 0.5) / len(v)), z


@dataclass(frozen=True, eq=False)
class Hist2d:
    """A two-dimensional histogram (matplotlib's ``hist2d``): the plane of ``x`` against ``y`` cut into ``bins`` by
    ``bins`` cells, each coloured by how many observations fall in it, with a colour bar. A scatter plot for many
    observations, where points would hide each other."""

    x: pd.Series
    y: pd.Series
    bins: int = 30
    cmap: str = "Blues"
    label: str = "observations"  # the colour bar's label


@dataclass(frozen=True, eq=False)
class Estimates:
    """Point estimates with intervals, as matplotlib's ``errorbar`` draws them: a marker at each estimate and a line
    across its interval. Named estimates (coefficients, models) are listed down the side with a dashed line at zero;
    dated ones (a coefficient over time) run along the dates. Several in one panel stand side by side, as bars do."""

    estimate: pd.Series
    lower: pd.Series
    upper: pd.Series
    label: str = ""
    color: str | None = None
    marker: str = "o"


@dataclass(frozen=True, eq=False)
class Heatmap:
    """A table of values as coloured cells (matplotlib's ``imshow``) with a colour bar: correlations, responses by
    shock and horizon, p-values. Values of both signs (or with ``center`` given) get a diverging scale, symmetric about
    the centre (zero by default); others a sequential one. ``annotate`` writes each value in its cell, as ``fmt``
    formats it, when there are at most 400 cells, in dark or white type, whichever reads better on the cell.

    ``scale`` "table" colours every cell on one scale; "column" colours each column on its own (its smallest value at
    one end of the colour map, its largest at the other; ``center``, ``vmin`` and ``vmax`` do not apply), for a table
    whose columns are in different units, such as a posterior summary's means beside its effective draws: the cells
    still show their own values, and a note under the table takes the colour bar's place. "auto" picks "column" when
    one column's typical size (the median of its absolute values) is more than ten times the largest absolute value
    in another, and "table" otherwise, and always for a square matrix with the same labels on its rows and columns
    (correlations, covariances), for columns that are horizons, lags or dates (responses by horizon), for columns
    that all take values above and below zero (responses by shock, residuals), and for shares
    (no value below zero, and every row, or every column, adding up to the same whole multiple of 1 or of 100:
    variance decompositions)."""

    values: pd.DataFrame
    label: str = ""  # the colour bar's label
    cmap: str = ""  # a matplotlib colour map; empty: RdBu_r for a diverging scale, Blues otherwise
    center: float | None = None
    vmin: float | None = None
    vmax: float | None = None
    annotate: bool = True
    fmt: str = "{:.2f}"
    scale: Literal["table", "column", "auto"] = "table"

    def __post_init__(self) -> None:
        if self.scale not in ("table", "column", "auto"):
            raise SpecificationError(f"a heatmap's scale is 'table', 'column' or 'auto', not {self.scale!r}")


@dataclass(frozen=True, eq=False)
class Hexbin:
    """Hexagonal bins (matplotlib's ``hexbin``): the plane of ``x`` against ``y`` cut into hexagons, ``gridsize``
    across, each coloured by how many observations fall in it, with a colour bar; empty hexagons are left out. For
    large scatters, as ``Hist2d``, with cells that tile the plane more evenly."""

    x: pd.Series
    y: pd.Series
    gridsize: int = 30
    cmap: str = "Blues"
    label: str = "observations"  # the colour bar's label


@dataclass(frozen=True, eq=False)
class Violin:
    """Violin plots (matplotlib's ``violinplot``): for each series, its kernel density (Gaussian, Scott's
    bandwidth) drawn mirrored about an upright line from its lowest to its highest value, with the median marked.
    A box plot that shows the distribution's shape: two humps, a long tail."""

    series: tuple[pd.Series, ...]
    labels: tuple[str, ...] = ()
    medians: bool = True


@dataclass(frozen=True, eq=False)
class Contour:
    """Contours of a table over a grid (matplotlib's ``contourf``, filled, or ``contour``, lines): the table's columns
    along x, its rows along y (numbers, dates or names), each cell's value the height. ``levels``: how many
    (matplotlib chooses round values) or the levels themselves; ``clabel`` writes each line's level on it (filled
    contours get thin lines at their levels to carry them, as matplotlib advises); ``color``
    draws the lines in one colour instead of the colour map (lines over a filled contour); ``colorbar`` with
    ``label`` names the levels."""

    values: pd.DataFrame
    levels: int | tuple[float, ...] = 10
    filled: bool = True
    clabel: bool = False
    cmap: str = "viridis"
    color: str | None = None
    colorbar: bool = True
    label: str = ""  # the colour bar's label


@dataclass(frozen=True, eq=False)
class Surface:
    """A surface over a grid, on a 3-D panel (matplotlib's ``plot_surface``, or ``plot_wireframe`` with
    ``wireframe``): the table's columns along x (maturities, horizons, strikes), its rows along y (dates, expiries),
    each cell's value the height z. A surface is coloured by height, with a colour bar named ``label``; a wireframe is
    lines in ``color``. Every cell is drawn (no thinning). Turn it with ``Figure.view_init``."""

    values: pd.DataFrame
    label: str = ""
    cmap: str = "viridis"
    wireframe: bool = False
    color: str | None = None
    alpha: float = 1.0


@dataclass(frozen=True, eq=False)
class Line3d:
    """A line through the points (x, y, z) on a 3-D panel (``Axes3D.plot``). A series of periods on any axis is
    drawn as dates. Styled as a line (``fig.line(label, ...)``)."""

    x: pd.Series
    y: pd.Series
    z: pd.Series
    label: str = ""
    color: str | None = None
    width: float = 1.6


@dataclass(frozen=True, eq=False)
class Scatter3d:
    """Points (x, y, z) on a 3-D panel (``Axes3D.scatter``); a series of periods on any axis is drawn as dates."""

    x: pd.Series
    y: pd.Series
    z: pd.Series
    label: str = ""
    color: str | None = None


_SOLID = (Surface, Line3d, Scatter3d)  # the layers of a 3-D panel


@dataclass(frozen=True, eq=False)
class Fan:
    """A fan chart: a central path, bands around it at several coverages (the inner ones darker, as each band
    overlaps the wider ones), and a dashed line where the bands, the forecast, begin. Drawn as its parts: a band per
    coverage (labelled "90%", "60%"...), the central line (``label``) and the start line, so each can be styled.

    ``Fan.from_frame(frame, "gdp")`` reads the bands from ``gdp_lo90``/``gdp_hi90`` columns (any coverages present), or
    makes normal bands from a ``gdp_se`` column; ``Fan.normal(center, se)`` from a standard error."""

    center: pd.Series
    lower: tuple[pd.Series, ...]  # one per coverage, in the order of ``coverages``
    upper: tuple[pd.Series, ...]
    coverages: tuple[float, ...] = (0.9, 0.6, 0.3)
    label: str = "central"
    color: str | None = None
    start: bool = True  # a dashed line where the bands begin
    axis: str = "left"

    def __post_init__(self) -> None:
        if not (len(self.lower) == len(self.upper) == len(self.coverages)) or not self.coverages:
            raise SpecificationError("a fan needs a lower and an upper band for each coverage")
        if any(not 0 < c < 1 for c in self.coverages):
            raise SpecificationError(f"coverages are between 0 and 1, e.g. 0.9 for 90%; got {list(self.coverages)}")

    @classmethod
    def normal(cls, center: pd.Series, se: pd.Series, coverages: Sequence[float] = (0.3, 0.6, 0.9), **options: Any) -> Fan:
        """Normal bands: the centre plus and minus the normal quantile for each coverage, times ``se``."""
        from scipy import stats

        cov = tuple(sorted((float(c) for c in coverages), reverse=True))
        z = [float(stats.norm.ppf(0.5 + c / 2)) for c in cov]
        return cls(center, tuple(center - k * se for k in z), tuple(center + k * se for k in z), cov, **options)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, name: str, coverages: Sequence[float] | None = None, **options: Any) -> Fan:
        """The fan of column ``name``: bands from ``{name}_lo{pct}`` and ``{name}_hi{pct}`` columns (those of
        ``coverages``, or every one present), else normal bands from ``{name}_se`` (coverages 30%, 60% and 90% by
        default)."""
        import re

        if name not in frame:
            raise SpecificationError(f"no series {name!r} for the fan's centre")
        present = sorted({int(m.group(1)) for c in frame.columns if (m := re.fullmatch(re.escape(name) + r"_lo(\d+)", str(c)))
                          and f"{name}_hi{m.group(1)}" in frame}, reverse=True)
        wanted = [int(round(100 * c)) for c in coverages] if coverages else present
        found = [pct for pct in sorted(wanted, reverse=True) if pct in present]
        options.setdefault("label", name)
        if found and len(found) == len(wanted):
            return cls(frame[name], tuple(frame[f"{name}_lo{pct}"] for pct in found),
                       tuple(frame[f"{name}_hi{pct}"] for pct in found), tuple(pct / 100 for pct in found), **options)
        if f"{name}_se" in frame and frame[f"{name}_se"].notna().any():
            return cls.normal(frame[name], frame[f"{name}_se"], coverages or (0.3, 0.6, 0.9), **options)
        raise SpecificationError(f"no bands for {name!r}: give {name}_lo90 and {name}_hi90 columns (any coverages), "
                                 f"or {name}_se for normal bands")

    def parts(self) -> tuple[Any, ...]:
        """What is drawn: the bands, widest first, the central line, and the start line."""
        color = self.color or PALETTE[0]
        order = sorted(range(len(self.coverages)), key=lambda k: -self.coverages[k])
        bands = [Band(self.lower[k], self.upper[k], f"{int(round(100 * self.coverages[k]))}%", color=color, alpha=0.22,
                      axis=self.axis) for k in order]
        out: list[Any] = [*bands, Line(self.center, self.label, color=color, axis=self.axis)]
        known = pd.concat([s.notna() for s in self.lower], axis=1).any(axis=1)
        if self.start and known.any() and isinstance(self.center.index, pd.PeriodIndex):
            first = known[known].index[0]
            if first != self.center.index[0]:  # bands from the start: no forecast to mark
                out.append(Rule(str(first), style="--", color="#888888", vertical=True))
        return tuple(out)


@dataclass(frozen=True, eq=False)
class Rule:
    """A horizontal reference line (zero, or a significance band); ``vertical`` for one across the x-axis instead (the
    zero of horizontal bars, or a date, given as a period, on a dated axis)."""

    y: float = 0.0
    style: str = "-"
    color: str = "#888888"
    vertical: bool = False


@dataclass(frozen=True, eq=False)
class Panel:
    """One panel of a figure (a matplotlib Axes): its layers, title and axis labels. A panel of surfaces, 3-D lines
    and 3-D points is a 3-D panel (``projection="3d"``), with a ``zlabel``; it holds no flat layers."""

    layers: tuple[Any, ...]
    title: str = ""
    ylabel: str = ""
    xlabel: str = ""
    legend: bool = True
    ylabel_right: str = ""  # for layers drawn against the right-hand axis
    zlabel: str = ""  # a 3-D panel's height
    projection: str = ""  # "3d" for surfaces, 3-D lines and points (set by those layers); "" for a flat panel
    legend_loc: str = ""  # where the legend goes: "" the best place inside, "above" outside, over the panel's right end
    # (clear of the data and the dates), or any matplotlib loc ("upper left"...)

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        solid = sum(isinstance(layer, _SOLID) for layer in self.layers)
        projection = self.projection or ("3d" if solid else "")
        if projection not in ("", "3d"):
            raise SpecificationError(f"a panel's projection is '3d' or '' (flat), not {self.projection!r}")
        if projection == "3d" and solid != len(self.layers):
            flat = sorted({type(layer).__name__ for layer in self.layers if not isinstance(layer, _SOLID)})
            raise SpecificationError(f"a 3-D panel holds surfaces and 3-D lines and points (Surface, Line3d, Scatter3d), "
                                     f"not {', '.join(flat)}: put those in a panel of their own")
        object.__setattr__(self, "projection", projection)

    def drawn(self) -> tuple[Any, ...]:
        """The layers as drawn: a fan as its bands, central line and start line."""
        return tuple(part for layer in self.layers for part in (layer.parts() if isinstance(layer, Fan) else (layer,)))


@dataclass(frozen=True)
class StyleCall:
    """One styling call recorded on a figure, as matplotlib would make it: ``kind`` names the call (``line``,
    ``bars``, ``band``, ``set``, ``ticks``, ``grid``, ``spines``, ``legend``, ``layout``, ``view_init``...), ``where`` says which panel
    and which series or part, and ``props`` are matplotlib's keyword arguments (aliases resolved; plain data only).
    See ``docs/notes/figure-styling.md``."""

    kind: str  # was target
    where: tuple[Any, ...] = ()
    props: Any = None  # a dict; kept as given, compared by value

    def __post_init__(self) -> None:
        def frozen(v: Any) -> Any:  # one form for sequences, as the recorded code writes them: (8, 4), not [8, 4]
            if isinstance(v, (list, tuple)):
                return tuple(frozen(x) for x in v)
            return {k: frozen(x) for k, x in v.items()} if isinstance(v, dict) else v

        object.__setattr__(self, "where", tuple(self.where))
        object.__setattr__(self, "props", {k: frozen(v) for k, v in dict(self.props or {}).items()})

    def __hash__(self) -> int:
        return hash((self.kind, self.where, repr(sorted(self.props.items()))))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "where": list(self.where), "props": dict(self.props)}

    def code(self) -> str:
        """The Figure method call that records this: ``line('gc', color='C3')``, ``set(ax=0, ylabel='%')``."""
        def kw(props: dict[str, Any]) -> list[str]:
            return [f"{k}={v!r}" for k, v in props.items()]

        if self.kind in ("line", "bars", "band"):
            ax, label = self.where
            parts = [repr(label), *([f"ax={ax!r}"] if ax is not None else []), *kw(self.props)]
        elif self.kind == "call":
            ax, side = self.where
            parts = [repr(self.props["method"]), *(repr(a) for a in self.props.get("args", ())), f"ax={ax!r}",
                     *(["side='right'"] if side == "right" else []), *kw(self.props.get("kwargs", {}))]
        elif self.kind == "layout":
            parts = kw(self.props)
        else:
            parts = [f"ax={self.where[0]!r}"]
            if self.kind == "set" and self.where[1] == "right":
                parts.append("side='right'")
            elif self.kind == "ticks":
                parts.append(f"axis={self.where[1]!r}")
            elif self.kind == "spines":
                parts.append(f"which={self.where[1]!r}")
            parts = [*(kw(self.props) if self.kind in _ADDED else []), *parts, *([] if self.kind in _ADDED else kw(self.props))]
        return f"{self.kind}({', '.join(parts)})"

    target = deprecated_alias("kind", "target", why="the name of the matplotlib call; target is its old name")


StyleCall.__init__ = renamed_arguments(target="kind")(StyleCall.__init__)  # type: ignore[method-assign]


def _style_call(c: StyleCall | Mapping[str, Any]) -> StyleCall:
    """A style call, or its dict (``to_dict``), including one saved before ``target`` became ``kind``."""
    if isinstance(c, StyleCall):
        return c
    c = dict(c)
    if "target" in c:
        c["kind"] = c.pop("target")
    return StyleCall(**c)


@dataclass(frozen=True, eq=False)
class Figure:
    """Panels of layers, laid out ``columns`` across; ``width`` in inches. ``subtitle``, ``footnote``, ``theme`` and
    ``span`` (the dates drawn) are options: see ``with_options``."""

    panels: tuple[Panel, ...]
    title: str = ""
    columns: int = 1
    width: float = 7.5
    height: float | None = None
    subtitle: str = ""
    footnote: str = ""
    theme: str | Theme | None = None  # None: the session's default theme
    span: tuple[Any, Any] | None = None  # (start, end) periods; None at either end for open
    shade: tuple[tuple[str, str, str], ...] = ()  # (start, end, label) periods shaded in every dated panel, e.g. recessions
    events: tuple[tuple[str, str], ...] = ()  # (period, label): a dashed vertical line, labelled, in every dated panel
    reference: tuple[tuple[float, str], ...] = ()  # (value, label): a dotted line across every panel, e.g. a target
    log: bool = False  # a log scale for the values
    ylim: tuple[float | None, float | None] | None = None  # (lowest, highest) value shown; None at either end: automatic
    sharey: bool = False  # one value scale for all panels
    bar_labels: bool = False  # each bar's value written on it
    line_labels: bool = False  # each line named at its end, in its colour, instead of in the legend
    style: tuple[StyleCall, ...] = ()  # matplotlib calls recorded as data, applied in order when drawn (fig.line, fig.set...)
    house: bool = True  # drawn with the house style first (set_house_style), where it applies
    ybreak: tuple[float, float] | None = None  # (low, high): the values between left out, the value axis split in two
    xbreak: tuple[Any, Any] | None = None  # (first, last): the periods left out, the time axis split in two

    def __post_init__(self) -> None:
        object.__setattr__(self, "panels", tuple(self.panels))
        object.__setattr__(self, "ybreak", value_break(self.ybreak))
        object.__setattr__(self, "xbreak", period_break(self.xbreak))
        if self.ybreak is not None and self.xbreak is not None:
            raise SpecificationError("split one axis per figure: the values (ybreak) or the dates (xbreak), not both")
        object.__setattr__(self, "shade", shade_spans(self.shade))
        object.__setattr__(self, "events", event_marks(self.events))
        object.__setattr__(self, "reference", reference_lines(self.reference))
        object.__setattr__(self, "ylim", value_limits(self.ylim))
        object.__setattr__(self, "style", tuple(_style_call(c) for c in self.style))
        if not self.panels:
            raise SpecificationError("a figure needs at least one panel")

    @api()
    def with_options(self, **options: Any) -> Figure:
        """A copy drawn differently. The data are unchanged.

        ``title``, ``subtitle``, ``footnote``, ``theme``, ``span``, ``columns``, ``width``, ``height``;
        ``shade`` (periods to shade: ``"2008Q3:2009Q2=GFC, 2020Q1:2020Q2"``); ``events`` (labelled dates:
        ``"2016Q2=Brexit vote"``); ``reference`` (labelled values: ``"2=inflation target"``); ``log`` (a log scale);
        ``ylim`` ((lowest, highest), either None); ``sharey`` (one scale for all panels); ``bar_labels`` (values on
        the bars); ``line_labels`` (each line named at its end instead of in the legend); ``house`` (False: without
        the house style); ``ybreak`` ((low, high), or ``"5:15"``: the values between left out, each panel whose
        values span them drawn in two parts, one above the other, with break marks: for a spike that squashes the
        rest); ``xbreak`` ((first, last) periods, or ``"1986Q1:2004Q4"``: those periods left out, each dated panel
        that spans them drawn in two parts side by side). One of the two per figure; a split value axis is for
        panels without a right-hand axis."""
        allowed = {"title", "subtitle", "footnote", "theme", "span", "columns", "width", "height", "shade", "events",
                   "reference", "log", "ylim", "sharey", "bar_labels", "line_labels", "house", "ybreak", "xbreak"}
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise TypeError(f"no figure option {', '.join(unknown)}; options: {', '.join(sorted(allowed))}")
        if options.get("theme") is not None:
            get_theme(options["theme"])  # an unknown name fails now, not when drawn
        return replace(self, **options)

    # styling, in matplotlib's words: each returns a new figure with the call recorded (docs/notes/figure-styling.md)

    def _styled(self, target: str, where: tuple[Any, ...], props: dict[str, Any]) -> Figure:
        call = StyleCall(target, where, _check_style(target, where, props))
        return replace(self, style=(*self.style, call))

    @api()
    def line(self, label: str, ax: int | str | None = None, **props: Any) -> Figure:
        """Style the line of the series ``label``, as ``Line2D.set`` takes it: ``color``, ``linestyle``, ``linewidth``,
        ``marker``, ``markersize``, ``alpha``, ``zorder`` and the rest (aliases such as ``lw`` and ``ls`` too).
        ``ax``: the panel (a position or its title); by default, wherever the series is drawn."""
        return self._styled("line", (ax, str(label)), props)

    @api()
    def bars(self, label: str, ax: int | str | None = None, **props: Any) -> Figure:
        """Style the bars of the series ``label``, as ``Rectangle.set`` takes it: ``color``, ``edgecolor``,
        ``linewidth``, ``hatch``, ``alpha``..."""
        return self._styled("bars", (ax, str(label)), props)

    @api()
    def band(self, label: str, ax: int | str | None = None, **props: Any) -> Figure:
        """Style the shaded band (area) of ``label``, as a ``PolyCollection`` takes it: ``facecolor``, ``alpha``,
        ``hatch``, ``edgecolor``..."""
        return self._styled("band", (ax, str(label)), props)

    @api()
    def set(self, ax: int | str = 0, side: str = "left", **props: Any) -> Figure:
        """Set a panel's axes, as ``Axes.set`` takes it: ``xlim``, ``ylim`` (dates as periods, e.g. ("2010Q1",
        "2020Q4")), ``xscale``, ``yscale``, ``xlabel``, ``ylabel``, ``title``, ``aspect``... ``ax``: the panel (a position,
        its title, or "all"); ``side="right"``: its right-hand axis."""
        return self._styled("set", (ax, side), props)

    @api()
    def ticks(self, ax: int | str = 0, axis: str = "both", format: str | None = None, **props: Any) -> Figure:
        """Ticks, as ``Axes.tick_params`` takes them (``labelsize``, ``rotation``, ``direction``, ``length``...), and
        ``format``, a format string for the tick labels as ``StrMethodFormatter`` takes it: ``"{x:,.0f}"``, ``"{x:.0%}"``."""
        return self._styled("ticks", (ax, axis), {**props, **({"format": format} if format is not None else {})})

    @api()
    def grid(self, ax: int | str = 0, visible: bool = True, **props: Any) -> Figure:
        """Grid lines, as ``Axes.grid`` takes them: ``visible``, ``which``, ``axis``, ``color``, ``linestyle``, ``alpha``..."""
        return self._styled("grid", (ax,), {"visible": visible, **props})

    @api()
    def spines(self, ax: int | str = 0, which: str = "all", **props: Any) -> Figure:
        """The frame's sides (``left``, ``right``, ``top``, ``bottom`` or ``all``), as ``Spine.set`` takes them:
        ``visible``, ``color``, ``linewidth``..."""
        return self._styled("spines", (ax, which), props)

    @api()
    def legend(self, ax: int | str = 0, **props: Any) -> Figure:
        """The legend, as ``Axes.legend`` takes it: ``loc``, ``ncols``, ``frameon``, ``fontsize``, ``title``,
        ``bbox_to_anchor``...; ``visible=False`` removes it."""
        return self._styled("legend", (ax,), props)

    @api()
    def annotate(self, text: str, xy: Any, xytext: Any = None, ax: int | str = 0, **props: Any) -> Figure:
        """A note at ``xy``, as ``Axes.annotate`` takes it: ``fig.annotate("Lehman", xy=("2008Q3", 3.2),
        xytext=("2008Q3", 6), arrowprops=dict(arrowstyle="->"))``. Dates are the data's periods; ``textcoords="offset
        points"`` puts the text that many points from ``xy`` (``xytext=(15, 20)``); ``fontsize``, ``color``, ``ha``... too."""
        return self._styled("annotate", (ax,), {"text": str(text), "xy": xy, **({"xytext": xytext} if xytext is not None else {}), **props})

    @api()
    def text(self, x: Any, y: Any, s: str, ax: int | str = 0, **props: Any) -> Figure:
        """Text at (``x``, ``y``), as ``Axes.text`` takes it (x a period on a dated axis); ``fontsize``, ``color``, ``ha``,
        ``va``, ``rotation``, ``bbox``..."""
        return self._styled("text", (ax,), {"x": x, "y": y, "s": str(s), **props})

    @api()
    def axhline(self, y: float = 0, ax: int | str = 0, **props: Any) -> Figure:
        """A horizontal line across the panel at value ``y``, as ``Axes.axhline`` takes it (``color``, ``linestyle``...)."""
        return self._styled("axhline", (ax,), {"y": y, **props})

    @api()
    def axvline(self, x: Any, ax: int | str = 0, **props: Any) -> Figure:
        """A vertical line at ``x`` (a period on a dated axis), as ``Axes.axvline`` takes it."""
        return self._styled("axvline", (ax,), {"x": x, **props})

    @api()
    def axhspan(self, ymin: float, ymax: float, ax: int | str = 0, **props: Any) -> Figure:
        """A shaded band between two values, as ``Axes.axhspan`` takes it (``color``, ``alpha``...)."""
        return self._styled("axhspan", (ax,), {"ymin": ymin, "ymax": ymax, **props})

    @api()
    def axvspan(self, xmin: Any, xmax: Any, ax: int | str = 0, **props: Any) -> Figure:
        """A shaded band between two dates (periods; the whole of the last), as ``Axes.axvspan`` takes it."""
        return self._styled("axvspan", (ax,), {"xmin": xmin, "xmax": xmax, **props})

    @api()
    def change(self, series: str, start: Any, end: Any, show: str = "both", units: str = "auto", annualise: bool = False,
               ax: int | str | None = None, **props: Any) -> Figure:
        """The change in ``series`` between two points, drawn as a double-headed arrow (``style="bracket"``: a bracket)
        from one to the other, labelled at its middle: ``fig.change("unemployment", "2019Q4", "2020Q2")``.

        ``start`` and ``end`` are periods on a dated axis (numbers otherwise); the values are read from the series as
        drawn each time the figure is, so the label follows the data. ``show``: ``"change"``, ``"percent"`` (percent
        change) or ``"both"``. ``units`` for the change: ``"auto"`` (percentage points, ``pp``, when the panel's axis is
        a rate or a percentage, else none), ``"pp"``, ``"%"`` or ``"none"``. ``annualise=True`` adds the rate a year
        (compound for the percent change). ``color``, ``fontsize``, ``linewidth`` and ``style`` style it. ``ax``: the
        panel; by default wherever the series is drawn."""
        props = {"series": str(series), "start": str(start), "end": str(end), "show": show, "units": units, **props}
        if annualise:
            props["annualise"] = True
        return self._styled("change", (ax,), props)

    @api()
    def view_init(self, elev: float | None = None, azim: float | None = None, roll: float | None = None, ax: int | str = 0) -> Figure:
        """Turn a 3-D panel, as ``Axes3D.view_init`` does: ``elev``, the angle above the x-y plane, and ``azim``, the
        angle around the vertical axis, in degrees (matplotlib's default view is elev=30, azim=-60); ``roll`` about the
        line of sight. ``ax``: the panel (a position, its title, or "all", every 3-D panel)."""
        props = {k: v for k, v in (("elev", elev), ("azim", azim), ("roll", roll)) if v is not None}
        return self._styled("view_init", (ax,), props)

    @api()
    def call(self, method: str, *args: Any, ax: int | str = 0, side: str = "left", **kwargs: Any) -> Figure:
        """Any other Axes method that styles or annotates a panel, recorded as data: ``fig.call("set_aspect", "auto")``,
        ``fig.call("ticklabel_format", axis="y", style="sci", scilimits=(0, 0))``, ``fig.call("minorticks_on")``. The
        arguments are matplotlib's own, as plain data (positions in matplotlib's numbers). ``pymetrix.figure_properties.
        axes_methods()`` lists the methods: those that plot more data, clear things or need Python objects are not among
        them."""
        return self._styled("call", (ax, side), {"method": str(method), "args": tuple(args), "kwargs": dict(kwargs)})

    # the recorded styles: listed, removed, reordered, changed, and copied from another figure

    def style_steps(self) -> list[str]:
        """The recorded style calls, in the order applied, each as the call that records it: ``line('gc',
        color='C3')``, ``set(ax=0, ylabel='%')``, ``annotate(text='Lehman', xy=('2008Q3', 3.2), ax=0)``. Their positions
        are the step numbers ``without_style``, ``reorder_style`` and ``restyle`` take."""
        return [call.code() for call in self.style]

    def _step(self, step: int) -> int:
        n = len(self.style)
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or not -n <= int(step) < n:
            raise SpecificationError(f"no style step {step!r}: this figure has {n}" + (f" (0 to {n - 1})" if n else ""))
        return int(step) % n

    @api()
    def without_style(self, *steps: int) -> Figure:
        """A copy without the style calls at these positions (as ``style_steps`` numbers them; -1 is the last)."""
        drop = {self._step(s) for s in steps}
        return replace(self, style=tuple(call for k, call in enumerate(self.style) if k not in drop))

    @api()
    def reorder_style(self, order: Sequence[int]) -> Figure:
        """A copy with the style calls in this order, by position; those not listed are removed. Calls apply in order,
        so where two set the same thing, the later one shows."""
        positions = [self._step(s) for s in order]
        if len(set(positions)) != len(positions):
            raise SpecificationError(f"a style step is listed twice in {list(order)}")
        return replace(self, style=tuple(self.style[k] for k in positions))

    @api()
    def restyle(self, step: int, replace_all: bool = False, **props: Any) -> Figure:
        """A copy with style call ``step``'s keyword arguments changed: ``fig.restyle(0, color="C2")``. ``None``
        removes one; ``replace_all=True`` replaces them all with ``props``. The call's series, panel and place in the
        order are kept, and matplotlib checks the new values as when it was first set."""
        k = self._step(step)
        call = self.style[k]
        old = dict(call.props["kwargs"] if call.kind == "call" else call.props)
        new = {key: v for key, v in ({} if replace_all else old).items() if key not in props}
        new.update({key: v for key, v in props.items() if v is not None})
        changed = {**call.props, "kwargs": new} if call.kind == "call" else new
        if call.kind in _ADDED:
            missing = [key for key in _ADDED_NEEDS[call.kind] if key not in changed]
            if missing:
                raise SpecificationError(f"{call.kind} needs {', '.join(missing)}")
        updated = StyleCall(call.kind, call.where, _check_style(call.kind, call.where, changed))
        return replace(self, style=(*self.style[:k], updated, *self.style[k + 1:]))

    @api()
    def with_style_of(self, other: Figure | Sequence[StyleCall], notes: bool = False) -> Figure:
        """A copy styled as ``other`` is: its recorded style calls added after this figure's own, where they apply.

        - A series' call (``line``, ``bars``, ``band``) is copied if this figure draws that series that way.
        - A panel's call is copied if this figure has that panel (by position or title) and, for a right-hand axis,
          if the panel has one; ``layout`` always is.
        - Notes, text, lines and bands (``annotate``, ``text``, ``axhline``...) mark the other chart's data, so they
          are copied only with ``notes=True``.

        ``style_skipped(other)`` lists the calls left out."""
        calls = other.style if isinstance(other, Figure) else tuple(other)
        return replace(self, style=(*self.style, *(c for c in calls if _style_fits(self, c, notes))))

    def style_skipped(self, other: Figure | Sequence[StyleCall], notes: bool = False) -> list[str]:
        """The style calls of ``other`` that ``with_style_of`` would leave out here."""
        calls = other.style if isinstance(other, Figure) else tuple(other)
        return [c.code() for c in calls if not _style_fits(self, c, notes)]

    def to_matplotlib(self) -> Any:
        """The drawn ``matplotlib.figure.Figure``, to finish by hand. What is done to it is not recorded: record changes
        with the style calls (``line``, ``set``, ``annotate``, ``call``...) instead."""
        return self.draw()

    def matplotlib_code(self) -> str:
        """A standalone script (NumPy, pandas and matplotlib only) that draws this figure from its numbers, every
        option and style call written as the matplotlib call it is: to check the chart, or reuse it, without pymetrix."""
        from .figure_code import matplotlib_code

        return matplotlib_code(self)

    @api()
    def layout(self, **props: Any) -> Figure:
        """The whole figure: ``figsize`` (inches), or ``figwidth`` or ``figheight`` alone; ``dpi``, ``facecolor``,
        ``suptitle``, ``supxlabel``, ``supylabel``."""
        return self._styled("layout", (), props)

    def _panels(self) -> tuple[Panel, ...]:
        panels = tuple(replace(p, layers=p.drawn()) if any(isinstance(layer, Fan) for layer in p.layers) else p for p in self.panels)
        return panels if self.span is None else tuple(_clip_panel(p, self.span) for p in panels)

    @property
    def rows(self) -> int:
        return int(np.ceil(len(self.panels) / max(1, self.columns)))

    def drawn_height(self) -> float:
        """The height drawn, in inches: as given, or by the rows of panels (taller where a panel is 3-D, to show its
        depth)."""
        if self.height:
            return self.height
        solid = any(p.projection for p in self.panels)
        return max(2.6, (2.9 if self.rows > 1 else 3.6) * self.rows * (1.4 if solid else 1.0))

    def draw(self, font: str | None = None, surface: Any = None) -> Any:
        """A matplotlib figure (without pyplot, so it works in the app, notebooks and scripts alike), in the house look
        (``pymetrix.figure_style``). ``font``: a font family to draw the text in (the app passes its own); ``surface``:
        what it is drawn on, ``figure_style.LIGHT`` (the default, as every saved file is) or a window's colours
        (``figure_style.surface_from``). Neither changes the figure's size in inches."""
        return self._draw(font=font, surface=surface)[0]

    def _housed(self) -> Figure:
        """The figure with the house style's calls that apply to it before its own, as it is drawn."""
        if not self.house or not _HOUSE:
            return self
        return replace(self, style=(*(c for c in _HOUSE if _style_fits(self, c, True)), *self.style), house=False)

    def _draw(self, font: str | None = None, surface: Any = None) -> tuple[Any, list[dict[str, Any]]]:
        """The matplotlib figure, and for each panel what was drawn (its axes, twin, artists by label, frequency)."""
        if self.house and _HOUSE:
            return self._housed()._draw(font=font, surface=surface)
        try:
            from matplotlib.figure import Figure as MplFigure
        except ImportError as exc:
            raise ImportError("drawing figures needs matplotlib, which pymetrix requires: pip install matplotlib") from exc
        from matplotlib import rc_context

        from .figure_style import LIGHT, chart_rc, on_surface, scaled

        surface = surface or LIGHT
        theme = scaled(get_theme(self.theme), surface.text_scale)  # the app's text size on screen; files keep the theme's
        cols = max(1, min(self.columns, len(self.panels)))
        rows = self.rows
        height = self.drawn_height()
        with rc_context(chart_rc(theme, font, surface)):  # the house look, for this figure only: rcParams are restored after
            fig = MplFigure(figsize=(self.width, height), layout="constrained")
            axes = fig.subplots(rows, cols, squeeze=False, sharey=self.sharey)
            drawn = []
            for i, panel in enumerate(self._panels()):
                ax = axes[i // cols][i % cols]
                if panel.projection:  # a 3-D panel: its own kind of axes, in the same place
                    ax.remove()
                    ax = fig.add_subplot(rows, cols, i + 1, projection=panel.projection)
                    drawn.append(_draw_panel3d(ax, panel, theme, self, i))
                elif self.ybreak is not None or self.xbreak is not None:
                    drawn.append(_draw_split(fig, ax, panel, theme, self, i))
                else:
                    drawn.append(_draw_panel(ax, panel, theme, self, i))
            _check_series_found(self, drawn)
            for j in range(len(self.panels), rows * cols):
                axes[j // cols][j % cols].set_visible(False)
            if self.title or self.subtitle:
                fig.suptitle("\n".join(x for x in (self.title, self.subtitle) if x), fontsize=theme.title_size + 1.5)
            if self.footnote:
                fig.supxlabel(self.footnote, fontsize=theme.font_size - 0.5, x=0.01, ha="left", style="italic", color="#555555")
            _apply_styles(fig, self, drawn, theme)
            for entry in drawn:
                if entry.get("split"):
                    _finish_split(entry)
            on_surface(fig, surface)
        return fig, drawn

    def png(self, dpi: int = 110, size: Any = None) -> bytes:
        """The figure as PNG bytes; ``size`` as ``save`` takes it (a preset, or width and height)."""
        buffer = io.BytesIO()
        fig = self.draw()
        if size is not None:
            fig.set_size_inches(*_size_inches(size, tuple(fig.get_size_inches())))
        fig.savefig(buffer, format="png", dpi=dpi)
        return buffer.getvalue()

    @api()
    def save(self, path: str, dpi: int = 150, size: Any = None, transparent: bool = False, editable_text: bool = False) -> Path:
        """Write the figure; the format follows the extension (.png, .pdf, .svg, .eps, .jpg...). Returns the path.

        ``size``: a preset (``SIZES``: ``"column"``, ``"page"``, ``"slide"``, ``"half-a4"``), or (width, height) as
        inches or as text with units (``("120mm", "80mm")``, ``("4in", None)``); a height of None keeps the figure's
        shape. Text stays at its size in points, so a narrower figure has relatively larger type, as print needs.
        ``transparent``: no background. PDF and EPS embed their fonts as TrueType (what journals ask for); SVG draws
        text as outlines, or keeps it as text to edit with ``editable_text``. The file records what made it:
        pymetrix's and matplotlib's versions, the title, the series and the recorded style calls (in PNG text, PDF
        Subject or SVG Description)."""
        from matplotlib import rc_context

        target = Path(path)
        fig = self.draw()
        if size is not None:
            fig.set_size_inches(*_size_inches(size, tuple(fig.get_size_inches())))
        fmt = target.suffix.lower().lstrip(".") or "png"
        rc = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none" if editable_text else "path",
              "svg.hashsalt": "pymetrix"}  # the same figure writes the same file
        with rc_context(rc):
            fig.savefig(target, dpi=dpi, transparent=transparent, metadata=self._metadata(fmt))
        return target

    def describe(self, technical: bool = True) -> str:
        """What the figure shows and how it was made, in words: its title, the series in each panel, the options
        and the recorded style calls, and the versions that drew it (the text a saved file records).
        ``technical=False`` keeps what a reader needs (the title, subtitle, panels and series, the footnote): the
        alt text reports give figures."""
        import matplotlib

        from . import __version__

        if not technical:
            words = [x for x in (self.title, self.subtitle) if x] or ["A figure"]
            for i, panel in enumerate(self._panels()):
                names = list(dict.fromkeys(str(getattr(layer, "label", "")) for layer in panel.layers if getattr(layer, "label", "")))
                where = panel.title or (f"panel {i + 1}" if len(self.panels) > 1 else "")
                if names:
                    words.append((f"{where}: " if where else "") + ", ".join(names))
            if self.footnote:
                words.append(self.footnote)
            return ". ".join(w.rstrip(".") for w in words) + "."
        lines = [self.title or "A pymetrix figure", f"Drawn by pymetrix {__version__} with matplotlib {matplotlib.__version__}."]
        for i, panel in enumerate(self._panels()):
            names = [str(getattr(layer, "label", "")) for layer in panel.layers if getattr(layer, "label", "")]
            lines.append(f"Panel {i}" + (f" ({panel.title})" if panel.title else "") + ": " + (", ".join(dict.fromkeys(names)) or "no named series"))
        options = {k: getattr(self, k) for k in ("span", "shade", "events", "reference", "log", "ylim", "sharey", "bar_labels", "line_labels",
                                                 "ybreak", "xbreak")
                   if getattr(self, k) not in (None, (), False)}
        if options:
            lines.append("Options: " + ", ".join(f"{k}={v!r}" for k, v in options.items()))
        housed = self._housed()
        if len(housed.style) > len(self.style):
            lines.append("House style: " + "; ".join(c.code() for c in housed.style[: len(housed.style) - len(self.style)]))
        if self.style:
            lines.append("Style: " + "; ".join(self.style_steps()))
        return "\n".join(lines)

    def _metadata(self, fmt: str) -> dict[str, Any] | None:
        """What made the file, in the keys each format keeps; dates left out, so the same figure writes the same file."""
        import matplotlib

        from . import __version__

        software = f"pymetrix {__version__}, matplotlib {matplotlib.__version__}"
        title = self.title or "pymetrix figure"
        if fmt == "png":
            return {"Title": title, "Software": software, "Description": self.describe()}
        if fmt == "pdf":
            return {"Title": title, "Creator": software, "Subject": self.describe(), "CreationDate": None, "ModDate": None}
        if fmt == "svg":
            return {"Title": title, "Creator": software, "Description": self.describe(), "Date": None}
        if fmt in ("eps", "ps"):
            return {"Creator": software}
        return None

    def data(self) -> pd.DataFrame:
        """Every series drawn, side by side (to copy into a spreadsheet). Two series with the same name (two unnamed
        bands, say) both appear, the second as "name (2)"."""
        cols: dict[str, pd.Series] = _Columns()
        for p, panel in enumerate(self._panels()):
            prefix = f"{panel.title}: " if panel.title and len(self.panels) > 1 else ""
            for layer in panel.layers:
                if isinstance(layer, (Line, Bars)):
                    cols[prefix + (layer.label or str(layer.y.name or f"series {len(cols) + 1}"))] = layer.y
                elif isinstance(layer, Band):
                    name = prefix + (layer.label or "band")
                    cols[f"{name} lower"], cols[f"{name} upper"] = layer.lower, layer.upper
                elif isinstance(layer, Scatter):
                    cols[prefix + str(layer.x.name or "x")], cols[prefix + str(layer.y.name or "y")] = layer.x, layer.y
                elif isinstance(layer, Histogram):
                    cols[prefix + (layer.label or str(layer.values.name or "values"))] = layer.values
                elif isinstance(layer, Estimates):
                    name = prefix + (layer.label or "estimate")
                    cols[name], cols[f"{name} lower"], cols[f"{name} upper"] = layer.estimate, layer.lower, layer.upper
                elif isinstance(layer, (Ecdf, QQ)):
                    cols[prefix + (layer.label or str(layer.values.name or "values"))] = layer.values
                elif isinstance(layer, (Hist2d, Hexbin)):
                    cols[prefix + str(layer.x.name or "x")], cols[prefix + str(layer.y.name or "y")] = layer.x, layer.y
                elif isinstance(layer, (Line3d, Scatter3d)):
                    name = prefix + (f"{layer.label}: " if layer.label else "")
                    for axis in ("x", "y", "z"):
                        s = getattr(layer, axis)
                        cols[name + str(s.name or axis)] = s
                elif isinstance(layer, (Heatmap, Contour, Surface)):
                    for c in layer.values.columns:
                        cols[prefix + str(c)] = layer.values[c]
                elif isinstance(layer, (Box, Violin)):
                    for i, s in enumerate(layer.series):
                        cols[prefix + (layer.labels[i] if i < len(layer.labels) else str(s.name or f"series {i + 1}"))] = s
        return pd.concat(cols, axis=1) if cols else pd.DataFrame()

    def _repr_png_(self) -> bytes:
        return self.png()

    def __repr__(self) -> str:
        titles = [p.title for p in self.panels if p.title]
        name = f"Figure {self.title}:" if self.title else "Figure:"
        detail = f" ({', '.join(titles[:4])})" if titles else ""
        return f"<{name} {len(self.panels)} panel{'s' if len(self.panels) != 1 else ''}{detail}>"


# ---------------------------------------------------------------------------------------------
# drawing


def _x(series: pd.Series, mid: bool = False) -> Any:
    index = series.index
    if isinstance(index, pd.PeriodIndex):
        return _period_x(index, mid)
    return np.asarray(index)


def _period_x(index: pd.PeriodIndex, mid: bool = False) -> Any:
    """Where periods are drawn: at their starts, or, in a panel that mixes frequencies, at their middles, so a
    quarter's value sits over its three months."""
    start = index.to_timestamp(how="start")
    return start + (index.to_timestamp(how="end") - start) / 2 if mid else start


def _panel_dates(panel: Panel, horizontal: bool = False) -> tuple[Any, bool]:
    """A panel's date frequency, the finest of its dated layers' (monthly beside quarterly: monthly), and whether it
    mixes frequencies. A scatter plot's or 2-D histogram's x is a series, not the dates."""
    if horizontal:
        return None, False
    freqs = [getattr(layer, attr).index.freq for layer in panel.layers for attr in ("y", "lower", "estimate")
             if not isinstance(layer, (Scatter, Hist2d, Hexbin)) and isinstance(getattr(layer, attr, None), pd.Series)
             and isinstance(getattr(layer, attr).index, pd.PeriodIndex)]
    if not freqs:
        return None, False

    def length(freq: Any) -> pd.Timedelta:
        period = pd.Period("2001-01-01", freq=freq)
        return period.end_time - period.start_time

    return min(freqs, key=length), len({length(f) for f in freqs}) > 1


def legend_place(where: str, entries: int) -> dict[str, Any]:
    """``Axes.legend``'s placement for a panel's ``legend_loc``: nothing (the best place inside) for "", the legend in
    one row above the panel, at its right end, for "above", or matplotlib's own ``loc``."""
    if not where:
        return {}
    if where == "above":
        return {"loc": "lower right", "bbox_to_anchor": (1.0, 1.0), "ncols": max(1, min(entries, 6)), "borderaxespad": 0.2,
                "columnspacing": 1.0, "handlelength": 1.5}
    return {"loc": where}


def _period_axis(ax: Any, freq: Any) -> None:
    """Label a date axis in the data's own periods: 2023Q1, 2023-01, 2023."""
    from matplotlib import dates, ticker

    def label(value: float, _pos: int) -> str:
        return str(pd.Period(dates.num2date(value).replace(tzinfo=None), freq=freq))

    ax.xaxis.set_major_locator(_PeriodLocator(freq))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(label))


@lru_cache(maxsize=1)
def _period_locator_class() -> Any:
    from matplotlib import dates, ticker

    class PeriodLocator(ticker.Locator):
        """Ticks on period boundaries at a regular step (every quarter, half-year, year, ...), as many as fit the
        axis's width at each draw (a label such as 2023Q1 needs about 0.75 inch)."""

        def __init__(self, freq: Any) -> None:
            self.freq = freq

        def __call__(self) -> list[float]:
            lo, hi = self.axis.get_view_interval()
            axes = self.axis.axes
            width = axes.get_window_extent().width / axes.figure.dpi if axes.figure.dpi else 3.0
            room = max(2, int(width / 0.75))
            end = getattr(axes, "pymetrix_data_end", None)  # room made for labels at the lines' ends gets no dates
            ticks = _period_ticks((lo, hi if end is None else min(hi, end)), self.freq, room)
            if ticks is None:
                fallback = dates.AutoDateLocator(minticks=2, maxticks=room)
                fallback.set_axis(self.axis)
                return [t for t in fallback() if end is None or t <= end]
            return ticks

    return PeriodLocator


def _PeriodLocator(freq: Any) -> Any:  # noqa: N802 - built lazily: matplotlib is needed only to draw
    return _period_locator_class()(freq)


_STEPS = {"Q": (1, 2, 4, 8, 12, 20, 40, 80), "M": (1, 2, 3, 6, 12, 24, 36, 60, 120), "Y": (1, 2, 5, 10, 20, 50, 100)}


def _period_ticks(limits: tuple[float, float], freq: Any, room: int) -> list[float] | None:
    """Tick positions at every k-th period (k from a list of round steps), aligned so that years start on a tick."""
    from matplotlib import dates

    code = str(getattr(freq, "freqstr", freq)).split("-")[0].rstrip("ES")[:1]
    if code == "A":
        code = "Y"
    if code not in _STEPS:
        return None
    lo, hi = (pd.Period(dates.num2date(v).replace(tzinfo=None), freq=freq) for v in limits)
    first, last = lo + 1 if dates.date2num(lo.to_timestamp()) < limits[0] else lo, hi
    if last < first:
        return None
    count = (last - first).n + 1
    per_year = {"Q": 4, "M": 12, "Y": 1}[code]
    step = next((k for k in _STEPS[code] if count / k <= room), _STEPS[code][-1])
    periods = pd.period_range(first, last, freq=freq)
    keep = [p for p in periods if (p.ordinal % step == 0 if step < per_year or per_year == 1 else (p.ordinal % per_year == 0 and (p.ordinal // per_year) % (step // per_year) == 0))]
    return [float(dates.date2num(p.to_timestamp())) for p in keep] or None


def _clip(series: pd.Series, span: tuple[Any, Any]) -> pd.Series:
    if not isinstance(series.index, pd.PeriodIndex):
        return series
    from .data.sample import as_span
    span = as_span(span)  # '1990Q1:2019Q4' or a pair
    start, end = span
    freq = series.index.freq
    keep = np.ones(len(series), dtype=bool)
    if start is not None:  # "2023Q1" for a monthly series starts with 2023-01, and as an end, stops after 2023-03
        keep &= series.index >= _period(start).asfreq(freq, how="start")
    if end is not None:
        keep &= series.index <= _period(end).asfreq(freq, how="end")
    return series[keep]


def _period(value: Any) -> pd.Period:
    """A span's end as a period of its own frequency: "2023Q4" a quarter, "2023-05" a month, "2023" a year."""
    return value if isinstance(value, pd.Period) else pd.Period(str(value))


def _clip_panel(panel: Panel, span: tuple[Any, Any]) -> Panel:
    layers = []
    for layer in panel.layers:
        if isinstance(layer, (Line, Bars)):
            layer = replace(layer, y=_clip(layer.y, span))
        elif isinstance(layer, Band):
            layer = replace(layer, lower=_clip(layer.lower, span), upper=_clip(layer.upper, span))
        elif isinstance(layer, (Box, Violin)):
            layer = replace(layer, series=tuple(_clip(s, span) for s in layer.series))
        elif isinstance(layer, Estimates) and isinstance(layer.estimate.index, pd.PeriodIndex):
            layer = replace(layer, estimate=_clip(layer.estimate, span), lower=_clip(layer.lower, span), upper=_clip(layer.upper, span))
        elif isinstance(layer, (Histogram, Ecdf, QQ)):
            layer = replace(layer, values=_clip(layer.values, span))
        elif isinstance(layer, (Scatter, Hist2d, Hexbin)):
            layer = replace(layer, x=_clip(layer.x, span), y=_clip(layer.y, span))
        elif isinstance(layer, (Line3d, Scatter3d)):
            layer = replace(layer, x=_clip(layer.x, span), y=_clip(layer.y, span), z=_clip(layer.z, span))
        elif isinstance(layer, (Contour, Surface)) and isinstance(layer.values.index, pd.PeriodIndex):  # dated rows
            layer = replace(layer, values=layer.values.loc[_clip(pd.Series(0.0, index=layer.values.index), span).index])
        layers.append(layer)
    return replace(panel, layers=tuple(layers))


def _color(color: str | None, theme: Theme, fallback: Any) -> str:
    """A result's colour in the theme: the n-th colour of the standard palette becomes the theme's n-th."""
    if color is None:
        return next(fallback)
    if color in PALETTE:
        return theme.palette[PALETTE.index(color) % len(theme.palette)]
    return color


_STYLE_TARGETS = ("line", "bars", "band", "set", "ticks", "grid", "spines", "legend", "layout", "annotate", "text",
                  "axhline", "axvline", "axhspan", "axvspan", "call", "view_init", "change")
_HOUSE: tuple[StyleCall, ...] = ()  # the house style: style calls every figure is drawn with first, where they apply


def set_house_style(style: Any = None, notes: bool = False) -> tuple[StyleCall, ...]:
    """Make a figure's recorded style the house style: every figure drawn from now on starts from it, call by call
    where the call applies (as ``with_style_of``: its series, panels and right-hand axes), and its own calls come
    after, so they win. ``style`` is a figure or its style calls; None clears the house style. Notes, text, lines and
    bands are left out unless ``notes``. A figure drawn with ``house=False`` ignores it. Returns the previous
    house style's calls, as ``set_theme`` returns the previous theme."""
    global _HOUSE
    previous = _HOUSE
    if style is None:
        _HOUSE = ()
    else:
        calls = style.style if isinstance(style, Figure) else tuple(_style_call(c) for c in style)
        _HOUSE = tuple(c for c in calls if notes or c.kind not in _ADDED)
    return previous


def house_style() -> tuple[StyleCall, ...]:
    """The house style's calls (empty: none)."""
    return _HOUSE


#: Figure sizes for publication, in millimetres (width, height); a height of None keeps the figure's shape.
SIZES: dict[str, tuple[float, float | None]] = {
    "column": (89.0, None),       # one column of a two-column journal page
    "page": (183.0, None),        # the full width of a journal page
    "slide": (254.0, 142.875),    # a 16:9 slide, 10 by 5.625 inches
    "half-a4": (170.0, 120.0),    # half an A4 page, inside its margins
}


def _length(value: Any) -> float | None:
    """A length in inches: a number (inches), or text with units (89mm, 8.9cm, 3.5in)."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    for unit, per_inch in (("mm", 25.4), ("cm", 2.54), ("in", 1.0)):
        if text.endswith(unit):
            try:
                return float(text[: -len(unit)]) / per_inch
            except ValueError:
                break
    try:
        return float(text)
    except ValueError:
        raise SpecificationError(f"{value!r} is not a length: give inches, or a number with mm, cm or in") from None


def _size_inches(size: Any, natural: tuple[float, float]) -> tuple[float, float]:
    """(width, height) in inches for ``Figure.save``'s ``size``; a missing height keeps the natural shape."""
    if isinstance(size, str):
        if size not in SIZES:
            raise SpecificationError(f"no figure size {size!r}; sizes: {', '.join(SIZES)}, or (width, height)")
        width_mm, height_mm = SIZES[size]
        width, height = width_mm / 25.4, None if height_mm is None else height_mm / 25.4
    else:
        try:
            first, second = size
        except (TypeError, ValueError):
            raise SpecificationError("size is a preset name or (width, height)") from None
        width, height = _length(first), _length(second)
        if width is None:
            raise SpecificationError("give the figure's width")
    if width <= 0 or (height is not None and height <= 0):
        raise SpecificationError("a figure's width and height are positive")
    return width, height if height is not None else width * natural[1] / natural[0]


_ADDED = ("annotate", "text", "axhline", "axvline", "axhspan", "axvspan", "change")  # added to a panel, in data coordinates
_ADDED_NEEDS = {"annotate": ("text", "xy"), "text": ("x", "y", "s"), "axhline": ("y",), "axvline": ("x",),
                "axhspan": ("ymin", "ymax"), "axvspan": ("xmin", "xmax"), "change": ("series", "start", "end")}
CHANGE_SHOW = ("change", "percent", "both")  # Figure.change: the change, the percent change, or both
CHANGE_UNITS = ("auto", "pp", "%", "none")  # the change's units: percentage points for a rate (auto), %, or none
CHANGE_STYLES = {"arrow": "<->", "bracket": "|-|"}
_CHANGE_KEYS = {"series", "start", "end", "show", "units", "annualise", "style", "color", "fontsize", "linewidth"}
_PER_YEAR = {"Y": 1, "A": 1, "Q": 4, "M": 12, "W": 52, "D": 365, "B": 261, "H": 2}
_LAYOUT = {"figsize", "figwidth", "figheight", "dpi", "facecolor", "suptitle", "supxlabel", "supylabel"}
_SPINES = ("left", "right", "top", "bottom", "all")


def _plain(value: Any, where: str) -> Any:
    """Plain data only (numbers, text, true/false, None, lists, dicts), so a style replays and saves as JSON."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v, where) for v in value] if isinstance(value, list) else tuple(_plain(v, where) for v in value)
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _plain(v, where) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    raise SpecificationError(f"{where}: style values must be numbers, text, lists or dicts, not {type(value).__name__}")


def _dates_as_numbers(value: Any) -> Any:
    """xlim given as periods ("2010Q1") or dates, as matplotlib's date numbers; numbers are left as they are."""
    from matplotlib import dates

    if isinstance(value, (list, tuple)):
        return type(value)(_dates_as_numbers(v) for v in value)
    if isinstance(value, str):
        try:
            return float(dates.date2num(pd.Period(value).to_timestamp()))
        except (ValueError, TypeError):
            return float(dates.date2num(pd.Timestamp(value)))
    return value


def _check_style(target: str, where: tuple[Any, ...], props: dict[str, Any]) -> dict[str, Any]:
    """The call's keyword arguments, aliases resolved and tried on a scratch matplotlib object: a value matplotlib
    rejects is reported now, not when the figure is drawn."""
    from matplotlib import cbook
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure as MplFigure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    if target not in _STYLE_TARGETS:
        raise SpecificationError(f"no style call {target!r}; the calls are {', '.join(_STYLE_TARGETS)}")
    props = _plain(dict(props), target)
    solid = target == "view_init" or (target == "set" and any(k.startswith("z") for k in props)) or (target == "ticks" and where[1] == "z")
    scratch = MplFigure().add_subplot(projection="3d" if solid else None)  # a 3-D panel's calls on 3-D axes
    try:
        if target == "line":
            props = cbook.normalize_kwargs(props, Line2D)
            Line2D([0], [0]).set(**props)
        elif target == "bars":
            props = cbook.normalize_kwargs(props, Rectangle)
            Rectangle((0, 0), 1, 1).set(**props)
        elif target == "band":
            props = cbook.normalize_kwargs(props, PolyCollection)
            scratch.fill_between([0, 1], [0, 0], [1, 1]).set(**props)
        elif target == "set":
            if where[1] not in ("left", "right"):
                raise SpecificationError("side must be 'left' or 'right'")
            scratch.set(**{k: _dates_as_numbers(v) if k in ("xlim", "xbound") else v for k, v in props.items()})
        elif target == "ticks":
            if where[1] not in ("x", "y", "z", "both"):
                raise SpecificationError("axis must be 'x', 'y', 'both' or, on a 3-D panel, 'z'")
            rest = {k: v for k, v in props.items() if k != "format"}
            scratch.tick_params(axis=where[1], **rest)
            if "format" in props:
                str(props["format"]).format(x=1234.5678, pos=0)
        elif target == "grid":
            scratch.grid(**props)
        elif target == "spines":
            if where[1] not in _SPINES:
                raise SpecificationError(f"which must be one of {', '.join(_SPINES)}")
            scratch.spines["left"].set(**props)
        elif target == "change":
            _check_change(props)
        elif target in _ADDED:
            _add_to(scratch, target, props, None)
        elif target == "call":
            _call_axes(scratch, props)
        elif target == "view_init":
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in props.values()):
                raise SpecificationError("view_init: elev, azim and roll are angles in degrees, e.g. elev=30, azim=-60")
            scratch.view_init(**props)
        elif target == "legend":
            visible = props.get("visible", True)
            scratch.plot([0], [0], label="a")
            scratch.legend(**{k: v for k, v in props.items() if k != "visible"}).set_visible(bool(visible))
        else:  # layout
            unknown = sorted(set(props) - _LAYOUT)
            if unknown:
                raise SpecificationError(f"layout takes {', '.join(sorted(_LAYOUT))}, not {', '.join(unknown)}")
            if "figsize" in props and (len(props["figsize"]) != 2 or min(float(v) for v in props["figsize"]) <= 0):
                raise SpecificationError("figsize is (width, height) in inches")
            if any(float(props[k]) <= 0 for k in ("figwidth", "figheight") if k in props):
                raise SpecificationError("figwidth and figheight are positive, in inches")
            if "facecolor" in props:
                from matplotlib.colors import to_rgba

                to_rgba(props["facecolor"])
            if "dpi" in props and float(props["dpi"]) <= 0:
                raise SpecificationError("dpi must be positive")
    except SpecificationError:
        raise
    except (AttributeError, TypeError, ValueError, KeyError) as exc:  # matplotlib's own message says what is wrong
        raise SpecificationError(f"{target}: {exc}") from exc
    return props


def _x_number(value: Any, freq: Any, end: bool = False) -> Any:
    """An x position as matplotlib's number: a period ("2008Q3") at its start (or, with ``end``, the end of it) on a
    dated axis; numbers as they are."""
    from matplotlib import dates

    if isinstance(value, str):
        if freq is None:  # an undated axis: "5" is the number 5
            try:
                return float(value)
            except ValueError:
                pass
        try:  # "2013Q4" is that quarter, on a monthly axis too: through December with end=True
            period = pd.Period(value)
        except (ValueError, TypeError):
            try:
                period = pd.Period(value, freq=freq)
            except (ValueError, TypeError) as exc:
                raise SpecificationError(f"{value!r} is not a period of this figure ({exc})") from exc
        return float(dates.date2num((period + 1 if end else period).to_timestamp()))
    return value


def _add_to(ax: Any, target: str, props: dict[str, Any], freq: Any) -> Any:
    """Draw an annotate, text, axhline, axvline, axhspan or axvspan call on ``ax``, positions in the data's terms."""
    if getattr(ax, "name", "") == "3d":
        raise SpecificationError(f"{target} marks a flat panel; a 3-D panel takes view_init, set, ticks, grid, legend and call")
    p = dict(props)
    if target == "annotate":
        text, xy = p.pop("text"), p.pop("xy")
        xy = (_x_number(xy[0], freq), xy[1])
        textcoords = p.get("textcoords", p.get("xycoords", "data"))
        if "xytext" in p and textcoords == "data":
            p["xytext"] = (_x_number(p["xytext"][0], freq), p["xytext"][1])
        return ax.annotate(text, xy, **p)
    if target == "text":
        return ax.text(_x_number(p.pop("x"), freq), p.pop("y"), p.pop("s"), **p)
    if target == "axhline":
        return ax.axhline(p.pop("y"), **p)
    if target == "axvline":
        return ax.axvline(_x_number(p.pop("x"), freq), **p)
    if target == "axhspan":
        return ax.axhspan(p.pop("ymin"), p.pop("ymax"), **p)
    return ax.axvspan(_x_number(p.pop("xmin"), freq), _x_number(p.pop("xmax"), freq, end=True), **p)


def _check_change(props: dict[str, Any]) -> None:
    unknown = sorted(set(props) - _CHANGE_KEYS)
    if unknown:
        raise SpecificationError(f"change takes {', '.join(sorted(_CHANGE_KEYS))}, not {', '.join(unknown)}")
    if props.get("show", "both") not in CHANGE_SHOW:
        raise SpecificationError(f"show is one of {', '.join(CHANGE_SHOW)}")
    if props.get("units", "auto") not in CHANGE_UNITS:
        raise SpecificationError(f"units is one of {', '.join(CHANGE_UNITS)}")
    if props.get("style", "arrow") not in CHANGE_STYLES:
        raise SpecificationError(f"style is one of {', '.join(CHANGE_STYLES)}")
    if "color" in props:
        from matplotlib.colors import to_rgba

        to_rgba(props["color"])
    for key in ("fontsize", "linewidth"):
        if key in props and float(props[key]) <= 0:
            raise SpecificationError(f"{key} must be positive")
    if str(props.get("start", "")) == str(props.get("end", "")):
        raise SpecificationError("the change is between two different points: give a start and a later end")


def _signed(value: float) -> str:
    """A change with its sign and three significant figures or so: +1.23, -0.0456, +1,234."""
    size = abs(value)
    digits = 0 if size >= 100 else 1 if size >= 10 else 2 if size >= 0.1 else 4
    return f"{value:+,.{digits}f}"


def _rate_like(panel: Panel) -> bool:
    """A panel of rates or percentages (its axis label says so): a change in it is in percentage points."""
    words = f"{panel.ylabel} {panel.ylabel_right}".lower()
    return "%" in words or "percent" in words or "per cent" in words or "rate" in words


def change_between(panel: Panel, series: str, start: Any, end: Any, show: str = "both", units: str = "auto",
                   annualise: bool = False) -> dict[str, Any]:
    """The change in ``series`` (as ``panel`` draws it) from ``start`` to ``end``: the values, the change, the percent
    change, the annualised rates and the label ``Figure.change`` writes. Periods on a dated axis, numbers otherwise."""
    layer = next((l for l in panel.layers if getattr(l, "label", None) == series and isinstance(getattr(l, "y", None), pd.Series)), None)
    if layer is None:
        names = [l.label for l in panel.layers if getattr(l, "label", "") and isinstance(getattr(l, "y", None), pd.Series)]
        raise SpecificationError(f"no series {series!r} in this panel" + (f"; it draws {', '.join(names)}" if names else ""))
    y = layer.y.dropna()

    def at(when: Any) -> Any:
        if isinstance(y.index, pd.PeriodIndex):
            try:
                key = pd.Period(str(when), freq=y.index.freq)
            except (ValueError, TypeError) as exc:
                raise SpecificationError(f"{when!r} is not a period of {series} ({exc})") from exc
        else:
            key = float(when)
        if key not in y.index:
            raise SpecificationError(f"{series} has no value at {when}")
        return key

    a, b = at(start), at(end)
    if y.index.get_loc(a) > y.index.get_loc(b):
        a, b = b, a
    y0, y1 = float(y.loc[a]), float(y.loc[b])
    change = y1 - y0
    percent = (y1 / y0 - 1) * 100 if y0 != 0 else float("nan")
    unit = {"auto": " pp" if _rate_like(panel) else "", "pp": " pp", "%": "%", "none": ""}[units]
    parts = {"change": _signed(change) + unit, "percent": "n/a %" if percent != percent else f"{percent:+.1f}%"}
    label = {"change": parts["change"], "percent": parts["percent"], "both": f"{parts['change']} ({parts['percent']})"}[show]
    per_year = None
    if annualise:
        if not isinstance(y.index, pd.PeriodIndex):
            raise SpecificationError("an annualised rate needs dates: this axis has none")
        steps = y.index.get_loc(b) - y.index.get_loc(a)
        code = y.index.freqstr.split("-")[0].rstrip("SE").upper() or "A"
        per_year = _PER_YEAR.get(code[-1] if code[-1] in _PER_YEAR else code[:1])
        if per_year is None:
            raise SpecificationError(f"no annual rate for a frequency of {y.index.freqstr}")
        years = steps / per_year
        rates = []
        if show in ("percent", "both"):
            rates.append("n/a" if not (y0 > 0 and y1 > 0) else f"{((y1 / y0) ** (1 / years) - 1) * 100:+.1f}% a year")
        if show in ("change", "both"):
            rates.append(f"{_signed(change / years)}{unit} a year")
        label += "\n" + ", ".join(rates)
    return {"start": str(a), "end": str(b), "y0": y0, "y1": y1, "change": change, "percent": percent, "label": label,
            "dated": isinstance(y.index, pd.PeriodIndex), "per_year": per_year}


def _change_x(when: str, dated: bool, mid: bool) -> float:
    from matplotlib import dates

    if not dated:
        return float(when)
    return float(dates.date2num(_period_x(pd.PeriodIndex([pd.Period(when)]), mid)[0]))


def _draw_change(ax: Any, panel: Panel, props: dict[str, Any], mid: bool) -> list[Any]:
    """A ``Figure.change`` call: an arrow (or bracket) between the two points, its ends marked, labelled at its middle."""
    p = dict(props)
    c = change_between(panel, p["series"], p["start"], p["end"], p.get("show", "both"), p.get("units", "auto"),
                       bool(p.get("annualise", False)))
    x0, x1 = _change_x(c["start"], c["dated"], mid), _change_x(c["end"], c["dated"], mid)
    color = p.get("color", "#333333")
    arrow = ax.annotate("", xy=(x1, c["y1"]), xytext=(x0, c["y0"]),
                        arrowprops={"arrowstyle": CHANGE_STYLES[p.get("style", "arrow")], "color": color,
                                    "lw": float(p.get("linewidth", 1.2)), "shrinkA": 0, "shrinkB": 0})
    ends = ax.plot([x0, x1], [c["y0"], c["y1"]], linestyle="none", marker="o", markersize=4, color=color, label="_change")[0]
    label = ax.annotate(c["label"], xy=((x0 + x1) / 2, (c["y0"] + c["y1"]) / 2), xytext=(6, 6), textcoords="offset points",
                        fontsize=float(p.get("fontsize", 8.5)), color=color, ha="left", va="bottom",
                        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.85})
    return [arrow, ends, label]


def _call_axes(ax: Any, props: dict[str, Any]) -> Any:
    """Make a recorded ``call`` on ``ax``: an allowed Axes method, its arguments checked against its signature."""
    import inspect

    from .figure_properties import axes_methods

    method = props["method"]
    if method not in axes_methods():
        raise SpecificationError(f"{method!r} is not an Axes method fig.call records (it styles or annotates a panel); "
                                 "see pymetrix.figure_properties.axes_methods()")
    bound = getattr(ax, method)
    try:
        inspect.signature(bound).bind(*props.get("args", ()), **props.get("kwargs", {}))
    except TypeError as exc:
        raise SpecificationError(f"{method}{inspect.signature(bound)}: {exc}") from exc
    return bound(*props.get("args", ()), **props.get("kwargs", {}))


def _panel_indexes(figure: Figure, ax: Any) -> list[int]:
    if ax is None or ax == "all":
        return list(range(len(figure.panels)))
    if isinstance(ax, int) and not isinstance(ax, bool):
        if not 0 <= ax < len(figure.panels):
            raise SpecificationError(f"no panel {ax}: this figure has {len(figure.panels)} (0 to {len(figure.panels) - 1})")
        return [ax]
    titles = [p.title for p in figure.panels]
    if ax not in titles:
        raise SpecificationError(f"no panel titled {ax!r}; the panels are {', '.join(repr(x) for x in titles)}")
    return [titles.index(ax)]


def _style_fits(figure: Figure, call: StyleCall, notes: bool) -> bool:
    """Whether a style call recorded on another figure applies to this one (``Figure.with_style_of``)."""
    if call.kind in _ADDED and not notes:
        return False
    if call.kind == "layout":
        return True
    try:
        panels = _panel_indexes(figure, call.where[0])
    except SpecificationError:
        return False
    if call.kind in ("line", "bars", "band"):
        kind = {"line": (Line, Estimates, Ecdf, Line3d), "bars": Bars, "band": Band}[call.kind]  # estimates' markers style as a line
        return any(isinstance(layer, kind) and getattr(layer, "label", "") in (call.where[1], f"{call.where[1]} (right)")
                   for i in panels for layer in figure.panels[i].drawn())
    if call.kind in ("set", "call") and call.where[1] == "right":
        return all(any(getattr(layer, "axis", "left") == "right" for layer in figure.panels[i].layers) for i in panels)
    if call.kind == "view_init":  # turns 3-D panels only
        return any(figure.panels[i].projection == "3d" for i in panels)
    return True


def _style_series(figure: Figure, index: int, panel: Panel, artists: dict[str, list[Any]]) -> set[int]:
    """Apply the line, bars and band calls that reach this panel, before its legend is made (so the legend shows the
    styled lines); returns the positions of the calls that found their series here."""
    found = set()
    for k, call in enumerate(figure.style):
        if call.kind not in ("line", "bars", "band"):
            continue
        ax, label = call.where
        if index not in _panel_indexes(figure, ax):
            continue
        drawn = artists.get(label) or artists.get(f"{label} (right)") or []
        for kind, artist in drawn:
            if kind != call.kind:
                continue
            for part in (artist if kind == "bars" else [artist]):
                part.set(**call.props)
            found.add(k)
    return found


def _check_series_found(figure: Figure, drawn: list[dict[str, Any]]) -> None:
    found = set().union(*(d["found"] for d in drawn)) if drawn else set()
    for k, call in enumerate(figure.style):
        if call.kind in ("line", "bars", "band") and k not in found:
            names = sorted({label for d in drawn for label, items in d["artists"].items()
                            if label and any(kind == call.kind for kind, _ in items)})
            plural = {"line": "lines", "bars": "bar series", "band": "bands"}[call.kind]
            raise SpecificationError(f"no {call.kind} {call.where[1]!r} in this figure"
                                     + (f"; its {plural} are {', '.join(names)}" if names else ""))


def _apply_styles(fig: Any, figure: Figure, drawn: list[dict[str, Any]], theme: Theme) -> None:
    """The axes, ticks, grid, spines, legend and layout calls, in the order recorded, once every panel is drawn."""
    from matplotlib.ticker import StrMethodFormatter

    for call in figure.style:
        if call.kind in ("line", "bars", "band"):
            continue
        if call.kind == "layout":
            props = call.props
            if "figsize" in props:
                fig.set_size_inches(*props["figsize"])
            if "figwidth" in props:  # one side; the other as it is
                fig.set_figwidth(float(props["figwidth"]))
            if "figheight" in props:
                fig.set_figheight(float(props["figheight"]))
            if "dpi" in props:
                fig.set_dpi(float(props["dpi"]))
            if "facecolor" in props:
                fig.set_facecolor(props["facecolor"])
            for key in ("suptitle", "supxlabel", "supylabel"):
                if key in props:
                    getattr(fig, key)(props[key])
            continue
        if call.kind == "change" and call.where[0] is None and not any(
                getattr(l, "label", None) == call.props["series"] for p in figure._panels() for l in p.layers):
            raise SpecificationError(f"no series {call.props['series']!r} in this figure")
        for i in _panel_indexes(figure, call.where[0]):
            for d in (drawn[i], *(() if call.kind == "legend" else drawn[i].get("parts", ()))):  # a split panel: both parts
                ax = d["ax"]
                if call.kind == "set":
                    if call.where[1] == "right":
                        if d["right"] is None:
                            raise SpecificationError(f"panel {i} has no right-hand axis")
                        ax = d["right"]
                    ax.set(**{k: _dates_as_numbers(v) if k in ("xlim", "xbound") else v for k, v in call.props.items()})
                elif call.kind == "ticks":
                    if call.where[1] == "z" and getattr(ax, "name", "") != "3d":
                        raise SpecificationError(f"panel {i} is flat: it has no z axis")
                    rest = {k: v for k, v in call.props.items() if k != "format"}
                    if rest:
                        ax.tick_params(axis=call.where[1], **rest)
                    if "format" in call.props:
                        for name in ("x", "y") if call.where[1] == "both" else (call.where[1],):
                            getattr(ax, f"{name}axis").set_major_formatter(StrMethodFormatter(call.props["format"]))
                elif call.kind == "change":
                    panel = figure._panels()[i]
                    if call.where[0] is None and not any(getattr(l, "label", None) == call.props["series"] for l in panel.layers):
                        continue  # wherever the series is drawn: the other panels are left as they are
                    for artist in _draw_change(ax, panel, call.props, drawn[i]["mid"]):
                        artist.pymetrix_style = call  # the recorded step it draws, for the app's direct editing
                elif call.kind in _ADDED:
                    artist = _add_to(ax, call.kind, call.props, drawn[i]["freq"])
                    artist.pymetrix_style = call  # the recorded step it draws, for the app's direct editing
                elif call.kind == "call":
                    if call.where[1] == "right":
                        if d["right"] is None:
                            raise SpecificationError(f"panel {i} has no right-hand axis")
                        ax = d["right"]
                    _call_axes(ax, call.props)
                elif call.kind == "view_init":
                    if getattr(ax, "name", "") != "3d":
                        if call.where[0] in (None, "all"):
                            continue  # every 3-D panel: the flat ones are left as they are
                        raise SpecificationError(f"panel {i} is flat: view_init turns 3-D panels")
                    ax.view_init(**call.props)
                elif call.kind == "grid":
                    ax.grid(**call.props)
                elif call.kind == "spines":
                    for name in (("left", "right", "top", "bottom") if call.where[1] == "all" else (call.where[1],)):
                        ax.spines[name].set(**call.props)
                elif call.kind == "legend":
                    props = dict(call.props)
                    visible = props.pop("visible", True)
                    old = ax.get_legend()
                    if old is not None:
                        handles = list(getattr(old, "legend_handles", None) or getattr(old, "legendHandles", []))
                        names = [t.get_text() for t in old.get_texts()]
                    else:
                        handles, names = ax.get_legend_handles_labels()
                    if not visible:
                        if old is not None:
                            old.remove()
                        continue
                    if handles:
                        ax.legend(handles, names, **{"fontsize": theme.font_size - 0.5, "frameon": False, **props})


def _bar_positions(index: pd.Index, horizontal: bool, mid: bool = False) -> tuple[Any, float, list[str] | None]:
    """Where a series' bars go: the positions, the space between neighbours (in the axis' units), and tick labels when
    the positions stand for categories. Periods are dates on a vertical-bar chart and categories on a horizontal one."""
    if isinstance(index, pd.PeriodIndex) and not horizontal:
        x = _period_x(index, mid)
        gap = float(np.median(np.diff(x.to_numpy()).astype("timedelta64[D]").astype(float))) if len(x) > 1 else 30.0
        return x, gap, None
    if pd.api.types.is_numeric_dtype(index) and not horizontal:
        x = np.asarray(index, float)
        return x, float(np.median(np.diff(x))) if len(x) > 1 else 1.0, None
    return np.arange(len(index), dtype=float), 1.0, [str(i) for i in index]


def _shift(x: Any, offset: float) -> Any:
    """Bar positions moved by ``offset`` (days on a date axis)."""
    return x + pd.to_timedelta(offset, unit="D") if isinstance(x, pd.DatetimeIndex) else x + offset


def _category_ticks(ax: Any, labels: list[str], horizontal: bool) -> None:
    step = max(1, int(np.ceil(len(labels) / 30)))  # at most about 30 labels
    positions = np.arange(0, len(labels), step)
    shown = [labels[i] for i in positions]
    if horizontal:
        ax.set_yticks(positions, shown)
        ax.invert_yaxis()  # the first category at the top, as a table reads
    else:
        ax.set_xticks(positions, shown, rotation=45 if max(map(len, shown), default=0) * len(shown) > 60 else 0,
                      ha="right" if max(map(len, shown), default=0) * len(shown) > 60 else "center")


def shade_spans(spec: Any) -> tuple[tuple[str, str, str], ...]:
    """Periods to shade, from text (``"2008Q3:2009Q2, 2020Q1:2020Q2"``; a label after ``=``, as in
    ``"2008Q3:2009Q2=GFC"``; a single period shades just that period), from (start, end[, label]) tuples, or from a
    dated 0/1 series (each run of non-zero values, labelled with the series' name). ``"nber"`` or ``"cepr"`` shade the
    business-cycle committees' recessions (``pymetrix.recessions``), dated at the figure's own frequency."""
    from .recessions import SOURCES

    if isinstance(spec, pd.Series):
        return _runs(spec)
    if not spec:
        return ()
    if isinstance(spec, str):
        spec = [part.strip() for part in spec.split(",") if part.strip()]
    spans = []
    for item in spec:
        if isinstance(item, str) and item.strip().lower() in SOURCES:
            spans.append((f"@{item.strip().lower()}", "", SOURCES[item.strip().lower()][0]))
            continue
        if isinstance(item, (tuple, list)) and str(item[0]).startswith("@"):
            spans.append((str(item[0]), "", str(item[2]) if len(item) > 2 else ""))
            continue
        if isinstance(item, str):
            text, _, label = item.partition("=")
            start, _, end = text.partition(":")
            item = (start.strip(), (end or start).strip(), label.strip())
        start, end, *rest = (str(x).strip() for x in item)
        if not start or not end:
            raise SpecificationError(f"a shaded span needs a start and an end, e.g. 2008Q3:2009Q2; got {item!r}")
        spans.append((start, end, rest[0] if rest else ""))
    return tuple(spans)


def _labelled(spec: Any, what: str) -> list[tuple[str, str]]:
    """``"a=label, b"`` or [(a, label), a, ...] as (a, label) pairs of text."""
    if not spec:
        return []
    if isinstance(spec, (str, int, float)):
        spec = [part.strip() for part in str(spec).split(",") if part.strip()]
    pairs = []
    for item in spec:
        if isinstance(item, (str, int, float)):
            key, _, label = str(item).partition("=")
        else:
            key, *rest = item
            label = rest[0] if rest else ""
        key = str(key).strip()
        if not key:
            raise SpecificationError(f"give each {what} as value=label or value; got {item!r}")
        pairs.append((key, str(label).strip()))
    return pairs


def event_marks(spec: Any) -> tuple[tuple[str, str], ...]:
    """Dated events, from text (``"2016Q2=Brexit vote, 2020Q1"``) or (period, label) pairs."""
    return tuple(_labelled(spec, "event"))


def reference_lines(spec: Any) -> tuple[tuple[float, str], ...]:
    """Reference values, from text (``"2=inflation target, 60"``), numbers, or (value, label) pairs."""
    out = []
    for key, label in _labelled(spec, "reference line"):
        try:
            out.append((float(key), label))
        except ValueError as exc:
            raise SpecificationError(f"a reference line is a number, optionally =label; got {key!r}") from exc
    return tuple(out)


def value_limits(spec: Any) -> tuple[float | None, float | None] | None:
    """(lowest, highest) values shown, either None (automatic); text such as ``"0:10"`` or ``":10"`` too."""
    if spec is None or spec == "" or spec == (None, None):
        return None
    if isinstance(spec, str):
        spec = tuple(spec.split(":")) if ":" in spec else tuple(spec.split(","))
    if len(spec) != 2:
        raise SpecificationError(f"give the value limits as (lowest, highest); got {spec!r}")
    try:
        low, high = (None if v is None or str(v).strip() == "" else float(v) for v in spec)
    except ValueError as exc:
        raise SpecificationError(f"the value limits must be numbers; got {spec!r}") from exc
    if low is not None and high is not None and low >= high:
        raise SpecificationError(f"the lowest value shown ({low:g}) must be below the highest ({high:g})")
    return (low, high)


def _two(spec: Any, what: str) -> tuple[Any, Any] | None:
    if spec is None or spec == "" or spec == (None, None):
        return None
    if isinstance(spec, str):
        spec = tuple(spec.split(":")) if ":" in spec else tuple(spec.split(","))
    if len(spec) != 2 or any(v is None or str(v).strip() == "" for v in spec):
        raise SpecificationError(f"give the {what} as two values, e.g. {'5:15' if what.startswith('values') else '1986Q1:2004Q4'}; got {spec!r}")
    return tuple(spec)


def value_break(spec: Any) -> tuple[float, float] | None:
    """The values a split value axis leaves out, (low, high); text such as ``"5:15"`` too."""
    pair = _two(spec, "values to leave out")
    if pair is None:
        return None
    try:
        low, high = (float(v) for v in pair)
    except ValueError as exc:
        raise SpecificationError(f"the values to leave out must be numbers; got {spec!r}") from exc
    if low >= high:
        raise SpecificationError(f"the values left out run from the lower ({low:g}) to the higher ({high:g})")
    return (low, high)


def period_break(spec: Any) -> tuple[Any, Any] | None:
    """The periods a split time axis leaves out, (first, last), both included; text such as ``"1986Q1:2004Q4"`` too.
    Text is a period on a dated axis ("1990" the year) and a number on an undated one; numbers are numbers."""
    pair = _two(spec, "periods to leave out")
    if pair is None:
        return None
    out: list[Any] = []
    for v in pair:
        if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
            out.append(float(v))
            continue
        text = str(v).strip()
        try:
            pd.Period(text)
        except (ValueError, TypeError):
            try:
                float(text)
            except ValueError as exc:
                raise SpecificationError(f"{text!r} is not a period (e.g. 1986Q1, 2004-12 or 1990) or a number") from exc
        out.append(text)
    try:
        backwards = (pd.Period(out[0]).start_time > pd.Period(out[1]).end_time if all(isinstance(v, str) for v in out)
                     else float(out[0]) > float(out[1]))
    except (ValueError, TypeError):
        raise SpecificationError(f"give both ends as periods, or both as numbers; got {spec!r}") from None
    if backwards:
        raise SpecificationError(f"the periods left out run from the first ({out[0]}) to the last ({out[1]})")
    return tuple(out)


BREAK_MARK = {"marker": [(-1, -0.5), (1, 0.5)], "markersize": 10, "linestyle": "none", "color": "#444444",
              "mec": "#444444", "mew": 1, "clip_on": False}
BREAK_MARK_X = {**BREAK_MARK, "marker": [(-0.5, -1), (0.5, 1)]}


def _break_fits(entry: dict[str, Any], figure: Figure) -> tuple[float, float, float, float] | None:
    """(start, low cut, high cut, end) of a split along the drawn axis, if the break lies inside what is drawn."""
    ax = entry["ax"]
    if figure.ybreak is not None:
        if entry["horizontal"] or getattr(ax, "name", "") == "3d":
            return None
        y0, y1 = ax.get_ylim()
        low, high = figure.ybreak
        return (y0, low, high, y1) if y0 < low < high < y1 else None
    x0, x1 = ax.get_xlim()
    first, last = figure.xbreak
    freq = entry["freq"]
    if entry["horizontal"] or entry["categories"] is not None or (freq is not None and not isinstance(first, str)):
        return None  # periods on a dated axis, numbers on an undated one
    try:
        if freq is None:
            cut0, cut1 = float(first), float(last)
        else:
            cut0, cut1 = float(_x_number(first, freq)), float(_x_number(last, freq, end=True))
    except (ValueError, TypeError, SpecificationError):
        return None
    return (x0, cut0, cut1, x1) if x0 < cut0 < cut1 < x1 else None


def _draw_split(fig: Any, ax: Any, panel: Panel, theme: Theme, figure: Figure, index: int) -> dict[str, Any]:
    """A panel drawn in two parts with the break between them, if the break lies inside its values (or dates); else
    drawn whole. The parts are finished (limits, spines, marks) by ``_finish_split`` once the styles are applied."""
    whole = _draw_panel(ax, panel, theme, figure, index)
    edges = _break_fits(whole, figure)
    if edges is None:
        return whole
    if figure.ybreak is not None and whole["right"] is not None:
        raise SpecificationError(f"panel {index} has a right-hand axis: split its dates (xbreak), or draw it without one, "
                                 "to split its values")
    spec = ax.get_subplotspec()
    ax.remove()
    if whole["right"] is not None:
        whole["right"].remove()
    start, cut0, cut1, end = edges
    vertical = figure.ybreak is not None
    grid = (spec.subgridspec(2, 1, hspace=0.08, height_ratios=[end - cut1, cut0 - start]) if vertical
            else spec.subgridspec(1, 2, wspace=0.06, width_ratios=[cut0 - start, end - cut1]))
    first = _draw_panel(fig.add_subplot(grid[0]), panel, theme, figure, index)
    second = _draw_panel(fig.add_subplot(grid[1]), panel, theme, figure, index)
    first["parts"] = (second,)
    first["split"] = {"axis": "y" if vertical else "x", "edges": edges}
    return first


def _finish_split(entry: dict[str, Any]) -> None:
    """A split panel's two parts, once styled: each its side of the break, the inner spines and labels taken away,
    break marks where the axis is cut. The first part keeps the title, the legend and the value label."""
    a, b = entry["ax"], entry["parts"][0]["ax"]
    start, cut0, cut1, end = entry["split"]["edges"]
    if b.get_legend() is not None:
        b.get_legend().remove()
    for loc in ("left", "center", "right"):
        b.set_title("", loc=loc)
    if entry["split"]["axis"] == "y":  # a above b
        a.set_ylim(cut1, end)
        b.set_ylim(start, cut0)
        a.spines["bottom"].set_visible(False)
        a.tick_params(axis="x", bottom=False, labelbottom=False)
        a.set_xlabel("")
        b.set_ylabel("")
        a.plot([0], [0], transform=a.transAxes, label="_break", gid="pymetrix-break", **BREAK_MARK)
        b.plot([0], [1], transform=b.transAxes, label="_break", gid="pymetrix-break", **BREAK_MARK)
    else:  # a to the left of b
        a.set_xlim(start, cut0)
        b.set_xlim(cut1, end)
        a.spines["right"].set_visible(False)
        b.spines["left"].set_visible(False)
        b.tick_params(axis="y", left=False, labelleft=False)
        b.set_ylabel("")
        if entry["right"] is not None:  # the right-hand scale on the right of the whole panel only
            entry["right"].tick_params(axis="y", right=False, labelright=False)
            entry["right"].spines["right"].set_visible(False)
            entry["right"].set_ylabel("")
        if entry["parts"][0]["right"] is not None:
            entry["parts"][0]["right"].spines["left"].set_visible(False)
        a.plot([1], [0], transform=a.transAxes, label="_break", gid="pymetrix-break", **BREAK_MARK_X)
        b.plot([0], [0], transform=b.transAxes, label="_break", gid="pymetrix-break", **BREAK_MARK_X)


def _shade(ax: Any, spans: tuple[tuple[str, str, str], ...], freq: Any) -> set[str]:
    """Shade each span, from the start of its first period to the end of its last, behind the data; True when a
    shaded span has a label for the legend."""
    from .recessions import recessions

    labelled = set()
    expanded = []
    for start, end, label in spans:
        expanded += list(recessions(start[1:], freq)) if start.startswith("@") else [(start, end, label)]
    from matplotlib import dates

    lo, hi = ax.get_xlim()  # the data's dates: shading outside them is left out, and never widens the axis
    for start, end, label in expanded:
        try:
            first, last = pd.Period(start, freq=freq), pd.Period(end, freq=freq)
        except (ValueError, TypeError) as exc:
            raise SpecificationError(f"cannot shade {start}:{end}: not periods of this figure's frequency ({exc})") from exc
        left, right = dates.date2num(first.to_timestamp()), dates.date2num((last + 1).to_timestamp())
        if right <= lo or left >= hi:
            continue
        ax.axvspan(max(left, lo), min(right, hi), color="#7F7F7F", alpha=0.18, linewidth=0, zorder=0,
                   label=label if label and label not in labelled else None)
        if label:
            labelled.add(label)
    ax.set_xlim(lo, hi)
    return labelled


def _period_start(text: str, freq: Any) -> Any:
    try:
        return pd.Period(text, freq=freq).to_timestamp()
    except (ValueError, TypeError) as exc:
        raise SpecificationError(f"{text}: not a period of this figure's frequency ({exc})") from exc


def _marks(ax: Any, figure: Figure, freq: Any, horizontal: bool, theme: Theme) -> None:
    """Events (dated panels) and reference lines, labelled in small type at the axis' edge."""
    from matplotlib import transforms

    size = theme.font_size - 1
    box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0}
    for value, label in figure.reference:
        if horizontal:
            ax.axvline(value, color="#555555", linewidth=0.9, linestyle=":", zorder=1)
            if label:
                ax.text(value, 1.0, f" {label}", transform=transforms.blended_transform_factory(ax.transData, ax.transAxes),
                        ha="left", va="top", fontsize=size, color="#444444", bbox=box)
        else:
            ax.axhline(value, color="#555555", linewidth=0.9, linestyle=":", zorder=1)
            if label:
                ax.text(1.0, value, f"{label} ", transform=transforms.blended_transform_factory(ax.transAxes, ax.transData),
                        ha="right", va="bottom", fontsize=size, color="#444444", bbox=box)
    if freq is None:
        return
    for period, label in figure.events:
        x = _period_start(period, freq)
        if not _inside(ax, x):  # outside the dates shown: left out, never widening the axis
            continue
        ax.axvline(x, color="#555555", linewidth=0.9, linestyle="--", zorder=1)
        if label:
            ax.text(x, 0.98, f"{label} ", transform=transforms.blended_transform_factory(ax.transData, ax.transAxes),
                    rotation=90, ha="right", va="top", fontsize=size, color="#444444", bbox=box)


def _inside(ax: Any, x: Any) -> bool:
    """Whether a date (or number) lies within the x-axis as drawn so far."""
    from matplotlib import dates

    value = float(dates.date2num(x)) if not isinstance(x, (int, float, np.floating, np.integer)) else float(x)
    lo, hi = ax.get_xlim()
    return lo <= value <= hi


def _line_ends(ax: Any, lines: list[tuple[Any, Any, str]], theme: Theme) -> None:
    """Each line's name just right of its last value, in its colour; names that would overlap are moved apart. The
    x-axis is widened to make room."""
    from matplotlib import dates

    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.16 * (hi - lo))
    ends = [np.asarray(line.get_xdata())[-1] for _, line, _ in lines if len(line.get_xdata())]
    if ends:  # the dates stop where the data do
        ax.pymetrix_data_end = max(float(dates.date2num(e)) if not isinstance(e, (int, float, np.floating)) else float(e) for e in ends)
    by_axes: dict[int, list[tuple[Any, float, float, str, str]]] = {}
    for target, line, label in lines:
        xs, ys = np.asarray(line.get_xdata()), np.asarray(line.get_ydata(), float)
        ok = np.flatnonzero(np.isfinite(ys))
        if len(ok):
            by_axes.setdefault(id(target), []).append((target, xs[ok[-1]], float(ys[ok[-1]]), label, line.get_color()))
    for items in by_axes.values():
        target = items[0][0]
        bottom, top = target.get_ylim()
        gap = 0.055 * (top - bottom)
        items.sort(key=lambda item: item[2])
        placed: list[float] = []
        for _, _, y, _, _ in items:  # upwards, each at least a gap above the one below
            placed.append(max(y, placed[-1] + gap) if placed else y)
        overflow = placed[-1] - (top - gap / 2) if placed else 0.0
        if overflow > 0:  # too high: move the stack down, keeping the gaps
            placed = [p - overflow for p in placed]
        for (_, x, y, label, color), at in zip(items, placed):
            target.annotate(label, xy=(x, at), xytext=(4, 0), textcoords="offset points", va="center", ha="left",
                            color=color, fontsize=theme.font_size - 0.5)


def _bar_values(bars: list[tuple[Any, Any, bool]], theme: Theme) -> None:
    """Each bar's value on it: inside stacked segments (in white), at the end of plain bars; one number format for the
    panel, with as many decimals as its largest value needs."""
    if not bars:
        return
    values = np.abs(np.concatenate([np.asarray(c.datavalues, float) for _, c, _ in bars]))
    top = float(np.nanmax(values)) if np.isfinite(values).any() else 0.0
    fmt = "{:,.0f}" if top >= 100 else "{:,.1f}"
    for target, container, stacked in bars:
        target.bar_label(container, labels=[fmt.format(v) if np.isfinite(v) and v != 0 else "" for v in container.datavalues],
                         label_type="center" if stacked else "edge", fontsize=theme.font_size - 1.5, padding=0 if stacked else 2,
                         color="white" if stacked else "#333333")


def _fan_keys(handles: list[Any], names: list[str]) -> list[Any]:
    """A fan's bands in the legend as they look on the chart: the bands overlap, so the inner ones are darker there,
    while matplotlib would key each with its own (equal) shading. Bands named as coverages ("90%", "60%"...) in one
    colour, widest first, are keyed with the shading where they are drawn: 1 - (1 - alpha)^k for the k-th."""
    import re

    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    seen: dict[tuple[float, ...], int] = {}
    out = list(handles)
    for k, (handle, name) in enumerate(zip(handles, names)):
        if not isinstance(handle, PolyCollection) or not re.fullmatch(r"\d+(\.\d+)?%", str(name)):
            continue
        faces = handle.get_facecolor()
        if not len(faces):
            continue
        rgb = tuple(round(v, 4) for v in to_rgb(faces[0]))
        alpha = handle.get_alpha() if handle.get_alpha() is not None else float(faces[0][3])
        layers = seen.get(rgb, 0) + 1
        seen[rgb] = layers
        if layers > 1:
            out[k] = Patch(facecolor=rgb, alpha=1 - (1 - alpha) ** layers, linewidth=0, label=name)
    return out


def _house_axes(ax: Any, theme: Theme, grid: bool = True, value_grid: str | None = None) -> None:
    """The house look's ticks and grid (``figure_style.chart_rc``, in force while a figure is drawn) pinned on the axes
    itself, so ticks matplotlib makes later (on a resize, a zoom, a new scale) keep it whatever the global settings:
    ticks outwards, the grid's major lines only, light grey, behind the data. ``value_grid``: the one axis to grid
    (bars: lines across the values only)."""
    from matplotlib import rcParams

    label = rcParams["xtick.labelcolor"]
    ax.tick_params(which="major", direction=rcParams["xtick.direction"], length=rcParams["xtick.major.size"],
                   width=rcParams["xtick.major.width"], color=rcParams["xtick.color"],
                   labelcolor=rcParams["xtick.color"] if label == "inherit" else label, labelsize=theme.font_size)
    ax.tick_params(which="minor", direction=rcParams["xtick.direction"], length=rcParams["xtick.minor.size"],
                   color=rcParams["xtick.color"])
    ax.set_axisbelow(True)
    if grid and theme.grid:
        ax.grid(True, which="major", axis=value_grid or "both", color=rcParams["grid.color"], linewidth=rcParams["grid.linewidth"],
                alpha=rcParams["grid.alpha"], linestyle=rcParams["grid.linestyle"])
        if value_grid:
            ax.grid(False, axis="y" if value_grid == "x" else "x")
    else:
        ax.grid(False)


def _draw_panel(ax: Any, panel: Panel, theme: Theme | None = None, figure: Figure | None = None, index: int = 0) -> Any:
    theme = theme or get_theme()
    colors = iter(theme.palette * 4)
    styles = iter(theme.line_styles * 40)
    horizontal = any((isinstance(layer, Bars) and layer.horizontal)  # named estimates are listed down the side too
                     or (isinstance(layer, Estimates) and not isinstance(layer.estimate.index, pd.PeriodIndex)) for layer in panel.layers)
    estimates = [layer for layer in panel.layers if isinstance(layer, Estimates)]
    freq, mid = _panel_dates(panel, horizontal)  # mixed frequencies: each value at the middle of its period
    ax.spines[["top", "right"]].set_visible(False)
    _house_axes(ax, theme, value_grid=("x" if horizontal else "y") if any(isinstance(layer, Bars) for layer in panel.layers) else None)
    right = ax.twinx() if any(getattr(layer, "axis", "left") == "right" for layer in panel.layers) else None
    if right is not None:
        _house_axes(right, theme, grid=False)
        right.spines[["top"]].set_visible(False)
        right.spines[["right"]].set_visible(True)  # a right-hand scale keeps its axis line
    labelled = False
    stacks: dict[tuple[str, bool], dict[str, float]] = {}  # running ends of stacked bars, by axis and sign, by period
    side_by_side = [layer for layer in panel.layers if isinstance(layer, Bars) and not layer.stacked]
    drawn_bars: list[tuple[Any, Any, bool]] = []  # (axes, bar container, stacked), for values on the bars
    shade_labels: set[str] = set()
    named_lines: list[tuple[Any, Any, str]] = []  # (axes, line, label), for labels at the lines' ends
    artists: dict[str, list[Any]] = {}  # each layer's label -> what it drew, for styling
    end_labels = figure is not None and figure.line_labels
    categories: list[str] | None = None
    for layer in panel.layers:
        target = right if getattr(layer, "axis", "left") == "right" and right is not None else ax
        if isinstance(layer, Band):
            color = _color(layer.color or PALETTE[0], theme, colors)
            alpha = theme.band_alpha if layer.alpha in (0.15, 0.18) else layer.alpha  # the standard shading follows the theme
            band = target.fill_between(_x(layer.lower, mid), layer.lower.to_numpy(float), layer.upper.to_numpy(float),
                                       color=color, alpha=alpha, linewidth=0, label=layer.label or None,
                                       step="post" if layer.step else None)
            artists.setdefault(layer.label, []).append(("band", band))
            labelled |= bool(layer.label)
        elif isinstance(layer, Line):
            style = next(styles) if layer.style == "-" else layer.style
            width = theme.line_width if layer.width == 1.6 else layer.width * theme.line_width / 1.6
            (line,) = target.plot(_x(layer.y, mid), layer.y.to_numpy(float), style, color=_color(layer.color, theme, colors),
                                  linewidth=width, label=None if end_labels else (layer.label or None),
                                  drawstyle="steps-post" if layer.step else "default")
            artists.setdefault(layer.label, []).append(("line", line))
            if end_labels and layer.label:
                named_lines.append((target, line, layer.label))
            else:
                labelled |= bool(layer.label)
        elif isinstance(layer, Bars):
            y = np.nan_to_num(layer.y.to_numpy(float)) if layer.stacked else layer.y.to_numpy(float)
            x, gap, labels = _bar_positions(layer.y.index, layer.horizontal, mid)
            categories = labels if labels is not None else categories
            n = 1 if layer.stacked else len(side_by_side)
            width = 0.8 * gap / max(n, 1)
            if not layer.stacked and n > 1:
                x = _shift(x, (side_by_side.index(layer) - (n - 1) / 2) * width)
            base = None
            if layer.stacked:
                keys = [str(k) for k in layer.y.index]
                ends = stacks.setdefault((layer.axis, layer.horizontal), {})
                base = np.array([ends.get((k, v >= 0), 0.0) for k, v in zip(keys, y)])  # type: ignore[call-overload]
                for k, v, b in zip(keys, y, base):
                    ends[(k, v >= 0)] = b + v  # type: ignore[index]
            color = _color(layer.color, theme, colors)
            if layer.horizontal:
                container = target.barh(x, y, height=width, left=base, color=color, label=layer.label or None)
            else:
                container = target.bar(x, y, width=width, bottom=base, color=color, label=layer.label or None)
            drawn_bars.append((target, container, layer.stacked))
            artists.setdefault(layer.label, []).append(("bars", container))
            labelled |= bool(layer.label)
        elif isinstance(layer, Box):
            values = [s.dropna().to_numpy(float) for s in layer.series]
            boxes = ax.boxplot(values, widths=0.5, patch_artist=True, medianprops={"color": "#222222", "linewidth": 1.6},
                               flierprops={"markersize": 3, "alpha": 0.6})
            for i, patch in enumerate(boxes["boxes"]):
                patch.set_facecolor(theme.palette[i % len(theme.palette)])
                patch.set_alpha(0.45)
            ax.set_xticks(range(1, len(values) + 1), list(layer.labels) or [str(s.name or i + 1) for i, s in enumerate(layer.series)])
        elif isinstance(layer, Scatter):
            y = layer.y.to_numpy(float)
            if isinstance(layer.x.dtype, pd.PeriodDtype):  # dates, as points on the panel's dated axis (VaR violations, say)
                x = np.asarray(_period_x(pd.PeriodIndex(layer.x), mid))
                ok = np.isfinite(y) & layer.x.notna().to_numpy()
            else:
                x = layer.x.to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
            color = next(colors)
            ax.scatter(x[ok], y[ok], s=12, alpha=0.7, color=color, label=layer.label or None)
            if layer.fit and ok.sum() > 1:
                slope, intercept = np.polyfit(x[ok], y[ok], 1)
                grid = np.linspace(x[ok].min(), x[ok].max(), 50)
                ax.plot(grid, intercept + slope * grid, color="#444444", linewidth=1.2, label=f"fit: slope {slope:.3g}")
                labelled = True
            ax.set_xlabel(panel.xlabel or str(layer.x.name or ""), fontsize=theme.font_size)
            ax.set_ylabel(panel.ylabel or str(layer.y.name or ""), fontsize=theme.font_size)
            labelled |= bool(layer.label)
        elif isinstance(layer, Histogram):
            values = layer.values.dropna().to_numpy(float)
            if not len(values):  # nothing to draw (all missing, or none in the span)
                continue
            ax.hist(values, bins=layer.bins, density=True, color="#56B4E9", edgecolor="white", linewidth=0.4,
                    label=layer.label or None)
            if layer.normal and len(values) > 1:
                from scipy import stats

                grid = np.linspace(values.min(), values.max(), 200)
                ax.plot(grid, stats.norm.pdf(grid, values.mean(), values.std(ddof=1)), color="#D55E00", linewidth=1.4, label="normal")
                labelled = True
        elif isinstance(layer, Estimates):
            line = _draw_estimates(target, layer, estimates, theme, colors, mid)
            categories = _estimate_names(estimates) if not isinstance(layer.estimate.index, pd.PeriodIndex) else categories
            artists.setdefault(layer.label, []).append(("line", line))
            labelled |= bool(layer.label)
        elif isinstance(layer, Heatmap):
            _draw_heatmap(ax, layer, theme)
        elif isinstance(layer, QQ):
            theory, z = qq_points(layer.values)
            ax.scatter(theory, z, s=12, alpha=0.75, color=_color(None, theme, colors), label=layer.label or None)
            ax.axline((0.0, 0.0), slope=1.0, color="#888888", linewidth=0.9, linestyle="--")  # where normal values lie
            ax.set_xlabel(panel.xlabel or "normal quantiles", fontsize=theme.font_size)
            ax.set_ylabel(panel.ylabel or "standardised values", fontsize=theme.font_size)
            labelled |= bool(layer.label)
        elif isinstance(layer, Ecdf):
            values = layer.values.dropna().to_numpy(float)
            if not len(values):  # nothing to draw (all missing, or none in the span)
                continue
            line = ax.ecdf(values, complementary=layer.complementary, color=_color(layer.color, theme, colors),
                           linewidth=theme.line_width, label=layer.label or None)
            artists.setdefault(layer.label, []).append(("line", line))
            labelled |= bool(layer.label)
            ax.set_ylabel(panel.ylabel or ("share above" if layer.complementary else "share at or below"), fontsize=theme.font_size)
        elif isinstance(layer, Hist2d):
            x, y = layer.x.to_numpy(float), layer.y.to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            *_, image = ax.hist2d(x[ok], y[ok], bins=layer.bins, cmap=layer.cmap, cmin=1)
            bar = ax.figure.colorbar(image, ax=ax, shrink=0.9)
            bar.ax.tick_params(labelsize=theme.font_size - 1)
            bar.set_label(layer.label, fontsize=theme.font_size)
            ax.set_xlabel(panel.xlabel or str(layer.x.name or ""), fontsize=theme.font_size)
            ax.set_ylabel(panel.ylabel or str(layer.y.name or ""), fontsize=theme.font_size)
        elif isinstance(layer, Hexbin):
            x, y = layer.x.to_numpy(float), layer.y.to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            cells = ax.hexbin(x[ok], y[ok], gridsize=layer.gridsize, cmap=layer.cmap, mincnt=1)
            _colorbar(ax, cells, layer.label, theme)
            ax.set_xlabel(panel.xlabel or str(layer.x.name or ""), fontsize=theme.font_size)
            ax.set_ylabel(panel.ylabel or str(layer.y.name or ""), fontsize=theme.font_size)
        elif isinstance(layer, Violin):
            values = [s.dropna().to_numpy(float) for s in layer.series]
            keep = [k for k, v in enumerate(values) if len(v)]  # nothing to draw for an empty series
            if not keep:
                continue
            parts = ax.violinplot([values[k] for k in keep], positions=[k + 1 for k in keep], widths=0.7, showmedians=layer.medians)
            for k, body in zip(keep, parts["bodies"]):
                body.set_facecolor(theme.palette[k % len(theme.palette)])
                body.set_edgecolor("#444444")
                body.set_alpha(0.45)
            for name in ("cmins", "cmaxes", "cbars", "cmedians"):
                if name in parts:
                    parts[name].set_color("#222222")
                    parts[name].set_linewidth(1.6 if name == "cmedians" else 0.9)
            ax.set_xticks(range(1, len(values) + 1), list(layer.labels) or [str(s.name or k + 1) for k, s in enumerate(layer.series)])
        elif isinstance(layer, Contour):
            _draw_contour(ax, layer, theme)
        elif isinstance(layer, Rule):
            at = _x_number(layer.y, freq) if isinstance(layer.y, str) else layer.y  # a period, on a dated axis
            if isinstance(layer.y, str) and not _inside(ax, at):
                continue
            (ax.axvline if layer.vertical else ax.axhline)(at, color=layer.color, linewidth=0.8, linestyle=layer.style)
    if figure is not None:
        if figure.shade and freq is not None:  # only dated panels: a histogram or scatter plot has no periods to shade
            shade_labels = _shade(ax, figure.shade, freq)
            labelled |= bool(shade_labels)
        _marks(ax, figure, freq, horizontal, theme)
        if figure.bar_labels:
            _bar_values(drawn_bars, theme)
        if named_lines:
            _line_ends(ax, named_lines, theme)
        if figure.log:
            from matplotlib import ticker

            (ax.set_xscale if horizontal else ax.set_yscale)("log")
            axis = ax.xaxis if horizontal else ax.yaxis
            axis.set_major_locator(ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
            axis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
            axis.set_minor_formatter(ticker.NullFormatter())
        if figure.ylim is not None:
            (ax.set_xlim if horizontal else ax.set_ylim)(*figure.ylim)
    found = _style_series(figure, index, panel, artists) if figure is not None else set()
    if categories is not None:
        _category_ticks(ax, categories, horizontal)
    if panel.title:
        ax.set_title(panel.title, fontsize=theme.title_size, loc="left")
    if panel.ylabel and not any(isinstance(layer, Scatter) for layer in panel.layers):
        ax.set_ylabel(panel.ylabel, fontsize=theme.font_size)
    if panel.xlabel and not any(isinstance(layer, Scatter) for layer in panel.layers):
        ax.set_xlabel(panel.xlabel, fontsize=theme.font_size)
    if right is not None and panel.ylabel_right:
        right.set_ylabel(panel.ylabel_right, fontsize=theme.font_size)
    if (panel.legend and labelled) or shade_labels:  # labelled shading is always explained, legend or not
        handles, names = ax.get_legend_handles_labels()
        if right is not None:
            more, more_names = right.get_legend_handles_labels()
            handles, names = handles + more, names + more_names
        if not panel.legend:
            handles, names = [h for h, n in zip(handles, names) if n in shade_labels], [n for n in names if n in shade_labels]
        handles = _fan_keys(handles, names)
        ax.legend(handles, names, fontsize=theme.font_size - 0.5, frameon=False, **legend_place(panel.legend_loc, len(names)))
    if freq is not None:
        _period_axis(ax, freq)
    elif categories is None and _whole_x(panel):  # horizons and periods: 0, 5, 10, not 2.5, 7.5
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
    ax.pymetrix_panel, ax.pymetrix_freq = index, freq  # where a click lands, for placing a note in the app
    if right is not None:
        right.pymetrix_panel, right.pymetrix_freq = index, freq
    return {"ax": ax, "right": right, "artists": artists, "found": found, "freq": freq, "mid": mid, "horizontal": horizontal,
            "categories": categories}


def _estimate_names(estimates: list[Estimates]) -> list[str]:
    """The names of a panel's named estimates, in the order they first appear."""
    return list(dict.fromkeys(str(n) for layer in estimates for n in layer.estimate.index))


def _draw_estimates(ax: Any, layer: Estimates, group: list[Estimates], theme: Theme, colors: Any, mid: bool = False) -> Any:
    """One set of estimates as ``errorbar``; side by side with the panel's others. Returns the markers' line."""
    n, k = len(group), group.index(layer)
    color = _color(layer.color, theme, colors)
    est, lo, hi = (s.to_numpy(float) for s in (layer.estimate, layer.lower, layer.upper))
    errors = np.vstack([est - lo, hi - est])
    if isinstance(layer.estimate.index, pd.PeriodIndex):  # along the dates
        x, gap, _ = _bar_positions(layer.estimate.index, False, mid)
        x = _shift(x, (k - (n - 1) / 2) * 0.6 * gap / max(n, 1)) if n > 1 else x
        drawn = ax.errorbar(x, est, yerr=errors, fmt=layer.marker, color=color, markersize=4, capsize=2, linewidth=1.2,
                            label=layer.label or None)
    else:  # down the side, a row per name, the first at the top
        names = _estimate_names(group)
        rows = np.array([names.index(str(i)) for i in layer.estimate.index], float) + (k - (n - 1) / 2) * 0.6 / max(n, 1)
        drawn = ax.errorbar(est, rows, xerr=errors, fmt=layer.marker, color=color, markersize=4, capsize=2, linewidth=1.2,
                            label=layer.label or None)
        if k == 0:
            ax.axvline(0.0, color="#888888", linewidth=0.8, linestyle="--")
    return drawn.lines[0]


_AUTO_SCALE_RATIO = 10.0  # "auto": a column's typical size more than this times another's largest value


def _is_shares(v: np.ndarray) -> bool:
    """Shares: no value below zero, and every row, or every column, adding up to the same whole multiple of 1 or of 100
    (to half a percent: tables rounded for display), with at least two rows and two columns."""
    finite = v[np.isfinite(v)]
    if finite.size == 0 or finite.min() < 0 or min(v.shape) < 2:
        return False
    for axis in (1, 0):
        sums = np.nansum(v, axis=axis)
        for unit in (1.0, 100.0):
            k = round(float(sums[0]) / unit)
            if k >= 1 and np.all(np.abs(sums - k * unit) <= 0.005 * k * unit):
                return True
    return False


def _heatmap_auto_scale(values: pd.DataFrame) -> Literal["table", "column"]:
    """What ``scale="auto"`` picks for a heatmap of ``values``: "column" when one column's typical size (the median of
    its absolute values, leaving out zeros) is more than ten times the largest absolute value in another column, so
    the columns are in different units or magnitudes (a posterior summary's effective draws beside its means; a Q
    statistic beside autocorrelations), and "table" otherwise. A table of one kind keeps one scale whatever its
    sizes: a square matrix with the same labels on its rows and columns (correlations, covariances), columns that are
    horizons, lags or dates (whole numbers or periods: responses by horizon), every column taking values both above
    and below zero (deviations: responses by shock, residuals), and shares (see ``_is_shares``: variance
    decompositions)."""
    if values.shape[0] == values.shape[1] and [str(i) for i in values.index] == [str(c) for c in values.columns]:
        return "table"
    columns = values.columns
    if isinstance(columns, (pd.PeriodIndex, pd.DatetimeIndex)) or (
            len(columns) and all(isinstance(c, (int, np.integer)) and not isinstance(c, bool) for c in columns)):
        return "table"  # one quantity along horizons, lags or dates (responses by horizon)
    v = values.to_numpy(float)
    if _is_shares(v):
        return "table"
    signs = [(np.nanmin(c) < 0 < np.nanmax(c)) for c in v.T if np.isfinite(c).any() and np.nanmax(np.abs(c)) > 0]
    if signs and all(signs):
        return "table"  # every column above and below zero: deviations, such as responses to shocks, on one scale about it
    typical, largest = [], []
    for j in range(v.shape[1]):
        a = np.abs(v[:, j][np.isfinite(v[:, j])])
        a = a[a > 0]
        if a.size:
            typical.append(float(np.median(a)))
            largest.append(float(a.max()))
    if len(typical) < 2:
        return "table"
    return "column" if max(typical) > _AUTO_SCALE_RATIO * min(largest) else "table"


def _heatmap_mode(layer: Heatmap) -> Literal["table", "column"]:
    """The scale a heatmap is drawn on: its own, or what "auto" picks for its values."""
    return _heatmap_auto_scale(layer.values) if layer.scale == "auto" else layer.scale  # type: ignore[return-value]


def _column_shades(v: np.ndarray) -> np.ndarray:
    """Each column of ``v`` mapped onto 0 to 1, its smallest value to 0 and its largest to 1 (a column of one value:
    0.5); missing values stay missing."""
    out = np.full(v.shape, np.nan)
    for j in range(v.shape[1]):
        col = v[:, j]
        ok = np.isfinite(col)
        if ok.any():
            lo, hi = float(col[ok].min()), float(col[ok].max())
            out[ok, j] = (col[ok] - lo) / (hi - lo) if hi > lo else 0.5
    return out


COLUMN_SCALE_NOTE = "each column coloured on its own scale"
_DARK_TEXT, _WHITE_TEXT, _BLACK_TEXT = "#1a1a1a", "#ffffff", "#000000"


def _luminance(rgb: Sequence[float]) -> float:
    """A colour's relative luminance (WCAG 2), from its red, green and blue as 0 to 1."""
    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(float(c)) for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_ratio(a: Any, b: Any) -> float:
    """The WCAG contrast ratio of two colours (anything matplotlib takes), 1 to 21."""
    from matplotlib.colors import to_rgb

    la, lb = sorted((_luminance(to_rgb(a)), _luminance(to_rgb(b))), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _text_colour(cell: Any) -> str:
    """The colour to write on a cell of colour ``cell``: dark grey or white, whichever has the higher contrast; black
    instead of the dark grey when neither reaches 4.5:1 (black or white always does)."""
    best = max((_DARK_TEXT, _WHITE_TEXT), key=lambda t: _contrast_ratio(t, cell))
    if _contrast_ratio(best, cell) >= 4.5:
        return best
    return max((_BLACK_TEXT, _WHITE_TEXT), key=lambda t: _contrast_ratio(t, cell))


def _heatmap_scale(layer: Heatmap) -> tuple[str, float, float]:
    """The colour map and the values at its ends: symmetric about the centre for a diverging scale; 0 and 1 for a
    heatmap coloured column by column (each column's shades run from 0 to 1)."""
    if _heatmap_mode(layer) == "column":
        return layer.cmap or "Blues", 0.0, 1.0
    v = layer.values.to_numpy(float)
    finite = v[np.isfinite(v)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    diverging = layer.center is not None or (lo < 0 < hi)
    if diverging:
        c = 0.0 if layer.center is None else float(layer.center)
        half = max(abs(lo - c), abs(hi - c)) or 1.0
        lo, hi = c - half, c + half
    lo = lo if layer.vmin is None else float(layer.vmin)
    hi = hi if layer.vmax is None else float(layer.vmax)
    return layer.cmap or ("RdBu_r" if diverging else "Blues"), lo, hi


def _cell_text(fmt: str, value: float) -> str:
    """A cell's value as ``fmt`` writes it, without the sign of a value that rounds to zero ("0.00", not "-0.00")."""
    text = fmt.format(value)
    return text[1:] if text.startswith("-") and not text.strip("-0.,% ") else text


def _draw_heatmap(ax: Any, layer: Heatmap, theme: Theme) -> Any:
    cmap, lo, hi = _heatmap_scale(layer)
    v = layer.values.to_numpy(float)
    by_column = _heatmap_mode(layer) == "column"
    shades = _column_shades(v) if by_column else v  # what is coloured; the cells still show their own values
    image = ax.imshow(shades, cmap=cmap, vmin=lo, vmax=hi, aspect="auto", interpolation="nearest")
    image.pymetrix_values = v  # the values under the colours, for the readout and a click
    rows, cols = [str(i) for i in layer.values.index], [str(c) for c in layer.values.columns]
    ax.set_yticks(range(len(rows)), rows)
    long = max(map(len, cols), default=0) * len(cols) > 60
    ax.set_xticks(range(len(cols)), cols, rotation=45 if long else 0, ha="right" if long else "center")
    ax.grid(False)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)
    if by_column:  # no one colour bar fits: a note says how the colours were chosen
        ax.set_xlabel(COLUMN_SCALE_NOTE, fontsize=theme.font_size - 0.5, style="italic", color="#555555")
    else:
        bar = ax.figure.colorbar(image, ax=ax, shrink=0.9)
        bar.ax.tick_params(labelsize=theme.font_size - 1)
        if layer.label:
            bar.set_label(layer.label, fontsize=theme.font_size)
    if layer.annotate and v.size <= 400:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                if np.isfinite(v[i, j]):  # dark type on light cells, white on dark ones
                    ax.text(j, i, _cell_text(layer.fmt, v[i, j]), ha="center", va="center", fontsize=theme.font_size - 1.5,
                            color=_text_colour(image.to_rgba(shades[i, j])))
    return image


def _colorbar(ax: Any, mappable: Any, label: str, theme: Theme, **options: Any) -> Any:
    """A colour bar beside the panel, its ticks and label in the theme's type."""
    bar = ax.figure.colorbar(mappable, ax=ax, **{"shrink": 0.9, **options})
    bar.ax.tick_params(labelsize=theme.font_size - 1)
    if label:
        bar.set_label(label, fontsize=theme.font_size)
    return bar


def grid_positions(index: pd.Index) -> tuple[np.ndarray, Any, list[str] | None]:
    """Where a grid's rows or columns are drawn: periods as matplotlib's date numbers (with their frequency), numbers
    as they are, and anything else (names) at 0, 1, 2... with the names as labels."""
    from matplotlib import dates

    if isinstance(index, pd.PeriodIndex):
        return np.asarray(dates.date2num(index.to_timestamp()), float), index.freq, None
    if isinstance(index, pd.DatetimeIndex):
        return np.asarray(dates.date2num(index), float), "D", None
    if pd.api.types.is_numeric_dtype(index) and not pd.api.types.is_bool_dtype(index):
        return np.asarray(index, float), None, None
    return np.arange(len(index), dtype=float), None, [str(v) for v in index]


def _series_positions(s: pd.Series) -> tuple[np.ndarray, Any]:
    """A 3-D layer's coordinates: a series of periods (or dates) as date numbers, with the frequency; numbers as they are."""
    from matplotlib import dates

    if isinstance(s.dtype, pd.PeriodDtype):
        periods = pd.PeriodIndex(s)
        return np.where(s.notna().to_numpy(), dates.date2num(periods.to_timestamp()), np.nan), periods.freq
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return np.asarray(dates.date2num(pd.DatetimeIndex(s)), float), "D"
    return s.to_numpy(float), None


def _label_axis(axis: Any, freq: Any, labels: list[str] | None) -> None:
    """Dates labelled as the data's periods (2023Q1), at most six; names at their positions, at most twelve."""
    from matplotlib import dates, ticker

    if freq is not None:
        def label(value: float, _pos: int = 0) -> str:
            return str(pd.Period(dates.num2date(value).replace(tzinfo=None), freq=freq))

        axis.set_major_locator(dates.AutoDateLocator(minticks=3, maxticks=6))
        axis.set_major_formatter(ticker.FuncFormatter(label))
    elif labels is not None:
        step = max(1, int(np.ceil(len(labels) / 12)))
        axis.set_ticks(np.arange(0, len(labels), step), [labels[k] for k in range(0, len(labels), step)])


def _draw_contour(ax: Any, layer: Contour, theme: Theme) -> Any:
    xs, xfreq, xnames = grid_positions(layer.values.columns)
    ys, yfreq, ynames = grid_positions(layer.values.index)
    levels = layer.levels if isinstance(layer.levels, (int, np.integer)) else list(layer.levels)
    colors = {"colors": layer.color} if layer.color else {"cmap": layer.cmap}
    draw = ax.contourf if layer.filled else ax.contour
    contours = draw(xs, ys, layer.values.to_numpy(float), levels=levels, **colors)
    if layer.clabel:  # filled contours are labelled on thin lines at their levels, as matplotlib advises
        lines = contours if not layer.filled else ax.contour(xs, ys, layer.values.to_numpy(float), levels=contours.levels,
                                                             colors="#333333", linewidths=0.5)
        ax.clabel(lines, fontsize=theme.font_size - 1.5, fmt="%.3g")
    if layer.colorbar:
        _colorbar(ax, contours, layer.label, theme)
    ax.grid(False)
    _label_axis(ax.xaxis, xfreq, xnames)
    _label_axis(ax.yaxis, yfreq, ynames)
    ax.set_xlabel(str(layer.values.columns.name or ""), fontsize=theme.font_size)
    ax.set_ylabel(str(layer.values.index.name or ""), fontsize=theme.font_size)
    return contours


def _draw_panel3d(ax: Any, panel: Panel, theme: Theme, figure: Figure | None = None, index: int = 0) -> dict[str, Any]:
    """A 3-D panel: surfaces, wireframes, lines and points; dates on any axis labelled as periods."""
    colors = iter(theme.palette * 4)
    artists: dict[str, list[Any]] = {}
    labelled = False
    names = {"x": "", "y": "", "z": ""}  # the axes' labels from the data, where the panel gives none
    ax.tick_params(labelsize=theme.font_size - 1)
    for layer in panel.layers:
        if isinstance(layer, Surface):
            xs, xfreq, xnames = grid_positions(layer.values.columns)
            ys, yfreq, ynames = grid_positions(layer.values.index)
            X, Y = np.meshgrid(xs, ys)
            Z = layer.values.to_numpy(float)
            if layer.wireframe:
                art = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color=_color(layer.color, theme, colors), linewidth=0.7,
                                        alpha=layer.alpha, label=layer.label or None)
                artists.setdefault(layer.label, []).append(("wireframe", art))
                labelled |= bool(layer.label)
            else:
                art = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=layer.cmap, linewidth=0, antialiased=False, alpha=layer.alpha)
                _colorbar(ax, art, layer.label, theme, shrink=0.6, pad=0.1)
                ax.set_anchor("C")  # the colour bar anchors its axes east; a 3-D box cannot widen, so it hugged the bar
            _label_axis(ax.xaxis, xfreq, xnames)
            _label_axis(ax.yaxis, yfreq, ynames)
            names = {"x": str(layer.values.columns.name or ""), "y": str(layer.values.index.name or ""), "z": layer.label}
        elif isinstance(layer, (Line3d, Scatter3d)):
            (x, xfreq), (y, yfreq), (z, zfreq) = (_series_positions(getattr(layer, a)) for a in ("x", "y", "z"))
            color = _color(layer.color, theme, colors)
            if isinstance(layer, Line3d):
                width = theme.line_width if layer.width == 1.6 else layer.width * theme.line_width / 1.6
                (art,) = ax.plot(x, y, z, color=color, linewidth=width, label=layer.label or None)
                artists.setdefault(layer.label, []).append(("line", art))
            else:
                ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                art = ax.scatter(x[ok], y[ok], z[ok], s=12, alpha=0.8, color=color, label=layer.label or None)
                artists.setdefault(layer.label, []).append(("points", art))
            for axis, freq in ((ax.xaxis, xfreq), (ax.yaxis, yfreq), (ax.zaxis, zfreq)):
                if freq is not None:
                    _label_axis(axis, freq, None)
            names = {a: str(getattr(layer, a).name or "") for a in ("x", "y", "z")}
            labelled |= bool(layer.label)
    if figure is not None:
        if figure.log:
            raise SpecificationError("a log scale is for flat panels; a 3-D panel's heights can be logged in the data")
        if figure.ylim is not None:  # the values shown: the heights
            ax.set_zlim(*figure.ylim)
    found = _style_series(figure, index, panel, artists) if figure is not None else set()
    if panel.title:
        ax.set_title(panel.title, fontsize=theme.title_size, loc="left")
    for axis in ("x", "y", "z"):
        text = getattr(panel, f"{axis}label") or names[axis]
        if text:
            getattr(ax, f"set_{axis}label")(text, fontsize=theme.font_size)
    if panel.legend and labelled:
        ax.legend(fontsize=theme.font_size - 0.5, frameon=False, **legend_place(panel.legend_loc, len(ax.get_legend_handles_labels()[1])))
    ax.pymetrix_panel, ax.pymetrix_freq = index, None
    return {"ax": ax, "right": None, "artists": artists, "found": found, "freq": None, "mid": False, "horizontal": False,
            "categories": None}


def _x_text(x: float, freq: Any) -> str:
    """A position along the x-axis as text: the period on a dated axis, else the number."""
    if freq is not None:
        from matplotlib import dates

        try:
            return str(pd.Period(dates.num2date(x).replace(tzinfo=None), freq=freq))
        except (ValueError, OverflowError):
            return ""
    return f"{x:.4g}"


def readout(ax: Any, px: float, py: float, reach: float = 20.0) -> str:
    """What is under the point (``px``, ``py``, in pixels) of a drawn panel, in words: a heatmap's cell and value,
    the bar there and its value, or the nearest point of a line (within ``reach`` pixels; either axis of a panel
    with a right-hand one), with the date as a period; else the position. For the app's hover readout."""
    if getattr(ax, "name", "") == "3d":  # a 3-D panel: matplotlib's own reading of the point on its panes
        return ax.format_coord(*ax.transData.inverted().transform((px, py)))
    freq = getattr(ax, "pymetrix_freq", None)
    panel = getattr(ax, "pymetrix_panel", None)
    family =[a for a in ax.figure.axes if panel is not None and getattr(a, "pymetrix_panel", None) == panel] or [ax]
    for a in family:
        for image in a.get_images():  # a heatmap: the cell under the pointer
            x, y = a.transData.inverted().transform((px, py))
            cells = getattr(image, "pymetrix_values", None)  # the values, where the colours are shades (by column)
            cells = image.get_array() if cells is None else cells
            row, col = int(round(y)), int(round(x))
            if 0 <= row < cells.shape[0] and 0 <= col < cells.shape[1]:
                rows = [t.get_text() for t in a.get_yticklabels()]
                cols = [t.get_text() for t in a.get_xticklabels()]
                name = f"{rows[row] if row < len(rows) else row}, {cols[col] if col < len(cols) else col}"
                value = cells[row, col]
                return f"{name}: " + ("no value" if np.ma.is_masked(value) or not np.isfinite(value) else f"{float(value):.4g}")
        for container in a.containers:  # bars: the one under the pointer
            patches = getattr(container, "patches", None)
            values = getattr(container, "datavalues", None)
            if patches is None or values is None:
                continue
            for patch, value in zip(patches, values):
                if patch.contains_point((px, py)):
                    horizontal = getattr(container, "orientation", "vertical") == "horizontal"
                    where = (_tick_text(a, patch.get_y() + patch.get_height() / 2, "y") if horizontal
                             else _x_text(patch.get_x() + patch.get_width() / 2, freq))
                    label = container.get_label()
                    return (f"{label}  " if label and not label.startswith("_") else "") + (f"{where}: " if where else "") + f"{float(value):.4g}"
    best: tuple[float, str] | None = None
    for a in family:  # lines: the nearest point
        for line in a.get_lines():
            label = line.get_label()
            if not line.get_visible() or not label or label.startswith("_"):
                continue
            xs = np.asarray(line.get_xdata(orig=False), dtype=float)  # dates as matplotlib's date numbers
            ys = np.asarray(line.get_ydata(), dtype=float)
            ok = np.isfinite(xs) & np.isfinite(ys)
            if not ok.any():
                continue
            pixels = a.transData.transform(np.column_stack([xs[ok], ys[ok]]))
            distance = np.hypot(pixels[:, 0] - px, pixels[:, 1] - py)
            k = int(np.argmin(distance))
            if distance[k] <= reach and (best is None or distance[k] < best[0]):
                x, y = xs[ok][k], ys[ok][k]
                best = (float(distance[k]), f"{label}  {_x_text(x, freq)}: {y:.4g}")
    if best is not None:
        return best[1]
    x, y = ax.transData.inverted().transform((px, py))
    return f"{_x_text(x, freq)}, {y:.4g}"


def _tick_text(ax: Any, position: float, axis: str) -> str:
    """The category label at ``position`` on a category axis (horizontal bars)."""
    ticks = ax.get_yticks() if axis == "y" else ax.get_xticks()
    labels = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for tick, label in zip(ticks, labels):
        if abs(tick - position) < 0.5:
            return label.get_text()
    return f"{position:.4g}"


def _whole_x(panel: Panel) -> bool:
    """True when every line, band and bar in the panel is drawn against whole numbers (periods, horizons, draws)."""
    indexes = [getattr(layer, attr).index for layer in panel.layers for attr in ("y", "lower")
               if isinstance(getattr(layer, attr, None), pd.Series) and not isinstance(layer, (Scatter, Hist2d, Hexbin))]
    return bool(indexes) and all(pd.api.types.is_integer_dtype(i) for i in indexes)


# ---------------------------------------------------------------------------------------------
# figures for data and results

Kind = Literal["line", "step", "bar", "barh", "area", "scatter", "histogram", "box", "contributions", "fan", "vintages", "correlation",
               "ecdf", "hist2d", "violin", "hexbin", "surface"]
KINDS = ("line", "step", "bar", "barh", "area", "scatter", "histogram", "box", "contributions", "fan", "vintages", "correlation",
         "ecdf", "hist2d", "violin", "hexbin", "surface")
BINNED = ("histogram", "hist2d", "hexbin")  # the kinds that take bins
STACKABLE = ("bar", "barh", "area")
#: How series can be transformed before they are drawn, and the axis label that says so.
TRANSFORMS = {"none": "", "index": "index, first period shown = 100", "change": "change on the previous period",
              "growth": "% change on the previous period", "yoy": "% change on a year earlier", "log": "natural log"}


def _transform(s: pd.Series, how: str) -> pd.Series:
    """``s`` transformed (``TRANSFORMS``); ``index`` is applied later, to the periods shown."""
    if how in ("none", "index"):
        return s
    if how == "change":
        return s.diff()
    if how == "growth":
        return 100 * s.pct_change(fill_method=None)
    if how == "yoy":
        from .filters import periods_per_year

        if not isinstance(s.index, pd.PeriodIndex):
            raise SpecificationError("year-on-year changes need dated (time-series) data")
        return 100 * s.pct_change(periods_per_year(s.index.freqstr), fill_method=None)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.Series(np.log(s.where(s > 0).to_numpy(float)), index=s.index, name=s.name)


def _index_to_100(s: pd.Series) -> pd.Series:
    first = s.dropna()
    if first.empty or first.iloc[0] == 0:
        raise SpecificationError(f"{s.name}: cannot index to 100: its first value shown is missing or zero")
    return 100 * s / first.iloc[0]


def _columns(data: Dataset, frame: pd.DataFrame, names: list[str], kind: str, entity: Any) -> dict[str, pd.Series]:
    """The series to draw, by label: for panels, one entity's, or one series with a column per entity."""
    entities = data.entity_axes
    if entities and entity is not None:
        frame = frame.xs(entity, level=entities[0].name)
    if entities and entity is None:
        if data.time_axis is None:  # a cross-section: the entities label the bars
            return {n: frame[n].droplevel([a.name for a in entities[1:]]) if len(entities) > 1 else frame[n] for n in names}
        if kind in ("scatter", "histogram", "box", "violin"):
            return {n: frame[n].reset_index(drop=True) for n in names}
        if len(names) == 1:
            level = entities[0].name
            return {f"{names[0]} ({key})": s.droplevel(level) for key, s in frame[names[0]].groupby(level=level)}
        raise SpecificationError("for several series of a panel, choose an entity (entity=...)")
    return {n: frame[n] for n in names}


def series_figure(data: Dataset, names: Sequence[str] | None = None, kind: Kind = "line", layout: Literal["one", "each"] = "one",
                  span: Any = None, entity: Any = None, title: str = "", stacked: bool = False, right: Sequence[str] = (),
                  transform: str = "none", shade: Any = (), total: str | None = None,
                  coverages: Sequence[float] | None = None, bins: int | None = None) -> Figure:
    """Series of a dataset: in one panel or one panel each; for panels, one entity or (for one series) a line per entity.

    ``kind``: line, step, bar (side by side), barh (horizontal bars), area, scatter (x first), histogram, box,
    contributions (the series as stacked parts, with ``total``, a series, or their sum as a line), fan (each series
    with its bands, from ``{name}_lo90``/``{name}_hi90`` or ``{name}_se`` columns, at ``coverages``), vintages (each
    series a release of the same data, oldest first), correlation (the series' correlations, as a heatmap), ecdf (each
    series' empirical distribution, in one panel), hist2d (a two-dimensional histogram of y against x, x first), violin
    (each series' density, as box plots are drawn), hexbin (hexagonal bins of y against x, x first) or surface (the
    series side by side along x, the dates along y, the values as heights: a yield curve over time, in 3-D).
    ``bins``: the histogram's or hist2d's bins, or how many hexagons across for hexbin.
    ``stacked`` stacks bars or areas (negative values below zero). ``right`` names series drawn against a right-hand
    axis. ``transform``: none, index (first period shown = 100), change, growth (% on the previous period), yoy (% on a
    year earlier) or log. ``shade``: periods to shade (``"2008Q3:2009Q2, 2020Q1:2020Q2"``), or the name of a 0/1 series
    of the dataset (such as recession dates), whose runs of ones are shaded."""
    if bins is not None and kind not in BINNED:
        raise SpecificationError("bins is for kind='histogram', 'hist2d' or 'hexbin'")
    if bins is not None and (isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 1):
        raise SpecificationError(f"bins is a whole number of at least 1; got {bins!r}")
    if kind in ("histogram", "hist2d", "hexbin", "ecdf") and bins is not None or kind in ("hist2d", "hexbin", "ecdf"):
        figure = _distribution_figure(data, names, kind, span, entity, title, transform, bins)
    elif kind == "surface":
        figure = _surface_series_figure(data, names, span, entity, title, transform)
    elif kind == "correlation":
        names = list(names) if names else [c for c in data.frame.select_dtypes("number").columns][:12]
        if len(names) < 2:
            raise SpecificationError("correlations need at least two series")
        corr = data.frame.loc[data.sample_mask(span), names].astype(float).corr()
        figure = heatmap_figure(corr, title=title, label="correlation", center=0.0, vmin=-1.0, vmax=1.0)
    elif kind in ("contributions", "fan", "vintages"):
        figure = _macro_figure(data, names, kind, span, title, total, coverages)
    else:
        if total is not None or coverages is not None:
            raise SpecificationError("total is for kind='contributions', coverages for kind='fan'")
        figure = _series_figure(data, names, kind, layout, span, entity, title, stacked, right, transform)
    return replace(figure, shade=_dataset_shade(data, shade)) if shade else figure


def _distribution_figure(data: Dataset, names: Sequence[str] | None, kind: str, span: Any, entity: Any, title: str,
                         transform: str, bins: int | None) -> Figure:
    """Histograms with a number of bins, empirical distributions, and two-dimensional histograms."""
    if kind == "histogram":
        figure = _series_figure(data, names, kind, "each", span, entity, title, False, (), transform)
        return replace(figure, panels=tuple(replace(p, layers=tuple(replace(layer, bins=int(bins)) if isinstance(layer, Histogram) else layer
                                                                      for layer in p.layers)) for p in figure.panels))
    names = list(names) if names else [c for c in data.frame.select_dtypes("number").columns][:8]
    columns = _columns(data, data.frame[data.sample_mask(span)], names, "line", entity)
    if transform != "none":
        whole = _columns(data, data.frame, names, "line", entity)
        columns = {n: _transform(whole[n], transform).reindex(s.index) for n, s in columns.items()}
    if kind == "ecdf":
        layers = tuple(Ecdf(s, n, color=PALETTE[k % len(PALETTE)]) for k, (n, s) in enumerate(columns.items()))
        return Figure((Panel(layers, legend=len(layers) > 1, xlabel=TRANSFORMS[transform]),), title=title)
    if len(names) != 2:
        what = "hexagonal bins need" if kind == "hexbin" else "a two-dimensional histogram needs"
        raise SpecificationError(f"{what} two series: x first, then y")
    x, y = (columns[n].rename(n) for n in names)
    layer = Hexbin(x, y, gridsize=int(bins or 30)) if kind == "hexbin" else Hist2d(x, y, bins=int(bins or 30))
    return Figure((Panel((layer,), title=f"{names[1]} against {names[0]}", legend=False),), title=title)


def _surface_series_figure(data: Dataset, names: Sequence[str] | None, span: Any, entity: Any, title: str, transform: str) -> Figure:
    """The series side by side along x (in the order named), the dates along y, their values as heights."""
    names = list(names) if names else [c for c in data.frame.select_dtypes("number").columns][:12]
    if len(names) < 2:
        raise SpecificationError("a surface needs at least two series, side by side along x (a yield curve's maturities, say)")
    columns = _columns(data, data.frame[data.sample_mask(span)], names, "line", entity)
    if transform != "none":
        whole = _columns(data, data.frame, names, "line", entity)
        columns = {n: _transform(whole[n], transform).reindex(s.index) for n, s in columns.items()}
    frame = pd.DataFrame(columns)
    if not isinstance(frame.index, pd.PeriodIndex):
        raise SpecificationError("a surface of series needs dated (time-series) data: the dates run along y")
    frame.columns = pd.Index([str(c) for c in frame.columns], name="series")
    return Figure((Panel((Surface(frame.astype(float), label=TRANSFORMS[transform] or "value"),), legend=False),), title=title)


def _macro_figure(data: Dataset, names: Sequence[str] | None, kind: str, span: Any, title: str, total: str | None,
                  coverages: Sequence[float] | None) -> Figure:
    """Contributions, fan and vintages charts of a dataset's series."""
    names = list(names or [])
    if not names:
        raise SpecificationError(f"name the series for the {kind} chart")
    missing = [n for n in names + ([total] if total else []) if n not in data.frame]
    if missing:
        raise SpecificationError(f"no series {', '.join(repr(m) for m in missing)} in the data")
    frame = data.frame[data.sample_mask(span)]
    if not isinstance(frame.index, pd.PeriodIndex):
        raise SpecificationError(f"a {kind} chart needs dated (time-series) data")
    if kind == "contributions":
        if total in names:
            raise SpecificationError("the total is not one of its own parts")
        return contributions_figure(frame[names], frame[total] if total else None, title=title, total_label=total or "total")
    if total is not None:
        raise SpecificationError("total is for kind='contributions'")
    if kind == "vintages":
        return vintages_figure(frame[names], title=title)
    panels = [Panel((Fan.from_frame(frame, n, coverages, color=PALETTE[k % len(PALETTE)]),), title=n if len(names) > 1 else "",
                    legend=k == 0, legend_loc="above") for k, n in enumerate(names)]
    return Figure(tuple(panels), title=title, columns=2 if len(panels) > 3 else 1)


def _runs(s: pd.Series) -> tuple[tuple[str, str, str], ...]:
    """The runs of non-zero values of a dated 0/1 series, as (start, end, the series' name) spans."""
    name = str(s.name or "")
    if not isinstance(s.index, pd.PeriodIndex):
        raise SpecificationError(f"shading from {name or 'a series'} needs dated (time-series) data")
    on = s.fillna(0).to_numpy(float) != 0
    spans, start = [], None
    for period, flag in zip(s.index, on):
        if flag and start is None:
            start = period
        if not flag and start is not None:
            spans.append((str(start), str(period - 1), name))
            start = None
    if start is not None:
        spans.append((str(start), str(s.index[-1]), name))
    return tuple(spans)


def shade_text(spans: tuple[tuple[str, str, str], ...]) -> str:
    """Spans written as text, the form ``shade_spans`` reads: ``"2008Q3:2009Q2=GFC, 2020Q1"``."""
    return ", ".join(start[1:] if start.startswith("@") else (start if start == end else f"{start}:{end}") + (f"={label}" if label else "")
                     for start, end, label in spans)


def _dataset_shade(data: Dataset, shade: Any) -> tuple[tuple[str, str, str], ...]:
    """Spans from a 0/1 series of ``data`` named by ``shade`` (each run of non-zero values), or ``shade`` itself
    (periods, or nber / cepr for the committees' recessions); a series of that name comes first."""
    if isinstance(shade, str) and shade.strip() in data.names:
        return _runs(data[shade.strip()].rename(shade.strip()))
    return shade_spans(shade)


def _series_figure(data: Dataset, names: Sequence[str] | None, kind: str, layout: str, span: Any, entity: Any, title: str,
                   stacked: bool, right: Sequence[str], transform: str) -> Figure:
    if kind not in KINDS:
        raise SpecificationError(f"kind must be one of {', '.join(KINDS)}")
    if transform not in TRANSFORMS:
        raise SpecificationError(f"transform must be one of {', '.join(TRANSFORMS)}")
    if stacked and kind not in STACKABLE:
        raise SpecificationError(f"only bars and areas stack, not {kind}")
    if stacked and layout == "each":
        raise SpecificationError("stacking needs the series in one panel (layout='one')")
    names = list(names) if names else [c for c in data.frame.select_dtypes("number").columns][:8]
    if not names:
        raise SpecificationError("no numeric series to plot")
    right = [right] if isinstance(right, str) else list(right)
    if right:
        if kind not in ("line", "step", "bar", "area") or layout == "each" or stacked:
            raise SpecificationError("a right-hand axis is for lines, steps, bars or areas in one panel, unstacked")
        if set(right) - set(names) or set(right) == set(names):
            raise SpecificationError("right must name some, not all, of the series plotted")
    shown = _columns(data, data.frame[data.sample_mask(span)], names, kind, entity)
    if transform != "none":  # changes need the periods before the first one shown
        whole = _columns(data, data.frame, names, kind, entity)
        shown = {n: _transform(whole[n], transform).reindex(s.index) for n, s in shown.items()}
        if transform == "index":
            shown = {n: _index_to_100(s) for n, s in shown.items()}
    columns = shown
    axis_label = TRANSFORMS[transform]
    if kind == "scatter":
        if len(names) < 2:
            raise SpecificationError("a scatter plot needs two series: x first, then y")
        x = columns[names[0]]
        panels = [Panel((Scatter(x, columns[n], fit=True),), title=f"{n} against {names[0]}") for n in names[1:]]
        return Figure(tuple(panels), title=title, columns=min(2, len(panels)))
    if kind == "histogram":
        panels = [Panel((Histogram(s, label=n),), title=n) for n, s in columns.items()]
        return Figure(tuple(panels), title=title, columns=min(2, len(panels)))
    if kind in ("box", "violin"):
        shape = Box if kind == "box" else Violin
        if layout == "each":
            panels = [Panel((shape((s,), (n,)),), title=n, ylabel=axis_label, legend=False) for n, s in columns.items()]
            return Figure(tuple(panels), title=title, columns=2 if len(panels) > 3 else 1)
        return Figure((Panel((shape(tuple(columns.values()), tuple(columns)),), ylabel=axis_label, legend=False),), title=title)

    def layer(name: str, s: pd.Series, color: str | None = None) -> Any:
        side = "right" if name in right else "left"
        label = f"{name} (right)" if side == "right" else name
        if kind in ("bar", "barh"):
            return Bars(s, label=label, color=color, stacked=stacked, horizontal=kind == "barh", axis=side)
        if kind == "area":
            return Band(pd.Series(0.0, index=s.index), s, label=label, color=color or PALETTE[0], alpha=0.35, axis=side)
        return Line(s, label=label, color=color, step=kind == "step", axis=side)

    def zero(series: Sequence[pd.Series]) -> tuple[Any, ...]:
        """A zero line where bars, areas or changes cross zero."""
        crosses = any((s.dropna() < 0).any() for s in series)
        wanted = kind in STACKABLE or transform in ("change", "growth", "yoy")
        return (Rule(0.0, vertical=kind == "barh"),) if crosses and wanted else ()

    if layout == "each":
        panels = [Panel((*zero([s]), layer(n, s, PALETTE[i % len(PALETTE)])), title=n, legend=False, ylabel=axis_label)
                  for i, (n, s) in enumerate(columns.items())]
        return Figure(tuple(panels), title=title, columns=2 if len(panels) > 3 else 1)
    if kind == "area" and stacked:
        layers, below, above = [], None, None
        for i, (n, s) in enumerate(columns.items()):
            v = s.fillna(0.0)
            above = pd.Series(0.0, index=v.index) if above is None else above.reindex(v.index, fill_value=0.0)
            below = pd.Series(0.0, index=v.index) if below is None else below.reindex(v.index, fill_value=0.0)
            base = above.where(v >= 0, below)
            layers.append(Band(base, base + v, label=n, color=PALETTE[i % len(PALETTE)], alpha=0.75))
            above, below = above + v.clip(lower=0), below + v.clip(upper=0)
        return Figure((Panel((*zero(list(columns.values())), *layers), legend=len(columns) > 1, ylabel=axis_label),), title=title)
    left_names = [n for n in columns if n not in right]
    panel = Panel((*zero(list(columns.values())), *(layer(n, s) for n, s in columns.items())), legend=len(columns) > 1,
                  ylabel=axis_label if not right else ", ".join(left_names) + (f" ({axis_label})" if axis_label else ""),
                  ylabel_right=", ".join(right) + (f" ({axis_label})" if axis_label and right else ""))
    return Figure((panel,), title=title)


def contribution_layers(parts: Mapping[str, pd.Series] | pd.DataFrame, total: pd.Series | None = None,
                        total_label: str = "total", other_label: str = "other") -> tuple[Any, ...]:
    """Layers of a contributions chart: each part as stacked bars (positive above zero, negative below), the total as
    a dark line, and a zero line. Without ``total``, the parts' sum is the total. With one, any part of it the parts
    leave unexplained is drawn as ``other_label``, so the bars always add up to the line."""
    frame = pd.DataFrame(dict(parts)) if not isinstance(parts, pd.DataFrame) else parts
    if frame.empty:
        raise SpecificationError("a contributions chart needs at least one part")
    total = frame.sum(axis=1, min_count=1) if total is None else total.reindex(frame.index)
    colors = PALETTE[:-1] if len(PALETTE) > 1 else PALETTE
    layers: list[Any] = [Bars(frame[c], str(c), color=colors[k % len(colors)], stacked=True) for k, c in enumerate(frame.columns)]
    other = total - frame.sum(axis=1, min_count=1)
    scale = max(1.0, float(np.nanmax(np.abs(total.to_numpy(float))))) if total.notna().any() else 1.0
    if other.notna().any() and float(np.nanmax(np.abs(other.to_numpy(float)))) > 1e-9 * scale:
        layers.append(Bars(other, other_label, color="#BBBBBB", stacked=True))
    return (*layers, Line(total, total_label, color="#222222"), Rule(0.0))


def contributions_figure(parts: Mapping[str, pd.Series] | pd.DataFrame, total: pd.Series | None = None, title: str = "",
                         total_label: str = "total", ylabel: str = "") -> Figure:
    """A contributions chart: what each part added to a total, period by period, as stacked bars with the total as a
    line (``contribution_layers``). For growth decompositions, shock decompositions and the like."""
    return Figure((Panel(contribution_layers(parts, total, total_label), ylabel=ylabel),), title=title)


@api("pymetrix:coefficient_figure")
def coefficient_figure(results: Any, terms: Sequence[str] | None = None, level: float = 0.95, title: str = "") -> Figure:
    """A coefficient plot: each estimate with its interval at ``level``, from one result or several (a mapping of names
    to results, or a sequence), side by side for each coefficient. Any result whose ``params`` give ``interval``
    works (equations, and draws for Bayesian ones). ``terms`` picks and orders the coefficients."""
    if not 0 < level < 1:
        raise SpecificationError(f"level is between 0 and 1, e.g. 0.95; got {level}")
    if isinstance(results, Mapping):
        named = list(results.items())
    elif isinstance(results, (list, tuple)):
        named = [(getattr(r, "name", "") or f"model {k + 1}", r) for k, r in enumerate(results)]
    else:
        named = [("", results)]
    layers = []
    for k, (name, result) in enumerate(named):
        params = getattr(result, "params", None)
        if params is None or not hasattr(params, "interval"):
            raise SpecificationError(f"{name or type(result).__name__} has no estimates with intervals")
        estimate = pd.Series(np.asarray(params.values, float), index=list(params.names))
        bounds = params.interval(1 - level)
        if terms:
            missing = [t for t in terms if t not in estimate.index]
            if len(missing) == len(terms):
                raise SpecificationError(f"none of {', '.join(terms)} among {name or 'the'} coefficients: {', '.join(estimate.index)}")
            keep = [t for t in terms if t in estimate.index]
            estimate, bounds = estimate[keep], bounds.loc[keep]
        layers.append(Estimates(estimate, bounds["lower"], bounds["upper"], label=str(name), color=PALETTE[k % len(PALETTE)]))
    pct = f"{100 * level:g}%"
    return Figure((Panel(tuple(layers), xlabel=f"estimate, with its {pct} interval", legend=len(layers) > 1),), title=title)


@api("pymetrix:heatmap_figure")
def heatmap_figure(values: Any, title: str = "", label: str = "", view: str | None = None,
                   scale: Literal["table", "column", "auto"] = "table", diagnostics: bool = True, **options: Any) -> Figure:
    """A table as a heatmap (``Heatmap``): a DataFrame, a ``Table`` (its numeric columns), or a result's table named
    by ``view`` (``heatmap_figure(granger, view="Results")``). ``label`` names the colour bar; ``scale`` "table"
    (one scale), "column" (each column its own) or "auto" (by the columns' sizes, as ``Heatmap`` says);
    ``diagnostics=False`` leaves out a posterior table's sampling diagnostics (its R-hat, effective draws, ESS and
    MCSE columns), which describe the sampler rather than the posterior; ``cmap``, ``center``, ``vmin``, ``vmax``,
    ``annotate`` and ``fmt`` as the layer takes them."""
    from .output import Table

    if view is not None:
        from .views import view as view_of

        values = view_of(values, view)
    if isinstance(values, Table):
        title = title or values.title
        flat = values.to_frame()
        frame = flat.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        frame = frame.loc[:, frame.notna().any().to_numpy()]  # by position: a table may repeat a name ("(se)")
    else:
        frame = pd.DataFrame(values)
    if not diagnostics:
        keep = [c for c in frame.columns if not _is_sampling_diagnostic(c)]
        frame = frame.loc[:, keep] if keep else frame  # a table of diagnostics alone keeps them
    if frame.empty:
        raise SpecificationError("a heatmap needs a table of numbers")
    height = max(2.6, min(9.0, 0.32 * len(frame) + 1.2))
    width = max(4.5, min(7.5, 0.75 * len(frame.columns) + 2.2))  # cells near square, room for the colour bar
    return Figure((Panel((Heatmap(frame.astype(float), label=label, scale=scale, **options),), legend=False),), title=title,
                  height=height, width=width)


_SAMPLING = {"r-hat", "rhat", "eff. draws", "ess", "mcse", "effective sample size", "effective draws"}


def _is_sampling_diagnostic(column: Any) -> bool:
    """A posterior table's column that describes the sampler (R-hat, effective draws, ESS, MCSE), by its name."""
    name = str(column).strip().lower()
    return name in _SAMPLING or name.startswith(("mcse ", "mcse_", "mcse(", "ess ", "ess_", "ess(", "r-hat "))


@api("pymetrix:surface_figure")
def surface_figure(values: Any, kind: Literal["surface", "wireframe", "contourf", "contour"] = "surface", title: str = "", label: str = "",
                   xlabel: str = "", ylabel: str = "", **options: Any) -> Figure:
    """A table over a grid, its columns along x and its rows along y (numbers, dates or names), each value a height:
    ``kind`` surface or wireframe (3-D, ``Surface``), contourf (filled contours) or contour (lines; ``Contour``).
    ``label`` names the heights (the colour bar, and a surface's z axis); ``xlabel`` and ``ylabel`` the axes (the
    table's column and row names by default). Other keywords as the layer takes them: ``cmap``, ``alpha``, ``color``
    for surfaces; ``levels``, ``clabel``, ``color``, ``colorbar`` for contours. A yield curve over time is a table of
    dates by maturities; responses over time, of dates by horizons."""
    if kind not in ("surface", "wireframe", "contourf", "contour"):
        raise SpecificationError(f"kind is surface, wireframe, contourf or contour, not {kind!r}")
    frame = pd.DataFrame(values)
    if frame.empty or frame.shape[0] < 2 or frame.shape[1] < 2:
        raise SpecificationError("a surface or contours need a table of numbers at least two by two")
    frame = frame.astype(float)
    if kind in ("surface", "wireframe"):
        layer: Any = Surface(frame, label=label, wireframe=kind == "wireframe", **options)
        panel = Panel((layer,), xlabel=xlabel, ylabel=ylabel, zlabel=label, legend=False)
    else:
        panel = Panel((Contour(frame, filled=kind == "contourf", label=label, **options),), xlabel=xlabel, ylabel=ylabel, legend=False)
    return Figure((panel,), title=title)


def vintages_figure(vintages: Mapping[str, pd.Series] | pd.DataFrame, title: str = "", ylabel: str = "") -> Figure:
    """One line per release of a series, oldest first: the older ones lighter and thinner, the latest darkest, to
    show how the data were revised."""
    frame = pd.DataFrame(dict(vintages)) if not isinstance(vintages, pd.DataFrame) else vintages
    if frame.empty:
        raise SpecificationError("a vintages chart needs at least one release")
    return Figure((Panel(_vintage_lines(frame), ylabel=ylabel),), title=title)


def _vintage_lines(frame: pd.DataFrame) -> tuple[Any, ...]:
    from matplotlib.colors import to_hex, to_rgb

    base, n = np.array(to_rgb(PALETTE[1] if len(PALETTE) > 1 else PALETTE[0])), len(frame.columns)
    lines = []
    for k, c in enumerate(frame.columns):
        weight = 0.25 + 0.75 * (k + 1) / n  # the share of the colour; the rest is white
        color = to_hex(np.clip(weight * base * (0.75 if k == n - 1 else 1.0) + (1 - weight), 0, 1))
        lines.append(Line(frame[c], str(c), color=color, width=2.0 if k == n - 1 else 1.2))
    return tuple(lines)


def fit_figure(fit: Any) -> Figure:
    """Actual, fitted and residuals (with ±2 standard errors of the regression)."""
    if fit.resid is None:
        raise SpecificationError("this fit has no residuals to plot")
    resid, fitted = fit.resid, fit.fitted
    actual = fitted + resid
    if isinstance(resid.index, pd.MultiIndex) or not isinstance(resid.index, pd.PeriodIndex):
        panel = Panel((Scatter(fitted.rename("fitted"), actual.rename("actual"), fit=True),), title="Actual against fitted")
        return Figure((panel, Panel((Histogram(resid.rename("residuals"), label="residuals"),), title="Residuals")),
                      title=fit.equation.text, columns=2)
    se = fit.stats.get("se_regression") or float(resid.std())
    top = Panel((Line(actual, "actual"), Line(fitted, "fitted")), title="Actual and fitted")
    bottom = Panel((Band(pd.Series(-2 * se, index=resid.index), pd.Series(2 * se, index=resid.index), "±2 s.e.", color="#888888", alpha=0.15),
                    Rule(0.0), Line(resid, "residuals", color=PALETTE[2], width=1.2)), title="Residuals")
    return Figure((top, bottom), title=fit.equation.text)


def forecast_figure(forecast: Dataset, data: Dataset | None = None, history: int = 16) -> Figure:
    """Forecasts with their intervals, after the last ``history`` periods of actual data when ``data`` is given."""
    frame = forecast.frame
    panels = []
    if "forecast" in frame.columns:  # a single equation's forecast
        variables = [("forecast", "lower", "upper", "actual", forecast.meta.title or "Forecast")]
    else:
        variables = [(c, f"{c}_lower", f"{c}_upper", None, c) for c in frame.columns if f"{c}_lower" in frame.columns]
    if not variables:
        raise SpecificationError("no forecast columns with intervals in this dataset")
    for mean, lo, hi, actual, title in variables:
        centre, lower, upper = frame[mean], frame[lo], frame[hi]
        past = None
        if data is not None and mean != "forecast" and mean in data.names:
            past = data[mean]
            past = past[past.index < frame.index[0]].dropna().iloc[-history:]
        if past is not None and len(past):  # the forecast and its interval start where the history ends
            joint = pd.Series([float(past.iloc[-1])], index=past.index[-1:])
            centre, lower, upper = (pd.concat([joint, x]) for x in (centre, lower, upper))
        layers: list[Any] = [Band(lower, upper, "interval")]
        if past is not None and len(past):
            layers.append(Line(past, "actual", color="#444444"))
        if actual and actual in frame.columns and frame[actual].notna().any():
            layers.append(Line(frame[actual], "actual", color="#444444"))
        layers.append(Line(centre, "forecast", color=PALETTE[0]))
        # one legend, above the first panel, clear of the data and the dates: it is the same in every panel
        panels.append(Panel(tuple(layers), title=title, legend=not panels, legend_loc="above"))
    return Figure(tuple(panels), columns=2 if len(panels) > 3 else 1)


def responses_figure(irf: Any, responses: Sequence[str] | None = None, shocks: Sequence[str] | None = None) -> Figure:
    """Impulse responses: a grid of responses (rows) by shocks (columns), with bands when available."""
    responses = list(responses or irf.responses)
    shocks = list(shocks or irf.shocks)
    horizon = pd.RangeIndex(len(irf.values), name="horizon")
    panels = []
    for r in responses:
        i = list(irf.responses).index(r)
        for s in shocks:
            k = list(irf.shocks).index(s)
            layers: list[Any] = [Rule(0.0)]
            if irf.lower is not None:
                layers.append(Band(pd.Series(irf.lower[:, i, k], index=horizon), pd.Series(irf.upper[:, i, k], index=horizon),
                                   f"{irf.coverage:.0%} band"))
            layers.append(Line(pd.Series(irf.values[:, i, k], index=horizon), "response"))
            bottom = r == responses[-1]
            panels.append(Panel(tuple(layers), title=f"{r}, shock to {s}", legend=False, xlabel="horizon" if bottom else ""))
    kind = "Cumulative responses" if irf.cumulative else "Impulse responses"
    return Figure(tuple(panels), title=f"{kind}: {irf.identification}", columns=len(shocks), width=max(7.5, 2.6 * len(shocks)))


def local_projection_figure(lp: Any) -> Figure:
    irf = lp.irf
    horizon = pd.Index(irf.index, name="horizon")
    regimes = [""] if "response" in irf.columns else ["state ", "other "]
    panels = []
    for regime in regimes:
        layers = [Rule(0.0),
                  Band(pd.Series(irf[f"{regime}lower"].to_numpy(), index=horizon), pd.Series(irf[f"{regime}upper"].to_numpy(), index=horizon),
                       f"{1 - lp.alpha:.0%}", alpha=0.14),
                  Band(pd.Series(irf[f"{regime}lower68"].to_numpy(), index=horizon), pd.Series(irf[f"{regime}upper68"].to_numpy(), index=horizon),
                       "68%", alpha=0.28),
                  Line(pd.Series(irf[f"{regime}response"].to_numpy(), index=horizon), "response")]
        title = {"": "", "state ": "In the state", "other ": "Otherwise"}[regime]
        panels.append(Panel(tuple(layers), title=title, xlabel="horizon"))
    return Figure(tuple(panels), title=f"Response of {lp.spec.response} to {lp.spec.shock}", columns=len(panels))


def states_figure(fit: Any, which: Literal["smoothed", "filtered"] = "smoothed") -> Figure:
    """Estimated states with 95% bands (smoothed: two-sided; filtered: real-time)."""
    frame = getattr(fit, which).frame
    panels = []
    for s in fit.model.states:
        est, se = frame[s], frame[f"{s}_se"]
        layers: list[Any] = [Band(est - 1.96 * se, est + 1.96 * se, "95%"), Line(est, s)]
        if s in fit.model.stationary:
            layers.insert(0, Rule(0.0))
        panels.append(Panel(tuple(layers), title=f"{s} ({which})", legend=False))
    return Figure(tuple(panels), columns=2 if len(panels) > 3 else 1)


def filter_figure(data: Dataset, name: str) -> Figure:
    """A series with its trend, and the cycle."""
    top = Panel((Line(data[name], name, color="#444444", width=1.2), Line(data[f"{name}_trend"], "trend")), title=f"{name} and its trend")
    bottom = Panel((Rule(0.0), Line(data[f"{name}_cycle"], "cycle", color=PALETTE[1])), title="Cycle", legend=False)
    return Figure((top, bottom))


def correlogram_figure(table: Any) -> Figure:
    """Autocorrelations and partial autocorrelations with the approximate 95% band."""
    frame = table.to_frame()
    lag = pd.Index(frame["lag"].astype(int), name="lag")
    import re

    note = " ".join(getattr(table, "notes", []) or [])
    m = re.search(r"±([0-9.]+)", note)
    band = float(m.group(1)) if m else None
    panels = []
    for col in ("AC", "PAC"):
        layers: list[Any] = [Rule(0.0), Bars(pd.Series(frame[col].to_numpy(float), index=lag), col)]
        if band:
            layers += [Rule(band, "--"), Rule(-band, "--")]
        panels.append(Panel(tuple(layers), title=col, legend=False))
    return Figure(tuple(panels), title=table.title, columns=2)


def roots_figure(eigenvalues: Any, title: str = "Roots of the companion matrix", note: str = "") -> Figure:
    """The companion matrix's eigenvalues in the complex plane with the unit circle: a stable VAR's lie inside it;
    a VECM's unit roots sit on it."""
    z = np.asarray(eigenvalues, dtype=complex)
    angle = np.linspace(0.0, np.pi, 181)  # the circle as two halves, each a function of the real part
    x = pd.Index(np.cos(angle)[::-1], name="real")
    halves = [Line(pd.Series(sign * np.sin(angle)[::-1], index=x), "unit circle", color="#4c78a8", style="--", width=1.0) for sign in (1, -1)]
    points = Scatter(pd.Series(z.real, name="real"), pd.Series(z.imag, name="imaginary"), "roots")
    edge = max(1.1, 1.1 * float(np.abs(z).max(initial=0.0)))
    fig = Figure((Panel((Rule(0.0), Rule(0.0, vertical=True), *halves, points), title=title, xlabel="real", ylabel="imaginary", legend=False),),
                 width=5.0, height=5.0, footnote=note)
    return fig.set(0, aspect="equal", xlim=(-edge, edge), ylim=(-edge, edge))


def coefficient_paths_figure(rolling: Any, names: Sequence[str] | None = None) -> Figure:
    params, se = rolling.params.frame, rolling.se.frame
    names = list(names) if names else list(params.columns)
    panels = [Panel((Band(params[n] - 2 * se[n], params[n] + 2 * se[n], "±2 s.e."), Line(params[n], n)), title=n, legend=False) for n in names]
    return Figure(tuple(panels), title=f"{rolling._kind()} estimates: {rolling.equation.text}", columns=2 if len(panels) > 3 else 1)
