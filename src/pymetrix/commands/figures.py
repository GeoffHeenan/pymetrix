"""The Plot series dialog: a figure of series, kept in the workspace like any result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def _figure_chain(source: str, tab: str, calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]], output: str,
                  recorded_by: str) -> Any:
    """``figure_of(source, tab).method(...).method(...)`` as one call, each ``Figure`` method nested on the one before:
    what the figure dialogs record. ``calls`` are (method, positional arguments, keyword arguments)."""
    import inspect

    from ..calls import make_call
    from ..codec import Ref
    from ..figures import Figure

    call = make_call("pymetrix:figure_of", obj=Ref(source), tab=tab)
    for i, (method, args, kwargs) in enumerate(calls):
        fn = getattr(Figure, method)
        signature = inspect.signature(fn)
        named = dict(signature.bind(None, *args, **kwargs).arguments)
        named.pop("self")
        for p in signature.parameters.values():
            if p.kind is p.VAR_KEYWORD:
                named.update(named.pop(p.name, {}))  # matplotlib's keyword arguments, as given
        last = i == len(calls) - 1
        call = make_call(fn.__api__.id, call, output=output if last else "", recorded_by=recorded_by if last else "", **named)
    if not calls:
        call = call.with_output(output)
    return call


@command(menu="Data/Plot series...")
@dataclass(frozen=True)
class PlotSeries(Command):
    """Plot series as lines, steps, bars (side by side or stacked, upright or across), areas, a scatter plot,
    histograms or box plots, in one panel or one each; or as a macro chart: contributions (the series as stacked
    parts of a total, drawn as a line), a fan (each series with its bands, from its _lo90/_hi90 or _se columns) or
    vintages (each series a release of the same data, oldest first); or the series' correlations as a heatmap. For
    distributions: histograms (with a number of bins), empirical distributions (ecdf, every series in one panel),
    two-dimensional histograms (hist2d: tick the x series first, then y), violin plots (each series' density, drawn as
    box plots are) and hexagonal bins (hexbin, for many observations: x first, then y). A surface draws the series
    side by side along x and the dates along y, each value a height, in 3-D: a yield curve over time, with its
    maturities ticked in order.

    Stacked bars and areas put negative values below zero, as contributions to growth do. Series named under
    Right axis are drawn against a second scale on the right. A transform plots each series as an index (first
    period shown = 100), its change, its growth on the previous period or on a year earlier, or its log. Shade
    marks periods such as recessions: give them (2008Q3:2009Q2, 2020Q1:2020Q2) or a 0/1 series. Events mark dates
    and reference lines mark values (an inflation target), each with an optional label after =. Values can be
    written on bars, panels can share one scale, and Advanced settings have a log scale and fixed limits. For a
    scatter plot, tick the x series first. For a panel dataset, name an entity, or plot one series to get a line
    per entity; a cross-section's entities label its bars. The figure opens in its own window, where it can be
    copied as an image, saved as PNG, SVG or PDF, or its data copied. Results have figures of their own in their
    windows' Graph tabs.
    """

    topics: ClassVar[tuple[str, ...]] = ()

    series: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the series to plot (x first for a scatter plot)")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    kind: Annotated[Literal["line", "step", "bar", "barh", "area", "scatter", "histogram", "box", "contributions", "fan", "vintages",
                            "correlation", "ecdf", "hist2d", "violin", "hexbin", "surface"],
                    Widget("choice", help="chart type: line for time series, bar for contributions, scatter for relations, histogram for distributions")] = "line"
    total: Annotated[str, Widget("series", depends_on="data", show_if=("kind", ("contributions",)),
                                 help="the series the parts add up to (empty: their sum)")] = ""
    bins: Annotated[str, Widget("text", show_if=("kind", ("histogram", "hist2d", "hexbin")),
                                help="the number of bins, or hexagons across (empty: 30)")] = ""
    coverages: Annotated[str, Widget("text", show_if=("kind", ("fan",)),
                                     help="the bands' coverages in %, e.g. 30, 60, 90 (empty: those in the data)")] = ""
    layout: Annotated[Literal["one", "each"], Widget("choice", help="all series in one panel, or one panel each")] = "one"
    stacked: Annotated[bool, Widget("checkbox", show_if=("kind", ("bar", "barh", "area")),
                                    help="stack the series (negative values below zero), e.g. contributions")] = False
    transform: Annotated[Literal["none", "index", "change", "growth", "yoy", "log"],
                         Widget("choice", help="plot each series as it is, or transformed")] = "none"
    shade: Annotated[str, Widget("text", advanced=True, help="periods to shade, e.g. 2008Q3:2009Q2, 2020Q1:2020Q2 (a label after =); "
                                               "nber or cepr for US or euro-area recessions; or a 0/1 series such as recession")] = ""
    events: Annotated[str, Widget("text", advanced=True, help="dates to mark with a labelled line, e.g. 2016Q2=Brexit vote, 2020Q1=COVID")] = ""
    reference: Annotated[str, Widget("text", advanced=True, help="values to mark with a labelled line, e.g. 2=inflation target")] = ""
    bar_labels: Annotated[bool, Widget("checkbox", advanced=True, show_if=("kind", ("bar", "barh")), help="write each bar's value on it")] = False
    line_labels: Annotated[bool, Widget("checkbox", advanced=True, show_if=("kind", ("line", "step")),
                                        help="name each line at its end, in its colour, instead of in a legend")] = False
    same_scale: Annotated[bool, Widget("checkbox", show_if=("layout", ("each",)), help="one value scale for every panel")] = False
    log_scale: Annotated[bool, Widget("checkbox", advanced=True, help="a log scale for the values (positive series only)")] = False
    lowest: Annotated[str, Widget("text", advanced=True, help="the lowest value shown (empty: automatic)")] = ""
    highest: Annotated[str, Widget("text", advanced=True, help="the highest value shown (empty: automatic)")] = ""
    split_values: Annotated[str, Widget("text", advanced=True, help="values to leave out, splitting the value axis in two, e.g. "
                                                                    "6:12 for a spike that squashes the rest (empty: none)")] = ""
    split_dates: Annotated[str, Widget("text", advanced=True, help="periods to leave out, splitting the time axis in two, e.g. "
                                                                   "1986Q1:2004Q4 (empty: none)")] = ""
    right: Annotated[str, Widget("series", depends_on="data", multiple=True, advanced=True, show_if=("kind", ("line", "step", "bar", "area")),
                                 help="series to draw against a second scale on the right")] = ""
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    title: Annotated[str, Widget("text", help="a title for the figure (optional)")] = ""
    subtitle: Annotated[str, Widget("text", advanced=True, help="a line under the title (optional)")] = ""
    footnote: Annotated[str, Widget("text", advanced=True, help="a note under the figure, e.g. the source (optional)")] = ""
    name: str = "figure"
    entity: Annotated[str, Widget("text", advanced=True, help="for a panel dataset: the entity to plot")] = ""

    def _span(self) -> tuple[str | None, str | None] | None:
        return (self.start or None, self.end or None) if self.start or self.end else None

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self.kind != "line":
            options["kind"] = self.kind
        if self.kind in ("histogram", "hist2d", "hexbin") and self.bins.strip():
            options["bins"] = int(self.bins)
        if self.layout != "one" and self.kind not in ("contributions", "fan", "vintages", "correlation", "ecdf", "hist2d", "hexbin", "surface"):
            options["layout"] = self.layout
        if self.kind == "contributions" and self.total.strip():
            options["total"] = self.total.strip()
        if self.kind == "fan" and self._coverages():
            options["coverages"] = self._coverages()
        if self.stacked:
            options["stacked"] = True
        if _names(self.right):
            options["right"] = _names(self.right)
        if self.transform != "none":
            options["transform"] = self.transform
        if self.shade.strip():
            options["shade"] = self.shade.strip()
        if self.events.strip():
            options["events"] = self.events.strip()
        if self.reference.strip():
            options["reference"] = self.reference.strip()
        if self.bar_labels:
            options["bar_labels"] = True
        if self.line_labels:
            options["line_labels"] = True
        if self.same_scale:
            options["sharey"] = True
        if self.log_scale:
            options["log"] = True
        if self.lowest.strip() or self.highest.strip():
            options["ylim"] = tuple(float(v) if v.strip() else None for v in (self.lowest, self.highest))
        if self.split_values.strip():
            options["ybreak"] = self.split_values.strip()
        if self.split_dates.strip():
            options["xbreak"] = self.split_dates.strip()
        if self._span():
            options["span"] = self._span()
        if self.entity:
            options["entity"] = self.entity
        if self.title:
            options["title"] = self.title
        if self.subtitle:
            options["subtitle"] = self.subtitle
        if self.footnote:
            options["footnote"] = self.footnote
        return options

    def _coverages(self) -> tuple[float, ...]:
        values = [float(v) for v in _names(self.coverages.replace("%", ""))]
        return tuple(v / 100 if v > 1 else v for v in values)

    def validate(self, session: Session) -> list[str]:
        if not _names(self.series):
            return ["tick the series to plot"]
        if self.kind == "contributions" and self.total.strip() in _names(self.series):
            return ["the total is not one of its own parts: untick it among the series"]
        if self.kind in ("histogram", "hist2d", "hexbin") and self.bins.strip() and (not self.bins.strip().isdigit() or int(self.bins) < 1):
            return ["bins is a whole number of at least 1, e.g. 20"]
        if self.kind == "hist2d" and len(_names(self.series)) != 2:
            return ["a two-dimensional histogram needs two series: the x series first, then y"]
        if self.kind == "hexbin" and len(_names(self.series)) != 2:
            return ["hexagonal bins need two series: the x series first, then y"]
        if self.kind == "surface" and len(_names(self.series)) < 2:
            return ["a surface needs two or more series, side by side along x (tick them in order)"]
        if self.kind == "fan":
            try:
                if any(not 0 < c < 1 for c in self._coverages()):
                    return ["coverages are percentages between 0 and 100, e.g. 30, 60, 90"]
            except ValueError:
                return ["coverages are numbers, e.g. 30, 60, 90"]
        if self.kind == "scatter" and len(_names(self.series)) < 2:
            return ["a scatter plot needs two series: the x series first, then y"]
        if self.stacked and self.kind not in ("bar", "barh", "area"):
            return ["only bars and areas stack"]
        if self.stacked and self.layout == "each":
            return ["stacking needs the series in one panel"]
        if set(_names(self.right)) - set(_names(self.series)):
            return ["the right-axis series must be among the series plotted"]
        try:
            from ..figures import event_marks, reference_lines, value_limits

            event_marks(self.events), reference_lines(self.reference)
            value_limits((self.lowest, self.highest))
            _check_splits(self.split_values, self.split_dates)
        except ValueError as exc:
            return [str(exc)]
        return [] if self.data in session.objects else [f"no dataset named {self.data!r}"]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        names = _names(self.series)
        return make_call("pymetrix:dataset.plot", Ref(self.data), output=self.name, recorded_by=self.command_id,
                         names=names[0] if len(names) == 1 else names, **self._options())

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu=None)  # opened by a figure's Edit… button
@dataclass(frozen=True)
class FigureOptions(Command):
    """Change how a figure is drawn: its title, subtitle and footnote, shaded periods (such as recessions), labelled
    lines at dates and at values, and its value scale. The data are unchanged.

    It works on any figure: one made by Plot series, or a result's own (an equation's Graph tab, a forecast's fan
    chart, impulse responses). The figure is kept as a new object, so the result itself is untouched; a figure in the
    workspace is changed in place.
    """

    topics: ClassVar[tuple[str, ...]] = ()

    source: Annotated[str, Widget("text", help="the figure, or the result whose figure it is")]
    tab: Annotated[str, Widget("text", advanced=True, help="for a result: the tab its figure is on")] = "Graph"
    title: Annotated[str, Widget("text", help="the title (empty: none)")] = ""
    subtitle: Annotated[str, Widget("text", help="a line under the title")] = ""
    footnote: Annotated[str, Widget("text", help="a note under the figure, e.g. the source")] = ""
    shade: Annotated[str, Widget("text", help="periods to shade, e.g. 2008Q3:2009Q2=GFC, 2020Q1:2020Q2; nber or cepr "
                                              "for US or euro-area recessions; or a 0/1 series of the dataset below")] = ""
    events: Annotated[str, Widget("text", help="dates to mark with a labelled line, e.g. 2016Q2=Brexit vote")] = ""
    reference: Annotated[str, Widget("text", help="values to mark with a labelled line, e.g. 2=inflation target")] = ""
    line_labels: Annotated[bool, Widget("checkbox", help="name each line at its end, in its colour, instead of in a legend")] = False
    log_scale: Annotated[bool, Widget("checkbox", help="a log scale for the values (positive series only)")] = False
    lowest: Annotated[str, Widget("text", help="the lowest value shown (empty: automatic)")] = ""
    highest: Annotated[str, Widget("text", help="the highest value shown (empty: automatic)")] = ""
    data: Annotated[str, Widget("object", source="Dataset", advanced=True,
                                help="the dataset whose 0/1 series Shade may name")] = "data"
    name: Annotated[str, Widget("text", help="save as (empty: <source>_figure)")] = ""
    start: Annotated[str, Widget("period", advanced=True, group="Dates drawn")] = ""
    end: Annotated[str, Widget("period", advanced=True, group="Dates drawn")] = ""
    theme: Annotated[str, Widget("named", source="themes", advanced=True, empty="the session's default",
                                 help="how the figure looks; empty = the session's default")] = ""
    columns: Annotated[int | None, Widget("number", min=1, max=6, advanced=True, help="panels side by side; empty = as drawn")] = None
    sharey: Annotated[bool, Widget("checkbox", advanced=True, help="one value scale for all panels")] = False
    bar_labels: Annotated[bool, Widget("checkbox", advanced=True, help="each bar's value written on it")] = False
    house: Annotated[bool, Widget("checkbox", advanced=True, help="drawn with the house style first, where one is set")] = True
    split_values: Annotated[str, Widget("text", advanced=True, help="values to leave out, splitting the value axis in two, e.g. 6:12 "
                                                     "for a spike that squashes the rest (empty: none)")] = ""
    split_dates: Annotated[str, Widget("text", advanced=True, help="periods to leave out, splitting the time axis in two, e.g. "
                                                    "1986Q1:2004Q4 (empty: none)")] = ""

    @classmethod
    def for_figure(cls, source: str, figure: Any, tab: str = "Graph", is_figure: bool = False) -> FigureOptions:
        """The dialog filled in with ``figure``'s current options, to edit: a workspace figure (``is_figure``) in place,
        a result's as ``<result>_<tab>``."""
        from ..figures import shade_text

        low, high = figure.ylim or (None, None)
        return cls(source, tab, figure.title, figure.subtitle, figure.footnote, shade_text(figure.shade),
                   ", ".join(p + (f"={label}" if label else "") for p, label in figure.events),
                   ", ".join(f"{v:.15g}" + (f"={label}" if label else "") for v, label in figure.reference),  # every digit
                   figure.line_labels, figure.log, "" if low is None else f"{low:.15g}", "" if high is None else f"{high:.15g}",
                   name=source if is_figure else f"{source}_{'_'.join(tab.lower().split())}",
                   start="" if not figure.span or figure.span[0] is None else str(figure.span[0]),
                   end="" if not figure.span or figure.span[1] is None else str(figure.span[1]),
                   theme=figure.theme if isinstance(figure.theme, str) else "", columns=figure.columns, sharey=figure.sharey,
                   bar_labels=figure.bar_labels, house=figure.house,
                   split_values="" if figure.ybreak is None else ":".join(f"{v:.15g}" for v in figure.ybreak),
                   split_dates="" if figure.xbreak is None else ":".join(f"{v:.15g}" if isinstance(v, float) else str(v)
                                                                         for v in figure.xbreak))

    def _layout(self) -> dict[str, Any]:
        """The dates, theme and panel options. The dates, theme and columns are given only when set, so a figure keeps
        its own otherwise; the boxes always, like Line labels, so unticking one is kept too."""
        out: dict[str, Any] = {"span": (self.start or None, self.end or None)} if self.start or self.end else {}
        out.update({k: v for k, v in (("theme", self.theme or None), ("columns", self.columns)) if v is not None})
        return out | {"sharey": self.sharey, "bar_labels": self.bar_labels, "house": self.house,
                      "ybreak": self.split_values.strip() or None, "xbreak": self.split_dates.strip() or None}

    def _target(self) -> str:
        return self.name or f"{self.source}_figure"

    def _ylim(self) -> Any:
        if not (self.lowest.strip() or self.highest.strip()):
            return None
        return tuple(float(v) if v.strip() else None for v in (self.lowest, self.highest))

    def _is_series(self) -> bool:
        """A plain name (recession) is a 0/1 series of the dataset; nber and cepr are the committees' dates."""
        from ..recessions import SOURCES

        text = self.shade.strip()
        return text.isidentifier() and text.lower() not in SOURCES

    def _shade_code(self) -> str:
        text = self.shade.strip()
        return f"{self.data}[{text!r}]" if self._is_series() else repr(text)

    def validate(self, session: Session) -> list[str]:
        from ..figures import event_marks, reference_lines, shade_spans, value_limits
        from ..views import figure_of

        if self.source not in session.objects:
            return [f"no object named {self.source!r}"]
        text = self.shade.strip()
        if self._is_series() and text not in getattr(session.objects.get(self.data), "names", []):
            return [f"no series {text!r} in {self.data} to shade from (or write nber or cepr for recession dates)"]
        try:
            figure_of(session.objects[self.source], self.tab)
            event_marks(self.events), reference_lines(self.reference)
            value_limits((self.lowest, self.highest))
            _check_splits(self.split_values, self.split_dates)
            if not self._is_series():
                shade_spans(text)
        except Exception as exc:  # whatever would fail when drawn: say it in the dialog
            return [str(exc)]
        return []

    def to_call(self) -> Any:
        from ..errors import SpecificationError

        if self._is_series():  # a 0/1 series of the data: not a setting a call can save
            raise SpecificationError("shading from a series is recorded as code")
        options = {"title": self.title, "subtitle": self.subtitle, "footnote": self.footnote, "shade": self.shade.strip(),
                   "events": self.events.strip(), "reference": self.reference.strip(), "line_labels": self.line_labels,
                   "log": self.log_scale, "ylim": self._ylim(), **self._layout()}
        return _figure_chain(self.source, self.tab, [("with_options", (), options)], self._target(), self.command_id)

    def run(self, session: Session) -> Any:
        if not self._is_series():
            return self.to_call().run(session)
        from ..views import figure_of

        shade = session[self.data][self.shade.strip()]
        figure = figure_of(session[self.source], self.tab).with_options(
            title=self.title, subtitle=self.subtitle, footnote=self.footnote, shade=shade, events=self.events.strip(),
            reference=self.reference.strip(), line_labels=self.line_labels, log=self.log_scale, ylim=self._ylim(), **self._layout())
        session.objects[self._target()] = figure
        return figure

    def to_code(self) -> str:
        if not self._is_series():
            return self.to_call().to_code()
        tab = f", {self.tab!r}" if self.tab != "Graph" else ""
        return (f"{self._target()} = figure_of({self.source}{tab}).with_options(title={self.title!r}, subtitle={self.subtitle!r}, "
                f"footnote={self.footnote!r}, shade={self._shade_code()}, events={self.events.strip()!r}, "
                f"reference={self.reference.strip()!r}, line_labels={self.line_labels!r}, log={self.log_scale!r}, ylim={self._ylim()!r}"
                + "".join(f", {k}={v!r}" for k, v in self._layout().items()) + ")")


def _check_splits(values: str, dates: str) -> None:
    """A split axis's values or periods, checked as the figure will check them; one of the two."""
    from ..errors import SpecificationError
    from ..figures import period_break, value_break

    value_break(values.strip() or None), period_break(dates.strip() or None)
    if values.strip() and dates.strip():
        raise SpecificationError("split the values or the dates, not both")


_LINESTYLES = ("", "solid", "dashed", "dashdot", "dotted")
_MARKERS = ("", "none", "o", "s", "^", "D", "x", "+", ".")
_LOCATIONS = ("", "best", "upper left", "upper right", "lower left", "lower right", "upper center", "lower center",
              "center left", "center right", "center", "none")


def _number(text: str) -> Any:
    text = str(text).strip()
    if not text:
        return None
    value = float(text)
    return int(value) if value.is_integer() and "." not in text else value


@command(menu=None)  # opened by a figure's Edit… > Style…
@dataclass(frozen=True)
class StyleFigure(Command):
    """Style a figure in matplotlib's terms: a series' line (colour, line style, width, markers), a panel's axes
    (scale, limits, labels, tick format, grid, frame), the legend and the figure's size.

    Empty fields leave things as they are. Each change is recorded as the matplotlib-style call that makes it, e.g.
    ``fig = figure_of(fit).line('actual', color='C3', linestyle='dashed').set(ax=0, yscale='log')``, so it replays.
    Values are checked by matplotlib as the step runs: a colour or line style it does not know is reported here.
    """

    topics: ClassVar[tuple[str, ...]] = ()
    dialog_title: ClassVar[str] = "Figure style"
    group_tabs: ClassVar[bool] = True  # Series, Axes, Legend and Layout on tabs of their own, as matplotlib arranges them

    source: Annotated[str, Widget("text", help="the figure, or the result whose figure it is")]
    tab: Annotated[str, Widget("text", advanced=True, help="for a result: the tab its figure is on")] = "Graph"
    series: Annotated[str, Widget("layer", depends_on="source", group="Series", help="the series to style")] = ""
    color: Annotated[str, Widget("text", group="Series", help="a colour: a name (navy), #RRGGBB, or C0 to C9 (matplotlib's cycle)")] = ""
    linestyle: Annotated[Literal["", "solid", "dashed", "dashdot", "dotted"], Widget("choice", group="Series", help="the line's style")] = ""
    linewidth: Annotated[str, Widget("text", group="Series", help="the line's width in points")] = ""
    marker: Annotated[Literal["", "none", "o", "s", "^", "D", "x", "+", "."], Widget("choice", group="Series", help="a marker at each point")] = ""
    markersize: Annotated[str, Widget("text", group="Series", help="the markers' size in points")] = ""
    alpha: Annotated[str, Widget("text", group="Series", help="opacity, 0 (invisible) to 1")] = ""
    panel: Annotated[str, Widget("text", group="Axes", help="the panel: 0 for the first, its title, or all")] = "0"
    side: Annotated[Literal["left", "right"], Widget("choice", group="Axes", help="the left-hand axis, or the right-hand one")] = "left"
    yscale: Annotated[Literal["", "linear", "log", "symlog"], Widget("choice", group="Axes", help="the value scale")] = ""
    ymin: Annotated[str, Widget("text", group="Axes", help="the lowest value shown")] = ""
    ymax: Annotated[str, Widget("text", group="Axes", help="the highest value shown")] = ""
    xmin: Annotated[str, Widget("text", group="Axes", help="the first date shown, e.g. 2010Q1")] = ""
    xmax: Annotated[str, Widget("text", group="Axes", help="the last date shown")] = ""
    ylabel: Annotated[str, Widget("text", group="Axes", help="the value axis' label")] = ""
    xlabel: Annotated[str, Widget("text", group="Axes", help="the date axis' label")] = ""
    tick_format: Annotated[str, Widget("text", group="Axes", help="the value labels' format, e.g. {x:,.0f} or {x:.0%}")] = ""
    tick_size: Annotated[str, Widget("text", group="Axes", help="the tick labels' size in points")] = ""
    grid: Annotated[Literal["", "on", "off"], Widget("choice", group="Axes", help="grid lines")] = ""
    hide_spines: Annotated[str, Widget("text", group="Axes", help="sides of the frame to hide, e.g. top, right")] = ""
    legend_loc: Annotated[Literal["", "best", "upper left", "upper right", "lower left", "lower right", "upper center",
                                  "lower center", "center left", "center right", "center", "none"],
                          Widget("choice", group="Legend", help="where the legend goes (none: no legend)")] = ""
    legend_columns: Annotated[str, Widget("text", group="Legend", help="entries side by side")] = ""
    legend_frame: Annotated[Literal["", "on", "off"], Widget("choice", group="Legend", help="a frame around the legend")] = ""
    legend_size: Annotated[str, Widget("text", group="Legend", help="the legend's text size in points")] = ""
    width: Annotated[str, Widget("text", group="Layout", help="the figure's width in inches")] = ""
    height: Annotated[str, Widget("text", group="Layout", help="the figure's height in inches")] = ""
    name: Annotated[str, Widget("text", help="save as (empty: <source>_figure)")] = ""

    @classmethod
    def for_figure(cls, source: str, figure: Any, tab: str = "Graph", is_figure: bool = False) -> StyleFigure:
        """The dialog, opened on ``figure``: a workspace figure is styled in place, a result's kept as ``<result>_<tab>``."""
        return cls(source, tab, name=source if is_figure else f"{source}_{'_'.join(tab.lower().split())}")

    def _target(self) -> str:
        return self.name or f"{self.source}_figure"

    def _calls(self) -> list[tuple[str, tuple[Any, ...], dict[str, Any]]]:
        """The matplotlib-style calls the settings make: (method, positional arguments, keyword arguments)."""
        calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        if self.series.strip():
            label, _, kind = self.series.strip().rpartition(" (")
            kind = kind.rstrip(")") if label else "line"
            label = label or self.series.strip()
            props = {"color": self.color.strip() or None, "linestyle": self.linestyle or None, "linewidth": _number(self.linewidth),
                     "marker": self.marker or None, "markersize": _number(self.markersize), "alpha": _number(self.alpha)}
            if kind != "line":
                props = {k: v for k, v in props.items() if k not in ("marker", "markersize")}
            props = {k: v for k, v in props.items() if v is not None}
            if props:
                calls.append((kind, (label,), props))
        panel: Any = self.panel.strip() or "0"
        panel = int(panel) if panel.isdigit() else panel
        axes: dict[str, Any] = {}
        if self.yscale:
            axes["yscale"] = self.yscale
        if self.ymin.strip() or self.ymax.strip():
            axes["ylim"] = (_number(self.ymin), _number(self.ymax))
        if self.xmin.strip() or self.xmax.strip():
            axes["xlim"] = (self.xmin.strip() or None, self.xmax.strip() or None)
        if self.ylabel.strip():
            axes["ylabel"] = self.ylabel.strip()
        if self.xlabel.strip():
            axes["xlabel"] = self.xlabel.strip()
        if axes:
            calls.append(("set", (), {"ax": panel, **({"side": "right"} if self.side == "right" else {}), **axes}))
        ticks: dict[str, Any] = {}
        if self.tick_format.strip():
            ticks["format"] = self.tick_format.strip()
        if self.tick_size.strip():
            ticks["labelsize"] = _number(self.tick_size)
        if ticks:
            calls.append(("ticks", (), {"ax": panel, "axis": "y" if "format" in ticks else "both", **ticks}))
        if self.grid:
            calls.append(("grid", (), {"ax": panel, "visible": self.grid == "on"}))
        for side in _names(self.hide_spines):
            calls.append(("spines", (), {"ax": panel, "which": side, "visible": False}))
        legend: dict[str, Any] = {}
        if self.legend_loc == "none":
            legend["visible"] = False
        elif self.legend_loc:
            legend["loc"] = self.legend_loc
        if self.legend_columns.strip():
            legend["ncols"] = _number(self.legend_columns)
        if self.legend_frame:
            legend["frameon"] = self.legend_frame == "on"
        if self.legend_size.strip():
            legend["fontsize"] = _number(self.legend_size)
        if legend:
            calls.append(("legend", (), {"ax": panel, **legend}))
        if self.width.strip() and self.height.strip():
            calls.append(("layout", (), {"figsize": (_number(self.width), _number(self.height))}))
        elif self.width.strip() or self.height.strip():  # one side; the other as it is
            calls.append(("layout", (), {"figwidth": _number(self.width)} if self.width.strip() else {"figheight": _number(self.height)}))
        return calls

    def _styled(self, session: Session) -> Any:
        from ..views import figure_of

        figure = figure_of(session[self.source], self.tab)
        for method, args, kwargs in self._calls():
            figure = getattr(figure, method)(*args, **kwargs)
        return figure

    def validate(self, session: Session) -> list[str]:
        if self.source not in session.objects:
            return [f"no object named {self.source!r}"]
        try:
            self._styled(session).draw()  # matplotlib checks every value, and the series are found, before the step runs
        except Exception as exc:
            return [str(exc)]
        return []

    def to_call(self) -> Any:
        return _figure_chain(self.source, self.tab, self._calls(), self._target(), self.command_id)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


def _part(text: str) -> tuple[str, Any, str]:
    """A figure part as the property editor lists it: ("line", None, "gdp") for "gdp (line)"; ("axes", 0, "left") for
    "panel 0: Actual and fitted (axes)"; ("axes", 0, "right") for "panel 0 right (axes)"."""
    import re

    from ..errors import SpecificationError

    text = text.strip()
    m = re.fullmatch(r"panel (\d+)( right)?(?:: .*)? \(axes\)", text)
    if m:
        return "axes", int(m.group(1)), "right" if m.group(2) else "left"
    label, _, kind = text.rpartition(" (")
    if not label or kind.rstrip(")") not in ("line", "bars", "band"):
        raise SpecificationError(f"no figure part {text!r}: pick a series or a panel from the list")
    return kind.rstrip(")"), None, label


@command(menu=None)  # opened by a figure's Edit… > All properties…
@dataclass(frozen=True)
class SetProperties(Command):
    """Set any style property matplotlib has for a part of a figure: a series' line, bars or band, or a panel's axes.

    The properties are matplotlib's own, listed from its description of each artist (ArtistInspector), with its own
    note on their valid values; those holding the plotted data or positions are left out. Empty fields are left as
    they are. The step is recorded as the call that makes it, e.g. ``figure_of(fit).line('actual', drawstyle='steps-post',
    markevery=4)``, and matplotlib checks each value as it runs.
    """

    topics: ClassVar[tuple[str, ...]] = ()
    dialog_title: ClassVar[str] = "Figure properties"

    source: Annotated[str, Widget("text", help="the figure, or the result whose figure it is")]
    tab: Annotated[str, Widget("text", advanced=True, help="for a result: the tab its figure is on")] = "Graph"
    part: Annotated[str, Widget("figurepart", depends_on="source", help="the series or panel whose properties to set")] = ""
    properties: Annotated[str, Widget("options", depends_on="part", source="figure-properties",
                                      help="matplotlib's properties for that part")] = ""
    name: Annotated[str, Widget("text", help="save as (empty: <source>_figure)")] = ""

    @classmethod
    def for_figure(cls, source: str, figure: Any, tab: str = "Graph", is_figure: bool = False) -> SetProperties:
        return cls(source, tab, name=source if is_figure else f"{source}_{'_'.join(tab.lower().split())}")

    def _target(self) -> str:
        return self.name or f"{self.source}_figure"

    def _call(self) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
        from ..forms import parse_options

        kind, panel, which = _part(self.part)
        _, props = parse_options(self.properties)
        if kind == "axes":
            return "set", (), {"ax": panel, **({"side": "right"} if which == "right" else {}), **props}
        return kind, (which,), props

    def validate(self, session: Session) -> list[str]:
        if self.source not in session.objects:
            return [f"no object named {self.source!r}"]
        try:
            self._styled(session).draw()
        except Exception as exc:  # matplotlib's message, a missing series or panel: shown in the dialog
            return [str(exc)]
        return []

    def _styled(self, session: Session) -> Any:
        from ..views import figure_of

        method, args, kwargs = self._call()
        return getattr(figure_of(session[self.source], self.tab), method)(*args, **kwargs)

    def to_call(self) -> Any:
        return _figure_chain(self.source, self.tab, [self._call()], self._target(), self.command_id)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


def _position(text: str) -> Any:
    """A position typed in a form: a decimal number is a number; a whole number (2007: a year on an annual axis, or 5
    on a horizon axis) and a period (2008Q3) stay text, read by the figure as a period on a dated axis and as a number
    on another."""
    text = str(text).strip()
    if text.lstrip("-").isdigit():
        return text
    try:
        return _number(text)
    except ValueError:
        return text


@command(menu=None)  # opened by a figure's Add a note (click where it goes) or Edit… > Add a note…
@dataclass(frozen=True)
class AnnotateFigure(Command):
    """Add something to a figure, as matplotlib's Axes would: a note with an arrow (annotate), an arrow from one point
    to another (with optional words at its tail), text, a text box, a line at a date or a value (axvline, axhline),
    or a band between dates or values (axvspan, axhspan).

    An arrow has a style (->, -|>, <->, fancy, wedge), a straight or curved connection (curvature: matplotlib's rad),
    a colour, a width and a head size. A text box has a box style (round, square, sawtooth, or none), fill and edge
    colours, padding, a font size and an alignment. Both are recorded as annotate or text with matplotlib's
    arrowprops and bbox, so they replay, appear in reports and in the figure's matplotlib code.

    Positions are in the data's terms: dates as periods (2008Q3), values as numbers. In the app, click Add a note on
    the figure, then click where it goes: the panel, date and value are filled in. The step is recorded as the call,
    e.g. ``figure_of(fit).annotate('Lehman', xy=('2008Q3', 3.2), xytext=(15, 20), textcoords='offset points',
    arrowprops={'arrowstyle': '->'})``, so it replays at any size.
    """

    topics: ClassVar[tuple[str, ...]] = ()
    dialog_title: ClassVar[str] = "Add a note"

    source: Annotated[str, Widget("text", help="the figure, or the result whose figure it is")]
    tab: Annotated[str, Widget("text", advanced=True, help="for a result: the tab its figure is on")] = "Graph"
    what: Annotated[Literal["annotate", "arrow", "text", "textbox", "axvline", "axhline", "axvspan", "axhspan"],
                    Widget("choice", help="what to add")] = "annotate"
    text: Annotated[str, Widget("text", show_if=("what", ("annotate", "text", "arrow", "textbox")),
                                help="the words (for an arrow: optional, at its tail)")] = ""
    panel: Annotated[str, Widget("text", help="the panel: 0 for the first, or its title")] = "0"
    x: Annotated[str, Widget("text", show_if=("what", ("annotate", "text", "textbox", "arrow", "axvline", "axvspan")),
                             help="the date (a period such as 2008Q3), or a number on an axis without dates; an arrow's head")] = ""
    y: Annotated[str, Widget("text", show_if=("what", ("annotate", "text", "textbox", "arrow", "axhline", "axhspan")),
                             help="the value; an arrow's head")] = ""
    from_x: Annotated[str, Widget("text", show_if=("what", ("arrow",)), help="the arrow's tail: a date or a number")] = ""
    from_y: Annotated[str, Widget("text", show_if=("what", ("arrow",)), help="the arrow's tail: a value")] = ""
    arrowstyle: Annotated[Literal["->", "-|>", "<->", "fancy", "wedge", "-"],
                          Widget("choice", advanced=True, show_if=("what", ("arrow",)), help="the arrow's style")] = "->"
    connection: Annotated[Literal["straight", "curved"], Widget("choice", advanced=True, show_if=("what", ("arrow",)),
                                                                help="a straight line, or a curve")] = "straight"
    curvature: Annotated[str, Widget("text", advanced=True, show_if=("connection", ("curved",)),
                                     help="how much it bends: matplotlib's rad, e.g. 0.3 (negative bends the other way)")] = "0.3"
    width: Annotated[str, Widget("text", advanced=True, show_if=("what", ("arrow",)), help="the line's width in points (empty: 1)")] = ""
    head: Annotated[str, Widget("text", advanced=True, show_if=("what", ("arrow",)), help="the head's size (matplotlib's mutation_scale; empty: 10)")] = ""
    boxstyle: Annotated[Literal["round", "square", "sawtooth", "none"],
                        Widget("choice", advanced=True, show_if=("what", ("textbox",)), help="the box around the words")] = "round"
    facecolor: Annotated[str, Widget("text", advanced=True, show_if=("what", ("textbox",)), help="the box's fill colour (empty: white)")] = ""
    edgecolor: Annotated[str, Widget("text", advanced=True, show_if=("what", ("textbox",)), help="the box's edge colour (empty: grey)")] = ""
    pad: Annotated[str, Widget("text", advanced=True, show_if=("what", ("textbox",)), help="space inside the box, in font sizes (empty: 0.3)")] = ""
    ha: Annotated[Literal["left", "center", "right"], Widget("choice", advanced=True, show_if=("what", ("textbox",)),
                                                            help="where the box sits across from the point")] = "left"
    va: Annotated[Literal["bottom", "center", "top"], Widget("choice", advanced=True, show_if=("what", ("textbox",)),
                                                            help="where the box sits up and down from the point")] = "bottom"
    x2: Annotated[str, Widget("text", show_if=("what", ("axvspan",)), help="the band's last date")] = ""
    y2: Annotated[str, Widget("text", show_if=("what", ("axhspan",)), help="the band's upper value")] = ""
    text_x: Annotated[str, Widget("text", show_if=("what", ("annotate",)), advanced=True,
                                  help="where the words go: a date (empty: just above and right of the point)")] = ""
    text_y: Annotated[str, Widget("text", show_if=("what", ("annotate",)), advanced=True, help="where the words go: a value")] = ""
    arrow: Annotated[bool, Widget("checkbox", show_if=("what", ("annotate",)), help="an arrow from the words to the point")] = True
    color: Annotated[str, Widget("text", help="a colour: a name, #RRGGBB, or C0 to C9 (empty: matplotlib's default)")] = ""
    fontsize: Annotated[str, Widget("text", advanced=True, show_if=("what", ("annotate", "text", "textbox", "arrow")),
                                    help="the words' size in points")] = ""
    linestyle: Annotated[Literal["", "solid", "dashed", "dashdot", "dotted"],
                         Widget("choice", show_if=("what", ("axvline", "axhline")), help="the line's style")] = ""
    alpha: Annotated[str, Widget("text", show_if=("what", ("axvspan", "axhspan")), help="the band's opacity, 0 to 1")] = ""
    name: Annotated[str, Widget("text", help="save as (empty: <source>_figure)")] = ""

    @classmethod
    def at(cls, source: str, figure: Any, tab: str = "Graph", is_figure: bool = False, panel: int = 0, x: Any = "",
           y: Any = "", what: str = "annotate", start: Any = None) -> AnnotateFigure:
        """The dialog for a note at a clicked point: the panel, date (or number) and value filled in. ``what="arrow"``
        with ``start``, the tail's (x, y): an arrow from the first click to the second."""
        value = "" if y == "" else f"{float(y):.4g}"
        tail = {} if start is None else {"from_x": str(start[0]), "from_y": f"{float(start[1]):.4g}"}
        return cls(source, tab, what=what, panel=str(panel), x=str(x), y=value, **tail,
                   name=source if is_figure else f"{source}_{'_'.join(tab.lower().split())}")

    def _target(self) -> str:
        return self.name or f"{self.source}_figure"

    def _call(self) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
        panel: Any = self.panel.strip() or "0"
        panel = int(panel) if panel.isdigit() else panel
        style: dict[str, Any] = {}
        if self.color.strip():
            style["color"] = self.color.strip()
        if self.what in ("annotate", "text", "textbox", "arrow") and self.fontsize.strip():
            style["fontsize"] = _number(self.fontsize)
        if self.what in ("axvline", "axhline") and self.linestyle:
            style["linestyle"] = self.linestyle
        if self.what in ("axvspan", "axhspan") and self.alpha.strip():
            style["alpha"] = _number(self.alpha)
        x, y = _position(self.x), _number(self.y) if self.y.strip() else None
        if self.what == "annotate":
            kwargs: dict[str, Any] = {"xy": (x, y)}
            if self.text_x.strip() or self.text_y.strip():
                kwargs["xytext"] = (_position(self.text_x) if self.text_x.strip() else x,
                                    _number(self.text_y) if self.text_y.strip() else y)
            else:
                kwargs.update(xytext=(15, 20), textcoords="offset points")
            if self.arrow:
                kwargs["arrowprops"] = {"arrowstyle": "->", **({"color": style["color"]} if "color" in style else {})}
            return "annotate", (self.text,), {**kwargs, "ax": panel, **style}
        if self.what == "arrow":  # annotate with the words (if any) at the tail and the arrow's head at the point
            arrow: dict[str, Any] = {"arrowstyle": self.arrowstyle}
            if self.connection == "curved":
                arrow["connectionstyle"] = f"arc3,rad={_number(self.curvature or '0.3')}"
            if "color" in style:
                arrow["color"] = style["color"]
            if self.width.strip():
                arrow["linewidth"] = _number(self.width)
            if self.head.strip():
                arrow["mutation_scale"] = _number(self.head)
            tail = (_position(self.from_x), _number(self.from_y) if self.from_y.strip() else None)
            return "annotate", (self.text,), {"xy": (x, y), "xytext": tail, "arrowprops": arrow, "ax": panel, **style}
        if self.what == "textbox":
            box: dict[str, Any] | None = None
            if self.boxstyle != "none":
                box = {"boxstyle": f"{self.boxstyle},pad={_number(self.pad) if self.pad.strip() else 0.3}",
                       "facecolor": self.facecolor.strip() or "white", "edgecolor": self.edgecolor.strip() or "#888888"}
            return "text", (x, y, self.text), {"ax": panel, "ha": self.ha, "va": self.va, **({"bbox": box} if box else {}), **style}
        if self.what == "text":
            return "text", (x, y, self.text), {"ax": panel, **style}
        if self.what == "axvline":
            return "axvline", (x,), {"ax": panel, **style}
        if self.what == "axhline":
            return "axhline", (y,), {"ax": panel, **style}
        if self.what == "axvspan":
            return "axvspan", (x, _position(self.x2) if self.x2.strip() else x), {"ax": panel, **style}
        return "axhspan", (y, _number(self.y2) if self.y2.strip() else y), {"ax": panel, **style}

    def validate(self, session: Session) -> list[str]:
        if self.source not in session.objects:
            return [f"no object named {self.source!r}"]
        needs = {"annotate": ("x", "y", "text"), "text": ("x", "y", "text"), "textbox": ("x", "y", "text"),
                 "arrow": ("x", "y", "from_x", "from_y"), "axvline": ("x",), "axhline": ("y",),
                 "axvspan": ("x", "x2"), "axhspan": ("y", "y2")}[self.what]
        missing = [f for f in needs if not str(getattr(self, f)).strip()]
        if "text" in missing:
            return ["give the words"]
        if missing:
            return [f"give the {', '.join(missing)} (click on the figure, or type {'them' if len(missing) > 1 else 'it'})"]
        try:
            self._styled(session).draw()
        except Exception as exc:
            return [str(exc)]
        return []

    def _styled(self, session: Session) -> Any:
        from ..views import figure_of

        method, args, kwargs = self._call()
        return getattr(figure_of(session[self.source], self.tab), method)(*args, **kwargs)

    def to_call(self) -> Any:
        return _figure_chain(self.source, self.tab, [self._call()], self._target(), self.command_id)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu=None)  # opened by a figure's Edit… > Mark a change… (or two clicks on the figure)
@dataclass(frozen=True)
class MarkChange(Command):
    """Mark the change in a series between two points: an arrow (or a bracket) from one to the other, labelled with
    the change, the percent change, or both, and optionally the rate a year. In the app, choose Mark a change on the
    figure and click two points of a series (each snaps to its nearest data point), or give the periods here.

    The change is in percentage points (pp) where the panel's axis is a rate or a percentage, unless units say
    otherwise. The step is recorded as ``figure_of(x).change('u', '2019Q4', '2020Q2', show='both')``: the values are
    read from the data each time the figure is drawn, so the label stays right when the data change."""

    topics: ClassVar[tuple[str, ...]] = ()
    dialog_title: ClassVar[str] = "Mark a change"

    source: Annotated[str, Widget("text", help="the figure, or the result whose figure it is")]
    tab: Annotated[str, Widget("text", advanced=True, help="for a result: the tab its figure is on")] = "Graph"
    series: Annotated[str, Widget("text", help="the series (its name in the legend)")] = ""
    start: Annotated[str, Widget("text", help="the first point: a period such as 2019Q4 (a number on an axis without dates)")] = ""
    end: Annotated[str, Widget("text", help="the second point")] = ""
    show: Annotated[Literal["both", "change", "percent"], Widget("choice", help="what the label says")] = "both"
    units: Annotated[Literal["auto", "pp", "%", "none"],
                     Widget("choice", help="the change's units: auto is percentage points for a rate or a percentage")] = "auto"
    annualise: Annotated[bool, Widget("checkbox", help="add the rate a year (compound, for the percent change)")] = False
    style: Annotated[Literal["arrow", "bracket"], Widget("choice", help="a double-headed arrow or a bracket")] = "arrow"
    color: Annotated[str, Widget("text", help="a colour (empty: dark grey)")] = ""
    panel: Annotated[str, Widget("text", advanced=True, help="the panel: empty for wherever the series is drawn")] = ""
    name: Annotated[str, Widget("text", help="save as (empty: <source>_figure)")] = ""

    @classmethod
    def at(cls, source: str, figure: Any, tab: str = "Graph", is_figure: bool = False, series: str = "", start: Any = "",
           end: Any = "") -> MarkChange:
        """The dialog for two clicked points of ``series``."""
        return cls(source, tab, series=series, start=str(start), end=str(end),
                   name=source if is_figure else f"{source}_{'_'.join(tab.lower().split())}")

    def _target(self) -> str:
        return self.name or f"{self.source}_figure"

    def _call(self) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
        panel: Any = self.panel.strip() or None
        if isinstance(panel, str) and panel.isdigit():
            panel = int(panel)
        kwargs: dict[str, Any] = {"show": self.show, "units": self.units, "annualise": self.annualise, "ax": panel}
        if self.style != "arrow":
            kwargs["style"] = self.style
        if self.color.strip():
            kwargs["color"] = self.color.strip()
        return "change", (self.series.strip(), self.start.strip(), self.end.strip()), kwargs

    def validate(self, session: Session) -> list[str]:
        if self.source not in session.objects:
            return [f"no object named {self.source!r}"]
        missing = [f for f in ("series", "start", "end") if not str(getattr(self, f)).strip()]
        if missing:
            return [f"give the {', '.join(missing)} (click two points of a series on the figure, or type them)"]
        try:
            self._styled(session).draw()
        except Exception as exc:
            return [str(exc)]
        return []

    def _styled(self, session: Session) -> Any:
        from ..views import figure_of

        method, args, kwargs = self._call()
        return getattr(figure_of(session[self.source], self.tab), method)(*args, **kwargs)

    def to_call(self) -> Any:
        return _figure_chain(self.source, self.tab, [self._call()], self._target(), self.command_id)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu=None)  # opened by a figure's Edit… > matplotlib method…
@dataclass(frozen=True)
class CallAxesMethod(Command):
    """Call any matplotlib Axes method that styles or annotates a panel, recorded as data: ``set_aspect``,
    ``ticklabel_format``, ``minorticks_on``, ``axline``, ``hlines``... (methods that plot more data, clear things or need
    Python objects are not offered). Give its arguments as matplotlib takes them, in Python's notation: positional
    (``'equal'``) and keyword (``axis='y', style='sci', scilimits=(0, 0)``). The step is recorded as
    ``figure_of(fit).call('ticklabel_format', axis='y', style='sci')``, and matplotlib checks the call as it runs.
    """

    topics: ClassVar[tuple[str, ...]] = ()
    dialog_title: ClassVar[str] = "matplotlib Axes method"

    source: Annotated[str, Widget("text", help="the figure, or the result whose figure it is")]
    tab: Annotated[str, Widget("text", advanced=True, help="for a result: the tab its figure is on")] = "Graph"
    method: Annotated[str, Widget("axesmethod", help="the Axes method")] = "minorticks_on"
    arguments: Annotated[str, Widget("text", help="positional arguments, e.g. 'equal' (empty: none)")] = ""
    keywords: Annotated[str, Widget("text", help="keyword arguments, e.g. axis='y', style='sci' (empty: none)")] = ""
    panel: Annotated[str, Widget("text", help="the panel: 0 for the first, its title, or all")] = "0"
    side: Annotated[Literal["left", "right"], Widget("choice", help="the left-hand axes, or the right-hand one")] = "left"
    name: Annotated[str, Widget("text", help="save as (empty: <source>_figure)")] = ""

    @classmethod
    def for_figure(cls, source: str, figure: Any, tab: str = "Graph", is_figure: bool = False) -> CallAxesMethod:
        return cls(source, tab, name=source if is_figure else f"{source}_{'_'.join(tab.lower().split())}")

    def _target(self) -> str:
        return self.name or f"{self.source}_figure"

    def _call(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        from ..forms import parse_options

        args, _ = parse_options(self.arguments)
        _, kwargs = parse_options(self.keywords)
        panel: Any = self.panel.strip() or "0"
        panel = int(panel) if panel.isdigit() else panel
        return tuple(args), {**kwargs, "ax": panel, **({"side": "right"} if self.side == "right" else {})}

    def _styled(self, session: Session) -> Any:
        from ..views import figure_of

        args, kwargs = self._call()
        return figure_of(session[self.source], self.tab).call(self.method, *args, **kwargs)

    def validate(self, session: Session) -> list[str]:
        if self.source not in session.objects:
            return [f"no object named {self.source!r}"]
        try:
            self._styled(session).draw()
        except Exception as exc:
            return [str(exc)]
        return []

    def to_call(self) -> Any:
        args, kwargs = self._call()
        return _figure_chain(self.source, self.tab, [("call", (self.method, *args), kwargs)], self._target(), self.command_id)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


def _steps(text: str) -> list[int]:
    """Style steps as typed or chosen: "0, 2, 1", or "2: line('gc', color='C3')" (a step from the list)."""
    from ..errors import SpecificationError

    out = []
    for part in ([text.split(":", 1)[0]] if ":" in text else text.split(",")):
        part = part.strip()
        if part:
            if not part.lstrip("-").isdigit():
                raise SpecificationError(f"{part!r} is not a style step number")
            out.append(int(part))
    return out


@command(menu=None)  # opened by a figure's Edit… > Recorded styles…
@dataclass(frozen=True)
class ArrangeStyles(Command):
    """Remove or reorder the style calls recorded on a figure. ``keep`` lists the steps to keep, in the order they
    should apply, by their numbers in the figure's list of recorded styles (``fig.style_steps()``): ``0, 2, 1`` swaps
    the last two of three; ``0, 2`` removes step 1. Recorded as ``figure_of(fit).without_style(1)`` or
    ``.reorder_style([0, 2, 1])``.
    """

    topics: ClassVar[tuple[str, ...]] = ()
    dialog_title: ClassVar[str] = "Recorded styles"

    source: Annotated[str, Widget("text", help="the figure, or the result whose figure it is")]
    tab: Annotated[str, Widget("text", advanced=True, help="for a result: the tab its figure is on")] = "Graph"
    keep: Annotated[str, Widget("text", help="the steps to keep, in order, e.g. 0, 2, 1 (empty: none)")] = ""
    name: Annotated[str, Widget("text", help="save as (empty: <source>_figure)")] = ""
    steps: Annotated[int, Widget("hidden", help="how many style calls the figure had (-1: not known)")] = -1

    @classmethod
    def for_figure(cls, source: str, figure: Any, tab: str = "Graph", is_figure: bool = False) -> ArrangeStyles:
        return cls(source, tab, keep=", ".join(str(k) for k in range(len(figure.style))),
                   name=source if is_figure else f"{source}_{'_'.join(tab.lower().split())}", steps=len(figure.style))

    def _target(self) -> str:
        return self.name or f"{self.source}_figure"

    def _figure(self, session: Session) -> Any:
        from ..views import figure_of

        return figure_of(session[self.source], self.tab)

    def _call(self, count: int) -> tuple[str, Any]:
        order = [k % count if count > 0 and -count <= k < count else k for k in _steps(self.keep)]  # out of range: refused below
        inside = all(0 <= k < count for k in order)
        if count >= 0 and inside and order == sorted(order) and len(set(order)) == len(order):  # only removed: say which
            return "without_style", [k for k in range(count) if k not in order]
        return "reorder_style", order

    def validate(self, session: Session) -> list[str]:
        if self.source not in session.objects:
            return [f"no object named {self.source!r}"]
        try:
            self._arranged(session)
        except Exception as exc:
            return [str(exc)]
        return []

    def _arranged(self, session: Session) -> Any:
        figure = self._figure(session)
        method, steps = self._call(len(figure.style))
        return figure.without_style(*steps) if method == "without_style" else figure.reorder_style(steps)

    def to_call(self, session: Session) -> Any:
        method, steps = self._call(len(self._figure(session).style))
        call = (method, tuple(steps), {}) if method == "without_style" else (method, (steps,), {})
        return _figure_chain(self.source, self.tab, [call], self._target(), self.command_id)

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        method, steps = self._call(self.steps)  # without the count, the steps kept, in order
        tab = f", {self.tab!r}" if self.tab != "Graph" else ""
        inner = ", ".join(str(k) for k in steps) if method == "without_style" else repr(steps)
        return f"{self._target()} = figure_of({self.source}{tab}).{method}({inner})"


@command(menu=None)  # opened by Recorded styles… > Change…
@dataclass(frozen=True)
class ChangeStyle(Command):
    """Change one recorded style call's keyword arguments: its colour, line style, label, position... The call keeps
    its series, panel and place in the order. Give matplotlib's keyword arguments in Python's notation
    (``color='C2', linewidth=2``); with **replace all** they replace the call's arguments, otherwise they change only
    those given (``None`` removes one). Recorded as ``figure_of(fit).restyle(0, color='C2')``.
    """

    topics: ClassVar[tuple[str, ...]] = ()
    dialog_title: ClassVar[str] = "Change a style"

    source: Annotated[str, Widget("text", help="the figure, or the result whose figure it is")]
    tab: Annotated[str, Widget("text", advanced=True, help="for a result: the tab its figure is on")] = "Graph"
    step: Annotated[str, Widget("stylestep", depends_on="source", help="the recorded style call to change")] = ""
    properties: Annotated[str, Widget("text", help="its keyword arguments, e.g. color='C2', linewidth=2")] = ""
    replace_all: Annotated[bool, Widget("checkbox", help="replace all its keyword arguments with these")] = False
    name: Annotated[str, Widget("text", help="save as (empty: <source>_figure)")] = ""

    @classmethod
    def for_step(cls, source: str, figure: Any, step: int, tab: str = "Graph", is_figure: bool = False) -> ChangeStyle:
        """The dialog on one step, its current keyword arguments filled in (all of them, so they replace the old)."""
        call = figure.style[step]
        props = call.props["kwargs"] if call.kind == "call" else call.props
        return cls(source, tab, step=f"{step}: {call.code()}", replace_all=True,
                   properties=", ".join(f"{k}={v!r}" for k, v in props.items()),
                   name=source if is_figure else f"{source}_{'_'.join(tab.lower().split())}")

    def _target(self) -> str:
        return self.name or f"{self.source}_figure"

    def _call(self) -> tuple[int, dict[str, Any]]:
        from ..errors import SpecificationError
        from ..forms import parse_options

        steps = _steps(self.step)
        if len(steps) != 1:
            raise SpecificationError("pick the style call to change")
        _, props = parse_options(self.properties)
        return steps[0], props

    def _styled(self, session: Session) -> Any:
        from ..views import figure_of

        step, props = self._call()
        return figure_of(session[self.source], self.tab).restyle(step, replace_all=self.replace_all, **props)

    def validate(self, session: Session) -> list[str]:
        if self.source not in session.objects:
            return [f"no object named {self.source!r}"]
        try:
            self._styled(session).draw()
        except Exception as exc:
            return [str(exc)]
        return []

    def to_call(self) -> Any:
        step, props = self._call()
        return _figure_chain(self.source, self.tab, [("restyle", (step,), {"replace_all": self.replace_all, **props})], self._target(),
                             self.command_id)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu=None)  # opened by a figure's Edit… > Apply style from…
@dataclass(frozen=True)
class CopyStyle(Command):
    """Style a figure as another one is: the other figure's recorded style calls are added after this one's own,
    where they apply. A series' call is copied if this figure draws that series; a panel's, if it has that panel (and
    the right-hand axis, for one on it). Notes, text, lines and bands mark the other chart's data, so they are
    copied only if asked. Recorded as ``figure_of(fit).with_style_of(figure_of(house_chart))``.
    """

    topics: ClassVar[tuple[str, ...]] = ()
    dialog_title: ClassVar[str] = "Apply style from"

    source: Annotated[str, Widget("text", help="the figure to style, or the result whose figure it is")]
    tab: Annotated[str, Widget("text", advanced=True, help="for a result: the tab its figure is on")] = "Graph"
    style_from: Annotated[str, Widget("text", help="the figure (or result) whose style to copy")] = ""
    from_tab: Annotated[str, Widget("text", advanced=True, help="for a result: the tab its figure is on")] = "Graph"
    notes: Annotated[bool, Widget("checkbox", help="copy its notes, text, lines and bands too")] = False
    name: Annotated[str, Widget("text", help="save as (empty: <source>_figure)")] = ""

    @classmethod
    def for_figure(cls, source: str, figure: Any, tab: str = "Graph", is_figure: bool = False) -> CopyStyle:
        return cls(source, tab, name=source if is_figure else f"{source}_{'_'.join(tab.lower().split())}")

    def _target(self) -> str:
        return self.name or f"{self.source}_figure"

    def _figures(self, session: Session) -> tuple[Any, Any]:
        from ..views import figure_of

        return figure_of(session[self.source], self.tab), figure_of(session[self.style_from], self.from_tab)

    def validate(self, session: Session) -> list[str]:
        for name in (self.source, self.style_from):
            if not name.strip():
                return ["name the figure whose style to copy"]
            if name not in session.objects:
                return [f"no object named {name!r}"]
        try:
            figure, other = self._figures(session)
            if not other.style:
                return [f"{self.style_from} has no recorded styles to copy"]
            if len(figure.with_style_of(other, notes=self.notes).style) == len(figure.style):
                return [f"none of {self.style_from}'s styles apply here: " + "; ".join(figure.style_skipped(other, self.notes))]
            figure.with_style_of(other, notes=self.notes).draw()
        except Exception as exc:
            return [str(exc)]
        return []

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        other = make_call("pymetrix:figure_of", obj=Ref(self.style_from), tab=self.from_tab)
        return _figure_chain(self.source, self.tab, [("with_style_of", (other,), {"notes": self.notes})], self._target(), self.command_id)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu=None)  # opened by a figure's Save…
@dataclass(frozen=True)
class SaveFigure(Command):
    """Save a figure for publication: as PNG, PDF, SVG or EPS (from the file's extension), at a size made for print
    or slides, at a resolution, with or without a background. PDF and EPS embed their fonts; SVG draws text as
    outlines unless it is kept editable. Text keeps its size in points. The file records what made it (pymetrix and
    matplotlib versions, the series, the style calls). Recorded as ``figure_of(fit).save('fit.pdf', size='column')``.
    """

    topics: ClassVar[tuple[str, ...]] = ()
    dialog_title: ClassVar[str] = "Save figure"

    source: Annotated[str, Widget("text", help="the figure, or the result whose figure it is")]
    tab: Annotated[str, Widget("text", advanced=True, help="for a result: the tab its figure is on")] = "Graph"
    path: Annotated[str, Widget("savefile", source="PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;EPS (*.eps)",
                                help="the file to write; its extension sets the format")] = ""
    size: Annotated[Literal["as drawn", "column", "page", "slide", "half-a4", "custom"],
                    Widget("choice", help="column 89 mm, page 183 mm wide (journals), slide 16:9, half an A4 page")] = "as drawn"
    width: Annotated[str, Widget("text", show_if=("size", ("custom",)), help="the width, e.g. 120mm, 12cm or 4.7in")] = ""
    height: Annotated[str, Widget("text", show_if=("size", ("custom",)), help="the height (empty: keep the figure's shape)")] = ""
    dpi: Annotated[int, Widget("number", min=50, max=1200, help="dots per inch, for PNG (300 for print)")] = 300
    transparent: Annotated[bool, Widget("checkbox", help="no background")] = False
    editable_text: Annotated[bool, Widget("checkbox", advanced=True, help="SVG: keep text as text, to edit")] = False

    @classmethod
    def for_figure(cls, source: str, figure: Any, tab: str = "Graph", is_figure: bool = False) -> SaveFigure:
        return cls(source, tab, path=f"{source}.png" if is_figure else f"{source}_{'_'.join(tab.lower().split())}.png")

    def _size(self) -> Any:
        if self.size == "as drawn":
            return None
        if self.size == "custom":
            return (self.width.strip(), self.height.strip() or None)
        return self.size

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self._size() is not None:
            options["size"] = self._size()
        if self.dpi != 150:
            options["dpi"] = self.dpi
        if self.transparent:
            options["transparent"] = True
        if self.editable_text:
            options["editable_text"] = True
        return options

    def validate(self, session: Session) -> list[str]:
        from ..figures import _size_inches

        if self.source not in session.objects:
            return [f"no object named {self.source!r}"]
        if not self.path.strip():
            return ["choose the file to write"]
        if not self.path.lower().endswith((".png", ".pdf", ".svg", ".eps", ".jpg", ".jpeg", ".tif", ".tiff")):
            return ["the file's extension sets the format: .png, .pdf, .svg or .eps"]
        if self.size == "custom" and not self.width.strip():
            return ["give the width, e.g. 120mm"]
        try:
            if self._size() is not None:
                _size_inches(self._size(), (7.5, 3.6))
        except Exception as exc:
            return [str(exc)]
        return []

    def to_call(self) -> Any:
        return _figure_chain(self.source, self.tab, [("save", (self.path,), self._options())], "", self.command_id)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/Coefficient plot...")
@dataclass(frozen=True)
class CoefficientPlot(Command):
    """Plot estimated coefficients with their confidence intervals: a marker at each estimate and a line across its
    interval, one row per coefficient, with a dashed line at zero. Tick several equations to compare them side by
    side, coefficient by coefficient (each in its own colour, named as in the workspace). Bayesian estimates show
    their posterior intervals.

    For classical estimates the intervals come from each equation's own standard errors, so they are only as
    reliable as its covariance choice (classic, robust, HAC or clustered): re-estimate with a robust covariance where heteroskedasticity or
    serial correlation is likely. An interval that excludes zero corresponds to a coefficient significant at one
    minus the chosen level, one coefficient at a time; it says nothing about joint significance. Coefficients are
    comparable across rows only if the regressors share units, so standardise or choose comparable terms first.
    Overlapping intervals of two equations do not imply that the coefficients are not significantly different.
    """

    topics: ClassVar[tuple[str, ...]] = ()

    results: Annotated[str, Widget("object", source="FittedEquation", multiple=True, help="estimated equations; several are compared coefficient by coefficient")]
    terms: Annotated[str, Widget("text", help="the coefficients to show, in order, e.g. dlog(y), dlog(c)[-1] (empty: all)")] = ""
    level: Annotated[Literal["90", "95", "99"], Widget("choice", help="confidence level of the intervals, in %; 95 is conventional; 90 gives narrower intervals at less confidence")] = "95"
    title: Annotated[str, Widget("text", help="a title for the figure (optional)")] = ""
    name: str = "coefficients"

    def _terms(self) -> list[str]:
        import re

        return [t.strip() for t in re.split(r",(?![^()\[\]]*[)\]])", self.terms) if t.strip()]  # commas inside brackets stay

    def _call(self) -> str:
        names = _names(self.results)
        target = names[0] if len(names) == 1 else "{" + ", ".join(f"{n!r}: {n}" for n in names) + "}"
        options = (f", terms={self._terms()!r}" if self._terms() else "") + (f", level={int(self.level) / 100}" if self.level != "95" else "")
        return f"coefficient_figure({target}{options}{f', title={self.title!r}' if self.title else ''})"

    def validate(self, session: Session) -> list[str]:
        names = _names(self.results)
        if not names:
            return ["tick the estimated equations to plot"]
        missing = [n for n in names if n not in session.objects]
        if missing:
            return [f"no object named {missing[0]!r}"]
        try:
            self.run_figure(session)
        except Exception as exc:
            return [str(exc)]
        return []

    def run_figure(self, session: Session) -> Any:
        from ..figures import coefficient_figure

        names = _names(self.results)
        results = session[names[0]] if len(names) == 1 else {n: session[n] for n in names}
        return coefficient_figure(results, self._terms() or None, int(self.level) / 100, self.title)

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        names = _names(self.results)
        return make_call("pymetrix:coefficient_figure", output=self.name, recorded_by=self.command_id,
                         results=Ref(names[0]) if len(names) == 1 else {n: Ref(n) for n in names}, terms=self._terms() or None,
                         level=int(self.level) / 100, title=self.title)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu=None)  # opened by a table's Heatmap button
@dataclass(frozen=True)
class TableHeatmap(Command):
    """Show a result's table as a heatmap: each number a coloured cell, with a colour bar (Granger p-values,
    correlations, responses by horizon...). The table's numeric columns are used, less a posterior table's sampling
    diagnostics (R-hat, effective draws), which describe the sampler rather than the posterior. Values of both signs
    get a scale symmetric about zero. The scale is chosen by the columns (``auto``): one for the table, or each column
    its own where the columns are in different units, such as a posterior's means beside its effective draws.
    Recorded as ``heatmap_figure(granger, view='Results', scale='auto', diagnostics=False)``.
    """

    topics: ClassVar[tuple[str, ...]] = ()
    dialog_title: ClassVar[str] = "Table as a heatmap"

    source: Annotated[str, Widget("text", help="the result whose table to show")]
    table: Annotated[str, Widget("text", help="the table: the tab it is on")] = "Results"
    label: Annotated[str, Widget("text", help="what the colours measure, for the colour bar (optional)")] = ""
    scale: Annotated[Literal["auto", "table", "column"],
                     Widget("choice", help="auto: each column its own scale when the columns differ in units or size; "
                                           "table: one scale; column: each column its own")] = "auto"
    name: Annotated[str, Widget("text", help="save as (empty: <source>_heatmap)")] = ""

    def _target(self) -> str:
        return self.name or f"{self.source}_heatmap"

    def _figure(self, session: Session) -> Any:
        from ..figures import heatmap_figure

        return heatmap_figure(session[self.source], view=self.table, label=self.label, scale=self.scale, diagnostics=False)

    def validate(self, session: Session) -> list[str]:
        if self.source not in session.objects:
            return [f"no object named {self.source!r}"]
        try:
            self._figure(session)
        except Exception as exc:
            return [str(exc)]
        return []

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:heatmap_figure", output=self._target(), recorded_by=self.command_id, values=Ref(self.source),
                         view=self.table, label=self.label, scale=self.scale, diagnostics=False)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu=None)  # opened by a figure's Edit… > Use as house style…
@dataclass(frozen=True)
class UseHouseStyle(Command):
    """Make this figure's recorded style the house style: every figure drawn from now on starts from it, where its
    calls apply (a series' call where that series is drawn, a panel's where the panel is), with the figure's own
    calls after it. Notes, text, lines and bands stay with this figure unless copied too. Tick **stop** to draw
    figures without a house style again. Recorded as ``set_house_style(figure_of(chart))``; open windows show it when
    they are drawn again.
    """

    topics: ClassVar[tuple[str, ...]] = ()
    dialog_title: ClassVar[str] = "House style"

    source: Annotated[str, Widget("text", help="the figure whose style to use, or the result whose figure it is")] = ""
    tab: Annotated[str, Widget("text", advanced=True, help="for a result: the tab its figure is on")] = "Graph"
    notes: Annotated[bool, Widget("checkbox", help="its notes, text, lines and bands too")] = False
    stop: Annotated[bool, Widget("checkbox", help="stop using a house style")] = False

    def validate(self, session: Session) -> list[str]:
        if self.stop:
            return []
        if self.source not in session.objects:
            return [f"no object named {self.source!r}"]
        try:
            from ..views import figure_of

            if not figure_of(session[self.source], self.tab).style:
                return [f"{self.source} has no recorded style to use: style it first (Edit… → Style)"]
        except Exception as exc:
            return [str(exc)]
        return []

    def run(self, session: Session) -> Any:
        from ..figures import set_house_style
        from ..views import figure_of

        set_house_style(None if self.stop else figure_of(session[self.source], self.tab), notes=self.notes)
        return None

    def to_code(self) -> str:
        if self.stop:
            return "set_house_style(None)"
        tab = f", {self.tab!r}" if self.tab != "Graph" else ""
        return f"set_house_style(figure_of({self.source}{tab}){', notes=True' if self.notes else ''})"
