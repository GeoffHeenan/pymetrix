"""Panel data tools: pivot tables, reshaping between wide and long, slices, balancing, panel statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session

_TOPICS: tuple[str, ...] = ("panel-data-tools",)


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def _list_code(items: list[str]) -> str:
    return repr(items[0]) if len(items) == 1 else repr(items)


class _OnData(Command):
    data: str

    def _check_data(self, session: Session) -> list[str]:
        from ..data import Dataset

        return [] if isinstance(session.objects.get(self.data), Dataset) else [f"no dataset named {self.data!r}"]


@command(menu="Data/Pivot table...")
@dataclass(frozen=True)
class PivotTable(_OnData):
    """Summarise series by the values of axes or categories: a pivot table, as in a spreadsheet.

    Rows and columns can be axes (country, time) or series of categories (region). Choose the series to
    summarise and how: mean, sum, count, min, max, median, std, first or last, or several separated by commas.
    Totals adds a Total row and column. The table can be copied into Excel.
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS

    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    rows: Annotated[str, Widget("level", depends_on="data", multiple=True, help="axes or categories down the side, e.g. region, country")] = ""
    columns: Annotated[str, Widget("level", depends_on="data", multiple=True, help="axes or categories across the top, e.g. time; empty = none")] = ""
    values: Annotated[str, Widget("series", depends_on="data", multiple=True, help="series to summarise; empty = all")] = ""
    agg: Annotated[str, Widget("text", help="mean, sum, count, min, max, median, std, first, last; several separated by commas")] = "mean"
    margins: Annotated[bool, Widget("checkbox", advanced=True, help="add a Total row and column, each computed with the same summary over all the data")] = False
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    name: str = "pivot"
    fill: Annotated[str, Widget("text", advanced=True, help="value for empty cells, e.g. 0; empty = leave them blank")] = ""

    def _args(self) -> dict[str, Any]:
        out: dict[str, Any] = {"rows": _names(self.rows)}
        if _names(self.columns):
            out["columns"] = _names(self.columns)
        if _names(self.values):
            out["values"] = _names(self.values)
        aggs = _names(self.agg) or ["mean"]
        if aggs != ["mean"]:
            out["agg"] = aggs[0] if len(aggs) == 1 else aggs
        if self.margins:
            out["margins"] = True
        if self.fill.strip():
            out["fill"] = float(self.fill)
        if self.start or self.end:
            out["span"] = (self.start or None, self.end or None)
        return out

    def validate(self, session: Session) -> list[str]:
        errors = self._check_data(session)
        if not _names(self.rows):
            errors.append("choose at least one row")
        try:
            float(self.fill) if self.fill.strip() else None
        except ValueError:
            errors.append(f"fill must be a number, not {self.fill!r}")
        return errors

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        args = {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in self._args().items()}  # one row: a name
        return make_call("pymetrix:dataset.pivot", Ref(self.data), output=self.name, recorded_by=self.command_id, **args)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Data/Panel to wide...")
@dataclass(frozen=True)
class ToWide(_OnData):
    """One series per entity: y becomes y_AUS, y_NZL... and the panel becomes a set of time series.

    Useful for graphs comparing countries, for exporting to a spreadsheet, and for methods that need time series.
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS

    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    series: Annotated[str, Widget("series", depends_on="data", multiple=True, help="series to spread; empty = all")] = ""
    name: str = "wide"
    sep: Annotated[str, Widget("text", advanced=True, help="between the series name and the entity")] = "_"

    def validate(self, session: Session) -> list[str]:
        return self._check_data(session)

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:dataset.wide", Ref(self.data), output=self.name, recorded_by=self.command_id, names=tuple(_names(self.series)), sep=self.sep)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Data/Wide to panel...")
@dataclass(frozen=True)
class ToLong(_OnData):
    """Make a panel from one column per series and entity (y_AUS, c_AUS, y_NZL...), or from one column per entity
    of a single series (a sheet with a column per country).

    The series names are what comes before the last separator, when two or more columns share it; give them
    under Series to be exact. Columns that match none are kept, the same for every entity (e.g. a world price).
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS

    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    entity: Annotated[str, Widget("text", help="name of the new axis, e.g. country")] = "country"
    layout: Annotated[Literal["series_entity", "one_series"], Widget("choice", help="series_entity: columns y_AUS, c_AUS...; one_series: a column per entity")] = "series_entity"
    stubs: Annotated[str, Widget("text", help="series_entity: the series names, e.g. y, c; empty = found from the columns")] = ""
    series_name: Annotated[str, Widget("text", help="one_series: the name of the series, e.g. gdp")] = ""
    name: str = "panel"
    sep: Annotated[str, Widget("text", advanced=True, help="between the series name and the entity")] = "_"

    def _kwargs(self) -> dict[str, Any]:
        out: dict[str, Any] = {"entity": self.entity}
        if self.layout == "one_series":
            out["name"] = self.series_name
        elif _names(self.stubs):
            out["stubs"] = _names(self.stubs)
        if self.sep != "_":
            out["sep"] = self.sep
        return out

    def validate(self, session: Session) -> list[str]:
        errors = self._check_data(session)
        if not self.entity.isidentifier():
            errors.append(f"the axis name {self.entity!r} must be a plain name")
        if self.layout == "one_series" and not self.series_name.isidentifier():
            errors.append("give the name of the series")
        return errors

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:dataset.long", Ref(self.data), output=self.name, recorded_by=self.command_id, **self._kwargs())

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Data/One entity or period...")
@dataclass(frozen=True)
class Slice(_OnData):
    """Take one entity's time series out of a panel, or the cross-section of all entities in one period.

    The result is an ordinary dataset: an entity's slice is a time series, so time-series tools (unit-root tests,
    ARIMA, VARs) apply to it; a period's slice is a cross-section for regressions across entities at one date. Use
    it to look at one unit in detail, but remember that conclusions from one entity or one period need not hold for
    the panel as a whole. Choose either an entity or a period, not both.
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS

    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    entity: Annotated[str, Widget("entity", depends_on="data", help="one entity, e.g. WSM: its time series")] = ""
    period: Annotated[str, Widget("period", help="or one period, e.g. 2020Q1: the cross-section")] = ""
    name: str = "panel_slice"

    def validate(self, session: Session) -> list[str]:
        errors = self._check_data(session)
        if bool(self.entity) == bool(self.period):
            errors.append("choose either an entity or a period")
        return errors

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        if self.entity:
            return make_call("pymetrix:dataset.entity", Ref(self.data), output=self.name, recorded_by=self.command_id, value=self.entity)
        return make_call("pymetrix:dataset.cross_section", Ref(self.data), output=self.name, recorded_by=self.command_id, period=self.period)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Data/Balance panel...")
@dataclass(frozen=True)
class BalancePanel(_OnData):
    """Make a balanced panel: keep the periods in which every entity has data on the chosen series, or the
    entities that have data in every period. Use Panel statistics (coverage) to see what is missing first.

    Some estimators and tests need a balanced panel, but balancing throws data away, and the choice of which matters:
    dropping periods shortens every series, dropping entities narrows the cross-section. If whether an entity has
    gaps is related to the outcome (a firm leaving because it is doing badly), the balanced panel is a selected
    sample and estimates on it can be biased. Choose only the series the analysis uses, so that gaps in other series
    do not cost observations. A missing entity-period counts as a gap even when the row is absent altogether.
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS

    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    series: Annotated[str, Widget("series", depends_on="data", multiple=True, help="series that must be present; empty = all")] = ""
    keep: Annotated[Literal["periods", "entities"], Widget("choice", help="periods: drop periods with gaps; entities: drop entities with gaps")] = "periods"
    name: str = "balanced"

    def validate(self, session: Session) -> list[str]:
        return self._check_data(session)

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:dataset.balance", Ref(self.data), output=self.name, recorded_by=self.command_id, names=tuple(_names(self.series)), keep=self.keep)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Data/Panel statistics...")
@dataclass(frozen=True)
class PanelStatistics(_OnData):
    """Describe a panel: coverage (each entity's first and last periods, observations and gaps), or overall,
    between-entity and within-entity variation of each series.

    Coverage shows, for each entity, the first and last periods with data, the observations and the gaps, and says
    whether the panel is balanced. Variation splits each series' spread: the between standard deviation is that of
    the entity means, the within standard deviation that of the deviations from them (plus the overall mean, so the
    within rows read on the series' scale). Fixed effects estimators use only the within variation, so a regressor that
    varies mostly between entities is poorly identified by them; random effects and pooled estimators use both. N is
    the observations, n the entities and T the average periods per entity.
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS

    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    series: Annotated[str, Widget("series", depends_on="data", multiple=True, help="series to describe; empty = all")] = ""
    show: Annotated[Literal["coverage", "variation"], Widget("choice", help="coverage: who has data when; variation: between and within")] = "coverage"
    name: str = "panel_stats"
    by: Annotated[str, Widget("text", advanced=True, show_if=("show", ("variation",)),
                              help="variation: the entity axis, for panels with several (e.g. country); empty = the only one")] = ""

    def validate(self, session: Session) -> list[str]:
        return self._check_data(session)

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        verb = "coverage" if self.show == "coverage" else "describe_panel"
        extra = {"by": self.by.strip()} if self.show == "variation" and self.by.strip() else {}
        return make_call(f"pymetrix:dataset.{verb}", Ref(self.data), output=self.name, recorded_by=self.command_id,
                         names=tuple(_names(self.series)), **extra)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
