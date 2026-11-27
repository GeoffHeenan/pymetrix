"""Data dialogs: frequency conversion, aggregation over a panel's entities, and loading from a data connector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session

_HOWS = ("sum", "mean", "first", "last", "min", "max", "median")


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def _rules(text: str) -> dict[str, str]:
    """``"ip=sum, emp=last"`` -> {"ip": "sum", "emp": "last"} (unchecked: see ``_rule_errors``)."""
    out = {}
    for part in _names(text):
        series, _, how = part.partition("=")
        out[series.strip()] = how.strip()
    return out


def _rule_errors(text: str, names: list[str]) -> list[str]:
    errors = []
    for part in _names(text):
        series, eq, how = (p.strip() for p in part.partition("="))
        if not eq or not series or not how:
            errors.append(f"write each series' rule as series=how (e.g. ip=sum, emp=last), not {part!r}")
        elif how not in _HOWS:
            errors.append(f"{series}: {how!r} is not one of {', '.join(_HOWS)}")
        elif names and series not in names:
            errors.append(f"no series {series!r} in the data")
    return errors


def _dataset(session: Session, name: str) -> Any:
    from ..data import Dataset

    obj = session.objects.get(name)
    return obj if isinstance(obj, Dataset) else None


def _how_argument(how: str, rules: str) -> Any:
    return how if how else (_rules(rules) or None)


def _measure_errors(data: Any, how: str, rules: str, grouped: tuple[str, ...] = ()) -> list[str]:
    """By measure, every numeric series (but those ``grouped`` by) needs one (or a rule): say which do not, before the
    call fails on the first."""
    if how:
        return []
    covered = set(_rules(rules)) | set(grouped)
    numeric = [str(c) for c in data.frame.select_dtypes("number").columns]
    lacking = [n for n in numeric if n not in covered and (n not in data.variables or data.variables[n].measure is None)]
    if not lacking:
        return []
    listed = ", ".join(lacking[:5]) + (" ..." if len(lacking) > 5 else "")
    return [f"{listed} {'has' if len(lacking) == 1 else 'have'} no measure (flow, stock, rate...): choose how to convert "
            "every series, or give a rule for each under Advanced (Series rules)"]


# -- frequency conversion -----------------------------------------------------------------------------------------------


def _days(freq: str) -> float:
    import pandas as pd

    period = pd.Period("2001-07-01", freq=freq)
    return (period.end_time - period.start_time).total_seconds() / 86400


@command(menu="Data/Convert frequency...")
@dataclass(frozen=True)
class ConvertFrequency(Command):
    """Convert a dataset to a lower frequency: monthly to quarterly, quarterly to annual (or a July-June fiscal year).

    Each new period combines the old ones it contains: their sum (flows such as GDP or sales), their average (rates,
    indices, prices), or the first or last value (stocks such as the money supply at the end of the quarter). By
    measure (the default) each series is combined as its metadata says: flows and counts summed, stocks the last
    value, rates, indices and ratios averaged; a series without a measure needs a rule (Advanced) or one choice for
    all. A period with a sub-period missing (a quarter with two of its months) is left missing, not computed from
    part of the data. Panels are converted entity by entity.

    To a higher frequency there is nothing to add up: Data → Transform → Temporal disaggregation spreads annual or
    quarterly totals over the quarters or months, guided by indicators.
    """

    topics: ClassVar[tuple[str, ...]] = ("frequency-conversion",)

    data: Annotated[str, Widget("object", source="Dataset", help="the dataset to convert (it is kept as it is)")] = "data"
    freq: Annotated[Literal["Q", "Y", "Y-JUN", "M", "W"], Widget("choice", label="New frequency",
                                                                 help="Q quarterly, Y annual, Y-JUN a July-June fiscal year, M monthly, W weekly; lower than the data's")] = "Q"
    how: Annotated[Literal["", "sum", "mean", "first", "last"],
                   Widget("choice", help="how the periods of each new period combine: sum for flows, average for rates and indices, "
                                         "first or last for stocks; by measure, as each series' metadata says")] = ""
    name: str = "converted"
    rules: Annotated[str, Widget("text", advanced=True, show_if=("how", ("",)), label="Series rules",
                                 help="rules for some series, overriding their measure: ip=sum, emp=last, pmi=mean "
                                      "(sum, mean, first, last, min, max or median)")] = ""

    def validate(self, session: Session) -> list[str]:
        data = _dataset(session, self.data)
        if data is None:
            return [f"no dataset named {self.data!r}"]
        t = data.time_axis
        if t is None:
            return [f"{self.data} has no time axis: there is no frequency to convert"]
        old = data.frame.index.get_level_values(t.name).freqstr
        if _days(self.freq) < _days(old):
            return [f"{self.data} is {old}; {self.freq} is a higher frequency: use Data → Transform → Temporal "
                    "disaggregation to go from low to high"]
        if _days(self.freq) == _days(old):
            return [f"{self.data} is already {old}"]
        errors = _rule_errors(self.rules, list(data.names)) if not self.how else []
        return errors + _measure_errors(data, self.how, self.rules)

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:dataset.to_freq", Ref(self.data), output=self.name, recorded_by=self.command_id, freq=self.freq,
                         how=_how_argument(self.how, self.rules))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Data/Aggregate over entities...")
@dataclass(frozen=True)
class AggregateEntities(Command):
    """Add up (or average) a panel over its entities: the countries' GDP summed into a world total, period by period.

    Keep the axes or categories under By: the time axis gives one total per period; a category such as region as
    well gives one per region and period. Every other row is combined: by measure, flows, counts and stocks are
    summed, rates, indices and ratios averaged (an unweighted average: for a weighted one, derive the weighted
    series first). A series without a measure needs a rule (Advanced) or one choice for all.
    """

    topics: ClassVar[tuple[str, ...]] = ("frequency-conversion", "panel-data-tools")

    data: Annotated[str, Widget("object", source="Dataset", help="the panel (or any dataset) to aggregate")] = "data"
    by: Annotated[str, Widget("level", depends_on="data", multiple=True,
                              help="what to keep: the time axis for a total per period; add a category (region) for one per group")] = ""
    how: Annotated[Literal["", "sum", "mean", "first", "last"],
                   Widget("choice", help="how the rows combine: sum, average, first or last; by measure, as each series' metadata says")] = ""
    name: str = "aggregated"
    rules: Annotated[str, Widget("text", advanced=True, show_if=("how", ("",)), label="Series rules",
                                 help="rules for some series, overriding their measure: gdp=sum, rate=mean "
                                      "(sum, mean, first, last, min, max or median)")] = ""

    def validate(self, session: Session) -> list[str]:
        data = _dataset(session, self.data)
        if data is None:
            return [f"no dataset named {self.data!r}"]
        by = _names(self.by)
        if not by:
            return ["choose what to keep under By (the time axis, for a total per period)"]
        known = set(data.names) | {str(n) for n in data.frame.index.names}
        errors = [f"no axis or series {b!r} in {self.data}" for b in by if b not in known]
        if not self.how:
            errors += _rule_errors(self.rules, list(data.names)) + _measure_errors(data, self.how, self.rules, tuple(by))
        return errors

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        by = _names(self.by)
        return make_call("pymetrix:dataset.aggregate", Ref(self.data), output=self.name, recorded_by=self.command_id,
                         by=by[0] if len(by) == 1 else by, how=_how_argument(self.how, self.rules))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


# -- data connectors ----------------------------------------------------------------------------------------------------


def _with_vintages() -> list[str]:
    from ..io import connectors

    return [k for k, c in connectors().items() if c.vintage]


def _with_start() -> list[str]:
    from ..io import connectors

    return [k for k, c in connectors().items() if c.start]


def _with_end() -> list[str]:
    from ..io import connectors

    return [k for k, c in connectors().items() if c.end]


@command(menu="File/Load from connector...")
@dataclass(frozen=True)
class LoadFromConnector(Command):
    """Read series from a data service: FRED, the World Bank, SDMX providers (the ECB, Eurostat, the OECD, the IMF,
    the BIS, or an institution's own service) and those plugins add.

    Give the provider and the series codes as the provider writes them: FRED IDs (GDPC1,UNRATE), World Bank
    indicators (NY.GDP.MKTP.CD, a panel by country), or an SDMX PROVIDER/FLOW/KEY (ECB/EXR/M.USD.EUR.SP00.A). The
    dialog writes the connector's address, the same text File → Load data… accepts as its path
    (``fred://GDPC1?start=1990Q1``), so the step can be repeated, or edited, in code. Where the provider keeps
    vintages (FRED, through ALFRED) a date gives the data as published that day, for real-time analysis; empty
    gives the latest. File → Browse series… finds codes by searching or browsing the provider, and fills them in here.
    A project keeps the data as read, so it opens with the same numbers whatever the provider holds by then; File →
    Refresh data… reads it again.
    """

    topics: ClassVar[tuple[str, ...]] = ("data-connectors",)

    provider: Annotated[str, Widget("named", source="connectors",
                                    help="the data service: fred, worldbank, sdmx, or a plugin's (each reads addresses provider://codes)")] = "fred"
    codes: Annotated[str, Widget("text", help="FRED: series IDs, e.g. GDPC1,UNRATE; World Bank: indicators, e.g. NY.GDP.MKTP.CD; "
                                              "SDMX: PROVIDER/FLOW/KEY, e.g. ECB/EXR/M.USD.EUR.SP00.A")] = ""
    start: Annotated[str, Widget("period", group="Periods", show_if=("provider", _with_start),
                                 help="first period, e.g. 1990Q1 or 1990-01 (World Bank: a year); empty = from the start")] = ""
    end: Annotated[str, Widget("period", group="Periods", show_if=("provider", _with_end),
                               help="last period; empty = to the latest")] = ""
    vintage: Annotated[str, Widget("text", show_if=("provider", _with_vintages),
                                   help="a date, e.g. 2020-01-30: the data as published that day (ALFRED for FRED); empty = the latest")] = ""
    name: str = "data"
    options: Annotated[str, Widget("text", advanced=True,
                                   help="other parameters of the address, as the connector takes them, e.g. country=WSM;TON (World Bank) "
                                        "or entity=REF_AREA (SDMX); several separated by &")] = ""

    def _connector(self) -> Any:
        from ..io import connectors

        return connectors().get(self.provider.strip().lower())

    def address(self) -> str:
        """The connector's address the dialog reads: ``provider://codes?start=...&end=...&options``."""
        conn = self._connector()
        params = []
        if self.start.strip() and conn is not None and conn.start:
            params.append(f"{conn.start}={self.start.strip()}")
        if self.end.strip() and conn is not None and conn.end:
            params.append(f"{conn.end}={self.end.strip()}")
        params += [p.strip() for p in self.options.split("&") if p.strip()]
        codes = self.codes.strip().replace(" ", "")
        return f"{self.provider.strip().lower()}://{codes}" + (f"?{'&'.join(params)}" if params else "")

    def validate(self, session: Session) -> list[str]:
        import pandas as pd

        from ..io import connectors

        conn = self._connector()
        if conn is None:
            return [f"no connector {self.provider!r}; installed: {', '.join(connectors()) or 'none'}"]
        errors = [] if self.codes.strip() else [f"give the series codes ({conn.codes or 'as the provider writes them'})"]
        if "?" in self.codes:
            errors.append("give other parameters under Options, not after ? in the codes")
        for label, value, param in (("a first period", self.start, conn.start), ("a last period", self.end, conn.end)):
            if value.strip() and not param:
                errors.append(f"the {self.provider} connector takes no {label}: leave it empty")
        bad = [p for p in self.options.split("&") if p.strip() and "=" not in p]
        errors += [f"write each option as name=value, not {p.strip()!r}" for p in bad]
        if self.vintage.strip():
            if not conn.vintage:
                errors.append(f"the {self.provider} connector keeps no vintages: leave Vintage empty for the latest data")
            else:
                try:
                    pd.Timestamp(self.vintage.strip())
                except ValueError:
                    errors.append(f"the vintage is a date such as 2020-01-30, not {self.vintage!r}")
        return errors

    def to_call(self) -> Any:
        from ..calls import make_call

        return make_call("pymetrix:dataset.read", output=self.name, recorded_by=self.command_id, source=self.address(),
                         vintage=self.vintage.strip() or "latest")

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
