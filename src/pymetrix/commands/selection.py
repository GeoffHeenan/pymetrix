"""Dialogs for selections of series: select series, then derive, estimate or test for each.

Each records one step whose code is the loop, e.g.
``fits = selection.fit('d{s} ~ d{s}[-1]', data)``. Templates use ``{s}`` for the
series name.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from ..errors import SpecificationError
from ..forms import parse_options
from ..plugins import registry
from .registry import Command, Widget, command, existing_series

if TYPE_CHECKING:
    from .session import Session

_TOPICS = ("selections-and-loops",)
_TEMPLATE_HELP = "use {s} for the series name, e.g. d{s} ~ d{s}[-1]"


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


_SERIES_HELP = "tick the series to loop over (or choose a named selection below)"
_SELECTION_HELP = "a selection made with Data > Select series; instead of ticking series"


class _Loop:
    """What the loop dialogs share: the series to loop over, ticked here or a named selection."""

    def _members_error(self, session: Session) -> list[str]:
        if self.selection and _names(self.series):  # type: ignore[attr-defined]
            return ["tick series or choose a selection, not both"]
        if self.selection:  # type: ignore[attr-defined]
            return [] if self.selection in session.objects else [f"no selection named {self.selection!r}"]  # type: ignore[attr-defined]
        if not _names(self.series):  # type: ignore[attr-defined]
            return ["tick the series to loop over, or choose a selection"]
        return [] if self.data in session.objects else [f"no dataset named {self.data!r}"]  # type: ignore[attr-defined]

    def _selection(self, session: Session) -> Any:
        if self.selection:  # type: ignore[attr-defined]
            return session[self.selection]  # type: ignore[attr-defined]
        return session[self.data].series(*_names(self.series))  # type: ignore[attr-defined]

    def _selection_code(self) -> str:
        if self.selection:  # type: ignore[attr-defined]
            return self.selection  # type: ignore[attr-defined]
        return f"{self.data}.series({', '.join(map(repr, _names(self.series)))})"  # type: ignore[attr-defined]

    def _selection_call(self) -> Any:
        """The selection a loop's call runs on: the named selection, or ``data.series(...)`` of the ticked series."""
        from ..calls import make_call
        from ..codec import Ref

        if self.selection:  # type: ignore[attr-defined]
            return Ref(self.selection)  # type: ignore[attr-defined]
        return make_call("pymetrix:dataset.series", Ref(self.data), patterns=tuple(_names(self.series)))  # type: ignore[attr-defined]


@command(menu="Data/Select series...")
@dataclass(frozen=True)
class SelectSeries(Command):
    """Select series by name, pattern or both, to derive, estimate or test each of them in one step.

    Tick series, give patterns (`*_cycle` for every cycle, `d*` for every series starting with d), or both.
    In code, series can also be selected by metadata: `data.series(units="USD bn")`.
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS

    series: Annotated[str, Widget("series", depends_on="data", multiple=True, help="series to include by name; added to any that match the patterns")] = ""
    patterns: Annotated[str, Widget("text", help="comma-separated name patterns, e.g. *_cycle, d*")] = ""
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    name: Annotated[str, Widget("text", help="name of the selection in the workspace")] = "selection"

    def validate(self, session: Session) -> list[str]:
        if not (_names(self.series) or _names(self.patterns)):
            return ["tick some series or give a pattern"]
        if not self.name.isidentifier():
            return [f"{self.name!r} is not a valid name"]
        return [] if self.data in session.objects else [f"no dataset named {self.data!r}"]

    def _members(self) -> list[str]:
        return _names(self.series) + _names(self.patterns)

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:dataset.series", Ref(self.data), output=self.name, recorded_by=self.command_id,
                         patterns=tuple(self._members()))

    def run(self, session: Session) -> Any:
        selection = self.to_call().invoke(session.objects)
        if not len(selection):
            raise SpecificationError("no series match")
        session.objects[self.name] = selection
        return selection

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Data/Derived series for each selected series...")
@dataclass(frozen=True)
class DeriveEach(_Loop, Command):
    """Add a series for each selected series, from templates: `d{s}` = `dlog({s})` gives dc, di, dg.

    Tick the series here, or choose a selection made with Data > Select series. With Keep updated (the default)
    they are derived series that follow their inputs; otherwise their values are computed once.
    """

    topics: ClassVar[tuple[str, ...]] = (*_TOPICS, "derived-series")

    name: Annotated[str, Widget("text", help="name template for the new series, e.g. d{s}")]
    formula: Annotated[str, Widget("text", help="formula template, e.g. dlog({s}) or {s} / y * 100")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    series: Annotated[str, Widget("series", depends_on="data", multiple=True, help=_SERIES_HELP)] = ""
    selection: Annotated[str, Widget("object", source="Selection", help=_SELECTION_HELP)] = ""
    keep_updated: Annotated[bool, Widget("checkbox", help="keep the formulas, so the series follow their inputs")] = True
    replace: Annotated[bool, Widget("checkbox", help="replace a series of the same name, if there is one")] = False

    schema_version: ClassVar[int] = 2  # 2: replace (schema-1 steps replaced silently)

    def validate(self, session: Session) -> list[str]:
        if "{s}" not in self.name:
            return ["the name template needs {s}, or every series would get the same name"]
        errors = self._members_error(session)
        if errors:
            return errors
        if self.data not in session.objects:
            return [f"no dataset named {self.data!r}"]
        return existing_series(session, self.data, list(self._selection(session).formulas(self.name, self.formula)), self.replace)

    def to_call(self, session: Session) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        formulas = self._selection(session).formulas(self.name, self.formula)  # one recipe per series, written out
        verb = "derive" if self.keep_updated else "with_series"
        return make_call(f"pymetrix:dataset.{verb}", Ref(self.data), output=self.data, recorded_by=self.command_id, **formulas)

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        verb = "derive" if self.keep_updated else "with_series"
        return f"{self.data} = {self.data}.{verb}(**{self._selection_code()}.formulas({self.name!r}, {self.formula!r}))"


@command(menu="Estimate/Equation for each selected series...")
@dataclass(frozen=True)
class FitEach(_Loop, Command):
    """Estimate the same equation for each selected series, with `{s}` standing for the series.

    For example `d{s} ~ d{s}[-1] + dlog(y)` estimates one equation per series. Tick the series here, or choose a
    selection made with Data > Select series. The result shows the coefficients side by side, with rows lined up
    across series (`d{s}[-1]`), and the fit statistics.
    """

    topics: ClassVar[tuple[str, ...]] = (*_TOPICS, "least-squares")

    formula: Annotated[str, Widget("formula", help=_TEMPLATE_HELP)]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    series: Annotated[str, Widget("series", depends_on="data", multiple=True, help=_SERIES_HELP)] = ""
    selection: Annotated[str, Widget("object", source="Selection", help=_SELECTION_HELP)] = ""
    method: Annotated[str, Widget("extension", source="methods", help="estimation method for every equation")] = "OLS"
    cov: Annotated[Literal["classic", "white", "hac"], Widget("choice")] = "classic"
    start: Annotated[str, Widget("period")] = ""
    end: Annotated[str, Widget("period")] = ""
    name: str = "fits"
    method_settings: Annotated[str, Widget("options", depends_on="method", source="methods",
                                           exclude=("endog", "instruments", "cov", "hac_lags"), group="Method settings")] = ""
    keep_going: Annotated[bool, Widget("checkbox", advanced=True, help="if one equation fails, estimate the rest and list the failure")] = False

    def validate(self, session: Session) -> list[str]:
        errors = self._members_error(session)
        if "{s}" not in self.formula:
            errors.append("the formula needs {s} where the series goes")
        entry = registry("methods").find(self.method)
        if entry is None:
            errors.append(f"unknown method {self.method!r}")
        elif entry.kind == "iv":
            errors.append("instrumental-variable methods need instruments for each equation; write the loop in code")
        return errors

    def _method(self) -> Any:
        entry = registry("methods").get(self.method)
        _, kw = parse_options(self.method_settings)
        if "cov" in {f.name for f in fields(entry.obj)}:
            kw["cov"] = self.cov
        return entry.build(**kw)

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self.start or self.end:
            options["span"] = (self.start or None, self.end or None)
        if self.keep_going:
            options["errors"] = "collect"
        return options

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:selection.fit", self._selection_call(), output=self.name, recorded_by=self.command_id,
                         formula=self.formula, data=Ref(self.data), method=self._method(), **self._options())

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Test/Test each selected series...")
@dataclass(frozen=True)
class TestEach(_Loop, Command):
    """Run the same test on each selected series, e.g. a unit-root test, with the results in one table. Tick the
    series here, or choose a selection made with Data > Select series.

    The table gives each series' statistic, p-value and decision at 5%, under the null hypothesis stated below it.
    The same test settings (lags, deterministic terms) apply to every series, so choose settings that suit them all,
    or test series that need different settings separately. With many series, some will reject at 5% by chance even
    when the null is true for all of them (about one in twenty), so read the decisions as a screen, not as separate
    findings; a multiple-testing correction or a panel test is the way to draw a joint conclusion.
    """

    __test__ = False  # not a pytest test class
    topics: ClassVar[tuple[str, ...]] = (*_TOPICS, "unit-roots")

    test: Annotated[str, Widget("test", depends_on="data")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    series: Annotated[str, Widget("series", depends_on="data", multiple=True, help=_SERIES_HELP)] = ""
    selection: Annotated[str, Widget("object", source="Selection", help=_SELECTION_HELP)] = ""
    options: Annotated[str, Widget("options", depends_on="test", exclude=("variable",), group="Test settings")] = ""
    name: str = "tests"

    def validate(self, session: Session) -> list[str]:
        return self._members_error(session)

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        args, kw = parse_options(self.options)
        return make_call("pymetrix:selection.test", self._selection_call(), output=self.name, recorded_by=self.command_id,
                         data=Ref(self.data), test=self.test, args=tuple(args), **kw)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
