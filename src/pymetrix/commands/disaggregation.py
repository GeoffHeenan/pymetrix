"""The temporal disaggregation and reconciliation dialogs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from ..errors import SpecificationError
from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session


_RHO_METHODS = ("chow-lin", "chow-lin-minrss", "litterman", "litterman-minrss")


@command(menu="Data/Temporal disaggregation...")
@dataclass(frozen=True)
class Disaggregate(Command):
    """Make a quarterly or monthly series consistent with annual (or quarterly) data, from indicators.

    Write the low-frequency series on its high-frequency indicators (gdp ~ ip + emp), or on 1 for none. Chow-Lin,
    Fernandez and Litterman estimate the relation and spread the rest; Denton moves the indicator (or a constant) as
    little as possible, in first or seasonal differences. The result adds up to the low-frequency data (sums for
    flows, averages for rates, first or last values for stocks, or the weighted sums of an aggregation matrix, such
    as calendar- and fiscal-year totals together), exactly unless soft constraints let it miss them, and extends to
    the end of the indicators. Chow-Lin's and Litterman's rho is estimated between 0 and 0.999 unless its range is
    widened; an estimate at an end of the range is noted in the summary.
    """

    topics: ClassVar[tuple[str, ...]] = ("temporal-disaggregation",)

    formula: Annotated[str, Widget("formula", help="e.g. gdp ~ ip, or gdp ~ 1 with no indicator")]
    low: Annotated[str, Widget("object", source="Dataset", help="the low-frequency data (e.g. annual)")]
    high: Annotated[str, Widget("object", source="Dataset", help="the high-frequency data with the indicators (e.g. quarterly)")] = "data"
    method: Annotated[Literal["chow-lin", "chow-lin-minrss", "fernandez", "litterman", "litterman-minrss", "denton", "denton-additive"],
                      Widget("choice", help="Chow-Lin (AR(1) residual), Fernandez (random walk), Litterman (random walk with AR(1) steps), or Denton")] = "chow-lin"
    conversion: Annotated[Literal["sum", "average", "first", "last"], Widget("choice", help="sum for flows, average for rates, first or last for stocks")] = "sum"
    add_to_high: Annotated[bool, Widget("checkbox", help="add the series to the high-frequency data")] = True
    name: str = "disaggregation"
    rho: Annotated[float | None, Widget("number", min=-0.999, max=0.999, step=0.01, advanced=True, show_if=("method", _RHO_METHODS),
                                        help="Chow-Lin and Litterman: fix rho; empty = estimated")] = None
    rho_bounds: Annotated[str, Widget("text", advanced=True, show_if=("method", _RHO_METHODS),
                                      help="Chow-Lin and Litterman: the range rho is estimated in, low, high; the default 0, 0.999 keeps it "
                                           "non-negative, -0.999, 0.999 allows negative rho")] = "0, 0.999"
    differences: Annotated[Literal["first", "seasonal"], Widget("choice", show_if=("method", ("denton", "denton-additive")),
                                                                help="Denton: seasonal differences (lag 4 or 12) keep the indicator's seasonal pattern")] = "first"
    level_penalty: Annotated[float | None, Widget("number", min=0, step=0.001, advanced=True, show_if=("differences", ("seasonal",)),
                                                  help="seasonal Denton: the weight on first differences that pins the pattern; empty = 0.001")] = None
    soft: Annotated[float | None, Widget("number", min=0, advanced=True,
                                         help="soft constraints: the standard deviation of the low-frequency values' error, in their units; "
                                              "empty = met exactly")] = None
    aggregation: Annotated[str, Widget("object", source="Matrix,Dataset", advanced=True, empty="none: use the conversion",
                                       help="an aggregation matrix instead of the conversion: a matrix (rows named as the low-frequency "
                                            "data's, columns as the high-frequency periods) or a high-frequency dataset of weights, "
                                            "one series per low-frequency value")] = ""
    aggregation_file: Annotated[str, Widget("file", source="Tables (*.csv *.xlsx *.xls *.parquet);;All files (*)", advanced=True,
                                            help="the aggregation matrix from a file: a first column of row names, a header of "
                                                 "high-frequency periods")] = ""

    def _spec(self) -> Any:
        from ..disaggregation import TemporalDisaggregation

        try:
            bounds = tuple(float(b) for b in self.rho_bounds.replace(";", ",").split(",") if b.strip())
        except ValueError:
            bounds = ()
        if len(bounds) != 2 or not -1 < bounds[0] < bounds[1] < 1:
            raise SpecificationError(f"rho's range is two numbers between -1 and 1, low first, not {self.rho_bounds!r}")
        extra: dict[str, Any] = {"rho": self.rho} if self.rho is not None else {}
        if bounds != (0.0, 0.999):
            extra["rho_bounds"] = bounds
        if self.soft is not None:
            extra["soft"] = self.soft
        if self.method.startswith("denton") and self.differences != "first":
            extra["differences"] = self.differences
            if self.level_penalty is not None:
                extra["level_penalty"] = self.level_penalty
        if self.aggregation and self.aggregation_file:
            raise SpecificationError("give the aggregation matrix as a workspace object or a file, not both")
        if (self.aggregation or self.aggregation_file) and self.conversion != "sum":
            raise SpecificationError("an aggregation matrix replaces the conversion: leave conversion at sum and put the weights in the matrix")
        return TemporalDisaggregation(self.formula, self.method, self.conversion, **extra)

    def _aggregation(self) -> str:
        """The aggregation argument as code: a workspace name, a file path, or nothing."""
        if self.aggregation:
            return f", aggregation={self.aggregation}"
        return f", aggregation={self.aggregation_file!r}" if self.aggregation_file else ""

    def validate(self, session: Session) -> list[str]:
        try:
            self._spec()
        except ValueError as exc:
            return [str(exc)]
        missing = [n for n in (self.low, self.high) if n not in session.objects]
        problems = [f"no dataset named {n!r}" for n in missing]
        if self.aggregation and self.aggregation not in session.objects:
            problems.append(f"no matrix or dataset named {self.aggregation!r}")
        return problems

    def run(self, session: Session) -> Any:
        aggregation = session[self.aggregation] if self.aggregation else (self.aggregation_file or None)
        fit = self._spec().fit(session[self.low], session[self.high], aggregation=aggregation)
        session.objects[self.name] = fit
        if self.add_to_high:
            target = fit.series.name
            session.objects[self.high] = session[self.high].with_series(**{target: fit.series})
        return fit

    def to_code(self) -> str:
        from .builtin import _code

        lines = [f"{self.name} = {_code(self._spec())}.fit({self.low}, {self.high}{self._aggregation()})"]
        if self.add_to_high:
            target = self.formula.split("~")[0].strip()
            lines.append(f"{self.high} = {self.high}.with_series({target}={self.name}.series)")
        return "\n".join(lines)


@command(menu="Data/Reconcile components...")
@dataclass(frozen=True)
class Reconcile(Command):
    """Make components (often disaggregated series) add up to their high-frequency total, keeping their own totals.

    Each period the components are moved, as little as the method allows, until they add up to the total; each
    component with a series of the same name in the low-frequency data still adds up to it; fixed components are not
    moved. Denton moves the components' first differences as little as possible, in proportion to their size or in
    their units (Di Fonzo and Marini's multivariate Denton); least squares spreads each period's gap in proportion
    to the values; weighted spreads it in proportion to the components' weights, whatever their levels and signs.
    The summary lists the size of each component's adjustment and the gaps before and after.
    """

    topics: ClassVar[tuple[str, ...]] = ("temporal-disaggregation",)

    components: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the series that should add up to the total")]
    total: Annotated[str, Widget("series", depends_on="data", help="their high-frequency total")]
    data: Annotated[str, Widget("object", source="Dataset", help="the high-frequency data with the components and the total")] = "data"
    low: Annotated[str, Widget("object", source="Dataset", empty="none: only the total",
                               help="low-frequency data with a series for each component whose totals must hold")] = ""
    fixed: Annotated[str, Widget("series", depends_on="data", multiple=True, help="components left as they are")] = ""
    method: Annotated[Literal["denton", "denton-additive", "least-squares", "weighted"],
                      Widget("choice", help="Denton: smooth adjustments (proportional or additive); least squares: in proportion to the values; "
                                            "weighted: in proportion to the weights")] = "denton"
    conversion: Annotated[Literal["sum", "average", "first", "last"], Widget("choice", help="how the low-frequency values add up")] = "sum"
    weights: Annotated[str, Widget("text", advanced=True, help="how much each component moves, e.g. cons=1, inv=4 (larger moves more; "
                                                               "empty = equally)")] = ""
    replace: Annotated[bool, Widget("checkbox", help="replace the components in the data with the reconciled series")] = True
    name: str = "reconciliation"

    def _weights(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for part in self.weights.replace(";", ",").split(","):
            if not part.strip():
                continue
            key, sep, value = part.partition("=")
            try:
                out[key.strip()] = float(value)
            except ValueError:
                raise SpecificationError(f"weights are name=number pairs, e.g. cons=1, inv=4, not {part.strip()!r}") from None
            if not sep or not key.strip():
                raise SpecificationError(f"weights are name=number pairs, e.g. cons=1, inv=4, not {part.strip()!r}")
        return out

    def _names(self, value: str) -> list[str]:
        from ..reconciliation import _names

        return _names(value)

    def _kwargs(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.fixed.strip():
            out["fixed"] = self._names(self.fixed)
        if self.method != "denton":
            out["method"] = self.method
        if self.conversion != "sum":
            out["conversion"] = self.conversion
        if self._weights():
            out["weights"] = self._weights()
        return out

    def validate(self, session: Session) -> list[str]:
        try:
            self._weights()
        except SpecificationError as exc:
            return [str(exc)]
        problems = [f"no dataset named {n!r}" for n in (self.data, self.low) if n and n not in session.objects]
        if len(self._names(self.components)) < 2:
            problems.append("pick two or more components")
        return problems

    def to_call(self) -> Any:
        """The reconcile call (replacing the components in the data, when asked, is a second line of the script)."""
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:reconcile", output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                         components=self._names(self.components), total=self.total, low=Ref(self.low) if self.low else None,
                         **self._kwargs())

    def run(self, session: Session) -> Any:
        from ..reconciliation import reconcile

        low = session[self.low] if self.low else None
        result = reconcile(session[self.data], self._names(self.components), self.total, low, **self._kwargs())
        session.objects[self.name] = result
        if self.replace:
            session.objects[self.data] = session[self.data].with_series(**{c: result.series[c] for c in result.components
                                                                             if c not in result.fixed})
        return result

    def to_code(self) -> str:
        args = [self.data, repr(self._names(self.components)), repr(self.total)]
        if self.low:
            args.append(f"low={self.low}")
        args += [f"{k}={v!r}" for k, v in self._kwargs().items()]
        lines = [f"{self.name} = reconcile({', '.join(args)})"]
        if self.replace:
            moved = [c for c in self._names(self.components) if c not in self._names(self.fixed)]
            lines.append(f"{self.data} = {self.data}.with_series(" + ", ".join(f"{c}={self.name}.series[{c!r}]" for c in moved) + ")")
        return "\n".join(lines)
