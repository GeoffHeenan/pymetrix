"""Seasonal adjustment and calendar regressor dialogs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from ..forms import build
from ..plugins import registry
from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session


# in pymetrix-seasonal: a project that used one says so when it is missing (names, class names and labels)
_PLUGIN_METHODS = {"x13", "x-13arima-seats", "tramoseats", "tramo-seats"}


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


@command(menu="Data/Seasonal adjustment...")
@dataclass(frozen=True)
class SeasonalAdjust(Command):
    """Seasonally adjust monthly or quarterly series with STL, X-13ARIMA-SEATS or classical decomposition.

    `<series>_sa` (adjusted) and `<series>_seasonal` (the seasonal factor, or component for additive adjustment)
    are added to the dataset. Positive series are adjusted multiplicatively unless you choose otherwise. With Keep
    updated they are derived series, recomputed when the original changes (a new release revises the adjustment).
    X-13ARIMA-SEATS and TRAMO-SEATS come with the pymetrix-seasonal plugin, and need the Census Bureau's x13as program
    and Java with JDemetra+ respectively.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Classical, Dataset

        data = Dataset.example("sample_monthly")
        data = data.seasonal_adjust("retail")      # STL; adds retail_sa and retail_seasonal, kept up to date
        data["retail_seasonal"]                    # retail is positive, so the seasonal factor is a ratio near 1
        data.seasonal_adjust(["ip", "emp"], method=Classical())   # several series, by classical decomposition
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("{series}_sa", "the seasonally adjusted series, added to the dataset (or to the new one named in Name)"),
        ("{series}_seasonal", ("multiplicative: a factor near 1, series / factor = adjusted; additive: a component near "
                               "0, series - component = adjusted")),
        ("Variables tab", "derived_from names the method and its settings; 'kept up to date' is yes with Keep updated"),
    )

    topics: ClassVar[tuple[str, ...]] = ("seasonal-adjustment", "derived-series")

    variables: Annotated[str, Widget("series", depends_on="data", multiple=True, help="monthly or quarterly series to adjust")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    method: Annotated[str, Widget("extension", source="seasonal", help="STL, X-13ARIMA-SEATS or classical")] = "STL"
    settings: Annotated[str, Widget("options", depends_on="method", source="seasonal", group="Method settings")] = ""
    keep_updated: Annotated[bool, Widget("checkbox", help="recompute when the series changes, so a new release also revises the adjusted history")] = True
    name: Annotated[str, Widget("text", help="a new dataset for the result; empty = add to the dataset itself")] = ""

    def _method(self) -> Any:
        methods = registry("seasonal")
        if self.method.lower() in _PLUGIN_METHODS and methods.find(self.method) is None:
            from ..errors import PluginUnavailable

            raise PluginUnavailable(f"pymetrix-seasonal: {self.method} is in the seasonal-adjustment plugin: "
                                    'pip install "pymetrix-seasonal[jdemetra]"')
        return build(methods.get(self.method), self.settings)

    def validate(self, session: Session) -> list[str]:
        if not _names(self.variables):
            return ["choose the series to adjust"]
        from ..errors import PluginUnavailable

        try:
            self._method()
        except PluginUnavailable:
            raise  # its plugin is missing: a project that used it keeps the step, unrun, and names the plugin
        except (KeyError, TypeError, ValueError) as exc:
            return [str(exc.args[0]) if exc.args else str(exc)]
        return [] if self.data in session.objects else [f"no dataset named {self.data!r}"]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        names = _names(self.variables)
        return make_call("pymetrix:dataset.seasonal_adjust", Ref(self.data), output=self.name or self.data, recorded_by=self.command_id,
                         names=names[0] if len(names) == 1 else names, method=self._method(), derived=self.keep_updated)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Data/Calendar regressors...")
@dataclass(frozen=True)
class CalendarRegressors(Command):
    """Add trading-day, Easter and leap-year regressors (X-13ARIMA-SEATS definitions) for use in equations.

    Six trading-day contrasts (td_mon ... td_sat: each weekday's count less Sundays), or one (td_weekdays: weekdays
    less 5/2 of weekend days); easter_w, the share of the w days before Easter in the period; leap_year.
    """

    topics: ClassVar[tuple[str, ...]] = ("seasonal-adjustment",)

    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    trading_days: Annotated[Literal["six", "one", "none"], Widget("choice", help="six contrasts, one weekday contrast, or none")] = "six"
    easter: Annotated[int, Widget("number", min=0, max=25, help="days before Easter in the window; 0 = no Easter regressor")] = 8
    leap_year: Annotated[bool, Widget("checkbox", help="add a regressor for the extra day in February of leap years (monthly and quarterly data)")] = True

    def validate(self, session: Session) -> list[str]:
        return [] if self.data in session.objects else [f"no dataset named {self.data!r}"]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:dataset.calendar", Ref(self.data), output=self.data, recorded_by=self.command_id,
                         trading_days=self.trading_days, easter=self.easter or None, leap_year=self.leap_year)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
