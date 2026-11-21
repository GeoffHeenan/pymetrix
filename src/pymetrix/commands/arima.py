"""Dialogs for univariate forecasting models: seasonal ARIMA, the theta method and STL forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from ..errors import SpecificationError
from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session


def _span(start: str, end: str) -> tuple[str | None, str | None] | None:
    return (start or None, end or None) if start or end else None


def _triple(text: str, what: str) -> tuple[int, int, int]:
    """'1, 0, 1' or '1 0 1' → (1, 0, 1)."""
    parts = [p for p in text.replace(",", " ").replace("(", " ").replace(")", " ").split() if p]
    try:
        values = tuple(int(p) for p in parts)
    except ValueError:
        raise SpecificationError(f"{what}: three whole numbers, such as 1, 0, 1") from None
    if len(values) != 3 or min(values) < 0:
        raise SpecificationError(f"{what}: three whole numbers, none negative, such as 1, 0, 1")
    return values  # type: ignore[return-value]


def _names(text: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in text.replace("\n", ",").split(",") if s.strip())


def _backend_options(text: str) -> dict[str, Any]:
    """'enforce_stationarity=False, fit.maxiter=200' → {"enforce_stationarity": False, "fit.maxiter": 200}."""
    import re

    from ..forms import parse_options

    args, options = parse_options(re.sub(r"\b(model|fit)\.(\w+)\s*=", r"\1__\2=", text))
    if args:
        raise SpecificationError(f"other statsmodels arguments are name=value pairs, not {args[0]!r}")
    return {k.replace("__", ".", 1) if k.startswith(("model__", "fit__")) else k: v for k, v in options.items()}


_OTHER = Widget("text", advanced=True, help="other statsmodels arguments, e.g. model.enforce_stationarity=False, fit.maxiter=200")


class _Fit(Command):
    """A model fitted to one series of a dataset, recorded as the API call ``call_id``."""

    call_id: ClassVar[str] = ""

    def _spec(self) -> Any:
        raise NotImplementedError

    def validate(self, session: Session) -> list[str]:
        try:
            self._spec()
        except SpecificationError as exc:
            return [str(exc)]
        return []

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call(self.call_id, self._spec(), output=self.name, recorded_by=self.command_id,  # type: ignore[attr-defined]
                         data=Ref(self.data), span=_span(self.start, self.end))  # type: ignore[attr-defined]

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/Seasonal ARIMA...")
@dataclass(frozen=True)
class FitSARIMA(_Fit):
    """Seasonal ARIMA(p, d, q)(P, D, Q)s, with optional regressors, by exact maximum likelihood (statsmodels' SARIMAX).

    The series is differenced d times and seasonally D times (at the season length s: 4 for quarters, 12 for
    months); AR and MA terms at lags 1 to p and q, and seasonal ones at multiples of s, model what is left. The
    airline model of Box and Jenkins is (0, 1, 1)(0, 1, 1) on the log of the series. Forecast from the result with
    Forecast, or evaluate it with Forecast → Backtest; regressors need their values for the forecast periods.

    Choose d and D from the data, not the fit: a unit-root test or the correlogram should suggest differencing,
    and too much differencing leaves a non-invertible MA root near one. Each difference also changes the forecasts'
    long-run behaviour: with d = 1 the forecasts settle at a level (or on a trend, with a drift term) and the
    forecast intervals widen without limit. Keep p and q small and compare candidates by information criteria
    and residual diagnostics (the residuals should look like white noise); a better in-sample fit from more terms
    often forecasts worse.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import SARIMA, Dataset

        data = Dataset.example("sample_monthly").derive(lip="100*log(ip)")
        arima = SARIMA("lip", order=(1, 1, 0)).fit(data)    # ARIMA(1, 1, 0): an AR(1) in the monthly changes
        print(arima.summary())         # estimates by exact maximum likelihood; log likelihood, AIC, BIC
        arima.forecast(data, ("2025-04", "2026-03"))        # forecast, se, lower and upper for each month
        SARIMA("lip", order=(2, 1, 0)).fit(data)            # an AR(2) instead: compare the two by AIC and BIC
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Summary", ("AR(k), MA(k) and seasonal terms with z and p-values, and the shocks' variance; the notes give the "
                     "log likelihood, AIC and BIC to compare orders")),
        ("Residuals", "the series, its one-step-ahead fit and the residual, which should look like white noise"),
        ("Graph", "the series and its one-step-ahead fit"),
        ("arima.forecast(data, span)", ("forecast, se, lower and upper for each period (as the Forecast dialog gives); "
                                        "with d = 1 the intervals keep widening")),
    )

    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow",)
    call_id: ClassVar[str] = "pymetrix:sarima.fit"

    series: Annotated[str, Widget("series", depends_on="data", help="the series to model, often in logs; ARIMA differences it as d and D say")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    order: Annotated[str, Widget("text", help="p, d, q: AR order, differences, MA order")] = "1, 0, 0"
    seasonal_order: Annotated[str, Widget("text", help="P, D, Q: seasonal AR order, seasonal differences, seasonal MA order")] = "0, 0, 0"
    exog: Annotated[str, Widget("series", depends_on="data", multiple=True, help="regressors (their future values are needed to forecast)")] = ""
    trend: Annotated[Literal["auto", "n", "c", "t", "ct"], Widget("choice", help="auto: a constant without differencing, none with it; c constant, t trend, ct both")] = "auto"
    period: Annotated[int, Widget("spin", min=0, max=366, advanced=True, help="the season length; 0 = from the frequency (4 or 12)")] = 0
    start: Annotated[str, Widget("period", group="Sample", help="first period of the sample (empty = the first available)")] = ""
    end: Annotated[str, Widget("period", group="Sample", help="last period of the sample (empty = the last available)")] = ""
    name: str = "sarima"
    backend_options: Annotated[str, _OTHER] = ""

    def _spec(self) -> Any:
        from ..arima import SARIMA

        return SARIMA(self.series, _triple(self.order, "order"), _triple(self.seasonal_order, "seasonal order"), self.period,
                      _names(self.exog), self.trend, _backend_options(self.backend_options))


@command(menu="Estimate/Theta method...")
@dataclass(frozen=True)
class FitTheta(_Fit):
    """The theta method: simple exponential smoothing plus half the slope of a linear trend, after a seasonal
    adjustment when the series is seasonal (statsmodels' ThetaModel).

    A simple method that did well in forecasting competitions (M3), and so a benchmark worth beating. Forecast from
    the result with Forecast, or evaluate it with Forecast → Backtest.

    The forecast extends the series' smoothed recent level by half the slope of its long-run linear trend, so it suits
    smooth series with a steady drift and does poorly after a break in trend. It has almost nothing to choose, which
    is its strength as a benchmark: a more elaborate model that cannot beat it in a backtest is not adding much.
    """

    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow",)
    call_id: ClassVar[str] = "pymetrix:theta.fit"

    series: Annotated[str, Widget("series", depends_on="data", help="the series to forecast, in levels; a steady drift suits the method")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    deseasonalize: Annotated[Literal["auto", "yes", "no"], Widget("choice", help="auto: adjust when a test finds seasonality")] = "auto"
    period: Annotated[int, Widget("spin", min=0, max=366, advanced=True, help="the season length; 0 = from the frequency")] = 0
    method: Annotated[Literal["auto", "additive", "multiplicative"], Widget("choice", advanced=True, help="seasonal adjustment: multiplicative when the seasonal swings grow with the level; auto picks by the data")] = "auto"
    start: Annotated[str, Widget("period", group="Sample", help="first period of the sample (empty = the first available)")] = ""
    end: Annotated[str, Widget("period", group="Sample", help="last period of the sample (empty = the last available)")] = ""
    name: str = "theta"
    backend_options: Annotated[str, _OTHER] = ""

    def _spec(self) -> Any:
        from ..arima import Theta

        return Theta(self.series, self.deseasonalize, self.period, self.method, _backend_options(self.backend_options))


@command(menu="Estimate/STL forecasting...")
@dataclass(frozen=True)
class FitSTLForecast(_Fit):
    """STL forecasting: remove the seasonal component by STL, forecast the adjusted series with an ARIMA or
    exponential smoothing model, and add back the last year's seasonal component (statsmodels' STLForecast).

    The components are in the result's window. Forecast from the result with Forecast, or evaluate it with
    Forecast → Backtest.

    Separating the seasonal pattern lets a simple non-seasonal model handle the trend and cycle, which is often more
    robust than a seasonal ARIMA when the seasonal pattern changes slowly over time. The seasonal component is
    carried forward unchanged from the last year, so the forecasts assume the latest seasonal pattern persists.
    STL is additive: for a series whose seasonal swings grow with its level, model its log.
    """

    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow", "seasonal-adjustment")
    call_id: ClassVar[str] = "pymetrix:stl_forecast.fit"

    series: Annotated[str, Widget("series", depends_on="data", help="a seasonal series to forecast; log it if its seasonal swings grow with its level")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    model: Annotated[Literal["arima", "ets"], Widget("choice", help="the model of the seasonally adjusted series")] = "arima"
    order: Annotated[str, Widget("text", show_if=("model", ("arima",)), help="p, d, q of the ARIMA for the adjusted series; d = 1 for a trending series")] = "1, 1, 0"
    trend: Annotated[Literal["none", "add"], Widget("choice", show_if=("model", ("ets",)), help="trend of the exponential smoothing for the adjusted series: none, or additive (Holt)")] = "add"
    damped: Annotated[bool, Widget("checkbox", show_if=("model", ("ets",)), help="damp the trend towards flat, which usually forecasts better beyond a few periods")] = True
    period: Annotated[int, Widget("spin", min=0, max=366, advanced=True, help="the season length; 0 = from the frequency (4 or 12)")] = 0
    start: Annotated[str, Widget("period", group="Sample", help="first period of the sample (empty = the first available)")] = ""
    end: Annotated[str, Widget("period", group="Sample", help="last period of the sample (empty = the last available)")] = ""
    name: str = "stl"
    backend_options: Annotated[str, _OTHER] = ""

    def _spec(self) -> Any:
        from ..arima import STLForecast

        return STLForecast(self.series, self.model, _triple(self.order, "order"), self.trend, self.damped, self.period,
                           _backend_options(self.backend_options))
