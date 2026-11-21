"""Seasonal ARIMA, the theta method and STL forecasting: univariate forecasting models estimated by statsmodels.

    sarima = SARIMA("lair", order=(0, 1, 1), seasonal_order=(0, 1, 1)).fit(data)    # the airline model
    sarima.summary(); sarima.forecast(data, ("1961-01", "1962-12"))
    Theta("y").fit(data).forecast(horizon=8)                                      # a strong, simple benchmark
    STLForecast("sales", model="arima", order=(1, 1, 0)).fit(data)                # STL, then a model of the adjusted series
    backtest(sarima, data, ("1957-01", "1960-12"))                                # re-estimated at each origin

``SARIMA`` is the seasonal ARIMA(p, d, q)(P, D, Q)s with optional regressors (SARIMAX), estimated by exact maximum
likelihood through the Kalman filter; the period ``s`` comes from the data's frequency (4 or 12) unless given, and
forecasts with regressors take the regressors' values for the forecast periods from the data. ``Theta`` is the theta
method of Assimakopoulos and Nikolopoulos (2000), as simple exponential smoothing with drift (Hyndman and Billah,
2003), after a multiplicative seasonal adjustment when the data are seasonal. ``STLForecast`` removes the seasonal
component by STL, forecasts the adjusted series with an ARIMA or ETS model, and adds back the seasonal component of
the last full year. See Hyndman and Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.), chapters 8-9.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd

from pymetrix._lazy import stats as sps

from ._api import api
from .adapters import require
from .adapters import statsmodels as statsmodels_adapter
from .adapters.base import (
    AdapterSpec,
    BackendOptions,
    check_backend_options,
    with_backend,
)
from .errors import SpecificationError
from .forms import Widget
from .output import Table
from .plugins import register
from .results import BackendResult, deprecated_alias

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["SARIMA", "STLForecast", "Theta", "FittedSARIMA", "FittedSTLForecast", "FittedTheta"]


def _season_length(index: pd.PeriodIndex, period: int) -> int:
    from .filters import periods_per_year

    if period:
        return int(period)
    s = periods_per_year(index.freqstr)
    if s not in (4, 12):
        raise SpecificationError("give the seasonal period (period=...): only quarterly and monthly data have a default")
    return s


def _unbroken(data: Dataset, name: str, span: Any, what: str, needed: int) -> pd.Series:
    """The series on the sample's periods, from its first to its last observation, which must be unbroken."""
    if data.kind != "time_series":
        raise SpecificationError(f"{what} needs a single time series")
    if name not in data.names:
        raise SpecificationError(f"no series {name!r} in the dataset")
    y = data[name][np.asarray(data.sample_mask(span))]
    first, last = y.first_valid_index(), y.last_valid_index()
    if first is None:
        raise SpecificationError(f"{name} has no observations in the sample")
    y = y.loc[first:last]
    if y.isna().any():
        raise SpecificationError(f"{name} has missing values inside the sample: {what} needs an unbroken series")
    if len(y) < needed:
        raise SpecificationError(f"fewer than {needed} observations for this model")
    return y


def _forecast_periods(last: pd.Period, span: Any, horizon: int | None, what: str) -> tuple[pd.Period, pd.Period]:
    if span is not None:
        from .data.sample import as_span

        span = as_span(span)  # '1990Q1:2019Q4' or a pair
        start, end = pd.Period(str(span[0]), freq=last.freq), pd.Period(str(span[1]), freq=last.freq)
    else:
        start, end = last + 1, last + int(horizon or 8)
    if start <= last:
        raise SpecificationError(f"{what} forecasts start after the sample, from {last + 1}")
    return start, end


def _forecast_dataset(fit: Any, mean: Any, se: Any, start: pd.Period, end: pd.Period, data: Any, alpha: float, note: str) -> Any:
    """The forecast as a dataset: forecast, se, lower, upper (a 1 - alpha interval), and actual where the data have it."""
    from .data import Dataset
    from .metadata import DatasetMeta

    last = fit.data.index[-1]
    periods = pd.period_range(last + 1, end, freq=last.freq)
    mean, se = np.asarray(mean, float), np.asarray(se, float)
    z = sps.norm.ppf(1 - alpha / 2)
    frame = pd.DataFrame({"forecast": mean, "se": se, "lower": mean - z * se, "upper": mean + z * se}, index=periods)
    frame = frame.loc[start:end]
    frame.index.name = fit.data.index.name or "time"
    if data is not None and fit.dependent in getattr(data, "names", []):
        frame["actual"] = data[fit.dependent].reindex(frame.index).to_numpy(float)
    meta = DatasetMeta(title=f"{fit.spec.describe()} forecast of {fit.dependent}",
                       description=f"Point forecasts with {1 - alpha:.0%} intervals ({note}).")
    return Dataset.from_frame(frame, meta=meta)


def _stats(result: Any, k: int | None = None) -> dict[str, float]:
    """Fit statistics in pymetrix's definitions (``estimation.criteria``); statsmodels' own as ``*_backend``."""
    from .estimation import criteria

    n, llf = int(result.nobs), float(result.llf)
    k = len(result.params) if k is None else k
    return {"nobs": float(n), "k": float(k), "loglike": llf, "aic": criteria.aic(llf, k), "bic": criteria.bic(llf, k, n),
            "aic_backend": float(result.aic), "bic_backend": float(result.bic)}


def _param_label(name: str) -> str:
    """statsmodels' ARIMA parameter names, as read: ar.L1 → AR(1), ma.S.L12 → seasonal MA(12), sigma2 → variance."""
    parts = name.split(".")
    if parts[0] in ("ar", "ma") and parts[-1].startswith("L"):
        kind = parts[0].upper()
        return f"seasonal {kind}({parts[-1][1:]})" if "S" in parts[1:-1] else f"{kind}({parts[-1][1:]})"
    return {"intercept": "constant", "drift": "trend", "sigma2": "variance"}.get(name, name)


# -- seasonal ARIMA ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SARIMA:
    """Seasonal ARIMA(p, d, q)(P, D, Q)s with optional regressors, by exact maximum likelihood (statsmodels' SARIMAX).

    The series is differenced d times and seasonally D times; AR and MA terms model what is left, at lags 1 to p
    and q and at multiples of the season length s.

    Difference only as much as needed to make the series stationary (d = 1 for a random walk, D = 1 for a
    seasonal unit root); differencing a stationary series adds a non-invertible MA root and wider intervals.
    Choose p and q from the autocorrelations of the differenced series or by information criteria, keep the
    model small, and check that the residuals are white noise. With ``trend="auto"`` a differenced model has no
    constant, so its forecasts have no drift: set the trend for a series that grows. With regressors the model is
    a regression with SARIMA errors, and forecasts need the regressors' future values. Forecast intervals widen
    with the horizon, without bound once the series is differenced; they assume the model and its parameters.
    """

    label: ClassVar[str] = "Seasonal ARIMA"
    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow",)
    requires: ClassVar[tuple[str, ...]] = ("statsmodels",)
    wraps: ClassVar[str] = "statsmodels.tsa.statespace.sarimax.SARIMAX"

    series: Annotated[str, Widget("text", help="the series, unbroken in the sample; let d difference it, so forecasts are in its levels")]
    order: Annotated[tuple[int, int, int], Widget("text", help="(p, d, q): AR order, differences, MA order")] = (1, 0, 0)
    seasonal_order: Annotated[tuple[int, int, int], Widget("text", help="(P, D, Q): seasonal AR order, seasonal differences, seasonal MA order")] = (0, 0, 0)
    period: Annotated[int, Widget("spin", help="the season length s; 0 = from the frequency (4 or 12)")] = 0
    exog: Annotated[tuple[str, ...], Widget("text", help="regressors (their future values are needed to forecast)")] = ()
    trend: Annotated[Literal["auto", "n", "c", "t", "ct"], Widget("choice", help="auto: a constant without differencing, none with it; c constant, t trend, ct both")] = "auto"
    backend_options: Annotated[BackendOptions, Widget("", advanced=True, help="other statsmodels arguments; model.x for the constructor")] = field(default_factory=dict)

    adapter_spec: ClassVar[AdapterSpec] = statsmodels_adapter.SARIMAX

    def __post_init__(self) -> None:
        object.__setattr__(self, "order", tuple(int(v) for v in self.order))
        object.__setattr__(self, "seasonal_order", tuple(int(v) for v in self.seasonal_order))
        object.__setattr__(self, "exog", (self.exog,) if isinstance(self.exog, str) else tuple(self.exog))
        if len(self.order) != 3 or len(self.seasonal_order) != 3 or min(self.order + self.seasonal_order) < 0:
            raise SpecificationError("order and seasonal_order are three numbers each, none negative: (p, d, q) and (P, D, Q)")
        check_backend_options(self)

    @property
    def seasonal(self) -> bool:
        return any(self.seasonal_order)

    def _trend(self) -> str:
        if self.trend != "auto":
            return self.trend
        return "n" if self.order[1] + self.seasonal_order[1] else "c"

    def describe(self, s: int | None = None) -> str:
        p, d, q = self.order
        text = f"ARIMA({p},{d},{q})"
        if self.seasonal:
            P, D, Q = self.seasonal_order
            text = f"SARIMA({p},{d},{q})({P},{D},{Q}){s or self.period or ''}"
        return text + (f" with {', '.join(self.exog)}" if self.exog else "")

    @api("pymetrix:sarima.fit")
    def fit(self, data: Dataset, span: Any = None) -> FittedSARIMA:
        """Estimate by exact maximum likelihood on the periods of ``span`` (first, last), which must hold an unbroken
        run of the series and its regressors."""
        from ._capture import capture, provenance_for

        sarimax = require("statsmodels.tsa.statespace.sarimax")
        missing = [x for x in self.exog if x not in data.names]
        if missing:
            raise SpecificationError(f"no series {', '.join(map(repr, missing))} in the dataset")
        s = 0
        if self.seasonal:
            probe = data[self.series]
            s = _season_length(pd.PeriodIndex(probe.index), self.period)
            if s < 2:
                raise SpecificationError("the seasonal period must be at least 2")
        p, d, q = self.order
        P, D, Q = self.seasonal_order
        needed = max(10, d + D * s + max(p, q) + s * max(P, Q) + 5)
        y = _unbroken(data, self.series, span, "ARIMA", needed)
        X = None
        if self.exog:
            X = data.frame.loc[y.index, list(self.exog)] if data.time_axis is None else pd.DataFrame({x: data[x].reindex(y.index) for x in self.exog})
            if X.isna().any().any():
                raise SpecificationError("the regressors have missing values in the sample")
        endog = pd.Series(y.to_numpy(float), index=y.index.to_timestamp(), name=self.series)
        exog = None if X is None else pd.DataFrame(X.to_numpy(float), index=endog.index, columns=list(self.exog))
        with capture() as caught, statsmodels_adapter.ADAPTER.run(self.adapter_spec, self) as run:
            model = run.call("model", sarimax.SARIMAX, endog, exog=exog, order=self.order,
                             seasonal_order=(P, D, Q, s) if self.seasonal else (0, 0, 0, 0), trend=self._trend())
            result = run.call("fit", model.fit)
        converged = (getattr(result, "mle_retvals", None) or {}).get("converged")
        prov = provenance_for(self, data, [self.series, *self.exog], y.index, y.to_frame(), caught,
                              {"type": "SARIMA", "model": self.describe(s)}, None if converged is None else bool(converged), span=span)
        return FittedSARIMA(self, result, y, s, with_backend(prov, run.record))


@dataclass(frozen=True, eq=False)
class FittedSARIMA(BackendResult):
    """An estimated seasonal ARIMA: its parameters, fit and forecasts."""

    spec: SARIMA
    result: Any = field(repr=False)
    data: pd.Series = field(repr=False)  # the series as estimated, with its periods
    season: int = 0  # the season length used (0: not seasonal)
    provenance: Any = None

    needs_backend: ClassVar[bool] = True  # forecasts call statsmodels' results: kept when pickled; projects recompute it

    @property
    def backend_result(self) -> Any:
        return self.result

    @property
    def dependent(self) -> str:
        return self.spec.series

    target = deprecated_alias("dependent", "target", why="the dependent variable; target is its old name")

    @property
    def params(self) -> Any:
        from .estimation.params import point

        names = [str(n) for n in self.result.param_names]
        try:
            cov = np.asarray(self.result.cov_params(), float)
        except Exception:  # noqa: BLE001 - the covariance needs the Hessian, which can fail
            cov = None
        return point([_param_label(n) for n in names], np.asarray(self.result.params, float), cov,
                     {n: _param_label(n) for n in names})

    @property
    def fitted(self) -> pd.Series:
        """One-step-ahead predictions (the first periods, before the differences are known, are rough)."""
        return pd.Series(np.asarray(self.result.fittedvalues, float), index=self.data.index, name=f"{self.dependent} fitted")

    @property
    def resid(self) -> pd.Series:
        """One-step-ahead forecast errors."""
        return pd.Series(np.asarray(self.result.resid, float), index=self.data.index, name="residual")

    @property
    def stats(self) -> dict[str, float]:
        return {**_stats(self.result), "sigma2": float(np.asarray(self.result.params, float)[list(self.result.param_names).index("sigma2")])}

    def params_table(self) -> Table:
        p = self.params
        se = np.sqrt(np.diag(p.cov)) if p.cov is not None else np.full(len(p.values), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = p.values / se
        rows = [[float(v), float(e), float(t), float(2 * sps.norm.sf(abs(t)))] for v, e, t in zip(p.values, se, z)]
        return Table(["estimate", "std err", "z", "p-value"], rows, index=list(p.names), index_name="parameter",
                     title=f"{self.spec.describe(self.season)} for {self.dependent}")

    def summary(self) -> Table:
        s = self.stats
        notes = [f"Sample: {self.data.index[0]} to {self.data.index[-1]}, {len(self.data)} observations",
                 f"Log likelihood: {s['loglike']:.4f}", f"Akaike info criterion: {s['aic']:.4f}",
                 f"Schwarz criterion: {s['bic']:.4f}",
                 "Exact maximum likelihood (Kalman filter); standard errors from the numerical Hessian."]
        return replace(self.params_table(), notes=tuple(notes))

    @api()
    def forecast(self, data: Any = None, span: tuple[Any, Any] | None = None, alpha: float = 0.05, dynamic: bool = True,
                 horizon: int | None = None) -> Any:
        """Forecasts after the end of the sample, for the periods of ``span`` (or the next ``horizon`` periods), with
        standard errors and a (1 - alpha) interval. With regressors, their values for the forecast periods come from
        ``data``."""
        last = self.data.index[-1]
        start, end = _forecast_periods(last, span, horizon, "ARIMA")
        steps = (end - last).n
        exog = None
        if self.spec.exog:
            if data is None:
                raise SpecificationError("forecasting with regressors needs data with their values for the forecast periods")
            future = pd.period_range(last + 1, end, freq=last.freq)
            exog = pd.DataFrame({x: data[x].reindex(future).to_numpy(float) for x in self.spec.exog})
            if exog.isna().any().any():
                raise SpecificationError(f"the regressors have no values for some forecast periods ({future[0]} to {future[-1]})")
        prediction = self.result.get_prediction(start=len(self.data), end=len(self.data) + steps - 1, exog=exog)
        return _forecast_dataset(self, prediction.predicted_mean, prediction.se_mean, start, end, data, alpha, "Kalman filter variances")

    def plot(self) -> Any:
        """The series and its one-step-ahead fit."""
        from .figures import Figure, Line, Panel

        start = self.spec.order[1] + self.spec.seasonal_order[1] * self.season  # skip the periods lost to differencing
        layers = (Line(self.data, self.dependent, color="#222222"), Line(self.fitted.iloc[start:], "fitted", style="--"))
        return Figure((Panel(layers, title=f"{self.spec.describe(self.season)}: {self.dependent}"),))

    def __repr__(self) -> str:
        return f"<FittedSARIMA {self.spec.describe(self.season)} for {self.dependent}, {len(self.data)} observations>"


# -- the theta method -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Theta:
    """The theta method: simple exponential smoothing plus half the slope of a linear trend (statsmodels' ThetaModel),
    after a multiplicative seasonal adjustment when the series is seasonal.

    A simple, hard-to-beat benchmark for univariate forecasts: the level adapts through exponential smoothing
    while the drift is the fitted linear trend, damped to half its slope. It has no regressors and assumes the
    sample's average trend carries on, so it suits short horizons and steady series better than ones with breaks.
    Alpha near 1 means the level follows the latest observations (close to a random walk); near 0 a long average.
    The intervals come from the exponential smoothing's variance, ignoring the uncertainty in the trend.
    """

    label: ClassVar[str] = "Theta method"
    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow",)
    requires: ClassVar[tuple[str, ...]] = ("statsmodels",)
    wraps: ClassVar[str] = "statsmodels.tsa.forecasting.theta.ThetaModel"

    series: Annotated[str, Widget("text", help="the series to forecast, unbroken in the sample; positive for multiplicative adjustment")]
    deseasonalize: Annotated[Literal["auto", "yes", "no"], Widget("choice", help="auto: adjust when a test finds seasonality")] = "auto"
    period: Annotated[int, Widget("spin", help="the season length; 0 = from the frequency")] = 0
    method: Annotated[Literal["auto", "additive", "multiplicative"], Widget("choice", advanced=True, help="the seasonal adjustment: multiplicative for positive series")] = "auto"
    backend_options: Annotated[BackendOptions, Widget("", advanced=True, help="other statsmodels arguments; model.x for the constructor")] = field(default_factory=dict)

    adapter_spec: ClassVar[AdapterSpec] = statsmodels_adapter.THETA

    def __post_init__(self) -> None:
        check_backend_options(self)

    def describe(self) -> str:
        return "Theta method"

    @api("pymetrix:theta.fit")
    def fit(self, data: Dataset, span: Any = None) -> FittedTheta:
        """Estimate the smoothing parameter and the trend on the periods of ``span``."""
        from ._capture import capture, provenance_for
        from .filters import periods_per_year

        theta = require("statsmodels.tsa.forecasting.theta")
        y = _unbroken(data, self.series, span, "the theta method", 8)
        period = self.period or None
        if period is None:
            try:
                period = periods_per_year(y.index.freqstr)
            except SpecificationError:
                period = 1
        endog = pd.Series(y.to_numpy(float), index=y.index.to_timestamp(), name=self.series)
        with capture() as caught, statsmodels_adapter.ADAPTER.run(self.adapter_spec, self) as run:
            model = run.call("model", theta.ThetaModel, endog, period=period, deseasonalize=self.deseasonalize != "no",
                             use_test=self.deseasonalize == "auto", method=self.method)
            result = run.call("fit", model.fit)
        prov = provenance_for(self, data, [self.series], y.index, y.to_frame(), caught, {"type": "Theta", "model": self.describe()},
                              span=span)
        return FittedTheta(self, result, y, with_backend(prov, run.record))


@dataclass(frozen=True, eq=False)
class FittedTheta(BackendResult):
    """An estimated theta method: the trend slope (b0), the smoothing parameter (alpha) and forecasts."""

    spec: Theta
    result: Any = field(repr=False)
    data: pd.Series = field(repr=False)
    provenance: Any = None

    needs_backend: ClassVar[bool] = True

    @property
    def backend_result(self) -> Any:
        return self.result

    @property
    def dependent(self) -> str:
        return self.spec.series

    target = deprecated_alias("dependent", "target", why="the dependent variable; target is its old name")

    @property
    def params(self) -> Any:
        from .estimation.params import point

        values = self.result.params
        return point(["b0 (trend slope)", "alpha (smoothing)"], [float(values["b0"]), float(values["alpha"])])

    @property
    def stats(self) -> dict[str, float]:
        return {"nobs": float(len(self.data)), "k": 2.0, "sigma2": float(self.result.sigma2)}

    @property
    def deseasonalized(self) -> bool:
        model = self.result.model  # with deseasonalize="auto", whether the test found seasonality
        return bool(getattr(model, "_has_seasonality", model.deseasonalize))

    def summary(self) -> Table:
        p = self.params
        rows = [[float(v)] for v in p.values]
        notes = (f"Sample: {self.data.index[0]} to {self.data.index[-1]}, {len(self.data)} observations",
                 "Forecasts are simple exponential smoothing plus half the slope b0 of a linear trend (theta = 2)"
                 + (", after a seasonal adjustment" if self.deseasonalized else "") + ".")
        return Table(["estimate"], rows, index=list(p.names), index_name="parameter", title=f"Theta method for {self.dependent}", notes=notes)

    @api()
    def forecast(self, data: Any = None, span: tuple[Any, Any] | None = None, alpha: float = 0.05, dynamic: bool = True,
                 horizon: int | None = None) -> Any:
        """Forecasts after the end of the sample, with intervals from the exponential smoothing's variance."""
        last = self.data.index[-1]
        start, end = _forecast_periods(last, span, horizon, "theta")
        steps = (end - last).n
        mean = np.asarray(self.result.forecast(steps), float)
        bands = self.result.prediction_intervals(steps, alpha=alpha)
        se = (np.asarray(bands["upper"], float) - np.asarray(bands["lower"], float)) / (2 * sps.norm.ppf(1 - alpha / 2))
        return _forecast_dataset(self, mean, se, start, end, data, alpha, "simple exponential smoothing variances")

    def plot(self) -> Any:
        from .figures import Figure, Line, Panel

        fc = self.forecast(horizon=8).frame
        return Figure((Panel((Line(self.data, self.dependent, color="#222222"), Line(fc["forecast"], "forecast", style="--")),
                             title=f"Theta method: {self.dependent}"),))

    def __repr__(self) -> str:
        return f"<FittedTheta for {self.dependent}, {len(self.data)} observations>"


# -- STL forecasting --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class STLForecast:
    """STL, then a model of the seasonally adjusted series: its forecasts plus the last year's seasonal component
    (statsmodels' STLForecast).

    STL estimates the seasonal pattern by local regression, letting it change slowly over the years; the model then
    has to capture the trend and dynamics of the adjusted series, so choose it as for a non-seasonal series. The
    forecasts repeat the last seasonal cycle unchanged, and their intervals come from the model of the adjusted
    series alone, treating the seasonal component as known. The decomposition is additive: take logs first when
    the seasonal swings grow with the level of the series.
    """

    label: ClassVar[str] = "STL forecasting"
    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow", "seasonal-adjustment")
    requires: ClassVar[tuple[str, ...]] = ("statsmodels",)
    wraps: ClassVar[str] = "statsmodels.tsa.forecasting.stl.STLForecast"

    series: Annotated[str, Widget("text", help="the series to forecast, unbroken in the sample; log it if its seasonality is proportional")]
    model: Annotated[Literal["arima", "ets"], Widget("choice", help="the model of the seasonally adjusted series")] = "arima"
    order: Annotated[tuple[int, int, int], Widget("text", help="the ARIMA's (p, d, q) for the adjusted series; with d > 0 it has no constant, so no drift")] = (1, 1, 0)
    trend: Annotated[Literal["none", "add"], Widget("choice", help="ETS: an additive trend in the adjusted series, or none (a level only, flat forecasts)")] = "add"
    damped: Annotated[bool, Widget("checkbox", help="ETS: damp the trend, so long-horizon forecasts level off instead of extending the latest slope")] = True
    period: Annotated[int, Widget("spin", help="the season length; 0 = from the frequency (4 or 12)")] = 0
    backend_options: Annotated[BackendOptions, Widget("", advanced=True, help="other statsmodels arguments; model.x for the constructor")] = field(default_factory=dict)

    adapter_spec: ClassVar[AdapterSpec] = statsmodels_adapter.STL_FORECAST

    def __post_init__(self) -> None:
        object.__setattr__(self, "order", tuple(int(v) for v in self.order))
        check_backend_options(self)

    def describe(self) -> str:
        if self.model == "arima":
            return f"STL + ARIMA({','.join(map(str, self.order))})"
        return f"STL + ETS(A,{'Ad' if self.damped else 'A'},N)" if self.trend == "add" else "STL + ETS(A,N,N)"

    @api("pymetrix:stl_forecast.fit")
    def fit(self, data: Dataset, span: Any = None) -> FittedSTLForecast:
        """Decompose by STL and estimate the model of the adjusted series on the periods of ``span``."""
        from ._capture import capture, provenance_for

        stl = require("statsmodels.tsa.forecasting.stl")
        probe = data[self.series] if self.series in data.names else None
        if probe is None:
            raise SpecificationError(f"no series {self.series!r} in the dataset")
        s = _season_length(pd.PeriodIndex(probe.index), self.period)
        y = _unbroken(data, self.series, span, "STL forecasting", 2 * s + 4)
        if self.model == "arima":
            arima = require("statsmodels.tsa.arima.model")
            kind, kwargs = arima.ARIMA, {"order": self.order, "trend": "n" if self.order[1] else "c"}
        else:
            ets = require("statsmodels.tsa.exponential_smoothing.ets")
            kind = ets.ETSModel
            kwargs = {"trend": None if self.trend == "none" else "add", "damped_trend": self.damped and self.trend != "none"}
        endog = pd.Series(y.to_numpy(float), index=y.index.to_timestamp(), name=self.series)
        with capture() as caught, statsmodels_adapter.ADAPTER.run(self.adapter_spec, self) as run:
            model = run.call("model", stl.STLForecast, endog, kind, model_kwargs=kwargs, period=s)
            result = run.call("fit", model.fit, fit_kwargs={"disp": False} if self.model == "ets" else None)
        prov = provenance_for(self, data, [self.series], y.index, y.to_frame(), caught, {"type": "STLForecast", "model": self.describe()},
                              span=span)
        return FittedSTLForecast(self, result, y, s, with_backend(prov, run.record))


@dataclass(frozen=True, eq=False)
class FittedSTLForecast(BackendResult):
    """STL forecasting, estimated: the decomposition, the model of the adjusted series and forecasts."""

    spec: STLForecast
    result: Any = field(repr=False)
    data: pd.Series = field(repr=False)
    season: int = 0
    provenance: Any = None

    needs_backend: ClassVar[bool] = True

    @property
    def backend_result(self) -> Any:
        return self.result

    @property
    def dependent(self) -> str:
        return self.spec.series

    target = deprecated_alias("dependent", "target", why="the dependent variable; target is its old name")

    @property
    def components(self) -> pd.DataFrame:
        """The STL decomposition: trend, seasonal and remainder, and the seasonally adjusted series."""
        r = self.result.result
        frame = pd.DataFrame({"trend": np.asarray(r.trend, float), "seasonal": np.asarray(r.seasonal, float),
                              "remainder": np.asarray(r.resid, float)}, index=self.data.index)
        frame["adjusted"] = self.data.to_numpy(float) - frame["seasonal"]
        return frame

    @property
    def params(self) -> Any:
        from .estimation.params import point

        inner = self.result.model_result
        names = [str(n) for n in inner.param_names] if hasattr(inner, "param_names") else [str(n) for n in inner.params.index]
        return point([_param_label(n) for n in names], np.asarray(inner.params, float))

    @property
    def stats(self) -> dict[str, float]:
        return _stats(self.result.model_result)

    def summary(self) -> Table:
        p = self.params
        rows = [[float(v)] for v in p.values]
        notes = (f"Sample: {self.data.index[0]} to {self.data.index[-1]}, {len(self.data)} observations; season length {self.season}",
                 "The seasonally adjusted series is modelled and forecast; the last year's seasonal component is added back.")
        return Table(["estimate"], rows, index=list(p.names), index_name="parameter", title=f"{self.spec.describe()} for {self.dependent}",
                     notes=notes)

    @api()
    def forecast(self, data: Any = None, span: tuple[Any, Any] | None = None, alpha: float = 0.05, dynamic: bool = True,
                 horizon: int | None = None) -> Any:
        """Forecasts after the end of the sample; the interval is the adjusted series' model's (the seasonal part is
        taken as known)."""
        last = self.data.index[-1]
        start, end = _forecast_periods(last, span, horizon, "STL")
        steps = (end - last).n
        prediction = self.result.get_prediction(start=len(self.data), end=len(self.data) + steps - 1)
        frame = prediction.summary_frame(alpha=alpha)
        return _forecast_dataset(self, frame["mean"], frame["mean_se"], start, end, data, alpha, "the adjusted series' model")

    def plot(self) -> Any:
        from .figures import Figure, Line, Panel

        c = self.components
        return Figure((Panel((Line(self.data, self.dependent, color="#222222"), Line(c["adjusted"], "seasonally adjusted"),
                              Line(c["trend"], "trend", style="--")), title=f"{self.spec.describe()}: {self.dependent}"),
                       Panel((Line(c["seasonal"], "seasonal"),), title="Seasonal component")))

    def __repr__(self) -> str:
        return f"<FittedSTLForecast {self.spec.describe()} for {self.dependent}, {len(self.data)} observations>"


register("models", SARIMA)
register("models", Theta)
register("models", STLForecast)
