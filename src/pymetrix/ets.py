"""Exponential smoothing: the ETS (error, trend, seasonal) state-space models, estimated by statsmodels.

    ets = ExponentialSmoothing("sales", trend="add", damped=True, seasonal="mul").fit(data)
    ets.summary(); ets.plot()
    ets.forecast(data, ("2025Q1", "2026Q4"))       # forecast, se, lower, upper
    backtest(ets, data, ("2015Q1", "2022Q4"))      # re-estimated at each origin; densities scored

Simple exponential smoothing is ``ExponentialSmoothing("y")``, Holt's linear trend ``trend="add"``, the damped
trend ``damped=True``, and Holt-Winters ``seasonal="add"`` or ``"mul"`` (four or twelve seasons, from the data's
frequency). ``error="mul"`` gives multiplicative errors (positive series only). The models are those of Hyndman,
Koehler, Ord and Snyder (2008), in statsmodels' ``ETSModel`` with parameters and initial states estimated by
maximum likelihood; forecast standard errors are analytic for additive models and simulated otherwise. See
Elliott and Timmermann (2016), section 7.5.
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

__all__ = ["ExponentialSmoothing", "FittedExponentialSmoothing"]

_NAMES = {"smoothing_level": "alpha (level)", "smoothing_trend": "beta (trend)", "smoothing_seasonal": "gamma (seasonal)",
          "damping_trend": "phi (damping)"}


@dataclass(frozen=True)
class ExponentialSmoothing:
    """Exponential smoothing (ETS), estimated by maximum likelihood (statsmodels' ETSModel).

    A level, optionally a trend (damped or not) and seasonal factors, each updated by a share of the latest
    forecast error.

    The smoothing parameters say how fast each component adapts: near 1 it follows the latest data, near 0 it is
    nearly fixed. Choose the components to match the series: a trend only if it grows (damped trends are the
    safer choice beyond a few periods, since undamped ones extend the latest slope for ever), additive
    seasonality when the swings are of constant size and multiplicative when they grow with the level. The error
    type matters mainly for the intervals, which widen with the level under multiplicative errors. Compare candidate models by information criteria (on the same sample) or by
    out-of-sample forecast errors; ETS has no regressors.
    """

    label: ClassVar[str] = "Exponential smoothing (ETS)"
    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow",)
    requires: ClassVar[tuple[str, ...]] = ("statsmodels",)
    wraps: ClassVar[str] = "statsmodels.tsa.exponential_smoothing.ets.ETSModel"

    series: Annotated[str, Widget("text", help="the series to forecast, unbroken in the sample; quarterly or monthly for seasonal models")]
    trend: Annotated[Literal["none", "add", "mul"], Widget("choice", help="none (simple smoothing), additive (Holt) or multiplicative")] = "none"
    damped: Annotated[bool, Widget("checkbox", help="damp the trend towards flat (needs a trend)")] = False
    seasonal: Annotated[Literal["none", "add", "mul"], Widget("choice", help="seasonal factors: none, additive or multiplicative (Holt-Winters)")] = "none"
    error: Annotated[Literal["add", "mul"], Widget("choice", advanced=True, help="additive or multiplicative errors (positive series)")] = "add"
    backend_options: Annotated[BackendOptions, Widget("", advanced=True, help="other statsmodels arguments; model.x for the constructor")] = field(default_factory=dict)

    adapter_spec: ClassVar[AdapterSpec] = statsmodels_adapter.ETS

    def __post_init__(self) -> None:
        check_backend_options(self)

    def describe(self) -> str:
        parts = {"none": "N", "add": "A", "mul": "M"}
        return f"ETS({parts[self.error]},{parts[self.trend]}{'d' if self.damped and self.trend != 'none' else ''},{parts[self.seasonal]})"

    @api("pymetrix:exponential_smoothing.fit")
    def fit(self, data: Dataset, span: Any = None) -> FittedExponentialSmoothing:
        """Estimate the smoothing parameters and initial states by maximum likelihood on the periods of ``span``
        (first, last), which must hold an unbroken run of the series."""
        from ._capture import capture, provenance_for
        from .filters import periods_per_year

        ets = require("statsmodels.tsa.exponential_smoothing.ets")
        if data.kind != "time_series":
            raise SpecificationError("exponential smoothing needs a single time series")
        if self.series not in data.names:
            raise SpecificationError(f"no series {self.series!r} in the dataset")
        if self.damped and self.trend == "none":
            raise SpecificationError("only a trend can be damped: choose an additive or multiplicative trend")
        y = data[self.series][np.asarray(data.sample_mask(span))]
        first, last = y.first_valid_index(), y.last_valid_index()
        if first is None:
            raise SpecificationError(f"{self.series} has no observations in the sample")
        y = y.loc[first:last]
        if y.isna().any():
            raise SpecificationError(f"{self.series} has missing values inside the sample: exponential smoothing needs an unbroken series")
        if (self.error == "mul" or self.trend == "mul" or self.seasonal == "mul") and (y <= 0).any():
            raise SpecificationError("multiplicative components need a positive series")
        periods = None
        if self.seasonal != "none":
            periods = periods_per_year(y.index.freqstr)
            if periods not in (4, 12):
                raise SpecificationError("seasonal exponential smoothing is for quarterly and monthly data")
        needed = 10 + (2 * periods if periods else 0)
        if len(y) < needed:
            raise SpecificationError(f"fewer than {needed} observations for this model")
        endog = pd.Series(y.to_numpy(float), index=y.index.to_timestamp(), name=self.series)
        with capture() as caught, statsmodels_adapter.ADAPTER.run(self.adapter_spec, self) as run:
            model = run.call("model", ets.ETSModel, endog, error=self.error, trend=None if self.trend == "none" else self.trend,
                             damped_trend=self.damped and self.trend != "none",
                             seasonal=None if self.seasonal == "none" else self.seasonal, seasonal_periods=periods)
            result = run.call("fit", model.fit)
        converged = (getattr(result, "mle_retvals", None) or {}).get("converged")
        prov = provenance_for(self, data, [self.series], y.index, y.to_frame(), caught,
                              {"type": "ExponentialSmoothing", "model": self.describe()},
                              None if converged is None else bool(converged), span=span)
        prov = with_backend(prov, run.record)
        return FittedExponentialSmoothing(self, result, y, prov)


@dataclass(frozen=True, eq=False)
class FittedExponentialSmoothing(BackendResult):
    """An estimated exponential smoothing model: its parameters, states, fit and forecasts."""

    spec: ExponentialSmoothing
    result: Any = field(repr=False)
    data: pd.Series = field(repr=False)  # the series as estimated, with its periods
    provenance: Any = None

    needs_backend: ClassVar[bool] = True  # forecasts call statsmodels' ETSResults: kept when pickled; projects recompute it

    @property
    def backend_result(self) -> Any:
        """The library's own result object (``result``)."""
        return self.result

    @property
    def dependent(self) -> str:
        return self.spec.series

    target = deprecated_alias("dependent", "target", why="the dependent variable; target is its old name")

    @property
    def fitted(self) -> pd.Series:
        """One-step-ahead fitted values."""
        return pd.Series(np.asarray(self.result.fittedvalues, float), index=self.data.index, name=f"{self.dependent} fitted")

    @property
    def resid(self) -> pd.Series:
        """The series less its one-step-ahead fitted values."""
        return (self.data - self.fitted).rename("residual")

    @property
    def states(self) -> pd.DataFrame:
        """The level, trend and seasonal states at the end of each period."""
        frame = pd.DataFrame(np.asarray(self.result.states, float), index=self.data.index, columns=list(self.result.states.columns))
        return frame.rename(columns={"slope": "trend", "seasonal": "seasonal"})

    def params_table(self) -> Table:
        names = list(self.result.param_names)
        values = np.asarray(self.result.params, float)
        try:
            se = np.asarray(self.result.bse, float)
        except Exception:  # noqa: BLE001 - standard errors need the Hessian, which can fail
            se = np.full(len(values), np.nan)
        rows = [[float(v), float(s)] for v, s in zip(values, se)]
        return Table(["estimate", "std err"], rows, index=[_NAMES.get(n, n.replace("_", " ")) for n in names], index_name="parameter",
                     title=f"{self.spec.describe()} for {self.dependent}")

    @property
    def stats(self) -> dict[str, float]:
        """Fit statistics, in pymetrix's definitions (``estimation.criteria``): the information criteria count the
        estimated parameters; statsmodels' own, which also count the error variance, are kept as ``*_backend``."""
        from .estimation import criteria

        r = self.result
        n, k, llf = int(r.nobs), len(r.params), float(r.llf)
        return {"nobs": float(n), "k": float(k), "loglike": llf, "aic": criteria.aic(llf, k), "bic": criteria.bic(llf, k, n),
                "aic_backend": float(r.aic), "bic_backend": float(r.bic)}

    def summary(self) -> Table:
        sample = f"{self.data.index[0]} to {self.data.index[-1]}"
        table = self.params_table()
        s = self.stats
        notes = [f"Sample: {sample}, {len(self.data)} observations",
                 f"Log likelihood: {s['loglike']:.4f}", f"Akaike info criterion: {s['aic']:.4f}",
                 f"Schwarz criterion: {s['bic']:.4f}",
                 "Smoothing parameters share each forecast error between the level, trend and seasonal states; initial states are estimated."]
        return replace(table, notes=tuple(notes))

    @api()
    def forecast(self, data: Any = None, span: tuple[Any, Any] | None = None, alpha: float = 0.05, dynamic: bool = True,
                 horizon: int | None = None) -> Any:
        """Forecasts after the end of the sample, for the periods of ``span`` (or the next ``horizon`` periods), with
        standard errors and a (1 - alpha) interval. Columns: forecast, se, lower, upper, and actual where ``data``
        has the series."""
        from .data import Dataset
        from .metadata import DatasetMeta

        last = self.data.index[-1]
        if span is not None:
            from .data.sample import as_span
            span = as_span(span)  # '1990Q1:2019Q4' or a pair
            start, end = pd.Period(str(span[0]), freq=last.freq), pd.Period(str(span[1]), freq=last.freq)
        else:
            start, end = last + 1, last + int(horizon or 8)
        if start <= last:
            raise SpecificationError(f"exponential smoothing forecasts start after the sample, from {last + 1}")
        steps = (end - last).n
        prediction = self.result.get_prediction(start=len(self.data), end=len(self.data) + steps - 1)
        mean = np.asarray(prediction.predicted_mean, float)
        se = np.sqrt(np.asarray(prediction.forecast_variance, float))
        periods = pd.period_range(last + 1, end, freq=last.freq)
        z = sps.norm.ppf(1 - alpha / 2)
        frame = pd.DataFrame({"forecast": mean, "se": se, "lower": mean - z * se, "upper": mean + z * se}, index=periods)
        frame = frame.loc[start:end]
        frame.index.name = self.data.index.name or "time"
        if data is not None and self.dependent in getattr(data, "names", []):
            frame["actual"] = data[self.dependent].reindex(frame.index).to_numpy(float)
        meta = DatasetMeta(title=f"{self.spec.describe()} forecast of {self.dependent}",
                           description=f"Point forecasts with {1 - alpha:.0%} intervals ("
                                       + ("analytic" if getattr(prediction, "method", "") != "simulated" else "simulated") + " variances).")
        return Dataset.from_frame(frame, meta=meta)

    def plot(self) -> Any:
        """The series, its one-step-ahead fit and the level."""
        from .figures import Figure, Line, Panel

        layers = [Line(self.data, self.dependent, color="#222222"), Line(self.fitted, "fitted", style="--")]
        if "level" in self.states.columns:
            layers.append(Line(self.states["level"], "level"))
        return Figure((Panel(tuple(layers), title=f"{self.spec.describe()}: {self.dependent}"),))

    def __repr__(self) -> str:
        return f"<FittedExponentialSmoothing {self.spec.describe()} for {self.dependent}, {len(self.data)} observations>"


register("models", ExponentialSmoothing)
