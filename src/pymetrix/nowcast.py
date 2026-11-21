"""Nowcasting tools: what was known when (the ragged edge), and the standard ways of handling it.

    monthly = Dataset.example("sample_monthly")      # each series knows its publication lag (Variable.release_lag)
    monthly.ragged_edge()                            # last observation of each series, and when it was published
    known = monthly.as_of("2025-02-15")              # what was known on that day: later values removed
    fill_edge(known, lags=2)                         # missing months at the end forecast by an AR model (bridge style)
    realign(known)                                   # each series shifted so its last value is in the same period

Data published with different delays end in different periods: a survey for last month is out, industrial
production only for the month before. ``as_of`` rebuilds that ragged edge for any day from the publication lags, so
a nowcast can be made (and checked) with only what was known then. Three ways of dealing with it: leave the gaps for
a model that handles missing values (a factor model, by the Kalman filter); fill them with forecasts from a small
autoregression of each series (what bridge equations do); or realign each series so its latest value is treated
as current (vertical realignment, Altissimo et al. 2010). Values are not revised here: ``as_of`` removes what was
not yet published, it does not restore earlier vintages (``Variable.vintage`` and ALFRED data do that).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

import numpy as np
import pandas as pd

from ._api import api
from .results import BackendResult
from .adapters import statsmodels as statsmodels_adapter
from .adapters.base import AdapterSpec, BackendOptions, check_backend_options, with_backend
from .errors import NotFoundError, SpecificationError
from .forms import Widget
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["Bridge", "DynamicFactor", "FittedBridge", "FittedDynamicFactor", "News", "as_of", "combine", "fill_edge", "ragged_edge", "realign"]


def _day(when: Any) -> pd.Timestamp:
    return pd.Timestamp(when if not isinstance(when, date) else when.isoformat()).normalize()


def _published(index: pd.PeriodIndex, lag: int) -> pd.DatetimeIndex:
    """The day each period's value is published: the period's last day plus the lag."""
    return index.to_timestamp(how="end").normalize() + pd.Timedelta(days=int(lag))


def _lags(data: Dataset, lags: Mapping[str, int] | None) -> dict[str, int]:
    out = {}
    for name in data.names:
        given = (lags or {}).get(name)
        known = data.variables[name].release_lag if name in data.variables else None
        out[name] = int(given if given is not None else (known if known is not None else 0))
    return out


def _time_index(data: Dataset) -> pd.PeriodIndex:
    index = data.frame.index
    if not isinstance(index, pd.PeriodIndex):
        raise SpecificationError("ragged-edge tools need a time series with a period index (one series per column)")
    return index


def as_of(data: Dataset, when: Any, lags: Mapping[str, int] | None = None) -> Dataset:
    """The data as known on ``when``: values published later (period end plus ``release_lag`` days) removed.
    ``lags`` overrides series' publication lags; series without one count as published when the period ends."""
    index = _time_index(data)
    day = _day(when)
    frame = data.frame.copy()
    for name, lag in _lags(data, lags).items():
        frame.loc[_published(index, lag) > day, name] = np.nan
    from .data import Dataset

    return Dataset(frame, data.axes, data.variables, data.flags, data.meta)


def ragged_edge(data: Dataset, when: Any = None, lags: Mapping[str, int] | None = None) -> Table:
    """Each series' last observation, its publication lag and the day that value was published; with ``when``,
    also how many periods it is behind the latest period known on that day."""
    index = _time_index(data)
    lag_of = _lags(data, lags)
    rows, last = [], {}
    for name in data.names:
        values = data.frame[name]
        valid = values.notna().to_numpy()
        if not valid.any():
            rows.append([name, "", lag_of[name], "", ""])
            continue
        period = index[np.flatnonzero(valid)[-1]]
        last[name] = period
        rows.append([name, str(period), lag_of[name], str(_published(pd.PeriodIndex([period]), lag_of[name])[0].date()), ""])
    if last:
        latest = max(last.values())
        for row in rows:
            if row[1]:
                row[4] = (latest - pd.Period(row[1], freq=index.freq)).n
    title = "Ragged edge" + (f" on {_day(when).date()}" if when is not None else "")
    return Table(["last observation", "release lag (days)", "published", "periods behind"], [r[1:] for r in rows],
                 index=[r[0] for r in rows], index_name="series", title=title,
                 notes=["periods behind: how far each series ends before the latest one"])


def fill_edge(data: Dataset, lags: int = 2, through: Any = None, names: Any = None) -> Dataset:
    """Missing values at the end of each series forecast by an autoregression with ``lags`` lags (and a constant),
    estimated on the series' own history, up to ``through`` (default: the last period of any series). The way
    bridge equations complete the current quarter's months."""
    index = _time_index(data)
    end = pd.Period(str(through), freq=index.freq) if through is not None else index[-1]
    frame = data.frame.reindex(pd.period_range(index[0], max(end, index[-1]), freq=index.freq, name=index.name))
    for name in list(names) if names is not None else data.names:
        series = frame[name]
        known = series.dropna()
        if known.empty or known.index[-1] >= end:
            continue
        y = known.to_numpy(float)
        if len(y) <= lags + 2:
            raise SpecificationError(f"{name}: too few observations for an AR({lags})")
        X = np.column_stack([np.ones(len(y) - lags)] + [y[lags - k - 1:len(y) - k - 1] for k in range(lags)])
        beta, *_ = np.linalg.lstsq(X, y[lags:], rcond=None)
        history = list(y[-lags:])
        for period in pd.period_range(known.index[-1] + 1, end, freq=index.freq):
            value = beta[0] + sum(beta[k + 1] * history[-k - 1] for k in range(lags))
            frame.loc[period, name] = value
            history.append(value)
    from .data import Dataset

    return Dataset(frame, data.axes, data.variables, None, data.meta)


def realign(data: Dataset) -> Dataset:
    """Vertical realignment: each series shifted forward so that its last observation falls in the latest period
    any series has (a series one month behind is moved one month on)."""
    index = _time_index(data)
    frame = data.frame.copy()
    ends = {n: frame[n].last_valid_index() for n in data.names}
    latest = max(e for e in ends.values() if e is not None)
    for name, end in ends.items():
        if end is not None and end < latest:
            frame[name] = frame[name].shift((latest - end).n)
    from .data import Dataset

    return Dataset(frame, data.axes, data.variables, None, data.meta)


# ---------------------------------------------------------------------------------------------
# bridge equations


@dataclass(frozen=True)
class Bridge:
    """A bridge equation: a quarterly series on monthly indicators averaged to quarters (Baffigi, Golinelli and
    Parigi 2004). The current quarter's missing months are forecast by an AR model of each indicator first.

        bridge = Bridge("400*dlog(y) ~ 400*dlog(retail) + pmi").fit(quarterly, monthly)
        bridge.nowcast(quarterly, monthly.as_of("2025-02-15"))   # the quarter(s) after the last GDP figure

    The formula uses the quarterly series' names and the indicators' names (averaged within each quarter, or as
    ``how`` says: "mean", "sum", "last"). ``fill_lags`` is the order of the autoregressions that complete the
    quarter. Any estimation method for equations can be given (``method``); OLS by default.
    """

    formula: str
    how: str = "mean"
    fill_lags: int = 2
    method: Any = None

    def _quarterly(self, low: Dataset, high: Dataset, through: Any = None) -> Dataset:
        from .data import Dataset

        freq = low.frame.index.freqstr
        filled = fill_edge(high, self.fill_lags, through=through) if through is not None else high
        aggregated = filled.to_freq(freq, how=self.how)
        clash = sorted(set(aggregated.names) & set(low.names))
        if clash:
            raise SpecificationError(f"{', '.join(clash)} are in both datasets; rename the indicators")
        frame = pd.concat([low.frame, aggregated.frame], axis=1)
        frame.index.name = low.frame.index.name
        return Dataset.from_frame(frame, variables={**low.variables, **aggregated.variables})

    @api("pymetrix:bridge.fit")
    def fit(self, low: Dataset, high: Dataset, span: Any = None) -> FittedBridge:
        from .estimation import OLS, Equation

        data = self._quarterly(low, high)
        fit = Equation(self.formula).fit(data, method=self.method or OLS(), span=span)
        return FittedBridge(self, fit)


@dataclass(frozen=True, eq=False)
class FittedBridge:
    """A bridge equation estimated: the quarterly equation (``equation``) and how to nowcast with it."""

    spec: Bridge
    equation: Any  # FittedEquation on the quarterly averages

    @property
    def provenance(self) -> Any:
        """The bridge equation's provenance (on the quarterly averages of the monthly indicators)."""
        return getattr(self.equation, "provenance", None)

    @property
    def params(self) -> Any:
        return self.equation.params

    @property
    def provenance(self) -> Any:
        return self.equation.provenance

    @api("pymetrix:fitted_bridge.nowcast")
    def nowcast(self, low: Dataset, high: Dataset, through: Any = None) -> pd.Series:
        """The left-hand side for each quarter after the last observation of the target, through the quarter of
        the indicators' latest month (or ``through``): missing months forecast, months averaged, the equation applied."""
        freq = low.frame.index.freqstr
        target = self.equation.equation.require_dependent() if hasattr(self.equation.equation, "require_dependent") else None
        last_month = max(high.frame[c].last_valid_index() for c in high.names if high.frame[c].notna().any())
        end = pd.Period(str(through), freq=freq) if through is not None else last_month.asfreq(freq)
        data = self.spec._quarterly(low, high, through=end.asfreq(high.frame.index.freqstr, "e"))
        known = low.frame[target].last_valid_index() if target in low.names else low.frame.index[-1]
        start = known + 1
        if start > end:
            raise SpecificationError(f"nothing to nowcast: {target} is known through {known}")
        forecast = self.equation.forecast(data, (str(start), str(end)))
        return forecast["forecast"].rename(self.equation.equation.text.split("~")[0].strip())

    def summary(self) -> Table:
        table = self.equation.summary()
        notes = [*table.notes, f"Bridge equation: indicators averaged within quarters ({self.spec.how}); missing months "
                 f"forecast by AR({self.spec.fill_lags})"]
        return Table(table.columns, table.rows, index=table.index, title=table.title, notes=notes, index_name=table.index_name)

    def __repr__(self) -> str:
        return f"<FittedBridge {self.spec.formula!r}>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


# ---------------------------------------------------------------------------------------------
# dynamic factor model, mixed frequency


@dataclass(frozen=True)
class DynamicFactor:
    """A dynamic factor model of monthly and quarterly series (Banbura and Modugno 2014; quarterly series as in
    Mariano and Murasawa 2003), estimated by EM and filtered by the Kalman filter, so any pattern of missing values
    (the ragged edge) is handled as it is. The engine is statsmodels' ``DynamicFactorMQ``.

        dfm = DynamicFactor(monthly=["gr", "gip", "gemp", "pmi"], quarterly=["gy"]).fit(quarterly, monthly)
        dfm.nowcast("gy")                          # each quarter after the last GDP figure, from the indicators
        later = dfm.update(quarterly, monthly2)    # the same model on newer data (not re-estimated)
        later.news(dfm, "gy")                      # how each new figure moved the nowcast

    Series must be stationary (growth rates, surveys): transform them first with ``with_series``.
    """

    monthly: tuple[str, ...]
    quarterly: tuple[str, ...] = ()
    factors: int = 1
    factor_order: int = 2
    idiosyncratic_ar1: bool = True
    max_iter: int = 500
    backend_options: Annotated[BackendOptions, Widget("", advanced=True, help="other statsmodels arguments; model.x for the constructor")] = field(default_factory=dict)

    adapter_spec: ClassVar[AdapterSpec] = statsmodels_adapter.DYNAMIC_FACTOR_MQ

    def __post_init__(self) -> None:
        check_backend_options(self)
        object.__setattr__(self, "monthly", tuple(self.monthly))
        object.__setattr__(self, "quarterly", tuple(self.quarterly))
        if not self.monthly:
            raise SpecificationError("give the monthly series")

    def _frames(self, low: Dataset | None, high: Dataset) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        missing = [n for n in self.monthly if n not in high.names] + [n for n in self.quarterly if low is None or n not in low.names]
        if missing:
            raise NotFoundError(f"no series {', '.join(missing)} in the data")
        month = high.frame[list(self.monthly)].astype(float)
        starts = [month[c].first_valid_index() for c in month.columns]
        if any(s is None for s in starts):
            raise SpecificationError(f"no data for {', '.join(c for c, s in zip(month.columns, starts) if s is None)}")
        # from the first month every monthly series has data: a start where only some are observed (e.g. a survey a
        # month before the growth rates) makes the EM steps ill-conditioned; missing values later are no problem
        month = month.loc[max(starts):]
        quarter = None
        if self.quarterly:
            quarter = low.frame[list(self.quarterly)].astype(float)
            quarter = quarter.loc[quarter.index >= month.index[0].asfreq("Q")]
        return month, quarter

    @api("pymetrix:dynamic_factor.fit")
    def fit(self, low: Dataset | None, high: Dataset) -> FittedDynamicFactor:
        """Estimate on monthly ``high`` and quarterly ``low`` data (``low`` may be None)."""
        from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

        from ._capture import capture

        month, quarter = self._frames(low, high)
        with capture() as caught, statsmodels_adapter.ADAPTER.run(self.adapter_spec, self) as run:
            model = run.call("model", DynamicFactorMQ, month, endog_quarterly=quarter, factors=self.factors,
                             factor_orders=self.factor_order, idiosyncratic_ar1=self.idiosyncratic_ar1)
            result = run.call("fit", model.fit, maxiter=self.max_iter)
        iterations = int(result.mle_retvals.get("iter", 0)) if isinstance(result.mle_retvals, dict) else 0
        prov = with_backend(_provenance(self, high, month, quarter, caught, iterations), run.record)
        return FittedDynamicFactor(self, result, prov)


def _provenance(spec: DynamicFactor, high: Dataset, month: pd.DataFrame, quarter: pd.DataFrame | None, caught: list[str],
                iterations: int) -> Any:
    from ._capture import fingerprint
    from ._version import __version__
    from .metadata import Provenance

    used = pd.concat([month.reset_index(drop=True), quarter.reset_index(drop=True)], axis=1) if quarter is not None else month
    return Provenance(
        data_vintages={n: high.variables[n].vintage for n in spec.monthly if n in high.variables},
        spec={"type": "DynamicFactor", "monthly": list(spec.monthly), "quarterly": list(spec.quarterly), "factors": spec.factors,
              "factor_order": spec.factor_order},
        method={"type": "dynamic factor (EM, Kalman filter)", "iterations": iterations},
        sample=(str(month.index[0]), str(month.index[-1])), nobs=len(month), dropped=0, package_version=__version__,
        data_fingerprint=fingerprint(used), warnings=tuple(caught), converged=iterations < spec.max_iter,
    )


@dataclass(frozen=True, eq=False)
class FittedDynamicFactor(BackendResult):
    """A dynamic factor model estimated: factors, nowcasts, and the same model on newer data."""

    spec: DynamicFactor
    result: Any  # statsmodels DynamicFactorMQResults
    provenance: Any = None

    needs_backend: ClassVar[bool] = True  # nowcasts and news call statsmodels' DynamicFactorMQResults: kept when pickled; projects recompute it

    @property
    def backend_result(self) -> Any:
        """The library's own result object (``result``)."""
        return self.result

    @property
    def factors(self) -> pd.DataFrame:
        """The smoothed factors, month by month."""
        from .adapters.base import to_axes

        raw = self.result.factors.smoothed
        frame = to_axes(raw, pd.PeriodIndex(raw.index, freq="M") if not isinstance(raw.index, pd.PeriodIndex) else raw.index)
        frame.columns = [f"factor {i + 1}" for i in range(frame.shape[1])]
        frame.index.name = "time"
        return frame

    @property
    def loadings(self) -> pd.DataFrame:
        """How much each series moves with each factor."""
        params = self.result.params
        factor_names = list(self.result.model.factor_names)
        names = [*self.spec.monthly, *self.spec.quarterly]
        rows = {n: [float(params.get(f"loading.{f}->{n}", np.nan)) for f in factor_names] for n in names}
        return pd.DataFrame.from_dict(rows, orient="index", columns=[f"factor {i + 1}" for i in range(len(factor_names))])

    @property
    def stats(self) -> dict[str, float]:
        return {"loglike": float(self.result.llf), "nobs": float(self.result.nobs),
                "iterations": float(self.provenance.method.get("iterations", 0) if self.provenance else 0)}

    def _end_month(self, through: Any) -> pd.Period:
        if through is None:
            last = statsmodels_adapter.dynamic_factor_index(self.result)[-1]
            return pd.Period(str(last), freq="M").asfreq("Q").asfreq("M", "e")
        return pd.Period(str(through)).asfreq("M", "e")

    @api("pymetrix:fitted_dynamic_factor.nowcast")
    def nowcast(self, variable: str, through: Any = None) -> pd.Series:
        """``variable`` for each period after its last observation, through the quarter of the latest month (or
        ``through``): quarterly series by quarter, monthly series by month. Uses all the data (smoothed estimates),
        so it equals the ``after`` value of ``news``."""
        end = self._end_month(through)
        predicted = self.result.get_prediction(end=end, information_set="smoothed").predicted_mean[variable]
        predicted.index = pd.PeriodIndex(predicted.index, freq="M")
        observed = pd.DataFrame(statsmodels_adapter.original_endog(self.result.model))[variable].dropna()
        last = pd.Period(str(observed.index[-1]), freq="M") if len(observed) else None
        if variable in self.spec.quarterly:
            quarters = predicted[np.asarray(predicted.index.month) % 3 == 0]
            quarters.index = quarters.index.asfreq("Q")
            return quarters[quarters.index > last.asfreq("Q")] if last is not None else quarters
        return predicted[predicted.index > last] if last is not None else predicted

    def update(self, low: Dataset | None, high: Dataset) -> FittedDynamicFactor:
        """The same model (parameters unchanged) on newer data: for nowcasts as new figures arrive, and for news."""
        from ._capture import capture

        month, quarter = self.spec._frames(low, high)
        with capture() as caught, statsmodels_adapter.ADAPTER.run(self.spec.adapter_spec) as run:
            result = run.call("apply", self.result.apply, month, endog_quarterly=quarter)
        # the new data's sample and fingerprint; the parameters, and whether their estimation converged, are the fit's
        old = self.provenance
        iterations = int((old.method or {}).get("iterations", 0))
        prov = with_backend(_provenance(self.spec, high, month, quarter, caught, iterations), run.record)
        prov = replace(prov, method={**prov.method, "parameters": f"estimated on {old.sample[0]} to {old.sample[1]}, not re-estimated"},
                       converged=old.converged)
        return FittedDynamicFactor(self.spec, result, prov)

    def news(self, previous: FittedDynamicFactor, variable: str, period: Any = None) -> News:
        """How the new figures in this model's data, compared with ``previous`` (an earlier ``update``), moved the
        nowcast of ``variable`` for ``period`` (default: the quarter of the latest month)."""
        end = self._end_month(period)
        raw = self.result.news(previous.result, impact_date=end, impacted_variable=variable, comparison_type="previous")
        details = raw.details_by_impact.reset_index()
        details = details[["update date", "updated variable", "observed", "forecast (prev)", "news", "weight", "impact"]].copy()
        details.columns = ["period", "series", "observed", "expected", "news", "weight", "impact"]
        details["period"] = details["period"].astype(str)
        before = float(np.asarray(raw.prev_impacted_forecasts.loc[end, variable]).ravel()[0])
        after = float(np.asarray(raw.post_impacted_forecasts.loc[end, variable]).ravel()[0])
        revisions = raw.revision_impacts
        revised = float(np.nan_to_num(np.asarray(revisions.loc[end, variable], dtype=float).ravel()[0])) if revisions is not None and len(revisions) else 0.0
        label = str(end.asfreq("Q")) if variable in self.spec.quarterly else str(end)
        return News(variable, label, before, after, details.reset_index(drop=True), revised)

    def summary(self) -> Table:
        notes = [f"Dynamic factor model: {self.spec.factors} factor(s), VAR({self.spec.factor_order}); monthly "
                 f"{', '.join(self.spec.monthly)}" + (f"; quarterly {', '.join(self.spec.quarterly)}" if self.spec.quarterly else ""),
                 f"Log likelihood: {self.result.llf:.3f}",
                 f"EM iterations: {int(self.stats['iterations'])}"
                 + ("" if not self.provenance or self.provenance.converged else " (did not converge)")]
        return Table.from_frame(self.loadings, title="Factor loadings", notes=notes)

    def plot(self) -> Any:
        from .figures import Figure, Line, Panel

        frame = self.factors
        frame.index = pd.PeriodIndex(frame.index, freq="M") if not isinstance(frame.index, pd.PeriodIndex) else frame.index
        return Figure(tuple(Panel((Line(frame[c], c),), title=c) for c in frame.columns), title="Factors (smoothed)")

    def __repr__(self) -> str:
        return (f"<FittedDynamicFactor: {self.spec.factors} factor(s), {len(self.spec.monthly)} monthly and "
                f"{len(self.spec.quarterly)} quarterly series>")

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


@dataclass(frozen=True, eq=False)
class News:
    """How new figures moved a nowcast: ``after - before`` is the sum of the ``details`` impacts plus ``revisions``."""

    variable: str
    period: str
    before: float
    after: float
    details: pd.DataFrame  # one row per new figure: period, series, observed, expected, news, weight, impact
    revisions: float = 0.0  # the impact of revised earlier figures

    @property
    def by_series(self) -> pd.Series:
        """The impact of each series' new figures, added up."""
        return self.details.groupby("series")["impact"].sum()

    def summary(self) -> Table:
        notes = [f"{self.variable} for {self.period}: {self.before:.4g} before, {self.after:.4g} after "
                 f"(revision {self.after - self.before:+.4g}; from revised data {self.revisions:+.4g})"]
        return Table.from_frame(self.details.set_index("series"), title=f"News: {self.variable} {self.period}", notes=notes)

    def __repr__(self) -> str:
        return f"<News {self.variable} {self.period}: {self.before:.4g} -> {self.after:.4g}>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


@api("pymetrix:nowcast.combine")
def combine(nowcasts: Mapping[str, pd.Series], weights: Mapping[str, float] | None = None) -> pd.Series:
    """A weighted average of nowcasts from several models (equal weights unless given; weights are normalised).
    Inverse mean squared errors from a backtest are a common choice of weights."""
    if not nowcasts:
        raise SpecificationError("give the nowcasts to combine")
    names = list(nowcasts)
    w = np.array([float((weights or {}).get(n, 1.0)) for n in names])
    if np.any(w < 0) or w.sum() == 0:
        raise SpecificationError("weights must be non-negative and not all zero")
    frame = pd.concat([nowcasts[n].rename(n) for n in names], axis=1)
    return (frame * (w / w.sum())).sum(axis=1, min_count=len(names)).rename("combined")
