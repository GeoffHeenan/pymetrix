"""Seasonal adjustment and calendar effects.

    data.seasonal_adjust("sales", STL())              # adds sales_sa and sales_seasonal
    data.seasonal_adjust(["c", "i"], X13(trading=True))   # X13 and TramoSeats: the pymetrix-seasonal plugin
    data.calendar(trading_days=True, easter=8)        # trading-day, Easter and leap-year regressors

Methods are configuration objects, like filters. ``STL`` (seasonal-trend decomposition by
LOESS) and ``Classical`` (moving-average decomposition) are computed by statsmodels, every argument of
theirs a setting. ``X13`` (X-13ARIMA-SEATS, the US Census Bureau's ``x13as`` program) and ``TramoSeats``
(TRAMO-SEATS in Eurostat's JDemetra+, in Java) are in the ``pymetrix-seasonal`` plugin, as they need programs
outside Python; installed, they register here like any plugin's method. A method whose class sets
``needs_periods = True`` is given the series' periods and a ``notes`` dict for its provenance, and the dataset's
series it names in ``inputs()`` as ``regressors``. Every adjusted series records what made it: ``backend`` (for
STL and classical, the statsmodels function and version), ``backend_options`` (the arguments) and any
``backend_warnings``.

Seasonal adjustment is multiplicative for positive series by default (``log=None``: logs
when every value is positive), so ``{name}_seasonal`` is then a factor around 1, and
``{name}_sa = {name} / {name}_seasonal``; for additive adjustment it is a component around
0, and ``{name}_sa = {name} - {name}_seasonal``. The adjusted series are derived: they are
recomputed when the original changes. Panels are adjusted within each entity.

Calendar regressors follow X-13's definitions: six trading-day contrasts (the number of
Mondays, ..., Saturdays in the period, each less the number of Sundays), or one weekday
contrast (weekdays less 5/2 of weekend days); Easter[w], the share of the w days before
Easter Sunday that fall in the period; and a leap-year regressor (0.75 in a February of 29
days, -0.25 in other Februaries, 0 otherwise; for quarters, in the first quarter).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd

from .errors import SpecificationError
from .adapters import statsmodels as statsmodels_adapter
from .filters import periods_per_year
from .forms import Widget
from .plugins import register

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["STL", "Classical", "calendar_effects", "seasonal_components"]

# moved to the pymetrix-seasonal plugin: still found here when it is installed (``from pymetrix.seasonal import X13``)
_PLUGIN_NAMES = {"X13": "X13", "TramoSeats": "TramoSeats", "run_x13": "run_x13", "x13_notes": "x13_notes",
                 "x13_program": "x13_program"}
PLUGIN_INSTALL = 'pip install "pymetrix-seasonal[jdemetra]"'


def __getattr__(name: str) -> Any:
    if name in _PLUGIN_NAMES:
        try:
            import pymetrix_seasonal
        except ImportError:
            raise AttributeError(f"pymetrix.seasonal.{name} is in the seasonal-adjustment plugin: {PLUGIN_INSTALL}") from None
        return getattr(pymetrix_seasonal, _PLUGIN_NAMES[name])
    raise AttributeError(f"module 'pymetrix.seasonal' has no attribute {name!r}")

_LOG = Widget("choice", help="multiplicative (logs) or additive; empty = multiplicative when every value is positive")


def _note_backend(notes: dict[str, str] | None, record: Any) -> None:
    """Keep what statsmodels was given and said with the adjusted series (``backend``, ``backend_options``,
    ``backend_warnings`` in the variables' ``extra``), as X-13's own notes are."""
    if notes is None:
        return
    notes["backend"] = f"{record.label} ({record.wraps})"
    notes["backend_options"] = "; ".join(f"{k}={v!r}" for stage in record.options.values() for k, v in stage.items())
    if record.warnings:
        notes["backend_warnings"] = " | ".join(record.warnings)


def _use_log(log: bool | None, y: np.ndarray, name: str) -> bool:
    if log is None:
        return bool(np.all(y > 0))
    if log and np.any(y <= 0):
        raise SpecificationError(f"{name} has zero or negative values: multiplicative adjustment (log=True) needs positive values")
    return bool(log)


_STL_DEFAULTS = {"low_pass": None, "seasonal_deg": 1, "trend_deg": 1, "low_pass_deg": 1, "seasonal_jump": 1, "trend_jump": 1,
                 "low_pass_jump": 1}  # statsmodels' own: left out of the call (and its record) when unchanged


@dataclass(frozen=True)
class STL:
    """Seasonal-trend decomposition by LOESS (Cleveland et al. 1990), computed by statsmodels."""

    label: ClassVar[str] = "STL"
    topics: ClassVar[tuple[str, ...]] = ("seasonal-adjustment",)
    requires: ClassVar[tuple[str, ...]] = ("statsmodels",)
    wraps: ClassVar[str] = "statsmodels.tsa.seasonal.STL"

    seasonal: Annotated[int, Widget("number", min=3, max=99, step=2, help="seasonal smoother length (odd); larger = more stable seasonal pattern")] = 7
    robust: Annotated[bool, Widget("checkbox", help="downweight outliers")] = True
    log: Annotated[bool | None, _LOG] = None
    trend: Annotated[int | None, Widget("number", min=3, max=999, advanced=True, label="Trend length",
                                           help="trend smoother length (odd); empty = automatic")] = None
    low_pass: Annotated[int | None, Widget("number", min=3, max=999, advanced=True,
                                           help="low-pass filter length (odd, more than the period); empty = the next odd number above it")] = None
    seasonal_deg: Annotated[Literal[0, 1], Widget("choice", advanced=True, help="degree of the seasonal LOESS: 0 constant, 1 linear")] = 1
    trend_deg: Annotated[Literal[0, 1], Widget("choice", advanced=True, help="degree of the trend LOESS: 0 constant, 1 linear")] = 1
    low_pass_deg: Annotated[Literal[0, 1], Widget("choice", advanced=True, help="degree of the low-pass LOESS: 0 constant, 1 linear")] = 1
    seasonal_jump: Annotated[int, Widget("number", min=1, max=100, advanced=True,
                                         help="fit the seasonal LOESS every this many points and interpolate (faster)")] = 1
    trend_jump: Annotated[int, Widget("number", min=1, max=100, advanced=True, help="fit the trend LOESS every this many points")] = 1
    low_pass_jump: Annotated[int, Widget("number", min=1, max=100, advanced=True, help="fit the low-pass LOESS every this many points")] = 1
    inner_iter: Annotated[int | None, Widget("number", min=1, max=50, advanced=True,
                                             help="inner-loop iterations; empty = automatic (2 if robust, else 5)")] = None
    outer_iter: Annotated[int | None, Widget("number", min=0, max=50, advanced=True,
                                             help="robustness (outer-loop) iterations; empty = automatic (15 if robust, else 0)")] = None

    def decompose(self, y: np.ndarray, period: int, name: str, notes: dict[str, str] | None = None) -> tuple[np.ndarray, bool]:
        from statsmodels.tsa.seasonal import STL as _STL

        log = _use_log(self.log, y, name)
        x = np.log(y) if log else y
        smoothing = {k: getattr(self, k) for k in ("low_pass", "seasonal_deg", "trend_deg", "low_pass_deg", "seasonal_jump",
                                                   "trend_jump", "low_pass_jump") if getattr(self, k) != _STL_DEFAULTS[k]}
        iterations = {k: getattr(self, k) for k in ("inner_iter", "outer_iter") if getattr(self, k) is not None}
        with statsmodels_adapter.ADAPTER.run(statsmodels_adapter.STL_DECOMPOSITION) as run:
            model = run.call("model", _STL, x, period=period, seasonal=self.seasonal, trend=self.trend, robust=self.robust, **smoothing)
            seasonal = np.asarray(run.call("fit", model.fit, **iterations).seasonal)
        _note_backend(notes, run.record)
        return (np.exp(seasonal), True) if log else (seasonal, False)

    def describe(self) -> str:
        return f"STL(seasonal={self.seasonal}{', robust' if self.robust else ''})"


@dataclass(frozen=True)
class Classical:
    """Classical decomposition: a centred moving-average trend, and the average deviation from it in
    each season. Simple and transparent, but the seasonal pattern cannot change over time."""

    label: ClassVar[str] = "Classical (moving average)"
    topics: ClassVar[tuple[str, ...]] = ("seasonal-adjustment",)
    requires: ClassVar[tuple[str, ...]] = ("statsmodels",)
    wraps: ClassVar[str] = "statsmodels.tsa.seasonal.seasonal_decompose"

    log: Annotated[bool | None, _LOG] = None
    two_sided: Annotated[bool, Widget("checkbox", advanced=True,
                                      help="a centred moving average (unticked: one-sided, past values only)")] = True
    extrapolate_trend: Annotated[int, Widget("number", min=0, max=1000, advanced=True,
                                             help="extend the trend to the ends by a line through this many points; 0 = leave the ends empty")] = 0

    def decompose(self, y: np.ndarray, period: int, name: str, notes: dict[str, str] | None = None) -> tuple[np.ndarray, bool]:
        from statsmodels.tsa.seasonal import seasonal_decompose

        log = _use_log(self.log, y, name)
        extra = {k: v for k, v in (("two_sided", self.two_sided), ("extrapolate_trend", self.extrapolate_trend))
                 if v != {"two_sided": True, "extrapolate_trend": 0}[k]}  # statsmodels' defaults stay out of the record
        with statsmodels_adapter.ADAPTER.run(statsmodels_adapter.CLASSICAL_DECOMPOSITION) as run:
            result = run.call("fit", seasonal_decompose, y, model="multiplicative" if log else "additive", period=period, **extra)
        _note_backend(notes, run.record)
        return np.asarray(result.seasonal), log

    def describe(self) -> str:
        return "classical decomposition"


for _cls in (STL, Classical):
    register("seasonal", _cls)


# ---------------------------------------------------------------------------------------------
# applying a method to a dataset


def _period(data: Dataset) -> int:
    t = data.time_axis
    if t is None:
        raise SpecificationError("seasonal adjustment needs a time axis")
    f = periods_per_year(data.frame.index.get_level_values(t.name).freqstr)
    if f not in (4, 12):
        raise SpecificationError("seasonal adjustment is for monthly and quarterly data")
    return f


def seasonal_components(data: Dataset, name: str, method: Any, notes: dict[str, str] | None = None) -> tuple[pd.Series, pd.Series]:
    """The seasonally adjusted series and its seasonal factor (or component), within each entity.

    ``notes``, if given, gets what statsmodels was given and said (``backend``, ``backend_options``,
    ``backend_warnings``) and what a plugin's method reports (an X-13 run's version, model and outliers); in a panel, a note that
    differs between entities is given for each, as ``"north: (0 1 1)(0 1 1); south: ..."``."""
    runs: list[tuple[Any, dict[str, str]]] = []
    t = data.time_axis
    period = _period(data)
    frame = data.frame
    entities = [e.name for e in data.entity_axes]
    sa = pd.Series(np.nan, index=frame.index, name=f"{name}_sa")
    seasonal = pd.Series(np.nan, index=frame.index, name=f"{name}_seasonal")
    level = entities[0] if len(entities) == 1 else entities  # one axis: plain keys, not 1-tuples
    groups = frame[name].groupby(level=level) if entities else [(None, frame[name])]
    from .progress import report

    for key, s in groups:
        label = name if key is None else f"{name} ({key})"
        report(None, f"seasonal adjustment: {label}")  # the app's progress bar; Cancel stops between series
        values = s.to_numpy(dtype=float)
        ok = np.flatnonzero(np.isfinite(values))
        if not len(ok):
            continue
        first, last = ok[0], ok[-1]
        if len(ok) != last - first + 1:
            raise SpecificationError(f"{label} has missing values inside its sample; seasonal adjustment needs an unbroken series")
        if last - first + 1 < 3 * period:
            raise SpecificationError(f"{label} is too short for seasonal adjustment: at least three years are needed")
        y = values[first : last + 1]
        run: dict[str, str] = {}
        if getattr(method, "needs_periods", False):  # X-13 and TRAMO-SEATS (pymetrix-seasonal): periods, notes, regressors
            periods = s.index.get_level_values(t.name)[first : last + 1]  # type: ignore[union-attr]
            extra = {}
            inputs = tuple(method.inputs()) if hasattr(method, "inputs") else ()
            if inputs:
                missing = [r for r in inputs if r not in frame.columns]
                if missing:
                    raise SpecificationError(f"{_label(method)} regressors {', '.join(missing)} are not series of the dataset")
                extra["regressors"] = {r: frame.loc[s.index, r].to_numpy(dtype=float)[first : last + 1] for r in inputs}
            factor, multiplicative = method.decompose(y, period, label, pd.PeriodIndex(periods), notes=run, **extra)
            runs.append((key, run))
        elif isinstance(method, (STL, Classical)):
            factor, multiplicative = method.decompose(y, period, label, notes=run)
            runs.append((key, run))
        else:  # a plugin's method
            factor, multiplicative = method.decompose(y, period, label)
        adjusted = y / factor if multiplicative else y - factor
        rows = s.index[first : last + 1]
        sa.loc[rows], seasonal.loc[rows] = adjusted, factor
    if notes is not None:
        for field in dict.fromkeys(f for _, run in runs for f in run):
            values = [(key, run.get(field, "")) for key, run in runs]
            same = {v for _, v in values}
            notes[field] = same.pop() if len(same) == 1 else "; ".join(f"{_entity(key)}: {v}" for key, v in values)
    return sa, seasonal


def _label(method: Any) -> str:
    return str(getattr(type(method), "label", "") or type(method).__name__)


def _entity(key: Any) -> str:
    return "/".join(map(str, key)) if isinstance(key, tuple) else str(key)


def apply(data: Dataset, names: str | list[str], method: Any, derived: bool = True) -> Dataset:
    from .data import Dataset
    from .data.derived import SeasonallyAdjusted
    from .metadata import Adjustment, Variable

    names = [names] if isinstance(names, str) else list(names)
    _period(data)
    if derived:
        out = data.derive(**{f"{n}_{part}": SeasonallyAdjusted(n, method, part) for n in names for part in ("sa", "seasonal")})
        variables = dict(out.variables)
    else:
        frame = data.frame.copy()
        variables = dict(data.variables)
        for n in names:
            notes: dict[str, str] = {}  # what statsmodels and X-13 report about the run: provenance
            sa, seasonal = seasonal_components(data, n, method, notes)
            frame[f"{n}_sa"], frame[f"{n}_seasonal"] = sa, seasonal
            for part in ("sa", "seasonal"):
                variables[f"{n}_{part}"] = Variable(f"{n}_{part}", derived_from=SeasonallyAdjusted(n, method, part).describe(data),
                                                    extra=dict(notes))
        out = Dataset(frame, data.axes, variables, data.flags, data.meta)
    for n in names:
        base = data.variables[n]
        variables[f"{n}_sa"] = replace(variables[f"{n}_sa"], units=base.units, scale=base.scale, measure=base.measure,
                                       label=f"{base.label or n}, seasonally adjusted", adjustment=Adjustment("sa"))
        variables[f"{n}_seasonal"] = replace(variables[f"{n}_seasonal"], label=f"{base.label or n}, seasonal factor")
    return Dataset(out.frame, out.axes, variables, out.flags, out.meta)


# ---------------------------------------------------------------------------------------------
# calendar effects


def _easter(year: int) -> pd.Timestamp:
    from dateutil.easter import easter

    return pd.Timestamp(easter(year))


def calendar_effects(index: pd.PeriodIndex, trading_days: Literal["six", "one", "none"] = "six", easter: int | None = 8,
                     leap_year: bool = True) -> pd.DataFrame:
    """Calendar regressors for monthly or quarterly periods (X-13's definitions; see the module notes)."""
    freq = periods_per_year(index.freqstr)
    if freq not in (4, 12):
        raise SpecificationError("calendar regressors are for monthly and quarterly data")
    out: dict[str, np.ndarray] = {}
    starts, ends = index.start_time.normalize(), index.end_time.normalize()
    if trading_days != "none":
        counts = np.zeros((len(index), 7))
        for i, (a, b) in enumerate(zip(starts, ends)):
            days = pd.date_range(a, b, freq="D")
            counts[i] = np.bincount(days.day_of_week, minlength=7)  # Monday = 0, ..., Sunday = 6
        if trading_days == "six":
            for d, label in enumerate(("mon", "tue", "wed", "thu", "fri", "sat")):
                out[f"td_{label}"] = counts[:, d] - counts[:, 6]
        elif trading_days == "one":
            out["td_weekdays"] = counts[:, :5].sum(axis=1) - 2.5 * counts[:, 5:].sum(axis=1)
        else:
            raise SpecificationError("trading_days must be 'six', 'one' or 'none'")
    if easter:
        if not 1 <= easter <= 25:
            raise SpecificationError("the Easter window must be 1 to 25 days")
        shares = np.zeros(len(index))
        for year in range(index.min().year - 1, index.max().year + 2):
            sunday = _easter(year)
            window = pd.date_range(sunday - pd.Timedelta(days=easter), sunday - pd.Timedelta(days=1), freq="D")
            for i, (a, b) in enumerate(zip(starts, ends)):
                shares[i] += ((window >= a) & (window <= b)).sum() / easter
        out[f"easter_{easter}"] = shares
    if leap_year:
        lp = np.zeros(len(index))
        for i, p in enumerate(index):
            if freq == 12 and p.month == 2 or freq == 4 and p.quarter == 1:
                lp[i] = 0.75 if pd.Timestamp(year=p.year, month=1, day=1).is_leap_year else -0.25
        out["leap_year"] = lp
    return pd.DataFrame(out, index=index)
