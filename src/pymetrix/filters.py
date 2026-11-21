"""Trend-cycle filters.

Filters are configuration objects applied with ``Dataset.filter``::

    ds.filter("ly", HP())                       # adds ly_trend and ly_cycle
    ds.filter("ly", HP(one_sided=True))         # real-time trend: each period uses data up to then
    ds.filter(["ly", "lc"], ChristianoFitzgerald(low=6, high=32))

Defaults follow the data's frequency (``f`` periods per year). Filters run
within each entity of a panel. A series must have no gaps between its first
and last observation; leading and trailing missing values are left missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, ClassVar

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .errors import SpecificationError
from .forms import Widget
from .plugins import register

if TYPE_CHECKING:
    from .data import Dataset

_PER_YEAR = {"Y": 1, "A": 1, "Q": 4, "M": 12, "W": 52, "B": 260, "D": 365}


def periods_per_year(freqstr: str) -> int:
    base = freqstr.split("-")[0].lstrip("0123456789").upper()
    for key, f in _PER_YEAR.items():
        if base.startswith(key):
            return f
    raise SpecificationError(f"no default filter settings for frequency {freqstr!r}; give them explicitly")


@dataclass(frozen=True)
class HP:
    """Hodrick-Prescott filter. ``lamb=None``: 1600*(f/4)**power (power 2: EViews' rule; 4: Ravn-Uhlig).

    The trend minimises the squared deviations of the series from it plus lambda times the squared changes in its
    slope; the cycle is the remainder. A larger lambda gives a smoother trend and longer cycles. The two-sided filter
    uses future data, so its last estimates are the least reliable and are revised as new data arrive. The one-sided
    version (each period's trend is the end point of the filter run on the data up to then) suits real-time measures
    such as gaps, but its trend lags turning points. Hamilton (2018) shows that the filter can produce cycles with
    spurious dynamics, even from a random walk; his regression filter is an alternative.
    """

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import HP, Dataset

        data = Dataset.example("sample_macro").derive(ly="log(y)")
        gap = data.filter("ly", HP())                            # adds ly_trend and ly_cycle; lambda 1600 for quarterly data
        real_time = data.filter("ly", HP(one_sided=True))        # each period's trend from the data up to then
        gap.frame[["ly", "ly_trend", "ly_cycle"]].tail()         # ly_cycle: the log deviation from trend, the output gap as a fraction
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("{name}_trend", "the smooth trend; with the two-sided filter its last values move as new data arrive"),
        ("{name}_cycle", ("the series less its trend; for a log series, the proportional deviation (output gap as a "
                          "fraction)")),
        ("missing values at the ends", ("no values are lost; the one-sided version's past values are not revised by new data, "
                                "but its trend lags turning points")),
        ("data.variables['{name}_trend'].derived_from", "the filter and lambda used, e.g. HP(lambda=1600) trend of ly"),
    )

    label: ClassVar[str] = "HP"
    topics: ClassVar[tuple[str, ...]] = ('trend-cycle-filters',)
    references: ClassVar[tuple[str, ...]] = (
        ("Hodrick, R. J. and Prescott, E. C. (1997). Postwar U.S. business cycles: an empirical investigation. Journal of Money, Credit and Banking 29(1), 1-16. doi:10.2307/2953682"),
        ("Ravn, M. O. and Uhlig, H. (2002). On adjusting the Hodrick-Prescott filter for the frequency of observations. Review of Economics and Statistics 84(2), 371-376. doi:10.1162/003465302317411604"),
        ("Hamilton, J. D. (2018). Why you should never use the Hodrick-Prescott filter. Review of Economics and Statistics 100(5), 831-843. doi:10.1162/rest_a_00706"),
    )

    lamb: Annotated[float | None, Widget("text", help="smoothing parameter lambda: larger, smoother trend; empty = frequency default (1600 quarterly)")] = None
    power: Annotated[int, Widget("number", min=1, max=6, advanced=True,
                                 help="default lambda = 1600*(f/4)^power: 2 is EViews' rule, 4 Ravn-Uhlig")] = 2
    one_sided: Annotated[bool, Widget("checkbox", help="real-time trend: each period uses data up to then only")] = False

    def describe(self, f: int) -> str:
        return f"{'one-sided ' if self.one_sided else ''}HP(lambda={self.smoothing(f):g})"

    def smoothing(self, f: int) -> float:
        return self.lamb if self.lamb is not None else 1600 * (f / 4) ** self.power

    def trend(self, y: np.ndarray, f: int) -> np.ndarray:
        lamb = self.smoothing(f)
        if not self.one_sided:
            return hp_trend(y, lamb)
        return np.array([hp_trend(y[: t + 1], lamb)[-1] for t in range(len(y))])


@dataclass(frozen=True)
class BaxterKing:
    """Baxter-King band-pass filter: keeps cycles between ``low`` and ``high`` periods; loses ``k`` points at each end.

    A symmetric moving average of 2k + 1 terms approximates the ideal filter that keeps only cycles of between
    ``low`` and ``high`` periods (by default the business-cycle band of 1.5 to 8 years); the trend is the remainder.
    Its weights sum to zero, so it removes trends, stochastic ones included, and being symmetric it does not shift
    the cycle in time. A larger k approximates the ideal filter better but loses more observations at each end, where
    the cycle is often of most interest; the Christiano-Fitzgerald filter keeps them.
    """

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import BaxterKing, Dataset

        data = Dataset.example("sample_macro").derive(ly="log(y)")
        data = data.filter("ly", BaxterKing())                   # cycles of 6 to 32 quarters; k = 12 points lost at each end
        data.frame[["ly", "ly_cycle"]].dropna()
        short = data.filter("ly", BaxterKing(k=6))               # a cruder band-pass that loses only 6 points at each end
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("{name}_cycle", "the band-pass component, cycles between low and high periods"),
        ("{name}_trend", "the series less its cycle, so it keeps the short, irregular movements as well as the trend"),
        ("missing values at the ends", "k values missing at each end (12 for quarterly data by default)"),
    )

    label: ClassVar[str] = "Baxter-King"
    topics: ClassVar[tuple[str, ...]] = ('trend-cycle-filters',)
    references: ClassVar[tuple[str, ...]] = (
        ("Baxter, M. and King, R. G. (1999). Measuring business cycles: approximate band-pass filters for economic time series. Review of Economics and Statistics 81(4), 575-593. doi:10.1162/003465399558454"),
    )

    low: Annotated[float | None, Widget("text", help="shortest cycle kept, in periods; empty = 1.5 years")] = None
    high: Annotated[float | None, Widget("text", help="longest cycle kept, in periods; empty = 8 years")] = None
    k: Annotated[int | None, Widget("text", advanced=True, help="lead/lag length (points lost at each end); empty = 3 years")] = None

    def settings(self, f: int) -> tuple[float, float, int]:
        return self.low or max(2.0, 1.5 * f), self.high or 8.0 * f, self.k or 3 * f

    def describe(self, f: int) -> str:
        low, high, k = self.settings(f)
        return f"BaxterKing({low:g}, {high:g}, k={k})"

    def cycle(self, y: np.ndarray, f: int) -> np.ndarray:
        low, high, k = self.settings(f)
        w1, w2 = 2 * np.pi / high, 2 * np.pi / low
        j = np.arange(1, k + 1)
        b = np.r_[(w2 - w1) / np.pi, (np.sin(w2 * j) - np.sin(w1 * j)) / (np.pi * j)]
        weights = np.r_[b[:0:-1], b]  # b_k .. b_1, b_0, b_1 .. b_k
        weights -= weights.mean()  # weights sum to zero: removes stochastic trends
        out = np.full(len(y), np.nan)
        if len(y) > 2 * k:
            out[k:-k] = np.convolve(y, weights, mode="valid")
        return out


@dataclass(frozen=True)
class ChristianoFitzgerald:
    """Christiano-Fitzgerald asymmetric band-pass filter (random-walk approximation), full sample.

    It approximates the ideal filter that keeps cycles of between ``low`` and ``high`` periods using every
    observation at every date, with asymmetric weights, so no observations are lost at the ends. The weights are
    those that are optimal when the series is a random walk (after removing a drift, if asked). Near the ends the
    estimates use mostly one side of the data, so they are revised as data arrive, and the asymmetric weights can
    shift the cycle in time there.
    """

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import ChristianoFitzgerald, Dataset

        data = Dataset.example("sample_macro").derive(ly="log(y)", lc="log(c)")
        data = data.filter(["ly", "lc"], ChristianoFitzgerald()) # cycles of 6 to 32 quarters, with no points lost at the ends
        data.frame[["ly_cycle", "lc_cycle"]].head()
        longer = data.filter("ly", ChristianoFitzgerald(high=40))  # keep cycles up to 10 years
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("{name}_cycle", "the band-pass component, cycles between low and high periods"),
        ("{name}_trend", "the series less its cycle, so it keeps the short, irregular movements as well as the trend"),
        ("missing values at the ends", ("no values are lost, but the estimates near the ends are one-sided: revised as data "
                                "arrive, and possibly shifted in time")),
    )

    label: ClassVar[str] = "Christiano-Fitzgerald"
    topics: ClassVar[tuple[str, ...]] = ('trend-cycle-filters',)
    references: ClassVar[tuple[str, ...]] = (
        "Christiano, L. and Fitzgerald, T. (2003). The band pass filter. International Economic Review 44(2), 435-465. doi:10.1111/1468-2354.t01-1-00076",
    )

    low: Annotated[float | None, Widget("text", help="shortest cycle kept, in periods; empty = 1.5 years")] = None
    high: Annotated[float | None, Widget("text", help="longest cycle kept, in periods; empty = 8 years")] = None
    drift: Annotated[bool, Widget("checkbox", advanced=True, help="first subtract the straight line from the first to the last value (a random walk's drift)")] = True

    def settings(self, f: int) -> tuple[float, float]:
        return self.low or max(2.0, 1.5 * f), self.high or 8.0 * f

    def describe(self, f: int) -> str:
        low, high = self.settings(f)
        return f"ChristianoFitzgerald({low:g}, {high:g})"

    def cycle(self, y: np.ndarray, f: int) -> np.ndarray:
        low, high = self.settings(f)
        n = len(y)
        x = y - np.arange(n) * (y[-1] - y[0]) / (n - 1) if self.drift else y
        w1, w2 = 2 * np.pi / high, 2 * np.pi / low
        j = np.arange(1, n)
        B = np.r_[(w2 - w1) / np.pi, (np.sin(w2 * j) - np.sin(w1 * j)) / (np.pi * j)]
        cum = np.r_[0.0, np.cumsum(B[1:])]  # cum[m] = B_1 + ... + B_m
        c = np.empty(n)
        for t in range(n):  # 0-based t; t forward and t backward points
            ahead, behind = n - 1 - t, t
            weight_last = -0.5 * B[0] - cum[max(ahead - 1, 0)]
            weight_first = -0.5 * B[0] - cum[max(behind - 1, 0)]
            value = B[0] * x[t]
            if ahead > 1:
                value += B[1:ahead] @ x[t + 1 : n - 1]
            if behind > 1:
                value += B[1:behind] @ x[t - 1 : 0 : -1]
            c[t] = value + weight_last * x[-1] + weight_first * x[0]
        return c


@dataclass(frozen=True)
class Hamilton:
    """Hamilton (2018) regression filter: cycle = error in predicting y[t] from y[t-h] .. y[t-h-p+1].

    The trend is the fitted value of a least-squares regression of y[t] on a constant and the p values ending h
    periods earlier; the cycle, the part of the series that could not be predicted h periods ahead. Hamilton proposes
    it in place of the HP filter, whose cycles can have spurious dynamics. The defaults are his: two years ahead
    and a year of lags (8 and 4 for quarterly data). The first h + p - 1 periods get no trend or cycle, and as the
    regression uses the whole sample its coefficients change as data arrive.
    """

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import Dataset, Hamilton

        data = Dataset.example("sample_macro").derive(ly="log(y)")
        data = data.filter("ly", Hamilton())                     # quarterly: y[t] on y[t-8] .. y[t-11]; ly_cycle is the 2-year-ahead surprise
        data.frame[["ly", "ly_trend", "ly_cycle"]].dropna().head()  # the first h + p - 1 = 11 quarters have no trend or cycle
        one_year = data.filter("ly", Hamilton(h=4))              # what is unpredictable a year ahead
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("{name}_trend", "the fitted value of y[t] from y[t-h] .. y[t-h-p+1]: what could be predicted h periods ahead"),
        ("{name}_cycle", "the prediction error: what was unpredictable h periods ahead"),
        ("missing values at the ends", ("the first h + p - 1 values are missing (11 for quarterly data by default); none at "
                                "the end")),
    )

    label: ClassVar[str] = "Hamilton"
    topics: ClassVar[tuple[str, ...]] = ('trend-cycle-filters',)
    references: ClassVar[tuple[str, ...]] = (
        ("Hamilton, J. D. (2018). Why you should never use the Hodrick-Prescott filter. Review of Economics and Statistics 100(5), 831-843. doi:10.1162/rest_a_00706"),
    )

    h: Annotated[int | None, Widget("text", help="horizon in periods: the cycle is what is unpredictable this far ahead; empty = 2 years")] = None
    p: Annotated[int | None, Widget("text", advanced=True, help="lags in the regression; empty = one year (at most 12)")] = None

    def settings(self, f: int) -> tuple[int, int]:
        return self.h or 2 * f, self.p or min(f, 12)

    def describe(self, f: int) -> str:
        h, p = self.settings(f)
        return f"Hamilton(h={h}, p={p})"

    def trend(self, y: np.ndarray, f: int) -> np.ndarray:
        h, p = self.settings(f)
        n = len(y)
        rows = np.arange(h + p - 1, n)
        if len(rows) <= p + 1:
            raise SpecificationError(f"the Hamilton filter needs more than {h + 2 * p} observations")
        X = np.column_stack([np.ones(len(rows))] + [y[rows - h - i] for i in range(p)])
        beta = np.linalg.lstsq(X, y[rows], rcond=None)[0]
        out = np.full(n, np.nan)
        out[rows] = X @ beta
        return out


@dataclass(frozen=True)
class MovingAverage:
    """Moving-average trend.

    The trend is the mean of ``window`` values, either the last ones (trailing) or those around each period
    (centred); the cycle is the remainder. A trailing average uses only past data but lags the series by about half
    a window; a centred one does not lag but leaves about half a window missing at each end. With an even window the
    centred average reaches one period further back than forward. A window of one year's periods averages out a
    stable seasonal pattern.
    """

    maturity: ClassVar[str] = "stable"
    label: ClassVar[str] = "Moving average"
    topics: ClassVar[tuple[str, ...]] = ('trend-cycle-filters',)
    references: ClassVar[tuple[str, ...]] = (
        ("Hyndman, R. J. and Athanasopoulos, G. (2021). Forecasting: Principles and Practice, 3rd ed., chapter 3, \"Time series decomposition\". OTexts. https://otexts.com/fpp3/"),
    )

    window: Annotated[int, Widget("number", min=2, max=1000, help="periods averaged: longer, smoother trend; a year's periods remove stable seasonality")] = 4
    centered: Annotated[bool, Widget("checkbox", help="centre the window (otherwise trailing)")] = True

    def describe(self, f: int) -> str:
        return f"MovingAverage({self.window}{', centered' if self.centered else ''})"

    def trend(self, y: np.ndarray, f: int) -> np.ndarray:
        return pd.Series(y).rolling(self.window, center=self.centered).mean().to_numpy()


Filter = HP | BaxterKing | ChristianoFitzgerald | Hamilton | MovingAverage


def hp_trend(y: np.ndarray, lamb: float) -> np.ndarray:
    n = len(y)
    if n < 3:
        return y.astype(float).copy()
    D = sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n))
    return spsolve((sparse.eye(n) + lamb * (D.T @ D)).tocsc(), y)


def _trend_cycle(method: Filter, y: np.ndarray, f: int) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(method, "trend"):
        trend = method.trend(y, f)
        return trend, y - trend
    cycle = method.cycle(y, f)
    return y - cycle, cycle


def _one_series(method: Filter, s: pd.Series, f: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    values = s.to_numpy(dtype=float)
    ok = np.flatnonzero(np.isfinite(values))
    trend, cycle = np.full(len(values), np.nan), np.full(len(values), np.nan)
    if not len(ok):
        return trend, cycle
    first, last = ok[0], ok[-1]
    if len(ok) != last - first + 1:
        raise SpecificationError(f"{name} has missing values inside its sample; filters need an unbroken series")
    t, c = _trend_cycle(method, values[first : last + 1], f)
    trend[first : last + 1], cycle[first : last + 1] = t, c
    return trend, cycle


def trend_cycle(data: Dataset, name: str, method: Filter) -> tuple[pd.Series, pd.Series]:
    """Trend and cycle of one series, within each entity of a panel."""
    t = data.time_axis
    if t is None:
        raise SpecificationError("filters need a time axis")
    f = periods_per_year(data.frame.index.get_level_values(t.name).freqstr)
    frame = data.frame
    entities = [e.name for e in data.entity_axes]
    trend = pd.Series(np.nan, index=frame.index, name=f"{name}_trend")
    cycle = pd.Series(np.nan, index=frame.index, name=f"{name}_cycle")
    level = entities[0] if len(entities) == 1 else entities  # one axis: plain keys, not 1-tuples
    groups = frame[name].groupby(level=level) if entities else [(None, frame[name])]
    for key, s in groups:
        tr, cy = _one_series(method, s, f, name if key is None else f"{name} ({key})")
        trend.loc[s.index], cycle.loc[s.index] = tr, cy
    return trend, cycle


def apply(data: Dataset, names: str | list[str], method: Filter, derived: bool = True) -> Dataset:
    """Add ``{name}_trend`` and ``{name}_cycle``. With ``derived`` (the default) they keep their
    recipe and are recomputed when ``name`` changes; otherwise they are plain values."""
    from dataclasses import replace

    from .data import Dataset
    from .data.derived import Filtered
    from .metadata import Variable

    names = [names] if isinstance(names, str) else list(names)
    if derived:
        out = data.derive(**{f"{n}_{part}": Filtered(n, method, part) for n in names for part in ("trend", "cycle")})
        variables = dict(out.variables)
        for n in names:  # units follow the filtered series
            for part in ("trend", "cycle"):
                base = data.variables[n]
                variables[f"{n}_{part}"] = replace(variables[f"{n}_{part}"], units=base.units, scale=base.scale)
        return Dataset(out.frame, out.axes, variables, out.flags, out.meta)
    frame = data.frame.copy()
    variables = dict(data.variables)
    for name in names:
        trend, cycle = trend_cycle(data, name, method)
        frame[f"{name}_trend"], frame[f"{name}_cycle"] = trend, cycle
        base = data.variables[name]
        for part in ("trend", "cycle"):
            label = Filtered(name, method, part).describe(data)
            variables[f"{name}_{part}"] = Variable(f"{name}_{part}", units=base.units, scale=base.scale, derived_from=label)
    return Dataset(frame, data.axes, variables, data.flags, data.meta)


register("filters", HP)
register("filters", HP, name="hp_one_sided", label="HP (one-sided)", preset={"one_sided": True})
register("filters", BaxterKing)
register("filters", ChristianoFitzgerald)
register("filters", Hamilton)
register("filters", MovingAverage)
