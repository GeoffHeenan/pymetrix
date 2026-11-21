"""Returns from prices, drawdowns, and the performance of return series.

    data = returns(data, ["spx", "bonds"])                 # spx_ret, bonds_ret: log returns (kind="simple" for p/p[-1] - 1)
    perf = performance(data, ["spx_ret", "bonds_ret"], rf="tbill", benchmark="spx_ret")
    perf.table(); perf.plot()                              # the measures; cumulative returns and drawdowns

Returns are fractions (0.01 for 1%), log or simple. Annualising uses the data's frequency (252 trading days a year
for daily data, 52, 12, 4 or 1) unless ``periods_per_year`` is given: returns and their excess over ``rf`` are
averaged and multiplied by it (log returns), or compounded (simple), and volatility is scaled by its square root.
The Sharpe ratio's standard error is Lo's (2002) for independent returns, sqrt((1 + SR^2 / 2) / T) per period.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from pymetrix.extension import api
from pymetrix.errors import SpecificationError
from pymetrix.output import Table
from pymetrix.extension import plugin_context
from . import PLUGIN

if TYPE_CHECKING:
    from pymetrix.data import Dataset
    from pymetrix.figures import Figure

_PER_YEAR = {"B": 252, "D": 252, "W": 52, "M": 12, "Q": 4, "A": 1, "Y": 1}


def annual_periods(index: Any) -> int:
    """The periods in a year for annualising: 252 for daily data (trading days), 52, 12, 4 or 1."""
    freq = getattr(index, "freqstr", "") or ""
    base = freq.split("-")[0].lstrip("0123456789").upper()[:1]
    if base not in _PER_YEAR:
        raise SpecificationError(f"cannot tell how many periods make a year from the frequency {freq!r}: give periods_per_year")
    return _PER_YEAR[base]


def drawdown(r: pd.Series, kind: Literal["log", "simple"] = "log") -> pd.Series:
    """The fall from the highest value so far of the wealth the returns make (0 at a new high, -0.2 when 20% below)."""
    wealth = np.exp(r.cumsum()) if kind == "log" else (1 + r).cumprod()
    return (wealth / np.maximum.accumulate(wealth) - 1).rename(f"{r.name} drawdown")


def measures(r: np.ndarray, q: int, kind: str, rf: np.ndarray | None = None, bench: np.ndarray | None = None) -> dict[str, float]:
    """The performance measures of one return series (see the module notes)."""
    T = len(r)
    simple = np.expm1(r) if kind == "log" else r
    excess = r - (rf if rf is not None else 0.0)
    mean_ex, sd = float(excess.mean()), float(r.std(ddof=1))
    sr = mean_ex / sd if sd > 0 else np.nan
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    growth = float(np.prod(1 + simple))
    ann_return = growth ** (q / T) - 1
    dd = drawdown(pd.Series(r, name="r"), kind).to_numpy()
    out = {"annual return": ann_return, "annual volatility": sd * np.sqrt(q), "Sharpe ratio": sr * np.sqrt(q),
           "Sharpe std err": float(np.sqrt((1 + sr**2 / 2) / T) * np.sqrt(q)) if np.isfinite(sr) else np.nan,
           "Sortino ratio": mean_ex / downside * np.sqrt(q) if downside > 0 else np.nan,
           "maximum drawdown": float(dd.min()), "hit rate": float(np.mean(r > 0)), "skewness": float(sps.skew(r, bias=False)),
           "excess kurtosis": float(sps.kurtosis(r, bias=False)), "periods": float(T)}
    out["Calmar ratio"] = ann_return / -out["maximum drawdown"] if out["maximum drawdown"] < 0 else np.nan
    if bench is not None:
        b_ex = bench - (rf if rf is not None else 0.0)
        X = np.column_stack([np.ones(T), b_ex])
        coef = np.linalg.lstsq(X, excess, rcond=None)[0]
        active = r - bench
        te = float(active.std(ddof=1))
        out.update({"alpha (annual)": float(coef[0]) * q, "beta": float(coef[1]), "tracking error": te * np.sqrt(q),
                    "information ratio": float(active.mean()) / te * np.sqrt(q) if te > 0 else np.nan})
    return out


@dataclass(frozen=True, eq=False)
class Performance:
    """Performance measures of one or more return series, with their cumulative returns and drawdowns."""

    series: tuple[str, ...]
    kind: str
    per_year: int
    values: pd.DataFrame  # measure by series
    returns: pd.DataFrame = field(repr=False)
    rf: str | None = None
    benchmark: str | None = None

    stage = "finance"

    def table(self) -> Table:
        notes = [f"{self.kind.capitalize()} returns; annualised with {self.per_year} periods a year."]
        if self.rf:
            notes.append(f"Excess returns over {self.rf} for the Sharpe and Sortino ratios and alpha.")
        if self.benchmark:
            notes.append(f"Against {self.benchmark}: alpha and beta by least squares on excess returns; the information ratio "
                         "is the mean active return over its volatility.")
        notes.append("The Sharpe ratio's standard error is Lo's (2002), for independent returns.")
        return Table(list(self.values.columns), self.values.to_numpy().tolist(), index=list(self.values.index), index_name="measure",
                     title="Performance", notes=tuple(notes))

    def summary(self) -> Table:
        return self.table()

    def cumulative(self) -> pd.DataFrame:
        """The growth of 1 invested in each series."""
        r = self.returns
        return np.exp(r.cumsum()) if self.kind == "log" else (1 + r).cumprod()

    def plot(self) -> Figure:
        from pymetrix.figures import Figure, Line, Panel

        c = self.cumulative()
        d = pd.DataFrame({s: drawdown(self.returns[s], self.kind) for s in self.returns.columns})
        return Figure((Panel(tuple(Line(c[s], s) for s in c.columns), title="Growth of 1"),
                       Panel(tuple(Line(d[s], s) for s in d.columns), title="Drawdown")))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from pymetrix.extension import views_frame

            return views_frame(self, view)
        return self.values

    def __repr__(self) -> str:
        return f"<Performance of {', '.join(self.series)}>"


with plugin_context(PLUGIN):
    @api("finance:returns")
    def returns(data: Dataset, prices: Sequence[str], kind: Literal["log", "simple"] = "log", periods: int = 1,
                suffix: str = "_ret") -> Dataset:
        """The dataset with each price's return over ``periods`` added as ``<price><suffix>``: log (the change in its log)
        or simple (its ratio to ``periods`` earlier, less one)."""
        prices = [prices] if isinstance(prices, str) else list(prices)
        missing = [p for p in prices if p not in data.names]
        if missing:
            raise SpecificationError(f"no series {', '.join(map(repr, missing))} in the dataset")
        if kind not in ("log", "simple") or periods < 1:
            raise SpecificationError("kind is 'log' or 'simple', and periods at least 1")
        for p in prices:
            if (data[p].dropna() <= 0).any():
                raise SpecificationError(f"{p} has prices at or below zero: returns need positive prices")
        lag = f"[-{periods}]"
        exprs = {f"{p}{suffix}": (f"log({p}) - log({p}{lag})" if kind == "log" else f"{p} / {p}{lag} - 1") for p in prices}
        return data.with_series(**exprs)

    @api("finance:performance")
    def performance(data: Dataset, series: Sequence[str], rf: str | None = None, benchmark: str | None = None,
                    kind: Literal["log", "simple"] = "log", periods_per_year: int | None = None) -> Performance:
        """Performance measures of the return ``series``: annual return and volatility, Sharpe (with its standard error)
        and Sortino ratios over ``rf``, maximum drawdown, the Calmar ratio, hit rate, skewness and kurtosis, and, against
        a ``benchmark``, alpha, beta, tracking error and the information ratio."""
        series = [series] if isinstance(series, str) else list(series)
        needed = [*series, *([rf] if rf else []), *([benchmark] if benchmark else [])]
        missing = [s for s in needed if s not in data.names]
        if missing:
            raise SpecificationError(f"no series {', '.join(map(repr, missing))} in the dataset")
        if data.time_axis is None or data.kind != "time_series":
            raise SpecificationError("performance needs returns over time")
        frame = data.frame[list(dict.fromkeys(needed))].dropna()
        if len(frame) < 3:
            raise SpecificationError("too few periods with every series known")
        q = int(periods_per_year) if periods_per_year else annual_periods(frame.index)
        rf_values = frame[rf].to_numpy(float) if rf else None
        bench = frame[benchmark].to_numpy(float) if benchmark else None
        values = pd.DataFrame({s: measures(frame[s].to_numpy(float), q, kind, rf_values,
                                           bench if benchmark and s != benchmark else None) for s in series})
        order = [m for m in ("annual return", "annual volatility", "Sharpe ratio", "Sharpe std err", "Sortino ratio",
                             "maximum drawdown", "Calmar ratio", "hit rate", "skewness", "excess kurtosis", "alpha (annual)",
                             "beta", "tracking error", "information ratio", "periods") if m in values.index]
        return Performance(tuple(series), kind, q, values.loc[order], frame[series], rf, benchmark)
