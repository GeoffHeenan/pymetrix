"""Realised volatility from intraday returns, jump tests, and the HAR model of Corsi (2009).

    daily = realized_measures(data, "r")              # intraday returns -> rv, bv, jump statistics, one row a day
    fit = har(daily, "rv")                            # RV(t+1) on yesterday's, last week's and last month's RV
    fit.summary(); fit.forecast(22); fit.value_at_risk(0.99)

**Realised measures**, for each day of n intraday returns r_1..r_n:
    RV = sum r_j^2;   BV = (pi / 2) n / (n - 1) sum |r_j| |r_(j-1)|   (bipower variation, robust to jumps);
    TQ = n mu^-3 n / (n - 2) sum |r_j r_(j-1) r_(j-2)|^(4/3),  mu = 2^(2/3) Gamma(7/6) / Gamma(1/2)   (tripower quarticity);
    the Barndorff-Nielsen and Shephard (2006) ratio statistic for a jump that day,
    z = ((RV - BV) / RV) / sqrt((pi^2 / 4 + pi - 5) max(1, TQ / BV^2) / n),   standard normal without jumps;
and the jump's variation max(RV - BV, 0) on days the test rejects at ``jump_level``. RV estimates the day's integrated
variance plus its squared jumps; BV the integrated variance alone, though in a finite day a large jump still
leaks into it a little through its neighbouring returns.

**HAR**, by least squares with Newey-West standard errors: RV_(t+1) = b0 + b_d RV_t + b_w RV_t^(w) + b_m RV_t^(m), where
RV^(w) and RV^(m) are the averages of the last 5 and 22 days (``lags``); with ``log=True`` in logs. Forecasts h days
ahead iterate the fitted equation, with each average updated by the forecasts before it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gamma, pi
from typing import TYPE_CHECKING, Any, Sequence

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

_MU43 = 2 ** (2 / 3) * gamma(7 / 6) / gamma(1 / 2)  # E|Z|^(4/3)


def day_measures(r: np.ndarray, jump_level: float = 0.999) -> dict[str, float]:
    """The realised measures of one day's intraday returns (see the module notes)."""
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3:
        return {k: np.nan for k in ("rv", "bv", "tq", "jump z", "jump p", "jump", "n")}
    a = np.abs(r)
    rv = float(np.sum(r**2))
    bv = float(pi / 2 * n / (n - 1) * np.sum(a[1:] * a[:-1]))
    tq = float(n * _MU43**-3 * n / (n - 2) * np.sum((a[2:] * a[1:-1] * a[:-2]) ** (4 / 3)))
    z = ((rv - bv) / rv) / np.sqrt((pi**2 / 4 + pi - 5) * max(1.0, tq / bv**2) / n) if rv > 0 and bv > 0 else np.nan
    p = float(sps.norm.sf(z)) if np.isfinite(z) else np.nan
    return {"rv": rv, "bv": bv, "tq": tq, "jump z": float(z), "jump p": p,
            "jump": max(rv - bv, 0.0) if np.isfinite(p) and p < 1 - jump_level else 0.0, "n": float(n)}


def har_design(rv: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    """The HAR regressors at each day t: RV_t and the averages of the last lags[1], lags[2], ... days."""
    s = pd.Series(rv)
    return np.column_stack([s.rolling(int(k)).mean().to_numpy() for k in lags])


@dataclass(frozen=True, eq=False)
class FittedHAR:
    """A heterogeneous autoregression of realised variance (Corsi 2009)."""

    series: str
    lags: tuple[int, ...]
    log: bool
    fit: Any = field(repr=False)  # the FittedEquation
    rv: pd.Series = field(repr=False)  # the realised variance, as given
    names: tuple[str, ...] = ()

    stage = "finance"

    @property
    def params(self) -> pd.Series:
        return pd.Series(self.fit.params.values, index=list(self.fit.params.names))

    def _transform(self, x: np.ndarray) -> np.ndarray:
        return np.log(x) if self.log else x

    def forecast(self, h: int = 22) -> pd.Series:
        """Realised variance forecast 1 to ``h`` days ahead (in levels; from logs by exponentiating, with no correction)."""
        b = self.params.to_numpy()
        hist = list(self._transform(self.rv.dropna().to_numpy(float)))
        out = []
        for _ in range(h):
            x = [1.0] + [float(np.mean(hist[-k:])) for k in self.lags]
            nxt = float(np.dot(b, x))
            hist.append(nxt)
            out.append(np.exp(nxt) if self.log else nxt)
        last = self.rv.dropna().index[-1]
        index = pd.period_range(last + 1, periods=h, freq=last.freq) if isinstance(last, pd.Period) else pd.RangeIndex(1, h + 1)
        return pd.Series(out, index=index, name=f"{self.series} forecast")

    def value_at_risk(self, level: float = 0.99, mean: float = 0.0) -> dict[str, float]:
        """Tomorrow's normal VaR and ES of the daily return, with the HAR's variance forecast."""
        from .risk import normal

        sd = float(np.sqrt(self.forecast(1).iloc[0]))
        v, e = normal(mean, sd, 1 - level)
        return {"var": float(v), "es": float(e), "sd": sd}

    def summary(self) -> Table:
        f = self.fit
        rows = [[float(v), float(s), float(v / s) if s > 0 else np.nan] for v, s in zip(f.params.values, f.params.se)]
        return Table(["coefficient", "std err", "t"], rows, index=list(self.names), index_name="", title="HAR",
                     notes=(f"{'log ' if self.log else ''}{self.series} on its averages over the last {', '.join(map(str, self.lags))} days; "
                            f"Newey-West standard errors; R2 {float(f.stats.get('r2', np.nan)):.4f}, {int(f.stats.get('nobs', 0))} days.",))

    def plot(self) -> Figure:
        from pymetrix.figures import Figure, Line, Panel

        fitted = self.fit.fitted
        fitted = np.exp(fitted) if self.log else fitted
        actual = self.rv.reindex(fitted.index)
        return Figure((Panel((Line(actual, self.series), Line(fitted.rename("fitted"), "fitted")), title="Realised variance and the HAR fit"),
                       Panel((Line(self.forecast(22), "forecast"),), title="Forecast")))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from pymetrix.extension import views_frame

            return views_frame(self, view)
        return self.summary().to_frame()

    def __repr__(self) -> str:
        return f"<FittedHAR of {self.series}, lags {self.lags}>"


with plugin_context(PLUGIN):
    @api("finance:realized_measures")
    def realized_measures(data: Dataset, series: str, jump_level: float = 0.999, annualize: int = 252) -> Dataset:
        """Each day's realised variance (rv), bipower variation (bv), tripower quarticity, the BNS jump statistic and its
        p-value, the jump variation on days it rejects at ``jump_level``, the intraday returns counted, and the
        annualised realised volatility (rvol), from the intraday returns ``series``."""
        from pymetrix.data import Dataset as _D

        if series not in data.names:
            raise SpecificationError(f"no series {series!r} in the dataset")
        index = data.frame.index
        if not isinstance(index, pd.PeriodIndex):
            raise SpecificationError("realised measures need intraday returns over time")
        r = data.frame[series]
        days = index.to_timestamp().normalize()
        if len(set(days)) == len(days):
            raise SpecificationError("each day has one return: realised measures need several returns a day (intraday data)")
        rows = {pd.Period(d, "D"): day_measures(g.to_numpy(float), jump_level) for d, g in r.groupby(days)}
        out = pd.DataFrame.from_dict(rows, orient="index")
        out.index = pd.PeriodIndex(out.index, freq="D", name="time")
        out["rvol"] = np.sqrt(out["rv"] * annualize)
        return _D.from_frame(out)

    @api("finance:har")
    def har(data: Dataset, series: str = "rv", lags: Sequence[int] = (1, 5, 22), log: bool = False) -> FittedHAR:
        """The HAR model of realised variance ``series``: tomorrow's on today's and its averages over the ``lags`` days,
        by least squares with Newey-West standard errors."""
        from pymetrix.data import Dataset as _D
        from pymetrix.estimation import OLS, Equation

        lags = tuple(int(k) for k in lags)
        if series not in data.names:
            raise SpecificationError(f"no series {series!r} in the dataset")
        if not lags or lags[0] != 1 or list(lags) != sorted(set(lags)):
            raise SpecificationError("lags start at 1 and rise, as (1, 5, 22)")
        rv = data[series].dropna()
        if (rv <= 0).any() and log:
            raise SpecificationError(f"{series} has values at or below zero: no logs")
        y = np.log(rv.to_numpy(float)) if log else rv.to_numpy(float)
        if len(y) < lags[-1] + 30:
            raise SpecificationError(f"{len(y)} days are too few for a HAR with a {lags[-1]}-day average")
        X = har_design(y, lags)
        names = ["d" if k == 1 else f"avg{k}" for k in lags]
        frame = pd.DataFrame(X[:-1], columns=names, index=rv.index[1:])
        frame.insert(0, "target", y[1:])
        fit = Equation(f"target ~ {' + '.join(names)}").fit(_D.from_frame(frame.dropna()), method=OLS(cov="hac"))
        labels = ("const", *(f"{'log ' if log else ''}RV {'day' if k == 1 else f'{k}-day average'}" for k in lags))
        return FittedHAR(series, lags, bool(log), fit, rv, labels)
