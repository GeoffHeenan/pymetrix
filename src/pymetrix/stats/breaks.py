"""Structural breaks at unknown dates: the Quandt-Andrews sup-F test and Bai-Perron's multiple breaks.

    fit.test("quandt_andrews")                 # the largest Chow F over the middle 70% of the sample; its break date
    breaks = fit.breaks(max_breaks=5)          # Bai-Perron: the best dates for 0 to 5 breaks, the number chosen by BIC
    breaks.dates; breaks.segments(); breaks.plot()

Both look for a change in all of a linear equation's coefficients. Quandt-Andrews' p-value is from the limiting
distribution (Andrews, 1993), the supremum of a normalised Brownian bridge, read from a table shipped with the
package (``quandt_andrews.npz``, made by ``scripts/make_quandt_andrews_table.py``: 1 to 20 coefficients, trimming
5% to 45% in steps of 5%) and simulated for any other case; Bai-Perron's dates minimise the sum of squared residuals over all partitions with
segments of at least ``trim`` of the sample, by dynamic programming (Bai and Perron, 2003), and the number of
breaks is chosen by the Bayesian information criterion, counting each date as a parameter (as R's strucchange).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

import numpy as np
import pandas as pd

from .._api import api
from ..errors import SpecificationError
from ..estimation import FittedEquation
from ..forms import Widget
from ..output import Table
from .base import Test, TestResult, register
from .equation_tests import _ols_parts
from .._random import generator as make_rng

if TYPE_CHECKING:
    from ..figures import Figure


def _time_series_parts(fit: FittedEquation, what: str) -> tuple[np.ndarray, np.ndarray, pd.PeriodIndex]:
    y, X, _ = _ols_parts(fit, what, linear=True)
    index = fit.resid.index  # type: ignore[union-attr]
    if not isinstance(index, pd.PeriodIndex):
        raise SpecificationError(f"{what} needs a single time series")
    return y, X, index


def _ssr(y: np.ndarray, X: np.ndarray) -> float:
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ b
    return float(u @ u)


@lru_cache(maxsize=32)
def _sup_wald_null(k: int, trim: float, reps: int = 20000, steps: int = 5000, seed: int = 19930701) -> np.ndarray:
    """Draws of sup over r in [trim, 1 - trim] of |B(r) - r B(1)|^2 / (r (1 - r)), B a k-dimensional Brownian
    motion: the limit of Quandt-Andrews' sup-Wald statistic (Andrews, 1993), sorted."""
    rng = make_rng(seed)
    r = np.arange(1, steps + 1) / steps
    inside = (r >= trim) & (r <= 1 - trim)
    rr = r[inside]
    out = np.empty(reps)
    chunk = max(1, 200_000 // (steps * k))  # draws at a time: memory stays small
    for start in range(0, reps, chunk):
        m = min(chunk, reps - start)
        B = np.cumsum(rng.standard_normal((m, steps, k)), axis=1) / np.sqrt(steps)
        bridge = B[:, inside, :] - rr[None, :, None] * B[:, -1:, :]
        out[start:start + m] = np.max(np.sum(bridge**2, axis=2) / (rr * (1 - rr))[None, :], axis=1)
    return np.sort(out)


_TABLE = Path(__file__).with_name("quandt_andrews.npz")


@lru_cache(maxsize=1)
def _table() -> dict[str, np.ndarray]:
    with np.load(_TABLE) as f:
        return {name: f[name] for name in f.files}


@lru_cache(maxsize=32)
def _null_quantiles(k: int, trim: float) -> tuple[np.ndarray, np.ndarray, bool]:
    """Probabilities and the null's quantiles at them (of the sup-Wald, k times sup F), and whether they come from
    the shipped table (else from a fresh simulation)."""
    t = _table()
    j = np.flatnonzero(np.isclose(t["trim"], trim, atol=1e-9))
    if 1 <= k <= int(t["k"][-1]) and len(j):
        return t["probs"], t["quantiles"][k - 1, j[0]].astype(float), True
    probs = t["probs"]
    return probs, np.quantile(_sup_wald_null(k, trim), probs), False


def sup_f_path(y: np.ndarray, X: np.ndarray, trim: float) -> tuple[np.ndarray, np.ndarray]:
    """The Chow F statistic for a break before each candidate observation (the first of the second regime)."""
    n, k = X.shape
    lo, hi = int(np.ceil(trim * n)), int(np.floor((1 - trim) * n))
    lo, hi = max(lo, k + 1), min(hi, n - k - 1)
    if lo > hi:
        raise SpecificationError(f"too few observations for {k} coefficients with trimming {trim}")
    ssr_r = _ssr(y, X)
    cands = np.arange(lo, hi + 1)
    F = np.empty(len(cands))
    for i, t in enumerate(cands):
        ssr_u = _ssr(y[:t], X[:t]) + _ssr(y[t:], X[t:])
        F[i] = ((ssr_r - ssr_u) / k) / (ssr_u / (n - 2 * k))
    return cands, F


@register
@dataclass(frozen=True)
class QuandtAndrews(Test):
    """Quandt-Andrews test for a break at an unknown date: the largest Chow F statistic over the candidate dates
    in the middle of the sample (all but ``trim`` at each end), with Andrews' asymptotic p-value.

    The null is that all the coefficients are constant; the alternative is one break at an unknown date. Because the
    date is the one where the Chow F is largest, the statistic does not have an F distribution: the p-value and
    critical values come from Andrews' limiting distribution for this number of coefficients and trimming. The date
    of the peak, in the details, estimates the break date; the exp F and average F statistics are there too, without
    p-values. The F statistics assume a constant error variance, and Andrews' distribution assumes stationary
    regressors, so with a time trend or integrated regressors the p-value is only approximate. For more than one
    break, use Bai-Perron's method.
    """

    maturity: ClassVar[str] = "beta"
    trim: Annotated[float, Widget("number", min=0.05, max=0.45, step=0.05, help="share of the sample left out at each end; 0.15 is usual, smaller searches nearer the ends")] = 0.15

    example: ClassVar[str] = """
        from pymetrix import Dataset, Equation

        nile = Dataset.example("nile")           # the Nile's annual flow, 1871-1970
        fit = Equation("volume ~ 1").fit(nile)
        fit.test("quandt_andrews")               # sup F, its p-value and critical values; the break date in the details
        fit.test("quandt_andrews", trim=0.05)    # search nearer the ends of the sample
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("statistic", ("sup F: the largest Chow F over the candidate dates; it does not follow the F distribution its "
                       "df suggest")),
        ("p-value, critical value 10%, 5%, 1%", ("from Andrews' distribution for this number of coefficients and "
                                                 "trimming; a small p-value rejects constant coefficients")),
        ("break date", "the date of the largest F: the estimate of when the break happened"),
        ("exp F, average F", "other summaries of the F path, which weigh all dates; given without p-values"),
        ("note", "present when the statistic is beyond the tabulated range: the p-value is then an upper bound"),
    )

    name = "quandt_andrews"
    topics = ("diagnostic-tests",)
    references = (
        ("Quandt, R. E. (1960). Tests of the hypothesis that a linear regression system obeys two separate regimes. Journal of the American Statistical Association 55(290), 324-330. doi:10.1080/01621459.1960.10482067"),
        ("Andrews, D. W. K. (1993). Tests for parameter instability and structural change with unknown change point. Econometrica 61(4), 821-856. doi:10.2307/2951764"),
        ("Andrews, D. W. K. (2003). Tests for parameter instability and structural change with unknown change point: a corrigendum. Econometrica 71(1), 395-397. doi:10.1111/1468-0262.00405"),
        ("Andrews, D. W. K. and Ploberger, W. (1994). Optimal tests when a nuisance parameter is present only under the alternative. Econometrica 62(6), 1383-1414. doi:10.2307/2951753"),
    )
    null = "no structural break (the coefficients are the same throughout)"
    applies_to = (FittedEquation,)

    def run(self, fit: FittedEquation) -> TestResult:
        if not 0 < self.trim < 0.5:
            raise SpecificationError("trim is a share between 0 and 0.5")
        y, X, index = _time_series_parts(fit, "the Quandt-Andrews test")
        k = X.shape[1]
        cands, F = sup_f_path(y, X, self.trim)
        i = int(np.argmax(F))
        stat = float(F[i])
        probs, quantiles, tabulated = _null_quantiles(k, round(self.trim, 4))
        p = float(1 - np.interp(k * stat, quantiles, probs))
        crit = {f"{int(100 * a)}%": float(np.interp(1 - a, probs, quantiles) / k) for a in (0.10, 0.05, 0.01)}
        source = "tabulated" if tabulated else "simulated"
        details = {"break date": str(index[cands[i]]), "exp F": float(np.log(np.mean(np.exp(np.minimum(k * F / 2, 700))))),
                   "average F": float(np.mean(F))}
        if k * stat > quantiles[-1]:  # beyond the largest quantile kept: the p-value is a bound
            details["note"] = f"the statistic is beyond the {source} distribution's range: the p-value is below {p:.4f}"
        return TestResult("Quandt-Andrews (sup F)", self.null, stat, (k, len(y) - 2 * k), p,
                          f"Andrews (1993), {k} coefficients, {self.trim:.0%} trimming ({source})", critical_values=crit,
                          details=details)


# -- Bai-Perron ------------------------------------------------------------------------------------------------------

def _segment_ssr(y: np.ndarray, X: np.ndarray, h: int) -> np.ndarray:
    """SSR[i, j]: the sum of squared residuals of a regression on observations i to j (inclusive), for segments of
    at least h observations (inf otherwise), from cumulative cross products."""
    n, k = X.shape
    SXX = np.concatenate([np.zeros((1, k, k)), np.cumsum(X[:, :, None] * X[:, None, :], axis=0)])
    SXy = np.concatenate([np.zeros((1, k)), np.cumsum(X * y[:, None], axis=0)])
    Syy = np.concatenate([[0.0], np.cumsum(y * y)])
    out = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + h - 1, n):
            A, b = SXX[j + 1] - SXX[i], SXy[j + 1] - SXy[i]
            coef = np.linalg.lstsq(A, b, rcond=None)[0]
            out[i, j] = max(float(Syy[j + 1] - Syy[i] - coef @ b), 0.0)
    return out


def partition(y: np.ndarray, X: np.ndarray, m: int, h: int, ssr: np.ndarray | None = None) -> tuple[float, list[int]]:
    """The m break points (each the first observation of a new segment) minimising the total SSR, by dynamic
    programming over the segment sums of squares."""
    n = len(y)
    S = _segment_ssr(y, X, h) if ssr is None else ssr
    if m == 0:
        return float(S[0, n - 1]), []
    if (m + 1) * h > n:
        raise SpecificationError(f"{m} breaks need {(m + 1) * h} observations with segments of {h}")
    best = S[0].copy()  # best[j]: the least SSR of observations 0..j in one segment, then with more breaks
    where: list[np.ndarray] = []
    for _ in range(m):
        new = np.full(n, np.inf)
        arg = np.zeros(n, dtype=int)
        for j in range(n):
            ts = np.arange(h, j - h + 2)  # the new segment starts at t and ends at j
            if len(ts) == 0:
                continue
            values = best[ts - 1] + S[ts, j]
            i = int(np.argmin(values))
            new[j], arg[j] = values[i], ts[i]
        where.append(arg)
        best = new
    dates, j = [], n - 1
    for arg in reversed(where):
        t = int(arg[j])
        dates.append(t)
        j = t - 1
    return float(best[n - 1]), sorted(dates)


@dataclass(frozen=True, eq=False)
class StructuralBreaks:
    """Bai-Perron multiple breaks in a linear equation: the best dates for each number of breaks, and the number
    chosen by BIC."""

    fit: FittedEquation = field(repr=False)
    trim: float
    by_count: pd.DataFrame  # for each number of breaks: SSR, log likelihood, BIC and the dates
    chosen: int
    breakpoints: tuple[int, ...]  # the chosen dates, as positions of the first observation of each new segment
    index: pd.PeriodIndex = field(repr=False)

    @property
    def dates(self) -> list[str]:
        """The chosen break dates: the first period of each new regime."""
        return [str(self.index[t]) for t in self.breakpoints]

    def segments(self) -> Table:
        """Each regime's coefficients, standard errors and periods."""
        y, X, _ = _time_series_parts(self.fit, "Bai-Perron breaks")
        names = list(self.fit.design.columns)  # type: ignore[union-attr]
        edges = [0, *self.breakpoints, len(y)]
        rows, index = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            Xs, ys = X[a:b], y[a:b]
            coef = np.linalg.lstsq(Xs, ys, rcond=None)[0]
            u = ys - Xs @ coef
            s2 = float(u @ u) / max(len(ys) - Xs.shape[1], 1)
            se = np.sqrt(np.diag(s2 * np.linalg.pinv(Xs.T @ Xs)))
            for name, c, s in zip(names, coef, se):
                index.append(f"{self.index[a]} to {self.index[b - 1]}: {name}")
                rows.append([float(c), float(s)])
        return Table(["coef", "std err"], rows, index=index, index_name="regime: coefficient", title="Coefficients in each regime",
                     notes=(f"{self.chosen} break{'s' if self.chosen != 1 else ''} chosen by BIC; standard errors within each regime.",))

    def summary(self) -> Table:
        rows = [[int(m), float(r["SSR"]), float(r["BIC"]), r["dates"] or "none"] for m, r in self.by_count.iterrows()]
        return Table(["SSR", "BIC", "dates"], [r[1:] for r in rows], index=[str(r[0]) for r in rows], index_name="breaks",
                     title=f"Bai-Perron breaks: {self.fit.equation.text}",
                     notes=(f"Chosen by BIC: {self.chosen} ({', '.join(self.dates) or 'no break'}); segments of at least {self.trim:.0%} "
                            "of the sample; all coefficients may change.",))

    def fitted(self) -> pd.Series:
        y, X, _ = _time_series_parts(self.fit, "Bai-Perron breaks")
        edges = [0, *self.breakpoints, len(y)]
        out = np.empty(len(y))
        for a, b in zip(edges[:-1], edges[1:]):
            out[a:b] = X[a:b] @ np.linalg.lstsq(X[a:b], y[a:b], rcond=None)[0]
        return pd.Series(out, index=self.index, name="fitted, by regime")

    def plot(self) -> Figure:
        from ..figures import Figure, Line, Panel, Rule

        y, _, _ = _time_series_parts(self.fit, "Bai-Perron breaks")
        actual = pd.Series(y, index=self.index, name=self.fit.dependent or "actual")
        layers = [Line(actual, actual.name, color="#222222"), Line(self.fitted(), "fitted, by regime")]
        layers += [Rule(y=self.index[t], vertical=True, style="--") for t in self.breakpoints]
        return Figure((Panel(tuple(layers), title=f"Breaks chosen by BIC: {', '.join(self.dates) or 'none'}"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from ..views import _views_to_frame

            return _views_to_frame(self, view)
        return self.by_count

    def __repr__(self) -> str:
        return f"<StructuralBreaks {self.chosen} chosen ({', '.join(self.dates) or 'none'}) for {self.fit.equation.text}>"


@api("pymetrix:bai_perron")
def bai_perron(fit: Any, max_breaks: int = 5, trim: float = 0.15) -> StructuralBreaks:
    """Bai-Perron multiple structural breaks in a linear equation (see the module notes)."""
    if not 0 < trim < 0.5:
        raise SpecificationError("trim is a share between 0 and 0.5")
    y, X, index = _time_series_parts(fit, "Bai-Perron breaks")
    n, k = X.shape
    h = max(int(np.floor(trim * n)), k + 1)
    top = min(int(max_breaks), n // h - 1)
    if top < 1:
        raise SpecificationError(f"too few observations for a break with segments of {h}")
    from ..progress import report

    report(None, "Bai-Perron: segment sums of squares")
    S = _segment_ssr(y, X, h)
    rows = {}
    for m in range(top + 1):
        report(m / (top + 1), f"Bai-Perron: the best {m} break{'s' if m != 1 else ''}")
        ssr, dates = partition(y, X, m, h, S)
        loglike = -n / 2 * (np.log(2 * np.pi) + np.log(ssr / n) + 1)
        df = (m + 1) * k + m + 1  # coefficients in each regime, the dates and the variance
        rows[m] = {"SSR": ssr, "loglike": loglike, "BIC": -2 * loglike + df * np.log(n),
                   "dates": ", ".join(str(index[t]) for t in dates), "positions": dates}
    table = pd.DataFrame(rows).T
    table.index.name = "breaks"
    chosen = int(pd.to_numeric(table["BIC"]).idxmin())
    return StructuralBreaks(fit, trim, table.drop(columns="positions"), chosen, tuple(rows[chosen]["positions"]), index)
