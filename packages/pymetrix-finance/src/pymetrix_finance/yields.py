"""Yield curves: Nelson-Siegel and Svensson, fitted period by period (Diebold and Li, 2006).

    yc = yield_curve(data, ["y3m", "y1", "y2", "y5", "y10"], [0.25, 1, 2, 5, 10])   # level, slope, curvature by period
    yc.factors; yc.curve("2020-06"); yc.forecast(12)                                # factors; a fitted curve; ahead
    yc.plot()

The yield at maturity m (years) is
    b0 + b1 (1 - e^(-l m)) / (l m) + b2 ((1 - e^(-l m)) / (l m) - e^(-l m))           (Nelson-Siegel)
plus b3 ((1 - e^(-l2 m)) / (l2 m) - e^(-l2 m)) for Svensson. b0 is the level (the yield at long maturities), b1 the
slope (short less long, as m -> 0 the yield tends to b0 + b1) and b2 (and b3) the curvature. The decay ``lam`` (and
``lam2``) is fixed or, when not given, one value for the whole sample by nonlinear least squares; given the decays,
each period's factors are least squares on that period's yields. ``forecast`` follows Diebold and Li: an AR(1) for
each factor, by least squares, and the curves it implies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import optimize

from pymetrix.extension import api
from pymetrix.errors import SpecificationError
from pymetrix.output import Table
from pymetrix.extension import plugin_context
from . import PLUGIN

if TYPE_CHECKING:
    from pymetrix.data import Dataset
    from pymetrix.figures import Figure


def loadings(m: Any, lam: float, lam2: float | None = None) -> np.ndarray:
    """The factor loadings at maturities ``m``: columns level, slope, curvature (and a second curvature with ``lam2``)."""
    m = np.asarray(m, float)
    x = lam * m
    slope = np.where(x > 1e-10, (1 - np.exp(-x)) / np.where(x > 1e-10, x, 1.0), 1.0)
    cols = [np.ones_like(m), slope, slope - np.exp(-x)]
    if lam2 is not None:
        x2 = lam2 * m
        s2 = np.where(x2 > 1e-10, (1 - np.exp(-x2)) / np.where(x2 > 1e-10, x2, 1.0), 1.0)
        cols.append(s2 - np.exp(-x2))
    return np.column_stack(cols)


def _fit_factors(Y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each row's factors by least squares, skipping missing yields; and the residuals."""
    B = np.full((len(Y), X.shape[1]), np.nan)
    resid = np.full(Y.shape, np.nan)
    for t, y in enumerate(Y):
        ok = np.isfinite(y)
        if ok.sum() > X.shape[1]:
            B[t] = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
            resid[t, ok] = y[ok] - X[ok] @ B[t]
    return B, resid


def _sse(Y: np.ndarray, m: np.ndarray, lam: float, lam2: float | None) -> float:
    _, e = _fit_factors(Y, loadings(m, lam, lam2))
    return float(np.nansum(e**2))


@dataclass(frozen=True, eq=False)
class YieldCurve:
    """Nelson-Siegel or Svensson factors of a yield curve, period by period."""

    series: tuple[str, ...]
    maturities: np.ndarray
    model: str
    lam: float
    lam2: float | None
    factors: pd.DataFrame  # level, slope, curvature (curvature 2) by period
    yields: pd.DataFrame = field(repr=False)
    residuals: pd.DataFrame = field(repr=False)
    estimated: bool = False

    stage = "finance"

    def fitted(self) -> pd.DataFrame:
        X = loadings(self.maturities, self.lam, self.lam2)
        return pd.DataFrame(self.factors.to_numpy() @ X.T, index=self.factors.index, columns=list(self.series))

    def curve(self, period: Any = None, maturities: Any = None) -> pd.Series:
        """The fitted curve at one period (default: the last), at ``maturities`` (default: 3 months to the longest)."""
        f = self.factors.dropna()
        row = f.iloc[-1] if period is None else f.loc[pd.Period(period, f.index.freq) if isinstance(f.index, pd.PeriodIndex) else period]
        m = np.asarray(maturities, float) if maturities is not None else np.linspace(min(0.25, self.maturities.min()), self.maturities.max(), 60)
        return pd.Series(loadings(m, self.lam, self.lam2) @ row.to_numpy(), index=pd.Index(m, name="maturity"), name="yield")

    def ar1(self) -> pd.DataFrame:
        """Each factor's AR(1), f_t = c + phi f_(t-1) + e_t, by least squares."""
        rows = {}
        for c in self.factors.columns:
            f = self.factors[c].dropna().to_numpy()
            X = np.column_stack([np.ones(len(f) - 1), f[:-1]])
            rows[c] = np.linalg.lstsq(X, f[1:], rcond=None)[0]
        return pd.DataFrame(rows, index=["const", "phi"]).T

    def forecast(self, h: int = 12) -> pd.DataFrame:
        """The yields forecast 1 to ``h`` periods ahead from each factor's AR(1)."""
        coef = self.ar1()
        f = self.factors.dropna().iloc[-1].to_numpy()
        out = []
        for _ in range(h):
            f = coef["const"].to_numpy() + coef["phi"].to_numpy() * f
            out.append(f.copy())
        F = np.array(out)
        last = self.factors.dropna().index[-1]
        index = pd.period_range(last + 1, periods=h, freq=last.freq) if isinstance(last, pd.Period) else pd.RangeIndex(1, h + 1, name="h")
        return pd.DataFrame(F @ loadings(self.maturities, self.lam, self.lam2).T, index=index, columns=list(self.series))

    def forwards(self, maturities: Any, period: Any = None) -> pd.Series:
        """Instantaneous forward rates at ``maturities`` from one period's fitted curve (default: the last)."""
        from .termstructure import forwards

        return forwards(self, maturities, period)

    def discount(self, maturities: Any, period: Any = None, percent: bool = True) -> pd.Series:
        """Discount factors exp(-y(m) m), reading the curve as continuously compounded zero rates (in percent unless
        ``percent=False``)."""
        from .termstructure import discount

        return discount(self, maturities, period, percent)

    def bond(self, coupon: float, maturity: float, freq: int = 2, period: Any = None, percent: bool = True) -> dict[str, float]:
        """A coupon bond's price per 100, continuous yield, duration, convexity and Fisher-Weil duration from the curve."""
        from .termstructure import bond

        return bond(self, coupon, maturity, freq, period, percent)

    def par_coupon(self, maturity: float, freq: int = 2, period: Any = None, percent: bool = True) -> float:
        """The coupon (percent a year) that prices a bond of ``maturity`` at par."""
        from .termstructure import par_coupon

        return par_coupon(self, maturity, freq, period, percent)

    def table(self) -> Table:
        r = self.residuals
        rows = [[float(m), float(np.sqrt(np.nanmean(r[s] ** 2))), float(np.nanmean(r[s]))] for s, m in zip(self.series, self.maturities)]
        decay = f"lambda {self.lam:.4f}" + (f", lambda2 {self.lam2:.4f}" if self.lam2 is not None else "")
        means = ", ".join(f"{c} {self.factors[c].mean():.4f}" for c in self.factors.columns)
        return Table(["maturity", "RMSE", "mean error"], rows, index=list(self.series), index_name="series",
                     title=f"Yield curve: {self.model.replace('_', '-').title()}",
                     notes=(f"Decay {decay} ({'estimated over the sample' if self.estimated else 'given'}), maturities in years.",
                            f"Average factors: {means}.", f"{int(self.factors.notna().all(axis=1).sum())} periods fitted."))

    def summary(self) -> Table:
        return self.table()

    def plot(self) -> Figure:
        from pymetrix.figures import Figure, Line, Panel, Scatter

        f = self.factors
        last = f.dropna().index[-1]
        seen = self.yields.loc[last]
        observed = Scatter(pd.Series(self.maturities), pd.Series(seen.to_numpy()), "observed")
        return Figure((Panel(tuple(Line(f[c], c) for c in f.columns), title="Factors"),
                       Panel((Line(self.curve(), "fitted"), observed), title=f"The curve at {last}", xlabel="maturity (years)")))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from pymetrix.extension import views_frame

            return views_frame(self, view)
        return self.factors

    def __repr__(self) -> str:
        return f"<YieldCurve {self.model} of {len(self.series)} maturities, lambda {self.lam:.4f}>"


with plugin_context(PLUGIN):
    @api("finance:yield_curve")
    def yield_curve(data: Dataset, series: Sequence[str], maturities: Sequence[float],
                    model: Literal["nelson_siegel", "svensson"] = "nelson_siegel", lam: float | None = None,
                    lam2: float | None = None) -> YieldCurve:
        """Nelson-Siegel (or Svensson) level, slope and curvature of the yields ``series`` at ``maturities`` (years),
        each period, with the decay ``lam`` (and ``lam2``) given or estimated over the whole sample."""
        series = [series] if isinstance(series, str) else list(series)
        m = np.asarray(maturities, float)
        missing = [s for s in series if s not in data.names]
        if missing:
            raise SpecificationError(f"no series {', '.join(map(repr, missing))} in the dataset")
        if len(m) != len(series) or np.any(m <= 0):
            raise SpecificationError("give one positive maturity (in years) for each series")
        if model not in ("nelson_siegel", "svensson"):
            raise SpecificationError("model is 'nelson_siegel' or 'svensson'")
        k = 3 if model == "nelson_siegel" else 4
        if len(series) <= k:
            raise SpecificationError(f"{model.replace('_', '-')} needs more than {k} maturities")
        yields = data.frame[series].dropna(how="all")
        Y = yields.to_numpy(float)
        estimated = lam is None or (model == "svensson" and lam2 is None)
        if model == "nelson_siegel":
            if lam is None:  # a grid over where the curvature peaks (3 months to 15 years), then refined
                grid = 1.7932 / np.geomspace(0.25, 15, 40)
                start = grid[int(np.argmin([_sse(Y, m, g, None) for g in grid]))]
                found = optimize.minimize_scalar(lambda v: _sse(Y, m, v, None), bounds=(start / 1.5, start * 1.5), method="bounded",
                                                 options={"xatol": 1e-8})
                lam = float(found.x)
            lam2 = None
        else:
            given = (lam, lam2)
            if estimated:
                grid = 1.7932 / np.geomspace(0.25, 15, 15)
                pairs = [(a, b) for a in grid for b in grid if a > b * 1.2]
                a0, b0 = min(pairs, key=lambda p: _sse(Y, m, p[0] if given[0] is None else given[0], p[1] if given[1] is None else given[1]))

                def objective(v: np.ndarray) -> float:
                    l1 = np.exp(v[0]) if given[0] is None else given[0]
                    l2 = np.exp(v[-1]) if given[1] is None else given[1]
                    return _sse(Y, m, l1, l2) if abs(l1 - l2) > 1e-3 * l1 else 1e12

                v0 = [np.log(x) for x, g in ((a0, given[0]), (b0, given[1])) if g is None]
                found = optimize.minimize(objective, v0, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-14, "maxiter": 4000})
                it = iter(np.exp(found.x))
                lam = float(next(it)) if given[0] is None else float(given[0])
                lam2 = float(next(it)) if given[1] is None else float(given[1])
        X = loadings(m, float(lam), lam2)
        B, e = _fit_factors(Y, X)
        names = ["level", "slope", "curvature"] + (["curvature 2"] if model == "svensson" else [])
        return YieldCurve(tuple(series), m, model, float(lam), lam2, pd.DataFrame(B, index=yields.index, columns=names), yields,
                          pd.DataFrame(e, index=yields.index, columns=series), estimated)
