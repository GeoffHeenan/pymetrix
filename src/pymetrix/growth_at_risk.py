"""Growth at risk: quantile local projections of future growth, and the skewed-t distributions they imply (Adrian,
Boyarchenko and Giannone 2019).

    gar = growth_at_risk(data, "dlog(gdp)*400", ["nfci", "dlog(gdp)*400"], horizons=(1, 4))
    gar.summary()                    # each horizon's quantile regressions: how conditions move each quantile
    gar.quantiles(4)                 # by period: the predicted 5%, 25%, 50%, 75% and 95% quantiles, four quarters out
    gar.at_risk(4); gar.plot()       # growth at risk (the 5% quantile) and expected shortfall, from the skewed t

**Quantile projections.** For each horizon h and quantile q, ``growth`` averaged over the next h periods,
(y_(t+1) + ... + y_(t+h)) / h, is regressed on the ``predictors`` at t by quantile regression (Koenker and Bassett).
The predicted quantiles of each period are put in increasing order where they cross (Chernozhukov, Fernandez-Val
and Galichon's rearrangement).

**The predictive distribution.** Each period's predicted 5%, 25%, 75% and 95% quantiles are matched by the four
parameters of Azzalini and Capitanio's skewed t, location xi, scale omega, shape alpha and degrees of freedom nu,
whose density is 2 / omega t_nu(z) T_(nu+1)(alpha z sqrt((nu + 1) / (nu + z^2))), z = (x - xi) / omega, by least squares.
Growth at risk is its 5% quantile and expected shortfall the mean below it, both computed from the fitted density.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import gammaln, stdtr

from ._api import api
from .errors import SpecificationError
from .output import Table
from .results import deprecated_alias

if TYPE_CHECKING:
    from .data import Dataset
    from .figures import Figure

__all__ = ["GrowthAtRisk", "growth_at_risk", "skewt_pdf", "skewt_quantiles"]

_Z = np.tan(np.linspace(-1.555, 1.555, 1201))  # the standardised grid: dense in the middle, out to +-60 in the tails
MATCHED = (0.05, 0.25, 0.75, 0.95)


def skewt_pdf(x: Any, xi: float, omega: float, alpha: float, nu: float) -> np.ndarray:
    """Azzalini and Capitanio's skewed t density."""
    z = (np.asarray(x, float) - xi) / omega
    log_t = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - (nu + 1) / 2 * np.log1p(z**2 / nu)
    return 2 / omega * np.exp(log_t) * stdtr(nu + 1, alpha * z * np.sqrt((nu + 1) / (nu + z**2)))


def _standard_cdf(alpha: float, nu: float) -> np.ndarray:
    dens = skewt_pdf(_Z, 0.0, 1.0, alpha, nu)
    cdf = np.concatenate([[0.0], np.cumsum((dens[1:] + dens[:-1]) / 2 * np.diff(_Z))])
    return cdf / cdf[-1]


def skewt_quantiles(q: Any, xi: float, omega: float, alpha: float, nu: float) -> np.ndarray:
    """The skewed t's quantiles (numerically, on a fine grid)."""
    cdf = _standard_cdf(alpha, nu)
    return xi + omega * np.interp(np.asarray(q, float), cdf, _Z)


def skewt_shortfall(p: float, xi: float, omega: float, alpha: float, nu: float) -> float:
    """The mean of the skewed t below its p quantile (adaptive quadrature)."""
    from scipy import integrate

    q = skewt_quantiles([p], xi, omega, alpha, nu)[0]
    mass = integrate.quad(lambda x: skewt_pdf(x, xi, omega, alpha, nu), -np.inf, q, limit=200)[0]
    return float(integrate.quad(lambda x: x * skewt_pdf(x, xi, omega, alpha, nu), -np.inf, q, limit=200)[0] / mass)


def _nu(p: float) -> float:
    return 2.0 + 58.0 / (1.0 + np.exp(-p))  # degrees of freedom kept between 2 (finite variance) and 60 (near normal)


def fit_skewt(levels: Sequence[float], values: Sequence[float], start: Any = None) -> np.ndarray:
    """(xi, omega, alpha, nu) whose quantiles at ``levels`` are nearest ``values`` (least squares). The standardised
    quantiles depend on alpha and nu alone, and xi and omega enter linearly, so for each (alpha, nu) they are a
    regression of the values on those quantiles; only (alpha, nu) are searched. ``start``, a neighbouring period's
    answer, saves the search from three starting points."""
    v = np.asarray(values, float)
    spread = max(float(v[-1] - v[0]), 1e-6)

    def location_scale(alpha: float, nu: float) -> tuple[float, float, float]:
        z = skewt_quantiles(levels, 0.0, 1.0, alpha, nu)
        Z = np.column_stack([np.ones_like(z), z])
        (xi, omega), *_ = np.linalg.lstsq(Z, v, rcond=None)
        return float(xi), float(omega), float(np.sum((Z @ [xi, omega] - v) ** 2)) / spread**2

    def loss(p: np.ndarray) -> float:
        alpha, nu_p = p
        if not np.all(np.isfinite(p)) or abs(alpha) > 50:
            return 1e6
        _, omega, err = location_scale(alpha, _nu(nu_p))
        return err if omega > 0 else 1e6

    if start is not None:
        nu0 = min(max(float(start[3]), 2.01), 59.99)
        starts = [np.array([start[2], np.log((nu0 - 2.0) / (60.0 - nu0))])]
    else:
        starts = [np.array([a0, np.log(2.0 / 56.0)]) for a0 in (-2.0, 0.0, 2.0)]  # nu = 4
    best = None
    for s0 in starts:
        found = optimize.minimize(loss, s0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-13, "maxiter": 1000})
        if best is None or found.fun < best.fun:
            best = found
    alpha, nu_p = best.x
    nu = _nu(nu_p)
    xi, omega, _ = location_scale(alpha, nu)
    return np.array([xi, omega, alpha, nu])


@dataclass(frozen=True, eq=False)
class GrowthAtRisk:
    """Quantile projections of future growth, and the skewed-t distribution fitted to each period's quantiles."""

    dependent: str  # was target
    predictors: tuple[str, ...]
    horizons: tuple[int, ...]
    levels: tuple[float, ...]
    fits: dict[tuple[int, float], Any] = field(repr=False)  # (horizon, quantile) -> FittedEquation
    predicted: dict[int, pd.DataFrame] = field(repr=False)  # horizon -> period by quantile
    skewt: dict[int, pd.DataFrame] = field(repr=False)  # horizon -> period by xi, omega, alpha, nu

    stage = "forecast"

    def quantiles(self, horizon: int | None = None) -> pd.DataFrame:
        """The predicted quantiles by period (rearranged where they cross)."""
        return self.predicted[self._h(horizon)]

    def at_risk(self, horizon: int | None = None, level: float = 0.05) -> pd.DataFrame:
        """Growth at risk (the ``level`` quantile) and expected shortfall below it, from each period's skewed t."""
        h = self._h(horizon)
        rows = [{"growth at risk": skewt_quantiles([level], *p)[0], "expected shortfall": skewt_shortfall(level, *p),
                 "median": skewt_quantiles([0.5], *p)[0]} for p in self.skewt[h].to_numpy()]
        return pd.DataFrame(rows, index=self.skewt[h].index)

    def density(self, period: Any = None, horizon: int | None = None, grid: Any = None) -> pd.Series:
        """The fitted skewed-t density for one period (default: the last)."""
        h = self._h(horizon)
        frame = self.skewt[h]
        row = frame.iloc[-1] if period is None else frame.loc[pd.Period(period, frame.index.freq)]
        xi, omega, alpha, nu = row.to_numpy()
        x = np.linspace(xi - 6 * omega, xi + 6 * omega, 400) if grid is None else np.asarray(grid, float)
        return pd.Series(skewt_pdf(x, xi, omega, alpha, nu), index=pd.Index(x, name=self.dependent), name="density")

    def _h(self, horizon: int | None) -> int:
        h = self.horizons[-1] if horizon is None else int(horizon)
        if h not in self.horizons:
            raise SpecificationError(f"horizon {h} was not estimated; horizons: {', '.join(map(str, self.horizons))}")
        return h

    def summary(self) -> Table:
        names = list(next(iter(self.fits.values())).params.names)
        cols, rows = [], []
        for h in self.horizons:
            for q in self.levels:
                cols.append(f"h={h} q={q:g}")
        for i, n in enumerate(names):
            rows.append([float(self.fits[(h, q)].params.values[i]) for h in self.horizons for q in self.levels])
        shown = [self.predictors[int(n[1:])] if n.startswith("x") and n[1:].isdigit() else n for n in names]
        return Table(cols, rows, index=shown, index_name="", title=f"Growth at risk: quantile projections of {self.dependent}",
                     notes=(f"Quantile regressions of {self.dependent} averaged over the next h periods on {', '.join(self.predictors)} "
                            "at t; robust (kernel) standard errors are in each fit.",))

    def plot(self) -> Figure:
        from .figures import Band, Figure, Line, Panel

        h = self._h(None)
        q = self.quantiles(h)
        risk = self.at_risk(h)
        first = Panel((Band(q[self.levels[0]], q[self.levels[-1]], f"{self.levels[0]:.0%}-{self.levels[-1]:.0%}"),
                       Line(q[0.5] if 0.5 in q else risk["median"], "median"), Line(risk["growth at risk"], "growth at risk (5%)")),
                      title=f"Predicted distribution of {self.dependent}, {h} periods ahead")
        d = self.density(horizon=h)
        second = Panel((Line(d, f"{self.skewt[h].index[-1]}"),), title="The latest predictive density", xlabel=self.dependent)
        return Figure((first, second))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.at_risk()

    def __repr__(self) -> str:
        return f"<GrowthAtRisk of {self.dependent}, horizons {self.horizons}>"

    def __setstate__(self, state: dict[str, Any]) -> None:  # results cached in projects saved before the rename
        if "target" in state:
            state["dependent"] = state.pop("target")
        self.__dict__.update(state)

    target = deprecated_alias("dependent", "target", why="the dependent variable; target is its old name")


@api("pymetrix:growth_at_risk")
def growth_at_risk(data: Dataset, growth: str, predictors: Sequence[str], horizons: Sequence[int] = (1, 4),
                   quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95), span: Any = None) -> GrowthAtRisk:
    """Quantile local projections of ``growth`` averaged over each horizon on ``predictors``, and a skewed t fitted to
    each period's predicted quantiles (see the module notes)."""
    from .data import Dataset as _D
    from .estimation import Equation
    from .estimation.quantile import Quantile

    predictors = [predictors] if isinstance(predictors, str) else list(predictors)
    horizons = tuple(int(h) for h in horizons)
    levels = tuple(sorted(float(q) for q in quantiles))
    if not all(m in levels for m in MATCHED):
        raise SpecificationError(f"the quantiles must include {', '.join(map(str, MATCHED))}, which the skewed t is matched to")
    if data.time_axis is None or data.kind != "time_series":
        raise SpecificationError("growth at risk needs time series")
    if not horizons or min(horizons) < 1:
        raise SpecificationError("horizons are at least 1 period")
    y = data.eval(growth)
    X = pd.DataFrame({f"x{i}": data.eval(p) for i, p in enumerate(predictors)})
    fits, predicted, skew = {}, {}, {}
    for h in horizons:
        future = sum(y.shift(-k) for k in range(1, h + 1)) / h
        frame = pd.concat([future.rename("future"), X], axis=1)
        work = _D.from_frame(frame.dropna())
        eq = Equation(f"future ~ {' + '.join(X.columns)}")
        pred = {}
        full = X.dropna()
        Z = np.column_stack([np.ones(len(full)), full.to_numpy(float)])
        for q in levels:
            fit = eq.fit(work, span=span, method=Quantile(q=q))
            fits[(h, q)] = fit
            pred[q] = Z @ fit.params.values
        P = np.sort(np.column_stack([pred[q] for q in levels]), axis=1)  # rearranged: quantiles never cross
        predicted[h] = pd.DataFrame(P, index=full.index, columns=list(levels))
        idx = [levels.index(m) for m in MATCHED]
        params, previous = [], None
        for row in P:
            previous = fit_skewt(MATCHED, row[idx], previous)
            params.append(previous)
        skew[h] = pd.DataFrame(params, index=full.index, columns=["xi", "omega", "alpha", "nu"])
    renamed = {f"x{i}": p for i, p in enumerate(predictors)}
    return GrowthAtRisk(growth, tuple(renamed.values()), horizons, levels, fits, predicted, skew)
