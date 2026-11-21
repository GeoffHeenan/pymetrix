"""Shock decompositions: how much each shock contributed to each variable's path.

    d = fitted_model.decompose()                 # the filtered history: each variable, shock by shock
    d.contributions["y"]                         # a DataFrame: periods x (shocks, initial conditions)
    d.plot(["y", "pi"])                          # stacked bars, with the path itself as a line
    solution.decompose(states, ...)              # any path of the model's variables (deviations from steady state)

With ``x(t) = P x(t-1) + Q e(t)``, a path of the model's variables (all of them, in deviations from the steady state)
gives back the shocks, ``e(t) = Q^+ (x(t) - P x(t-1))``, and then each shock's contribution, ``c_j(t) = P c_j(t-1) +
Q_j e_j(t)``, and the effect of where the path started, ``P^t x(0)``. They add up to the path exactly when the path
follows the model (the Kalman smoother's estimates do); ``residual`` shows any part that does not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..errors import SpecificationError
from ..output import Table

if TYPE_CHECKING:
    from .linear import LinearSolution

__all__ = ["Decomposition", "decompose"]

INITIAL = "initial conditions"


@dataclass(frozen=True, eq=False)
class Decomposition:
    """Each variable's deviation from its steady state, split into the contributions of the shocks and of the
    starting point."""

    contributions: dict[str, pd.DataFrame]  # variable -> periods x (shocks..., initial conditions)
    deviations: pd.DataFrame  # periods x variables: the path being decomposed
    shocks: pd.DataFrame  # periods x shocks: the shocks recovered from the path (empty for data contributions)
    title: str = "Shock decomposition"  # or "Data contributions": the parts are the observed series
    measure: str = "deviation from steady state"  # what the path is: a VAR's is relative to its baseline
    baseline: pd.DataFrame | None = None  # periods x variables: a VAR's path without shocks (deterministic terms, initial conditions)
    lower: dict[str, pd.DataFrame] | None = None  # variable -> periods x shocks: the bands' lower quantile, over draws
    median: dict[str, pd.DataFrame] | None = None  # the median over draws
    upper: dict[str, pd.DataFrame] | None = None
    coverage: float | None = None  # the bands' probability, e.g. 0.95
    band_source: str = ""  # what the draws are, e.g. "500 posterior draws"

    @property
    def has_bands(self) -> bool:
        return self.lower is not None and self.upper is not None

    @property
    def residual(self) -> pd.DataFrame:
        """The part of each path the shocks and initial conditions do not explain (zero for a path of the model)."""
        return pd.DataFrame({v: self.deviations[v] - frame.sum(axis=1) for v, frame in self.contributions.items()})

    def table(self, variable: str) -> Table:
        frame = self.contributions[variable].copy()
        frame["total"] = self.deviations[variable]
        frame.index = frame.index.astype(str)
        return Table.from_frame(frame, title=f"{self.title}: {variable} ({self.measure})")

    def plot(self, variables: Sequence[str] | None = None) -> Any:
        """A panel for each variable: the contributions as stacked bars, the path being decomposed as a line."""
        from ..figures import Figure, Panel, contribution_layers

        names = list(variables) if variables else list(self.contributions)
        panels = []
        for v in names:  # the parts as stacked bars, the path as the total (any residual as "other")
            panels.append(Panel(contribution_layers(self.contributions[v], self.deviations[v]), title=v, legend=not panels))
        return Figure(tuple(panels), title=self.title, columns=2 if len(panels) > 3 else 1)

    def bands_table(self, variable: str) -> Table:
        """The variable's contributions with their bands: for each shock, the point contribution and the lower,
        median and upper quantiles over the draws."""
        if not self.has_bands:
            raise SpecificationError("this decomposition has no bands: ask for them with draws=, e.g. historical_decomposition(draws=500)")
        point = self.contributions[variable]
        cols = {}
        for s in self.lower[variable].columns:  # type: ignore[index]
            cols[f"{s}"] = point[s]
            cols[f"{s} lower"] = self.lower[variable][s]  # type: ignore[index]
            cols[f"{s} median"] = self.median[variable][s]  # type: ignore[index]
            cols[f"{s} upper"] = self.upper[variable][s]  # type: ignore[index]
        frame = pd.DataFrame(cols)
        frame.index = frame.index.astype(str)
        notes = [f"{self.coverage:.0%} bands: pointwise quantiles over {self.band_source}" if self.coverage else self.band_source,
                 "Each draw's contributions and baseline add up to the data; the point column is the decomposition at the estimates"]
        return Table.from_frame(frame, title=f"{self.title}: {variable}, bands", notes=notes)

    def plot_bands(self, variables: Sequence[str] | None = None) -> Any:
        """A panel for each variable and shock: the contribution's band (shaded), its median over the draws and the
        point contribution (dashed)."""
        from ..figures import Band, Figure, Line, Panel, Rule

        if not self.has_bands:
            raise SpecificationError("this decomposition has no bands: ask for them with draws=, e.g. historical_decomposition(draws=500)")
        names = list(variables) if variables else list(self.contributions)
        shocks = list(self.lower[names[0]].columns)  # type: ignore[index]
        panels = []
        for v in names:
            for s in shocks:
                layers = (Band(self.lower[v][s], self.upper[v][s], f"{self.coverage:.0%} band" if self.coverage else "band"),  # type: ignore[index]
                          Line(self.median[v][s], "median"), Line(self.contributions[v][s], "point", style="--"), Rule(0.0))  # type: ignore[index]
                panels.append(Panel(layers, title=f"{v} <- {s}", legend=not panels))
        return Figure(tuple(panels), title=f"{self.title}, bands ({self.band_source})", columns=len(shocks))

    def __repr__(self) -> str:
        parts = len(next(iter(self.contributions.values())).columns) - 1 if self.contributions else 0
        bands = f", bands from {self.band_source}" if self.has_bands else ""
        return f"<Decomposition ({self.title}): {len(self.contributions)} variables, {parts} parts, {len(self.deviations)} periods{bands}>"

    def _repr_html_(self) -> str:
        return self.table(next(iter(self.contributions)))._repr_html_()


def decompose(sol: LinearSolution, levels: pd.DataFrame, start: str = "first", regimes: Any = None) -> Decomposition:
    """Decompose paths of the model's variables (``levels``: periods x variables, in levels) into the shocks'
    contributions and the effect of where the path started.

    ``start="first"`` counts the first period's state, its shocks included, as the initial condition. ``"presample"``
    takes the first period's state as the pre-sample state x(-1) and decomposes from the next period, so that period's
    shocks are the shocks' (give one period more at the front).

    ``regimes`` (known-date regimes over the path's periods): through their time-varying rules ``x(t) = k(t) + P(t)
    x(t-1) + Q(t) e(t)``, the shocks ``e(t) = Q(t)^+ (x(t) - k(t) - P(t) x(t-1))``, and the regimes' constants a part
    of their own, ``"regimes"`` (``path_decomposition``); deviations from the steady state of the regime in force
    before the first period."""
    plan = None
    if regimes:
        from .regimes import regime_plan

        periods = levels.index if isinstance(levels.index, pd.PeriodIndex) else pd.PeriodIndex(levels.index)
        sol, plan = regime_plan(sol, regimes, periods, "span")
    if start not in ("first", "presample"):
        raise SpecificationError("start is 'first' (the first period's state is the initial condition) or 'presample' "
                                 "(the first period is the pre-sample state, and the decomposition starts after it)")
    steady = dict(sol.model.steady.values)
    trend = dict(getattr(sol.model.steady, "growth", {}) or {})
    observed = list(sol.model.observed)
    missing = [v for v in observed if v not in levels.columns]
    if missing:
        raise SpecificationError(f"the paths need every model variable; missing {', '.join(missing)}")
    frame = levels[observed].astype(float)
    if frame.isna().any().any():
        raise SpecificationError("the paths have missing values; decompose smoothed estimates (a fitted model's smoothed data)")
    T = len(frame)
    steps = np.arange(T)
    dev = frame - pd.DataFrame({v: steady[v] + trend.get(v, 0.0) * steps for v in observed}, index=frame.index)

    def state(t: int) -> np.ndarray:  # every state variable, auxiliary lags included, at period t
        out = np.zeros(len(sol.variables))
        for j, name in enumerate(sol.variables):
            base, _, tail = name.rpartition("_")
            if name in observed:
                out[j] = dev[name].iloc[t]
            elif base in observed and tail.startswith("lag") and tail[3:].isdigit():
                k = int(tail[3:])
                out[j] = dev[base].iloc[t - k] if t - k >= 0 else np.nan
        return out

    first = max([int(n.rpartition("_")[2][3:]) for n in sol.variables if n.rpartition("_")[2].startswith("lag")
                 and n.rpartition("_")[2][3:].isdigit()] or [0])
    X = np.array([state(t) for t in range(first, T)])
    index = frame.index[first:]
    if plan is not None:
        from .regimes import REGIMES

        rules = plan.period_rules(T)[first:]
        Ts, cs, Rs = (np.array([r[i] for r in rules]) for i in (0, 1, 2))
        e = np.zeros((len(X), Rs.shape[2]))
        for t in range(1, len(X)):
            e[t] = np.linalg.pinv(Rs[t]) @ (X[t] - cs[t] - Ts[t] @ X[t - 1])
        return path_decomposition(Ts, cs, Rs, X, e, index, list(sol.variables), observed, list(sol.shocks), REGIMES,
                                  "Shock decomposition (known-date regimes)", None, start)
    pinv = np.linalg.pinv(sol.Q)
    m = sol.Q.shape[1]
    shocks = np.zeros((len(X), m))
    for t in range(1, len(X)):
        shocks[t] = pinv @ (X[t] - sol.P @ X[t - 1])
    contrib = np.zeros((len(X), m, len(sol.variables)))
    initial = np.zeros((len(X), len(sol.variables)))
    initial[0] = X[0]
    for t in range(1, len(X)):
        initial[t] = sol.P @ initial[t - 1]
        for j in range(m):
            contrib[t, j] = sol.P @ contrib[t - 1, j] + sol.Q[:, j] * shocks[t, j]
    if start == "presample":  # the first state is x(-1): its effect is P^(t+1) x(-1), every later shock the shocks'
        if len(X) < 2:
            raise SpecificationError("start='presample' needs the pre-sample period and at least one more")
        index, shocks, contrib = index[1:], shocks[1:], contrib[1:]
        initial = initial[:-1] @ sol.P.T
    contributions = {}
    for v in observed:
        i = list(sol.variables).index(v)
        cols = {s: contrib[:, j, i] for j, s in enumerate(sol.shocks)}
        cols[INITIAL] = initial[:, i]
        contributions[v] = pd.DataFrame(cols, index=index)
    return Decomposition(contributions, dev.loc[index], pd.DataFrame(shocks, index=index, columns=list(sol.shocks)))


def path_decomposition(T: np.ndarray, c: np.ndarray, R: np.ndarray, X: np.ndarray, e: np.ndarray, index: pd.Index,
                       variables: Sequence[str], observed: Sequence[str], shocks: Sequence[str], constant: str, title: str,
                       factor: np.ndarray | None = None, start: str = "first") -> Decomposition:
    """The decomposition of a path through time-varying rules ``X[t] = c[t] + T[t] X[t-1] + R[t] e[t]`` (deviations
    from the reference steady state; OccBin's binding regimes, known-date regimes): the first period's state is the
    starting point, each shock's contribution and the starting point's go through the rules, and the constants ``c``
    are a part of their own (``constant``); the parts add up to the path. ``factor`` (correlated shocks): the
    contributions of the shocks orthogonalised by it (``e = factor u``, Cholesky, as impulse responses), ``u`` in
    standard deviations. ``start="presample"``: the first period is the state before, and the decomposition starts
    after it."""
    if start not in ("first", "presample"):
        raise SpecificationError("start is 'first' (the first period's state is the initial condition) or 'presample' "
                                 "(the first period is the pre-sample state, and the decomposition starts after it)")
    nT, k = X.shape
    if factor is not None:
        e = e @ np.linalg.pinv(factor).T  # the orthogonalised shocks
        R = R @ factor
    m = R.shape[2]
    initial, constants, parts = np.zeros((nT, k)), np.zeros((nT, k)), np.zeros((nT, m, k))
    initial[0] = X[0]  # the first period's state is the starting point, as in the linear decomposition
    for t in range(1, nT):
        initial[t] = T[t] @ initial[t - 1]
        constants[t] = T[t] @ constants[t - 1] + c[t]
        for j in range(m):
            parts[t, j] = T[t] @ parts[t - 1, j] + R[t][:, j] * e[t, j]
    shocks_used = e.copy()
    shocks_used[0] = 0.0  # in the starting point
    rows = slice(1, None) if start == "presample" else slice(None)
    if start == "presample" and nT < 2:
        raise SpecificationError("start='presample' needs the pre-sample period and at least one more")
    index = index[rows]
    position = {v: i for i, v in enumerate(variables)}
    contributions = {}
    for v in observed:
        i = position[v]
        cols = {s: parts[rows, j, i] for j, s in enumerate(shocks)}
        cols[INITIAL] = initial[rows, i]
        cols[constant] = constants[rows, i]
        contributions[v] = pd.DataFrame(cols, index=index)
    deviations = pd.DataFrame({v: X[rows, position[v]] for v in observed}, index=index)
    if factor is not None:
        title += "; correlated shocks orthogonalised by Cholesky, in the order " + ", ".join(shocks)
    return Decomposition(contributions, deviations, pd.DataFrame(shocks_used[rows], index=index, columns=list(shocks)), title=title)
