"""Conditional forecasts and structural scenarios for VARs and Bayesian VARs: exact conditions, ranges, and which
shocks may move to meet them.

    var.conditional_forecast(data, span, {"r": 2.0})                                 # exact, every period
    var.conditional_forecast(data, span, {"pi": Bounds(3, 4)})                       # between 3 and 4 (a range)
    var.conditional_forecast(data, span, {"pi": {"2026Q1": Bounds(upper=2.5)}})      # at most 2.5 in one period
    var.conditional_forecast(data, span, {"r": 2.0}, shocks=["r"])                   # met by the policy shock alone
    var.conditional_forecast(data, span, {"r": 2.0}, shock_conditions={"y": 0.0})    # ... and no demand shocks

The future path stacks as ``y = point + M e`` with ``e ~ N(0, I)``: ``point`` is the forecast with no shocks and
``M`` the moving-average weights times the impact matrix. A condition is a row of ``R y`` (a variable in a period) or
of ``e`` (a structural shock in a period), either exact (``= value``) or a range (``lower <= . <= upper``, one side
may be open).

**Exact conditions** (Waggoner and Zha 1999). With every shock free to move, the conditional distribution of ``e``
given ``R M e = r - R point`` is normal with mean ``e* = D'(DD')^-1 (r - R point)``, ``D = R M``, the smallest
(minimum-norm) shocks that meet the conditions, and covariance ``I - D'(DD')^-1 D``; it does not depend on how the
shocks are identified.

**Which shocks move** (structural scenario analysis, Antolin-Diaz, Petrella and Rubio-Ramirez 2021). With
``shocks=[...]`` only those identified shocks (the *driving* shocks) adjust to the conditions; the others keep their
unconditional distribution ``N(0, 1)``. Given the others, the driving shocks are drawn from their distribution given the
conditions, so when they can meet them in more ways than one (more shock-periods than conditions) the mean is the
minimum-norm combination, and when they cannot meet them at all the forecast is refused. This does depend on the
identification. ``shock_conditions`` restricts the structural shocks themselves (``0.0`` switches a shock off;
``Bounds(lower=0)`` allows only positive ones); a conditioned shock must be one that may move.

**Ranges** (soft conditions in Waggoner and Zha's sense; the truncated-normal sampler of Geweke 1991 and Robert 1995,
used for conditional forecasts with inequality constraints as in Andersson, Palmqvist and Waggoner 2010). Given the
exact conditions, the constrained quantities ``z = R_b y`` are jointly normal; the forecast draws ``z`` from that
normal truncated to the box ``lower <= z <= upper`` by Gibbs sampling, one coordinate at a time from its univariate
truncated normal given the others (run as many independent chains as draws, each for ``SWEEPS`` sweeps from an
unconstrained draw clipped into the box), and then the rest of the path from its normal distribution given ``z`` and
the exact conditions. Every draw meets every bound; a range wide enough never to bind gives the unconditional forecast
and one that shrinks to a point gives the exact condition.

The forecast is then the mean, standard deviation, median and quantiles of the draws. With exact conditions only,
on a classical VAR, it is the closed form instead (mean and standard error, intervals mean +/- z se).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import special

from .errors import SpecificationError

__all__ = ["Bounds", "SWEEPS"]

SWEEPS = 100  # Gibbs sweeps of each chain for the ranges (each chain gives one draw)
UPDATE_SWEEPS = 5  # sweeps per iteration of the Waggoner-Zha sampler that also updates the posterior (the chain carries on)


@dataclass(frozen=True)
class Bounds:
    """A range condition: the variable (or shock) is at least ``lower`` and at most ``upper`` in a period; leave one
    out (None) for a one-sided condition. ``Bounds(3, 4)`` is "between 3 and 4"."""

    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        lo, hi = _limit(self.lower, -np.inf), _limit(self.upper, np.inf)
        if lo == -np.inf and hi == np.inf:
            raise SpecificationError("a range needs a lower bound, an upper bound or both")
        if lo > hi:
            raise SpecificationError(f"the lower bound {lo} is above the upper bound {hi}")

    def __repr__(self) -> str:
        if self.lower is None:
            return f"Bounds(upper={self.upper!r})"
        return f"Bounds({self.lower!r})" if self.upper is None else f"Bounds({self.lower!r}, {self.upper!r})"


def _limit(value: Any, default: float) -> float:
    if value is None:
        return default
    out = float(value)
    if np.isnan(out):
        raise SpecificationError("a bound is not a number")
    return out


@dataclass(frozen=True)
class Condition:
    """One condition: period ``i``, variable or shock ``k``, ``lower <= . <= upper`` (equal when exact)."""

    i: int
    k: int
    lower: float
    upper: float
    exact: bool
    on_shock: bool = False

    @property
    def value(self) -> float:
        return self.lower


def _cell(value: Any) -> tuple[float, float, bool] | None:
    """(lower, upper, exact) for one period's condition, or None if ``value`` is not a single condition."""
    if isinstance(value, Bounds):
        return _limit(value.lower, -np.inf), _limit(value.upper, np.inf), False
    if isinstance(value, Mapping) and value and set(value) <= {"lower", "upper"}:
        b = Bounds(value.get("lower"), value.get("upper"))
        return _limit(b.lower, -np.inf), _limit(b.upper, np.inf), False
    if isinstance(value, tuple) and len(value) == 2 and all(v is None or np.isscalar(v) for v in value):
        b = Bounds(*value)
        return _limit(b.lower, -np.inf), _limit(b.upper, np.inf), False
    if np.isscalar(value) and not isinstance(value, str):
        v = float(value)
        if not np.isfinite(v):
            return None
        return v, v, True
    return None


def parse(conditions: Mapping[str, Any] | None, names: Sequence[str], periods: pd.PeriodIndex, what: str = "variable",
          on_shock: bool = False) -> list[Condition]:
    """The conditions on ``names`` (variables, or shocks with ``on_shock``): each maps a name to a condition held over
    every forecast period (a number, ``Bounds``, ``(lower, upper)`` or ``{"lower": ..., "upper": ...}``) or to period ->
    condition."""
    out: list[Condition] = []
    for name, path in (conditions or {}).items():
        if name not in names:
            raise SpecificationError(f"{name} is not a {what} of this VAR ({', '.join(names)})")
        k = list(names).index(name)
        single = _cell(path)
        if single is not None:
            items: list[tuple[Any, Any]] = [(q, path) for q in periods]
        elif isinstance(path, (Mapping, pd.Series)):
            items = list(dict(path).items())
        else:
            raise SpecificationError(f"the condition on {name} is not a number, a range or a path of them")
        for q, value in items:
            period = pd.Period(str(q), freq=periods.freq)
            if period not in periods:
                raise SpecificationError(f"the condition on {name} in {period} is outside the forecast ({periods[0]} to {periods[-1]})")
            cell = _cell(value)
            if cell is None:
                raise SpecificationError(f"the condition on {name} in {period} is not a number")
            lo, hi, exact = cell
            out.append(Condition(periods.get_loc(period), k, lo, hi, exact, on_shock))
    return out


def check(conds: Sequence[Condition]) -> None:
    if not conds:
        raise SpecificationError("give at least one condition, e.g. {'r': 2.0}, {'r': {'2025Q1': 2.0}} or {'pi': Bounds(3, 4)}")
    if len({(c.i, c.k, c.on_shock) for c in conds}) < len(conds):
        raise SpecificationError("a variable has two conditions in the same period")


def record(conds: Sequence[Condition], periods: pd.PeriodIndex, variables: Sequence[str], shock_names: Sequence[str],
           driving: Sequence[str] | None) -> dict[str, Any]:
    """The conditions as plain data, for the forecast's metadata and a forecast round's record."""
    rows = []
    for c in conds:
        row: dict[str, Any] = {"shock" if c.on_shock else "variable": (shock_names if c.on_shock else variables)[c.k],
                               "period": str(periods[c.i])}
        if c.exact:
            row["value"] = c.value
        else:
            row.update({"lower": None if c.lower == -np.inf else c.lower, "upper": None if c.upper == np.inf else c.upper})
        rows.append(row)
    return {"conditions": rows, "shocks": None if driving is None else list(driving)}


def summary(conds: Sequence[Condition], variables: Sequence[str], shock_names: Sequence[str]) -> str:
    held = sorted({variables[c.k] for c in conds if not c.on_shock and c.exact})
    ranged = sorted({variables[c.k] for c in conds if not c.on_shock and not c.exact})
    shocked = sorted({shock_names[c.k] for c in conds if c.on_shock})
    parts = ([f"paths for {', '.join(held)}"] if held else []) + ([f"ranges for {', '.join(ranged)}"] if ranged else []) + (
        [f"conditions on the shocks {', '.join(shocked)}"] if shocked else [])
    return "; ".join(parts)


# ---------------------------------------------------------------------------------------------
# the algebra


def stacked(A: np.ndarray, impact: np.ndarray, H: int) -> np.ndarray:
    """M (H K x H S): the stacked path's response to the stacked shocks (period-major), lower block-triangular."""
    from .var import _ma

    K, S = impact.shape
    phi = _ma(A, H - 1)
    M = np.zeros((H * K, H * S))
    for i in range(H):
        for j in range(i + 1):
            M[i * K:(i + 1) * K, j * S:(j + 1) * S] = phi[i - j] @ impact
    return M


@dataclass(frozen=True, eq=False)
class Prepared:
    """Draws of the path as ``y = y0 + W (z - c)``, ``z`` from ``N(c, S)`` truncated to ``[lower, upper]`` (no ranges:
    ``y = y0``). Shapes: y0 (n, H K), c (n, nb), W (H K, nb), S (nb, nb)."""

    y0: np.ndarray
    c: np.ndarray
    W: np.ndarray
    S: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def _rows(conds: Sequence[Condition], M: np.ndarray, driving: np.ndarray, K: int, S: int) -> np.ndarray:
    """Each condition as a row acting on the driving shocks."""
    Md = M[:, driving]
    out = np.zeros((len(conds), int(driving.sum())))
    where = np.cumsum(driving) - 1
    for r, c in enumerate(conds):
        if c.on_shock:
            col = c.i * S + c.k
            if not driving[col]:
                raise SpecificationError("a condition on a shock needs that shock among those that may move (shocks=)")
            out[r, where[col]] = 1.0
        else:
            out[r] = Md[c.i * K + c.k]
    return out


def _offsets(conds: Sequence[Condition], base: np.ndarray, K: int) -> np.ndarray:
    """What each condition's quantity is before the driving shocks move: (n, conditions)."""
    out = np.zeros((len(base), len(conds)))
    for r, c in enumerate(conds):
        if not c.on_shock:
            out[:, r] = base[:, c.i * K + c.k]
    return out


def _pinv_rows(D: np.ndarray, what: str) -> np.ndarray:
    DD = D @ D.T
    if np.linalg.matrix_rank(DD) < len(D):
        raise SpecificationError(f"the {what} cannot all be met: some fix a variable that other conditions already determine, "
                                 "or the shocks that may move cannot reach it (allow more shocks, or condition less)")
    return D.T @ np.linalg.inv(DD)


def prepare(A: np.ndarray, impact: np.ndarray, point: np.ndarray, conds: Sequence[Condition], driving: np.ndarray, n: int,
            rng: np.random.Generator) -> Prepared:
    """The ingredients of ``n`` draws for one set of parameters (see the module docstring). ``driving`` marks the
    stacked shocks (period-major) that may move to meet the conditions; the others are drawn unconditionally."""
    H, K = point.shape
    S = impact.shape[1]
    M = stacked(A, impact, H)
    Md, Mo = M[:, driving], M[:, ~driving]
    nd = Md.shape[1]
    base = np.tile(point.reshape(-1), (n, 1))
    if Mo.shape[1]:
        base = base + rng.standard_normal((n, Mo.shape[1])) @ Mo.T
    exact = [c for c in conds if c.exact]
    ranged = [c for c in conds if not c.exact]
    if exact:
        De = _rows(exact, M, driving, K, S)
        Ge = _pinv_rows(De, "exact conditions")
        e_star = (np.array([c.value for c in exact])[None, :] - _offsets(exact, base, K)) @ Ge.T
        F = np.eye(nd) - Ge @ De
    else:
        e_star, F = np.zeros((n, nd)), np.eye(nd)
    mean = base + e_star @ Md.T
    xi = rng.standard_normal((n, nd))
    if not ranged:
        return Prepared(mean + xi @ (Md @ F).T, np.zeros((n, 0)), np.zeros((H * K, 0)), np.zeros((0, 0)), np.zeros(0), np.zeros(0))
    Db = _rows(ranged, M, driving, K, S)
    B = Db @ F  # z = c + B eta, eta ~ N(0, I), given the exact conditions
    Sz = B @ B.T
    if np.linalg.matrix_rank(Sz) < len(ranged):
        raise SpecificationError("the ranges cannot be met independently: some bound a quantity that the exact conditions or "
                                 "other ranges already determine, or the shocks that may move cannot reach it")
    Si = np.linalg.inv(Sz)
    c = _offsets(ranged, base, K) + e_star @ Db.T
    keep = np.eye(nd) - B.T @ Si @ B  # what z leaves free in eta
    y0 = mean + xi @ (Md @ F @ keep).T
    W = Md @ F @ B.T @ Si
    return Prepared(y0, c, W, (Sz + Sz.T) / 2, np.array([r.lower for r in ranged]), np.array([r.upper for r in ranged]))


def truncated_standard(a: np.ndarray, b: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Standard normal draws truncated to ``[a, b]`` by the inverse distribution function, in logs so that it stays
    accurate far in the tails; ``u`` uniform."""
    flip = a > 0  # use the lower tail: draw from [-b, -a] and negate
    lo, hi = np.where(flip, -b, a), np.where(flip, -a, b)
    la, lb = special.log_ndtr(lo), special.log_ndtr(hi)
    with np.errstate(divide="ignore"):
        logp = np.logaddexp(lb + np.log(u), la + np.log1p(-u))
    x = special.ndtri_exp(np.minimum(logp, 0.0))
    x = np.clip(np.where(np.isfinite(x), x, lo), lo, hi)
    return np.where(flip, -x, x)


def box_gibbs(c: np.ndarray, S: np.ndarray, lower: np.ndarray, upper: np.ndarray, rng: np.random.Generator,
              sweeps: int = SWEEPS, start: np.ndarray | None = None) -> np.ndarray:
    """Draws from ``N(c, S)`` truncated to ``lower <= z <= upper``: one Gibbs chain per row of ``c`` (n, nb), each
    updating one coordinate at a time from its truncated normal given the others. ``S`` is (nb, nb), or (n, nb, nb)
    for a covariance per chain. Chains start from ``start`` (inside the box) or an unconstrained draw clipped into it."""
    n, nb = c.shape
    Sb = np.broadcast_to(S, (n, nb, nb))
    Q = np.linalg.inv(Sb)
    if start is None:
        L = np.linalg.cholesky(Sb)
        start = c + np.einsum("nij,nj->ni", L, rng.standard_normal((n, nb)))
    z = np.clip(np.array(start, float), lower, upper)
    qd = np.diagonal(Q, axis1=1, axis2=2)  # (n, nb)
    sd = 1.0 / np.sqrt(qd)
    for _ in range(sweeps):
        for i in range(nb):
            dev = z - c
            m = c[:, i] - (np.einsum("nj,nj->n", Q[:, i, :], dev) - qd[:, i] * dev[:, i]) / qd[:, i]
            a, b = (lower[i] - m) / sd[:, i], (upper[i] - m) / sd[:, i]
            z[:, i] = np.clip(m + sd[:, i] * truncated_standard(a, b, rng.random(n)), lower[i], upper[i])
    return z


def draw_paths(p: Prepared, rng: np.random.Generator, sweeps: int = SWEEPS, start: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Paths (n, H K) from ``prepare``'s ingredients, and the constrained quantities z drawn."""
    if not p.c.shape[1]:
        return p.y0, p.c
    z = box_gibbs(p.c, p.S, p.lower, p.upper, rng, sweeps, start)
    return p.y0 + (z - p.c) @ p.W.T, z


def moments(A: np.ndarray, impact: np.ndarray, point: np.ndarray, conds: Sequence[Condition], driving: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
    """Exact conditions in closed form: the conditional mean path (H, K) and the covariance of the stacked path, with
    the shocks not marked ``driving`` at their unconditional distribution."""
    H, K = point.shape
    S = impact.shape[1]
    M = stacked(A, impact, H)
    Md, Mo = M[:, driving], M[:, ~driving]
    De = _rows(conds, M, driving, K, S)
    Ge = _pinv_rows(De, "conditions")
    q = np.array([c.value for c in conds]) - _offsets(conds, point.reshape(1, -1), K)[0]
    mean = point.reshape(-1) + Md @ (Ge @ q)
    Ro = np.array([np.zeros(Mo.shape[1]) if c.on_shock else Mo[c.i * K + c.k] for c in conds]).reshape(len(conds), Mo.shape[1])
    T = Mo - Md @ Ge @ Ro  # how the other shocks move the path, net of the driving shocks' answer to them
    F = np.eye(Md.shape[1]) - Ge @ De
    cov = T @ T.T + Md @ F @ Md.T
    return mean.reshape(H, K), cov


def driving_mask(shocks: Sequence[str] | None, names: Sequence[str], H: int) -> np.ndarray:
    """Which stacked shocks (period-major) may move: all, or the named ones in every period."""
    if shocks is None:
        return np.ones(H * len(names), bool)
    if isinstance(shocks, str):
        shocks = [shocks]
    unknown = [s for s in shocks if s not in names]
    if unknown:
        raise SpecificationError(f"{', '.join(unknown)} {'is' if len(unknown) == 1 else 'are'} not a shock of this VAR "
                                 f"({', '.join(names)})")
    if not shocks:
        raise SpecificationError("name at least one shock that may move to meet the conditions")
    return np.tile(np.isin(list(names), list(shocks)), H)
