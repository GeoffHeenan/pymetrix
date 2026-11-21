"""Simulation plans for linear(ised) models: anticipated and unanticipated shocks, exogenize and endogenize.

    sol = model.linearize().solve()                      # or a template's or a fitted model's solution
    path = sol.solve(data, ("2025Q1", "2026Q4"), Scenario(
        set={"ei": {"2025Q1": 0.25}},                    # a shock path (anticipated unless named below)
        hold={"i": {"2025Q1": 1.0, "2025Q2": 1.0}},      # hold the policy rate for two quarters...
        free=["ei"],                                     # ...with the policy shock (exogenize i, endogenize ei)
        unanticipated=["ey"],                            # surprises: known only when they happen
    ))
    model.solve(data, span, scenario, solver=FirstOrder())   # the same, linearising the model first

The solution ``x(t) = P x(t-1) + Q e(t)`` holds when nothing is known about future shocks. A shock path known in
advance (anticipated) moves the economy before it happens: with ``A0 E x(t+1) + A1 x(t) + A2 x(t-1) + B e(t) = 0``,

    x(t) = P x(t-1) + sum_k F^k Q a(t+k) + Q u(t),        F = -(A0 P + A1)^-1 A0

where ``a`` are the anticipated shocks, known from the first simulated period, and ``u`` the unanticipated ones,
known only in their own period. The sum is computed backwards, ``w(t) = Q a(t) + F w(t+1)``.

Targets (exogenize) are paths of endogenous variables, with missing values where a variable is free to move. Each
target is paired, in order, with a ``free`` shock (endogenize), which takes whatever values put its variable on the
target in those periods; an anticipated free shock is known from the start, an unanticipated one only in its
period. The model is linear, so the free shock values solve one linear system. Values are in levels: the steady
state is added and subtracted.

A least-norm conditional forecast instead, when the free shocks are not one per hold or ``hold_sd``/``free_sd`` is
given: every free shock is free in each period up to the last held one, and the plan takes the smallest shocks, in
units of ``free_sd`` (default 1), that meet the holds,

    min sum (z / free_sd)^2 + sum over soft holds ((x - hold) / hold_sd)^2    subject to the exact holds,

as Waggoner and Zha's (1999) conditional forecasts with soft conditions. A hold with a standard deviation is soft:
met more closely the smaller it is; none, or 0, is exact.

Ranges (``hold_lower`` and ``hold_upper``) keep a variable within bounds in the periods given: the same problem with
the inequality constraints ``lower <= x <= upper`` added, a convex quadratic programme solved by sequential least
squares (SLSQP) from the solution without them. A range that does not bind leaves the plan as it was; one that does
moves the free shocks only as far as needed, so the variable ends on the bound. This is the deterministic
counterpart of a VAR's range conditions, which draw from the truncated distribution instead. Shocks after the last held period stay as given. Starting values (the periods before the span) come from ``data``, in levels, or are
the steady state when no data are given.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .solvers import ModelError

if TYPE_CHECKING:
    from ..data import Dataset
    from .linear import LinearSolution
    from .scenario import Scenario
    from .solution import Solution


def _periods(span: tuple[Any, Any], freq: Any = None) -> pd.PeriodIndex:
    from ..data.sample import as_span
    span = as_span(span)  # '1990Q1:2019Q4' or a pair
    start = pd.Period(str(span[0]), freq=freq) if freq is not None else pd.Period(str(span[0]))
    end = pd.Period(str(span[1]), freq=start.freq)
    if end < start:
        raise ModelError(f"the span ends ({end}) before it starts ({start})")
    return pd.period_range(start, end, freq=start.freq, name="time")


def _path(value: Any, periods: pd.PeriodIndex) -> np.ndarray:
    """A path over the periods: a number (every period), a Series, or a mapping of periods to values."""
    if value is None:
        return np.full(len(periods), np.nan)
    if isinstance(value, Mapping) and not isinstance(value, pd.Series):
        value = pd.Series({pd.Period(str(k), freq=periods.freq): float(v) for k, v in value.items()})
    if isinstance(value, pd.Series):
        s = value
        if not isinstance(s.index, pd.PeriodIndex):
            s = pd.Series(s.to_numpy(float), index=pd.PeriodIndex(s.index.astype(str), freq=periods.freq))
        outside = [str(p) for p in s.dropna().index if p not in periods]
        if outside:
            raise ModelError(f"values for periods outside the span: {', '.join(outside[:4])}")
        return s.reindex(periods).to_numpy(float)
    return np.full(len(periods), float(value))


def forward(sol: LinearSolution) -> np.ndarray:
    """``F = -(A0 P + A1)^-1 A0``: how news about next period's shocks enters today's variables."""
    lin = sol.model
    return -np.linalg.solve(lin.A0 @ sol.P + lin.A1, lin.A0)


def paths(sol: LinearSolution, x0: np.ndarray, anticipated: np.ndarray, unanticipated: np.ndarray,
          F: np.ndarray | None = None) -> np.ndarray:
    """Deviations from the steady state, period by period (T x variables), given the starting state, the
    anticipated shocks (T x shocks, known from the first period) and the unanticipated ones (known when they occur)."""
    F = forward(sol) if F is None else F
    T = anticipated.shape[0]
    w = np.zeros((T + 1, len(x0)))
    for t in range(T - 1, -1, -1):
        w[t] = sol.Q @ anticipated[t] + F @ w[t + 1]
    out = np.empty((T, len(x0)))
    x = x0
    for t in range(T):
        x = sol.P @ x + w[t] + sol.Q @ unanticipated[t]
        out[t] = x
    return out


def solve_plan(sol: LinearSolution, data: Dataset | None, span: tuple[Any, Any], scenario: Scenario | None = None,
               regimes: Any = None) -> Solution:
    """The path over ``span`` under ``scenario`` (see the module docstring), with known-date ``regimes`` if any (see
    ``pymetrix.models.regimes``)."""
    return _solve_plan(sol, data, span, scenario, regimes)[0]


def _solve_plan(sol: LinearSolution, data: Dataset | None, span: tuple[Any, Any], scenario: Scenario | None = None,
                regimes: Any = None) -> tuple[Solution, Any, Any]:
    """``solve_plan``, with the solution it used (that of the regime in force before the first period) and the regimes'
    plan (None without a regime change in the span)."""
    from .._capture import data_fingerprint
    from .._version import __version__
    from ..data import Dataset
    from ..metadata import Provenance
    from .scenario import Scenario
    from .solution import Solution

    sc = scenario or Scenario(name="baseline")
    if sc.fix or sc.adjust:
        raise ModelError("fix= and adjust= need a nonlinear solver; with a first-order solution, hold a variable with "
                         "hold= and a free= shock")
    plan = None
    if regimes:
        from .regimes import regime_plan

        freq = data.frame.index.freq if data is not None and isinstance(data.frame.index, pd.PeriodIndex) else None
        sol, plan = regime_plan(sol, regimes, _periods(span, freq))
    variables, shocks, observed = list(sol.variables), list(sol.shocks), list(sol.model.observed)
    steady = dict(sol.model.steady.values)
    trend = dict(getattr(sol.model.steady, "growth", {}) or {})  # a balanced growth path: levels move each period

    def check(names: Any, allowed: list[str], what: str, kind: str) -> None:
        bad = [v for v in names if v not in allowed]
        if bad:
            raise ModelError(f"{what}: {', '.join(bad)} are not {kind} of this solution ({', '.join(allowed)})")

    check(sc.set, shocks, "set=", "shocks")
    check(sc.shift, shocks, "shift=", "shocks")
    check(sc.free, shocks, "free=", "shocks")
    check(sc.unanticipated, shocks, "unanticipated=", "shocks")
    check(sc.hold, observed, "hold=", "variables")
    check(sc.hold_sd, list(sc.hold), "hold_sd=", "held variables")
    check(sc.free_sd, list(sc.free), "free_sd=", "free shocks")
    check(sc.hold_lower, observed, "hold_lower=", "variables")
    check(sc.hold_upper, observed, "hold_upper=", "variables")
    holding = bool(sc.hold) or sc.ranges
    least_norm = bool(sc.hold_sd or sc.free_sd or sc.ranges) or len(sc.free) != len(sc.hold)
    if holding != bool(sc.free):
        raise ModelError(f"{len(sc.hold)} held variables but {len(sc.free)} free shocks; pair each hold= with one free= shock, "
                         "in order (or give more free shocks than holds, for a least-norm plan)")
    if any(not np.isfinite(float(w)) or float(w) <= 0 for w in sc.free_sd.values()):
        raise ModelError("free_sd= values are positive numbers")

    # periods: the span, after any history in the data
    freq = data.frame.index.freq if data is not None and isinstance(data.frame.index, pd.PeriodIndex) else None
    periods = _periods(span, freq)
    if data is not None:
        if not isinstance(data.frame.index, pd.PeriodIndex):
            raise ModelError("solving over a span needs a dataset with a time axis")
        idx = data.frame.index
        full = pd.period_range(min(idx.min(), periods[0]), max(idx.max(), periods[-1]), freq=periods.freq, name=idx.name or "time")
        frame = data.frame.reindex(full)
    else:
        full = pd.period_range(periods[0] - 4, periods[-1], freq=periods.freq, name="time")
        frame = pd.DataFrame(index=full)
    t0 = int(full.get_loc(periods[0]))
    T = len(periods)

    # the starting state, in deviations from the steady state
    x0 = np.zeros(len(variables))
    needed = np.any(np.abs(sol.P) > 0, axis=0) if plan is None else plan.needed
    missing = []
    for j, v in enumerate(variables):
        base, lag = _state_of(v, observed)
        if base is None or not needed[j]:
            continue
        pos = t0 - 1 - lag
        if data is None:
            continue  # the steady state
        value = frame[base].iloc[pos] if base in frame.columns and pos >= 0 else np.nan
        if not np.isfinite(value):
            missing.append(f"{base} in {full[pos] if pos >= 0 else 'a period before the data'}")
            continue
        x0[j] = float(value) - (steady[base] + trend.get(base, 0.0) * (pos - t0))  # the path's reference is the first period
    if missing:
        raise ModelError(f"starting values are missing for {', '.join(missing)}; give them in the data (a fitted model's "
                         "smoothed variables have them), or no data to start from the steady state")

    # shock paths (levels in the data or the scenario; deviations below)
    level = np.zeros((T, len(shocks)))
    for k, s in enumerate(shocks):
        if s in frame.columns:
            level[:, k] = np.nan_to_num(frame[s].iloc[t0:t0 + T].to_numpy(float), nan=steady.get(s, 0.0))
        else:
            level[:, k] = steady.get(s, 0.0)
        if s in sc.set:
            p = _path(sc.set[s], periods)
            level[:, k] = np.where(np.isfinite(p), p, level[:, k])
        if s in sc.shift:
            level[:, k] += np.nan_to_num(_path(sc.shift[s], periods))
    dev = level - np.array([steady.get(s, 0.0) for s in shocks])
    surprise = np.array([s in sc.unanticipated for s in shocks])
    F = forward(sol)

    def run(devs: np.ndarray, start: np.ndarray, constant: bool = True) -> np.ndarray:
        if plan is not None:  # time-varying rules through the regimes; without their constants for responses
            return plan.paths(start, devs * ~surprise, devs * surprise, constant)
        return paths(sol, start, devs * ~surprise, devs * surprise, F)

    # exogenize/endogenize: free shock values in the target periods, from the responses of the targets
    conditions: list[tuple[int, int, float]] = []  # (period, variable, target deviation)
    unknowns: list[tuple[int, int]] = []  # (period, shock)
    for (v, path), s in zip(sc.hold.items(), sc.free):
        p = _path(path, periods)
        for t in np.flatnonzero(np.isfinite(p)):
            conditions.append((int(t), variables.index(v), float(p[t]) - steady[v] - trend.get(v, 0.0) * int(t)))
            unknowns.append((int(t), shocks.index(s)))
    if least_norm and holding:
        dev = _least_norm(sc, periods, variables, shocks, steady, trend, dev, x0, run)
        conditions = []
    if conditions:
        base = run(dev, x0)
        M = np.empty((len(conditions), len(unknowns)))
        for c, (t, k) in enumerate(unknowns):
            unit = np.zeros_like(dev)
            unit[t, k] = 1.0
            response = run(unit, np.zeros_like(x0), constant=False)
            M[:, c] = [response[tt, j] for tt, j, _ in conditions]
        gap = np.array([target - base[tt, j] for tt, j, target in conditions])
        if np.linalg.matrix_rank(M) < len(unknowns):
            raise ModelError("the free shocks cannot put the targets on their paths (the responses are not independent): "
                             "check that each free shock moves its target, and that an unanticipated shock is not asked "
                             "to hit a target before it occurs")
        z = np.linalg.solve(M, gap)
        for (t, k), value in zip(unknowns, z):
            dev[t, k] += value
    x = run(dev, x0)

    out = frame.copy()
    for j, v in enumerate(observed):
        column = out[v].to_numpy(float, copy=True) if v in out.columns else np.full(len(full), np.nan)
        column[t0:t0 + T] = x[:, variables.index(v)] + steady[v] + trend.get(v, 0.0) * np.arange(T)
        out[v] = column
    for k, s in enumerate(shocks):
        column = out[s].to_numpy(float, copy=True) if s in out.columns else np.full(len(full), np.nan)
        column[t0:t0 + T] = dev[:, k] + steady.get(s, 0.0)
        out[s] = column
    method = {"type": "first-order rational expectations", "expectations": "anticipated unless named unanticipated",
              **({"conditioning": "least norm"} if least_norm and holding else {}),
              **({"ranges": "least-norm free shocks within the bounds (SLSQP)"} if sc.ranges else {})}
    if plan is not None:
        method["regimes"] = plan.spells(T)
    provenance = Provenance(
        data_vintages={v: data.variables[v].vintage for v in data.names if v in data.variables} if data is not None else {},
        spec={"variables": observed, "shocks": shocks, "scenario": sc.name, "unanticipated": list(sc.unanticipated),
              "targets": list(sc.hold), "free": list(sc.free),
              **({"ranges": sorted({*sc.hold_lower, *sc.hold_upper})} if sc.ranges else {})},
        method=method,
        sample=(str(periods[0]), str(periods[-1])), nobs=T, dropped=0, package_version=__version__,
        data_fingerprint=data_fingerprint(data),
    )
    shell = Dataset.from_frame(out) if data is None else data
    variables_meta = {k: v for k, v in shell.variables.items() if k in out.columns}
    return Solution(out, shell.axes, variables_meta, meta=shell.meta, model=sol, scenario=sc, converged=True, iterations=0,
                    provenance=provenance), sol, plan


def _least_norm(sc: Scenario, periods: pd.PeriodIndex, variables: list[str], shocks: list[str], steady: Mapping[str, float],
                trend: Mapping[str, float], dev: np.ndarray, x0: np.ndarray, run: Any) -> np.ndarray:
    """The free shocks of a least-norm conditional forecast, added to ``dev``: each free shock is free in every period
    up to the last held one; minimise sum (z / free_sd)^2 + sum over soft holds ((value - target) / hold_sd)^2,
    subject to the exact holds. The exact holds are met by the minimum-norm solution, the soft ones then traded off
    against the size of the shocks in the space the exact ones leave free."""
    hard: list[tuple[int, int, float]] = []
    soft: list[tuple[int, int, float, float]] = []
    for v, path in sc.hold.items():
        p = _path(path, periods)
        sd = _path(sc.hold_sd.get(v), periods)
        for t in np.flatnonzero(np.isfinite(p)):
            target = float(p[t]) - steady[v] - trend.get(v, 0.0) * int(t)
            s = float(sd[t]) if np.isfinite(sd[t]) else 0.0
            if s < 0:
                raise ModelError(f"hold_sd= for {v} is negative in {periods[t]}: give zero (exact) or a positive number")
            if s == 0:
                hard.append((int(t), variables.index(v), target))
            else:
                soft.append((int(t), variables.index(v), target, s))
    ranged: list[tuple[int, int, float, float]] = []  # (period, variable, lower, upper) in deviations
    for v in dict.fromkeys([*sc.hold_lower, *sc.hold_upper]):
        lo, hi = _path(sc.hold_lower.get(v), periods), _path(sc.hold_upper.get(v), periods)
        for t in np.flatnonzero(np.isfinite(lo) | np.isfinite(hi)):
            ref = steady[v] + trend.get(v, 0.0) * int(t)
            low = float(lo[t]) - ref if np.isfinite(lo[t]) else -np.inf
            high = float(hi[t]) - ref if np.isfinite(hi[t]) else np.inf
            if low > high:
                raise ModelError(f"the range for {v} in {periods[t]} has its lower bound above its upper bound")
            ranged.append((int(t), variables.index(v), low, high))
    last = max(t for t, *_ in (*hard, *soft, *ranged))
    unknowns = [(t, shocks.index(s)) for s in sc.free for t in range(last + 1)]
    weight = np.array([float(sc.free_sd.get(shocks[k], 1.0)) for _, k in unknowns])
    base = run(dev, x0)
    responses = np.empty((len(unknowns), *base.shape))
    for c, (t, k) in enumerate(unknowns):
        unit = np.zeros_like(dev)
        unit[t, k] = 1.0
        responses[c] = run(unit, np.zeros_like(x0), constant=False)  # without the regimes' constants
    # in scaled shocks u = z / free_sd: the conditions are A u = g
    A_h = np.array([responses[:, t, j] * weight for t, j, _ in hard]).reshape(len(hard), len(unknowns))
    g_h = np.array([target - base[t, j] for t, j, target in hard])
    A_s = np.array([responses[:, t, j] * weight / s for t, j, _, s in soft]).reshape(len(soft), len(unknowns))
    g_s = np.array([(target - base[t, j]) / s for t, j, target, s in soft])
    if hard:
        u = np.linalg.lstsq(A_h, g_h, rcond=None)[0]  # the minimum-norm shocks that meet the exact holds
        if not np.allclose(A_h @ u, g_h, atol=1e-8 * max(1.0, float(np.max(np.abs(g_h))))):
            raise ModelError("the free shocks cannot meet the exact holds: free more shocks, hold fewer values, or make "
                             "some holds soft with hold_sd= (an unanticipated shock cannot move a variable before it occurs)")
        _, sing, vt = np.linalg.svd(A_h)
        rank = int(np.sum(sing > 1e-10 * max(1.0, float(sing[0]) if len(sing) else 1.0)))
        N = vt[rank:].T  # what the exact holds leave free
    else:
        u, N = np.zeros(len(unknowns)), np.eye(len(unknowns))
    if soft and N.shape[1]:
        lhs = np.vstack([N, A_s @ N])
        rhs = np.concatenate([-u, g_s - A_s @ u])
        u = u + N @ np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    if ranged:
        u = _within(u, ranged, responses, base, weight, A_h, g_h, A_s, g_s)
    out = dev.copy()
    for (t, k), value in zip(unknowns, u * weight):
        out[t, k] += value
    return out


def _within(u: np.ndarray, ranged: list[tuple[int, int, float, float]], responses: np.ndarray, base: np.ndarray, weight: np.ndarray,
            A_h: np.ndarray, g_h: np.ndarray, A_s: np.ndarray, g_s: np.ndarray) -> np.ndarray:
    """The least-norm scaled shocks ``u`` with the ranges added: minimise |u|^2 + |A_s u - g_s|^2 subject to the exact
    holds and lower <= base + A_r u <= upper, from the solution without the ranges (optimal when it meets them)."""
    from scipy import optimize

    A_r = np.array([responses[:, t, j] * weight for t, j, _, _ in ranged]).reshape(len(ranged), len(u))
    level = np.array([base[t, j] for t, j, _, _ in ranged])
    lower = np.array([lo for *_, lo, _ in ranged]) - level
    upper = np.array([hi for *_, hi in ranged]) - level
    scale = max(1.0, float(np.max(np.abs(np.concatenate([lower[np.isfinite(lower)], upper[np.isfinite(upper)], [0.0]])))))
    tol = 1e-8 * scale
    x = A_r @ u
    if np.all(x >= lower - tol) and np.all(x <= upper + tol):
        return u
    constraints = []
    if len(g_h):
        constraints.append({"type": "eq", "fun": lambda v: A_h @ v - g_h, "jac": lambda v: A_h})
    lo_rows, hi_rows = np.isfinite(lower), np.isfinite(upper)
    if lo_rows.any():
        constraints.append({"type": "ineq", "fun": lambda v: A_r[lo_rows] @ v - lower[lo_rows], "jac": lambda v: A_r[lo_rows]})
    if hi_rows.any():
        constraints.append({"type": "ineq", "fun": lambda v: upper[hi_rows] - A_r[hi_rows] @ v, "jac": lambda v: -A_r[hi_rows]})

    def objective(v: np.ndarray) -> tuple[float, np.ndarray]:
        r = A_s @ v - g_s
        return float(v @ v + r @ r), 2 * v + 2 * A_s.T @ r

    found = optimize.minimize(objective, u, jac=True, method="SLSQP", constraints=constraints,
                              options={"ftol": 1e-14, "maxiter": 1000})
    v = found.x
    x = A_r @ v
    met = np.all(x >= lower - 1e-6 * scale) and np.all(x <= upper + 1e-6 * scale) and (
        not len(g_h) or np.allclose(A_h @ v, g_h, atol=1e-6 * max(1.0, float(np.max(np.abs(g_h))))))
    if not met:
        raise ModelError("the free shocks cannot keep the variables within the ranges (and meet the exact holds): free more "
                         "shocks, widen the ranges, or hold fewer values")
    return v


def _state_of(name: str, observed: list[str]) -> tuple[str | None, int]:
    """The variable and lag a state stands for: ``x`` is x at t-1 (lag 0 beyond the one in the state), ``x_lag2`` is
    x two periods further back; leads are not states."""
    if name in observed:
        return name, 0
    base, _, tail = name.rpartition("_")
    if base in observed and tail.startswith("lag") and tail[3:].isdigit():
        return base, int(tail[3:])
    return None, 0
