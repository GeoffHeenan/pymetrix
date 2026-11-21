"""Steady states with growth: a balanced growth path, on which every variable moves by a constant amount each period.

    ss = model.steady_state(growth=True)              # levels (in the reference period) and growth rates
    ss.growth["l_y"]                                   # e.g. 0.5 a quarter: log output times 100, 2 percent a year
    ss.undetermined                                    # levels the model does not pin down (only their growth)
    model.steady_state(growth=True, fix={"l_y": 460})  # pin them, e.g. to the data
    model.linearize(ss).solve()                        # around the path; unit roots allowed

Trending models (gap-trend forecasting models such as FINEX and the quarterly projection models) write levels as
logs times 100, so trends are additive: ``x(t) = level + growth * t``. Substituting the path into every equation
at two consecutive periods gives twice as many equations as variables, for the levels and the growth rates
together; a third period checks that the path is balanced (the equations hold at every period, not just two).
Levels of variables with unit roots are usually not pinned down: any level that grows at the right rate is a
steady state. They are found from the starting values (``guess``, 1 by default) and reported as undetermined, by
the null space of the Jacobian; ``fix=`` sets them. Variables in levels with multiplicative growth (``x(t) =
level * (1 + growth)^t``): write them in logs, ``model.in_logs(...)``, and their logs grow by ``log(1 + growth)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import linalg, optimize

from .solvers import ModelError

if TYPE_CHECKING:
    from .linear import SteadyState
    from .model import Model


def steady_state_with_growth(model: Model, exogenous: Mapping[str, float] | None = None, guess: Mapping[str, float] | None = None,
                             fix: Mapping[str, float] | None = None) -> SteadyState:
    from ..expr import evaluate
    from .fast import functions, parameter_values
    from .linear import SteadyState

    exogenous = {v: float((exogenous or {}).get(v, 0.0)) for v in model.exogenous}
    fix = {k: float(v) for k, v in (fix or {}).items()}
    endo = list(model.endogenous)
    bad = [k for k in fix if k not in endo]
    if bad:
        raise ModelError(f"fix=: {', '.join(bad)} are not endogenous variables of the model")
    free = [v for v in endo if v not in fix]  # unknown levels
    n_free, n = len(free), len(endo)
    pos = {v: j for j, v in enumerate(free)}
    gpos = {v: n_free + j for j, v in enumerate(endo)}  # growth of every endogenous variable is unknown

    def levels_and_growth(z: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
        levels = {**exogenous, **fix, **{v: float(z[pos[v]]) for v in free}}
        growth = {v: float(z[gpos[v]]) for v in endo}
        return levels, growth

    fns = functions(model)
    p = parameter_values(model, fns) if fns is not None else None
    residuals = [e.residual for e in model.model_equations]

    def at(levels: dict[str, float], growth: dict[str, float], name: str, lag: int, tau: int) -> float:
        return levels[name] + growth.get(name, 0.0) * (tau + lag)

    def F(z: np.ndarray, taus: tuple[int, ...] = (0, 1)) -> np.ndarray:
        levels, growth = levels_and_growth(z)
        out = []
        for tau in taus:
            if fns is not None:
                out.extend(fns.residuals([at(levels, growth, nm, lag, tau) for nm, lag in fns.residuals.refs], p))
            else:
                out.extend(evaluate(r, lambda nm, lag: at(levels, growth, nm, lag, tau)) for r in residuals)
        return np.asarray(out, dtype=float)

    def J(z: np.ndarray) -> np.ndarray:
        if fns is None:
            return _numeric_jacobian(F, z)
        levels, growth = levels_and_growth(z)
        out = np.zeros((2 * len(residuals), len(z)))
        for k, tau in enumerate((0, 1)):
            values = fns.derivatives([at(levels, growth, nm, lag, tau) for nm, lag in fns.derivatives.refs], p)
            for (i, name, lag), d in zip(fns.pairs, values):
                row = k * len(residuals) + i
                if name in pos:
                    out[row, pos[name]] += float(d)
                if name in gpos:
                    out[row, gpos[name]] += float(d) * (tau + lag)
        return out

    start = np.array([float((guess or {}).get(v, 1.0)) for v in free] + [0.0] * n)
    # a very weak pull of each free level towards its starting value: it decides only the levels the equations leave
    # free (variables with unit roots), which would otherwise drift, e.g. logs far below zero where every exp(.) term
    # vanishes and the equations hold trivially; the levels the equations pin down move by a negligible amount
    pull = 1e-7

    def F_pulled(z: np.ndarray) -> np.ndarray:
        return np.concatenate([F(z), pull * (z[:n_free] - start[:n_free])])

    def J_pulled(z: np.ndarray) -> np.ndarray:
        extra = np.zeros((n_free, len(z)))
        extra[np.arange(n_free), np.arange(n_free)] = pull
        return np.vstack([J(z), extra])

    found = optimize.least_squares(F_pulled, start, jac=J_pulled, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=2000)
    # then the equations alone, from there: exact again, and too close to drift along the free levels
    polished = optimize.least_squares(F, found.x, jac=J, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=50)
    z = polished.x if np.max(np.abs(F(polished.x)), initial=0.0) <= np.max(np.abs(F(found.x)), initial=0.0) else found.x
    worst = np.max(np.abs(F(z))) if len(residuals) else 0.0
    if worst > 1e-8:
        i = int(np.argmax(np.abs(F(z)))) % len(residuals)
        raise ModelError(f"no steady state with growth found from these starting values (equation {model.model_equations[i].name!r} "
                         f"off by {worst:.3g}); give guess={{...}} or fix={{...}}, or check the model has a balanced growth path")
    third = F(z, (2, 5))
    if np.max(np.abs(third)) > 1e-7:
        i = int(np.argmax(np.abs(third))) % len(residuals)
        raise ModelError(f"the path is not balanced: equation {model.model_equations[i].name!r} holds in two periods but not "
                         "in later ones (a variable may need to grow at a changing rate, or enter in levels)")
    levels, growth = levels_and_growth(z)
    undetermined = _undetermined(J(z), free)
    clean = {k: (0.0 if abs(v) < 1e-12 else v) for k, v in levels.items()}
    rates = {k: v for k, v in growth.items() if abs(v) > 1e-12}
    return SteadyState({**{v: clean[v] for v in endo}, **exogenous}, model, growth=rates, undetermined=tuple(undetermined))


def _numeric_jacobian(F: Any, z: np.ndarray) -> np.ndarray:
    f0 = F(z)
    out = np.empty((len(f0), len(z)))
    for j in range(len(z)):
        h = 1e-6 * max(1.0, abs(z[j]))
        step = z.copy()
        step[j] += h
        out[:, j] = (F(step) - f0) / h
    return out


def _undetermined(jac: np.ndarray, free: list[str]) -> list[str]:
    """Levels that can move without breaking any equation: the level part of the Jacobian's null space."""
    if not free or jac.size == 0:
        return []
    null = linalg.null_space(jac, rcond=1e-9)
    if null.size == 0:
        return []
    weight = np.abs(null[: len(free)]).max(axis=1)
    return [v for v, w in zip(free, weight) if w > 1e-6]
