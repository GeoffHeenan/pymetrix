"""Optimal control of a model's path: the paths of some exogenous instruments that minimise a quadratic loss, the
model solved as the constraint.

    oc = optimal_control(model, data, ("2026Q2", "2046Q1"), instruments=["rfftay_aerr"],
                         loss={"ugap": 1.0, "pic4": 1.0, "delrff": 1.0}, desired={"pic4": 2.0},
                         control=("2026Q2", "2041Q1"), discount=0.99, scenario=Scenario(adjust=track))
    oc.solution, oc.baseline       # the model solved with the optimal moves, and without them
    oc.moves                       # the instruments' optimal moves (added to their paths), by control period
    oc.loss, oc.baseline_loss, oc.gradient

This is the optimal-control ("opt") simulation of the Federal Reserve's ``mce_solve_library.prg`` as its
``ocpolicy.prg`` runs it on FRB/US: the policy instrument is an add factor on a policy rule (``rfftay_aerr``), moved
in each period of ``control``, and the loss is

    sum over t in span, v in loss of  discount^t * weight_v * (v(t) - desired_v(t))^2

with ``t`` counted from the span's first period. ``desired`` is a number or a path for each variable (zero when not
given). The policymaker commits to the whole path at the start, as in the Fed's program (a commitment-based
simulation); with VAR expectations nothing in the model anticipates it.

**Method.** Gauss-Newton on the stacked residuals ``sqrt(discount^t weight_v) (v(t) - desired_v(t))``. Their
derivatives in the moves come from the model linearised along its solution: one forward pass through the span, each
period solving the period's Jacobian (the one Newton's method uses) for the responses of every variable to every
move at once, from the derivatives of the equations in their lags and in the instruments. A step is taken in full
when it lowers the loss and halved until it does; the iterations stop when the gradient (``2 R' r``) is below
``tol`` times its size at the start, or when the step promises to lower the loss by less than 1e-12 of it (the
solver's own accuracy). For a linear model one step is the optimum.

Backward-looking models only (solved period by period, Newton or Gauss-Seidel): a model with leads (model-consistent
expectations) is not controlled here. Bounds on variables (the Fed's program adds a penalty for the zero lower bound)
are not imposed: add a penalty variable to the model and to the loss if one is needed. Derivatives that do not
evaluate (a saturated logistic switch: inf/inf) are taken as zero, where the function is flat.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from .._api import api
from ..threads import blas_threads
from ..errors import SpecificationError
from ..expr import evaluate
from ..output import Table
from .solvers import ModelError

__all__ = ["OptimalControl", "optimal_control"]


@dataclass(frozen=True, eq=False)
class OptimalControl:
    """An optimal-control solution: see the module docstring."""

    solution: Any  # Solution with the optimal moves
    baseline: Any  # Solution without them
    moves: pd.DataFrame  # control periods by instrument
    loss: float
    baseline_loss: float
    gradient: pd.DataFrame  # the loss's derivatives in the moves, at the optimum
    iterations: int
    weights: Mapping[str, float] = field(default_factory=dict)
    discount: float = 1.0
    history: tuple[float, ...] = ()  # the loss at each iteration

    @property
    def gradient_norm(self) -> float:
        return float(np.linalg.norm(self.gradient.to_numpy()))

    def summary(self) -> Table:
        rows = [["loss", self.baseline_loss, self.loss]]
        notes = [f"Instruments: {', '.join(self.moves.columns)}, moved in {len(self.moves)} periods "
                 f"({self.moves.index[0]} to {self.moves.index[-1]}); discount {self.discount:g}",
                 f"Loss weights: {', '.join(f'{v} {w:g}' for v, w in self.weights.items())}",
                 f"Gauss-Newton: {self.iterations} iterations; gradient norm at the optimum {self.gradient_norm:.3g}"]
        return Table(["baseline", "optimal"], [r[1:] for r in rows], index=[r[0] for r in rows], title="Optimal control", notes=notes)

    def __repr__(self) -> str:
        return (f"<OptimalControl: loss {self.loss:.6g} (baseline {self.baseline_loss:.6g}), {self.iterations} iterations, "
                f"gradient norm {self.gradient_norm:.3g}>")

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


def _periods(span: Any, freq: Any) -> pd.PeriodIndex:
    from ..data.sample import as_span

    first, last = as_span(span)
    return pd.period_range(pd.Period(str(first), freq=freq), pd.Period(str(last), freq=freq), freq=freq)


class _Problem:
    """The model, the loss and the moves: solving at given moves, the loss's residuals and their derivatives."""

    def __init__(self, model: Any, data: Any, span: Any, scenario: Any, solver: Any, instruments: list[str],
                 control: pd.PeriodIndex, loss: Mapping[str, float], desired: Mapping[str, Any], discount: float):
        from .model import _path, _prepare

        self.model, self.data, self.span, self.scenario, self.solver = model, data, span, scenario, solver
        self.instruments, self.control = instruments, control
        prep = _prepare(model, data, span, scenario)
        if prep.varying:
            raise SpecificationError("optimal control takes a scenario whose fixes and holds apply in every period: "
                                     "partial fixes and holds (Scenario(partial=...)) are not available here")
        self.full, self.positions = prep.full, np.flatnonzero(prep.rows)
        self.periods = prep.full[self.positions]
        self.eqs, self.unknowns = prep.eqs, list(prep.unknowns)
        missing = [p for p in control if p not in self.periods]
        if missing:
            raise SpecificationError(f"the control periods must lie in the span: {missing[0]} does not")
        self.base_shift = {v: np.nan_to_num(_path(scenario.shift.get(v), self.periods)) for v in instruments}
        self.names = list(loss)
        T = len(self.periods)
        self.scale = np.array([[np.sqrt(float(loss[v]) * discount ** t) for t in range(T)] for v in self.names])
        self.desired = np.array([np.nan_to_num(_path(desired.get(v, 0.0), self.periods)) for v in self.names])
        self.columns = [(v, p) for v in instruments for p in control]
        self._derivatives()

    def _derivatives(self) -> None:
        from .fast import _derivatives as derivatives, functions, parameter_values

        system = self.model._system
        fns = functions(self.model, self.eqs)
        if fns is not None:
            self.pairs = list(fns.pairs)
            self.compiled = (fns.derivatives, parameter_values(self.model, fns))
        else:
            residuals = tuple(system.equations[self.model.model_equations.index(e)].residual for e in self.eqs)
            found = derivatives(residuals)
            self.pairs = [p for p, _ in found]
            self.exprs = [d for _, d in found]
            self.compiled = None
            self.params = dict(system.parameters.values())

    def solve(self, moves: np.ndarray) -> Any:
        shift = dict(self.scenario.shift)
        k = len(self.control)
        for n, v in enumerate(self.instruments):
            path = self.base_shift[v].copy()
            path[self.periods.get_indexer(self.control)] += moves[n * k:(n + 1) * k]
            shift[v] = pd.Series(path, index=self.periods)
        return self.model.solve(self.data, self.span, replace(self.scenario, shift=shift), solver=self.solver)

    def residuals(self, solution: Any) -> np.ndarray:
        values = solution.frame.loc[self.periods, self.names].to_numpy(float).T
        return (self.scale * (values - self.desired)).ravel()

    def sensitivities(self, solution: Any) -> np.ndarray:
        """The loss residuals' derivatives in the moves, from the model linearised along ``solution``."""
        frame = solution.frame.reindex(self.full)
        positions, full = self.positions, len(self.full)

        def resolve(name: str, lag: int) -> np.ndarray:
            p = positions + lag
            values = frame[name].to_numpy(float) if name in frame else np.full(full, np.nan)
            out = np.full(len(p), np.nan)
            ok = (p >= 0) & (p < full)
            out[ok] = values[p[ok]]
            return out

        T = len(positions)
        if self.compiled is not None:
            fn, params = self.compiled
            values = fn([resolve(name, lag) for name, lag in fn.refs], params)
        else:
            values = [evaluate(d, resolve, self.params) for d in self.exprs]
        values = [np.nan_to_num(np.broadcast_to(np.asarray(v, dtype=float), (T,)), nan=0.0, posinf=0.0, neginf=0.0) for v in values]
        uidx = {u: j for j, u in enumerate(self.unknowns)}
        n, m = len(self.eqs), len(self.columns)
        S = {u: np.zeros((full, m)) for u in self.unknowns}  # responses of each unknown, by position
        U = {v: np.zeros((full, m)) for v in self.instruments}  # the moves themselves
        at = {p: j for j, p in enumerate(self.columns)}
        for (v, p), j in at.items():
            U[v][self.full.get_loc(p), j] = 1.0
        current = [(k, i, uidx[name]) for k, (i, name, lag) in enumerate(self.pairs) if lag == 0 and name in uidx]
        lagged = [(k, i, S[name], lag) for k, (i, name, lag) in enumerate(self.pairs) if lag != 0 and name in uidx]
        moved = [(k, i, U[name], lag) for k, (i, name, lag) in enumerate(self.pairs) if name in U]
        with blas_threads():  # one period's system is small: BLAS's threads cost more than they save
            for t, pos in enumerate(positions):
                self._period(t, pos, values, current, lagged, moved, uidx, S, n, m, full)
        rows = []
        for r, v in enumerate(self.names):
            response = S[v][positions] if v in S else np.zeros((T, m))
            rows.append(self.scale[r][:, None] * response)
        return np.vstack(rows)

    def _period(self, t: int, pos: int, values: list[np.ndarray], current: list[Any], lagged: list[Any], moved: list[Any],
                uidx: dict[str, int], S: dict[str, np.ndarray], n: int, m: int, full: int) -> None:
        """One period of the forward pass: the responses of this period's unknowns to every move."""
        A = np.zeros((n, n))
        for k, i, j in current:
            A[i, j] += values[k][t]
        B = np.zeros((n, m))
        for k, i, source, lag in (*lagged, *moved):
            q = pos + lag
            if 0 <= q < full and values[k][t] != 0.0:
                B[i] += values[k][t] * source[q]
        try:
            X = np.linalg.solve(A, -B)
        except np.linalg.LinAlgError:
            raise ModelError(f"{self.full[pos]}: the model's Jacobian is singular along the solution") from None
        for u, j in uidx.items():
            S[u][pos] = X[j]


@api("pymetrix:optimal_control")
def optimal_control(model: Any, data: Any, span: Any, instruments: Sequence[str], loss: Mapping[str, float],
                    desired: Mapping[str, Any] | None = None, control: Any = None, discount: float = 1.0,
                    scenario: Any = None, solver: Any = None, tol: float = 1e-6, max_iter: int = 20) -> OptimalControl:
    """The paths of ``instruments`` (exogenous series, moved in each period of ``control``: the span by default)
    that minimise the discounted quadratic ``loss`` over ``span``, with ``model`` solved under ``scenario`` as the
    constraint: see the module docstring."""
    from .model import Model
    from .scenario import Scenario
    from .solvers import GaussSeidel, Newton

    if not isinstance(model, Model):
        raise SpecificationError(f"optimal control needs a Model, not {type(model).__name__}")
    if model.has_leads:
        raise SpecificationError("optimal control is for backward-looking models (solved period by period): this model has "
                                 "leads (model-consistent expectations)")
    scenario = scenario or Scenario(name="baseline")
    solver = solver or Newton()
    if not isinstance(solver, (Newton, GaussSeidel)):
        raise SpecificationError("optimal control solves the model period by period: solver is Newton() or GaussSeidel()")
    instruments = list(dict.fromkeys(instruments))
    if not instruments:
        raise SpecificationError("name at least one instrument: an exogenous series to move")
    bad = [v for v in instruments if v not in model.exogenous]
    if bad:
        raise SpecificationError(f"instruments are exogenous series of the model: {', '.join(bad)} are not")
    if not loss:
        raise SpecificationError("give the loss: {variable: weight}")
    bad = [v for v in loss if v not in model.endogenous]
    if bad:
        raise SpecificationError(f"the loss names {', '.join(bad)}, which are not endogenous variables of the model")
    if any(not np.isfinite(float(w)) or float(w) < 0 for w in loss.values()) or not any(float(w) > 0 for w in loss.values()):
        raise SpecificationError("loss weights are 0 or more, and at least one is positive")
    bad = [v for v in (desired or {}) if v not in loss]
    if bad:
        raise SpecificationError(f"desired values for {', '.join(bad)}, which are not in the loss")
    if not 0 < float(discount) <= 1:
        raise SpecificationError("the discount factor is in (0, 1]")
    freq = data.frame.index.freq
    control_periods = _periods(span if control is None else control, freq)
    problem = _Problem(model, data, span, scenario, solver, instruments, control_periods, dict(loss), dict(desired or {}),
                       float(discount))

    moves = np.zeros(len(problem.columns))
    baseline = solution = problem.solve(moves)
    r = problem.residuals(solution)
    value = float(r @ r)
    history = [value]
    R = problem.sensitivities(solution)
    g0 = float(np.linalg.norm(2 * R.T @ r))
    iterations = 0
    while iterations < max_iter and np.linalg.norm(2 * R.T @ r) > tol * max(g0, 1e-300):
        step = np.linalg.lstsq(R, -r, rcond=None)[0]
        predicted = value - float((r + R @ step) @ (r + R @ step))  # the fall the linearised model promises
        if predicted <= 1e-12 * max(value, 1e-300):  # nothing left to gain beyond the solver's own accuracy
            break
        for _ in range(12):  # halved until the loss falls
            trial = problem.solve(moves + step)
            r_trial = problem.residuals(trial)
            if float(r_trial @ r_trial) < value:
                break
            step = step / 2
        else:
            break
        iterations += 1
        moves, solution, r = moves + step, trial, r_trial
        value = float(r @ r)
        history.append(value)
        R = problem.sensitivities(solution)
    gradient = 2 * R.T @ r
    k = len(control_periods)
    frame = pd.DataFrame({v: moves[n * k:(n + 1) * k] for n, v in enumerate(instruments)}, index=control_periods)
    grad = pd.DataFrame({v: gradient[n * k:(n + 1) * k] for n, v in enumerate(instruments)}, index=control_periods)
    return OptimalControl(solution, baseline, frame, value, history[0], grad, iterations, dict(loss), float(discount), tuple(history))
