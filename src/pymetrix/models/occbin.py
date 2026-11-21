"""Occasionally binding constraints: the piecewise-linear solution of Guerrieri and Iacoviello (2015), OccBin.

    nk = Model([..., Equation("i ~ max(rho*i[-1] + (1-rho)*(phi*pi + 0.5*y), floor) + ei", ...)], params=...)
    path = occbin(template, {"ey": {1: -3.0}}, periods=40)     # a large demand shock: the floor binds for a while
    path.frame                     # levels, period by period
    path.binding                   # when each constraint binds
    path.plot(["y", "pi", "i"])    # with the unconstrained (linear) path, dashed
    path.summary()

The constraints are the model's own ``max`` and ``min``: an equation such as ``i ~ max(i_rule, floor)`` defines
two regimes, the branch that holds at the steady state (the reference regime: the constraint is slack) and the
other (it binds). Each regime is linearised around the reference steady state; a binding regime keeps the constant
its equations leave there. Given a guess of the periods in which each constraint binds, the time-varying decision
rules are found backwards from the end of the horizon, where the reference regime holds for good, and the path is
simulated forwards; each constraint is then checked in levels along the path (the max or min, evaluated with the
model's own expressions), the guess updated, and so on until it no longer changes.

Shocks are surprises, period by period, as in Dynare's OccBin (``shocks(surprise)``): each period's shock is
unexpected, and agents then expect no more. With ``anticipated=True`` the whole shock path is known from the first
period (a perfect-foresight solution of the piecewise-linear model).

Known-date regimes (``regimes=``: ``Regime(5, {"phi": 2.0})``, periods numbered from 1; see
``pymetrix.models.regimes``) combine with the constraints: each period's matrices come from both its known regime
(its parameters and equations) and the constraints' guessed regime, linearised at the steady state of the regime in
force before the first period, and each period's constraints are checked with its regime's parameters. After the
horizon the constraints are slack and the known regimes follow their own rules, the last holding for good. A regime's
equations may rewrite a max or min (another floor), leave one out (a constraint switched off for a while: slack, never
binding, then) or bring one in: the guess-and-check runs over the constraints of all the regimes, matched by the
equation each is in, a constraint absent from a period's regime counted slack there; one brought in is slack on the
branch that holds at the reference steady state. A surprise regime (``expected=False``) is unknown until it starts:
each solution uses the regimes believed in its first period, so with the shocks known in advance (``anticipated``) the
path is solved again from the state reached when a surprise starts; with surprise shocks it is solved each period anyway.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from .._api import api
from ..output import Table
from .solvers import ModelError

__all__ = ["OccBinPath", "occbin"]


@dataclass(frozen=True)
class _Constraint:
    call: Any  # the max or min node
    reference: int  # the branch that holds at the steady state (0 or 1)
    label: str = ""  # the equation it is in, e.g. "policy (max)"

    @property
    def key(self) -> str:
        return str(self.call)

    def branch(self, k: int) -> Any:
        return self.call.args[k]


def _calls(model: Any) -> list[tuple[Any, str]]:
    """The model's max and min nodes, in order, each with its label (the equation it is in)."""
    found: dict[str, Any] = {}
    labels: dict[str, str] = {}
    for e in model.model_equations:
        for node in e.residual.walk():
            if getattr(node, "elementwise", False) and node.fn in ("max", "min") and str(node) not in found:
                found[str(node)] = node
                label = f"{e.name} ({node.fn})"
                labels[str(node)] = label if label not in labels.values() else f"{label} {len(labels) + 1}"
    return [(call, labels[key]) for key, call in found.items()]


def _constraints(model: Any, steady: Mapping[str, float], empty: bool = False) -> list[_Constraint]:
    """The model's max and min, each with the branch that holds at the steady state; ``empty``: a model without any is
    allowed (a known-date regime may bring one in)."""
    found = _calls(model)
    if not found and not empty:
        raise ModelError("the model has no max or min: OccBin needs an occasionally binding constraint, "
                         "e.g. i ~ max(i_rule, floor)")
    return [_slack(call, label, steady) for call, label in found]


def _slack(call: Any, label: str, steady: Mapping[str, float]) -> _Constraint:
    """The constraint with its reference branch: the one that holds at the steady state ``steady``."""
    from ..expr.evaluate import evaluate

    a, b = (float(evaluate(arg, lambda name, lag: steady[name])) for arg in call.args)
    if np.isclose(a, b, rtol=0, atol=1e-10 * max(1.0, abs(a))):
        raise ModelError(f"{call} has both branches equal at the steady state: OccBin needs the constraint slack there")
    larger = 0 if a > b else 1
    return _Constraint(call, larger if call.fn == "max" else 1 - larger, label)


def _regime_constraints(model: Any, reference: list[_Constraint | None], labels: list[str], steady: Mapping[str, float]
                        ) -> list[_Constraint | None]:
    """A known-date regime's own max and min (its parameters in them), matched by label (the equation each is in, and
    max or min) to ``labels``, the constraints of all the regimes together, which grows by those this regime brings in:
    a constraint the reference regime has keeps its branch at the steady state, one brought in takes the branch that
    holds at the reference steady state; one this regime leaves out is None (slack: it never binds)."""
    found = {label: call for call, label in _calls(model)}
    for label in found:
        if label not in labels:
            labels.append(label)
    out: list[_Constraint | None] = []
    for j, label in enumerate(labels):
        call = found.get(label)
        if call is None:
            out.append(None)
        elif j < len(reference) and reference[j] is not None:
            out.append(_Constraint(call, reference[j].reference, label))  # type: ignore[union-attr]
        else:
            out.append(_slack(call, label, steady))
    return out


def _regime(model: Any, ss: Any, shocks: Sequence[str], constraints: Sequence[_Constraint | None], choice: tuple[int, ...],
            variables: tuple[str, ...] | None = None) -> Any:
    from .linear import LinearModel, linearize

    branches = {c.key: c.branch(k) for c, k in zip(constraints, choice) if c is not None}

    def transform(residual: Any) -> Any:
        return residual.map(lambda n: branches.get(str(n)) if getattr(n, "elementwise", False) else None)

    if variables is None:
        return linearize(model, ss, shocks, transform=transform, constant=True)
    # a known-date regime's model: on the reference's variables and shocks (a shock its equations leave out does nothing)
    present = tuple(s for s in shocks if s in model.exogenous)
    lin = linearize(model, ss, present, transform=transform, constant=True)
    if lin.variables != variables:
        raise ModelError("a regime's equations use lags or leads beyond one period that the model's do not (or no longer "
                         "use ones they do); write the alternative equation with the same timing")
    B = np.zeros((len(lin.variables), len(shocks)))
    for j, s in enumerate(present):
        B[:, list(shocks).index(s)] = lin.B[:, j]
    c = lin.c if lin.c is not None else np.zeros(len(lin.variables))
    return LinearModel(lin.A0, lin.A1, lin.A2, B, lin.variables, tuple(shocks), ss, lin.observed, c)


@dataclass(frozen=True, eq=False)
class OccBinPath:
    """A piecewise-linear path: see the module docstring."""

    frame: pd.DataFrame  # levels of the model's variables, period by period (1, 2, ...)
    deviations: pd.DataFrame  # the same, less the steady state
    binding: pd.DataFrame  # True where each constraint binds
    linear: pd.DataFrame  # levels without the constraints (the reference regime throughout), shocks known as in ``frame``
    shocks: pd.DataFrame
    constraints: tuple[str, ...]
    iterations: int
    anticipated: bool = False
    steady: Mapping[str, float] = field(default_factory=dict)

    def summary(self) -> Table:
        rows = []
        for name in self.constraints:
            periods = [int(p) for p in self.binding.index[self.binding[name]]]
            rows.append([name, _spells(periods) or "never", len(periods)])
        notes = [f"Piecewise-linear solution (OccBin), {len(self.frame)} periods; shocks "
                 + ("anticipated from the first period" if self.anticipated else "unexpected, period by period"),
                 f"Converged in {self.iterations} iteration(s) of the guess of binding periods"]
        return Table(["constraint", "binds in periods", "periods"], rows, title="Occasionally binding constraints", notes=notes)

    def plot(self, variables: Sequence[str] | str | None = None) -> Any:
        """Each variable with the constraints (solid) and without them (dashed)."""
        from ..figures import Figure, Line, Panel

        names = [variables] if isinstance(variables, str) else list(variables or list(self.frame.columns)[:6])
        panels = tuple(Panel((Line(self.frame[v], "with constraints"), Line(self.linear[v], "without", style="--")), title=v,
                             legend=i == 0) for i, v in enumerate(names))
        return Figure(panels, title="Occasionally binding constraints", columns=min(2, len(panels)))

    def __repr__(self) -> str:
        spells = "; ".join(f"{n}: {_spells([int(p) for p in self.binding.index[self.binding[n]]]) or 'never'}" for n in self.constraints)
        return f"<OccBinPath: {len(self.frame)} periods; binds {spells}>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


def _spells(periods: list[int]) -> str:
    if not periods:
        return ""
    out, start, prev = [], periods[0], periods[0]
    for p in periods[1:] + [None]:  # type: ignore[list-item]
        if p is not None and p == prev + 1:
            prev = p
            continue
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
        if p is not None:
            start = prev = p
    return ", ".join(out)


def _source(obj: Any, shocks: Sequence[str] | None) -> tuple[Any, tuple[str, ...], dict[str, float]]:
    from .model import Model

    if isinstance(obj, Model):
        names = tuple(shocks or obj.compile().shocks)
        if not names:
            raise ModelError("name the model's shocks (shocks=[...]) or give them Role.SHOCK")
        return obj, names, {}
    model = getattr(obj, "model", None)
    if not isinstance(model, Model):
        raise TypeError(f"occbin solves a template, a Dynare or IRIS model, or a Model, not {type(obj).__name__}")
    names = tuple(shocks or getattr(obj, "shocks", None) or getattr(obj, "exogenous", ()))
    guess = dict(getattr(obj, "guess", None) or {})
    return model, names, guess


def _shock_matrix(shocks: Mapping[str, Any], names: tuple[str, ...], periods: int) -> np.ndarray:
    E = np.zeros((periods, len(names)))
    for name, path in shocks.items():
        if name not in names:
            raise ModelError(f"{name} is not a shock of the model; its shocks are {', '.join(names)}")
        items = path.items() if isinstance(path, Mapping) else enumerate(path, start=1)
        for t, value in items:
            if not 1 <= int(t) <= periods:
                raise ModelError(f"shock {name} in period {t}: periods run from 1 to {periods}")
            E[int(t) - 1, names.index(name)] = float(value)
    return E


class _Piecewise:
    """A model's regimes, linearised at the reference steady state, and the guess-and-check solution of its
    piecewise-linear dynamics: what ``occbin`` simulates with and what the OccBin filter (``occbin_filter``) filters
    with. ``horizon`` is how far ahead each solution looks; the reference regime holds after it.

    ``regimes``: known-date regimes (``pymetrix.models.regimes``) over the periods from ``first`` (a period of
    frequency ``freq``, or a number with ``freq=None``) for ``length`` periods. The regime in force before ``first`` is
    the reference (its steady state the point of linearisation); each period's matrices are then those of its known
    regime and of the constraints' guessed regime, each linearised there with the constant it leaves; after the
    horizon the constraints are slack, and the known regimes follow their own time-varying rules until the last, which
    holds for good with its own stable solution (``regimes.RegimePlan``'s rules). The methods take ``offset``, the
    period (0 the first) the solution starts in."""

    def __init__(self, model: Any, names: tuple[str, ...], guess: Mapping[str, float] | None = None, horizon: int = 100,
                 max_iter: int = 100, regimes: Any = None, first: Any = 1, freq: Any = None, length: int = 1) -> None:
        self.known = None  # the known regimes' schedule (the first one news gives), None without regimes that change anything
        self._views: list[tuple[int, Any]] = [(0, None)]  # (the period news arrives in, the schedule believed from then)
        if regimes:
            from .regimes import regime_schedules

            model, self._views = regime_schedules(model, regimes, first, freq, int(length), "OccBin")
            self.known = next((v for _, v in self._views if v is not None), None)
        self.model, self.names, self.H, self.max_iter = model, tuple(names), int(horizon), int(max_iter)
        self.tie = False  # the filter counts a rule exactly at its bound as binding (see _binding)
        self.ss = model.steady_state(guess=dict(guess or {}) or None)
        self.steady = dict(self.ss.values)
        own = _constraints(model, self.steady, empty=self.known is not None)
        self.constraints: list[_Constraint] = own
        calls: list[_Constraint | None] = list(own)
        if self.known is not None:  # the constraints of all the regimes together; each regime's, None where it has none
            self._ref, self._models = self.known.reference, self.known.models
            labels = [c.label for c in own]
            self._calls = {key: _regime_constraints(m, own, labels, self.steady)
                           for key, m in self._models.items() if key != self._ref}
            self._calls[self._ref] = calls
            for found in self._calls.values():
                found += [None] * (len(labels) - len(found))
            if not labels:
                raise ModelError("neither the model nor its regimes have a max or min: OccBin needs an occasionally binding "
                                 "constraint, e.g. i ~ max(i_rule, floor)")
            self.constraints = [next(c[j] for c in self._calls.values() if c[j] is not None) for j in range(len(labels))]  # type: ignore[misc]
        self.reference = tuple(c.reference for c in self.constraints)
        self.regimes = {choice: _regime(model, self.ss, self.names, calls, choice)
                        for choice in product((0, 1), repeat=len(self.constraints))}
        base = self.regimes[self.reference]
        if base.c is not None and np.max(np.abs(base.c)) > 1e-8:
            raise ModelError("the model does not hold at its steady state in the reference regime")
        self.base = base
        self.solution = base.solve()
        self.variables = base.variables
        self.n = len(self.variables)
        self.index = {v: i for i, v in enumerate(self.variables)}
        if self.known is not None:
            self._linear = {(self._ref, choice): lin for choice, lin in self.regimes.items()}
            self._tails: dict[int, tuple[list[np.ndarray], list[np.ndarray]]] = {}

    def _view(self, offset: int) -> int:
        """The news in force at period ``offset`` (0 the first): the last arrival by then."""
        return max(i for i, (a, _) in enumerate(self._views) if a <= max(offset, 0))

    def _schedule(self, offset: int) -> Any:
        return self._views[self._view(offset)][1]

    def _key(self, s: int, schedule: Any) -> Any:
        """Period ``s``'s known regime, as ``schedule`` has it."""
        return self._ref if schedule is None else schedule.key(s)

    def _name(self, schedule: Any) -> str:
        return schedule.names[-1] if schedule is not None else "the reference regime"

    def _lin(self, s: int, choice: tuple[int, ...], schedule: Any = None) -> Any:
        """Period ``s``'s linearised model under the constraints' regime ``choice`` (by the branches of ``constraints``;
        a regime's own constraint binds where that one does), its known regime as ``schedule`` has it."""
        if self.known is None:
            return self.regimes[choice]
        key = self._key(s, schedule)
        lin = self._linear.get((key, choice))
        if lin is None:
            calls = self._calls[key]
            local = tuple(0 if c is None else c.reference if k == r else 1 - c.reference for c, k, r in zip(calls, choice, self.reference))
            lin = self._linear[(key, choice)] = _regime(self._models[key], self.ss, self.names, calls, local, tuple(self.variables))
        return lin

    def _known_rules(self, schedule: Any) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """The rules with the constraints slack, period by period over the known regimes of ``schedule``: backwards
        from the last regime, which holds for good with its own stable solution and the constant that takes it to its
        steady state."""
        W = len(schedule.keys)
        last = self._lin(W - 1, self.reference, schedule)
        if schedule.key(W - 1) == self._ref:
            P, d = self.solution.P, np.zeros(self.n)
        else:
            try:
                P = last.solve().P
            except ModelError as exc:
                raise ModelError(f"the last regime ({self._name(schedule)}), which holds for good, has no unique stable solution: {exc}") from None
            try:
                d = -np.linalg.solve(last.A0 @ P + last.A1 + last.A0, last.c)
            except np.linalg.LinAlgError:
                raise ModelError(f"the last regime ({self._name(schedule)}) has no steady state to go to") from None
        Ps, ks = [P] * W, [d] * W
        for t in range(W - 2, -1, -1):
            lin = self._lin(t, self.reference, schedule)
            M = lin.A0 @ Ps[t + 1] + lin.A1
            try:
                Ps[t] = -np.linalg.solve(M, lin.A2)
                ks[t] = -np.linalg.solve(M, lin.A0 @ ks[t + 1] + lin.c)
            except np.linalg.LinAlgError:
                raise ModelError(f"the regime in period {t + 1} has no solution (A0 P + A1 singular)") from None
        return Ps, ks

    def _after(self, s: int, schedule: Any = None) -> tuple[np.ndarray, np.ndarray]:
        """The rule ``x = P x(-1) + d`` of period ``s`` with the constraints slack from then on (the known regimes as
        ``schedule`` has them)."""
        if schedule is None:
            return self.solution.P, np.zeros(self.n)
        view = id(schedule)
        if view not in self._tails:
            self._tails[view] = self._known_rules(schedule)
        Ps, ks = self._tails[view]
        s = min(s, len(Ps) - 1)
        return Ps[s], ks[s]

    def _choice(self, binds_t: np.ndarray) -> tuple[int, ...]:
        return tuple(self.reference[j] if not binds_t[j] else 1 - self.reference[j] for j in range(len(self.constraints)))

    def rules(self, binds: np.ndarray, shock_path: np.ndarray, offset: int = 0) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """The time-varying decision rules x(t) = P(t) x(t-1) + D(t), t = 1..H, given which constraints bind in each
        period and the (known) shock path; and Q(1), the response in period 1 to that period's shock. The known
        regimes are those believed in period ``offset`` (surprises that have started by then)."""
        H, n = self.H, self.n
        schedule = self._schedule(offset)
        if schedule is None and not binds.any() and not np.any(shock_path[1:]):  # the reference regime throughout: its own solution
            sol = self.solution
            first = sol.Q @ shock_path[0] if len(shock_path) else np.zeros(n)
            return [sol.P] * H, [first] + [np.zeros(n)] * (H - 1), sol.Q
        P_next, D_next = self._after(offset + H, schedule)
        Ps, Ds = [np.empty(0)] * H, [np.empty(0)] * H
        Q1 = np.empty(0)
        for t in range(H - 1, -1, -1):
            lin = self._lin(offset + t, self._choice(binds[t]), schedule)
            M = lin.A0 @ P_next + lin.A1
            e = shock_path[t] if t < len(shock_path) else np.zeros(len(self.names))
            const = lin.A0 @ D_next + (lin.c if lin.c is not None else 0.0) + lin.B @ e
            try:
                Ps[t] = -np.linalg.solve(M, lin.A2)
                Ds[t] = -np.linalg.solve(M, const)
                if t == 0:
                    Q1 = -np.linalg.solve(M, lin.B)
            except np.linalg.LinAlgError:
                raise ModelError(f"the regime in period {offset + t + 1} has no solution (A0 P + A1 singular)") from None
            P_next, D_next = Ps[t], Ds[t]
        return Ps, Ds, Q1

    def path(self, x_init: np.ndarray, shock_path: np.ndarray, binds: np.ndarray, offset: int = 0) -> np.ndarray:
        """Deviations for periods 1..H given which regime holds in each (and the reference one after H)."""
        Ps, Ds, _ = self.rules(binds, shock_path, offset)
        X = np.empty((self.H + 1, self.n))
        prev = x_init
        for t in range(self.H):
            prev = X[t] = Ps[t] @ prev + Ds[t]
        P, d = self._after(offset + self.H, self._schedule(offset))
        X[self.H] = P @ prev + d
        return X

    def binding(self, full: np.ndarray, E: np.ndarray, tie: bool = False, offset: int = 0, view: int | None = None) -> np.ndarray:
        """``_binding`` on a path whose first period is ``offset`` (``full``: the state before it, then the path): with
        known-date regimes, each period's constraints are its regime's (their parameters; a constraint its regime
        leaves out never binds), the regime in force then, or the one believed under the news of ``view``."""
        if self.known is None:
            return _binding(self.constraints, full, self.steady, self.index, self.names, E, tie)
        T = len(full) - 1
        if view is None:
            keys = [self._key(offset + t, self._schedule(offset + t)) for t in range(T)]
        else:
            keys = [self._key(offset + t, self._views[view][1]) for t in range(T)]
        out = np.zeros((T, len(self.constraints)), dtype=bool)
        for key in dict.fromkeys(keys):
            rows = np.array([k == key for k in keys])
            out[rows] = _binding(self._calls[key], full, self.steady, self.index, self.names, E, tie)[rows]
        return out

    def branches(self, full: np.ndarray, E: np.ndarray, t: int) -> np.ndarray:
        """``_branches`` in period ``t + 1`` of a path from the first period, with that period's constraints."""
        calls = self.constraints if self.known is None else self._calls[self._key(t, self._schedule(t))]
        return _branches(calls, full, self.steady, self.index, self.names, E, t)

    def check(self, X: np.ndarray, x_init: np.ndarray, shock_path: np.ndarray, offset: int = 0) -> np.ndarray:
        """Which constraints bind in periods 1..H of a path: the branch each max or min picks, in levels."""
        padded = np.zeros((self.H + 1, len(self.names)))
        padded[:len(shock_path)] = shock_path[:self.H + 1]
        return self.binding(np.vstack([x_init, X]), padded, self.tie, offset, view=self._view(offset))[:self.H]

    @property
    def arrivals(self) -> tuple[int, ...]:
        """The periods (0 the first) in which surprise regimes start: news that makes the path be solved again."""
        return tuple(a for a, _ in self._views[1:])

    def solve_from(self, x_init: np.ndarray, shock_path: np.ndarray, binds: np.ndarray | None = None,
                   offset: int = 0) -> tuple[np.ndarray, int, np.ndarray]:
        """The path from ``x_init`` after ``shock_path`` (known in advance), the iterations it took and the binding
        periods; ``binds`` is the first guess (default: never). ``offset``: the period the path starts in (with
        known-date regimes)."""
        binds = np.zeros((self.H, len(self.constraints)), dtype=bool) if binds is None else binds
        seen = set()
        for iteration in range(1, self.max_iter + 1):
            X = self.path(x_init, shock_path, binds, offset)
            new = self.check(X, x_init, shock_path, offset)
            if np.array_equal(new, binds):
                if new[-1].any():
                    raise ModelError(f"the constraint still binds at the end of the horizon ({self.H} periods): give a longer horizon")
                return X, iteration, binds
            key = new.tobytes()
            if key in seen:
                raise ModelError("the guess of binding periods cycles without converging: OccBin finds no solution here")
            seen.add(key)
            binds = new
        raise ModelError(f"no convergence of the binding periods in {self.max_iter} iterations")


def _simulate(pw: _Piecewise, x0: np.ndarray, E: np.ndarray, anticipated: bool) -> tuple[np.ndarray, int]:
    """The piecewise-linear path (deviations, periods x variables) from ``x0`` after the shocks ``E`` (periods x
    shocks): surprises period by period, or the whole path known from the first period; and the most iterations a
    solution took."""
    periods = len(E)
    if anticipated:  # solved again from each surprise regime's start, the shocks still known from the first period
        starts = [0, *(a for a in pw.arrivals if a < periods), periods]
        parts, iterations, state = [], 0, x0
        for a, b in zip(starts[:-1], starts[1:]):
            X, k, _ = pw.solve_from(state, E[a:], offset=a)
            iterations = max(iterations, k)
            parts.append(X[:b - a])
            state = X[b - a - 1]
        return np.vstack(parts)[:periods], iterations
    rows, iterations, state = [], 0, x0
    for t in range(periods):
        X, k, _ = pw.solve_from(state, E[t:t + 1], offset=t)
        iterations = max(iterations, k)
        state = X[0]
        rows.append(state)
    return np.array(rows).reshape(periods, len(x0)), iterations


@api("pymetrix:occbin")
def occbin(obj: Any, shocks: Mapping[str, Any], periods: int = 40, anticipated: bool = False, horizon: int | None = None,
           shock_names: Sequence[str] | None = None, initial: Mapping[str, float] | None = None, max_iter: int = 100,
           regimes: Any = None) -> OccBinPath:
    """The piecewise-linear path of ``obj`` (a template, a Dynare or IRIS model, or a Model) after ``shocks``
    (``{"ey": {1: -3.0}}``: shock ey of -3 in period 1; or a list of values from period 1): see the module docstring.
    ``initial`` gives starting levels (default: the steady state); ``horizon`` how far ahead each solution looks.
    ``regimes``: parameters or equations that change at known periods (``Regime(5, {"phi_pi": 2.0})``, or text such as
    ``"5..8: rho=1"``; periods numbered from 1), known in advance (or surprises, ``expected=False``: ``"5 surprise:
    rho=1"``), combined with the constraints: each period's matrices are those of its known regime and of the
    constraints' regime (see ``_Piecewise``)."""
    model, names, guess = _source(obj, shock_names)
    pw = _Piecewise(model, names, guess, int(horizon or max(4 * periods, 100)), max_iter, regimes=regimes, first=1, length=periods)
    steady, constraints, index, n = pw.steady, pw.constraints, pw.index, pw.n
    ref_solution = pw.solution
    E = _shock_matrix(shocks, names, periods)
    x0 = np.zeros(n)
    for name, value in (initial or {}).items():
        if name not in index:
            raise ModelError(f"{name} is not a variable of the model")
        x0[index[name]] = float(value) - steady[name]

    deviations, iterations = _simulate(pw, x0, E, anticipated)
    none = np.zeros((pw.H, len(constraints)), dtype=bool)
    if anticipated:  # the counterfactual has the same information: the whole shock path (and the known regimes) known from the first period
        starts = [0, *(a for a in pw.arrivals if a < periods), periods]  # and surprise regimes when they start
        parts, prev = [], x0
        for a, b in zip(starts[:-1], starts[1:]):
            parts.append(pw.path(prev, E[a:], none, offset=a)[:b - a])
            prev = parts[-1][-1]
        linear_rows = np.vstack(parts)[:periods]
    else:
        linear_rows, prev = [], x0
        for t in range(periods):
            if pw.known is None:
                prev = ref_solution.P @ prev + ref_solution.Q @ E[t]
            else:  # the known regimes' rules, the constraints ignored
                Ps, Ds, _ = pw.rules(none, E[t:t + 1], offset=t)
                prev = Ps[0] @ prev + Ds[0]
            linear_rows.append(prev)
    observed = list(pw.base.observed)
    idx = pd.RangeIndex(1, periods + 1, name="period")
    keep = [index[v] for v in observed]
    dev = pd.DataFrame(deviations[:, keep], index=idx, columns=observed)
    levels = dev + pd.Series({v: steady[v] for v in observed})
    linear = pd.DataFrame(np.array(linear_rows)[:, keep], index=idx, columns=observed) + pd.Series({v: steady[v] for v in observed})
    binding = pw.binding(np.vstack([x0, deviations]), E)
    return OccBinPath(levels, dev, pd.DataFrame(binding, index=idx, columns=[c.label for c in constraints]), linear,
                      pd.DataFrame(E, index=idx, columns=list(names)), tuple(c.label for c in constraints), iterations, anticipated,
                      steady)


def _branches(constraints: Sequence[_Constraint | None], full: np.ndarray, steady: Mapping[str, float], index: Mapping[str, int],
              names: tuple[str, ...], E: np.ndarray, t: int) -> np.ndarray:
    """The gap between each constraint's two branches (first less second), in levels, in period ``t + 1`` of a path
    (``full``: the initial state, then the path, in deviations; ``E``: the shocks, period by period)."""
    from ..expr.evaluate import evaluate

    T = len(full) - 1

    def resolve(name: str, lag: int) -> float:
        if name in index:
            return steady[name] + full[max(0, min(t + 1 + lag, T))][index[name]]
        if name in names:
            return (E[t][names.index(name)] if lag == 0 and t < len(E) else 0.0) + steady.get(name, 0.0)
        return steady[name]

    return np.array([0.0 if c is None else float(evaluate(c.call.args[0], resolve)) - float(evaluate(c.call.args[1], resolve))
                     for c in constraints])


def _binding(constraints: Sequence[_Constraint | None], full: np.ndarray, steady: Mapping[str, float], index: Mapping[str, int],
             names: tuple[str, ...], E: np.ndarray, tie: bool = False) -> np.ndarray:
    """Whether each constraint binds in each period of a path (``full``: the initial state, then the path, in
    deviations): whether its max or min, evaluated in levels, picks the other branch than at the steady state. With
    ``tie``, branches equal to rounding count as binding: a rule exactly at its floor (the filter's reading of a rate
    observed at the floor)."""
    from ..expr.evaluate import evaluate

    T = len(full) - 1
    out = np.zeros((T, len(constraints)), dtype=bool)
    periods = np.arange(1, T + 1)
    shocks = np.zeros((T, len(names)))
    shocks[:min(T, len(E))] = E[:T]

    def along(name: str, lag: int) -> Any:  # a name's values in periods 1..T at once
        if name in index:
            return steady[name] + full[np.clip(periods + lag, 0, T), index[name]]
        if name in names:
            return (shocks[:, names.index(name)] if lag == 0 else np.zeros(T)) + steady.get(name, 0.0)
        return np.full(T, steady[name])

    try:
        for j, c in enumerate(constraints):
            if c is None:  # a constraint the period's regime leaves out: slack
                continue
            a, b = (np.broadcast_to(np.asarray(evaluate(arg, along), dtype=float), (T,)) for arg in c.call.args)
            picked = np.where(a >= b, 0, 1) if c.call.fn == "max" else np.where(a <= b, 0, 1)
            out[:, j] = (picked != c.reference) | (tie & (np.abs(a - b) <= 1e-9 * np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))))
        return out
    except Exception:  # noqa: BLE001 - an expression that cannot take whole paths: period by period, as below
        out[:] = False
    for t in range(1, T + 1):
        def resolve(name: str, lag: int, t: int = t) -> float:
            if name in index:
                return steady[name] + full[max(0, min(t + lag, T))][index[name]]
            if name in names:
                return (E[t - 1][names.index(name)] if lag == 0 and t - 1 < len(E) else 0.0) + steady.get(name, 0.0)
            return steady[name]
        for j, c in enumerate(constraints):
            if c is None:
                continue
            a, b = (float(evaluate(arg, resolve)) for arg in c.call.args)
            picked = (0 if a >= b else 1) if c.call.fn == "max" else (0 if a <= b else 1)
            out[t - 1, j] = picked != c.reference or (tie and abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b)))
    return out
