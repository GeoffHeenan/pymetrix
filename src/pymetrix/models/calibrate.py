"""Calibration to steady-state targets: parameters chosen so that the steady state has given values.

    cal = model.calibrate(targets={"k/y": 10, "c/y": 0.75}, free=["beta", "delta"])
    cal.values                      # the parameters found: {"beta": ..., "delta": ...}
    cal.model                       # the model with them (calibrated), ready to solve
    cal.steady                      # its steady state, which hits the targets
    cal.summary()                   # each target, the value reached, and the parameters

A target is any expression of the model's variables in the steady state: a level (``"i": 3``), a ratio
(``"k/y": 10``), a spread. There must be as many targets as free parameters. With ``growth=True`` the steady state
is a balanced growth path (targets apply to levels in the reference period, e.g. ratios and rates that do not
grow). This is the first stage of calibrating a semi-structural model such as FINEX: the parameters that
determine the long run are set so the model's steady state matches the data's averages or the long-term views
of the economy; the parameters of the dynamics come after.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import optimize

from ..output import Table
from .solvers import ModelError

if TYPE_CHECKING:
    from .model import Model

__all__ = ["Calibration", "calibrate"]


@dataclass(frozen=True, eq=False)
class Calibration:
    """Parameters found so that the steady state hits targets."""

    model: Any  # the model with the parameters found
    steady: Any  # its steady state
    values: Mapping[str, float]  # the free parameters' values
    start: Mapping[str, float]  # their values before
    targets: Mapping[str, float]
    achieved: Mapping[str, float]
    evaluations: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> Table:
        rows = [[f"target: {t}", float(v), float(self.achieved[t]), ""] for t, v in self.targets.items()]
        rows += [[f"parameter: {p}", float(self.start[p]), float(v), "calibrated"] for p, v in self.values.items()]
        notes = [f"{len(self.targets)} targets met by {len(self.values)} parameters ({self.evaluations} steady states solved)",
                 *self.notes]
        return Table(["", "target / before", "reached / found", ""], rows, title="Calibration to the steady state", notes=notes)

    def __repr__(self) -> str:
        found = ", ".join(f"{p}={v:.6g}" for p, v in self.values.items())
        return f"<Calibration: {found}>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


def calibrate(model: Model, targets: Mapping[str, float], free: Sequence[str], growth: bool = False,
              exogenous: Mapping[str, float] | None = None, guess: Mapping[str, float] | None = None,
              fix: Mapping[str, float] | None = None, tol: float = 1e-10) -> Calibration:
    from ..expr import evaluate, parse_expression
    from .model import Model as _Model

    free = list(free)
    targets = {str(k): float(v) for k, v in targets.items()}
    if len(free) != len(targets):
        raise ModelError(f"{len(targets)} targets but {len(free)} free parameters: give one parameter for each target")
    params = model.parameters
    unknown = [p for p in free if p not in params.names]
    if unknown:
        raise ModelError(f"{', '.join(unknown)} are not parameters of the model; its parameters: {', '.join(params.names)}")
    not_calibrated = [p for p in free if params[p].status != "calibrated"]
    if not_calibrated:
        raise ModelError(f"{', '.join(not_calibrated)} are {params[not_calibrated[0]].status}, not calibrated; only calibrated "
                         "parameters are chosen to hit targets")
    exprs = {}
    names = set(model.endogenous) | set(model.exogenous)
    for t in targets:
        e = parse_expression(t)
        missing = sorted({n.id for n in e.refs()} - names)
        if missing:
            raise ModelError(f"target {t!r} uses {', '.join(missing)}, which are not variables of the model")
        exprs[t] = e

    start = {p: float(params[p].value) for p in free}
    state = {"guess": dict(guess or {}), "count": 0}

    def steady_at(theta: np.ndarray) -> Any:
        trial = _Model(model.equations, params=params.with_values(dict(zip(free, map(float, theta)))), roles=model.roles)
        ss = trial.steady_state(exogenous, state["guess"] or None, growth=growth, fix=fix)
        state["guess"] = {k: v for k, v in ss.values.items() if k in trial.endogenous}  # start the next one from here
        state["count"] += 1
        return trial, ss

    def gaps(theta: np.ndarray) -> np.ndarray:
        try:
            _, ss = steady_at(theta)
        except (ModelError, ValueError, np.linalg.LinAlgError, OverflowError, ZeroDivisionError):
            return np.full(len(targets), 1e6)
        return np.array([float(evaluate(e, lambda n, lag, s=ss: s.values[n])) - v for (t, e), v in zip(exprs.items(), targets.values())])

    x0 = np.array([start[p] for p in free])
    if np.max(np.abs(gaps(x0))) >= 1e6:
        raise ModelError("the model has no steady state at the starting parameter values; check the calibration and guess=")
    found = optimize.root(gaps, x0, method="hybr", options={"xtol": 1e-13})
    if np.max(np.abs(gaps(found.x))) > max(tol, 1e-8):
        found = optimize.root(gaps, found.x, method="lm", options={"xtol": 1e-14, "ftol": 1e-14})
    worst = np.max(np.abs(gaps(found.x)))
    if worst > max(tol, 1e-8):
        raise ModelError(f"the targets could not all be met (off by {worst:.3g}); check that each free parameter moves the steady "
                         "state, and that the targets are consistent")
    calibrated, ss = steady_at(found.x)
    achieved = {t: float(evaluate(e, lambda n, lag: ss.values[n])) for t, e in exprs.items()}
    notes = []
    for p, v in zip(free, found.x):
        low, high = calibrated.parameters[p].bounds
        if not (low <= v <= high):
            notes.append(f"{p} = {v:.6g} is outside its bounds [{low:g}, {high:g}]")
    return Calibration(calibrated, ss, dict(zip(free, map(float, found.x))), start, targets, achieved, state["count"], tuple(notes))
