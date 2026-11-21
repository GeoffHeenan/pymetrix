"""A model's equations compiled once, for the solvers: residuals, derivatives, and their steady-state forms.

The steady state, linearisation and the stacked perfect-foresight solver evaluate the same residuals and
derivatives many times. ``functions(model)`` compiles them to NumPy (see ``pymetrix.expr.compile``), from the
model's symbolic residuals, with parameters passed as values: the compiled code is cached by the equations' form,
so a model rebuilt with new parameter values (each step of an estimation or a calibration) reuses it.
Derivatives are taken symbolically once per form. The tree interpreter remains the reference: set ``ENABLED =
False`` to use it, and the tests check that both give the same results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import Any

import numpy as np

from ..expr import Name
from ..expr.compile import Compiled, compilable, compile_exprs
from ..expr.diff import diff

ENABLED = True  # False: every solver uses the tree interpreter (the reference)

_ZERO_TEXT = "0"


def _static(expr: Any) -> Any:
    return expr.map(lambda n: Name(n.id, 0) if isinstance(n, Name) and n.lag else None)


@dataclass(eq=False)
class ModelFunctions:
    """Compiled residuals of some of a model's equations (in order), their derivatives and steady-state forms."""

    residual_exprs: tuple[Any, ...]  # symbolic: parameters are Coef nodes
    endogenous: tuple[str, ...]
    coefs: tuple[str, ...]
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @cached_property
    def residuals(self) -> Compiled:
        return compile_exprs(self.residual_exprs, self.coefs)

    @cached_property
    def pairs(self) -> tuple[tuple[int, str, int], ...]:
        """(equation, variable, lag) of each non-zero derivative."""
        return tuple(p for p, _ in self._derivatives)

    @cached_property
    def derivatives(self) -> Compiled:
        return compile_exprs(tuple(d for _, d in self._derivatives), self.coefs)

    @cached_property
    def _derivatives(self) -> tuple[tuple[tuple[int, str, int], Any], ...]:
        return _derivatives(self.residual_exprs)

    @cached_property
    def static(self) -> Compiled:
        return compile_exprs(tuple(_static(r) for r in self.residual_exprs), self.coefs)

    @cached_property
    def static_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(p for p, _ in self._static_jacobian)

    @cached_property
    def static_jacobian(self) -> Compiled:
        return compile_exprs(tuple(d for _, d in self._static_jacobian), self.coefs)

    @cached_property
    def _static_jacobian(self) -> tuple[tuple[tuple[int, int], Any], ...]:
        return _static_jacobian(tuple(_static(r) for r in self.residual_exprs), self.endogenous)


@lru_cache(maxsize=64)
def _derivatives(exprs: tuple[Any, ...]) -> tuple[tuple[tuple[int, str, int], Any], ...]:
    out = []
    for i, r in enumerate(exprs):
        for ref in sorted(r.refs(), key=lambda n: (n.id, n.lag)):
            d = diff(r, Name(ref.id, ref.lag))
            if str(d) != _ZERO_TEXT:
                out.append(((i, ref.id, ref.lag), d))
    return tuple(out)


@lru_cache(maxsize=64)
def _static_jacobian(exprs: tuple[Any, ...], endogenous: tuple[str, ...]) -> tuple[tuple[tuple[int, int], Any], ...]:
    out = []
    for i, r in enumerate(exprs):
        names = {n.id for n in r.refs()}
        for j, u in enumerate(endogenous):
            if u in names:
                d = diff(r, Name(u, 0))
                if str(d) != _ZERO_TEXT:
                    out.append(((i, j), d))
    return tuple(out)


@lru_cache(maxsize=64)
def _functions(exprs: tuple[Any, ...], endogenous: tuple[str, ...]) -> ModelFunctions:
    coefs = tuple(sorted(set().union(*(e.params() for e in exprs)))) if exprs else ()
    return ModelFunctions(exprs, endogenous, coefs)


def functions(model: Any, equations: Any = None) -> ModelFunctions | None:
    """The compiled functions of ``model`` (or of some of its equations, a list of its ``model_equations``), or None
    when compiling is off or an equation cannot be compiled."""
    if not ENABLED:
        return None
    system = model._system
    chosen = tuple(range(len(system.equations))) if equations is None else tuple(model.model_equations.index(e) for e in equations)
    shared = model.__dict__.get("_shared")  # models at other parameter values share their structure (with_parameters)
    if shared is not None and ("functions", chosen) in shared:
        return shared[("functions", chosen)]
    exprs = tuple(system.equations[i].residual for i in chosen)
    found = _functions(exprs, tuple(model.endogenous)) if all(compilable(e) for e in exprs) else None
    if shared is not None:
        shared[("functions", chosen)] = found
    return found


def parameter_values(model: Any, fns: ModelFunctions) -> np.ndarray:
    values = model._system.parameters.values()
    return np.array([float(values[c]) for c in fns.coefs])
