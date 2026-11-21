"""Period-by-period solvers for backward-looking models.

A solver receives a ``PeriodSystem`` (residuals and Jacobian of the model's
equations in one period, as functions of the unknowns) and returns the
solution and the number of iterations used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, Protocol

import numpy as np

from ..errors import PyMetrixError
from .._planned import planned
from ..forms import Widget
from ..plugins import register


class ModelError(PyMetrixError, ValueError):
    """A model could not be built or solved."""


class PeriodSystem(Protocol):
    x0: np.ndarray
    equation_names: list[str]

    def residuals(self, x: np.ndarray) -> np.ndarray: ...
    def jacobian(self, x: np.ndarray) -> np.ndarray: ...
    def own(self, i: int, x: np.ndarray) -> tuple[float, float, int]: ...


def _stopped(message: str, iterations: int) -> ModelError:
    """A period that could not be solved, with the iterations taken (``Solution.convergence`` of the error)."""
    exc = ModelError(message)
    exc.iterations = iterations  # type: ignore[attr-defined]
    return exc


def _worst(system: PeriodSystem, F: np.ndarray) -> str:
    i = int(np.nanargmax(np.abs(np.where(np.isfinite(F), F, np.inf))))
    return f"equation {system.equation_names[i]!r} (residual {F[i]:.3g})"


def _finite_jacobian(system: PeriodSystem, x: np.ndarray, F: np.ndarray) -> np.ndarray:
    """The analytic Jacobian, with any entry that does not evaluate (an overflowing ``exp``: the derivative of a
    saturated logistic switch such as ``1/(1+exp(25*(u - k)))`` far from its threshold is inf/inf) replaced by a
    forward difference, where the function itself is finite and flat."""
    J = system.jacobian(x)
    bad = ~np.isfinite(J)
    if bad.any():
        for j in np.flatnonzero(bad.any(axis=0)):
            h = 1e-7 * max(1.0, abs(x[j]))
            shifted = x.copy()
            shifted[j] += h
            column = (system.residuals(shifted) - F) / h
            J[bad[:, j], j] = column[bad[:, j]]
        system.residuals(x)  # leave the system at x
    return J


Convergence = Literal["residual", "step", "both"]


def _note(system: PeriodSystem, **values: Any) -> None:
    """Tell the system how its period was solved (``Solution.convergence``'s optional columns), where it listens."""
    report = getattr(system, "report", None)
    if report is not None:
        report.update(values)


def _count_jacobian(system: PeriodSystem) -> None:
    report = getattr(system, "report", None)
    if report is not None:
        report["jacobians"] = report.get("jacobians", 0) + 1


def _step_size(new: np.ndarray, old: np.ndarray) -> float:
    """The scaled step: the largest ``|x_new - x_old| / (1 + |x_old|)``."""
    return float(np.max(np.abs(new - old) / (1.0 + np.abs(old)), initial=0.0))


def _stops(test: str, residual: float, step: float | None, tol: float) -> bool:
    """Whether a period has converged under ``test``: the largest residual, the scaled step (none before the first),
    or both."""
    small_step = step is not None and step < tol
    if test == "residual":
        return residual < tol
    if test == "step":
        return small_step
    return residual < tol and small_step


def _check_convergence(test: str) -> None:
    if test not in ("residual", "step", "both"):
        raise ModelError(f"convergence is 'residual', 'step' or 'both', not {test!r}")


@dataclass(frozen=True)
class Newton:
    """Newton's method with the analytic Jacobian; solves all equations jointly.

    In each period every step solves the model's equations linearised at the current values, so simultaneous blocks
    are solved together and convergence near the solution is fast. It also handles scenarios that swap a target for a
    free variable, which Gauss-Seidel cannot. It can fail when the starting values are far from the solution, or when
    the Jacobian is singular because some equation does not determine its variable; the error names the equation
    with the largest residual. The tolerance applies to the residuals as written, so it depends on the equations'
    units: ``convergence="step"`` stops on the scaled step instead, the largest ``|x_new - x_old| / (1 + |x_old|)``,
    which does not (``"both"`` requires both tests). ``line_search=True`` backtracks (Armijo) when the full step does
    not reduce the residuals enough, for starts far from the solution. ``refresh="period"`` is a modified Newton: the
    Jacobian is evaluated and factorised once and kept while each step cuts the largest residual at least to a
    quarter, and evaluated again when convergence slows; for models whose Jacobian is costly to evaluate.
    """

    label: ClassVar[str] = "Newton"
    topics: ClassVar[tuple[str, ...]] = ('model-solution',)

    tol: Annotated[float, Widget("number", min=1e-15, max=1e-2, advanced=True, help="largest equation residual accepted")] = 1e-10
    max_iter: Annotated[int, Widget("number", min=1, max=10000, advanced=True, help="iterations allowed in each period")] = 50
    convergence: Annotated[Convergence, Widget("choice", advanced=True,
                                               help="stop on the largest residual (in the equations' units), on the scaled "
                                                    "step |x_new - x_old| / (1 + |x_old|), or on both")] = "residual"
    line_search: Annotated[bool, Widget("checkbox", advanced=True,
                                        help="backtrack (Armijo) when the full step does not reduce the residuals")] = False
    refresh: Annotated[Literal["iteration", "period"], Widget("choice", advanced=True,
                                                              help="evaluate the Jacobian every iteration, or once a period "
                                                                   "and again only when convergence slows (modified Newton)")] = "iteration"

    def solve_period(self, system: PeriodSystem) -> tuple[np.ndarray, int]:
        _check_convergence(self.convergence)
        if self.refresh not in ("iteration", "period"):
            raise ModelError(f"refresh is 'iteration' or 'period', not {self.refresh!r}")
        x = system.x0.copy()
        F: np.ndarray | None = None  # the residuals at x, when the line search has computed them there
        step: float | None = None  # the last scaled step
        kept: Any = None  # refresh="period": the factorised Jacobian kept
        previous: float | None = None  # the largest residual one iteration before
        for it in range(self.max_iter + 1):
            if F is None:
                F = system.residuals(x)
            if not np.all(np.isfinite(F)):
                raise _stopped(f"non-finite residual in {_worst(system, F)}", it)
            largest = float(np.max(np.abs(F), initial=0.0))
            if _stops(self.convergence, largest, step, self.tol):
                _note(system, test=self.convergence)
                return x, it
            if it == self.max_iter:
                break
            fresh = True
            try:
                if self.refresh == "iteration":
                    dx = np.linalg.solve(self._jacobian(system, x, F), -F)
                else:
                    if kept is None or (previous is not None and largest > _SLOW * previous):
                        kept = _factorised(self._jacobian(system, x, F))
                    else:
                        fresh = False
                    dx = _lu_solve(kept, -F)
                if self.line_search:
                    found = _armijo(system, x, dx, F)
                    if found is None and not fresh:  # a direction from an old Jacobian: try a new one
                        kept = _factorised(self._jacobian(system, x, F))
                        dx = _lu_solve(kept, -F)
                        found = _armijo(system, x, dx, F)
                    if found is None:
                        system.residuals(x)  # leave the system at x
                        raise _stopped(f"the line search found no step that reduces the residuals; worst is {_worst(system, F)}", it)
                    new, F = found
                else:
                    new, F = x + dx, None
            except np.linalg.LinAlgError:
                raise _stopped("the Jacobian is singular; check that each equation determines its variable", it) from None
            if self.convergence != "residual":
                step = _step_size(new, x)
            x, previous = new, largest
        raise _stopped(f"Newton did not converge in {self.max_iter} iterations; worst is {_worst(system, F)}", self.max_iter)

    @staticmethod
    def _jacobian(system: PeriodSystem, x: np.ndarray, F: np.ndarray) -> np.ndarray:
        _count_jacobian(system)
        return _finite_jacobian(system, x, F)


_SLOW = 0.25  # refresh="period": a step that leaves more than this share of the largest residual takes a new Jacobian
_ARMIJO = 1e-4  # the sufficient decrease the line search asks of 0.5 ||F||^2
_HALVINGS = 30  # the line search's shortest step: 2**-30 of Newton's


def _factorised(J: np.ndarray) -> Any:
    """The LU factors of ``J`` (``scipy.linalg.lu_factor``); LinAlgError when it is singular."""
    import warnings

    from scipy import linalg

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", linalg.LinAlgWarning)
        lu = linalg.lu_factor(J, check_finite=False)
    if not np.all(np.isfinite(lu[0])) or np.any(np.diag(lu[0]) == 0):
        raise np.linalg.LinAlgError("singular")
    return lu


def _lu_solve(lu: Any, b: np.ndarray) -> np.ndarray:
    from scipy import linalg

    return linalg.lu_solve(lu, b, check_finite=False)


def _armijo(system: PeriodSystem, x: np.ndarray, dx: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Armijo backtracking along Newton's direction ``dx``: the longest step ``lam`` of 1, 1/2, 1/4, ... with
    ``||F(x + lam dx)||^2 <= (1 - 2 c lam) ||F(x)||^2``, the sufficient decrease of ``0.5 ||F||^2`` (whose slope along
    the Newton direction is ``-||F||^2``) with ``c = 1e-4``, and the residuals there; None when no step down to
    ``2**-30`` gives it. Residuals that do not evaluate (the log of a negative level) count as no decrease."""
    base = float(F @ F)
    lam = 1.0
    for _ in range(_HALVINGS + 1):
        trial = x + lam * dx
        Ft = system.residuals(trial)
        if np.all(np.isfinite(Ft)) and float(Ft @ Ft) <= (1.0 - 2.0 * _ARMIJO * lam) * base:
            return trial, Ft
        lam *= 0.5
    return None


@dataclass(frozen=True)
class GaussSeidel:
    """Update each equation's own variable in turn. Cannot handle target/free swaps.

    Each sweep takes the equations in order and moves each one's own variable towards satisfying it (a Newton step
    in that variable alone), using the latest values of the others, until every residual is below the tolerance. It needs no full Jacobian and suits large models whose
    equations are mostly recursive, but it can converge slowly or not at all when the simultaneous feedback is
    strong; damping below 1 often helps, at the cost of more iterations, and Newton is the alternative. Every
    equation must depend on its own variable. ``convergence="step"`` stops when a sweep's scaled step, the largest
    ``|x_new - x_old| / (1 + |x_old|)``, is below the tolerance, whatever the equations' units (``"both"`` requires
    both tests).
    """

    label: ClassVar[str] = "Gauss-Seidel"
    topics: ClassVar[tuple[str, ...]] = ('model-solution',)

    tol: Annotated[float, Widget("number", min=1e-15, max=1e-2, advanced=True, help="largest equation residual accepted")] = 1e-10
    max_iter: Annotated[int, Widget("number", min=1, max=100000, advanced=True, help="iterations allowed in each period")] = 500
    damping: Annotated[float, Widget("number", min=0.05, max=1, step=0.05, help="share of each update applied (below 1 can help convergence)")] = 1.0
    convergence: Annotated[Convergence, Widget("choice", advanced=True,
                                               help="stop on the largest residual (in the equations' units), on the scaled "
                                                    "step |x_new - x_old| / (1 + |x_old|), or on both")] = "residual"

    def solve_period(self, system: PeriodSystem) -> tuple[np.ndarray, int]:
        _check_convergence(self.convergence)
        x = system.x0.copy()
        n = len(x)
        F = None
        for it in range(1, self.max_iter + 1):
            before = x.copy() if self.convergence != "residual" else None
            for i in range(n):
                f, d, j = system.own(i, x)
                if j < 0:
                    raise ModelError("Gauss-Seidel needs every equation to solve for its own variable; use Newton for target/free scenarios")
                if d == 0 or not np.isfinite(d):
                    raise ModelError(f"equation {system.equation_names[i]!r} does not depend on its own variable here")
                x[j] -= self.damping * f / d
            step = None if before is None else _step_size(x, before)
            if self.convergence == "step":  # no residuals needed
                if step is not None and step < self.tol:
                    _note(system, test="step")
                    return x, it
                continue
            F = system.residuals(x)
            if _stops(self.convergence, float(np.max(np.abs(F), initial=0.0)), step, self.tol):
                _note(system, test=self.convergence)
                return x, it
        if F is None or self.convergence == "step":
            F = system.residuals(x)
        raise _stopped(f"Gauss-Seidel did not converge in {self.max_iter} iterations; worst is {_worst(system, F)}", self.max_iter)


def describe(solver: Any) -> str:
    """A solver as code, its settings that differ from the defaults: ``GaussSeidel(damping=0.5, max_iter=400)``."""
    from ..calls import python_code

    try:
        return python_code(solver)
    except Exception:  # a plugin's solver that does not write itself as code
        return type(solver).__name__


@dataclass(frozen=True, init=False)
class Fallback:
    """Period-by-period solvers tried in order: a period the first cannot solve is solved again by the next, from the
    same starting values, and so on; periods the first solves are as it solves them.

    ``Fallback(GaussSeidel(), GaussSeidel(damping=0.5, max_iter=400), Newton(line_search=True))`` keeps Gauss-Seidel's
    speed where it converges and retries only the periods where it does not. ``Solution.convergence`` names the solver
    that solved each period (its ``solver`` column) and counts every attempt's iterations; when all fail, the error
    lists each attempt with its worst equation. With ``ordering="blocks"`` the solvers are tried block by block.
    """

    code_positional: ClassVar[int] = 1  # generated code gives the solvers by position: Fallback([GaussSeidel(), Newton()])
    label: ClassVar[str] = "Fallback"
    topics: ClassVar[tuple[str, ...]] = ('model-solution',)

    solvers: tuple[Any, ...]

    def __init__(self, *attempts: Any, solvers: Any = ()) -> None:
        given = tuple(attempts)
        if len(given) == 1 and isinstance(given[0], (list, tuple)):  # Fallback([a, b]), as code is written
            given = tuple(given[0])
        given = given + tuple(solvers)
        if not given:
            raise ModelError("Fallback needs at least one solver: Fallback(GaussSeidel(), Newton())")
        for s in given:
            if isinstance(s, (PerfectForesight, FirstOrder)) or not callable(getattr(s, "solve_period", None)):
                raise ModelError(f"Fallback tries period-by-period solvers (Newton, GaussSeidel, ...), not {describe(s)}")
        object.__setattr__(self, "solvers", given)

    def solve_period(self, system: PeriodSystem) -> tuple[np.ndarray, int]:
        start = system.x0.copy()
        failures: list[str] = []
        spent = 0
        for solver in self.solvers:
            system.x0 = start.copy()
            try:
                x, iterations = solver.solve_period(system)
            except ModelError as exc:
                spent += int(getattr(exc, "iterations", 0) or 0)
                message = str(exc)
                if "worst is" not in message:
                    residual, equation = _last_worst(system)
                    if equation:
                        message += f"; worst is equation {equation!r} (residual {residual:.3g})"
                failures.append(f"{describe(solver)}: {message}")
                continue
            system.x0 = start
            report = getattr(system, "report", None)
            if report is not None:
                report.setdefault("solver", describe(solver))  # a Fallback inside has named its own
            return x, spent + iterations
        system.x0 = start
        raise _stopped("every solver failed: " + "; ".join(failures), spent)


def uses_options(solver: Any) -> bool:
    """Whether ``solver`` uses any option that ``Solution.convergence`` reports in its own columns (a convergence test
    other than the residual's, a line search, a Jacobian kept for the period, or a Fallback)."""
    if isinstance(solver, Fallback):
        return True
    return (getattr(solver, "convergence", "residual") != "residual" or bool(getattr(solver, "line_search", False))
            or getattr(solver, "refresh", "iteration") != "iteration")


def _last_worst(system: PeriodSystem) -> tuple[float, str]:
    last = getattr(system, "last_residuals", None)
    if last is None:
        return float("nan"), ""
    residual, equation = last()
    return (float("nan") if residual is None else residual), equation or ""


@dataclass(frozen=True)
class PerfectForesight:
    """Models with expectations (leads such as x[+1]): every period solved at once (stacked-time Newton).

    Expectations are model-consistent: agents foresee the whole simulated path, including announced
    changes to exogenous variables. Beyond the last period, leads take their terminal values: the last
    simulated value (constant), the data, or the steady state. Simulate well past the periods of interest
    so the terminal condition matters little.
    """

    label: ClassVar[str] = "Perfect foresight"
    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models", "model-solution")

    terminal: Annotated[Literal["constant", "data", "steady_state", "growth"],
                        Widget("choice", help="leads beyond the last period: hold the last value, take the data, the steady state, "
                               "or the last value on the balanced growth path")] = "constant"
    tol: Annotated[float, Widget("number", min=1e-15, max=1e-2, advanced=True, help="largest equation residual accepted")] = 1e-10
    max_iter: Annotated[int, Widget("number", min=1, max=1000, advanced=True, help="Newton iterations allowed")] = 50

    def solve_period(self, system: PeriodSystem) -> tuple[np.ndarray, int]:
        raise ModelError("perfect foresight solves all periods together; use it through Model.solve")


@dataclass(frozen=True)
class FirstOrder:
    """Models with expectations, approximated to first order around the steady state (linearised and solved by QZ).

    Every exogenous variable is a shock. Shocks and exogenous paths are anticipated (known from the first simulated
    period) unless the scenario names them ``unanticipated``; targets can be held with free shocks (exogenize and
    endogenize). Exact for linear models; for nonlinear ones, accurate near the steady state.
    """

    label: ClassVar[str] = "First order (linearised)"
    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models", "model-solution")

    growth: Annotated[bool, Widget("checkbox", help="a trending model: linearise around its balanced growth path, not a constant steady state")] = False

    def solve_period(self, system: PeriodSystem) -> tuple[np.ndarray, int]:
        raise ModelError("a first-order solution solves all periods together; use it through Model.solve")


Solver = Newton | GaussSeidel | Fallback | PerfectForesight | FirstOrder

register("solvers", Newton)
register("solvers", GaussSeidel, name="gaussseidel")
register("solvers", PerfectForesight, name="perfect_foresight")
register("solvers", FirstOrder, name="first_order")
