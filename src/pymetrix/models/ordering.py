"""Block-recursive solution of a backward-looking model, period by period (``Model.solve(..., ordering="blocks")``).

Each period's equations are split into blocks: each equation is paired with the unknown it determines (its own
variable, or with holds and free variables a pairing found by bipartite matching), and the strongly connected
components of the graph "this equation needs that unknown in the same period" (Tarjan's algorithm) are the blocks, in
an order where every block needs only the blocks before it. A block of one equation that is linear in its own
variable (an identity, a fitted equation written for its variable) is recursive: it is solved by one evaluation of the
equation and its derivative, ``x -= f / f'``, exact for such an equation. Every other block (simultaneous, or one
equation nonlinear in its variable, or one solved for a free variable) is solved by the solver given, alone, with the
values of the blocks before it fixed: Newton, Gauss-Seidel, or a Fallback, applied block by block. A model whose
equations are mostly recursive is then solved with far fewer evaluations than iterating on all of them each period.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..expr import Name, Num, diff
from .solvers import ModelError, _stopped, describe

_ZERO = Num(0.0)


@dataclass(frozen=True)
class Step:
    """One block of a period's solution: positions of its equations and unknowns in the period's lists, and whether it
    is a recursive equation solved by one evaluation."""

    equations: tuple[int, ...]
    unknowns: tuple[int, ...]
    once: bool


def plan(eqs: Sequence[Any], unknowns: Sequence[str]) -> list[Step]:
    """The blocks of equations ``eqs`` (``ModelEquation``) in unknowns ``unknowns``, in solution order."""
    n = len(eqs)
    uidx = {u: j for j, u in enumerate(unknowns)}
    needs = [sorted({uidx[r.id] for r in e.residual.refs() if r.lag == 0 and r.id in uidx}) for e in eqs]
    by_eq, by_unknown = [-1] * n, [-1] * len(unknowns)
    for i, e in enumerate(eqs):  # each equation with its own variable, where that is an unknown
        j = uidx.get(e.dependent, -1)
        if j >= 0 and j in needs[i] and by_unknown[j] < 0:
            by_eq[i], by_unknown[j] = j, i
    for i in range(n):  # the rest (holds with free variables) by augmenting paths
        if by_eq[i] < 0 and not _augment(i, needs, by_eq, by_unknown):
            raise ModelError(f"equation {eqs[i].name!r} has no unknown of its own to determine in the block ordering: "
                             "the equations do not determine the unknowns (check the holds and free variables)")
    deps = [sorted({by_unknown[j] for j in needs[i]} - {i}) for i in range(n)]
    steps = []
    for block in _components(deps):
        if len(block) == 1:
            i = block[0]
            j = by_eq[i]
            steps.append(Step((i,), (j,), unknowns[j] == eqs[i].dependent and _linear_in_own(eqs[i])))
        else:
            steps.append(Step(tuple(block), tuple(sorted(by_eq[i] for i in block)), False))
    return steps


def _linear_in_own(eq: Any) -> bool:
    """Whether the equation is linear in its own variable (and depends on it): one Newton step in it is exact."""
    own = Name(eq.dependent, 0)
    d = diff(eq.residual, own)
    return d != _ZERO and diff(d, own) == _ZERO


def _augment(i: int, needs: list[list[int]], by_eq: list[int], by_unknown: list[int]) -> bool:
    """Pair equation ``i`` with an unknown along an augmenting path (breadth first), re-pairing those on it."""
    reached: dict[int, int] = {}  # unknown -> the equation it was reached from
    seen = {i}
    queue = deque([i])
    while queue:
        k = queue.popleft()
        for j in needs[k]:
            if j in reached:
                continue
            reached[j] = k
            if by_unknown[j] < 0:  # free: flip the path back to i
                while True:
                    k = reached[j]
                    before = by_eq[k]
                    by_eq[k], by_unknown[j] = j, k
                    if k == i:
                        return True
                    j = before
            other = by_unknown[j]
            if other not in seen:
                seen.add(other)
                queue.append(other)
    return False


def _components(deps: list[list[int]]) -> list[list[int]]:
    """Strongly connected components of the graph ``i -> deps[i]`` (Tarjan's algorithm, without recursion), each
    after every component it needs."""
    n = len(deps)
    index, low, on = [-1] * n, [0] * n, [False] * n
    stack: list[int] = []
    out: list[list[int]] = []
    counter = 0
    for root in range(n):
        if index[root] >= 0:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on[root] = True
        work = [(root, 0)]
        while work:
            v, p = work[-1]
            if p < len(deps[v]):
                work[-1] = (v, p + 1)
                w = deps[v][p]
                if index[w] < 0:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on[w] = True
                    work.append((w, 0))
                elif on[w]:
                    low[v] = min(low[v], index[w])
                continue
            work.pop()
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on[w] = False
                    comp.append(w)
                    if w == v:
                        break
                out.append(sorted(comp))
    return out


class BlockSolve:
    """One layout's equations solved block by block in each period (see the module docstring). ``system`` is the
    layout's whole period system (``_PeriodSystem``), used for the recursive equations; ``make`` builds the system of
    some of its equations and unknowns, for a block the solver iterates on."""

    def __init__(self, system: Any, make: Callable[[list[int], list[int]], Any]) -> None:
        self.system = system
        self.steps = plan(system.eqs, system.unknowns)
        self._make = make
        self._subs: dict[int, Any] = {}
        self.final: tuple[float, str] = (0.0, "")

    @property
    def simultaneous(self) -> int:
        return sum(not s.once for s in self.steps)

    @property
    def recursive(self) -> int:
        return sum(s.once for s in self.steps)

    def names(self) -> list[tuple[str, ...]]:
        """The blocks' equations by name, in solution order."""
        return [tuple(self.system.equation_names[i] for i in s.equations) for s in self.steps]

    def _sub(self, k: int) -> Any:
        if k not in self._subs:
            step = self.steps[k]
            self._subs[k] = self._make(list(step.equations), list(step.unknowns))
        return self._subs[k]

    def solve_period(self, solver: Any, pos: int) -> tuple[np.ndarray, int]:
        """The period ``pos`` solved (the system already at it): the unknowns and the solver's iterations, summed over
        the blocks it iterates on. ``self.final`` is the largest residual those blocks leave, and its equation; the
        report (``system.report``) gathers theirs."""
        system = self.system
        x = system.x0.copy()
        arrays, unknowns = system.arrays, system.unknowns
        iterations = 0
        worst, where = 0.0, ""
        tests: list[str] = []
        solvers: list[str] = []
        jacobians = 0
        for k, step in enumerate(self.steps):
            if step.once:
                i = step.equations[0]
                f, d, j = system.own(i, x)
                if d == 0 or not np.isfinite(d):
                    raise _stopped(f"equation {system.equation_names[i]!r} does not depend on its own variable here", iterations)
                x[j] -= f / d
                if not np.isfinite(x[j]):
                    raise _stopped(f"equation {system.equation_names[i]!r} gives {unknowns[j]} a non-finite value", iterations)
                continue
            for jj, u in enumerate(unknowns):  # the blocks before, solved: what this one takes as given
                arrays[u][pos] = x[jj]
            sub = self._sub(k)
            sub.at(pos)
            try:
                xs, used = solver.solve_period(sub)
            except ModelError as exc:
                exc.iterations = iterations + int(getattr(exc, "iterations", 0) or 0)  # type: ignore[attr-defined]
                exc.last = sub.last_residuals()  # type: ignore[attr-defined]
                raise
            x[list(step.unknowns)] = xs
            iterations += used
            residual, equation = sub.final_residuals(xs)
            if residual > worst or not where:
                worst, where = residual, equation
            report = sub.report
            tests.append(report.get("test", ""))
            solvers.append(report.get("solver") or describe(solver))
            jacobians += int(report.get("jacobians", 0))
        system.report.update(test=", ".join(dict.fromkeys(t for t in tests if t)),
                             solver=", ".join(dict.fromkeys(solvers)), jacobians=jacobians)
        self.final = (worst, where)
        return x, iterations
