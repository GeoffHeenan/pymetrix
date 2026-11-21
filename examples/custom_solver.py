"""A custom solution method: damped Newton with a backtracking line search.

A solver solves one period at a time. ``Model.solve`` gives it a
``PeriodSystem``: ``x0`` (starting values: last period's solution),
``residuals(x)`` and ``jacobian(x)`` of the model's equations in that period,
``own(i, x)`` (equation i's residual, derivative and variable, for
equation-by-equation methods) and ``equation_names``. It returns the solution
and the number of iterations used.

Once registered it appears in Model → Solution → Solve model's solver list, and works in
scripts and projects like the built-in solvers.

Run it from the repository folder:  python examples/custom_solver.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np

from pymetrix import Dataset, Equation, Model, toolkit
from pymetrix.forms import Widget


@dataclass(frozen=True)
class DampedNewton:
    """Newton's method that halves its step until the residuals shrink: slower, but steadier far from the solution."""

    label: ClassVar[str] = "Damped Newton"
    topics: ClassVar[tuple[str, ...]] = ("model-solution",)

    tol: Annotated[float, Widget("number", min=1e-15, max=1e-2, help="largest equation residual accepted")] = 1e-10
    max_iter: Annotated[int, Widget("number", min=1, max=10000, help="iterations allowed in each period")] = 100
    halvings: Annotated[int, Widget("number", min=1, max=60, advanced=True, help="most step halvings per iteration")] = 30

    def solve_period(self, system: toolkit.PeriodSystem) -> tuple[np.ndarray, int]:
        x = system.x0.copy()
        F = system.residuals(x)
        for it in range(self.max_iter + 1):
            if not np.all(np.isfinite(F)):
                raise toolkit.ModelError("a residual is not finite; check the starting values")
            if np.max(np.abs(F), initial=0.0) < self.tol:
                return x, it
            step = np.linalg.solve(system.jacobian(x), -F)
            size = np.linalg.norm(F)
            for _ in range(self.halvings):  # backtrack until the residuals shrink
                trial = x + step
                F_trial = system.residuals(trial)
                if np.all(np.isfinite(F_trial)) and np.linalg.norm(F_trial) < size:
                    break
                step /= 2
            x, F = trial, F_trial
        raise toolkit.ModelError(f"damped Newton did not converge in {self.max_iter} iterations")


toolkit.register("solvers", DampedNewton)  # now in Model → Solution → Solve model's solver list


if __name__ == "__main__":
    data = Dataset.read("examples/sample_macro.parquet")
    cons = Equation("c ~ y + c[-1]").fit(data)
    model = Model([cons, Equation("y = c + i + g")])
    ours = model.solve(data, span=("2020Q1", "2024Q4"), solver=DampedNewton())
    builtin = model.solve(data, span=("2020Q1", "2024Q4"))
    print(ours.solved().frame[["c", "y"]].round(3).head())
    print("same as the built-in Newton:", np.allclose(ours.solved()["y"], builtin.solved()["y"]))
