"""``HASteadyState``: the calibration solved, as a table of the aggregates with the unknowns found and the targets
met; the household blocks' policies and distribution beside it."""

from __future__ import annotations

from typing import Any

import numpy as np

from pymetrix.errors import SpecificationError
from pymetrix.output import Table

from . import _provenance
from .blocks import het_blocks


class HASteadyState:
    """The steady state: ``ss["K"]``, ``ss.values`` (every aggregate), ``ss.table()``; ``ss.internals("household")``
    and ``ss.distribution()`` for a het block's policies and distribution; ``ss.checks()`` for the distribution's
    mass and the aggregates against it. ``ss.raw`` is sequence-jacobian's own ``SteadyStateDict``."""

    def __init__(self, model: Any, raw: Any, calls: dict[str, Any], evaluations: int, seconds: float,
                 caught: tuple[str, ...], printed: tuple[str, ...]) -> None:
        self.model, self.raw, self.evaluations, self.seconds = model, raw, int(evaluations), float(seconds)
        self.values: dict[str, float] = {}
        for k in raw.toplevel:
            try:
                self.values[str(k)] = float(raw[k])
            except (TypeError, ValueError):
                continue
        self.unknowns = {k: self.values[k] for k in model.ss_unknowns}
        self.residuals = {}
        for t, v in model.ss_targets.items():
            goal = self.values[v] if isinstance(v, str) else float(v)
            self.residuals[t] = self.values[t] - goal
        self.max_residual = max((abs(r) for r in self.residuals.values()), default=0.0)
        solver = model.options.steady_solver(model.ss_unknowns)
        self.provenance = _provenance.make(
            model, "steady state", calls, caught, printed, steady=self,
            resolved={"solver": solver, "evaluations": self.evaluations, "seconds": round(self.seconds, 3),
                      "unknowns": dict(self.unknowns), "target_residuals": dict(self.residuals)})

    def __getitem__(self, name: str) -> float:
        if name not in self.values:
            raise KeyError(f"no aggregate {name!r} in the steady state; it has {', '.join(self.values)}")
        return self.values[name]

    def __repr__(self) -> str:
        found = ", ".join(f"{k}={v:.6g}" for k, v in self.unknowns.items())
        return f"<HASteadyState of {self.model.name!r}: {found}; largest target residual {self.max_residual:.1e}>"

    def table(self) -> Table:
        """Every aggregate: its value, and whether it was calibrated, solved for (an unknown), a target (with the
        value asked for and the residual) or computed by a block."""
        targets = self.model.ss_targets
        rows, index = [], []
        for k, v in self.values.items():
            if k in self.model.ss_unknowns:
                role, goal, resid = "unknown (solved)", "", ""
            elif k in targets:
                goal = targets[k] if isinstance(targets[k], str) else float(targets[k])
                role, resid = "target", self.residuals[k]
            elif k in self.model.calibration:
                role, goal, resid = "calibrated", "", ""
            else:
                role, goal, resid = "computed", "", ""
            index.append(k)
            rows.append([v, role, goal, resid])
        notes = [f"Solved by {self.provenance.resolved['solver']} in {self.evaluations} evaluations of the model; "
                 f"largest target residual {self.max_residual:.1e}",
                 f"Backend: {self.provenance.method['backend']} (Auclert, Bardóczy, Rognlie and Straub 2021)"]
        return Table(["value", "role", "target", "residual"], rows, index=index, title=f"Steady state: {self.model.name}",
                     notes=notes, index_name="aggregate")

    def to_frame(self) -> Any:
        return self.table().to_frame()

    def _het(self, block: str | None) -> str:
        names = [str(b.name) for b in het_blocks(self.model.blocks)]
        if not names:
            raise SpecificationError("the model has no het block")
        if block is None:
            return names[0]
        if block not in names:
            raise SpecificationError(f"no het block {block!r}; the model's are {', '.join(names)}")
        return block

    def internals(self, block: str | None = None) -> dict[str, np.ndarray]:
        """A het block's internals: its grids, Markov matrix, policies (``a``, ``c``), backward variable and the
        distribution (``D``; ``Dbeg`` before the period's income shock), as arrays (income states by asset points)."""
        name = self._het(block)
        return {str(k): np.asarray(v) for k, v in self.raw.internals[name].items()}

    def distribution(self, block: str | None = None) -> np.ndarray:
        """The stationary distribution of a het block (``D``): the mass at each income state and asset point."""
        return self.internals(block)["D"]

    def checks(self) -> Table:
        """For each het block: the distribution's total mass less 1 and its smallest entry; each aggregate output
        less the distribution's sum of its policy (``A`` against the sum of ``D * a``); and each target's residual."""
        rows, index = [], []
        for b in het_blocks(self.model.blocks):
            inner = self.internals(str(b.name))
            D = inner["D"]
            index += [f"{b.name}: mass less 1", f"{b.name}: smallest mass"]
            rows += [[float(D.sum() - 1.0)], [float(D.min())]]
            for out in b.outputs:
                policy = str(out).lower()
                if policy in inner and str(out) in self.values:
                    index.append(f"{b.name}: {out} less sum of D * {policy}")
                    rows.append([float(self.values[str(out)] - np.vdot(D, inner[policy]))])
        for t, r in self.residuals.items():
            index.append(f"target {t}: residual")
            rows.append([float(r)])
        return Table(["value"], rows, index=index, title="Steady-state checks", index_name="check")
