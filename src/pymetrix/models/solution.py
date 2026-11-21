"""Solution: a Dataset that also records how it was produced."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..data import Dataset
from ..errors import SpecificationError
from ..metadata import Provenance


@dataclass(frozen=True, eq=False, repr=False)
class Solution(Dataset):
    model: Any = None
    scenario: Any = None
    converged: bool = True
    iterations: int = 0
    provenance: Provenance | None = None
    simulation: Any = None  # a stochastic simulation's draws (``Simulations``), if one was asked for
    # by period (Newton, Gauss-Seidel, perfect foresight): iterations, the largest absolute residual at the end and its
    # equation, whether it converged; None for a first-order solution (and for solutions saved before it was recorded)
    # period by period, its .attrs["seconds"] gives the seconds each period's solve took (not a column)
    convergence: pd.DataFrame | None = None

    @property
    def span(self) -> tuple[str, str] | None:
        return self.provenance.sample if self.provenance else None

    def solved(self) -> Dataset:
        """The simulated periods only."""
        return self.select(span=self.span) if self.span else self

    def __repr__(self) -> str:
        span = f"{self.span[0]} to {self.span[1]}" if self.span else "?"
        name = getattr(self.scenario, "name", "baseline")
        sims = self._simulations(required=False)
        drawn = f", {sims.values.shape[0]} draws" if sims is not None else ""
        return f"<Solution {span}, scenario {name!r}, {self.iterations} iterations{drawn}>"

    # a stochastic simulation's draws, summarised

    def _simulations(self, required: bool = True) -> Any:
        sims = getattr(self, "simulation", None)  # a Solution saved before simulations has no such attribute
        if sims is None and required:
            raise SpecificationError("this solution has no draws: solve with draws=..., e.g. model.solve(data, span, draws=1000)")
        return sims

    def _summary(self, values: np.ndarray, names: Sequence[str]) -> Dataset:
        sims = self._simulations()
        frame = pd.DataFrame(values, index=sims.periods, columns=list(names))
        frame.index.name = self.frame.index.name or "time"
        return Dataset.from_frame(frame)

    @staticmethod
    def _label(q: float) -> str:
        return f"q{q * 100:g}".replace(".", "_")

    def quantiles(self, qs: Sequence[float] = (0.05, 0.5, 0.95), variables: Sequence[str] | None = None) -> Dataset:
        """The draws' quantiles in each period of the span: ``{variable}_q5``, ``{variable}_q50``, ... (``q2_5`` for 2.5%)."""
        sims = self._simulations()
        names = list(variables) if variables else list(sims.names)
        if any(not 0 <= q <= 1 for q in qs):
            raise SpecificationError("quantiles are between 0 and 1")
        cols, values = [], []
        for v in names:
            j = sims.names.index(v)
            for q in qs:
                cols.append(f"{v}_{self._label(q)}")
                values.append(np.nanquantile(sims.values[:, :, j], q, axis=0))
        return self._summary(np.column_stack(values), cols)

    def mean(self) -> Dataset:
        """The draws' mean in each period (for a nonlinear model, not the deterministic solution)."""
        sims = self._simulations()
        return self._summary(np.nanmean(sims.values, axis=0), sims.names)

    def sd(self) -> Dataset:
        """The draws' standard deviation in each period."""
        sims = self._simulations()
        return self._summary(np.nanstd(sims.values, axis=0, ddof=1), sims.names)

    def bands(self, coverages: Sequence[float] = (0.3, 0.6, 0.9), variables: Sequence[str] | None = None) -> pd.DataFrame:
        """The median of the draws and central bands holding each share of them: ``{variable}``, ``{variable}_lo90``,
        ``{variable}_hi90``, ..., as forecast rounds and fan charts take them."""
        sims = self._simulations()
        out = {}
        for v in (list(variables) if variables else list(sims.names)):
            draws = sims.values[:, :, sims.names.index(v)]
            out[v] = np.nanmedian(draws, axis=0)
            for c in coverages:
                pct = int(round(c * 100))
                out[f"{v}_lo{pct}"] = np.nanquantile(draws, (1 - c) / 2, axis=0)
                out[f"{v}_hi{pct}"] = np.nanquantile(draws, (1 + c) / 2, axis=0)
        return pd.DataFrame(out, index=sims.periods)

    def fan(self, variables: Sequence[str] | None = None, coverages: Sequence[float] = (0.3, 0.6, 0.9), history: int = 16) -> Any:
        """Fan charts of the draws: the median and bands at each coverage, after the recent history in the data."""
        from ..forecasting import fan_figure

        sims = self._simulations()
        names = list(variables) if variables else list(sims.names)
        title = f"Stochastic simulation, {sims.values.shape[0]} draws"
        return fan_figure(self.bands(coverages, names), self, names, coverages, history, title=title)


@dataclass(frozen=True, eq=False, repr=False)
class PanelSolution(Dataset):
    """A model solved for each entity of a panel (``Model(..., over="country")``): a panel of every entity's
    solution, with each entity's own ``Solution`` (its scenario, convergence and any draws) in ``solutions``."""

    solutions: Mapping[Any, Solution] = field(default_factory=dict)
    over: str = ""
    links: tuple[str, ...] = ()  # variables each entity took from the others (Model links), solved jointly
    rounds: int = 0  # rounds of entity solutions until the links settled

    def solution(self, entity: Any) -> Solution:
        """One entity's solution."""
        if entity not in self.solutions:
            raise SpecificationError(f"no {self.over} {entity!r}; the solution has {', '.join(map(str, self.solutions))}")
        return self.solutions[entity]

    @property
    def span(self) -> tuple[str, str] | None:
        first = next(iter(self.solutions.values()), None)
        return first.span if first is not None else None

    @property
    def converged(self) -> bool:
        return all(s.converged for s in self.solutions.values())

    def solved(self) -> Dataset:
        """The simulated periods only, for every entity."""
        return self.select(span=self.span) if self.span else self

    def __repr__(self) -> str:
        span = f"{self.span[0]} to {self.span[1]}" if self.span else "?"
        linked = f", linked by {', '.join(self.links)} in {self.rounds} rounds" if self.links else ""
        return f"<PanelSolution {span} for {len(self.solutions)} {self.over} values{linked}>"
