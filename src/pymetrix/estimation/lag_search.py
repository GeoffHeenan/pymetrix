"""Choosing how many lags an equation takes, by information criteria on a common sample.

    search = Equation("dly ~ dlx").select_lags(data, max_lags=8)            # lags of the left-hand side: dly[-1], ...
    search = Equation("dly ~ dlx").select_lags(data, max_lags=4, of=["dly", "dlx"])   # distributed lags too
    search.table; search.best["bic"]; search.equation("bic").fit(data)

Lags 1 to p of each variable in ``of`` (by default the left-hand side as written, so ``dlog(y)[-1]`` for an
equation in ``dlog(y)``) are added for p = 0 ... ``max_lags``; lags the equation already has are not added again.
Every candidate is fitted on the sample the largest allows, by the equation's own method (least squares, or
maximum likelihood with ARMA errors or a GARCH variance), and compared by AIC, BIC and Hannan-Quinn, as totals.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..errors import SpecificationError
from ..output import Table

if TYPE_CHECKING:
    from ..data import Dataset
    from .equation import Equation

__all__ = ["EquationLagSelection", "select_lags"]

CRITERIA = (("aic", "AIC"), ("bic", "BIC"), ("hq", "HQ"))


@dataclass(frozen=True, eq=False)
class EquationLagSelection:
    """Information criteria for 0 ... max_lags lags of the variables searched; ``best`` maps each criterion to the
    number of lags it picks, and ``equation(criterion)`` gives that equation."""

    table: Table
    best: dict[str, int]
    equations: tuple[Equation, ...]  # the equation with 0, 1, ... max_lags lags
    sample: tuple[str, str]

    def equation(self, criterion: str = "bic") -> Equation:
        """The equation with the lags ``criterion`` ("aic", "bic" or "hq") picks."""
        if criterion not in self.best:
            raise SpecificationError(f"criterion is one of {', '.join(self.best)}")
        return self.equations[self.best[criterion]]

    def __str__(self) -> str:
        return str(self.table)

    def _repr_html_(self) -> str:
        return self.table._repr_html_()


def _lag_text(variable: str, k: int) -> str:
    simple = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*(\(.*\))?", variable.strip()) is not None
    return f"{variable.strip()}[-{k}]" if simple else f"({variable.strip()})[-{k}]"


def select_lags(eq: Equation, data: Dataset, max_lags: int = 8, of: Sequence[str] | None = None, span: tuple[Any, Any] | None = None,
                method: Any = None) -> EquationLagSelection:
    """See the module docstring."""
    from .equation import Equation

    if eq.is_identity or eq.form != "linear":
        raise SpecificationError("lag search is for estimated equations linear in their coefficients")
    if int(max_lags) < 1:
        raise SpecificationError("max_lags must be at least 1")
    variables = [eq.text.split("~", 1)[0].strip()] if not of else [str(v) for v in of]
    have = {t.expr for t in eq.terms}
    candidates: list[Equation] = []
    added: list[str] = []
    for p in range(int(max_lags) + 1):
        if p:
            for v in variables:
                text = _lag_text(v, p)
                term = Equation(f"{eq.text.split('~', 1)[0]} ~ {text}").terms[0].expr
                if term not in have:
                    added.append(text)
        rhs = " + ".join(added)
        candidates.append(replace(eq, text=f"{eq.text} + {rhs}") if rhs else eq)
    largest = candidates[-1].fit(data, span=span, method=method)
    if "aic" not in largest.stats:
        raise SpecificationError(f"{type(largest.method).__name__} gives no likelihood, so no information criteria; use least squares or maximum likelihood")
    sample = tuple(largest.provenance.sample)  # type: ignore[union-attr]
    fits = [c.fit(data, span=sample, method=method) for c in candidates[:-1]] + [largest]
    nobs = [int(f.stats["nobs"]) for f in fits]
    frame = pd.DataFrame([[f.stats[k] for k, _ in CRITERIA] + [n] for f, n in zip(fits, nobs)],
                         index=pd.Index(range(len(fits)), name="lags"), columns=[label for _, label in CRITERIA] + ["nobs"])
    best = {k: int(frame[label].to_numpy().argmin()) for k, label in CRITERIA}
    searched = ", ".join(variables)
    notes = [f"Lags 1 to p of {searched}; {nobs[-1]} observations ({sample[0]} to {sample[1]}) for every p",
             "Totals: AIC -2 log L + 2k, BIC -2 log L + k ln n, HQ -2 log L + 2k ln ln n (smaller is better)"]
    if len(set(nobs)) > 1:
        notes.append("The samples differ (missing values inside the span): compare with care")
    notes += [f"Best by {label}: {best[k]} lag{'s' * (best[k] != 1)}" for k, label in CRITERIA]
    table = Table.from_frame(frame, title=f"Lag search: {eq.text}", notes=notes)
    return EquationLagSelection(table, best, tuple(candidates), (str(sample[0]), str(sample[1])))

