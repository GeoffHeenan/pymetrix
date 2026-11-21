"""Scenario: how a simulation departs from the data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from .._api import renamed_arguments

Path = pd.Series | float


@dataclass(frozen=True, eq=False)
class Scenario:
    """How a simulation departs from the data. ``hold`` and ``free`` exogenize and endogenize: by default each held
    variable is paired, in order, with one free variable that takes whatever values hold it. With ``hold_sd`` or
    ``free_sd`` (first-order solutions only), or a different number of free shocks than holds, the plan is a
    least-norm conditional forecast instead: every free shock is free in each period up to the last held one, and
    the smallest free shocks (in units of ``free_sd``) that meet the holds are found; a hold with a standard deviation
    in ``hold_sd`` is soft, met only as far as it is worth, and one without (or 0) is exact. ``hold_lower`` and
    ``hold_upper`` (first-order solutions only) keep variables within a range instead: the free shocks move only as far
    as needed to keep each bounded value at or above its lower bound and at or below its upper one.

    ``partial`` (nonlinear solvers: Newton, Gauss-Seidel, perfect foresight) applies ``fix`` and ``hold`` only in the
    periods their paths give a value: ``True`` for every fixed and held variable, or the names of some. In a period
    with a value (not NaN) the variable is exogenous at it, its equation dropped (fixed) or its free variable solved
    for (held); in a period without one its equation determines it as usual, and a held variable's free partner keeps
    its value in the data. A fix at the data's values (``fix={"x": None}``) applies where the data have values. Without
    ``partial`` a fixed or held variable needs a value in every period of the span, as before. A first-order solution
    holds variables only where their paths give values in any case."""

    set: Mapping[str, Path] = field(default_factory=dict)  # replace exogenous paths
    shift: Mapping[str, Path] = field(default_factory=dict)  # add to exogenous paths
    fix: Mapping[str, Path | None] = field(default_factory=dict)  # pin endogenous, dropping their equations (None: at data values)
    adjust: Mapping[str, Path] = field(default_factory=dict)  # add factor, by equation name
    hold: Mapping[str, Path] = field(default_factory=dict)  # endogenous paths to hit... (was target)
    free: Sequence[str] = ()  # ...by solving for these exogenous variables
    unanticipated: Sequence[str] = ()  # shocks that are surprises (first-order solutions); others are known in advance
    name: str = "scenario"
    hold_sd: Mapping[str, Path] = field(default_factory=dict)  # soft holds: a standard deviation for a held value (0: exact)
    free_sd: Mapping[str, float] = field(default_factory=dict)  # least-norm weights of the free shocks (default 1)
    hold_lower: Mapping[str, Path] = field(default_factory=dict)  # ranges: keep a variable at or above these values...
    hold_upper: Mapping[str, Path] = field(default_factory=dict)  # ...and at or below these (least-norm free shocks)
    partial: bool | Sequence[str] = False  # fix= and hold= only where their paths have values (True: all; or names)

    @property
    def ranges(self) -> bool:
        """Whether the scenario keeps some variable within a range (``hold_lower`` or ``hold_upper``)."""
        return bool(self.hold_lower or self.hold_upper)


Scenario.__init__ = renamed_arguments(target="hold")(Scenario.__init__)  # type: ignore[method-assign]
