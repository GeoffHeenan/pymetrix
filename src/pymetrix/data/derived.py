"""Derived series: series that keep the recipe that made them.

A derived series stores its values and how to compute them, either a
formula (``Formula("dlog(y)")``) or a filter (``Filtered("y", HP(), "trend")``).
When a series it depends on changes, the dataset recomputes it, and anything
derived from it in turn. Recipes are saved with the data (Parquet metadata and
projects), so they survive a save and reopen.

``Dataset.with_series`` stays static, like EViews' ``genr``; ``Dataset.derive``
is dynamic, like EViews' auto-updating series. Two-sided filters revise their
history when new data arrive; that is what they do, and results built on them
are then flagged out of date by the session.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from ..errors import DataError
from .._serial import to_jsonable

if TYPE_CHECKING:
    from .dataset import Dataset

__all__ = ["Filtered", "Formula", "Recipe", "SeasonallyAdjusted", "recipe_from_dict"]


@dataclass(frozen=True)
class Formula:
    """A series computed from an expression of other series, e.g. ``dlog(y)`` or ``c / y``."""

    expr: str

    def inputs(self, data: Dataset) -> frozenset[str]:
        from ..expr import parse_expression

        from ..expr import Level

        node = parse_expression(self.expr)
        groups = {n.id for n in node.walk() if isinstance(n, Level) and n.id in data.names}  # grouping series
        return node.variables() | groups

    def compute(self, data: Dataset) -> pd.Series:
        return data.eval(self.expr)

    def describe(self, data: Dataset | None = None) -> str:
        return self.expr


@dataclass(frozen=True)
class Filtered:
    """The trend or cycle of series ``of`` from a trend-cycle filter (``HP()``, ``Hamilton()``...)."""

    of: str
    method: Any
    part: Literal["trend", "cycle"] = "trend"

    def __post_init__(self) -> None:
        if self.part not in ("trend", "cycle"):
            raise DataError("part must be 'trend' or 'cycle'")

    def inputs(self, data: Dataset) -> frozenset[str]:
        return frozenset({self.of})

    def compute(self, data: Dataset) -> pd.Series:
        from ..filters import trend_cycle

        trend, cycle = trend_cycle(data, self.of, self.method)
        return trend if self.part == "trend" else cycle

    def describe(self, data: Dataset | None = None) -> str:
        label = type(self.method).__name__
        if data is not None and data.time_axis is not None:
            from ..filters import periods_per_year

            try:
                label = self.method.describe(periods_per_year(data.time_axis.freq))
            except ValueError:
                pass
        return f"{label} {self.part} of {self.of}"


@dataclass(frozen=True)
class SeasonallyAdjusted:
    """The seasonally adjusted series (``sa``) or seasonal factor (``seasonal``) of series ``of``."""

    of: str
    method: Any
    part: Literal["sa", "seasonal"] = "sa"

    def __post_init__(self) -> None:
        if self.part not in ("sa", "seasonal"):
            raise DataError("part must be 'sa' or 'seasonal'")

    def inputs(self, data: Dataset) -> frozenset[str]:
        used = getattr(self.method, "inputs", None)  # a method's other series (TRAMO-SEATS's regressors)
        return frozenset({self.of, *(used() if callable(used) else ())})

    def compute(self, data: Dataset) -> pd.Series:
        from ..seasonal import seasonal_components

        return self.compute_noted(data, {})[0]

    def compute_noted(self, data: Dataset, extra: Mapping[str, Any]) -> tuple[pd.Series, dict[str, Any]]:
        """The series, and the variable's ``extra`` with what this run reports (the library and its arguments; X-13's
        program, version, model and outliers) in place of what the last run reported."""
        from ..seasonal import seasonal_components

        notes: dict[str, str] = {}
        sa, seasonal = seasonal_components(data, self.of, self.method, notes)
        kept = {k: v for k, v in extra.items() if not k.startswith(("x13_", "backend"))}
        return (sa if self.part == "sa" else seasonal), {**kept, **notes}

    def describe(self, data: Dataset | None = None) -> str:
        what = "seasonally adjusted" if self.part == "sa" else "seasonal factor of"
        return f"{self.of} {what} ({self.method.describe()})" if self.part == "sa" else f"{what} {self.of} ({self.method.describe()})"


Recipe = Formula | Filtered | SeasonallyAdjusted


def _filter_class(name: str, group: str = "filters") -> type:
    from ..plugins import registry

    for entry in registry(group).entries():
        if entry.obj.__name__ == name:
            return entry.obj
    what = "filter" if group == "filters" else "seasonal adjustment method"
    raise DataError(f"the {what} {name!r} used by a derived series is not installed")


def recipe_from_dict(d: Mapping[str, Any] | None) -> Recipe | None:
    """Rebuild a recipe saved with ``to_jsonable`` (as Parquet metadata and projects do)."""
    if not d:
        return None
    kind = d.get("type")
    if kind == "Formula":
        return Formula(d["expr"])
    if kind == "Filtered":
        spec = dict(d["method"])
        cls = _filter_class(spec.pop("type"))
        names = {f.name for f in fields(cls)}
        return Filtered(d["of"], cls(**{k: v for k, v in spec.items() if k in names}), d.get("part", "trend"))
    if kind == "SeasonallyAdjusted":
        spec = dict(d["method"])
        cls = _filter_class(spec.pop("type"), "seasonal")
        names = {f.name for f in fields(cls)}
        return SeasonallyAdjusted(d["of"], cls(**{k: v for k, v in spec.items() if k in names}), d.get("part", "sa"))
    raise DataError(f"unknown recipe {kind!r}")


def recipe_to_dict(recipe: Recipe | None) -> dict[str, Any] | None:
    return to_jsonable(recipe) if recipe is not None else None


def order(recipes: Mapping[str, Recipe], data: Dataset, changed: set[str] | None = None) -> list[str]:
    """Derived series to compute, in dependency order: all of them, or those that (directly or
    through other derived series) depend on ``changed``. Raises on circular definitions."""
    deps = {name: set(r.inputs(data)) for name, r in recipes.items()}
    done: list[str] = []
    state: dict[str, int] = {}

    def visit(name: str, path: tuple[str, ...]) -> None:
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            raise DataError("circular derived series: " + " -> ".join((*path, name)))
        state[name] = 1
        for dep in deps.get(name, ()):
            if dep in recipes:
                visit(dep, (*path, name))
        state[name] = 2
        done.append(name)

    for name in recipes:
        visit(name, ())
    if changed is None:
        return done
    affected: set[str] = set()
    for name in done:  # dependency order, so one pass collects everything downstream
        if deps[name] & (changed | affected):
            affected.add(name)
    return [n for n in done if n in affected]
