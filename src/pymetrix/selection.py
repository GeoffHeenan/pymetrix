"""Selections of series and collections of results: loops over series, in Python.

A ``Selection`` is an ordered set of series names, chosen by listing them, by
pattern or by metadata (like column selectors in R's tidyselect, polars and
Julia's DataFrames)::

    s = data.series("c", "i", "g")
    cycles = data.series("*_cycle")
    real = data.series(prices="constant")

Loop over it as over any list (``for name in s:``), or let pymetrix do the loop.
Templates use ``{s}`` for the series name::

    data = data.derive(**s.formulas("d{s}", "dlog({s})"))   # dc, di, dg
    fits = s.fit("d{s} ~ d{s}[-1]", data)                   # one fit per series
    adf = s.test(data, "adf", lags=4)                       # one test per series
    fits.coefficients()                                     # side by side

Each loop returns ``Results``: the results by series, with tables that put
them side by side.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ._api import api
from .errors import NotFoundError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["Results", "Selection", "SelectionLike", "as_selection", "fit_each", "test_each"]


@dataclass(frozen=True)
class Selection(Sequence[str]):
    """An ordered set of series names."""

    names: tuple[str, ...]
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", tuple(dict.fromkeys(str(n) for n in self.names)))

    # a sequence of names

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: Any) -> Any:
        return Selection(self.names[i]) if isinstance(i, slice) else self.names[i]

    def __contains__(self, item: object) -> bool:
        return item in self.names

    def __add__(self, other: Sequence[str]) -> Selection:
        return Selection((*self.names, *other))

    def __sub__(self, other: Sequence[str]) -> Selection:
        drop = set(other)
        return Selection(tuple(n for n in self.names if n not in drop))

    def __repr__(self) -> str:
        label = f"{self.name}: " if self.name else ""
        return f"<Selection {label}{', '.join(self.names)}>"

    # templates

    def format(self, template: str) -> Selection:
        """New names from a template: ``s.format("d{s}")`` -> dc, di, dg."""
        return Selection(tuple(template.format(s=s) for s in self.names))

    def formulas(self, name: str, formula: str) -> dict[str, str]:
        """``{new name: formula}`` for each series, for ``derive`` or ``with_series``:
        ``data.derive(**s.formulas("d{s}", "dlog({s})"))``."""
        return {name.format(s=s): formula.format(s=s) for s in self.names}

    # loops

    def map(self, fn: Callable[[str], Any], errors: str = "raise") -> Results:
        """``fn(series)`` for each series. ``errors="collect"`` keeps going and records failures."""
        out: dict[str, Any] = {}
        failed: dict[str, str] = {}
        for s in self.names:
            try:
                out[s] = fn(s)
            except Exception as exc:
                if errors == "raise":
                    raise
                failed[s] = f"{type(exc).__name__}: {exc}"
        return Results(out, failed, title=self.name)

    @api("pymetrix:selection.fit")
    def fit(self, formula: str, data: Dataset, errors: str = "raise", **fit_options: Any) -> Results:
        """Estimate the equation ``formula`` (with ``{s}``) for each series: ``s.fit("{s} ~ {s}[-1]", data)``.
        Options (``method=``, ``span=``...) go to ``Equation.fit``."""
        from .estimation import Equation

        results = self.map(lambda s: Equation(formula.format(s=s)).fit(data, **fit_options), errors)
        results.template = formula  # so coefficient tables line up across series (dc[-1], di[-1] -> d{s}[-1])
        return results

    @api("pymetrix:selection.test")
    def test(self, data: Dataset, test: str, *args: Any, errors: str = "raise", **options: Any) -> Results:
        """Run a test on each series of a dataset: ``s.test(data, "adf", lags=4)``."""
        return self.map(lambda s: data.test(test, s, *args, **options), errors)


class Results(Mapping[str, Any]):
    """Results by series (or by any name), in order, with side-by-side tables."""

    def __init__(self, results: Mapping[str, Any], failed: Mapping[str, str] | None = None, title: str = "") -> None:
        self._results = dict(results)
        self.failed = dict(failed or {})
        self.title = title
        self.template = ""  # set by Selection.fit: names in it with {s} line up across results

    def __getitem__(self, key: str) -> Any:
        return self._results[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def __repr__(self) -> str:
        kinds = sorted({type(v).__name__ for v in self._results.values()})
        more = f", {len(self.failed)} failed" if self.failed else ""
        return f"<Results: {len(self)} × {'/'.join(kinds) or 'nothing'}{more}>"

    def map(self, fn: Callable[[Any], Any]) -> Results:
        return Results({k: fn(v) for k, v in self._results.items()}, self.failed, self.title)

    # side-by-side tables

    def _kind(self) -> str:
        from .estimation import FittedEquation
        from .stats import TestResult

        values = list(self._results.values())
        if values and all(isinstance(v, FittedEquation) for v in values):
            return "fits"
        if values and all(isinstance(v, TestResult) for v in values):
            return "tests"
        return "values"

    def coefficients(self, se: bool = True) -> Table:
        """Coefficients (and standard errors below them) with one column per result."""
        labelled = {key: {self._generic(key, n): (fit.params[n], fit.params.se[i]) for i, n in enumerate(fit.params.names)}
                    for key, fit in self._results.items()}
        names: list[str] = []
        for params in labelled.values():
            names += [n for n in params if n not in names]
        rows = []
        for n in names:
            rows.append([n] + [params[n][0] if n in params else None for params in labelled.values()])
            if se:
                rows.append([""] + [_bracket(params[n][1]) if n in params else "" for params in labelled.values()])
        return Table(["", *self._results], rows, title=self._title("Coefficients"), notes=["Standard errors in brackets."] if se else [])

    def _generic(self, key: str, name: str) -> str:
        """A parameter name with this result's series replaced by the template's ``{s}`` names."""
        import re

        for token in dict.fromkeys(re.findall(r"\w*\{s\}\w*", self.template)):
            pattern = r"(?<!\w)" + re.escape(token.format(s=key)) + r"(?!\w)"
            name = re.sub(pattern, lambda _m, t=token: t, name)
        return name

    def stats(self, *keys: str) -> Table:
        """Fit statistics side by side (default: observations, R², adjusted R², AIC, BIC, log likelihood)."""
        wanted = keys or ("nobs", "r2", "r2_adj", "aic", "bic", "loglike")
        present = [k for k in wanted if any(k in fit.stats for fit in self._results.values())]
        rows = [[k] + [fit.stats.get(k) for fit in self._results.values()] for k in present]
        return Table(["", *self._results], rows, title=self._title("Statistics"))

    def tests(self) -> Table:
        """Test statistics, p-values and decisions at 5%, one row per result."""
        rows = []
        for name, t in self._results.items():
            decision = "" if t.p_value is None else ("reject" if t.p_value < 0.05 else "do not reject")
            rows.append([name, t.statistic, t.p_value, decision])
        first = next(iter(self._results.values()), None)
        name = first.name.split(" (")[0] if first is not None else "Tests"  # "ADF (c)" -> "ADF"
        return Table(["", "statistic", "p-value", "at 5%"], rows, title=self._title(name),
                     notes=[f"Null hypothesis: {first.null}"] if first is not None and first.null else [])

    def value_table(self) -> Table:
        rows = [[name, v if isinstance(v, (int, float, np.floating)) else str(v)] for name, v in self._results.items()]
        return Table(["", "value"], rows, title=self._title("Results"))

    def summary(self) -> Table:
        kind = self._kind()
        if kind == "fits":
            return self.coefficients()
        if kind == "tests":
            return self.tests()
        return self.value_table()

    def failures(self) -> Table:
        return Table(["", "error"], [[k, v] for k, v in self.failed.items()], title=self._title("Failed"))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:  # one of its views by name, as for every result (pymetrix.views)
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.summary().to_frame()

    def _title(self, what: str) -> str:
        return f"{what}: {self.title}" if self.title else what

    def __str__(self) -> str:
        return str(self.summary())

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


def _bracket(value: float) -> str:
    return f"({value:.4g})" if np.isfinite(value) else ""


def select_series(data: Dataset, patterns: Sequence[str], where: Callable[[Any], bool] | None = None, **meta: Any) -> Selection:
    """Series of ``data`` matching any of ``patterns`` (names or shell patterns such as ``*_cycle``;
    none = all), whose metadata equal ``meta`` and for which ``where(variable)`` is true."""
    names = list(data.names)
    if patterns:
        chosen: list[str] = []
        for p in patterns:
            matches = [n for n in names if fnmatch.fnmatchcase(n, p)]
            if not matches and not any(ch in p for ch in "*?["):
                raise NotFoundError(f"no series {p!r} in the dataset")
            chosen += [m for m in matches if m not in chosen]
        names = chosen
    out = []
    for n in names:
        var = data.variables[n]
        ok = all(_meta_equal(getattr(var, k, None), v) for k, v in meta.items())
        if ok and (where is None or where(var)):
            out.append(n)
    return Selection(tuple(out))


def _meta_equal(actual: Any, wanted: Any) -> bool:
    if wanted is True:  # e.g. derived=True: any recipe
        return actual not in (None, "", False)
    if wanted is False:
        return actual in (None, "", False)
    return getattr(actual, "value", actual) == getattr(wanted, "value", wanted)


SelectionLike = Selection | Sequence[str] | str  # what any function taking series names accepts


def as_selection(value: SelectionLike, data: Dataset | None = None) -> Selection:
    """Series names as a ``Selection``: a Selection unchanged; a list or tuple of names; one name; or, with ``data``,
    a pattern such as ``"*_cycle"``. A string is one name (or a pattern if it has ``*`` or ``?``), never split
    at commas."""
    if isinstance(value, Selection):
        return value
    if isinstance(value, str):
        if data is not None and any(c in value for c in "*?["):
            return data.series(value)
        return Selection((value,))
    return Selection(tuple(str(v) for v in value))


def fit_each(names: SelectionLike, formula: str, data: Dataset, errors: str = "raise", **fit_options: Any) -> Results:
    """``Selection.fit`` for any list of names: ``fit_each(["c", "i"], "d{s} ~ d{s}[-1]", data)``."""
    return as_selection(names, data).fit(formula, data, errors=errors, **fit_options)


def test_each(names: SelectionLike, data: Dataset, test: str, *args: Any, errors: str = "raise", **options: Any) -> Results:
    """``Selection.test`` for any list of names: ``test_each(["c", "i"], data, "adf", lags=4)``."""
    return as_selection(names, data).test(data, test, *args, errors=errors, **options)


test_each.__test__ = False  # type: ignore[attr-defined]  # not a pytest test
