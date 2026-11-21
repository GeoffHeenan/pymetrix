"""Make one new series for each of several series, from a name template and a formula template.

``Dataset.with_each`` and the Generate series dialog's *For each* field use this::

    data.with_each("gdp cons inv", "{x}_g", "100*d(log({x}))")      # gdp_g, cons_g, inv_g
    data.with_each("ip_*", "{1}_growth", "100*dlog({x})")            # ip_us -> us_growth, ...

*For each* lists series (separated by spaces or commas), wildcards (``*`` any run of characters, ``?`` one character)
or both. In the templates ``{x}`` is the series (``{s}`` too, as in selection templates), and ``{1}``, ``{2}``... are
the parts of its name a wildcard's ``*`` or ``?`` matched, in order (each ``*`` takes as much as it can, left first).
``{{`` and ``}}`` are literal braces.

``expand_each`` is the pure expansion (new name -> formula); ``preview_each`` checks every row against a dataset: names
that are not valid, formulas that do not parse or use series the dataset lacks, names made twice, and series that exist
already (allowed with ``replace``).
"""

from __future__ import annotations

import keyword
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ..errors import SpecificationError

__all__ = ["EachPreview", "EachRow", "expand_each", "fill_template", "preview_each", "split_series"]

_TOKEN = re.compile(r"\{\{|\}\}|\{([^{}]*)\}|[{}]")
_PLACEHOLDER_HELP = "use {x} for the series and {1}, {2}... for the parts a wildcard matched ({{ and }} for literal braces)"


def split_series(series: str | Iterable[str]) -> list[str]:
    """The entries of *For each*: ``"gdp cons, inv"`` -> ``["gdp", "cons", "inv"]`` (a list or a selection as it is)."""
    if isinstance(series, str):
        return [s for s in re.split(r"[\s,]+", series) if s]
    return [str(s).strip() for s in series if str(s).strip()]


def _is_wildcard(entry: str) -> bool:
    return "*" in entry or "?" in entry


def _pattern(entry: str) -> re.Pattern[str]:
    parts = ["(.*)" if ch == "*" else "(.)" if ch == "?" else re.escape(ch) for ch in entry]
    return re.compile("".join(parts))


def fill_template(template: str, series: str, parts: Sequence[str] = ()) -> str:
    """``template`` with ``{x}`` (or ``{s}``) replaced by ``series`` and ``{1}``, ``{2}``... by ``parts``."""

    def sub(m: re.Match[str]) -> str:
        token = m.group(0)
        if token == "{{":
            return "{"
        if token == "}}":
            return "}"
        if token in "{}":
            raise SpecificationError(f"{template!r} has a lone {token!r}: write {token * 2} for a literal brace")
        key = m.group(1).strip()
        if key in ("x", "s"):
            return series
        if key.isdigit() and int(key) >= 1:
            k = int(key)
            if k <= len(parts):
                return parts[k - 1]
            have = f"its wildcard matched {len(parts)} part{'s' * (len(parts) != 1)}" if parts else "it was not matched by a wildcard"
            raise SpecificationError(f"{{{key}}} has nothing for {series}: {have}")
        raise SpecificationError(f"{{{key}}} is not a placeholder: {_PLACEHOLDER_HELP}")

    return _TOKEN.sub(sub, template)


def _has_placeholder(template: str) -> bool:
    return any(m.group(1) is not None for m in _TOKEN.finditer(template))


def _sources(series: str | Iterable[str], names: Sequence[str]) -> tuple[list[tuple[str, tuple[str, ...]]], list[str]]:
    """The series *For each* stands for, each with its wildcard parts, and what could not be matched."""
    out: dict[str, tuple[str, ...]] = {}
    problems: list[str] = []
    have = set(names)
    for entry in split_series(series):
        if _is_wildcard(entry):
            pattern = _pattern(entry)
            found = [(n, pattern.fullmatch(n)) for n in names]
            matched = [(n, m.groups()) for n, m in found if m is not None]
            if not matched:
                problems.append(f"no series match {entry}")
            for n, parts in matched:
                out.setdefault(n, parts)
        elif entry in have:
            out.setdefault(entry, ())
        else:
            problems.append(f"no series {entry}")
    return list(out.items()), problems


@dataclass(frozen=True)
class EachRow:
    """One series to be made: from ``source``, the new ``name`` and its ``formula``. ``status`` is ``"new"``,
    ``"replaces"`` (it exists, and Replace is ticked), ``"exists"`` (it exists: tick Replace) or ``"error"``;
    ``note`` says why."""

    source: str
    name: str
    formula: str
    status: str = "new"
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("new", "replaces")


@dataclass(frozen=True)
class EachPreview:
    """What *For each* would make: a row per new series, and problems with the request as a whole (an entry that
    matches nothing, a template that cannot be filled in)."""

    rows: tuple[EachRow, ...]
    problems: tuple[str, ...] = ()
    data: str = "data"

    @property
    def ok(self) -> bool:
        return not self.problems and bool(self.rows) and all(r.ok for r in self.rows)

    def formulas(self) -> dict[str, str]:
        """New name -> formula, in order: the arguments of ``with_series`` or ``derive``."""
        return {r.name: r.formula for r in self.rows}

    def errors(self) -> list[str]:
        """Why it cannot run (empty when it can), worded for a dialog."""
        out = list(self.problems)
        if not self.rows and not out:
            out.append("nothing to generate: For each names no series")
        bad = [r for r in self.rows if r.status == "error"]
        out += [f"{r.name or r.source}: {r.note}" for r in bad[:5]] + ([f"... and {len(bad) - 5} more"] if len(bad) > 5 else [])
        clash = [r.name for r in self.rows if r.status == "exists"]
        if clash:
            listed = ", ".join(clash[:5]) + (" ..." if len(clash) > 5 else "")
            out.append(f"{self.data} already has {'a series' if len(clash) == 1 else 'series'} {listed}: choose another name, "
                       "or tick Replace (replace=True) to overwrite")
        return out

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        lines = [f"{r.name or '?'} = {r.formula or '?'}" + (f"  [{r.note}]" if r.note else "") for r in self.rows]
        return "\n".join([*lines, *(f"! {p}" for p in self.problems)]) or "(nothing)"


def _names_of(data: Any) -> list[str]:
    if data is None:
        return []
    names = getattr(data, "names", None)
    return list(names) if names is not None else [str(n) for n in data]


def preview_each(data: Any, series: str | Iterable[str], name: str, formula: str, replace: bool = False,
                 derived: bool = False, data_name: str = "data") -> EachPreview:
    """Each new series *For each* would make in ``data`` (a dataset, or its series' names), checked: see the module."""
    names = _names_of(data)
    sources, problems = _sources(series, names)
    if not name.strip():
        problems.append("give a name template, e.g. {x}_g")
    if not formula.strip():
        problems.append("give a formula template, e.g. 100*d(log({x}))")
    if len(sources) > 1 and not _has_placeholder(name):
        problems.append(f"the name {name!r} has no placeholder, so every series would get the same name: e.g. {name}_{{x}}")
    if problems and not sources:
        return EachPreview((), tuple(problems), data_name)
    from ..expr import parse_expression

    have = set(names)
    filled: list[tuple[str, str, str, str]] = []  # source, name, formula, error
    for source, parts in sources:
        new = text = error = ""
        try:
            new = fill_template(name, source, parts)
        except SpecificationError as exc:
            error = str(exc)
        try:
            text = fill_template(formula, source, parts)
        except SpecificationError as exc:
            error = error or str(exc)
        filled.append((source, new, text, error))
    made = {n for _, n, _, e in filled if n and not e}
    seen: dict[str, str] = {}
    rows = []
    for source, new, text, error in filled:
        if not error and not new.isidentifier():
            error = f"{new!r} is not a valid series name (letters, digits and _, not starting with a digit)"
        elif not error and keyword.iskeyword(new):
            error = f"{new!r} is a Python keyword, not a valid series name"
        elif not error and new in seen:
            error = f"made twice (also from {seen[new]}): put {{x}} or a wildcard part in the name"
        if not error:
            try:
                used = parse_expression(text).variables()
            except SpecificationError as exc:
                error = f"the formula does not parse: {exc}"
            else:
                available = have | (made if derived else set())
                missing = sorted(set(used) - available)
                if missing:
                    error = f"no series {', '.join(missing)} in {data_name}"
        if new and not error:
            seen[new] = source
        if error:
            rows.append(EachRow(source, new, text, "error", error))
        elif new in have:
            rows.append(EachRow(source, new, text, "replaces" if replace else "exists",
                                "replaces the existing series" if replace else "already exists: tick Replace to overwrite"))
        else:
            rows.append(EachRow(source, new, text))
    return EachPreview(tuple(rows), tuple(problems), data_name)


def expand_each(series: str | Iterable[str], name: str, formula: str, names: Any, derived: bool = False) -> dict[str, str]:
    """New name -> formula for each series *For each* stands for among ``names`` (a dataset or its series' names):
    ``expand_each("gdp cons", "{x}_g", "100*d(log({x}))", data)`` -> ``{"gdp_g": "100*d(log(gdp))", ...}``. Raises
    ``SpecificationError`` saying what is wrong; names that exist already are allowed (whether to replace them is
    the caller's choice). With ``derived``, a formula may use another of the new series."""
    found = preview_each(names, series, name, formula, replace=True, derived=derived)
    if not found.ok:
        raise SpecificationError("; ".join(found.errors()))
    return found.formulas()
