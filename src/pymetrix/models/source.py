"""Models written as text, in pymetrix's own syntax, Dynare's, IRIS's or EViews's: checked as typed, built into a model.

    src = ModelSource(text)                      # pymetrix's syntax (below); ModelSource(text, "dynare") for a .mod,
                                                 # "iris" for a .model, "eviews" for a model's text or a .prg
                                                 # (pymetrix.models.eviews)
    src.check()                                  # problems with their line and column, the symbols and their roles
    nk = src.build()                             # a Template (pymetrix syntax) or a DynareModel (Dynare syntax)
    src.algebraic(); src.latex()                 # the equations in algebraic notation, or as LaTeX
    src.check_solution()                         # the steady state, the roots and the Blanchard-Kahn conditions
    src.check_solution().irf(20)                 # when unique: the impulse responses (the base regime's with regimes)

The text is what is kept: a project saves it as written, in its syntax, and builds the model from it when it opens,
so comments, layout and descriptions survive. ``check`` is cheap enough to run on every keystroke: it reads the text,
parses each equation and looks up each name, without solving anything.

pymetrix's syntax: sections, each a keyword on its own line, then one entry a line; ``#`` starts a comment, and a
description in double quotes may end an entry.

    parameters
        beta = 0.99          "discount factor"
        r_ss = 1/beta - 1    # expressions of the parameters above
        rho = 0.8  bounds 0 1  prior beta(0.7, 0.1)  symbol ρ   "policy smoothing"   # for estimation (below)
    shocks
        ey = 0.5             "demand"          # standard deviation (default 1)
        ei = 0.2  prior invgamma1(0.2, 2)      # a prior on its size, sd_ei
        corr ey, ei = 0.3  prior uniform(-0.9, 0.9)   # their correlation (estimated as corr_ey_ei)
    exogenous
        g = 0                "spending"        # steady-state value
    variables
        y = 0                "output gap"      # optional: descriptions and steady-state guesses
    equations
        IS: y = y[+1] - (1/sigma)*(i - pi[+1]) + ey
        policy: i = max(rho*i[-1] + (1 - rho)*phi*pi + ei, i_floor)
    measurement
        infl = 4*pi + pi_ss  error 0.2  "inflation, annual"
        growth = g + 4*(y - y[-1])  from 400*dlog(gdp)  "GDP growth"   # made from a raw series, and rebuilt into it
    reported
        r_real = 4*(i - pi)  "real rate, annual"   # reported with the smoothed history and forecasts, never fitted
    regimes
        2009Q1 post: phi = 1.2                    "post-crisis rule"   # from 2009Q1 on, for good
        2025Q1..2026Q4 peg: rho = 1, phi = 0      # a while; the regime before it resumes after 2026Q4
        2030Q1 alt: policy: i = 0.9*i[-1] + 0.2*pi + ei     # an equation replaced, by its name
        2031Q1 hawk surprise: phi = 2; policy: i = rho*i[-1] + (1 - rho)*phi*pi + ei   # unexpected; both at once

A parameter or shock may carry what estimation needs, after its value: ``bounds low high``, ``prior shape(mean, sd)``
(normal, beta, gamma, invgamma1, invgamma2, uniform(low, high)), ``start value``, ``fixed`` (held at its value) and
``symbol φ_π`` (its symbol in algebraic notation); a shock's apply to its size, ``sd_<shock>``. They become the model's
``Param`` specs, so ``estimate()`` needs no priors of its own.

Each equation determines the variable on its left (alone, or in an expression such as ``1/c``); ``x[-1]`` is a lag and ``x[+1]`` a lead, and an equation may run
over several lines while a bracket is open. ``max`` and ``min`` are occasionally binding constraints (``occbin``).

A correlation of two shocks is a line ``corr a, b = r`` in ``shocks`` (any order of the two; ``r`` in (-1, 1), all of them
together a valid correlation matrix), with estimation settings after it as a shock's, on ``corr_<a>_<b>`` (the two in
the order the shocks are declared); they become the template's ``shock_corr``.

Regimes (``pymetrix.models.regimes``), one a line: a start (or ``start..end``, a temporary regime), optionally a name
(letters, digits, ``_`` and ``.``) and ``surprise`` (``expected=False``), a colon, then parameter values ``name = value,
...`` (numbers, or expressions in the parameters) and equations replaced by name, ``label: lhs = rhs``, separated by
``;``; a description in quotes may end it. They are checked as typed (periods, order, overlaps, names) and become the
template's ``regimes``: ``solve()`` gives a solution whose ``solve(data, span)`` follows them, ``estimate`` uses those
that start inside the sample (their values estimated as ``<parameter>@<regime>``) and keeps those after it for the
fit's ``forecast``, and ``regime_filter`` filters through them. A parameter that only a regime's equation uses is
given to that regime with its declared value.

``ModelSource.completions`` and ``ModelSource.hover`` give the model editor its completion lists and hover help: the
names the text declares, the section and option keywords, and the functions with their signatures.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

from .._api import api
from ..errors import SpecificationError
from ..output import Table
from ..expr.functions import EXPANDING as _EXPANDING
from ..expr.functions import FUNCTIONS as _EXPRESSION_FUNCTIONS
from .iris import _EQUATION_NAME as _IRIS_EQUATION_NAME
from .iris import _KNOWN as _IRIS_SECTIONS
from .iris import _declared_at as _iris_declared_at
from .iris import _sections as _iris_sections
from .iris import _strip_comments as _iris_strip_comments
from .safe_eval import safe_eval
from .solvers import ModelError

__all__ = ["Diagnostic", "ModelSource", "SolutionCheck", "SourceCheck", "Symbol", "check_solution"]

Syntax = Literal["pymetrix", "dynare", "iris", "eviews"]
SYNTAXES: tuple[str, ...] = ("pymetrix", "dynare", "iris", "eviews")
_SECTIONS = ("parameters", "shocks", "exogenous", "variables", "equations", "measurement", "reported", "regimes")
_FUNCTIONS = frozenset({"log", "exp", "sqrt", "abs", "sign", "max", "min", "step", "not", "d", "dlog", "pch",
                        "steady_state", "STEADY_STATE"}) | _EXPRESSION_FUNCTIONS  # distributions, erf, erfc, moving windows


@dataclass(frozen=True)
class Diagnostic:
    """A problem in the text: where it is (line and column from 1), what it is, and how bad."""

    line: int
    column: int
    message: str
    severity: Literal["error", "warning", "note"] = "error"
    length: int = 1  # characters to underline

    def __str__(self) -> str:
        return f"line {self.line}, column {self.column}: {self.severity}: {self.message}"


@dataclass(frozen=True)
class Symbol:
    """A name the model declares or determines: its role, value, description and the lines that use it."""

    name: str
    role: Literal["endogenous", "exogenous", "shock", "parameter", "observable", "reported", "local", "correlation"]
    value: float | None = None
    description: str = ""
    line: int = 0  # where it is declared, or the equation that determines it
    used: tuple[int, ...] = ()  # lines of the equations that use it
    estimation: str = ""  # what estimation is told about it: prior, bounds, start, fixed, symbol


@dataclass(frozen=True, eq=False)
class SourceCheck:
    """What ``ModelSource.check`` found: problems, symbols, and the model's shape."""

    syntax: str
    diagnostics: tuple[Diagnostic, ...]
    symbols: tuple[Symbol, ...]
    equations: int
    max_lead: int = 0
    max_lag: int = 0
    constraints: tuple[str, ...] = ()  # equations with max or min

    @property
    def ok(self) -> bool:
        """No errors (warnings and notes allowed)."""
        return not any(d.severity == "error" for d in self.diagnostics)

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity == "error")

    def role(self, role: str) -> tuple[str, ...]:
        return tuple(s.name for s in self.symbols if s.role == role)

    def summary(self) -> Table:
        endo = self.role("endogenous")
        rows = [[s.name, s.role, s.value, s.description, ", ".join(map(str, s.used)), s.estimation] for s in self.symbols]
        notes = [f"{self.syntax} syntax; {self.equations} equations for {len(endo)} endogenous variables; "
                 f"{len(self.role('shock'))} shocks, {len(self.role('exogenous'))} exogenous, {len(self.role('parameter'))} parameters",
                 f"Leads up to {self.max_lead}, lags up to {self.max_lag}"
                 + (f"; occasionally binding constraints in {', '.join(self.constraints)}" if self.constraints else ""),
                 "No errors" if self.ok else f"{len(self.errors)} error(s): see Problems"]
        return Table(["symbol", "role", "value", "description", "used on lines", "estimation"], rows, title="Model source", notes=notes)

    def problems(self) -> Table:
        rows = [[d.line, d.column, d.severity, d.message] for d in self.diagnostics]
        return Table(["line", "column", "severity", "problem"], rows, title="Problems")

    def _repr_html_(self) -> str:
        return self.problems()._repr_html_() + self.summary()._repr_html_()

    def __repr__(self) -> str:
        counts = {s: sum(d.severity == s for d in self.diagnostics) for s in ("error", "warning")}
        return f"<SourceCheck {self.syntax}: {self.equations} equations, {counts['error']} errors, {counts['warning']} warnings>"


@dataclass(frozen=True, eq=False)
class SolutionCheck:
    """What ``ModelSource.check_solution`` found: the steady state, the roots of the linearised model and whether they
    give a unique stable solution (Blanchard-Kahn: as many stable roots as predetermined variables)."""

    verdict: Literal["unique", "no stable solution", "indeterminate", "no steady state", "not solved"]
    message: str
    steady: Mapping[str, float] | None = None  # the endogenous variables' steady-state values
    roots: tuple[complex, ...] = ()  # generalised eigenvalues; infinite ones as inf
    predetermined: int = 0
    labels: Mapping[str, str] = field(default_factory=dict)
    solution: Any = field(default=None, repr=False)  # the LinearSolution when unique (of the base regime with regimes)
    shock_sd: Mapping[str, float] = field(default_factory=dict, repr=False)  # the model's shock sizes, for ``irf``
    shock_corr: Mapping[Any, float] = field(default_factory=dict, repr=False)  # and their correlations
    regimes: int = 0  # the model's regimes; the solution (and ``irf``) is the base regime's

    @property
    def ok(self) -> bool:
        return self.verdict == "unique"

    @property
    def shocks(self) -> tuple[str, ...]:
        return tuple(self.solution.shocks) if self.solution is not None else ()

    def irf(self, periods: int = 20) -> Any:
        """Responses to one-standard-deviation shocks (the model's sizes; one unit where it gives none), correlated
        shocks orthogonalised as ``LinearSolution.irf``: the solution's ``irf`` with the model's ``shock_sd`` and
        ``shock_corr``. Only when the verdict is "unique"; with regimes, the base regime's responses."""
        if self.solution is None:
            raise SpecificationError(f"no responses: the model has no unique stable solution ({self.verdict})")
        return self.solution.irf(periods, shock_sd=dict(self.shock_sd) or None, shock_corr=dict(self.shock_corr) or None)

    @property
    def stable(self) -> int:
        """Roots inside the unit circle (unit roots, within 1e-6, counted as stable, as the solver does)."""
        return sum(abs(r) < 1 + 1e-6 for r in self.roots)

    def summary(self) -> Table:
        rows = [[k, v, self.labels.get(k, "")] for k, v in (self.steady or {}).items()]
        notes = [self.message]
        if self.roots:
            notes.append(f"{self.stable} stable roots for {self.predetermined} predetermined variables; "
                         f"{sum(not np.isfinite(abs(r)) for r in self.roots)} infinite")
        return Table(["variable", "steady state", "description"], rows, title=f"Solution check: {self.verdict}", notes=notes)

    def roots_table(self) -> Table:
        finite = sorted((r for r in self.roots if np.isfinite(abs(r))), key=abs)
        rows = [[i + 1, float(abs(r)), float(r.real), float(r.imag), "stable" if abs(r) < 1 + 1e-6 else "explosive"]
                for i, r in enumerate(finite)]
        return Table(["", "modulus", "real", "imaginary", ""], rows, title="Roots of the linearised model")

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_() + (self.roots_table()._repr_html_() if self.roots else "")

    def __repr__(self) -> str:
        return f"<SolutionCheck: {self.verdict}>"


@dataclass(frozen=True)
class ModelSource:
    """A model as text in a syntax: see the module docstring."""

    text: str
    syntax: Syntax = "pymetrix"
    name: str = "model"

    def __post_init__(self) -> None:
        if self.syntax not in SYNTAXES:
            raise SpecificationError(f"syntax is one of {', '.join(SYNTAXES)}, not {self.syntax!r}")

    def check(self) -> SourceCheck:
        """The problems in the text, with their places, and the symbols: without solving the model."""
        return {"pymetrix": _check_pymetrix, "dynare": _check_dynare, "iris": _check_iris, "eviews": _check_eviews}[self.syntax](self.text)

    @api("pymetrix:model_source.build")
    def build(self) -> Any:
        """The model the text describes: a ``Template`` (pymetrix syntax: the model, its shocks and their sizes, the
        measurement equations, guesses and descriptions), a ``DynareModel`` (Dynare syntax, as ``read_mod``) or a
        ``Template`` (IRIS syntax, as ``read_iris``: parameter values in ``!parameters``, shock sizes as ``std_<shock>``), or a
        ``Template`` (EViews syntax, as ``read_eviews``: model text or a program; its shocks are its ``@innov``s).
        Refused, with the first problems, when the text has errors."""
        found = self.check()
        if not found.ok:
            raise ModelError("the model text has errors: " + "; ".join(str(d) for d in found.errors[:3]))
        if self.syntax == "dynare":
            from .dynare import read_mod

            return read_mod(self.text)
        if self.syntax == "iris":
            from .iris import read_iris

            return read_iris(self.text, name=self.name)
        if self.syntax == "eviews":
            from .eviews import read_eviews

            return read_eviews(self.text, name=self.name)
        return _build_pymetrix(_understand(self.text), self.name)

    def export(self, syntax: Syntax) -> Any:
        """The model in another syntax (a ``Translation``: the text, notes on what did not carry across, and whether it
        solves the same): built from this text, then written. See ``pymetrix.models.writers``."""
        from .writers import translate

        return translate(self, syntax)

    def algebraic(self, style: Literal["unicode", "latex"] = "unicode") -> Any:
        """The equations in algebraic notation (``AlgebraicEquation``s: name, variable, text), the measurement
        equations last: see ``pymetrix.models.writers.algebraic_equations``."""
        from .writers import algebraic_equations

        return algebraic_equations(self.build(), style)

    def latex(self) -> str:
        """The equations as LaTeX ``align`` blocks, each tagged with its name."""
        from .writers import write_latex

        return write_latex(self.build())

    def check_solution(self) -> SolutionCheck:
        """Whether the model solves: its steady state, then the roots of the model linearised there and the
        Blanchard-Kahn conditions. The roots are found whether or not the conditions hold, to see which fail."""
        return check_solution(self.build())

    def completions(self, line: int, column: int, found: SourceCheck | None = None) -> list[str]:
        """What may complete the word before the cursor, at ``line`` and ``column`` (from 1; the cursor before that
        column's character): the names the text declares (``found``, a check of it, or a new one), then the
        functions, then the syntax's keywords, those that start with the word typed so far (any case). None inside
        a comment, a description or a number."""
        lines = self.text.split("\n")
        if not 1 <= line <= len(lines):
            return []
        before = lines[line - 1][:max(column - 1, 0)]
        if _in_comment(self.text, line, len(before), self.syntax):
            return []
        word = re.search(_WORD[self.syntax] + "$", before)
        prefix = word.group(0) if word else ""
        if not prefix and before[-1:].isalnum() or word and before[:word.start()][-1:].isdigit():
            return []  # a number, or 1e-3
        out = [w for w in _vocabulary(self, found or self.check()) if w.lower().startswith(prefix.lower()) and w != prefix]
        return out

    def hover(self, line: int, column: int, found: SourceCheck | None = None) -> str:
        """What the name or function at ``line`` and ``column`` (from 1) is: a symbol's role, value, description,
        estimation settings and the lines that use it; a function's signature and what it does; empty for anything
        else (and inside comments)."""
        lines = self.text.split("\n")
        if not 1 <= line <= len(lines) or _in_comment(self.text, line, column, self.syntax):
            return ""
        for m in re.finditer(_WORD[self.syntax], lines[line - 1]):
            if m.start() < column <= m.end():
                word = m.group(0).lower() if self.syntax == "eviews" else m.group(0)
                return _hover_text(word, lines[line - 1][m.end():].lstrip().startswith("("), self.syntax, found or self.check())
        return ""


def check_solution(obj: Any) -> SolutionCheck:
    """``ModelSource.check_solution`` for a built model (a template, a Dynare model or a Model with leads)."""
    from .dynare import DynareModel

    if isinstance(obj, DynareModel):
        model, shocks, labels = obj.model, list(obj.exogenous), {}
        sizes = {"shock_sd": dict(obj.shock_sd), "shock_corr": dict(obj.shock_corr)}
    elif hasattr(obj, "model") and hasattr(obj, "shocks"):  # a Template
        model, shocks, labels = obj.model, list(obj.shocks), dict(getattr(obj, "labels", {}) or {})
        sizes = {"shock_sd": dict(obj.shock_sd), "shock_corr": dict(getattr(obj, "shock_corr", {}) or {}),
                 "regimes": len(getattr(obj, "regimes", ()) or ())}
    else:
        model, shocks, labels, sizes = obj, None, {}, {}
    failures = (ModelError, SpecificationError, ValueError, ArithmeticError, np.linalg.LinAlgError)
    try:
        ss = obj.steady_state()
    except failures as exc:
        return SolutionCheck("no steady state", f"no steady state found: {exc}", labels=labels)
    steady = {v: float(ss[v]) for v in model.endogenous}
    try:
        lin = model.linearize(ss, shocks=shocks) if shocks is not None else model.linearize(ss)
        roots, m = lin.roots()
    except failures as exc:
        return SolutionCheck("not solved", f"the steady state is found, but the model could not be linearised there: {exc}",
                             steady, labels=labels)
    found = tuple(complex(r) for r in roots)
    try:
        solution = lin.solve()
    except ModelError as exc:
        stable = sum(abs(r) < 1 + 1e-6 for r in found)
        verdict = "no stable solution" if stable < m else "indeterminate" if stable > m else "not solved"
        return SolutionCheck(verdict, str(exc), steady, found, m, labels)
    return SolutionCheck("unique", f"a unique stable solution: {m} stable roots for {m} predetermined variables "
                         "(Blanchard-Kahn)", steady, found, m, labels, solution, **sizes)


# ---------------------------------------------------------------------------------------------
# pymetrix's syntax


@dataclass
class _Entry:
    section: str
    text: str
    line: int
    column: int  # of the text's first character
    description: str = ""
    unclosed: bool = False  # a bracket never closed (reported once, not again as a parse error)


@dataclass
class _Read:
    entries: list[_Entry] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)


def _strip_comment(line: str) -> str:
    quoted = False
    for i, ch in enumerate(line):
        if ch == '"':
            quoted = not quoted
        elif ch == "#" and not quoted:
            return line[:i]
    return line


def _read_pymetrix(text: str) -> _Read:
    """The entries of each section, with their places; equations joined over lines while a bracket is open."""
    out = _Read()
    section: str | None = None
    pending: _Entry | None = None
    depth = 0
    for number, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw)
        if not line.strip():
            continue
        starts_entry = bool(re.match(r"^\s*[A-Za-z_]\w*\s*(:(?!=)|=(?!=))", line)) or line.strip().rstrip(":").strip().lower() in _SECTIONS
        if pending is not None and starts_entry:  # a new entry: the bracket was never closed
            out.diagnostics.append(Diagnostic(pending.line, pending.column, "a bracket opened here is never closed", length=len(pending.text)))
            pending.unclosed = True
            out.entries.append(pending)
            pending, depth = None, 0
        if pending is not None:  # an equation still inside a bracket
            pending.text += " " + line.strip()
            depth += line.count("(") + line.count("[") - line.count(")") - line.count("]")
            if depth <= 0:
                out.entries.append(pending)
                pending, depth = None, 0
            continue
        word = line.strip().rstrip(":").strip().lower()
        if word in _SECTIONS and line.strip().rstrip(":").strip().lower() == word:
            section = word
            continue
        column = len(line) - len(line.lstrip()) + 1
        if section is None:
            out.diagnostics.append(Diagnostic(number, column, f"text before any section: start with one of {', '.join(_SECTIONS)}",
                                              length=len(line.strip())))
            continue
        body, description = line.strip(), ""
        m = re.search(r'"([^"]*)"\s*$', body)
        if m:
            body, description = body[:m.start()].rstrip(), m.group(1)
        entry = _Entry(section, body, number, column, description)
        open_ = body.count("(") + body.count("[") - body.count(")") - body.count("]")
        if section == "equations" and open_ > 0:
            pending, depth = entry, open_
            continue
        out.entries.append(entry)
    if pending is not None:
        out.diagnostics.append(Diagnostic(pending.line, pending.column, "a bracket opened here is never closed", length=len(pending.text)))
        pending.unclosed = True
        out.entries.append(pending)
    return out


def _nearest(name: str, known: list[str]) -> str:
    close = difflib.get_close_matches(name, known, n=1, cutoff=0.6)
    return f"; did you mean {close[0]!r}?" if close else ""


def _parse(text: str, line: int, column: int, diagnostics: list[Diagnostic]) -> Any:
    """An expression, or None with a diagnostic at the parser's position."""
    from ..expr.nodes import ExpressionError
    from ..expr.parser import parse_expression

    try:
        return parse_expression(text)
    except ExpressionError as exc:
        m = re.search(r"at position (\d+)", str(exc))
        at = int(m.group(1)) if m else 0
        message = re.sub(r" at position \d+ in .*$", "", str(exc))
        diagnostics.append(Diagnostic(line, column + at, message))
        return None


def _names(expr: Any) -> list[Any]:
    from ..expr.nodes import Name

    return [n for n in expr.walk() if isinstance(n, Name)]


def _calls(expr: Any) -> set[str]:
    from ..expr.nodes import Call

    return {n.fn for n in expr.walk() if isinstance(n, Call)}


@dataclass
class _Parsed:
    """pymetrix-syntax text understood: what ``check`` reports and ``build`` builds from."""

    params: dict[str, float] = field(default_factory=dict)
    shocks: dict[str, float] = field(default_factory=dict)
    exogenous: dict[str, float] = field(default_factory=dict)
    guesses: dict[str, float] = field(default_factory=dict)
    descriptions: dict[str, str] = field(default_factory=dict)
    lines: dict[str, int] = field(default_factory=dict)
    columns: dict[str, int] = field(default_factory=dict)  # where each declaration's name starts
    equations: list[tuple[str, str, str, int]] = field(default_factory=list)  # label, lhs name, rhs text, line
    measurement: list[tuple[str, str, float, str]] = field(default_factory=list)  # observable, expression, error sd, label
    transforms: dict[str, str] = field(default_factory=dict)  # observable: its transform of raw data series
    specs: dict[str, dict[str, Any]] = field(default_factory=dict)  # name (sd_<shock> for shocks): Param fields given
    reported: list[tuple[str, str, str, int]] = field(default_factory=list)  # name, expression, label, line
    used: dict[str, set[int]] = field(default_factory=dict)
    max_lead: int = 0
    max_lag: int = 0
    constraints: list[str] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    unparsed: set[str] = field(default_factory=set)  # left-hand sides of equations that did not parse
    lhs: dict[str, str] = field(default_factory=dict)  # left-hand sides that are expressions (1/c = ...), by variable
    corr: dict[tuple[str, str], float] = field(default_factory=dict)  # correlations of the shocks, in their order
    corr_lines: dict[tuple[str, str], tuple[int, int, int]] = field(default_factory=dict)  # line, column, length
    regimes: list[Any] = field(default_factory=list)  # Regime objects, in order
    regime_lines: list[tuple[int, int, Any]] = field(default_factory=list)  # each regime's line, column and head (a match)


def _understand(text: str) -> _Parsed:
    from ..expr.evaluate import evaluate

    read = _read_pymetrix(text)
    p = _Parsed(diagnostics=list(read.diagnostics))
    declared: dict[str, str] = {}  # name -> section

    def declare(e: _Entry, name: str) -> bool:
        if not name.isidentifier():
            p.diagnostics.append(Diagnostic(e.line, e.column, f"{name!r} is not a name", length=max(1, len(name))))
            return False
        if name in declared:
            p.diagnostics.append(Diagnostic(e.line, e.column, f"{name} is declared twice (also in {declared[name]}, line {p.lines[name]})",
                                            length=len(name)))
            return False
        declared[name] = e.section
        p.lines[name] = e.line
        p.columns[name] = e.column
        if e.description:
            p.descriptions[name] = e.description
        return True

    def value_of(e: _Entry, expr_text: str, at: int) -> float | None:
        expr = _parse(expr_text, e.line, at, p.diagnostics)
        if expr is None:
            return None
        unknown = [n.id for n in _names(expr) if n.id not in p.params]
        if unknown:
            p.diagnostics.append(Diagnostic(e.line, at, f"{unknown[0]} has no value yet: a value may use the parameters above it"
                                            + _nearest(unknown[0], list(p.params)), length=len(expr_text)))
            return None
        try:
            return float(evaluate(expr, lambda name, lag: p.params[name]))
        except Exception as exc:  # noqa: BLE001 - arithmetic on the text's own numbers
            p.diagnostics.append(Diagnostic(e.line, at, f"cannot evaluate {expr_text!r}: {exc}", length=len(expr_text)))
            return None

    def settings(e: _Entry, rest: str, at: int, spec: str) -> str | None:
        """The value's text with the estimation settings after it taken off (into ``p.specs[spec]``); None when a
        setting does not read (the problem placed). ``at``: the column of ``rest``."""
        options_at = re.search(r"\s(?:bounds|prior|start|fixed|symbol)\b", " " + rest)
        stray = re.match(r"^(\S+)\s+([A-Za-z_]\w*)", rest)
        if options_at is None and stray and difflib.get_close_matches(stray.group(2), _OPTION_WORDS, 1, 0.6):
            p.diagnostics.append(Diagnostic(e.line, at + stray.start(2), f"{stray.group(2)!r} does not read here: after a value come "
                                            "bounds low high, prior shape(mean, sd), start value, fixed, symbol S"
                                            + _nearest(stray.group(2), list(_OPTION_WORDS)), length=len(stray.group(2))))
            return None
        if options_at is not None:
            cut = max(options_at.start() - 1, 0)
            given = _options(rest[cut:], e.line, at + cut, p.diagnostics)
            if given is not None:
                p.specs[spec] = given
            rest = rest[:cut].strip()
        return rest

    correlations: list[_Entry] = []
    for e in read.entries:
        if e.section == "shocks" and re.match(r"corr\b(?!\s*=)", e.text):
            correlations.append(e)  # read once every shock is declared
            continue
        if e.section in ("parameters", "shocks", "exogenous", "variables"):
            name, eq, rest = (s.strip() for s in e.text.partition("="))
            if not declare(e, name):
                continue
            if eq and e.section in ("parameters", "shocks"):  # estimation settings after the value
                after = e.text.partition("=")[2]
                taken = settings(e, rest, e.column + e.text.index("=") + 1 + len(after) - len(after.lstrip()),
                                 name if e.section == "parameters" else f"sd_{name}")
                if taken is None:
                    continue
                rest = taken
            at = e.column + e.text.index("=") + 1 + (len(e.text.partition("=")[2]) - len(e.text.partition("=")[2].lstrip())) if eq else e.column
            if e.section == "parameters":
                if not eq:
                    p.diagnostics.append(Diagnostic(e.line, e.column, f"parameter {name} needs a value: {name} = ...", length=len(name)))
                    continue
                v = value_of(e, rest, at)
                if v is not None:
                    p.params[name] = v
            else:
                v = value_of(e, rest, at) if eq else None
                target = {"shocks": p.shocks, "exogenous": p.exogenous, "variables": p.guesses}[e.section]
                target[name] = v if v is not None else (1.0 if e.section == "shocks" else 0.0)
    for e in correlations:
        _correlation(e, p, settings, value_of)
    _correlations_valid(p)
    for e in read.entries:
        if e.section == "equations":
            m = re.match(r"^([A-Za-z_]\w*)\s*:(?!=)\s*", e.text)
            label, body, offset = (m.group(1), e.text[m.end():], m.end()) if m else ("", e.text, 0)
            lhs, eq, rhs = body.partition("=")
            if not eq:
                p.diagnostics.append(Diagnostic(e.line, e.column + offset, "an equation needs '=': variable = expression",
                                                length=len(body)))
                continue
            sink = [] if e.unclosed else p.diagnostics  # an unclosed bracket is reported already
            lhs_expr = _parse(lhs.strip(), e.line, e.column + offset + len(lhs) - len(lhs.lstrip()), sink)
            rhs_at = e.column + offset + len(lhs) + 1 + len(rhs) - len(rhs.lstrip())
            rhs_expr = _parse(rhs.strip(), e.line, rhs_at, sink)
            if lhs_expr is None or rhs_expr is None:  # still: the variable it means to determine, and the names it uses
                if lhs.strip().isidentifier():
                    p.unparsed.add(lhs.strip())
                for m2 in re.finditer(r"[A-Za-z_]\w*", body):
                    p.used.setdefault(m2.group(0), set()).add(e.line)
                continue
            from ..expr.nodes import Name

            if isinstance(lhs_expr, Name) and not lhs_expr.lag:
                dependent = lhs_expr.id
            else:  # an expression in one variable of the current period, as 1/c or log(y): that variable
                current = sorted({n.id for n in _names(lhs_expr) if n.lag == 0 and n.id not in p.params and n.id not in p.shocks
                                  and n.id not in p.exogenous})
                if len(current) != 1:
                    p.diagnostics.append(Diagnostic(e.line, e.column + offset, "the left-hand side holds the variable the equation "
                                                    "determines, in the current period, and no other (rearrange the equation)",
                                                    length=len(lhs.strip())))
                    continue
                dependent = current[0]
                p.lhs[dependent] = lhs.strip()
            if dependent in {d for _, d, _, _ in p.equations}:
                first = next(ln for _, d, _, ln in p.equations if d == dependent)
                p.diagnostics.append(Diagnostic(e.line, e.column + offset, f"{dependent} is already determined by the equation on line {first}",
                                                length=len(dependent)))
                continue
            if dependent in declared and declared[dependent] != "variables":
                p.diagnostics.append(Diagnostic(e.line, e.column + offset, f"{dependent} is declared in {declared[dependent]}: "
                                                "an equation's left-hand side is an endogenous variable", length=len(dependent)))
                continue
            p.equations.append((label or dependent, dependent, rhs.strip(), e.line))
            if e.description:
                p.descriptions.setdefault(dependent, e.description)
            p.lines.setdefault(dependent, e.line)
            for n in _names(rhs_expr) + _names(lhs_expr):
                p.used.setdefault(n.id, set()).add(e.line)
                p.max_lead, p.max_lag = max(p.max_lead, n.lag), max(p.max_lag, -n.lag)
            if {"max", "min"} & _calls(rhs_expr):
                p.constraints.append(label or dependent)
    endogenous = {d for _, d, _, _ in p.equations}
    known = sorted(endogenous | set(declared) | p.unparsed)
    for e in read.entries:  # names an equation uses that nothing declares or determines
        if e.section != "equations":
            continue
        for m in re.finditer(r"[A-Za-z_]\w*", re.sub(r'"[^"]*"', "", e.text)):
            name = m.group(0)
            after = e.text[m.end():].lstrip()
            if name in known or name in _FUNCTIONS and after.startswith("(") or after.startswith(":") and m.start() == 0:
                continue
            if re.fullmatch(r"e\d*", name) and m.start() > 0 and e.text[m.start() - 1].isdigit():
                continue  # 1e-3
            p.diagnostics.append(Diagnostic(e.line, e.column + m.start(), f"{name} is not declared or determined by an equation"
                                            + _nearest(name, known), length=len(name)))
    for name in [n for n, s in declared.items() if s == "variables" and n not in endogenous]:
        p.diagnostics.append(Diagnostic(p.lines[name], 1, f"{name} is listed as a variable but no equation determines it", length=len(name)))
    for e in read.entries:
        if e.section in ("measurement", "reported"):
            measured = e.section == "measurement"
            pattern = (r"^([A-Za-z_]\w*)\s*=\s*(.*?)(?:\s+from\s+(.*?))?(?:\s+error\s+([0-9.eE+-]+))?$" if measured
                       else r"^([A-Za-z_]\w*)\s*=\s*(.*)$")
            m = re.match(pattern, e.text)
            if not m:
                p.diagnostics.append(Diagnostic(e.line, e.column, "a measurement equation is: observed = expression [from transform] "
                                                "[error sd]" if measured else "a reported series is: name = expression",
                                                length=len(e.text)))
                continue
            what = "a measurement equation" if measured else "a reported series"
            expr = _parse(m.group(2), e.line, e.column + m.start(2), p.diagnostics)
            if expr is None:
                continue
            bad = [n.id for n in _names(expr) if n.id not in endogenous and n.id not in p.params]
            if bad:
                p.diagnostics.append(Diagnostic(e.line, e.column + m.start(2), f"{bad[0]} in {what} is not a model "
                                                "variable or parameter" + _nearest(bad[0], sorted(endogenous | set(p.params))),
                                                length=len(m.group(2))))
                continue
            leads = sorted({str(n) for n in _names(expr) if n.lag > 0})
            if leads:
                p.diagnostics.append(Diagnostic(e.line, e.column + m.start(2), f"{what} uses the present and the past, not {leads[0]}",
                                                length=len(m.group(2))))
                continue
            if measured and m.group(3) is not None:
                if _parse(m.group(3), e.line, e.column + m.start(3), p.diagnostics) is None:
                    continue
                p.transforms[m.group(1)] = m.group(3).strip()
            if measured:
                p.measurement.append((m.group(1), m.group(2), float(m.group(4) or 0.0), e.description))
            else:
                p.reported.append((m.group(1), m.group(2), e.description, e.line))
            for n in _names(expr):
                p.used.setdefault(n.id, set()).add(e.line)
    for e in read.entries:
        if e.section == "regimes":
            _regime(e, p, value_of)
    _regime_order(p)
    for kind, names in (("parameter", p.params), ("shock", p.shocks)):
        for name in names:
            if name not in p.used:
                p.diagnostics.append(Diagnostic(p.lines[name], p.columns.get(name, 1), f"{kind} {name} is not used", "warning",
                                                length=len(name)))
    if not p.equations and not any(d.severity == "error" for d in p.diagnostics):
        p.diagnostics.append(Diagnostic(1, 1, "no equations yet: add an equations section", "note"))
    return p


def _correlation(e: _Entry, p: _Parsed, settings: Any, value_of: Any) -> None:
    """``corr a, b = r [settings]`` in ``shocks``: checked (two different shocks, each pair once, -1 < r < 1) into
    ``p.corr``; whether all of them make a valid correlation matrix is checked after (``_correlations_valid``)."""
    from .covariance import corr_pairs

    m = re.match(r"^corr\s+(\S+?)\s*,\s*(\S+?)\s*=\s*(.*)$", e.text)
    if not m:
        p.diagnostics.append(Diagnostic(e.line, e.column, "a correlation of two shocks is: corr a, b = value", length=len(e.text)))
        return
    shocks = list(p.shocks)
    for g in (1, 2):
        if m.group(g) not in p.shocks:
            p.diagnostics.append(Diagnostic(e.line, e.column + m.start(g), f"{m.group(g)} is not a shock (declared in shocks)"
                                            + _nearest(m.group(g), shocks), length=len(m.group(g))))
            return
    a, b = m.group(1), m.group(2)
    if a == b:
        p.diagnostics.append(Diagnostic(e.line, e.column + m.start(2), f"a correlation is between two different shocks: "
                                        f"{a}'s size is its value in shocks", length=len(b)))
        return
    pair = (a, b) if shocks.index(a) < shocks.index(b) else (b, a)
    if pair in p.corr:
        p.diagnostics.append(Diagnostic(e.line, e.column, f"the correlation of {a} and {b} is given twice (also on line "
                                        f"{p.corr_lines[pair][0]})", length=len(e.text)))
        return
    name = f"corr_{pair[0]}_{pair[1]}"
    if name not in corr_pairs(shocks):
        p.diagnostics.append(Diagnostic(e.line, e.column, f"{name} would name two pairs of shocks: rename one of them (without _)",
                                        length=len(e.text)))
        return
    rest = settings(e, m.group(3), e.column + m.start(3), name)
    if rest is None:
        return
    value = value_of(e, rest, e.column + m.start(3))
    if value is None:
        return
    if not -1 < value < 1:
        p.diagnostics.append(Diagnostic(e.line, e.column + m.start(3), f"a correlation is between -1 and 1 (not either): {value:g}",
                                        length=max(len(rest), 1)))
        return
    p.corr[pair] = value
    p.corr_lines[pair] = (e.line, e.column, len(e.text))
    p.lines[name], p.columns[name] = e.line, e.column
    if e.description:
        p.descriptions[name] = e.description


def _correlations_valid(p: _Parsed) -> None:
    """All the correlations together a valid correlation matrix (positive definite): if not, the problem at the first
    line from which those so far are not."""
    from .covariance import correlation_matrix

    try:
        correlation_matrix(list(p.shocks), p.corr)
        return
    except ModelError:
        pass
    so_far: dict[tuple[str, str], float] = {}
    for pair in sorted(p.corr, key=lambda q: p.corr_lines[q]):
        so_far[pair] = p.corr[pair]
        try:
            correlation_matrix(list(p.shocks), so_far)
        except ModelError:
            line, column, length = p.corr_lines[pair]
            p.diagnostics.append(Diagnostic(line, column, "with this one, the shocks' correlations are not a valid correlation matrix "
                                            "(not positive definite): " + ", ".join(f"{a}-{b} {r:g}" for (a, b), r in so_far.items()),
                                            length=length))
            return


_REGIME_HEAD = re.compile(r"^(?P<start>[^\s:.]+)(?:\s*\.\.\s*(?P<end>[^\s:]+))?(?:\s+(?P<name>(?!surprise\b)[A-Za-z0-9_.]+))?"
                          r"(?:\s+(?P<surprise>surprise))?\s*:(?!=)")


def _split(text: str, sep: str) -> list[tuple[str, int]]:
    """``text`` split at ``sep`` outside brackets: each piece with its offset."""
    out, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        depth += (ch in "([") - (ch in ")]")
        if ch == sep and depth == 0:
            out.append((text[start:i], start))
            start = i + 1
    out.append((text[start:], start))
    return out


def _period_problem(text: str) -> str:
    import pandas as pd

    try:
        pd.Period(text)
    except (ValueError, TypeError):
        return f"{text!r} is not a period: e.g. 2009Q1, 2009-03 or 2009"
    return ""


def _regime(e: _Entry, p: _Parsed, value_of: Any) -> None:
    """A line of ``regimes``, checked against the model; a ``Regime`` into ``p.regimes`` when it reads."""
    from ..expr.nodes import Name
    from .regimes import Regime

    head = _REGIME_HEAD.match(e.text)
    if head is None:
        p.diagnostics.append(Diagnostic(e.line, e.column, "a regime is: start[..end] [name] [surprise]: parameter = value, ...; "
                                        "equation: lhs = rhs", length=len(e.text)))
        return
    ok = True
    for g in ("start", "end"):
        if head.group(g) and _period_problem(head.group(g)):
            p.diagnostics.append(Diagnostic(e.line, e.column + head.start(g), _period_problem(head.group(g)), length=len(head.group(g))))
            ok = False
    endogenous = {d: (label, line) for label, d, _, line in p.equations}
    labels = {label: d for label, d, _, _ in p.equations}
    params: dict[str, float] = {}
    equations: dict[str, str] = {}
    body_at = head.end()
    pieces = _split(e.text[body_at:], ";")
    if not e.text[body_at:].strip():
        p.diagnostics.append(Diagnostic(e.line, e.column + len(e.text) - 1, "the regime changes nothing: give parameter values "
                                        "(name = value) or an equation replaced (label: lhs = rhs)"))
        return
    assignments: list[tuple[str, int]] = []
    for piece, offset in pieces:
        at = body_at + offset + len(piece) - len(piece.lstrip())
        piece = piece.strip()
        if not piece:
            p.diagnostics.append(Diagnostic(e.line, e.column + at - 1, "nothing between two ';'"))
            ok = False
            continue
        replaced = re.match(r"^([A-Za-z_]\w*)\s*:(?!=)\s*", piece)
        if replaced is None:
            assignments += [(part, at + o) for part, o in _split(piece, ",")]
            continue
        label, text, text_at = replaced.group(1), piece[replaced.end():], at + replaced.end()
        if label not in labels:
            p.diagnostics.append(Diagnostic(e.line, e.column + at, f"{label} is not the name of an equation"
                                            + _nearest(label, list(labels)), length=len(label)))
            ok = False
            continue
        if label in equations:
            p.diagnostics.append(Diagnostic(e.line, e.column + at, f"the equation {label} is replaced twice in this regime", length=len(label)))
            ok = False
            continue
        lhs, eq, rhs = text.partition("=")
        if not eq:
            p.diagnostics.append(Diagnostic(e.line, e.column + text_at, "an equation needs '=': variable = expression", length=max(len(text), 1)))
            ok = False
            continue
        lhs_expr = _parse(lhs.strip(), e.line, e.column + text_at + len(lhs) - len(lhs.lstrip()), p.diagnostics)
        rhs_at = text_at + len(lhs) + 1 + len(rhs) - len(rhs.lstrip())
        rhs_expr = _parse(rhs.strip(), e.line, e.column + rhs_at, p.diagnostics)
        if lhs_expr is None or rhs_expr is None:
            ok = False
            continue
        known = set(endogenous) | set(p.shocks) | set(p.exogenous) | set(p.params)
        strange = False
        for m in re.finditer(r"[A-Za-z_]\w*", text):
            name, after = m.group(0), text[m.end():].lstrip()
            if name in known or name in _FUNCTIONS and after.startswith("("):
                continue
            if re.fullmatch(r"e\d*", name) and m.start() > 0 and text[m.start() - 1].isdigit():
                continue
            p.diagnostics.append(Diagnostic(e.line, e.column + text_at + m.start(), f"{name} is not a variable, shock or parameter "
                                            "of the model" + _nearest(name, sorted(known)), length=len(name)))
            strange = ok = False
        if strange:
            continue
        current = sorted({n.id for n in _names(lhs_expr) if n.lag == 0 and n.id in endogenous})
        dependent = lhs_expr.id if isinstance(lhs_expr, Name) and not lhs_expr.lag else current[0] if len(current) == 1 else None
        if dependent != labels[label]:
            p.diagnostics.append(Diagnostic(e.line, e.column + text_at, f"the equation {label} determines {labels[label]}: its "
                                            f"replacement keeps it on the left-hand side", length=len(lhs.strip()) or 1))
            ok = False
            continue
        equations[label] = f"{lhs.strip()} = {rhs.strip()}"
        for n in _names(lhs_expr) + _names(rhs_expr):
            p.used.setdefault(n.id, set()).add(e.line)
    model_lines = {line for _, _, _, line in p.equations} | {line for line, _, _ in p.regime_lines} | {e.line}
    in_equations = {n for n, lines in p.used.items() if lines & model_lines}  # the model's, earlier regimes' or this one's
    for part, at in assignments:
        lead = len(part) - len(part.lstrip())
        name, eq, value = part.strip().partition("=")
        name = name.strip()
        if not eq or not name.isidentifier():
            p.diagnostics.append(Diagnostic(e.line, e.column + at + lead, "a regime gives parameters values as name = value, ...",
                                            length=max(len(part.strip()), 1)))
            ok = False
            continue
        if name not in p.params:
            p.diagnostics.append(Diagnostic(e.line, e.column + at + lead, f"{name} is not a parameter" + _nearest(name, list(p.params)),
                                            length=len(name)))
            ok = False
            continue
        if name not in in_equations:
            p.diagnostics.append(Diagnostic(e.line, e.column + at + lead, f"{name} is a parameter no equation uses: a regime changes "
                                            "the parameters of the model's equations", length=len(name)))
            ok = False
            continue
        if name in params:
            p.diagnostics.append(Diagnostic(e.line, e.column + at + lead, f"{name} is given twice in this regime", length=len(name)))
            ok = False
            continue
        value_text = value.strip()
        v = value_of(e, value_text, e.column + at + len(part) - len(part.lstrip()) + part.strip().index("=") + 1
                     + len(value) - len(value.lstrip()))
        if v is None:
            ok = False
            continue
        params[name] = v
    if not ok:
        return
    try:
        regime = Regime(head.group("start"), params, equations, name=head.group("name") or "", end=head.group("end"),
                        expected=head.group("surprise") is None, description=e.description)
    except SpecificationError as exc:
        p.diagnostics.append(Diagnostic(e.line, e.column + (head.start("name") if head.group("name") else 0), str(exc),
                                        length=len(head.group("name") or head.group("start"))))
        return
    p.regimes.append(regime)
    p.regime_lines.append((e.line, e.column, head))


def _regime_order(p: _Parsed) -> None:
    """The regimes together: one frequency, in order of their starts, not overlapping a temporary one, named once (as
    ``pymetrix.models.regimes`` refuses them), each problem at the regime it starts with."""
    import pandas as pd

    from .regimes import _phases

    if not p.regimes:
        return
    freq = pd.Period(p.regimes[0].start).freqstr
    kept, places = [], []
    for regime, (line, column, head) in zip(p.regimes, p.regime_lines):
        if pd.Period(regime.start).freqstr != freq:
            p.diagnostics.append(Diagnostic(line, column + head.start("start"), f"{regime.start} is not of the first regime's "
                                            f"frequency ({p.regimes[0].start})", length=len(regime.start)))
            continue
        try:
            _phases([*kept, regime], pd.Period(p.regimes[0].start).freq)
        except SpecificationError as exc:
            message = str(exc)
            g = "end" if "before it starts" in message else "name" if "called" in message and head.group("name") else "start"
            p.diagnostics.append(Diagnostic(line, column + head.start(g), message, length=len(head.group(g))))
            continue
        kept.append(regime)
        places.append((line, column, head))
    p.regimes, p.regime_lines = kept, places


_OPTION_WORDS = ("bounds", "prior", "start", "fixed", "symbol")
_PRIOR_SHAPES = {"normal": "Normal", "beta": "Beta", "gamma": "Gamma", "invgamma": "InvGamma1", "invgamma1": "InvGamma1",
                 "invgamma2": "InvGamma2", "uniform": "Uniform"}


def _options(text: str, line: int, column: int, diagnostics: list[Diagnostic]) -> dict[str, Any] | None:
    """``bounds 0 1  prior beta(0.7, 0.1)  start 0.8  fixed  symbol ρ`` as ``Param`` fields; None, with a problem placed
    where it is, when they do not read."""
    from ..estimation import priors as P

    out: dict[str, Any] = {}
    pattern = re.compile(r"\s*(?:(bounds)\s+(\S+)\s+(\S+)|(prior)\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)|(start)\s+(\S+)|(fixed)\b|(symbol)\s+(\S+))")
    at = 0
    while at < len(text.rstrip()):
        m = pattern.match(text, at)
        if m is None:
            word = text[at:].split()[0] if text[at:].split() else text[at:]
            diagnostics.append(Diagnostic(line, column + at + len(text[at:]) - len(text[at:].lstrip()),
                                          f"{word!r} does not read here: after a value come bounds low high, prior shape(mean, sd), "
                                          "start value, fixed, symbol S", length=max(1, len(word))))
            return None
        try:
            if m.group(1):
                out["bounds"] = (float(m.group(2)), float(m.group(3)))
            elif m.group(4):
                shape = _PRIOR_SHAPES.get(m.group(5).lower())
                if shape is None:
                    diagnostics.append(Diagnostic(line, column + m.start(5), f"{m.group(5)} is not a prior shape: "
                                                  + ", ".join(_PRIOR_SHAPES) + _nearest(m.group(5).lower(), list(_PRIOR_SHAPES)),
                                                  length=len(m.group(5))))
                    return None
                args = [float(a) for a in m.group(6).split(",") if a.strip()]
                out["prior"] = getattr(P, shape)(*args)
            elif m.group(7):
                out["start"] = float(m.group(8))
            elif m.group(9):
                out["fixed"] = True
            else:
                out["symbol"] = m.group(11)
        except (ValueError, TypeError, SpecificationError) as exc:
            diagnostics.append(Diagnostic(line, column + m.start(), f"{m.group(0).strip()!r}: {exc}", length=len(m.group(0).strip())))
            return None
        at = m.end()
    return out


def _estimation_text(given: Mapping[str, Any]) -> str:
    from .system import prior_text

    parts = []
    if "prior" in given:
        parts.append(prior_text(given["prior"]))
    if "bounds" in given:
        parts.append(f"bounds {given['bounds'][0]:g} to {given['bounds'][1]:g}")
    if "start" in given:
        parts.append(f"start {given['start']:g}")
    if given.get("fixed"):
        parts.append("fixed")
    if "symbol" in given:
        parts.append(f"symbol {given['symbol']}")
    return "; ".join(parts)


def _symbols(p: _Parsed) -> tuple[Symbol, ...]:
    out = []
    for _, name, _, line in p.equations:
        out.append(Symbol(name, "endogenous", p.guesses.get(name), p.descriptions.get(name, ""), line, tuple(sorted(p.used.get(name, ())))))
    for role, values in (("shock", p.shocks), ("exogenous", p.exogenous), ("parameter", p.params)):
        for name, value in values.items():
            given = p.specs.get(name if role == "parameter" else f"sd_{name}", {})
            out.append(Symbol(name, role, value, p.descriptions.get(name, ""), p.lines.get(name, 0), tuple(sorted(p.used.get(name, ()))),
                              _estimation_text(given)))
    for (a, b), r in p.corr.items():
        name = f"corr_{a}_{b}"
        out.append(Symbol(name, "correlation", r, p.descriptions.get(name, "") or f"the correlation of {a} and {b}",
                          p.lines.get(name, 0), (), _estimation_text(p.specs.get(name, {}))))
    for observable, _, _, label in p.measurement:
        out.append(Symbol(observable, "observable", None, label))
    for name, _, label, line in p.reported:
        out.append(Symbol(name, "reported", None, label, line))
    return tuple(out)


def _check_pymetrix(text: str) -> SourceCheck:
    p = _understand(text)
    diagnostics = list(p.diagnostics)
    if p.equations and not any(d.severity == "error" for d in diagnostics):
        try:  # the model compiles: what the reading above cannot see (a parameter left out of an equation, ...)
            built = _build_pymetrix(p, "check")
        except (ModelError, SpecificationError) as exc:
            diagnostics.append(Diagnostic(1, 1, f"the model does not compile: {exc}"))
        else:
            diagnostics += _regime_models(built, p)
    order = {"error": 0, "warning": 1, "note": 2}
    diagnostics.sort(key=lambda d: (d.line, d.column, order[d.severity]))
    return SourceCheck("pymetrix", tuple(diagnostics), _symbols(p), len(p.equations), p.max_lead, p.max_lag, tuple(p.constraints))


def _regime_models(template: Any, p: _Parsed) -> list[Diagnostic]:
    """Each regime's model built (its parameters and equations over those before it), as solving it will: a problem
    the line-by-line reading cannot see, at the regime's line."""
    from .regimes import _carried, regime_model

    out, params, equations = [], {}, {}
    carried = _carried(template.regimes)
    for regime, (line, column, _) in zip(template.regimes, p.regime_lines):
        params.update(regime.params)
        equations.update(regime.equations)
        try:
            regime_model(template.model, params, equations, carried)
        except (ModelError, SpecificationError) as exc:
            out.append(Diagnostic(line, column, f"the regime's model does not build: {exc}", length=len(regime.start)))
    return out


def _build_pymetrix(p: Any, name: str) -> Any:
    from ..estimation import Equation
    from ..metadata import Param, Role
    from .measurement import MeasurementMap
    from .model import Model
    from .templates import Template

    eqs = []
    for label, dependent, rhs, _ in p.equations:
        lhs = p.lhs.get(dependent, dependent)
        used = [q for q in p.params if re.search(rf"\b{re.escape(q)}\b", lhs + " " + rhs)]
        text = f"{lhs} {'~' if used else '='} {rhs}"
        eqs.append(Equation(text, params=used, name=label) if used else Equation(text, name=label))
    roles = {**{s: Role.SHOCK for s in p.shocks}, **{x: Role.EXOGENOUS for x in p.exogenous}}
    used = {q for e in eqs for q in e.param_names}
    corr = [f"corr_{a}_{b}" for a, b in p.corr]
    specs = []  # the parameters' descriptions and estimation settings, the shocks' (on their sizes) and correlations'
    for q in [*used, *(f"sd_{s}" for s in p.shocks), *corr]:
        label = p.descriptions.get(q, "") if q in used or q in corr else ""
        if q in p.specs or label:
            specs.append(Param(q, label=label, **p.specs.get(q, {})))
    model = Model(eqs, roles=roles or None, params={q: v for q, v in p.params.items() if q in used} or None, specs=specs)
    measurement = None
    if p.measurement or p.reported:
        texts = [expr for _, expr, _, _ in p.measurement] + [expr for _, expr, _, _ in p.reported] + list(p.transforms.values())
        mparams = {q: v for q, v in p.params.items() if any(re.search(rf"\b{re.escape(q)}\b", expr) for expr in texts)}
        labels = {o: lab for o, _, _, lab in p.measurement if lab} | {r: lab for r, _, lab, _ in p.reported if lab}
        measurement = MeasurementMap([f"{o} = {expr}" for o, expr, _, _ in p.measurement], params=mparams,
                                     errors={o: sd for o, _, sd, _ in p.measurement if sd}, labels=labels,
                                     transforms=dict(p.transforms), reported=[f"{r} = {expr}" for r, expr, _, _ in p.reported])
    guess = {**p.guesses, **p.exogenous}
    labels = {k: v for k, v in p.descriptions.items() if k not in p.params}
    regimes = []
    for r in p.regimes:  # a parameter only a regime's equation uses: given to the regime, at its declared value
        words = set(re.findall(r"[A-Za-z_]\w*", " ".join(r.equations.values())))
        extra = {q: p.params[q] for q in p.params if q in words and q not in model.parameters and q not in r.params}
        regimes.append(replace(r, params={**r.params, **extra}) if extra else r)
    labels = {k: v for k, v in labels.items() if not k.startswith("corr_") or k not in corr}
    return Template(name, model, tuple(p.shocks), dict(p.shocks), measurement, guess, "", labels, dict(p.corr), tuple(regimes))


# ---------------------------------------------------------------------------------------------
# Dynare's syntax


def _blank_comments(text: str) -> str:
    """Comments replaced by spaces, so that positions stay where they were; a quoted string (a long_name) is kept as it
    is, so a % in it, as in 'Output gap (% of potential)', does not start a comment."""
    def blank(m: re.Match) -> str:
        return m.group(0) if m.group(0)[0] in "'\"" else re.sub(r"[^\n]", " ", m.group(0))
    return re.sub(r"'[^'\n]*'|\"[^\"\n]*\"|/\*.*?\*/|//[^\n]*|%[^\n]*", blank, text, flags=re.S)


def _place(text: str, offset: int) -> tuple[int, int]:
    """The line and column (both from 1) of an offset (from 0) in ``text``."""
    line = text.count("\n", 0, offset) + 1
    return line, offset - (text.rfind("\n", 0, offset) + 1) + 1


_DYNARE_BLOCKS = frozenset({"shocks", "initval", "endval", "histval", "steady_state_model", "estimated_params", "estimated_params_init",
                            "estimated_params_bounds", "observation_trends", "homotopy_setup", "optim_weights", "osr_params_bounds",
                            "conditional_forecast_paths", "mshocks", "irf_calibration", "moment_calibration", "planner_objective",
                            "occbin_constraints", "filter_initial_state", "verbatim"})


def _check_dynare(text: str) -> SourceCheck:
    """Dynare's text checked; with macro directives, after Dynare's macro processor has run (``dynare_macros``): a macro
    error at the line of its directive, and each problem of the expanded text at the line it came from (its column
    only where the line is as written; a line from an included file at its ``@#include``)."""
    from dataclasses import replace

    from .dynare_macros import MacroError, expand_macros, has_macros

    if not has_macros(_blank_comments(text)):
        return _check_dynare_text(text)
    original = text.split("\n")
    try:
        expanded, lines = expand_macros(text)
    except MacroError as exc:
        line = min(max(exc.line, 1), len(original))
        found = original[line - 1].find("@")
        message = re.sub(r"^line \d+: ", "", str(exc))
        return SourceCheck("dynare", (Diagnostic(line, found + 1 if found >= 0 else 1, f"macro processor: {message}",
                                                 length=max(len(original[line - 1].strip()), 1)),), (), 0)
    out = expanded.split("\n")

    def where(line: int, column: int) -> tuple[int, int]:
        if not lines:
            return 1, 1
        i = min(max(line, 1), len(lines)) - 1
        source = lines[i]
        same = i < len(out) and source - 1 < len(original) and out[i] == original[source - 1]
        return source, column if same else 1

    found = _check_dynare_text(expanded)
    diagnostics = []
    for d in found.diagnostics:
        line, column = where(d.line, d.column)
        diagnostics.append(replace(d, line=line, column=column))
    symbols = tuple(replace(s, line=where(s.line, 1)[0], used=tuple(sorted({where(u, 1)[0] for u in s.used}))) for s in found.symbols)
    return replace(found, diagnostics=tuple(diagnostics), symbols=symbols)


def _check_dynare_text(text: str) -> SourceCheck:
    from ..expr.nodes import ExpressionError
    from ..expr.parser import parse_equation
    from .dynare import _convert

    clean = _blank_comments(text)
    diagnostics: list[Diagnostic] = []
    statements, start = [], 0
    for m in re.finditer(r";", clean):
        statements.append((clean[start:m.start()], start))
        start = m.end()
    if clean[start:].strip():
        statements.append((clean[start:], start))
    declared: dict[str, tuple[str, int]] = {}
    columns: dict[str, int] = {}
    descriptions: dict[str, str] = {}
    values: dict[str, float] = {}
    equations: list[tuple[str, int]] = []  # text, offset
    locals_: set[str] = set()
    in_model = False
    in_block = False  # shocks, initval, estimated_params...: statements until end; that declare nothing
    for body, offset in statements:
        lead = len(body) - len(body.lstrip())
        st, at = body.strip(), offset + lead
        if not st:
            continue
        if in_block:
            in_block = st != "end"
            continue
        if st.split(None, 1)[0].split("(")[0] in _DYNARE_BLOCKS:
            in_block = True
            continue
        if in_model:
            if st == "end":
                in_model = False
                continue
            eq = re.sub(r"^\[[^\]]*\]\s*", "", st)
            shift = len(st) - len(eq)
            if eq.startswith("#"):
                locals_.add(eq[1:].partition("=")[0].strip())
            equations.append((eq, at + shift))
            continue
        head = st.split(None, 1)[0]
        if head in ("var", "varexo", "parameters"):
            role = {"var": "endogenous", "varexo": "shock", "parameters": "parameter"}[head]
            rest = st[len(head):]
            for m in re.finditer(r"([A-Za-z_]\w*)(\s*\$[^$]*\$)?(\s*\(((?:[^()'\"]|'[^']*'|\"[^\"]*\")*)\))?", rest):
                name = m.group(1)
                if m.group(4):
                    d = re.search(r"long_name\s*=\s*'([^']*)'", m.group(4))
                    if d:
                        descriptions[name] = d.group(1)
                if name in declared:
                    diagnostics.append(Diagnostic(*_place(clean, at + len(head) + m.start(1)), f"{name} is declared twice", length=len(name)))
                declared[name] = (role, _place(clean, at + len(head) + m.start(1))[0])
                columns[name] = _place(clean, at + len(head) + m.start(1))[1]
        elif re.match(r"^model\b", st):
            in_model = True
        elif "=" in st and st.split("=")[0].strip() in declared and declared[st.split("=")[0].strip()][0] == "parameter":
            name, _, expr = st.partition("=")
            try:
                values[name.strip()] = float(safe_eval(expr.replace("^", "**"), values))  # checked: numbers, names, arithmetic
            except Exception:  # noqa: BLE001 - shown as no value; read_mod reports it when the model is built
                pass
    if in_model:
        diagnostics.append(Diagnostic(*_place(clean, statements[-1][1]), "the model block has no 'end;'"))
    endo = [n for n, (r, _) in declared.items() if r == "endogenous"]
    exo = [n for n, (r, _) in declared.items() if r == "shock"]
    params = [n for n, (r, _) in declared.items() if r == "parameter"]
    known = sorted(set(declared) | locals_)
    used: dict[str, set[int]] = {}
    max_lead = max_lag = 0
    constraints = []
    for eq, at in equations:
        for m in re.finditer(r"[A-Za-z_]\w*", eq):
            name = m.group(0)
            if eq[m.start() - 1:m.start()].isdigit() and re.fullmatch(r"[eE]\d*", name):
                continue
            after = eq[m.end():].lstrip()
            if name in _FUNCTIONS and after.startswith("("):
                if name in ("max", "min"):
                    constraints.append(f"line {_place(clean, at)[0]}")
                continue
            line, column = _place(clean, at + m.start())
            if name not in known:
                diagnostics.append(Diagnostic(line, column, f"{name} is not declared (var, varexo or parameters)" + _nearest(name, known),
                                              length=len(name)))
                continue
            used.setdefault(name, set()).add(line)
            timing = re.match(r"\(\s*([+-]?\d+)\s*\)", eq[m.end():])
            if timing and name in endo + exo:
                k = int(timing.group(1))
                max_lead, max_lag = max(max_lead, k), max(max_lag, -k)
    real = [(eq, at) for eq, at in equations if not eq.startswith("#")]
    if not any(d.severity == "error" for d in diagnostics):
        for eq, at in real:  # each equation parses
            try:
                (lhs, rhs, _), = _convert([eq], endo, exo, params)
                parse_equation(f"{lhs} = {rhs}")
            except (ExpressionError, ModelError, ValueError) as exc:
                diagnostics.append(Diagnostic(*_place(clean, at), f"cannot read this equation: {re.sub(r' at position .*$', '', str(exc))}",
                                              length=len(eq)))
    if endo and real and len(real) != len(endo) and not any(d.severity == "error" for d in diagnostics):
        diagnostics.append(Diagnostic(*_place(clean, real[0][1]), f"{len(real)} equations for {len(endo)} endogenous variables (var)"))
    for name in params:
        if name not in used:
            diagnostics.append(Diagnostic(declared[name][1], columns.get(name, 1), f"parameter {name} is not used in the model", "warning",
                                          length=len(name)))
    if real and not any(d.severity == "error" for d in diagnostics):
        from .dynare import read_mod

        try:
            built = read_mod(text)
            values.update(built.params)
        except (ModelError, SpecificationError, ValueError) as exc:
            line, column, length = _at_symbol(str(exc), {n: (ln, columns.get(n, 1)) for n, (_, ln) in declared.items()})
            diagnostics.append(Diagnostic(line, column, f"the file does not build: {exc}", length=length))
    symbols = tuple(Symbol(n, role, values.get(n), descriptions.get(n, ""), line, tuple(sorted(used.get(n, ()))))
                    for n, (role, line) in declared.items())
    order = {"error": 0, "warning": 1, "note": 2}
    diagnostics.sort(key=lambda d: (d.line, d.column, order[d.severity]))
    return SourceCheck("dynare", tuple(diagnostics), symbols, len(real), max_lead, max_lag, tuple(constraints))


# ---------------------------------------------------------------------------------------------
# IRIS's syntax

_IRIS_ROLES = {"!transition_variables": "endogenous", "!variables": "endogenous", "!transition_shocks": "shock", "!shocks": "shock",
               "!measurement_variables": "observable", "!measurement_shocks": "shock", "!parameters": "parameter",
               "!exogenous_variables": "exogenous"}
# read_iris's sections (its own table: the editor reads what it reads) and the preprocessor's words
_IRIS_KNOWN = {*_IRIS_SECTIONS, "!for", "!do", "!end", "!if", "!elseif", "!else"}
_IRIS_FUNCTIONS = _FUNCTIONS | {"diff", "difflog", "ttrend"}


def _at_symbol(message: str, places: Mapping[str, tuple[int, int]]) -> tuple[int, int, int]:
    """Where a build error that names a declared symbol belongs: its declaration (line, column, length); else line 1."""
    for name in re.findall(r"'([A-Za-z_]\w*)'|\b([A-Za-z_]\w*)\b", message):
        name = name[0] or name[1]
        if name in places:
            return (*places[name], len(name))
    return 1, 1, 1


def _check_iris(text: str) -> SourceCheck:
    """IRIS's text checked: comments, sections (with the QPM toolkit's spellings), declarations and their descriptions
    and equation names found as ``read_iris`` finds them (``pymetrix.models.iris``'s own functions), places kept."""
    clean = _iris_strip_comments(text, blank=True)
    diagnostics: list[Diagnostic] = []
    sections = _iris_sections(clean)
    declared: dict[str, tuple[str, int, int]] = {}  # name -> role, line, column
    descriptions: dict[str, str] = {}
    equations: dict[str, list[tuple[str, int]]] = {"transition": [], "measurement": [], "reporting": []}
    assigned: list[tuple[str, int]] = []  # !parameter_values: each name and its place
    preprocessed = any(keyword in ("!for", "!if", "!substitutions") for keyword, *_ in sections)
    for keyword, written, mark, body_start, body_end in sections:
        body = clean[body_start:body_end]
        if keyword not in _IRIS_KNOWN:
            close = difflib.get_close_matches(keyword, sorted(_IRIS_KNOWN), n=1, cutoff=0.6)
            diagnostics.append(Diagnostic(*_place(clean, mark), f"unknown section {written}"
                                          + (f"; did you mean {close[0]!r}?" if close else ""), length=len(written)))
            continue
        if keyword in _IRIS_ROLES:
            for name, label, _, offset in _iris_declared_at(body):
                line, column = _place(clean, body_start + offset)
                if name in declared:
                    diagnostics.append(Diagnostic(line, column, f"{name} is declared twice", length=len(name)))
                declared[name] = (_IRIS_ROLES[keyword], line, column)
                if label:
                    descriptions[name] = label
        elif keyword == "!parameter_values":
            assigned += [(m.group(1), body_start + m.start(1)) for m in re.finditer(r"(?:^|;)\s*([A-Za-z_]\w*)\s*=", body)]
        elif keyword in ("!transition_equations", "!equations", "!measurement_equations", "!reporting_equations"):
            kind = {"!measurement_equations": "measurement", "!reporting_equations": "reporting"}.get(keyword, "transition")
            start = 0
            for piece in body.split(";"):
                stripped = piece.strip()
                if stripped:
                    lead = len(piece) - len(piece.lstrip())
                    described_eq = re.match(r"(\"[^\"]*\"|'[^']*')\s*", piece[lead:])
                    eq = re.sub(r"^(\"[^\"]*\"|'[^']*')\s*", lambda q: " " * len(q.group(0)), piece[lead:])
                    named = re.match(r"\s*" + _IRIS_EQUATION_NAME.pattern, eq)  # the toolkit's "is_curve: y = ..."
                    if named:
                        eq = " " * named.end() + eq[named.end():]
                    at = body_start + start + lead
                    if kind == "reporting":  # the name on the left is new: a reported series
                        target = re.match(r"\s*([A-Za-z_]\w*)\s*=", eq)
                        if target:
                            name = target.group(1)
                            line, column = _place(clean, at + target.start(1))
                            if name in declared:
                                diagnostics.append(Diagnostic(line, column, f"{name} is declared twice", length=len(name)))
                            declared[name] = ("reported", line, column)
                            if described_eq:
                                descriptions[name] = described_eq.group(1)[1:-1]
                            eq = " " * target.end() + eq[target.end():]
                        else:
                            diagnostics.append(Diagnostic(*_place(clean, at), "a reporting equation is name = expression"))
                    equations[kind].append((eq.split("!!")[0], at))
                start += len(piece) + 1
    for name, at in assigned:  # !parameter_values gives values to declared parameters only
        if declared.get(name, ("",))[0] != "parameter":
            diagnostics.append(Diagnostic(*_place(clean, at), f"!parameter_values gives {name}, which is not in !parameters",
                                          length=len(name)))
    known = sorted(declared)
    endo = [n for n, (r, _, _) in declared.items() if r == "endogenous"]
    used: dict[str, set[int]] = {}
    max_lead = max_lag = 0
    constraints = []
    for kind, items in equations.items():
        for eq, at in items:
            for m in re.finditer(r"&?\$?[A-Za-z_?]\w*\$?", eq):
                token = m.group(0)
                name = token.lstrip("&")
                if "?" in name or name.startswith("$"):  # !for controls and !substitutions: checked when the file is read
                    continue
                after = eq[m.end():].lstrip()
                if eq[m.start() - 1:m.start()].isdigit() and re.fullmatch(r"[eE]\d*", name):
                    continue
                if name in _IRIS_FUNCTIONS and (after.startswith("(") or name == "ttrend"):
                    if name in ("max", "min"):
                        constraints.append(f"line {_place(clean, at)[0]}")
                    continue
                line, column = _place(clean, at + m.start())
                if name not in declared:
                    diagnostics.append(Diagnostic(line, column, f"{name} is not declared" + _nearest(name, known), length=len(name)))
                    continue
                used.setdefault(name, set()).add(line)
                timing = re.match(r"\{\s*([+-]?\d+)\s*\}", eq[m.end():])
                if timing:
                    k = int(timing.group(1))
                    max_lead, max_lag = max(max_lead, k), max(max_lag, -k)
    transition = equations["transition"]
    if not preprocessed and transition and len(transition) != len(endo) and not any(d.severity == "error" for d in diagnostics):
        diagnostics.append(Diagnostic(*_place(clean, transition[0][1]), f"{len(transition)} transition equations for {len(endo)} "
                                      "transition variables"))
    for name, (role, line, column) in declared.items():
        if role == "parameter" and name not in used and not name.startswith("std_") and not (name.startswith("corr_") and "__" in name):
            diagnostics.append(Diagnostic(line, column, f"parameter {name} is not used", "warning", length=len(name)))
    values: dict[str, float] = {}
    if transition and not any(d.severity == "error" for d in diagnostics):
        from .iris import read_iris

        try:
            built = read_iris(text)
            values.update({p.name: p.value for p in built.model.parameters if p.value is not None})
            values.update({f"std_{s}": v for s, v in built.shock_sd.items()})
            for n in declared:  # correlations, corr_<a>__<b>
                pair = tuple(n[5:].split("__")) if n.startswith("corr_") else ()
                if len(pair) == 2 and (pair in built.shock_corr or pair[::-1] in built.shock_corr):
                    values[n] = float(built.shock_corr.get(pair, built.shock_corr.get(pair[::-1], 0.0)))
        except (ModelError, SpecificationError, ValueError) as exc:
            line, column, length = _at_symbol(str(exc), {n: (ln, col) for n, (_, ln, col) in declared.items()})
            diagnostics.append(Diagnostic(line, column, f"the file does not build: {exc}", length=length))
    symbols = tuple(Symbol(n, role, values.get(n), descriptions.get(n, ""), line, tuple(sorted(used.get(n, ()))))
                    for n, (role, line, _) in declared.items())
    order = {"error": 0, "warning": 1, "note": 2}
    diagnostics.sort(key=lambda d: (d.line, d.column, order[d.severity]))
    return SourceCheck("iris", tuple(diagnostics), symbols, len(transition), max_lead, max_lag, tuple(constraints))


def _check_eviews(text: str) -> SourceCheck:
    """EViews's text checked (``pymetrix.models.eviews``): model text or a program."""
    from .eviews import check_eviews

    return check_eviews(text)


def check_text(text: str, syntax: Syntax = "pymetrix") -> SourceCheck:
    """``ModelSource(text, syntax).check()``."""
    return ModelSource(text, syntax).check()


def known_syntaxes() -> Mapping[str, str]:
    return {"pymetrix": "pymetrix", "dynare": "Dynare (.mod)", "iris": "IRIS (.model)", "eviews": "EViews (.prg / model text)"}


# ---------------------------------------------------------------------------------------------
# the editor's completions and hover help

_BLANKED = {"pymetrix": re.compile(r'"[^"\n]*(?:"|$)|#[^\n]*', re.M),
            "dynare": re.compile(r"/\*.*?(?:\*/|\Z)|//[^\n]*|%[^\n]*|'[^'\n]*(?:'|$)", re.M | re.S),
            "iris": re.compile(r"%\{.*?(?:%\}|\Z)|%[^\n]*|\"[^\"\n]*(?:\"|$)|'[^'\n]*(?:'|$)", re.M | re.S),
            "eviews": re.compile(r"\"[^\"\n]*(?:\"|$)|'[^\n]*", re.M)}


def _in_comment(text: str, line: int, column: int, syntax: str) -> bool:
    """Whether the character at ``line`` and ``column`` (from 1; with 0, the start of the line) is in a comment or a
    description (a quoted string), of the text as it is (an unclosed quote runs to the end of its line)."""
    lines = text.split("\n")
    offset = sum(len(t) + 1 for t in lines[:line - 1]) + max(column - 1, 0)
    return any(m.start() <= offset < m.end() for m in _BLANKED[syntax].finditer(text, 0, offset + 1))


_PYMETRIX_KEYWORDS = {
    "parameters": "section: parameters and their values, name = value  [bounds low high] [prior shape(mean, sd)] [start v] "
                  "[fixed] [symbol S]  \"description\"",
    "shocks": "section: shocks and their standard deviations, name = sd; correlations, corr a, b = r",
    "exogenous": "section: exogenous variables and their steady-state values",
    "variables": "section: endogenous variables' steady-state guesses and descriptions (optional)",
    "equations": "section: one equation a line, [name:] variable = expression; x[-1] a lag, x[+1] a lead",
    "measurement": "section: observed = expression [from transform of raw data] [error sd]",
    "reported": "section: series reported with the history and forecasts, never fitted: name = expression",
    "regimes": "section: start[..end] [name] [surprise]: parameter = value, ...; equation: lhs = rhs",
    "bounds": "bounds low high: the parameter's range in estimation",
    "prior": "prior shape(mean, sd): normal, beta, gamma, invgamma1, invgamma2, uniform(low, high)",
    "start": "start value: where estimation starts",
    "fixed": "fixed: held at its value in estimation",
    "symbol": "symbol S: its symbol in algebraic notation (φ_π)",
    "from": "from transform: the observed series made from a raw one, e.g. from 400*dlog(gdp)",
    "error": "error sd: the measurement error's standard deviation",
    "corr": "corr a, b = r: the correlation of two shocks (estimated as corr_a_b)",
    "surprise": "surprise: the regime is not known until it starts (expected=False)",
}
_EVIEWS_KEYWORDS = {
    "model": "model m: declares the model (a program)", "append": "m.append <equation>: a line of the model's text",
    "merge": "m.merge eq: the equation object eq, specified in the same text, linked into the model",
    "equation": "equation eq.ls y c y(-1) x: a least-squares specification (c the intercept; c(1), c(2) ... in order), or "
                "equation eq.ls y = c(1) + c(2)*x",
    "ls": "least squares: fitted with read_eviews(text, data=...)",
    "@identity": "@identity y = ...: an identity (no coefficients, no add factor)",
    "@add": "@add y [y_a]: an add factor, a series added to the right of the equation for y",
    "@innov": "@innov y sd: a shock y_innov, of standard deviation sd, added to the equation for y",
    "scalar": "scalar name = value: a parameter", "coef": "coef(n) beta: a coefficient vector, beta(1) ... beta(n) (beta_1 ...)",
    "vector": "vector(n) beta: a coefficient vector, as coef", "param": "param c(1) value c(2) value ...: coefficient values",
    "smpl": "smpl first last: the sample of the specifications after it (1990q1, 1990m3, 1990, @first, @last, @all)",
    "series": "series x = number: an exogenous series' value (or an endogenous variable's starting guess); other series "
              "generation is ignored",
}
_PRIOR_WORDS = ("normal", "beta", "gamma", "invgamma1", "invgamma2", "uniform")
_DYNARE_KEYWORDS = ("var", "varexo", "varexo_det", "parameters", "model", "end", "shocks", "initval", "endval", "histval",
                    "steady_state_model", "estimated_params", "estimated_params_init", "estimated_params_bounds", "varobs",
                    "estimation", "stoch_simul", "steady", "check", "stderr", "corr", "periods", "values", "model_local_variable",
                    "occbin_constraints", "observation_trends", "long_name", "normal_pdf", "beta_pdf", "gamma_pdf",
                    "inv_gamma_pdf", "inv_gamma1_pdf", "inv_gamma2_pdf", "uniform_pdf")
_FAMILY_WORDS = {"norm": "normal", "t": "Student's t", "chi2": "chi-squared", "f": "F", "gam": "gamma", "beta": "beta",
                 "logn": "lognormal", "unif": "uniform", "exp": "exponential", "poiss": "Poisson", "bino": "binomial"}
_BASIC_FUNCTIONS = {
    "log": ("log(x)", "the natural logarithm"), "exp": ("exp(x)", "e to the power x"), "sqrt": ("sqrt(x)", "the square root"),
    "abs": ("abs(x)", "the absolute value"), "sign": ("sign(x)", "-1, 0 or 1, the sign of x"),
    "max": ("max(a, b)", "the larger of a and b: in a model's equation, an occasionally binding constraint (OccBin)"),
    "min": ("min(a, b)", "the smaller of a and b: in a model's equation, an occasionally binding constraint (OccBin)"),
    "step": ("step(x)", "1 where x >= 0, else 0"), "not": ("not(x)", "1 where x is 0, else 0"),
    "d": ("d(x)", "the first difference, x - x[-1]"), "dlog": ("dlog(x)", "the difference of the log, log(x) - log(x[-1])"),
    "pch": ("pch(x)", "the percent change, 100*(x/x[-1] - 1)"),
    "diff": ("diff(x)", "the first difference, x - x{-1} (IRIS)"), "difflog": ("difflog(x)", "log(x) - log(x{-1}) (IRIS)"),
    "ttrend": ("ttrend", "the time trend, periods from the base period (IRIS's !dtrends)"),
    "steady_state": ("steady_state(x)", "x's steady-state value (Dynare)"),
    "STEADY_STATE": ("STEADY_STATE(x)", "x's steady-state value (Dynare)"),
    "erf": ("erf(x)", "the error function"), "erfc": ("erfc(x)", "the complementary error function, 1 - erf(x)"),
}
_MOVING_WORDS = {"movavg": "average", "movsum": "sum", "movsd": "standard deviation", "movvar": "variance", "movmin": "minimum",
                 "movmax": "maximum", "movcov": "covariance of x and y", "movcorr": "correlation of x and y",
                 "movbeta": "slope of x on y"}
_CENTRED_WORDS = {f"c{fn}": what for fn, what in _MOVING_WORDS.items()}
_EXPANDING_WORDS = {"cum" + fn[3:]: what for fn, what in _MOVING_WORDS.items()}


def function_help(name: str) -> str:
    """A function's signature and what it does, in a line (empty for a name that is not a function)."""
    from ..expr.functions import DISTRIBUTIONS, EXPANDING, MOVING, signature

    if name in _BASIC_FUNCTIONS:
        sig, what = _BASIC_FUNCTIONS[name]
        return f"{sig}: {what}"
    if name in _CENTRED_WORDS:
        return (f"{signature(name)}: the {_CENTRED_WORDS[name]} over a centred window of n periods (leads and lags: a model "
                "reads its leads as expectations)")
    if name in EXPANDING:
        return (f"{signature(name)}: the {_EXPANDING_WORDS[name]} over every period up to t (an expanding window; data only, "
                "not in a model)")
    if name in MOVING:
        return f"{signature(name)}: the {_MOVING_WORDS.get(name, name[3:])} over a trailing window of n periods"
    if name in DISTRIBUTIONS:
        family, kind = DISTRIBUTIONS[name]
        what = {"pdf": "probability mass" if family in ("poiss", "bino") else "density", "cdf": "cumulative distribution function",
                "inv": "quantile function: the smallest integer k with cdf(k) >= p (no derivative)" if family in ("poiss", "bino")
                else "quantile function (the inverse of the cdf)"}[kind]
        return f"{signature(name)}: the {_FAMILY_WORDS.get(family, family)} {what}"
    return ""


def _functions(syntax: str) -> list[str]:
    if syntax == "eviews":
        from .eviews import FUNCTIONS as EVIEWS_FUNCTIONS

        return sorted(EVIEWS_FUNCTIONS, key=lambda f: f.lstrip("@"))
    names = (_IRIS_FUNCTIONS if syntax == "iris" else _FUNCTIONS) - set(_EXPANDING)  # expanding windows: data only
    if syntax == "pymetrix":
        names = names - {"steady_state", "STEADY_STATE"}
    return sorted(names, key=str.lower)


def _vocabulary(source: ModelSource, found: SourceCheck) -> list[str]:
    """Every word the editor may complete: declared names, equation names, functions, then keywords."""
    words = [s.name for s in found.symbols]
    if source.syntax == "pymetrix":
        words += re.findall(r"^\s*([A-Za-z_]\w*)\s*:(?!=)", source.text, re.M)  # equations' names, for regimes
        keywords = [*_PYMETRIX_KEYWORDS, *_PRIOR_WORDS]
    elif source.syntax == "dynare":
        keywords = list(_DYNARE_KEYWORDS)
    elif source.syntax == "eviews":
        keywords = list(_EVIEWS_KEYWORDS)
    else:
        keywords = sorted(_IRIS_KNOWN)
    return list(dict.fromkeys([*words, *_functions(source.syntax), *keywords]))


_WORD = {"pymetrix": r"[A-Za-z_]\w*", "dynare": r"[A-Za-z_]\w*", "iris": r"!?[A-Za-z_][\w\-]*", "eviews": r"@?[A-Za-z_]\w*"}
_ROLE_WORDS = {"endogenous": "endogenous variable", "exogenous": "exogenous variable", "shock": "shock", "parameter": "parameter",
               "observable": "observed series", "reported": "reported series", "local": "model-local variable",
               "correlation": "correlation of two shocks"}


def _hover_text(word: str, call: bool, syntax: str, found: SourceCheck) -> str:
    symbol = next((s for s in found.symbols if s.name == word), None)
    if symbol is not None and not (call and function_help(word)):
        value = "" if symbol.value is None else {"shock": f", standard deviation {symbol.value:g}",
                                                 "endogenous": f", steady-state guess {symbol.value:g}",
                                                 "exogenous": f", steady-state value {symbol.value:g}"}.get(symbol.role, f" = {symbol.value:g}")
        lines = [f"{word}: {_ROLE_WORDS.get(symbol.role, symbol.role)}{value}"]
        if symbol.description:
            lines.append(symbol.description)
        if symbol.estimation:
            lines.append(f"estimation: {symbol.estimation}")
        where = []
        if symbol.line:
            where.append(f"{'determined' if symbol.role == 'endogenous' else 'declared'} on line {symbol.line}")
        if symbol.used:
            where.append(f"used on line{'s' if len(symbol.used) > 1 else ''} {', '.join(map(str, symbol.used))}")
        elif symbol.role in ("parameter", "shock"):
            where.append("not used")
        if where:
            lines.append("; ".join(where))
        return "\n".join(lines)
    if syntax == "eviews":
        from .eviews import FUNCTIONS as EVIEWS_FUNCTIONS

        if word.lower() in EVIEWS_FUNCTIONS:
            sig, what = EVIEWS_FUNCTIONS[word.lower()]
            return f"{sig}: {what}"
        return _EVIEWS_KEYWORDS.get(word.lower(), "")
    if word in _functions(syntax) or call:
        return function_help(word)
    if syntax == "pymetrix":
        return _PYMETRIX_KEYWORDS.get(word, "")
    return ""
