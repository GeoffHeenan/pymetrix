"""Phase 1: data work, and the translator. Each statement becomes ``Op``s (``ir.py``) with a status and a reason, or is
left out with one; estimation and tests (``estimation.py``) and models (``models.py``) are phase 2's handlers.

The translator walks the program in order, as EViews runs it, keeping what a straight-line program needs: the pages
(each a Dataset variable, the current one, their frequencies, series and samples) and the program's own variables
(scalars, ``!`` numbers, ``%`` strings, groups, samples, matrices), with their values where they are known before
the program runs, so ``{%s}`` and ``{!i}`` in a line can be put in. Loops, ``if`` blocks and subroutines are
phase 3: they and the lines inside them are left out with that reason, as is a line that needs a value only a
subroutine or a loop gives.
"""

from __future__ import annotations

import builtins
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pymetrix.extension import EViewsBinary, EViewsBrackets, EViewsCall, EViewsCoefficient, EViewsExpressionError, EViewsName
from pymetrix.extension import EViewsNegative, EViewsNumber, parse_eviews_expression, render_eviews_expression
from pymetrix.extension import walk_eviews_expression

from ..bundle import _py_name
from .estimation import ESTIMATORS, LEFT_ESTIMATORS, Estimation
from .ir import Arg, Code, Op, SampleSpec, Step, Text, Var, quote
from .left import (GAP_FUNCTIONS, PROGRAM_FUNCTIONS, PHASE2_COMMANDS, PHASE3, PHASE3_WORDS, PHASE4_COMMANDS, PHASE4_METHODS,
                   STATISTICS, left_reason)
from .models import Models
from .parse import Statement, _statement
from .report import ERROR, LEFT, NOTE, TRANSLATED

__all__ = ["Outcome", "Translator", "is_closer", "is_opener"]

_UNKNOWN = object()
_NAN = 'float("nan")'
_OPENERS = {"for": "next", "while": "wend", "if": "endif", "subroutine": "endsub"}
_CLOSERS = {"next", "wend", "endif", "endsub"}
_HARMLESS = frozenset({*PHASE4_COMMANDS, "wfsave", "pagesave", "wfclose"})  # left out without making series
_SCRIPT_NAMES = {"Dataset", "Matrix", "Sample", "pymetrix", "pymetrix_eviews"}


class _Out(Exception):
    """A statement that does not translate: its status (left out or error) and the reason."""

    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status, self.reason = status, reason


def _left(reason: str) -> _Out:
    return _Out(LEFT, reason)


def _error(reason: str) -> _Out:
    return _Out(ERROR, reason)


@dataclass
class _Page:
    var: str
    eviews: str
    freq: str | None  # Y, Q, M, W, D; None undated or unknown
    series: set[str] | None  # pymetrix names; None when not known
    sample: Any = None  # None (all), a SampleSpec, or a Var (a named sample)
    first: str | None = None
    last: str | None = None


@dataclass
class _Py:
    name: str
    kind: str  # scalar, control, string, group, sample, matrix
    value: Any = _UNKNOWN
    shape: tuple[int, int] | None = None
    members: list[str] = field(default_factory=list)
    page: str = ""


@dataclass
class Outcome:
    """One statement translated: its status, reason and ops."""

    statement: Statement
    status: str
    reason: str = ""
    ops: list[Op] = field(default_factory=list)


def _public_names() -> set[str]:
    import pymetrix

    return {n for n in getattr(pymetrix, "__all__", ())} | {"data"} | {n for n in dir(builtins) if not n.startswith("_")}


def _number_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() and abs(value) < 1e15 else repr(float(value))


_FREQ = {"a": "Y", "y": "Y", "q": "Q", "m": "M", "w": "W", "d": "D", "d7": "D", "7": "D"}
_FREQ_GAP = {"s": "semi-annual pages: pandas has no half-year period (a gap)",
             "d5": "business-day pages are not translated (a gap)", "5": "business-day pages are not translated (a gap)",
             "u": "undated pages are not made here: read the data with Dataset.read, or use a dated page",
             "h": "intraday pages are not translated (a gap)", "min": "intraday pages are not translated (a gap)",
             "sec": "intraday pages are not translated (a gap)", "bm": "bimonthly pages are not translated (a gap)",
             "f": "fortnightly pages are not translated (a gap)", "t": "ten-day pages are not translated (a gap)"}


def _period(token: str, freq: str | None) -> str | None:
    """An EViews date as a pymetrix period of frequency ``freq`` (Y, Q or M): 1990q1, 1990:1, 1990m3, 1990."""
    token = token.strip().strip('"').lower()
    m = re.fullmatch(r"(\d{4})(?:([qm:])(\d{1,2}))?", token)
    if m is None or freq not in ("Y", "Q", "M"):
        return None
    year, kind, sub = m.group(1), m.group(2), int(m.group(3) or 1)
    if freq == "Y":
        return year if kind is None else None
    if freq == "Q" and kind in ("q", ":") and 1 <= sub <= 4:
        return f"{year}Q{sub}"
    if freq == "M" and kind in ("m", ":") and 1 <= sub <= 12:
        return f"{year}-{sub:02d}"
    return None  # a year alone in a quarterly or monthly page: the caller decides its first or last period


def _shifted(period: str, freq: str, n: int) -> str:
    import pandas as pd

    return str(pd.Period(period, freq=freq) + n)


class Translator(Estimation, Models):
    """Translates a program's statements in order (see the module's docstring); estimation and tests
    (``estimation.py``) and models (``models.py``) are phase 2's."""

    def __init__(self, bundle: str | None = None, read_options: dict[str, Any] | None = None) -> None:
        self.bundle = bundle
        self.read_options = dict(read_options or {})
        self.pages: dict[str, _Page] = {}  # by EViews name, lower case
        self.current: _Page | None = None
        self.py: dict[str, _Py] = {}  # by EViews name: "s", "!i", "%x"
        self.taken: dict[str, str] = {}  # Python name -> what holds it
        self.types: dict[str, str] = {}  # other objects' types, by EViews name: model, graph, table, equation, ...
        self.opened: str | None = None
        self.uses_bundle = False
        self.public = _public_names()
        self.workfile_frequency: str | None = None
        self.lost = False  # a line that may have set values was left out
        self.unset: dict[str, int] = {}  # variables a left-out line sets, by EViews name -> that line
        self.fits: dict[str, Any] = {}  # estimated objects, by EViews name (estimation.Fit)
        self.systems: dict[str, dict[str, Any]] = {}  # systems being collected
        self.models: dict[str, Any] = {}  # models (models._Model)
        self.last_fit: Any = None  # the estimation that last filled c
        self.building: list[tuple[str, int]] = []  # (model, line) of each line that builds a model
        self.refusals: dict[int, list[str]] = {}  # the model reader's problems, by the program's line
        self._declared: dict[str, int] = {}  # scalars -> the line that made them
        self._fit_line: dict[str, int] = {}  # estimated objects -> the line that estimated them last
        self.line = 0  # the line being translated

    # -- names ------------------------------------------------------------------------------------------------------

    def _name(self, eviews: str, what: str) -> tuple[str, str]:
        """A Python variable for an EViews object, and a note when its name changed."""
        base = _py_name(eviews.lstrip("!%"))
        note = f"{eviews} is called {base} ({eviews.lstrip('!%').lower()} is a Python word)" if base != eviews.lstrip("!%").lower() else ""
        if base in self.public or base in _SCRIPT_NAMES:
            note = f"{eviews} is called {base}_v ({base} is a Python or pymetrix name)"
            base = f"{base}_v"
        name, i = base, 2
        while name in self.taken and self.taken[name] != what:
            name, i = f"{base}_{i}", i + 1
        if name != base:
            note = f"{eviews} is called {name} ({base} is taken)"
        self.taken[name] = what
        return name, note

    def _page_named(self, name: str) -> _Page:
        page = self.pages.get(name.lower())
        if page is None:
            raise _error(f"no page {name!r}" + (f"; the pages: {', '.join(p.eviews for p in self.pages.values())}" if self.pages else ""))
        return page

    def _page(self) -> _Page:
        if self.current is None:
            raise _error("no workfile is open yet: wfopen or wfcreate comes first")
        return self.current

    # -- the program's values ----------------------------------------------------------------------------------------

    def _known(self, key: str) -> Any:
        v = self.py.get(key.lower())
        return _UNKNOWN if v is None else v.value

    def _substitute(self, st: Statement) -> tuple[Statement, list[str]]:
        """``{%s}``, ``{!i}`` and ``{s}`` put in with their values, and ``%s`` in a command's arguments; a value not
        known before the program runs is phase 3."""
        notes: list[str] = []

        def value_text(key: str, braces: bool) -> str:
            value = self._known(key)
            if value is _UNKNOWN:
                kind = (f"set by line {self.unset[key.lower()]}, which is left out" if key.lower() in self.unset else
                        "set by a subroutine, a loop or the data" if key.lower() in self.py or self.lost else "never set")
                raise _left(f"{'{' + key + '}' if braces else key} has no value before the program runs ({kind}): {PHASE3}")
            text = _number_text(value) if isinstance(value, (int, float)) else str(value)
            shown = "{" + key + "}" if braces else key
            if f"{shown} is {text!r}" not in notes:
                notes.append(f"{shown} is {text!r}")
            return text

        text = st.text
        if "{" in text:
            text = re.sub(r"\{\s*([!%]?\w+)\s*\}", lambda m: value_text(m.group(1), True), text)
            if "{" in text:
                raise _left(f"a name made from an expression in braces: {PHASE3}")
        if st.kind != "assign" and st.word not in ("string",) and "%" in text:
            pieces = re.split(r'("[^"]*")', text)
            text = "".join(p if p.startswith('"') else re.sub(r"%(\w+)", lambda m: value_text("%" + m.group(1), False), p)
                           for p in pieces)
        if text == st.text:
            return st, []
        lead = len(st.text) - len(st.text.lstrip())
        new = _statement(text, st.at, st.line, st.last_line)
        return replace(new, rest_at=st.rest_at if new.rest == st.rest else st.at + lead), notes

    # -- expressions ------------------------------------------------------------------------------------------------

    def _scan(self, text: str) -> None:
        """Functions pymetrix's expressions have no equivalent of (the audit), and statistics: left out by name."""
        if re.search(r"\w\.@?\w", re.sub(r'"[^"]*"|\d+\.\d*', "", text)):
            raise _left("an object's data member (eq.@coefs(2), x.@count) in a series expression is not translated: an "
                        "equation's is, as a scalar's value (scalar b = eq.@coefs(2)), then use the scalar; the others are "
                        f"EViews's programming language: {PHASE3}")
        for m in re.finditer(r'"[^"]*"|(?<![\w@!%])(@?[A-Za-z_]\w*)(\s*\()?', text):
            if m.group(1) is None:
                continue
            name, call = m.group(1).lower(), bool(m.group(2))
            if name in STATISTICS:
                raise _left(f"{name} is the sample's {STATISTICS[name].replace('std', 'standard deviation').replace('count', 'number of observations')}, "
                            f"a number: give a scalar its value first (scalar m = {name}(x)), then use the scalar")
            if name in PROGRAM_FUNCTIONS:
                raise _left(f"{name} is part of EViews's programming language (strings, objects, matrices): {PHASE3}")
            if name in GAP_FUNCTIONS:
                raise _left(f"{name} ({GAP_FUNCTIONS[name]}) has no equivalent in pymetrix's expressions: a gap "
                            "(docs/known-gaps.md)")
            if name == "na" and not call:
                raise _left("na inside an expression: pymetrix's expressions have no missing-value constant (a gap); "
                            "y = na alone translates")
            if name.startswith("@") and name in ("@uiprompt", "@uidialog", "@uiedit", "@uilist", "@uiradio"):
                raise _left(left_reason("interactive prompts", "the dialog asks for the program's arguments"))

    def _scalars(self) -> dict[str, str]:
        """Names as the expression parser sees them (``!i`` as ``_i``) -> Python variables, for the numbers."""
        out = {}
        for key, v in self.py.items():
            if v.kind in ("scalar", "control"):
                out["_" + key[1:] if key.startswith("!") else key] = v.name
        return out

    def _tree(self, text: str, at: int, comparison: bool = False, coefs: bool = False) -> Any:
        self._scan(text)
        if re.search(r'%\w', re.sub(r'"[^"]*"', "", text)):
            raise _left(f"a string variable in an expression: {PHASE3}")
        work = re.sub(r"!(?=[A-Za-z_])", "_", text)
        vectors = {"c"} | {k for k, v in self.types.items() if v == "coef"}
        try:
            tree = parse_eviews_expression(work, at, vectors, comparison)
        except EViewsExpressionError as exc:
            if "is not read here" in exc.message:
                raise _left(exc.message.split(";")[0] + ": not in pymetrix's expressions, nor in the audit (docs/known-gaps.md)") from None
            raise _error(exc.message or "this does not read as an EViews expression") from None
        if not coefs:
            for node in walk_eviews_expression(tree):
                if isinstance(node, EViewsCoefficient):
                    raise _left(f"{node.vector}({node.index}) is an estimated coefficient: in a series expression it is not "
                                f"translated; give a scalar its value first (scalar b = {node.vector}({node.index})), then use the scalar")
        return tree

    def _names_in(self, rendered: str, page: _Page | None, scalars: dict[str, str], allowed: frozenset[str] = frozenset()) -> Text:
        """The rendered pymetrix text with the program's numbers as variables (in brackets) and series renamed as
        the bundle reader names them; a name that is neither, where the page's series are known, is an error."""
        parts: list[Any] = []
        pos = 0
        for m in re.finditer(r"(?<![\w.])([A-Za-z_]\w*)(?!\w)(\s*\()?", rendered):
            if m.group(2):
                continue
            name = m.group(1)
            if name in ("and", "or", "not"):
                continue
            if name in scalars:
                parts += [rendered[pos:m.start()] + "(", Var(scalars[name]), ")"]
                pos = m.end()
                continue
            if name in allowed:
                continue
            series = _py_name(name)
            if page is not None and page.series is not None and series not in page.series:
                if series == "resid":
                    raise _left("resid, EViews's series of the last residuals, is not translated: eq.makeresids r makes them "
                                "a series of the page")
                raise _error(f"{name} is not a series of page {page.eviews}, nor a scalar of the program")
            if series != name:
                parts.append(rendered[pos:m.start()] + series)
                pos = m.end()
        parts.append(rendered[pos:])
        merged: list[Any] = []
        for p in parts:
            if isinstance(p, str) and merged and isinstance(merged[-1], str):
                merged[-1] += p
            elif p != "":
                merged.append(p)
        return Text(tuple(merged))

    def expression(self, text: str, at: int, page: _Page, coefs: bool = False) -> Text:
        """A series expression as pymetrix's text, with the program's numbers in it by name; with ``coefs``, an
        estimated equation's right-hand side, its coefficients ``c(i)`` as ``c_i``."""
        tree = self._tree(text, at, comparison=True, coefs=coefs)  # EViews: series d = x > 3, 1 where it holds
        scalars = self._scalars()
        try:
            rendered = render_eviews_expression(tree, tuple(n for n in scalars), frequency="A" if page.freq == "Y" else page.freq)
        except EViewsExpressionError as exc:
            raise _error(exc.message) from None
        except Exception as exc:  # the frequency of @pchy and the like
            raise _error(str(exc)) from None
        allowed = frozenset(f"{n.vector}_{n.index}" for n in walk_eviews_expression(tree) if isinstance(n, EViewsCoefficient)) if coefs else frozenset()
        return self._names_in(rendered, page, scalars, allowed)

    def condition(self, text: str, at: int, page: _Page) -> Text:
        """A sample's condition (``if x > 0 and y < 3``) as a row condition on the page's columns (``Sample(where=)``)."""
        pieces: list[tuple[str, int, str]] = []  # (text, offset, the connector before it)
        depth, start, connector, i = 0, 0, "", 0
        quoted = False
        while i < len(text):
            ch = text[i]
            if ch == '"':
                quoted = not quoted
            elif not quoted and ch == "(":
                depth += 1
            elif not quoted and ch == ")":
                depth -= 1
            elif not quoted and depth == 0:
                m = re.match(r"(?i)(?<![\w])(and|or)(?![\w])", text[i:]) if (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")) else None
                if m:
                    pieces.append((text[start:i], start, connector))
                    connector = m.group(1).lower()
                    i += len(m.group(1))
                    start = i
                    continue
            i += 1
        pieces.append((text[start:], start, connector))
        scalars = self._scalars()
        out: list[Any] = []
        for piece, offset, conn in pieces:
            if not piece.strip():
                raise _error("a condition is missing on one side of and/or")
            negated = re.match(r"\s*not\s+", piece, re.I)
            body = piece[negated.end():] if negated else piece
            tree = self._tree(body, at + offset + (negated.end() if negated else 0), comparison=True)
            for node in walk_eviews_expression(tree):
                if isinstance(node, EViewsName) and node.lag:
                    raise _left("a lag in a sample's condition: a sample selects rows by the data's columns; make the lagged "
                                "series first (series x_1 = x(-1)) and use it")
                if isinstance(node, EViewsCall) and node.fn not in ("log", "@log", "exp", "@exp", "abs", "@abs", "sqr", "@sqrt"):
                    raise _left(f"{node.fn} in a sample's condition: a sample's condition uses columns, numbers, comparisons "
                                "and log, exp, abs and sqrt; make the series first and use it")
            try:
                rendered = render_eviews_expression(tree, tuple(scalars))
            except EViewsExpressionError as exc:
                raise _error(exc.message) from None
            rendered = rendered.replace("^", "**")
            if conn:
                out.append(f" {conn} ")
            if negated:
                out.append("not ")
            out += list(self._names_in(rendered, page, scalars).parts)
        merged: list[Any] = []
        for p in out:
            if isinstance(p, str) and merged and isinstance(merged[-1], str):
                merged[-1] += p
            else:
                merged.append(p)
        return Text(tuple(merged))

    def _number(self, node: Any) -> tuple[str, Any]:
        """A scalar expression's tree as Python, and its value when known before the program runs."""
        if isinstance(node, EViewsNumber):
            text = "0" + node.text if node.text.startswith(".") else node.text
            return text, float(text)
        if isinstance(node, EViewsName):
            key = "!" + node.name[1:] if node.name.startswith("_") and ("!" + node.name[1:]) in self.py else node.name
            v = self.py.get(key)
            if v is None or v.kind not in ("scalar", "control"):
                if key in self.py:
                    raise _error(f"{node.name} is a {self.py[key].kind}, not a number")
                if key in self.unset or self.lost:
                    raise self._missing(key)
                raise _error(f"{node.name.lstrip('_')} is not a scalar of the program: a scalar's value needs numbers "
                             "(a series' value at a date is @elem, not read yet; a statistic is @mean(x) and the like)")
            if node.lag:
                raise _error(f"{node.name} is a scalar: it has no lags or leads")
            return v.name, v.value
        if isinstance(node, EViewsBrackets):
            code, value = self._number(node.x)
            return f"({code})", value
        if isinstance(node, EViewsNegative):
            code, value = self._number(node.x)
            return f"-{code}", (-value if value is not _UNKNOWN else _UNKNOWN)
        if isinstance(node, EViewsBinary):
            (lc, lv), (rc, rv) = self._number(node.left), self._number(node.right)
            op = {"^": "**", "=": "==", "<>": "!="}.get(node.op, node.op)
            code = f"{lc} {op} {rc}"
            value = _UNKNOWN
            if lv is not _UNKNOWN and rv is not _UNKNOWN:
                try:
                    value = float(eval(f"{lv!r} {op} {rv!r}", {"__builtins__": {}}))  # noqa: S307 - two numbers and an operator
                except (ZeroDivisionError, OverflowError):
                    value = math.nan
            if op in ("<", ">", "<=", ">=", "==", "!="):
                code = f"float({code})"
            return code, value
        if isinstance(node, EViewsCall):
            fns = {"log": math.log, "@log": math.log, "exp": math.exp, "@exp": math.exp, "abs": abs, "@abs": abs,
                   "sqr": math.sqrt, "@sqrt": math.sqrt}
            if node.fn in fns and len(node.args) == 1:
                _, value = self._number(node.args[0])
                if value is not _UNKNOWN:
                    try:
                        result = float(fns[node.fn](value))
                    except (ValueError, OverflowError):
                        result = math.nan
                    return _number_text(result) if math.isfinite(result) else _NAN, result
                raise _left(f"{node.fn} of a value known only when the program runs: compute it in Python (math.{fns[node.fn].__name__})")
            raise _left(f"{node.fn} in a scalar's expression is not translated")
        raise _error("this does not read as a number")

    def number(self, text: str, at: int) -> tuple[str, Any, bool]:
        """A scalar expression: Python, its value where known, and whether a function was worked out."""
        tree = self._tree(text, at)
        code, value = self._number(tree)
        worked = any(isinstance(n, EViewsCall) for n in walk_eviews_expression(tree))
        if worked and value is not _UNKNOWN:
            code = _number_text(value) if isinstance(value, float) and math.isfinite(value) else _NAN
        return code, value, worked

    def string(self, text: str) -> tuple[str, Any]:
        """A string expression (literals and ``%`` variables joined by ``+``): Python, and its value when known."""
        tokens = re.findall(r'\s*("[^"]*"|%\w+|[A-Za-z_]\w*|\+|\S+)', text)
        if not tokens:
            raise _error("a string assignment needs a value")
        codes, values = [], []
        expect_value = True
        for tok in tokens:
            tok = tok.strip()
            if expect_value:
                if tok.startswith('"'):
                    codes.append(quote(tok[1:-1]))
                    values.append(tok[1:-1])
                elif tok.startswith("%") or tok.lower() in self.py and self.py[tok.lower()].kind == "string" or tok.lower() in self.unset:
                    v = self.py.get(tok.lower())
                    if v is None:
                        raise self._missing(tok)
                    codes.append(v.name)
                    values.append(v.value)
                elif tok.startswith("@"):
                    raise _left(f"string functions ({tok}) are part of EViews's programming language: {PHASE3}")
                elif self.lost:
                    raise self._missing(tok)
                else:
                    raise _error(f"{tok} is not a string or a string variable")
            elif tok != "+":
                raise _error(f"expected + between strings, not {tok!r}")
            expect_value = not expect_value
        if expect_value:
            raise _error("the string ends with +")
        value = "".join(values) if all(v is not _UNKNOWN for v in values) else _UNKNOWN
        return " + ".join(codes), value

    # -- statements -------------------------------------------------------------------------------------------------

    def statement(self, st: Statement) -> Outcome:
        self.line = st.line
        try:
            st, notes = self._substitute(st)
            ops, note = self._dispatch(st)
            notes = [*notes, *([note] if note else [])]
            return Outcome(st, NOTE if notes else TRANSLATED, "; ".join(notes), ops)
        except _Out as out:
            self._unset(st)
            if not (st.kind == "command" and st.word in _HARMLESS or st.kind == "method" and st.word in PHASE4_METHODS
                    or self.types.get(st.obj.lower()) in ("graph", "table", "spool", "text")):
                self.forget()
            return Outcome(st, out.status, out.reason)

    def inside_block(self, st: Statement) -> None:
        """A line inside a loop, an if block or a subroutine: not translated, but what it sets is no longer known."""
        self._unset(st)

    def _unset(self, st: Statement) -> None:
        """A variable that a line left out would set: later lines that use it are left out too, saying why."""
        key = ""
        if st.kind == "assign" and st.lhs[:1] in "!%":
            key = st.lhs.lower()
        elif st.kind == "command" and st.word in ("scalar", "string"):
            key = _first_word(st.rest).split("=")[0].lower()
        if key:
            self.unset[key] = st.line
            self.py.pop(key, None)

    def _missing(self, name: str) -> _Out:
        """A name the program has not set: left out when a line left out sets it (or may), else an error."""
        key = name.lower()
        if key in self.unset:
            return _left(f"{name} is set by line {self.unset[key]}, which is left out")
        if self.lost:
            return _left(f"{name} is not set by a line translated so far (a loop, a subroutine or a line left out may set it): {PHASE3}")
        return _error(f"{name} is not set before this line")

    def forget(self) -> None:
        """After a line left out that may have made series (a loop, a subroutine, a fetch, an add-in), which series
        each page has is no longer known: names are no longer checked."""
        self.lost = True
        for page in self.pages.values():
            page.series = None

    def _dispatch(self, st: Statement) -> tuple[list[Op], str]:
        if st.kind == "assign":
            return self._assign(st)
        if st.kind == "method":
            return self._method(st)
        word = st.word
        if word in PHASE3_WORDS:
            raise _left(f"{PHASE3_WORDS[word]}: {PHASE3}")
        handler = getattr(self, f"_cmd_{word}", None)
        if handler is not None:
            return handler(st)
        if word in PHASE2_COMMANDS:
            if word in ("sspace",):
                self.types[_first_word(st.rest).split(".")[0].lower()] = word
            raise _left(left_reason("state-space objects are read by read_eviews_sspace, not translated line by line",
                                    PHASE2_COMMANDS[word]))
        if word in PHASE4_COMMANDS:
            why, eq = PHASE4_COMMANDS[word]
            if word in ("graph", "table", "spool", "text"):
                self.types[_first_word(st.rest).split(".")[0].lower()] = word
            raise _left(left_reason(why, eq))
        if word.startswith("{") or not word:
            raise _error("this does not read as an EViews statement")
        raise _left(f"{word} is not an EViews command this translator knows: an add-in or a user-defined command, "
                    "another program's code, is not translated")

    # workfiles and pages

    def _cmd_wfopen(self, st: Statement) -> tuple[list[Op], str]:
        args = _arguments(st.rest)
        if not args:
            raise _error("wfopen names a workfile")
        named = Path(args[0].strip('"').replace("\\", "/")).stem.lower()
        if self.bundle is None:
            raise _error("wfopen needs the workfile's data: give translate_program the bundle the export program wrote "
                         "(bundle=\".../<workfile>.evbundle\"), or a CSV or Excel file saved from the workfile in EViews")
        if self.opened is not None and self.opened != named:
            raise _error(f"a second workfile ({named}) after {self.opened}: one bundle or data file stands for one workfile")
        first = self.opened is None
        self.opened = named
        if not first:
            raise _left(f"{named} is open already (line before): nothing more to read")
        if str(self.bundle).lower().endswith(".evbundle"):
            return self._open_bundle(named)
        return self._open_file(named)

    def _open_bundle(self, named: str) -> tuple[list[Op], str]:
        from pymetrix.errors import PyMetrixError

        from ..bundle import _open, bundle_pages, eviews_sample

        try:
            manifest = _open(self.bundle).manifest
            infos = bundle_pages(self.bundle)
        except PyMetrixError as exc:
            raise _error(str(exc)) from None
        self.uses_bundle = True
        ops: list[Op] = []
        notes: list[str] = []
        wf = manifest.get("workfile", {}).get("name", "")
        if wf and wf.lower() != named:
            notes.append(f"the bundle is of workfile {wf}, the program opens {named}")
        active = str(manifest.get("workfile", {}).get("active_page", "")).lower()
        for p, info in zip(manifest["pages"], infos):
            if info["refused"]:
                notes.append(f"page {p['name']} is not read: {info['refused']}")
                continue
            var, note = self._name(p["name"], "page")
            if note:
                notes.append(note)
            code = str(p.get("frequency") or "").upper()
            freq = {"A": "Y", "Y": "Y", "Q": "Q", "M": "M", "W": "W", "D7": "D", "D": "D"}.get(code) if p["structure"] == "dated" else None
            rng = p.get("range") or [None, None]
            page = _Page(var, p["name"], freq, {_py_name(s["name"]) for s in p["series"]},
                         first=_period(str(rng[0]), freq) if rng[0] and freq else None,
                         last=_period(str(rng[-1]), freq) if rng[-1] and freq else None)
            self.pages[p["name"].lower()] = page
            ops.append(Step("pymetrix:dataset.read", None, var, "Dataset.read", (Arg("source", str(self.bundle), "pos"),
                                                                                  Arg("kw", {"page": p["name"]}, "varkw"))))
            saved = str(p.get("sample") or "").strip()
            if saved and saved.lower() != "@all":
                sample = eviews_sample(saved, code if code in ("Q", "M", "A") else None)
                if sample is None:
                    notes.append(f"page {p['name']}'s saved sample ({saved}) does not translate: all its rows are used")
                elif not sample.everything:
                    spec = SampleSpec(sample.start, sample.end)
                    page.sample = spec
                    ops.append(self._with_sample(page, spec))
                    notes.append(f"page {p['name']} opens with its saved sample, {saved}")
            for s in p.get("scalars", []):
                key = s["name"].lower()
                name, _ = self._name(key, f"scalar {key}")
                value = math.nan if s.get("value") is None else float(s["value"])
                self.py[key] = _Py(name, "scalar", value)
                ops.append(Code(f"{name} = {_number_text(value) if math.isfinite(value) else _NAN}  "
                                f"# the workfile's scalar {s['name']}", (name,)))
            for s in p.get("strings", []):
                key = s["name"].lower()
                name, _ = self._name(key, f"string {key}")
                self.py[key] = _Py(name, "string", str(s.get("value", "")))
                ops.append(Code(f"{name} = {quote(str(s.get('value', '')))}  # the workfile's string {s['name']}", (name,)))
            if any(p.get(k) for k in ("groups", "samples", "arrays", "equations", "models")):
                notes.append(f"page {p['name']}'s groups, samples, matrices, equations and models: eviews_objects({var})")
        if not ops:
            raise _error("the bundle has no page that reads")
        self.current = self.pages.get(active) or next(iter(self.pages.values()))
        return ops, "; ".join(notes)

    def _open_file(self, named: str) -> tuple[list[Op], str]:
        from pymetrix import Dataset
        from pymetrix.errors import PyMetrixError

        try:
            data = Dataset.read(str(self.bundle), **self.read_options)
        except (PyMetrixError, OSError, ValueError) as exc:
            raise _error(f"{self.bundle} does not read: {exc}") from None
        var, note = self._name(named, "page")
        t = data.time_axis
        freq = None
        if t is not None:
            code = str(t.freq).upper()
            freq = "Y" if code.startswith(("Y", "A")) else code[:1] if code[:1] in ("Q", "M", "W", "D") else None
        periods = data.frame.index.get_level_values(t.name) if t is not None else None
        page = _Page(var, named, freq, set(data.names), first=str(periods.min()) if periods is not None and len(periods) else None,
                     last=str(periods.max()) if periods is not None and len(periods) else None)
        self.pages[named] = page
        self.current = page
        step = Step("pymetrix:dataset.read", None, var, "Dataset.read",
                    (Arg("source", str(self.bundle), "pos"), *([Arg("kw", dict(self.read_options), "varkw")] if self.read_options else [])))
        notes = [f"the workfile {named} is read from {Path(str(self.bundle)).name}"] + ([note] if note else [])
        return [step], "; ".join(notes)

    def _new_page(self, st: Statement, name: str, freq_text: str, start: str, end: str) -> tuple[list[Op], str]:
        f = freq_text.lower()
        if f in _FREQ_GAP:
            raise _left(_FREQ_GAP[f])
        freq = _FREQ.get(f)
        if freq is None:
            raise _error(f"{freq_text!r} is not a frequency read here: a, q, m, w or d")
        if freq not in ("Y", "Q", "M"):
            raise _left("weekly and daily pages are not translated yet: their dates are written differently "
                        "(Dataset.empty(\"W\" or \"D\", start, end))")
        first, last = _period(start, freq), _period(end, freq)
        if first is None or last is None:
            raise _error(f"{start} to {end} are not dates of frequency {freq_text}: 1990q1, 1990:1, 1990m3 or 1990")
        var, note = self._name(name, "page")
        page = _Page(var, name, freq, set(), first=first, last=last)
        self.pages[name.lower()] = page
        self.current = page
        return [Step("pymetrix:dataset.empty", None, var, "Dataset.empty",
                     (Arg("freq", freq, "pos"), Arg("start", first, "pos"), Arg("end", last, "pos")))], note

    def _cmd_wfcreate(self, st: Statement) -> tuple[list[Op], str]:
        options = _option_map(st.options)
        args = _arguments(st.rest)
        name = options.get("page") or options.get("wf")
        if len(args) == 4:
            name = name or args[0]
            args = args[1:]
        if len(args) == 2 and args[0].lower() == "u":
            raise _left(_FREQ_GAP["u"])
        if len(args) != 3:
            raise _error("wfcreate takes a frequency and the first and last dates: wfcreate q 1990q1 2030q4")
        self.workfile_frequency = args[0]
        return self._new_page(st, name or "untitled", *args)

    def _cmd_pagecreate(self, st: Statement) -> tuple[list[Op], str]:
        options = _option_map(st.options)
        args = _arguments(st.rest)
        if len(args) == 2 and args[0].lower() == "u":
            raise _left(_FREQ_GAP["u"])
        if len(args) != 3:
            raise _error("pagecreate takes a frequency and the first and last dates: pagecreate(page=a) a 1990 2030")
        name = options.get("page")
        note = ""
        if not name:
            name = f"untitled{len(self.pages) + 1:02d}"
            note = f"the new page has no name (page=): it is called {name}"
        ops, n = self._new_page(st, name, *args)
        return ops, "; ".join(x for x in (note, n) if x)

    def _cmd_pageselect(self, st: Statement) -> tuple[list[Op], str]:
        args = _arguments(st.rest)
        if len(args) != 1:
            raise _error("pageselect names one page")
        self.current = self._page_named(args[0])
        return [], f"later lines use {self.current.var} (no step)"

    def _save(self, st: Statement) -> tuple[list[Op], str]:
        raise _left("the project is saved by the user: save the project (File → Save)")

    _cmd_wfsave = _cmd_pagesave = _save

    def _cmd_wfclose(self, st: Statement) -> tuple[list[Op], str]:
        raise _left("closing the workfile: nothing to do (the session keeps its objects; save the project with File → Save)")

    def _cmd_read(self, st: Statement) -> tuple[list[Op], str]:
        raise _left(left_reason(f"{st.word} adds a file's series to the page; its options are not translated",
                                "Dataset.read(path, time=..., freq=...), then add its series to the page"))

    _cmd_import = _cmd_read

    def _database(self, st: Statement) -> tuple[list[Op], str]:
        raise _left("EViews databases are not read: export the database's series to the bundle (fetch them into the "
                    "workfile, then run the export program)")

    _cmd_fetch = _cmd_store = _cmd_dbopen = _cmd_dbclose = _cmd_dbcreate = _cmd_db = _database

    def _page_op(self, st: Statement) -> tuple[list[Op], str]:
        raise _left(left_reason(f"{st.word}: page operations are not translated", "a Dataset per page; xs, keep, to_freq"))

    _cmd_pagestruct = _cmd_pagedelete = _cmd_pagerename = _cmd_pagecopy = _cmd_pageappend = _cmd_pagecontract = _page_op
    _cmd_pagestack = _cmd_pageunstack = _cmd_wfstruct = _page_op

    def _cmd_copy(self, st: Statement) -> tuple[list[Op], str]:
        raise _left(left_reason("copying between pages is not translated", "Dataset.to_freq(freq, how=...) (Convert frequency)"))

    # samples

    def _sample_spec(self, text: str, at: int, page: _Page) -> Any:
        """An EViews sample (``1990q1 2007q4``, ``@all``, ``@first 2000q4 if x > 0``, a named sample) as a
        ``SampleSpec``, a ``Var`` or None (all rows)."""
        body, cond = text, None
        m = re.search(r"(?i)(?:^|\s)if\s", text)
        if m:
            body, cond = text[:m.start()], text[m.end():]
            cond_at = at + m.end()
        tokens = re.sub(r"\s*([+-])\s*", r"\1", body).split()
        where = self.condition(cond, cond_at, page) if cond is not None else None
        if len(tokens) == 1 and tokens[0].lower() not in ("@all",) and tokens[0].lower() in self.py:
            v = self.py[tokens[0].lower()]
            if v.kind != "sample":
                raise _error(f"{tokens[0]} is a {v.kind}, not a sample")
            if where is not None:
                raise _left("a named sample with a condition: write the sample out")
            return Var(v.name)
        if not tokens or [t.lower() for t in tokens] == ["@all"]:
            return SampleSpec(where=where) if where is not None else None
        if len(tokens) != 2:
            raise _left("a sample of several pairs of dates is not translated: one pair, with a condition, is "
                        "(Sample(...) | Sample(...) combines them)")
        ends = []
        for i, tok in enumerate(tokens):
            ends.append(self._end(tok, page, i == 0))
        if ends == [None, None] and where is None:
            return None
        return SampleSpec(ends[0], ends[1], where)

    def _end(self, token: str, page: _Page, first: bool) -> str | None:
        t = token.lower()
        m = re.fullmatch(r"(@first|@last|[^+\-]+)((?:[+-](?:\d+|!\w+))*)", t)
        if m is None:
            raise _error(f"{token} is not a date")
        base, n = m.group(1), 0
        for sign, amount in re.findall(r"([+-])(\d+|!\w+)", m.group(2)):
            if amount.startswith("!"):
                value = self._known(amount)
                if value is _UNKNOWN or not float(value).is_integer():
                    raise self._missing(amount) if value is _UNKNOWN else _error(f"{amount} is not a whole number")
                amount = str(int(value))
            n += int(amount) if sign == "+" else -int(amount)
        if page.freq not in ("Y", "Q", "M"):
            raise _left("samples by date are translated for annual, quarterly and monthly pages")
        if base in ("@first", "@last"):
            if not n:
                return None
            known = page.first if base == "@first" else page.last
            if known is None:
                raise _left(f"{token}: the page's {base[1:]} period is not known here: write the date")
            return _shifted(known, page.freq, n)
        period = _period(base, page.freq)
        if period is None:
            if re.fullmatch(r"\d{4}", base) and page.freq in ("Q", "M"):  # a year: its first or last period
                period = _shifted(f"{base}Q1" if page.freq == "Q" else f"{base}-01", page.freq, 0 if first else (3 if page.freq == "Q" else 11))
            else:
                raise _error(f"{token} is not a date of page {page.eviews} ({page.freq})")
        return _shifted(period, page.freq, n) if n else period

    def _with_sample(self, page: _Page, spec: Any) -> Step:
        return Step("pymetrix:dataset.with_sample", page.var, page.var, f"{page.var}.with_sample", (Arg("sample", spec, "pos"),))

    def _cmd_smpl(self, st: Statement) -> tuple[list[Op], str]:
        page = self._page()
        spec = self._sample_spec(st.rest, st.rest_at, page)
        page.sample = spec
        return [self._with_sample(page, spec)], ""

    def _cmd_sample(self, st: Statement) -> tuple[list[Op], str]:
        name = _first_word(st.rest)
        if not name:
            raise _error("sample names the sample: sample s 1990q1 2007q4")
        page = self._page()
        rest = st.rest[len(name):]
        spec = self._sample_spec(rest, st.rest_at + len(name), page) if rest.strip() else None
        var, note = self._name(name, f"sample {name.lower()}")
        self.py[name.lower()] = _Py(var, "sample", page=page.var)
        if isinstance(spec, Var):
            return [Code(f"{var} = {spec.name}", (var,))], note
        spec = spec if spec is not None else SampleSpec()
        return [Code(f"{var} = {spec.code()}", (var,))], note

    # series

    def _assign_series(self, name: str, rhs: str, at: int, declare: bool = False) -> tuple[list[Op], str]:
        page = self._page()
        if "\\" in name or not re.fullmatch(r"[A-Za-z_]\w*", name):
            raise _left(f"{name}: assigning to anything but a series name (a page\\series, an element) is not translated")
        series = _py_name(name)
        if series == "within":
            raise _left("a series named within: with_series takes within= for the sample; rename the series "
                        "(or add it with data.add_series)")
        notes = [f"{name} is called {series}"] if series != name.lower() else []
        if rhs.strip().lower() == "na":
            value: Any = None
        else:
            value = self.expression(rhs, at, page)
        args: list[Arg] = [Arg("exprs", {series: value}, "varkw")]
        own = value is not None and _uses(str(value), series)
        within = page.sample
        if own and within is None:
            within = SampleSpec()
            notes.append("it uses its own values, so it is worked out period by period, as EViews does")
        elif own:
            notes.append("it uses its own values: worked out period by period within the sample, as EViews does")
        if within is not None:
            args.append(Arg("within", within))
        if page.series is not None:
            page.series.add(series)
        return [Step("pymetrix:dataset.with_series", page.var, page.var, f"{page.var}.with_series", tuple(args))], "; ".join(notes)

    def _cmd_series(self, st: Statement) -> tuple[list[Op], str]:
        m = re.fullmatch(r"\s*([^\s=]+)\s*(?:=(.*))?", st.rest, re.S)
        if m is None:
            raise _error(f"{st.word} names a series: series y = x")
        name, rhs = m.group(1), m.group(2)
        if rhs is None:
            if st.word == "genr":
                raise _error("genr assigns a formula: genr y = x")
            return self._assign_series(name, "na", st.rest_at, declare=True)
        return self._assign_series(name, rhs, st.rest_at + m.start(2))

    _cmd_genr = _cmd_series

    def _cmd_frml(self, st: Statement) -> tuple[list[Op], str]:
        m = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*=(.*)", st.rest, re.S)
        if m is None:
            raise _error("frml names a series and its formula: frml y = x")
        page = self._page()
        series = _py_name(m.group(1))
        value = self.expression(m.group(2), st.rest_at + m.start(2), page)
        if page.series is not None:
            page.series.add(series)
        return [Step("pymetrix:dataset.derive", page.var, page.var, f"{page.var}.derive", (Arg("recipes", {series: value}, "varkw"),))], \
            "a derived series: it follows its inputs (Keep updated), over all rows, as a formula series ignores the sample"

    def _cmd_alpha(self, st: Statement) -> tuple[list[Op], str]:
        raise _left("alpha (text) series are not translated: a text column comes with the data (a gap for assigning text)")

    def _cmd_group(self, st: Statement) -> tuple[list[Op], str]:
        words = _arguments(st.rest)
        if not words:
            raise _error("group names the group: group g a b c")
        page = self._page()
        return self._group(words[0], words[1:], page, "")

    def _group(self, name: str, members: list[str], page: _Page, note: str) -> tuple[list[Op], str]:
        for m in members:
            if not re.fullmatch(r"[A-Za-z_][\w*?]*|[*?][\w*?]*", m):
                raise _left(f"{m}: a group of expressions is not translated: a Selection holds series; make the series first")
        series = [_py_name(m) if "*" not in m and "?" not in m else m.lower() for m in members]
        if page.series is not None:
            missing = [m for m in series if "*" not in m and "?" not in m and m not in page.series]
            if missing:
                raise _error(f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not series of page {page.eviews}")
        key = name.lower()
        var, n = self._name(name, f"group {key}")
        self.py[key] = _Py(var, "group", members=series, page=page.var)
        notes = "; ".join(x for x in (note, n) if x)
        return [Step("pymetrix:dataset.series", page.var, var, f"{page.var}.series", (Arg("patterns", tuple(series), "star"),))], notes

    def _cmd_rename(self, st: Statement) -> tuple[list[Op], str]:
        args = _arguments(st.rest)
        if len(args) != 2 or any(ch in "".join(args) for ch in "*?"):
            raise _left("rename of several names or a pattern is not translated: data.rename(old=\"new\") for each")
        old, new = args
        page = self._page()
        if old.lower() in self.py:
            raise _left(f"{old} is a {self.py[old.lower()].kind}: renaming the program's variables is not translated")
        o, n = _py_name(old), _py_name(new)
        if page.series is not None:
            if o not in page.series:
                raise _error(f"{old} is not a series of page {page.eviews}")
            page.series.discard(o)
            page.series.add(n)
        return [Step("pymetrix:dataset.rename", page.var, page.var, f"{page.var}.rename", (Arg("names", {o: n}, "varkw"),))], ""

    def _cmd_delete(self, st: Statement) -> tuple[list[Op], str]:
        args = _arguments(st.rest)
        if not args or any(ch in "".join(args) for ch in "*?\\"):
            raise _left("delete with a pattern or a page is not translated: data.drop(\"a\", \"b\") with the names")
        page = self._page()
        series, gone = [], []
        for a in args:
            key = a.lower()
            if key in self.py:
                gone.append(self.py.pop(key).name)
            elif key in self.types:
                del self.types[key]
                gone.append(key)
            else:
                s = _py_name(a)
                if page.series is not None and s not in page.series:
                    raise _error(f"{a} is not a series of page {page.eviews}, nor an object of the program")
                series.append(s)
        ops: list[Op] = []
        if series:
            if page.series is not None:
                page.series -= set(series)
            ops.append(Step("pymetrix:dataset.drop", page.var, page.var, f"{page.var}.drop", (Arg("names", tuple(series), "star"),)))
        note = f"{', '.join(gone)}: not series, so nothing to remove in the script" if gone else ""
        return ops, note

    # scalars, strings, matrices

    def _scalar(self, name: str, rhs: str | None, at: int, kind: str) -> tuple[list[Op], str]:
        key = name.lower()
        old = self.py.get(key)
        if old is not None and old.kind not in ("scalar", "control"):
            raise _error(f"{name} is a {old.kind}, not a number")
        var, note = (old.name, "") if old is not None else self._name(name, f"{kind} {key}")
        if rhs is None:
            if old is None:
                self._declared[key] = self.line
            self.py[key] = _Py(var, kind, math.nan)
            return [Code(f"{var} = {_NAN}", (var,))], "; ".join(x for x in (note, "a new scalar has no value: NA (verify)") if x)
        stat = re.fullmatch(r"\s*(@\w+)\s*\((.*)\)\s*", rhs, re.S)
        if stat and stat.group(1).lower() in STATISTICS and _balanced(stat.group(2)):
            page = self._page()
            fn = STATISTICS[stat.group(1).lower()]
            inner = self.expression(stat.group(2), at + stat.start(2), page)
            self.py[key] = _Py(var, kind, _UNKNOWN)
            code = f"{var} = float({page.var}.eval({inner.code()})[{page.var}.sample_mask()].{fn}())"
            why = (f"{stat.group(1).lower()} over the page's sample, missing values skipped"
                   + (" (the standard deviation with n - 1, as EViews's @stdev)" if fn == "std" else ""))
            return [Code(code, (var,))], "; ".join(x for x in (note, why) if x)
        rhs, members, member_notes = self.members_in(rhs)
        for placeholder, member in members.items():
            self.py[placeholder] = _Py(member, "scalar", _UNKNOWN)
        try:
            code, value, worked = self.number(rhs, at)
        finally:
            for placeholder in members:
                self.py.pop(placeholder, None)
        if members:
            code = f"float({code})"
        if old is None:
            self._declared[key] = self.line
        self.py[key] = _Py(var, kind, value)
        why = f"worked out: {rhs.strip()}" if worked else ""
        return [Code(f"{var} = {code}", (var,))], "; ".join(x for x in (note, why, *member_notes) if x)

    def _cmd_scalar(self, st: Statement) -> tuple[list[Op], str]:
        m = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*(?:=(.*))?", st.rest, re.S)
        if m is None:
            raise _error("scalar names the scalar: scalar s = 0.3")
        return self._scalar(m.group(1), m.group(2), st.rest_at + (m.start(2) if m.group(2) is not None else 0), "scalar")

    def _string(self, key: str, name: str, rhs: str | None) -> tuple[list[Op], str]:
        old = self.py.get(key)
        if old is not None and old.kind != "string":
            raise _error(f"{name} is a {old.kind}, not a string")
        var, note = (old.name, "") if old is not None else self._name(name, f"string {key}")
        if rhs is None:
            self.py[key] = _Py(var, "string", "")
            return [Code(f'{var} = ""', (var,))], note
        code, value = self.string(rhs)
        self.py[key] = _Py(var, "string", value)
        return [Code(f"{var} = {code}", (var,))], note

    def _cmd_string(self, st: Statement) -> tuple[list[Op], str]:
        m = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*(?:=(.*))?", st.rest, re.S)
        if m is None:
            raise _error("string names the string: string s = \"abc\"")
        return self._string(m.group(1).lower(), m.group(1), m.group(2))

    def _matrix(self, st: Statement, kind: str) -> tuple[list[Op], str]:
        size = [s.strip() for s in (st.options or "").split(",") if s.strip()]
        m = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*(?:=(.*))?", st.rest, re.S)
        if m is None:
            raise _error(f"{kind} names the object: {kind}(3) v")
        if m.group(2) is not None:
            raise _left(f"{kind} filled from an expression is not translated: Matrix arithmetic (Model → Matrix)")
        try:
            dims = [int(self._static_int(s)) for s in size]
        except ValueError:
            raise _left(f"{kind}'s size is known only when the program runs: {PHASE3}") from None
        if kind == "matrix":
            rows, cols = (dims + [1, 1])[:2] if dims else (1, 1)
        elif kind == "sym":
            rows = cols = dims[0] if dims else 1
        else:
            rows, cols = (dims[0] if dims else 1), 1
        if kind == "coef":
            self.types[m.group(1).lower()] = "coef"
            return [], f"a coefficient vector: an expression specification's {m.group(1).lower()}(i) are its parameters {m.group(1).lower()}_i (no step)"
        key = m.group(1).lower()
        var, note = self._name(m.group(1), f"matrix {key}")
        self.py[key] = _Py(var, "matrix", shape=(rows, cols))
        extra = "; a sym is a full matrix here: keep it symmetric" if kind == "sym" else ""
        return [Code(f"{var} = Matrix.zeros({rows}, {cols})", (var,))], (note + extra).strip("; ")

    def _static_int(self, text: str) -> float:
        if re.fullmatch(r"\d+", text):
            return float(text)
        if text.startswith("!") and self._known(text) is not _UNKNOWN:
            return float(self._known(text))
        raise ValueError(text)

    def _cmd_matrix(self, st: Statement) -> tuple[list[Op], str]:
        return self._matrix(st, st.word)

    _cmd_vector = _cmd_sym = _cmd_coef = _cmd_rowvector = _cmd_matrix

    # assignments

    def _assign(self, st: Statement) -> tuple[list[Op], str]:
        lhs = st.lhs
        if lhs.startswith("!"):
            return self._scalar(lhs, st.rest, st.rest_at, "control")
        if lhs.startswith("%"):
            return self._string(lhs.lower(), lhs, st.rest)
        if "(" in lhs:
            name = lhs.split("(")[0].lower()
            if self.types.get(name) in ("table",):
                why, eq = PHASE4_COMMANDS["table"]
                raise _left(left_reason(why, eq))
            if self.py.get(name) and self.py[name].kind == "matrix":
                raise _left("assigning one element of a matrix is not translated: Matrix.set (Model → Matrix)")
            raise _left(f"{lhs}: assigning to a lag or an element is not translated")
        key = lhs.lower()
        v = self.py.get(key)
        if v is not None:
            if v.kind in ("scalar", "control"):
                return self._scalar(lhs, st.rest, st.rest_at, v.kind)
            if v.kind == "string":
                return self._string(key, lhs, st.rest)
            raise _left(f"{lhs} is a {v.kind}: assigning to it is not translated")
        if '"' in st.rest:  # a string object, as EViews makes one from a string's value
            return self._string(key, lhs, st.rest)
        if key in self.types:
            raise _left(f"{lhs} is a {self.types[key]}: assigning to it is not translated")
        return self._assign_series(lhs, st.rest, st.rest_at)

    # methods

    def _method(self, st: Statement) -> tuple[list[Op], str]:
        obj, method = st.obj, st.word
        key = obj.lower()
        kind = self.py[key].kind if key in self.py else self.types.get(key, "")
        if kind == "group" and method in ("add", "drop"):
            g = self.py[key]
            words = _arguments(st.rest)
            members = list(g.members)
            for w in words:
                s = _py_name(w)
                if method == "add" and s not in members:
                    members.append(s)
                elif method == "drop" and s in members:
                    members.remove(s)
            page = next((p for p in self.pages.values() if p.var == g.page), self._page())
            del self.taken[g.name]
            return self._group(obj, members, page, f"the group made again with {'more' if method == 'add' else 'fewer'} series")
        if kind == "sample" and method == "set":
            page = self._page()
            spec = self._sample_spec(st.rest, st.rest_at, page)
            v = self.py[key]
            return [Code(f"{v.name} = {(spec or SampleSpec()).code() if not isinstance(spec, Var) else spec.name}", (v.name,))], ""
        if kind == "matrix" and method == "fill":
            return self._fill(self.py[key], st)
        if method == "fill" and not kind:
            return self._series_fill(st)
        if method in ("label", "displayname", "setattr") and not kind:
            return self._label(st)
        if key in self.models:
            return self._model_method(st, key)
        if key in self.systems:
            return self._system_method(st, key)
        if kind == "group" and method == "coint":
            return self._coint(st, self.py[key])
        if method == "uroot" and not kind:
            return self._uroot(st)
        if kind in ("equation", "var") or (not kind and (method in ESTIMATORS or method in LEFT_ESTIMATORS)):
            return self._fit_method(st, key)
        if kind in ("graph", "table", "spool", "text") or (method in PHASE4_METHODS and kind not in PHASE2_COMMANDS):
            why, eq = PHASE4_METHODS.get(method, PHASE4_COMMANDS.get(kind, ("not translated", "")))
            raise _left(left_reason(why, eq))
        if method in _OBJECT_METHODS and not kind:
            if self.lost:
                raise _left(f"{obj} is not made by a line translated so far (a subroutine, an add-in or a line left out may make "
                            f"it): {PHASE3}")
            raise _error(f"{obj} is not an object of the program: {obj}.{method} needs a model, an equation or a system")
        raise _left(f"{obj}.{method} is not translated" + (f" (a {kind})" if kind else ""))

    def _label(self, st: Statement) -> tuple[list[Op], str]:
        page = self._page()
        series = _py_name(st.obj)
        if page.series is not None and series not in page.series:
            raise _error(f"{st.obj} is not a series of page {page.eviews}")
        text = st.rest.strip()
        if text.startswith('"') and text.endswith('"') and len(text) > 1:
            text = text[1:-1]
        option = (st.options or "").strip().lower()
        if st.word == "displayname":
            info, note = {"label": text}, ""
        elif st.word == "setattr":
            field_ = {"units": "units", "source": "source", "description": "description", "unit": "units"}.get(option)
            if field_ is None:
                raise _left(f"the attribute {option!r} is not a field of a series' description: Variable.extra")
            info, note = {field_: text}, ""
        else:
            fields = {"d": "description", "u": "units", "s": "source"}
            if option == "c":
                info, note = {"label": "", "description": "", "units": "", "source": ""}, "the series' description cleared"
            elif option in fields:
                info, note = {fields[option]: text}, ""
            elif option == "r":
                raise _left("remarks have no field in a series' description: Variable.extra[\"remarks\"]")
            elif not option:
                info, note = {"description": text}, "label with no option read as the description (verify)"
            else:
                raise _left(f"label({option}) is not translated: d, u, s and c are")
        return [Step("pymetrix:dataset.with_info", page.var, page.var, f"{page.var}.with_info",
                     (Arg("name", series, "pos"), Arg("info", info, "varkw")))], note

    def _series_fill(self, st: Statement) -> tuple[list[Op], str]:
        page = self._page()
        series = _py_name(st.obj)
        if page.series is not None and series not in page.series:
            raise _error(f"{st.obj} is not a series of page {page.eviews}")
        options = _option_map(st.options)
        flags = [o.strip().lower() for o in (st.options or "").split(",") if o.strip() and "=" not in o]
        if flags or set(options) - {"o"}:
            raise _left(f"fill({st.options}) is not translated: values from a date (o=) are")
        start = options.get("o")
        note = ""
        if start is None:
            if page.first is None:
                raise _left("fill from the page's first period, which is not known here: give o=")
            first, note = page.first, "filled from the page's first period (verify: EViews's default)"
        else:
            first = self._end(start, page, True)
            if first is None:
                raise _error(f"o={start} is not a date")
        values: dict[str, Any] = {}
        for i, tok in enumerate(t.strip() for t in st.rest.split(",")):
            try:
                number = float(tok)
            except ValueError:
                if tok.lower() != "na":
                    raise _left("fill with expressions is not translated: numbers are") from None
                number = math.nan
            values[_shifted(first, page.freq or "Q", i)] = None if math.isnan(number) else number
        return [Step("pymetrix:dataset.edit", page.var, page.var, f"{page.var}.edit", (Arg("values", {series: values}, "pos"),))], \
            "; ".join(x for x in (note, "the values are marked edited") if x)

    def _fill(self, v: _Py, st: Statement) -> tuple[list[Op], str]:
        options = (st.options or "").replace(" ", "").lower()
        by_row = options in ("b=r", "byrow")
        if options and not by_row and options not in ("b=c",):
            raise _left(f"fill({st.options}) is not translated: by column and by row (b=r) are")
        values = []
        for tok in st.rest.split(","):
            tok = tok.strip()
            try:
                values.append(float(tok))
            except ValueError:
                if tok.lower() == "na":
                    values.append(math.nan)
                else:
                    raise _left("fill with expressions is not translated: numbers are") from None
        rows, cols = v.shape or (len(values), 1)
        if len(values) != rows * cols:
            raise _left(f"fill with {len(values)} values for a {rows} by {cols} matrix (EViews fills part, or repeats): "
                        "not translated")
        grid = [[values[r * cols + c] if by_row else values[c * rows + r] for c in range(cols)] for r in range(rows)]
        text = "[" + ", ".join("[" + ", ".join(_number_text(x) if math.isfinite(x) else _NAN for x in row) + "]"
                               for row in grid) + "]"
        return [Code(f"{v.name} = Matrix({text})", (v.name,))], "" if by_row else "filled by column, as EViews does"


# methods of models, equations and systems: on a name the program has not made, the object comes from a line left out
_OBJECT_METHODS = frozenset({"solve", "solveopt", "addassign", "addinit", "scenario", "override", "exclude", "append", "merge",
                             "makegroup", "wald", "auto", "hettest", "archtest", "reset", "chow", "hist", "rls", "testdrop",
                             "testadd", "forecast", "fit", "makeresids", "ec", "sur", "3sls", "fiml", "makemodel", "results", "spec",
                             "output"})


def _top_level(text: str, ch: str) -> list[int]:
    """Where ``ch`` is outside brackets (an ``=`` not part of ``<=``, ``>=`` or ``<>``)."""
    out, depth = [], 0
    for i, c in enumerate(text):
        depth += (c == "(") - (c == ")")
        if c == ch and depth == 0 and text[i - 1:i] not in ("<", ">") and text[i + 1:i + 2] not in (">",):
            out.append(i)
    return out


def _uses(expr: str, name: str) -> bool:
    return re.search(rf"(?<![\w.]){re.escape(name)}(?!\w)(?!\s*\()", expr) is not None


def _balanced(text: str) -> bool:
    depth = 0
    for ch in text:
        depth += ch == "("
        depth -= ch == ")"
        if depth < 0:
            return False
    return depth == 0


def _first_word(text: str) -> str:
    m = re.match(r"\s*([^\s(]+)", text)
    return m.group(1) if m else ""


def _arguments(text: str) -> list[str]:
    """Words separated by spaces, a quoted text one word."""
    return [a for a in re.findall(r'"[^"]*"|\S+', text)]


def _option_map(options: str | None) -> dict[str, str]:
    out = {}
    for part in (options or "").split(","):
        if "=" in part:
            k, _, v = part.partition("=")
            out[k.strip().lower()] = v.strip().strip('"')
    return out


def is_opener(st: Statement) -> str:
    """The block a statement opens (``for``, ``if``, ``while``, ``subroutine``), or ''."""
    return st.word if st.kind == "command" and st.word in _OPENERS else ""


def is_closer(st: Statement) -> bool:
    return st.kind == "command" and st.word in _CLOSERS

