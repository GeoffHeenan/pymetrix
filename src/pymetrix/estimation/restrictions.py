"""Restrictions on coefficients written as text: ``"a + b = 1, c = 2 * d"``.

``parse_restrictions`` reads each restriction into an expression in the named coefficients; system estimation
(``pymetrix.systems``) uses it for its cross-equation restrictions, the Wald test (``fit.test("wald", ...)``) for its
hypotheses, and ``linear_restrictions`` turns linear ones into ``R b = r`` for restricted least squares
(``OLS(restrictions=...)``, ``IV(restrictions=...)``).

Coefficients are named as the fit names them: ``const``, a linear term's label (``y``, ``x[-1]``, ``dlog(y)``), an
explicit-form parameter, a system's ``equation.coefficient``, or a column name of ``pymetrix.ols(y, X)`` (``log(P)``,
``L(x, 1)``). A name is matched whole, longest first, with any spaces in it optional (``L(x,1)`` is ``L(x, 1)``); a
name in backticks (`` `dlog(y)` ``) is always read as one coefficient.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..errors import SpecificationError
from ..expr import Coef, Expr, ExpressionError, diff, evaluate, parse_equation

__all__ = ["linear_restrictions", "parse_restrictions", "split_restrictions"]


def split_restrictions(restrictions: str | Sequence[str] | None) -> list[str]:
    """The restrictions one by one: text is split at its top-level commas, a list item by item (each split likewise)."""
    from ..stats.equation_tests import _split_top_level

    if restrictions is None:
        return []
    items = [restrictions] if isinstance(restrictions, str) else list(restrictions)
    return [part for item in items for part in _split_top_level(str(item))]


def parse_restrictions(restrictions: str | Sequence[str] | None, names: Sequence[str]) -> list[tuple[str, Expr]]:
    """Each restriction as written, with the expression ``lhs - rhs`` in the coefficients ``names`` (zero when it
    holds). A restriction that cannot be read, or that names something other than a coefficient, raises an error
    naming it."""
    names = list(names)
    internal = {n: f"__coef{j}" for j, n in enumerate(names)}
    external = {v: k for k, v in internal.items()}
    spaced = {_spaced(n): n for n in names}  # one regex per name, its spaces optional
    pattern = (re.compile(r"(?<![\w.])(?:" + "|".join(f"({p})" for p in sorted(spaced, key=len, reverse=True)) + r")(?![\w.])")
               if names else None)
    order = sorted(spaced, key=len, reverse=True)
    out = []
    for part in split_restrictions(restrictions):
        text = _backticked(part, names, internal)
        if pattern is not None:
            text = pattern.sub(lambda m: internal[spaced[order[m.lastindex - 1]]], text)
        try:
            lhs, op, rhs = parse_equation(text, list(internal.values()))
        except ExpressionError:
            raise ExpressionError(f"could not read the restriction {part!r}: write it as an equation in the coefficients, "
                                  f"e.g. {_example(names)}; coefficients are {names}") from None
        if op != "=":
            raise ExpressionError(f"the restriction {part!r} needs '=', not '{op}'")
        expr = (lhs - rhs).map(lambda node: Coef(external.get(node.id, node.id)) if isinstance(node, Coef) else None)
        if expr.variables():
            raise ExpressionError(f"unknown coefficient(s) {sorted(expr.variables())} in restriction {part!r}; "
                                  f"coefficients are {names}")
        out.append((part, expr))
    return out


def linear_restrictions(restrictions: str | Sequence[str] | None, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """``(R, r, texts)`` with ``R b = r`` the restrictions on the coefficients ``names``, one row each. Each must be linear
    in the coefficients, involve at least one, and add something the others do not say (R has full row rank)."""
    names = list(names)
    parsed = parse_restrictions(restrictions, names)
    R, r = np.zeros((len(parsed), len(names))), np.zeros(len(parsed))
    nowhere = lambda *_: np.nan  # noqa: E731 - restrictions involve no data
    for i, (part, expr) in enumerate(parsed):
        for j, name in enumerate(names):
            d = diff(expr, Coef(name))
            if d.params():
                raise SpecificationError(f"the restriction {part!r} is not linear in the coefficients: write it as a linear "
                                         f"equation, e.g. {_example(names)}")
            R[i, j] = float(evaluate(d, nowhere, {}))
        r[i] = -float(evaluate(expr, nowhere, {n: 0.0 for n in names}))
        if not np.any(R[i]):
            raise SpecificationError(f"the restriction {part!r} involves no coefficient")
    if parsed and np.linalg.matrix_rank(R) < len(parsed):
        raise SpecificationError(f"the restrictions {[p for p, _ in parsed]} repeat or contradict one another: "
                                 "give each only once")
    return R, r, [p for p, _ in parsed]


def restricted(beta: np.ndarray, bread: np.ndarray, R: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The restricted estimate ``b - A R' (R A R')^-1 (R b - r)`` with ``A = bread`` (``(X'X)^-1``, or ``(Xhat'Xhat)^-1``
    for two-stage least squares), and ``M = I - A R' (R A R')^-1 R``, which maps the unrestricted covariance's sandwich
    to the restricted one (``M V M'``)."""
    AR = bread @ R.T
    K = AR @ np.linalg.inv(R @ AR)
    return beta - K @ (R @ beta - r), np.eye(len(beta)) - K @ R


def normalise(restrictions: Any) -> tuple[str, ...] | None:
    """A method's ``restrictions`` as stored: a tuple of the restrictions one by one, or None for none."""
    parts = split_restrictions(restrictions)
    return tuple(parts) or None


def _spaced(name: str) -> str:
    """A regex for ``name`` with each run of spaces in it optional, and optional spaces after its commas."""
    out = []
    for piece in re.split(r"(\s+)", name):
        out.append(r"\s*" if piece.isspace() else re.escape(piece).replace(",", r",\s*"))
    return "".join(out)


def _backticked(part: str, names: Sequence[str], internal: dict[str, str]) -> str:
    """``part`` with each `` `name` `` replaced by the coefficient's placeholder; an unknown quoted name is an error."""
    def one(m: re.Match[str]) -> str:
        inner = m.group(1).strip()
        if inner in internal:
            return internal[inner]
        loose = [n for n in names if re.sub(r"\s+", "", n) == re.sub(r"\s+", "", inner)]
        if len(loose) == 1:
            return internal[loose[0]]
        raise ExpressionError(f"unknown coefficient `{inner}` in {part!r}; coefficients are {list(names)}")

    return re.sub(r"`([^`]*)`", one, part)


def _example(names: Sequence[str]) -> str:
    named = [n for n in names if n != "const"][:2]
    return f"{named[0]} + {named[1]} = 1" if len(named) == 2 else (f"{named[0]} = 1" if named else "a + b = 1")
