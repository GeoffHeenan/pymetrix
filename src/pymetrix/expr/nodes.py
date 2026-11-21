"""Expression tree nodes.

One tree serves data transformations, regression formulas, model equations,
restrictions and analytic derivatives. Nodes are immutable. After parsing,
every lag sits on a ``Name`` (``x[-1]`` is ``Name("x", -1)``), and difference
functions such as ``dlog`` are expanded into primitive operations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

from ..errors import SpecificationError


class ExpressionError(SpecificationError):
    """An expression or equation could not be parsed, evaluated or used."""


PRIMITIVES = frozenset({"log", "exp", "sqrt", "abs", "sign", "step", "not"})  # step(x): 1 where x >= 0, else 0; not(x): 1 where x is 0
# Comparisons and logical connectives give 1 where true and 0 where false (missing where an operand is missing), so a
# condition is a 0/1 series: ``recession = gy < 0``, ``both = (u > 5) and (infl > 2)``.
COMPARISONS = frozenset({"<", "<=", ">", ">=", "==", "!="})
LOGICAL = frozenset({"and", "or"})
# ``max(a, b, ...)`` and ``min(a, b, ...)`` with two or more arguments are period by period (an effective lower bound:
# ``max(i_rule, i_floor)``); with one argument they are group-wise statistics, below. Stored as nested pairs.
ELEMENTWISE = frozenset({"max", "min"})
MARKERS = frozenset({"fe", "C", "seasonal"})  # only valid as terms of a linear-form equation
# Group-wise functions of series: ``mean(y, country)`` (or ``mean(y, by=country)``) is each country's mean of y, on
# every row; without a group, over all rows. ``max`` and ``min`` group only with ``by=``. They need series (pandas), so work in data.eval, derived series
# and linear-form regressors, not inside model solvers.
GROUPWISE = frozenset({"mean", "sum", "min", "max", "sd", "median", "count", "demean", "share", "rank"})
# Moving-window statistics over the time axis (see pymetrix.expr.functions): the name and how many series each takes
# (the window length comes last). Stored with every lag explicit: movavg(x, 3) is Call("movavg", (x, x[-1], x[-2])).
_TRAILING: dict[str, int] = {"movavg": 1, "movsum": 1, "movsd": 1, "movvar": 1, "movmin": 1, "movmax": 1,
                             "movcov": 2, "movcorr": 2, "movbeta": 2}
# Centred windows (``cmovavg(x, 5)``: x[+2] ... x[-2]; an even n has one more lag than leads, as pandas'
# rolling(n, center=True)) are stored the same way, their leads explicit, so models see them as leads.
CENTRED: dict[str, str] = {f"c{fn}": fn for fn in _TRAILING}
MOVING: dict[str, int] = {**_TRAILING, **{c: _TRAILING[fn] for c, fn in CENTRED.items()}}
# Expanding windows (``cumavg(x)``: every period up to t, within each entity) cannot be written with fixed lags: like
# the group-wise functions they need series (data.eval, derived series, linear-form regressors), not model solvers.
EXPANDING: dict[str, str] = {"cum" + fn[3:]: fn for fn in _TRAILING}


def window_lead(fn: str, n: int) -> int:
    """The longest lead of an n-period window of moving function ``fn``: 0 for a trailing one, (n - 1) // 2 for a
    centred one; its arguments run from that lead down, one period at a time."""
    return (n - 1) // 2 if fn in CENTRED else 0

_PREC = {"or": -2, "and": -1, **dict.fromkeys(COMPARISONS, 0), "+": 1, "-": 1, "*": 2, "/": 2, "^": 4}


class Expr:
    """Base class for expression nodes."""

    __slots__ = ()

    def children(self) -> tuple[Expr, ...]:
        return ()

    def rebuild(self, children: tuple[Expr, ...]) -> Expr:
        return self

    def walk(self) -> Iterator[Expr]:
        """Every node of the tree, depth first: this one, then each child's subtree in order. With a stack rather than
        nested generators, which hand each node up through every level above it (quadratic in a long sum's depth)."""
        stack: list[Expr] = [self]
        while stack:
            node = stack.pop()
            yield node
            kids = node.children()
            if kids:
                stack.extend(reversed(kids))

    def map(self, fn: Callable[[Expr], Expr | None]) -> Expr:
        """Rebuild the tree; ``fn`` returns a replacement node or None to recurse."""
        out = fn(self)
        if out is not None:
            return out
        kids = self.children()
        if not kids:
            return self
        return self.rebuild(tuple(k.map(fn) for k in kids))

    def refs(self) -> frozenset[Name]:
        """Variables with their lags, e.g. ``{Name("c", 0), Name("c", -1)}``."""
        return frozenset(n for n in self.walk() if isinstance(n, Name))

    def variables(self) -> frozenset[str]:
        return frozenset(n.id for n in self.refs())

    def params(self) -> frozenset[str]:
        return frozenset(n.id for n in self.walk() if isinstance(n, Coef))

    def calls(self, fn: str) -> list[Call]:
        return [n for n in self.walk() if isinstance(n, Call) and n.fn == fn]

    def shift(self, k: int) -> Expr:
        """Lag (k < 0) or lead (k > 0) every variable in the expression."""
        if k == 0:
            return self
        out = self.map(lambda n: Name(n.id, n.lag + k) if isinstance(n, Name) else None)
        if isinstance(self, BinOp) and self.source and isinstance(out, BinOp):  # dlog(c) -> dlog(c)[-1]
            import re

            m = re.fullmatch(r"(.*)\[([+-]\d+)\]", self.source)
            base, lag = (m.group(1), int(m.group(2)) + k) if m else (self.source, k)
            return BinOp(out.op, out.left, out.right, out.grouped or self.grouped, base if lag == 0 else f"{base}[{lag:+d}]")
        return out

    def substitute(self, values: Mapping[str, float]) -> Expr:
        """Replace parameters by numbers."""
        return self.map(
            lambda n: Num(float(values[n.id])) if isinstance(n, Coef) and n.id in values else None
        )

    def __add__(self, other: object) -> Expr:
        return add(self, lift(other))

    def __radd__(self, other: object) -> Expr:
        return add(lift(other), self)

    def __sub__(self, other: object) -> Expr:
        return sub(self, lift(other))

    def __rsub__(self, other: object) -> Expr:
        return sub(lift(other), self)

    def __mul__(self, other: object) -> Expr:
        return mul(self, lift(other))

    def __rmul__(self, other: object) -> Expr:
        return mul(lift(other), self)

    def __truediv__(self, other: object) -> Expr:
        return div(self, lift(other))

    def __rtruediv__(self, other: object) -> Expr:
        return div(lift(other), self)

    def __pow__(self, other: object) -> Expr:
        return power(self, lift(other))

    def __neg__(self) -> Expr:
        return neg(self)


@dataclass(frozen=True, slots=True)
class Num(Expr):
    value: float

    def __str__(self) -> str:
        v = float(self.value)
        return str(int(v)) if v.is_integer() and abs(v) < 1e15 else repr(v)


@dataclass(frozen=True, slots=True)
class Name(Expr):
    """A variable, with its lag (negative) or lead (positive)."""

    id: str
    lag: int = 0

    def __str__(self) -> str:
        return self.id if self.lag == 0 else f"{self.id}[{self.lag:+d}]"


@dataclass(frozen=True, slots=True)
class Level(Expr):
    """The grouping in a group-wise function: an axis (``country``, ``time``) or a series of categories."""

    id: str

    def __str__(self) -> str:
        return self.id


@dataclass(frozen=True, slots=True)
class Coef(Expr):
    """A parameter (coefficient)."""

    id: str

    def __str__(self) -> str:
        return self.id


@dataclass(frozen=True, slots=True)
class Neg(Expr):
    operand: Expr

    def children(self) -> tuple[Expr, ...]:
        return (self.operand,)

    def rebuild(self, children: tuple[Expr, ...]) -> Expr:
        return Neg(children[0])

    def __str__(self) -> str:
        return f"-{_wrap(self.operand, 3)}"


@dataclass(frozen=True, slots=True)
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr
    # One term in a linear-form equation, not split at its + or -: written in parentheses,
    # or expanded from d(), dlog() or pch(), whose text ``source`` keeps for labels.
    grouped: bool = field(default=False, compare=False)
    source: str = field(default="", compare=False)

    def children(self) -> tuple[Expr, ...]:
        return (self.left, self.right)

    def rebuild(self, children: tuple[Expr, ...]) -> Expr:
        same = children == (self.left, self.right)
        return BinOp(self.op, children[0], children[1], self.grouped, self.source if same else "")

    def __str__(self) -> str:
        if self.source:
            return self.source
        p = _PREC[self.op]
        if self.op == "^":
            return f"{_wrap(self.left, p + 1)}^{_wrap(self.right, p)}"
        left = _wrap(self.left, p)
        right = _wrap(self.right, p + (self.op in "-/"))
        sep = f" {self.op} " if self.op in ("+", "-") or self.op in COMPARISONS or self.op in LOGICAL else self.op
        return f"{left}{sep}{right}"


@dataclass(frozen=True, slots=True)
class Call(Expr):
    fn: str
    args: tuple[Expr, ...]

    def children(self) -> tuple[Expr, ...]:
        return self.args

    def rebuild(self, children: tuple[Expr, ...]) -> Expr:
        return Call(self.fn, children)

    def __str__(self) -> str:
        if self.fn in MOVING:  # written with its window, not its lags: movavg(x, 4)
            k = MOVING[self.fn]
            n = len(self.args) // k
            at = window_lead(self.fn, n)  # the argument without a lead or lag
            return f"{self.fn}({', '.join(str(self.args[i * n + at]) for i in range(k))}, {n})"
        if self.fn in ELEMENTWISE and len(self.args) == 2 and isinstance(self.args[1], Level):
            return f"{self.fn}({self.args[0]}, by={self.args[1]})"  # max(y, country) would mean the larger of the two
        return f"{self.fn}({', '.join(str(a) for a in self.args)})"

    @property
    def elementwise(self) -> bool:
        """``max(a, b)`` or ``min(a, b)`` period by period (not a group-wise statistic)."""
        return self.fn in ELEMENTWISE and len(self.args) == 2 and not isinstance(self.args[1], Level)


def _prec(e: Expr) -> int:
    if isinstance(e, BinOp):
        return _PREC[e.op]
    if isinstance(e, Neg) or (isinstance(e, Num) and e.value < 0):
        return 3
    return 5


def _wrap(e: Expr, p: int) -> str:
    s = str(e)
    return f"({s})" if _prec(e) < p else s


# Constructors that fold constants; used when building expressions in code
# (derivatives, linear-form equations). The parser keeps the user's structure.


def lift(v: object) -> Expr:
    return v if isinstance(v, Expr) else Num(float(v))  # type: ignore[arg-type]


def _is(e: Expr, v: float) -> bool:
    return isinstance(e, Num) and e.value == v


def add(a: Expr, b: Expr) -> Expr:
    if _is(a, 0):
        return b
    if _is(b, 0):
        return a
    if isinstance(a, Num) and isinstance(b, Num):
        return Num(a.value + b.value)
    return BinOp("+", a, b)


def sub(a: Expr, b: Expr) -> Expr:
    if _is(b, 0):
        return a
    if _is(a, 0):
        return neg(b)
    if isinstance(a, Num) and isinstance(b, Num):
        return Num(a.value - b.value)
    return BinOp("-", a, b)


def mul(a: Expr, b: Expr) -> Expr:
    if _is(a, 0) or _is(b, 0):
        return Num(0.0)
    if _is(a, 1):
        return b
    if _is(b, 1):
        return a
    if isinstance(a, Num) and isinstance(b, Num):
        return Num(a.value * b.value)
    return BinOp("*", a, b)


def div(a: Expr, b: Expr) -> Expr:
    if _is(a, 0):
        return Num(0.0)
    if _is(b, 1):
        return a
    if isinstance(a, Num) and isinstance(b, Num) and b.value != 0:
        return Num(a.value / b.value)
    return BinOp("/", a, b)


def power(a: Expr, b: Expr) -> Expr:
    if _is(b, 0):
        return Num(1.0)
    if _is(b, 1):
        return a
    if isinstance(a, Num) and isinstance(b, Num):
        return Num(a.value**b.value)
    return BinOp("^", a, b)


def neg(a: Expr) -> Expr:
    if isinstance(a, Num):
        return Num(-a.value)
    if isinstance(a, Neg):
        return a.operand
    return Neg(a)
