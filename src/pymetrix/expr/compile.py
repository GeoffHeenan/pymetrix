"""Expressions compiled to NumPy functions, for model solvers that evaluate the same equations many times.

    f = compile_exprs([residual, derivative, ...], coefs=("beta", "kappa"))
    f.refs                                  # ((name, lag), ...): the values the function needs, in order
    values = f([x_now, x_lag, ...], [0.99, 0.1])  # one result per expression: floats, or arrays over periods

The tree interpreter (``evaluate``) walks every node on every call; a compiled function is Python source generated
once from the tree, in which every variable is a slot ``v[k]`` and every parameter ``p[j]``, so the same function
works for scalars (a steady state) and for whole arrays of periods (a stacked simulation), and for any parameter
values. Subexpressions that occur more than once (equal trees: symbolic derivatives repeat the same factors many
times over, and a model's parameter expressions recur in every equation) are worked out once, into temporaries: the
second derivatives of Smets and Wouters (2007) shrink from 70,000 nodes to 810. The source is built only from the
parsed tree: names never appear in it, only slot numbers, numbers and NumPy functions, so it cannot run anything but
arithmetic. Group-wise functions and linear-form markers are not compilable (they need pandas or the estimator);
``compilable`` says so, and callers keep the interpreter for them. Results agree with ``evaluate``: the same NumPy
operations, with the same handling of division by zero and missing values. Called with plain numbers (every slot and
parameter a float: a steady state, a linearisation), the same source runs on Python's float arithmetic and ``math``
instead, which costs a small fraction of NumPy's per-call overhead on scalars; each operation hands over to NumPy where
Python would raise or differ (division by zero, logs of zero or negatives, overflow, NaN in a maximum), so the results
are NumPy's (powers can differ in the last digit, and the sign of a zero).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any

import numpy as np

from . import functions, rows
from .nodes import (
    DATA_ONLY,
    EXPANDING,
    GROUPWISE,
    MARKERS,
    NA,
    BinOp,
    Call,
    Coef,
    Expr,
    ExpressionError,
    Level,
    Name,
    Neg,
    Num,
)

__all__ = ["Compiled", "compilable", "compile_exprs"]

_FUNCTIONS = {"log": "_np.log", "exp": "_np.exp", "sqrt": "_np.sqrt", "abs": "_np.abs", "sign": "_np.sign", "log10": "_np.log10",
              "floor": "_np.floor", "ceil": "_np.ceil"}
_COMPARE = {"<": "less", "<=": "less_equal", ">": "greater", ">=": "greater_equal", "==": "equal", "!=": "not_equal"}


def compilable(expr: Expr) -> bool:
    """Whether ``expr`` can be compiled (no group-wise functions, expanding windows, functions of the data's rows or
    linear-form markers)."""
    def ok(node: Expr) -> bool:
        if isinstance(node, Level):
            return False
        if isinstance(node, Call) and not node.elementwise and (node.fn in GROUPWISE or node.fn in MARKERS or node.fn in EXPANDING
                                                             or node.fn in DATA_ONLY):
            return False
        return all(ok(c) for c in node.children())

    return ok(expr)


@dataclass(frozen=True, eq=False)
class Compiled:
    """A list of expressions as one NumPy function of variable slots and parameter values."""

    exprs: tuple[Expr, ...]
    refs: tuple[tuple[str, int], ...]  # (name, lag) of each variable slot
    coefs: tuple[str, ...]  # parameter names, in the order of the values passed
    source: str
    fn: Callable[[Sequence[Any], Sequence[float]], list[Any]]

    def __call__(self, values: Sequence[Any], params: Sequence[float] = ()) -> list[Any]:
        if _numbers(values) and _numbers(params):
            return self._scalar([float(v) for v in values], [float(v) for v in params])
        return self.fn(values, params)

    @cached_property
    def _scalar(self) -> Callable[[Sequence[float], Sequence[float]], list[Any]]:
        """The same source on Python floats (``_Scalar`` in place of NumPy)."""
        namespace: dict[str, Any] = {"_np": _SCALAR, "_F": functions, "_R": rows}
        exec(compile(self.source, "<pymetrix compiled expressions>", "exec"), namespace)  # noqa: S102 - generated from the tree only
        return namespace["_f"]

    def slot(self, name: str, lag: int = 0) -> int:
        return self.refs.index((name, lag))


_FLOATS = (float, int, np.floating, np.integer)


def _numbers(values: Any) -> bool:
    """Whether every value is a plain number (not an array over periods)."""
    if isinstance(values, np.ndarray):
        return values.ndim == 1 and values.dtype.kind in "fi"
    return all(isinstance(v, _FLOATS) and not isinstance(v, bool) for v in values)


def _np_scalar(name: str, *args: float) -> Any:
    with np.errstate(all="ignore"):
        return getattr(np, name)(*args)


class _Scalar:
    """NumPy's functions on Python floats, for ``Compiled``'s scalar path: Python's arithmetic and ``math`` where they
    give NumPy's result (to the last digit), NumPy itself where they would raise or differ. Anything else is NumPy's."""

    errstate = staticmethod(lambda **_: nullcontext())

    @staticmethod
    def divide(a: float, b: float) -> Any:
        try:
            return a / b
        except ZeroDivisionError:
            return _np_scalar("divide", a, b)

    @staticmethod
    def power(a: float, b: float) -> Any:
        if not math.isfinite(b) or (a < 0 and b != int(b)):  # Python gives a complex number for the second
            return _np_scalar("power", a, b)
        try:
            return a ** b
        except (ZeroDivisionError, OverflowError):
            return _np_scalar("power", a, b)

    @staticmethod
    def log(a: float) -> Any:
        return math.log(a) if a > 0 and a != math.inf else _np_scalar("log", a)

    @staticmethod
    def exp(a: float) -> Any:
        try:
            return math.exp(a)
        except OverflowError:
            return _np_scalar("exp", a)

    @staticmethod
    def sqrt(a: float) -> Any:
        return math.sqrt(a) if a >= 0 and a != math.inf else _np_scalar("sqrt", a)

    @staticmethod
    def abs(a: float) -> Any:
        return abs(a)

    @staticmethod
    def maximum(a: float, b: float) -> Any:
        return max(a, b) if not (math.isnan(a) or math.isnan(b)) else _np_scalar("maximum", a, b)

    @staticmethod
    def minimum(a: float, b: float) -> Any:
        return min(a, b) if not (math.isnan(a) or math.isnan(b)) else _np_scalar("minimum", a, b)

    def __getattr__(self, name: str) -> Any:
        return getattr(np, name)


_SCALAR = _Scalar()


def compile_exprs(exprs: Sequence[Expr], coefs: Sequence[str] = ()) -> Compiled:
    """Compile expressions (cached: the same expressions and parameter names give the same function)."""
    return _compile(tuple(exprs), tuple(coefs))


@lru_cache(maxsize=256)
def _compile(exprs: tuple[Expr, ...], coefs: tuple[str, ...]) -> Compiled:
    refs: dict[tuple[str, int], int] = {}
    for e in exprs:
        for n in sorted(e.refs(), key=lambda r: (r.id, r.lag)):
            refs.setdefault((n.id, n.lag), len(refs))
    params = {c: j for j, c in enumerate(coefs)}
    shared: dict[int, str] = {}  # id of a node -> the temporary holding its value
    body = []
    for objects, node in _common(exprs):
        code = _code(node, refs, params, shared)
        name = f"_t{len(body)}"
        body.append(f"        {name} = {code}")
        shared.update(dict.fromkeys(objects, name))
    lines = ["def _f(v, p):", "    with _np.errstate(all='ignore'):", *body, "        return ["]
    lines += [f"            {_code(e, refs, params, shared)}," for e in exprs]
    lines += ["        ]"]
    source = "\n".join(lines)
    namespace: dict[str, Any] = {"_np": np, "_F": functions, "_R": rows}
    exec(compile(source, "<pymetrix compiled expressions>", "exec"), namespace)  # noqa: S102 - generated from the tree only
    return Compiled(exprs, tuple(refs), coefs, source, namespace["_f"])


def _common(exprs: Sequence[Expr]) -> list[tuple[list[int], Expr]]:
    """The subexpressions that occur more than once (equal trees, wherever they are: derivatives repeat the same
    factors many times over), children before parents: each with the ids of its node objects. The compiled function
    works each out once, into a temporary, as a common-subexpression pass would."""
    canon: dict[int, int] = {}  # id of a node object -> its class of equal trees
    table: dict[tuple, int] = {}
    first: list[Expr] = []
    objects: list[list[int]] = []
    uses: list[int] = []

    def visit(node: Expr) -> int:
        found = canon.get(id(node))
        if found is not None:
            return found
        kids = tuple(visit(c) for c in node.children())
        if isinstance(node, Num):
            own: Any = repr(float(node.value))
        elif isinstance(node, Name):
            own = (node.id, node.lag)
        elif isinstance(node, (Coef, Call)):
            own = node.id if isinstance(node, Coef) else node.fn
        elif isinstance(node, BinOp):
            own = node.op
        elif isinstance(node, Neg):
            own = None
        else:  # anything else is never shared
            own = id(node)
        key = (type(node).__name__, own, kids)
        cls = table.get(key)
        if cls is None:
            cls = table[key] = len(first)
            first.append(node)
            objects.append([])
            uses.append(0)
            for k in kids:
                uses[k] += 1
        canon[id(node)] = cls
        objects[cls].append(id(node))
        return cls

    for e in exprs:
        uses[visit(e)] += 1
    return [(objects[c], first[c]) for c in range(len(first)) if uses[c] > 1 and first[c].children()]


def _code(node: Expr, refs: dict[tuple[str, int], int], params: dict[str, int], shared: dict[int, str]) -> str:
    if id(node) in shared:
        return shared[id(node)]
    if isinstance(node, Num):
        return repr(float(node.value))
    if isinstance(node, Name):
        return f"v[{refs[(node.id, node.lag)]}]"
    if isinstance(node, Coef):
        if node.id not in params:
            raise ExpressionError(f"no value given for parameter {node.id!r}")
        return f"p[{params[node.id]}]"
    if isinstance(node, Neg):
        return f"(-{_code(node.operand, refs, params, shared)})"
    if isinstance(node, BinOp):
        a, b = _code(node.left, refs, params, shared), _code(node.right, refs, params, shared)
        if node.op in ("+", "-", "*"):
            return f"({a} {node.op} {b})"
        if node.op == "/":
            return f"_np.divide({a}, {b})"
        if node.op in _COMPARE or node.op in ("and", "or"):
            test = (f"_np.{_COMPARE[node.op]}({a}, {b})" if node.op in _COMPARE else
                    f"_np.logical_{node.op}(_np.not_equal({a}, 0.0), _np.not_equal({b}, 0.0))")
            return f"_np.where(_np.isnan({a}) | _np.isnan({b}), _np.nan, {test} * 1.0)"
        return f"_np.power({a}, {b})"
    if isinstance(node, Call):
        if node.elementwise:
            fn = "_np.maximum" if node.fn == "max" else "_np.minimum"
            return f"{fn}({_code(node.args[0], refs, params, shared)}, {_code(node.args[1], refs, params, shared)})"
        if node.fn == NA:
            return "_np.nan"
        if node.fn == "isna":
            return f"(_np.isnan({_code(node.args[0], refs, params, shared)}) * 1.0)"
        if node.fn == "round":
            args = ", ".join(_code(a, refs, params, shared) for a in node.args)
            return f"_R.round_half_away({args})"
        if node.fn == "fillna":
            x, v = (_code(a, refs, params, shared) for a in node.args)
            return f"_np.where(_np.isnan({x}), {v}, {x})"
        if node.fn == "iif":
            c, a, b = (_code(x, refs, params, shared) for x in node.args)
            return f"_np.where(_np.isnan({c}), _np.nan, _np.where(_np.not_equal({c}, 0.0), {a}, {b}))"
        if node.fn == "step":
            return f"_np.heaviside({_code(node.args[0], refs, params, shared)}, 1.0)"
        if node.fn == "not":
            x = _code(node.args[0], refs, params, shared)
            return f"_np.where(_np.isnan({x}), _np.nan, _np.equal({x}, 0.0) * 1.0)"
        if node.fn in functions.MOVING or node.fn in functions.FUNCTIONS:  # distributions and windows: a fixed name
            args = ", ".join(_code(a, refs, params, shared) for a in node.args)
            kind = "window" if node.fn in functions.MOVING else "numeric"
            return f"_F.{kind}({node.fn!r}, ({args},))"
        if node.fn in _FUNCTIONS:
            return f"{_FUNCTIONS[node.fn]}({_code(node.args[0], refs, params, shared)})"
    raise ExpressionError(f"{node} cannot be compiled (group-wise functions and linear-form markers need the interpreter)")
