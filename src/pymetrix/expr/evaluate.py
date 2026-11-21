"""Evaluation of expression trees.

``evaluate`` works on whatever the resolver returns: pandas Series when
evaluating over a ``Dataset``, NumPy arrays during estimation, or floats
inside a model solver.

Model solvers use compiled versions of the same trees (``pymetrix.expr.compile``,
``pymetrix.models.fast``); this interpreter stays the reference they are tested
against, and handles what cannot be compiled (group-wise functions, pandas).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from . import functions
from .nodes import EXPANDING, GROUPWISE, MARKERS, MOVING, BinOp, Call, Coef, Expr, ExpressionError, Level, Name, Neg, Num

Resolver = Callable[[str, int], Any]

_FUNCS: dict[str, Callable[[Any], Any]] = {
    "log": np.log,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sign": np.sign,
    "step": lambda x: np.heaviside(x, 1.0),  # 1 where x >= 0, else 0; missing stays missing
    "not": lambda x: _truth(np.equal(x, 0), x),
}

_COMPARE = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal, "==": np.equal, "!=": np.not_equal}


def _truth(result: Any, *operands: Any) -> Any:
    """``result`` (true or false) as 1.0 or 0.0, missing where any operand is missing; a Series stays a Series."""
    missing = np.zeros(np.shape(result), dtype=bool)
    for x in operands:
        missing = missing | np.isnan(np.asarray(x, dtype=float))
    values = np.where(missing, np.nan, np.asarray(result, dtype=float))
    series = next((x for x in operands if isinstance(x, pd.Series)), None)
    if series is not None:
        return pd.Series(values, index=series.index)
    return float(values) if np.ndim(values) == 0 else values


def _logic(op: str, a: Any, b: Any) -> Any:
    if op in _COMPARE:
        return _truth(_COMPARE[op](a, b), a, b)
    left, right = np.not_equal(a, 0), np.not_equal(b, 0)
    return _truth(np.logical_and(left, right) if op == "and" else np.logical_or(left, right), a, b)


def evaluate(node: Expr, resolve: Resolver, params: Mapping[str, float] | None = None) -> Any:
    """Evaluate ``node``; ``resolve(name, lag)`` supplies variable values."""
    with np.errstate(all="ignore"):
        return _eval(node, resolve, params or {})


def _eval(node: Expr, resolve: Resolver, params: Mapping[str, float]) -> Any:
    if isinstance(node, Num):
        return node.value
    if isinstance(node, Name):
        return resolve(node.id, node.lag)
    if isinstance(node, Coef):
        try:
            return params[node.id]
        except KeyError:
            raise ExpressionError(f"no value given for parameter {node.id!r}") from None
    if isinstance(node, Neg):
        return -_eval(node.operand, resolve, params)
    if isinstance(node, BinOp):
        a = _eval(node.left, resolve, params)
        b = _eval(node.right, resolve, params)
        if node.op == "+":
            return a + b
        if node.op == "-":
            return a - b
        if node.op == "*":
            return a * b
        if node.op == "/":
            return np.divide(a, b)
        if node.op in _COMPARE or node.op in ("and", "or"):
            return _logic(node.op, a, b)
        return np.power(a, b)
    if isinstance(node, Call):
        if node.elementwise:
            a, b = _eval(node.args[0], resolve, params), _eval(node.args[1], resolve, params)
            return np.maximum(a, b) if node.fn == "max" else np.minimum(a, b)
        if node.fn in MARKERS:
            raise ExpressionError(
                f"{node.fn}() can only appear as a term of a linear-form equation"
            )
        if node.fn in GROUPWISE:
            return _groupwise(node, resolve, params)
        if node.fn in MOVING:
            return functions.window(node.fn, [_eval(a, resolve, params) for a in node.args])
        if node.fn in EXPANDING:  # within each entity, along the dataset's time axis
            axis = getattr(getattr(resolve, "__self__", None), "time_axis", None)
            return functions.expanding(node.fn, [_eval(a, resolve, params) for a in node.args], getattr(axis, "name", None))
        if node.fn in functions.FUNCTIONS:
            return functions.numeric(node.fn, [_eval(a, resolve, params) for a in node.args])
        return _FUNCS[node.fn](_eval(node.args[0], resolve, params))
    raise ExpressionError(f"cannot evaluate {node!r}")


def _groupwise(node: Call, resolve: Resolver, params: Mapping[str, float]) -> pd.Series:
    x = _eval(node.args[0], resolve, params)
    if not isinstance(x, pd.Series):
        raise ExpressionError(f"{node}: group-wise functions work on series (data.eval, derived series, "
                              "linear-form regressors); make it a derived series first")
    keys: Any = np.zeros(len(x))
    if len(node.args) == 2:
        level = node.args[1]
        assert isinstance(level, Level)
        if level.id in (x.index.names or []):
            keys = np.asarray(x.index.get_level_values(level.id))
        else:
            try:
                keys = np.asarray(pd.Series(resolve(level.id, 0)).reindex(x.index))
            except (KeyError, ExpressionError):
                raise ExpressionError(f"{node}: {level.id!r} is neither an axis nor a series") from None
    x = x.astype(float)
    g = x.groupby(keys)
    fn = node.fn
    if fn == "rank":
        return g.rank(method="average")
    if fn == "count":
        return g.transform("count").astype(float)
    stat = {"mean": "mean", "demean": "mean", "sum": "sum", "share": "sum", "min": "min", "max": "max",
            "sd": "std", "median": "median"}[fn]
    value = g.transform(stat)
    if stat == "sum":  # a group with no observations has no total, not zero
        value = value.where(g.transform("count") > 0)
    if fn == "demean":
        return x - value
    if fn == "share":
        return x / value
    return value
