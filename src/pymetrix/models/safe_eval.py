"""Evaluating the small expressions model files hold, checked before they run.

Dynare parameter values (``alpha = 0.33; beta = 1/(1+r/400);``), IRIS ``!if`` conditions and ``!parameters`` values,
the model editor's parameter values and simple MATLAB data files (``growth = diff(log(raw(:,1)))``) are evaluated as
Python after a light rewrite (``^`` to ``**``, ``&&`` to ``and``, ``x(:,k)`` to ``x[:, k-1]``). A model file can come
from anyone, so, as for matrix expressions (``pymetrix.matrix_expr``), the expression is parsed and every node checked
against a short list before anything runs:

- numbers (and ``True``/``False`` where conditions are read);
- names given by the caller (parameters, earlier values, the allowed functions); none starting with ``__``;
- ``+ - * / **`` and unary ``+ -``; comparisons; ``and``, ``or`` and ``not`` only where conditions are read;
- calls only to the caller's functions by name, with positional arguments;
- indexing only as ``name[:, k]`` and only where the caller allows it (MATLAB data files).

Attribute access (``().__class__``), other indexing, ``lambda``, comprehensions, keyword or ``*`` arguments, strings and
anything else are refused with a ``ModelError`` naming what was found. Whole numbers are evaluated as floats, so a
power such as ``9**9**9`` overflows at once rather than computing a huge integer; the values are the same.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from typing import Any

from .solvers import ModelError

__all__ = ["check_expression", "safe_eval"]

_ARITHMETIC = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_UNARY = (ast.UAdd, ast.USub)
_COMPARE = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
_WORDS = {ast.Attribute: "attribute access", ast.Subscript: "indexing", ast.Lambda: "lambda",
          ast.ListComp: "a comprehension", ast.SetComp: "a comprehension", ast.DictComp: "a comprehension",
          ast.GeneratorExp: "a comprehension", ast.NamedExpr: "an assignment (:=)", ast.IfExp: "if ... else",
          ast.Dict: "a dictionary", ast.Set: "a set", ast.List: "a list", ast.Tuple: "a tuple",
          ast.Starred: "* unpacking", ast.JoinedStr: "an f-string", ast.Await: "await", ast.Yield: "yield",
          ast.YieldFrom: "yield", ast.Compare: "a comparison", ast.BoolOp: "and / or", ast.Slice: "a slice"}


def _refuse(what: str) -> ModelError:
    return ModelError(f"{what} is not allowed in a model file's expression")


def _column(node: ast.Subscript) -> bool:
    """``name[:, k]``: a whole column of a named array, k a whole number."""
    s = node.slice
    return (isinstance(node.value, ast.Name) and isinstance(s, ast.Tuple) and len(s.elts) == 2
            and isinstance(s.elts[0], ast.Slice) and s.elts[0].lower is s.elts[0].upper is s.elts[0].step is None
            and isinstance(s.elts[1], ast.Constant) and type(s.elts[1].value) is int)


def check_expression(expr: str, names: Iterable[str], functions: Iterable[str] = (), *, conditions: bool = False,
                     columns: bool = False) -> ast.Expression:
    """Parse ``expr`` and check it against the allowed nodes; returns the tree with whole numbers made floats.
    ``names`` are the values it may read, ``functions`` the names it may call; ``conditions`` also allows ``and``,
    ``or``, ``not`` and ``True``/``False``; ``columns`` allows ``name[:, k]``. Raises ``ModelError``."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ModelError(f"not an expression ({exc.msg})") from None
    known, callable_ = set(names), set(functions)
    skip: set[int] = set()  # the k of name[:, k], an index left as it is
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise _refuse(f"the name {node.id}")
            if not isinstance(node.ctx, ast.Load):
                raise _refuse("assignment")
            if node.id not in known and node.id not in callable_:
                raise ModelError(f"name {node.id!r} is not defined")
        elif isinstance(node, ast.Constant):
            if type(node.value) not in (int, float) and not (conditions and type(node.value) is bool):
                raise _refuse(f"a {type(node.value).__name__} value ({node.value!r})")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in callable_:
                target = node.func.id if isinstance(node.func, ast.Name) else _WORDS.get(type(node.func), "an expression")
                raise _refuse(f"calling {target}")
            if node.keywords:
                raise _refuse("a keyword argument")
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ARITHMETIC):
                raise _refuse(f"the operator {type(node.op).__name__}")
        elif isinstance(node, ast.UnaryOp):
            if not (isinstance(node.op, _UNARY) or (conditions and isinstance(node.op, ast.Not))):
                raise _refuse(f"the operator {type(node.op).__name__}")
        elif isinstance(node, ast.Compare):
            if not all(isinstance(op, _COMPARE) for op in node.ops):
                raise _refuse("this comparison (in, is)")
        elif isinstance(node, ast.BoolOp):
            if not conditions:
                raise _refuse("and / or")
        elif isinstance(node, ast.Subscript):
            if not (columns and _column(node)):
                raise _refuse("indexing")
            skip.update(id(n) for n in ast.walk(node.slice))
        elif not isinstance(node, (ast.Expression, ast.Load, ast.And, ast.Or, *_ARITHMETIC, *_UNARY, *_COMPARE, ast.Not)):
            raise _refuse(_WORDS.get(type(node), type(node).__name__))

    class _Floats(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            return node  # the column index stays a whole number

        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            return ast.copy_location(ast.Constant(float(node.value)), node) if type(node.value) is int else node

    return ast.fix_missing_locations(_Floats().visit(tree))


def safe_eval(expr: str, values: Mapping[str, Any], functions: Mapping[str, Any] | None = None, *,
              conditions: bool = False, columns: bool = False) -> Any:
    """``expr`` evaluated with ``values`` and ``functions`` only, after ``check_expression`` has accepted it."""
    functions = dict(functions or {})
    tree = check_expression(expr, values, functions, conditions=conditions, columns=columns)
    scope = {**functions, **values}
    return eval(compile(tree, "<model file expression>", "eval"), {"__builtins__": {}}, scope)  # noqa: S307 - checked above
