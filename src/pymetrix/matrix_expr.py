"""Matrix expressions: the code a matrix step records, checked before it runs.

The app's matrix dialogs and the actions of a matrix window record one step each, ``name = <expression>``, in the
package's own syntax::

    b = (X.T @ X).inv() @ X.T @ y
    V = Matrix.cov(fit)
    R = V.chol(upper=True)
    K = A.kron(Matrix.identity(3))

The expression may use ``Matrix``, ``np`` (NumPy), a few harmless built-ins (``abs``, ``min``, ``max``, ``round``,
``len``, ``float``, ``int``, ``range``, ``sum``) and the session's objects by name. Before it runs it is parsed and
checked: it must be one Python expression made of names, attribute access, calls, operators, indexing and literals.
Imports, ``lambda``, comprehensions, assignments, attributes starting with an underscore (``__class__``), built-ins
that run or reach code (``exec``, ``eval``, ``open``, ``getattr``, ...) and methods that read or write files
(``to_csv``, ``np.save``, ...) are refused with a message saying which. So a project holding matrix steps rebuilds
them on opening without the trust that typed code needs: the expression can compute, and nothing else.
"""

from __future__ import annotations

import ast
import dataclasses
import difflib
import threading
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from ._api import api
from .errors import SpecificationError
from .matrix import LU, QR, QZ, SVD, Eig, Matrix, Schur

__all__ = ["ALLOWED_BUILTINS", "ExpressionError", "ExpressionPreview", "MATRIX_EXPRESSION", "check_expression", "evaluate",
           "expression_names", "preview"]

MATRIX_EXPRESSION = "pymetrix:matrix.expression"

ALLOWED_BUILTINS: dict[str, Any] = {"abs": abs, "min": min, "max": max, "round": round, "len": len, "float": float,
                                    "int": int, "range": range, "sum": sum}
GIVEN = ("Matrix", "np")  # always there, as in a script (from pymetrix import *; import numpy as np)
_RESERVED = set(GIVEN) | set(ALLOWED_BUILTINS)

_FORBIDDEN_NAMES = {"exec", "eval", "compile", "open", "__import__", "getattr", "setattr", "delattr", "hasattr", "globals",
                    "locals", "vars", "dir", "input", "breakpoint", "exit", "quit", "help", "type", "object", "super",
                    "memoryview", "classmethod", "staticmethod", "property", "print", "__builtins__"}
_FILES = {"save", "savez", "savez_compressed", "savetxt", "tofile", "dump", "dumps", "load", "loadtxt", "genfromtxt",
          "fromfile", "fromregex", "memmap", "ctypeslib", "lib", "DataSource", "to_csv", "to_excel", "to_parquet",
          "to_pickle", "to_json", "to_hdf", "to_sql", "to_stata", "to_feather", "write", "read", "read_csv", "savefig",
          "save_figure", "save_result", "export"}
_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.MatMult, ast.Pow, ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd)
_ALLOWED = (ast.Expression, ast.Name, ast.Load, ast.Attribute, ast.Call, ast.keyword, ast.BinOp, ast.UnaryOp,
            ast.Constant, ast.List, ast.Tuple, ast.Subscript, ast.Slice, *_OPERATORS)
_WORDS = {ast.Lambda: "lambda", ast.ListComp: "a comprehension", ast.SetComp: "a comprehension",
          ast.DictComp: "a comprehension", ast.GeneratorExp: "a comprehension", ast.NamedExpr: "an assignment (:=)",
          ast.IfExp: "if ... else", ast.Compare: "a comparison", ast.BoolOp: "and / or", ast.Dict: "a dictionary",
          ast.Set: "a set", ast.Starred: "* unpacking", ast.JoinedStr: "an f-string", ast.Await: "await",
          ast.Yield: "yield", ast.YieldFrom: "yield"}
DECOMPOSITIONS = (SVD, QR, LU, Eig, Schur, QZ)


class ExpressionError(SpecificationError):
    """A matrix expression that cannot be run: not an expression, a name that is not there, or a construct that is
    not allowed."""


def _parse(text: str) -> ast.Expression:
    text = text.strip()
    if not text:
        raise ExpressionError("write a matrix expression, such as A.T @ A or Matrix.cov(fit)")
    try:
        return ast.parse(text, mode="eval")
    except SyntaxError as exc:
        try:
            body = ast.parse(text).body
        except SyntaxError:
            body = []
        if any(isinstance(s, (ast.Import, ast.ImportFrom)) for s in body):
            raise ExpressionError("imports are not allowed in a matrix expression: Matrix and np are there already") from None
        if any(isinstance(s, (ast.Assign, ast.AugAssign, ast.AnnAssign)) for s in body):
            raise ExpressionError("write the expression only, without 'name =': the name goes in the Name box") from None
        if len(body) > 1:
            raise ExpressionError("one expression only, not several statements") from None
        if body:
            raise ExpressionError(f"{type(body[0]).__name__.lower()} statements are not allowed in a matrix expression") from None
        where = f" at column {exc.offset}" if exc.offset else ""
        raise ExpressionError(f"not a Python expression ({exc.msg}{where})") from None


def expression_names(text: str) -> set[str]:
    """The names an expression reads (objects, Matrix, np, built-ins); empty if it does not parse."""
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError:
        return set()
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def check_expression(text: str, objects: Iterable[str]) -> set[str]:
    """Check an expression before it runs; returns the session objects it uses. ``objects`` are the names it may
    use besides ``Matrix``, ``np`` and the allowed built-ins. Raises ``ExpressionError`` saying what is wrong."""
    tree = _parse(text)
    known = set(objects)
    used: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            word = next((w for cls, w in _WORDS.items() if isinstance(node, cls)), type(node).__name__)
            raise ExpressionError(f"{word} is not allowed in a matrix expression")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ExpressionError(f"attributes starting with an underscore are not allowed ({node.attr})")
            if node.attr in _FILES:
                raise ExpressionError(f"{node.attr} reads or writes files: not allowed in a matrix expression")
        elif isinstance(node, ast.keyword) and node.arg is None:
            raise ExpressionError("** unpacking is not allowed in a matrix expression")
        elif isinstance(node, ast.Name):
            name = node.id
            if name in _FORBIDDEN_NAMES or name.startswith("__"):
                raise ExpressionError(f"{name} is not allowed in a matrix expression")
            if name in known:
                used.add(name)
            elif name not in _RESERVED:
                near = difflib.get_close_matches(name, sorted(known | set(GIVEN)), n=1)
                hint = f" (did you mean {near[0]}?)" if near else ""
                raise ExpressionError(f"no object named {name!r}{hint}")
        elif isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, complex, str, bool, type(None))):
            raise ExpressionError(f"{type(node.value).__name__} values are not allowed in a matrix expression")
    return used


def _run(text: str, objects: Mapping[str, Any]) -> Any:
    tree = _parse(text)
    used = check_expression(text, objects)
    scope: dict[str, Any] = {"__builtins__": ALLOWED_BUILTINS, "Matrix": Matrix, "np": np}
    scope.update({n: objects[n] for n in used})
    return eval(compile(tree, "<matrix expression>", "eval"), scope)  # noqa: S307 - checked above: names, calls, operators


def evaluate(text: str, objects: Mapping[str, Any] | None = None, *, pair: bool = False) -> Any:
    """The value of a checked matrix expression: a ``Matrix``, or a decomposition (``A.svd()``, ``A.eig()``, ...).
    ``pair``: a pair of matrices instead (``Matrix.design(...)``: X and y)."""
    value = _run(text, objects or {})
    if pair:
        if isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, Matrix) for v in value):
            return value
        raise ExpressionError("the expression should give two matrices")
    if isinstance(value, (Matrix, *DECOMPOSITIONS)):
        return value
    if isinstance(value, np.ndarray):
        raise ExpressionError("the expression gives a NumPy array, not a matrix: wrap it as Matrix(...) to name it")
    what = "a number" if isinstance(value, (int, float, complex, np.number)) else f"a {type(value).__name__}"
    raise ExpressionError(f"the expression gives {what}, not a matrix (a step keeps a matrix or a decomposition)")


PREVIEW_INPUTS = 40_000  # the most numbers the session objects an expression uses may hold for it to be previewed
PREVIEW_CONSTANT = 1_000  # the largest number written in it
PREVIEW_SECONDS = 0.5  # and the longest it may take


@dataclasses.dataclass(frozen=True)
class ExpressionPreview:
    """What ``preview`` found: the problem (empty when none) and, when the expression was evaluated, what it gives
    (``shape`` and the row and column names of a matrix; ``what`` in words), or why it was not evaluated (``skipped``)."""

    problem: str = ""
    what: str = ""
    shape: tuple[int, int] | None = None
    rows: tuple[str, ...] = ()
    cols: tuple[str, ...] = ()
    skipped: str = ""

    @property
    def ok(self) -> bool:
        return not self.problem


def _size(obj: Any) -> int:
    if isinstance(obj, Matrix):
        return int(obj.values.size)
    frame = getattr(obj, "frame", None)
    if frame is not None and hasattr(frame, "shape"):
        return int(np.prod(frame.shape))
    if isinstance(obj, DECOMPOSITIONS):
        return sum(_size(part) for part in obj)
    return int(np.size(obj)) if isinstance(obj, np.ndarray) else 0


def _numbers_only(node: ast.AST) -> bool:
    return not any(isinstance(n, ast.Name) for n in ast.walk(node))


def _too_costly(tree: ast.Expression, used: Iterable[str], objects: Mapping[str, Any]) -> str:
    """Why evaluating a checked expression might be slow or large (empty when it is cheap): large inputs, large
    numbers written in it, or arithmetic on numbers alone (``2**40``, ``1000*1000``)."""
    inputs = sum(_size(objects[n]) for n in used)
    if inputs > PREVIEW_INPUTS:
        return f"the objects it uses hold {inputs:,} numbers"
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, complex)) and not isinstance(node.value, bool):
            if abs(node.value) > PREVIEW_CONSTANT:
                return f"it writes a number above {PREVIEW_CONSTANT:,}"
        elif isinstance(node, ast.BinOp) and _numbers_only(node):
            return "it computes with numbers alone"
        elif isinstance(node, (ast.Tuple, ast.List)):
            sizes = [abs(e.value) for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, (int, float))]
            if len(sizes) > 1 and np.prod(sizes) > 1e6:
                return "it may make a large array"
        elif isinstance(node, ast.Attribute) and node.attr in ("kron", "tile", "repeat", "power") and inputs > 2_500:
            return f"{node.attr} of matrices this large"
    return ""


def _describe(value: Any) -> dict[str, Any]:
    if isinstance(value, Matrix):
        r, c = value.shape
        return {"what": f"a {r} x {c} matrix", "shape": (r, c), "rows": tuple(value.rows), "cols": tuple(value.cols)}
    if isinstance(value, DECOMPOSITIONS):
        parts = list(getattr(value, "_fields", ()))
        return {"what": f"a decomposition: {type(value).__name__}" + (f" ({', '.join(parts)})" if parts else "")}
    raise ExpressionError("not a matrix")  # evaluate has refused it already


def preview(text: str, objects: Mapping[str, Any] | None = None, *, evaluate_it: bool = True) -> ExpressionPreview:
    """Check an expression as it is typed (the matrix dialog, a moment after typing stops): the problem
    ``check_expression`` finds (an unknown name with a suggestion, a construct not allowed, a parse error with its
    column), else, when it is cheap (see ``_too_costly``: small inputs, small numbers, at most ``PREVIEW_SECONDS``),
    what it gives: its shape and names, or the problem evaluating it raised. Nothing is changed."""
    objects = dict(objects or {})
    try:
        tree = _parse(text)
        used = check_expression(text, objects)
    except ExpressionError as exc:
        return ExpressionPreview(problem=str(exc))
    if not evaluate_it:
        return ExpressionPreview(skipped="not evaluated")
    reason = _too_costly(tree, used, objects)
    if reason:
        return ExpressionPreview(skipped=f"not evaluated while typing: {reason}")
    out: dict[str, Any] = {}

    def work() -> None:
        try:
            out["value"] = evaluate(text, {n: objects[n] for n in used})
        except Exception as exc:  # noqa: BLE001 - shown as the expression's problem, as the step would report it
            out["error"] = exc

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    thread.join(PREVIEW_SECONDS)
    if thread.is_alive():
        return ExpressionPreview(skipped=f"not evaluated while typing: it takes over {PREVIEW_SECONDS:g} s")
    if "error" in out:
        exc = out["error"]
        from .errors import PyMetrixError

        return ExpressionPreview(problem=str(exc) if isinstance(exc, PyMetrixError) else f"{type(exc).__name__}: {exc}")
    return ExpressionPreview(**_describe(out["value"]))


@api(MATRIX_EXPRESSION)
def _matrix_expression(expression: str, inputs: Mapping[str, Any] | None = None) -> Any:
    """A matrix computed from an expression (``b = (X.T @ X).inv() @ X.T @ y``): what the matrix dialogs and a matrix
    window's actions record. ``inputs`` are the session objects it uses, by name; its code is the expression itself."""
    return evaluate(expression, dict(inputs or {}))
