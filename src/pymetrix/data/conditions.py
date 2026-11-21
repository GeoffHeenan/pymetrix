"""Row conditions (``where=``), checked before they run.

A sample's or a step's row condition (``Sample(where="pop > 1e6")``, ``data.select(where=...)``, the "Rows where"
field of a dialog) is kept as text, saved with projects and datasets, and evaluated by pandas' ``DataFrame.eval`` on
the data's rows. ``DataFrame.eval`` can read the caller's Python variables (``@name``) and call methods on them or on
the columns (``@self.write('...')``, ``x.__class__``), so a condition in a project from someone else could run code
when the project opens, even with its typed code not trusted. As for matrix expressions (``pymetrix.matrix_expr``) and
model files (``pymetrix.models.safe_eval``), the condition is parsed and every node checked against a short list
before anything runs:

- column names (and the axes' names: ``year``, ``country``, ``time``), also in backticks for names with spaces; none
  starting with ``__``;
- numbers, ``True``/``False`` and text in quotes, which is also how dates and periods are written
  (``time >= '2000Q1'``, ``date < '2008-09-15'``);
- comparisons ``< <= > >= == !=``, ``in`` and ``not in`` with a list (``country in ['DEU', 'FRA']``);
- ``and``, ``or``, ``not`` and pandas' ``&``, ``|``, ``~``;
- arithmetic ``+ - * / // % **`` on numbers and columns (a power's exponent a number, at most 100 in size);
- pandas' functions by name, with plain arguments: ``abs``, ``sqrt``, ``exp``, ``log``, ``log10``, ``log1p``,
  ``expm1``, ``floor``, ``ceil``, the trigonometric and hyperbolic functions and ``arctan2``.

Anything else (attribute access, indexing, ``@`` references to Python variables, calls to other functions, keyword
arguments, lambdas, comprehensions, arithmetic on text or lists) is refused with a ``SpecificationError`` naming what
was found (a ``ConditionError``, a kind of ``SpecificationError``). A condition that passes is evaluated exactly as before, by ``DataFrame.eval``, with no Python variables in
reach, so it selects the same rows.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from typing import Any

import numpy as np

from ..errors import SpecificationError

__all__ = ["FUNCTIONS", "ConditionError", "check_condition", "condition_mask"]

# pandas.core.computation.ops.MATHOPS: the functions DataFrame.eval knows by name
_UNARY_FUNCTIONS = {"sin", "cos", "tan", "exp", "log", "expm1", "log1p", "sqrt", "sinh", "cosh", "tanh", "arcsin",
                    "arccos", "arctan", "arccosh", "arcsinh", "arctanh", "abs", "log10", "floor", "ceil"}
FUNCTIONS = frozenset(_UNARY_FUNCTIONS | {"arctan2"})
MAX_POWER = 100
MAX_LENGTH = 10_000

_ARITHMETIC = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.BitAnd, ast.BitOr)
_COMPARE = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn)
_UNARY = (ast.Not, ast.Invert, ast.USub, ast.UAdd)
_BACKTICKS = re.compile(r"`([^`]*)`")
_WORDS = {ast.Attribute: "attribute access (.)", ast.Subscript: "indexing ([...])", ast.Lambda: "lambda",
          ast.ListComp: "a comprehension", ast.SetComp: "a comprehension", ast.DictComp: "a comprehension",
          ast.GeneratorExp: "a comprehension", ast.NamedExpr: "an assignment (:=)", ast.IfExp: "if ... else",
          ast.Dict: "a dictionary", ast.Set: "a set", ast.Starred: "* unpacking", ast.JoinedStr: "an f-string",
          ast.Await: "await", ast.Yield: "yield", ast.YieldFrom: "yield", ast.Slice: "a slice",
          ast.MatMult: "@", ast.LShift: "<<", ast.RShift: ">>", ast.BitXor: "^", ast.Is: "`is`", ast.IsNot: "`is not`"}
_HELP = ("a row condition uses column names, numbers, text and dates in quotes, comparisons, in [...], and/or/not, "
         "arithmetic and the functions " + ", ".join(sorted(FUNCTIONS)))


class ConditionError(SpecificationError):
    """A row condition that cannot be used: refused by the check, or not readable. A project step that meets one is kept
    in the history, not run, and the project still opens."""


def _refuse(text: str, what: str) -> ConditionError:
    return ConditionError(f"row condition {text!r}: {what} is not allowed ({_HELP})")


def _no_at(text: str, cleaned: str) -> None:
    """``@name``, pandas' reference to a Python variable, outside quoted text."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(cleaned).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return  # ast.parse below reports it
    if any(t.type == tokenize.OP and t.string in ("@", "@=") for t in tokens):
        raise _refuse(text, "@ (a reference to a Python variable)")


def check_condition(text: str) -> ast.Expression:
    """Parse the row condition ``text`` and check every node against the allowed ones (module docstring); returns the
    tree (backtick names as ``_bt0``, ``_bt1``, ...). Raises ``SpecificationError``."""
    if not isinstance(text, str):
        raise ConditionError(f"a row condition is text, e.g. \"pop > 1e6\", not {type(text).__name__}")
    if len(text) > MAX_LENGTH:
        raise ConditionError(f"row condition of {len(text):,} characters: at most {MAX_LENGTH:,}")
    quoted: list[str] = []

    def placeholder(m: re.Match) -> str:
        quoted.append(m.group(1))
        return f" _bt{len(quoted) - 1} "

    cleaned = _BACKTICKS.sub(placeholder, text)
    for name in quoted:
        if name.strip().startswith("__"):
            raise _refuse(text, f"the name `{name}` (names starting with __)")
    _no_at(text, cleaned)
    try:
        tree = ast.parse(cleaned.strip(), mode="eval")
    except SyntaxError as exc:
        raise ConditionError(f"row condition {text!r} cannot be read: {exc.msg}") from None
    _Check(text, len(quoted)).visit(tree.body)
    return tree


class _Check:
    def __init__(self, text: str, backticks: int):
        self.text = text
        self.backticks = {f"_bt{i}" for i in range(backticks)}

    def refuse(self, what: str) -> ConditionError:
        return _refuse(self.text, what)

    def describe(self, node: Any) -> str:
        if isinstance(node, ast.Call):
            fn = node.func.id if isinstance(node.func, ast.Name) else ast.unparse(node.func)
            return f"a call to {fn}()"
        if isinstance(node, ast.Constant):
            return f"the value {node.value!r}"
        return _WORDS.get(type(node), type(node).__name__)

    def visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            if node.id.startswith("__") and node.id not in self.backticks:
                raise self.refuse(f"the name {node.id} (names starting with __)")
            return
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float, str, bool):
                raise self.refuse(self.describe(node))
            return
        if isinstance(node, ast.BoolOp):
            for v in node.values:
                self.visit(v)
            return
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _UNARY):
                raise self.refuse(type(node.op).__name__)
            self.operand(node.operand)
            return
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ARITHMETIC):
                raise self.refuse(_WORDS.get(type(node.op), type(node.op).__name__))
            if isinstance(node.op, ast.Pow):
                self.power(node)
            self.operand(node.left)
            self.operand(node.right)
            return
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if not isinstance(op, _COMPARE):
                    raise self.refuse(_WORDS.get(type(op), type(op).__name__))
            for side in (node.left, *node.comparators):
                if isinstance(side, (ast.List, ast.Tuple)):
                    self.items(side)
                else:
                    self.visit(side)
            return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                raise self.refuse(self.describe(node))
            if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
                raise self.refuse(f"keyword or * arguments to {node.func.id}()")
            wanted = 2 if node.func.id == "arctan2" else 1
            if len(node.args) != wanted:
                raise ConditionError(f"row condition {self.text!r}: {node.func.id}() takes {wanted} argument"
                                         f"{'s' if wanted > 1 else ''}, not {len(node.args)}")
            for a in node.args:
                self.operand(a)
            return
        if isinstance(node, (ast.List, ast.Tuple)):
            raise self.refuse("a list outside a comparison (write x in [1, 2])")
        raise self.refuse(self.describe(node))

    def operand(self, node: ast.AST) -> None:
        """An operand of arithmetic or a function: not text, not a list (``'a' * 10**9`` would fill the memory)."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            raise self.refuse("arithmetic on text")
        if isinstance(node, (ast.List, ast.Tuple)):
            raise self.refuse("arithmetic on a list")
        self.visit(node)

    def power(self, node: ast.BinOp) -> None:
        """``x ** k``: k a number, at most ``MAX_POWER`` in size, and no power inside the base (``9**99**99`` and
        ``((9**99)**99)**99`` would compute a huge whole number before anything else happens)."""
        e = node.right
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, (ast.USub, ast.UAdd)):
            e = e.operand
        if not (isinstance(e, ast.Constant) and type(e.value) in (int, float) and abs(e.value) <= MAX_POWER):
            raise self.refuse(f"a power whose exponent is not a number of at most {MAX_POWER}")
        if any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow) for n in ast.walk(node.left)):
            raise self.refuse("a power of a power")

    def items(self, node: ast.List | ast.Tuple) -> None:
        """The list of ``x in [...]`` (or ``x == [...]``): numbers, text, True/False."""
        for item in node.elts:
            value = item.operand if isinstance(item, ast.UnaryOp) and isinstance(item.op, (ast.USub, ast.UAdd)) else item
            if not (isinstance(value, ast.Constant) and type(value.value) in (int, float, str, bool)):
                raise self.refuse(f"{self.describe(value)} in a list (a list holds numbers, text and True/False)")
            if value is not item and isinstance(value.value, str):
                raise self.refuse("arithmetic on text")


def condition_mask(frame: Any, where: str) -> np.ndarray:
    """The rows of ``frame`` (a Dataset's frame, its axes as index levels) where ``where`` holds: checked first
    (``check_condition``), then evaluated by ``DataFrame.eval`` on the frame with its index as columns, with no Python
    variables in reach."""
    check_condition(where)
    return np.asarray(frame.reset_index().eval(where, local_dict={}, global_dict={}), dtype=bool)
