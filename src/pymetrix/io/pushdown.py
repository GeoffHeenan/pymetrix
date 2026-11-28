"""Row conditions as pyarrow filter expressions, so a Parquet read skips what it does not need.

``Dataset.read("big.parquet", where="country in ['DEU', 'FRA'] and gdp > 100")`` hands pyarrow the parts of the
condition it can test from a file's metadata: partition folders (``year=2010/``) and row groups whose statistics (each
column's minimum and maximum) show that no row can match are not read. pyarrow's filter is only a first cut. The
condition is then evaluated on the rows read exactly as ``Dataset.select(where=...)`` evaluates it (pandas'
``DataFrame.eval``, after the check in ``pymetrix.data.conditions``), so the rows kept are those that reading the whole
file and selecting would keep, and any difference between the two engines (missing values above all) cannot change
the result.

What is pushed down (each part only when the column's type and the value's agree: numbers with numeric columns, text
with text columns, dates in quotes with date columns, True/False with true/false columns):

- comparisons of a column with a value (``x > 3``, ``3 < x``, ``country == 'DEU'``, ``x != 0``);
- ``in [...]`` and ``not in [...]`` with a list of values;
- on the time axis and on date columns, ``<``, ``<=``, ``>`` and ``>=`` with a period or date in quotes
  (``time >= '2001Q1'``, ``year >= '2005'``): a plain file's time column, whose values become periods only once
  read, is cut as ``span=`` is (``span_expression``), wide enough to keep every row that can match;
- ``and`` / ``&`` (each part that can be pushed is; the others are left to the exact evaluation);
- ``or`` / ``|`` when every part can be pushed.

Not pushed (evaluated on the rows read only): arithmetic and functions (``log(x) > 1``, ``x + y > 0``), comparisons
of two columns, ``not`` / ``~``, and ``==``, ``!=`` and ``in`` on periods and dates (pandas' ``DataFrame.eval``
compares a period or date with text for order but takes ``==`` as never true and ``!=`` as always true, and the
first cut must not drop a row it keeps). A condition with nothing to push reads every row group.

pyarrow drops rows where a comparison is missing (null); pandas keeps them for ``!=`` and ``not in`` (``NaN != 0`` is
true). The expressions below keep those rows too (``x != 0 or x is null``), so the first cut never drops a row the
exact evaluation would keep.
"""

from __future__ import annotations

import ast
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from ..data.conditions import check_condition
from ..errors import PyMetrixError

__all__ = ["condition_expression", "span_expression"]

_OPS = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}
_FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}


class _NotPushed(PyMetrixError):
    """This part of the condition is left to the exact evaluation after reading (caught here, never raised to the
    user; a PyMetrixError as every error pymetrix raises)."""


def condition_expression(where: str, schema: pa.Schema, time: tuple[str, str | None] | None = None,
                         periods: dict[str, str] | None = None) -> tuple[pc.Expression | None, list[str]]:
    """The pyarrow expression for the parts of row condition ``where`` that can be pushed down on a file of
    ``schema`` (None when none can), and the parts that cannot, as text, for the provenance. ``time``: a plain file's
    time column and frequency, whose values become periods only once read: a comparison of it with a period in quotes
    (``year >= '2005'``) is pushed as a span is (``span_expression``). ``periods``: text columns holding periods of a
    frequency (a pymetrix file's time axis), compared as periods written the standard way (``2001Q1``)."""
    tree = check_condition(where)
    names = _backtick_names(where)
    left_out: list[str] = []
    expr = _Translate(schema, time, periods or {}, names, left_out).conjunction(tree.body)
    return expr, left_out


def _backtick_names(text: str) -> dict[str, str]:
    from ..data.conditions import _BACKTICKS

    return {f"_bt{i}": m.group(1) for i, m in enumerate(_BACKTICKS.finditer(text))}


class _Translate:
    def __init__(self, schema: pa.Schema, time: tuple[str, str | None] | None, periods: dict[str, str],
                 backticks: dict[str, str], left_out: list[str]):
        self.schema, self.time, self.periods, self.backticks, self.left_out = schema, time, periods, backticks, left_out

    def conjunction(self, node: ast.AST) -> pc.Expression | None:
        """``a and b and c``: each part that can be pushed is (a superset of the rows stays a superset)."""
        parts = self._and_parts(node)
        pushed = []
        for part in parts:
            try:
                pushed.append(self.translate(part))
            except _NotPushed:
                self.left_out.append(self._text(part))
        if not pushed:
            return None
        out = pushed[0]
        for e in pushed[1:]:
            out = out & e
        return out

    def _and_parts(self, node: ast.AST) -> list[ast.AST]:
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            return [p for v in node.values for p in self._and_parts(v)]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            return self._and_parts(node.left) + self._and_parts(node.right)
        return [node]

    def _text(self, node: ast.AST) -> str:
        text = ast.unparse(node)
        for placeholder, name in self.backticks.items():
            text = text.replace(placeholder, f"`{name}`")
        return text

    def translate(self, node: ast.AST) -> pc.Expression:
        """The whole of ``node`` as an expression, or ``_NotPushed``."""
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And) or isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            parts = [self.translate(p) for p in self._and_parts(node)]
            out = parts[0]
            for e in parts[1:]:
                out = out & e
            return out
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or) or isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            sides = node.values if isinstance(node, ast.BoolOp) else [node.left, node.right]
            parts = [self.translate(p) for p in sides]  # every side, or the "or" is not pushed
            out = parts[0]
            for e in parts[1:]:
                out = out | e
            return out
        if isinstance(node, ast.Compare):
            return self.compare(node)
        raise _NotPushed

    def compare(self, node: ast.Compare) -> pc.Expression:
        out = None
        left = node.left
        for op, right in zip(node.ops, node.comparators):  # a < x < b: each link
            part = self.link(left, op, right)
            out = part if out is None else out & part
            left = right
        assert out is not None
        return out

    def link(self, left: ast.AST, op: ast.cmpop, right: ast.AST) -> pc.Expression:
        if isinstance(op, (ast.In, ast.NotIn)):
            column = self.column(left)
            if self.ordered_only(column):
                raise _NotPushed
            if not isinstance(right, (ast.List, ast.Tuple)):
                raise _NotPushed
            values = [self.value(column, item) for item in right.elts]
            field = pc.field(column)
            listed = field.isin(self.typed(lambda: pa.array(values, type=self.type_of(column))))
            return listed if isinstance(op, ast.In) else (~listed | field.is_null())
        word = _OPS.get(type(op))
        if word is None:
            raise _NotPushed
        if isinstance(left, ast.Name) and not isinstance(right, ast.Name):
            column, value = self.column(left), self.value(self.column(left), right)
        elif isinstance(right, ast.Name) and not isinstance(left, ast.Name):
            column, value, word = self.column(right), self.value(self.column(right), left), _FLIP[word]
        else:
            raise _NotPushed  # two columns, or two values
        if self.ordered_only(column) and word in ("==", "!="):
            raise _NotPushed
        if isinstance(value, _Bound):  # a plain file's time column against a period
            bounds = (value.text, None) if word in (">", ">=") else (None, value.text)
            cut = span_expression(column, self.type_of(column), bounds, self.time[1] if self.time else None)
            if cut is None:
                raise _NotPushed
            return cut
        field = pc.field(column)
        scalar = self.typed(lambda: pa.scalar(value, type=self.type_of(column)))
        expr = {"==": field == scalar, "!=": field != scalar, "<": field < scalar, "<=": field <= scalar,
                ">": field > scalar, ">=": field >= scalar}[word]
        return expr | field.is_null() if word == "!=" else expr  # pandas: NaN != v is true

    @staticmethod
    def typed(make: Any) -> Any:
        """A value or list of the column's type; one that does not fit (1e12 in a 32-bit column) is not pushed."""
        try:
            return make()
        except (pa.ArrowInvalid, pa.ArrowTypeError, OverflowError, ValueError, TypeError):
            raise _NotPushed from None

    def column(self, node: ast.AST) -> str:
        if not isinstance(node, ast.Name):
            raise _NotPushed
        name = self.backticks.get(node.id, node.id)
        if name not in self.schema.names:
            raise _NotPushed
        return name

    def ordered_only(self, column: str) -> bool:
        """Periods and dates: only < <= > >= are pushed. pandas' ``DataFrame.eval`` compares periods (and dates) with
        text for order, but takes ``==`` as never and ``!=`` as always true, so pushing those could drop rows the exact
        evaluation keeps."""
        kind = self.type_of(column)
        return (column in self.periods or (self.time is not None and column == self.time[0])
                or pa.types.is_timestamp(kind) or pa.types.is_date(kind))

    def type_of(self, column: str) -> pa.DataType:
        kind = self.schema.field(column).type
        return kind.value_type if pa.types.is_dictionary(kind) else kind

    def value(self, column: str, node: ast.AST) -> Any:
        """The constant ``node`` as a value of ``column``'s type, or ``_NotPushed`` when they do not agree."""
        sign = 1
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            sign = -1 if isinstance(node.op, ast.USub) else 1
            node = node.operand
        if not isinstance(node, ast.Constant):
            raise _NotPushed
        v = node.value
        kind = self.type_of(column)
        if self.time is not None and column == self.time[0] and not isinstance(v, str):
            raise _NotPushed  # a number against a column that becomes periods: pandas refuses it after reading
        if isinstance(v, bool):
            if pa.types.is_boolean(kind) and sign == 1:
                return v
            raise _NotPushed
        if isinstance(v, (int, float)):
            if pa.types.is_integer(kind):
                if isinstance(v, float) and not v.is_integer():
                    raise _NotPushed  # x > 2.5 on whole numbers: left to pandas, not rounded here
                return int(sign * v)
            if pa.types.is_floating(kind):
                return float(sign * v)
            raise _NotPushed
        if isinstance(v, str) and sign == 1:
            if self.time is not None and column == self.time[0]:
                return _Bound(v)  # the column's values become periods once read: pushed as a span
            if column in self.periods:  # a pymetrix file's time axis: periods as text, compared as written by it
                try:
                    return str(pd.Period(v, freq=self.periods[column]))
                except (ValueError, TypeError):
                    raise _NotPushed from None
            if pa.types.is_string(kind) or pa.types.is_large_string(kind):
                return v
            if pa.types.is_timestamp(kind) and kind.tz is None or pa.types.is_date(kind):
                try:
                    stamp = pd.Timestamp(v)
                except (ValueError, TypeError):
                    raise _NotPushed from None
                return stamp.date() if pa.types.is_date(kind) else stamp.to_pydatetime()
        raise _NotPushed


class _Bound:
    """A period in quotes compared with a plain file's time column."""

    def __init__(self, text: str):
        self.text = text


def span_expression(column: str, kind: pa.DataType, span: tuple[Any, Any], freq: str | None,
                    periods: str | None = None) -> pc.Expression | None:
    """A first cut for ``span=(start, end)`` on the time column ``column`` of type ``kind``, wide enough to keep every
    row the span keeps once the column is read as periods: whole years for a column of years, the first day of the
    first period to the last day of the last for dates, the periods themselves for a pymetrix file's time axis
    (``periods`` its frequency). None when the column's type allows no cut (periods as text in a plain file)."""
    start, end = span
    field = pc.field(column)
    parts = []
    for bound, first in ((start, True), (end, False)):
        if bound is None:
            continue
        guesses = []
        for f in dict.fromkeys((periods, freq, None)):
            try:
                guesses.append(pd.Period(str(bound), freq=f) if f else pd.Period(str(bound)))
            except (ValueError, TypeError):
                continue
        if not guesses:
            return None
        if periods is not None:
            if not (pa.types.is_string(kind) or pa.types.is_large_string(kind)):
                return None
            value: Any = str(guesses[0])
            parts.append(field >= value if first else field <= value)
        elif pa.types.is_integer(kind):  # years; a fiscal year's name may be either calendar year: one more each side
            years = [g.year for g in guesses]
            margin = 0 if str(freq or "").upper() in ("Y", "A", "Y-DEC", "A-DEC", "YE", "YE-DEC") else 1
            value = min(years) - margin if first else max(years) + margin
            scalar = pa.scalar(value, type=kind)
            parts.append(field >= scalar if first else field <= scalar)
        elif pa.types.is_timestamp(kind) and kind.tz is None or pa.types.is_date(kind):
            # from the first moment of the first period to before the first moment of the period after the last
            stamp = min(g.start_time for g in guesses) if first else max((g + 1).start_time for g in guesses)
            scalar = pa.scalar(stamp.date() if pa.types.is_date(kind) else stamp.to_pydatetime(), type=kind)
            parts.append(field >= scalar if first else field < scalar)
        else:
            return None
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else parts[0] & parts[1]
