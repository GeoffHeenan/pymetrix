"""Named matrices as steps: Model > Matrix (an expression, from data or a result, or values typed or pasted in a grid)
and the actions of a matrix window (Transpose, Inverse, Cholesky, Decompose, ...).

Every matrix step records one line, ``name = <expression>``, in the package's own syntax (``V = Matrix.cov(fit)``,
``R = V.chol(upper=True)``, ``M = Matrix([[1, 0.5], [0.5, 1]], rows=['a', 'b'], cols=['a', 'b'])``); the step runs
exactly that expression, checked first (``pymetrix.matrix_expr``), so the script and the dialog give the same matrix.
Steps read the objects their expression names, so a matrix made from a fit is recomputed when the fit is.
"""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from ..errors import SpecificationError
from ..matrix_expr import ALLOWED_BUILTINS, GIVEN, MATRIX_EXPRESSION, ExpressionError, check_expression, evaluate, expression_names
from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session

__all__ = ["DefineMatrix", "EnterMatrix", "MatrixFrom", "format_number", "live_check", "matrix_literal", "parse_grid", "parse_number"]

TOPICS = ("matrices",)
RESULTS = "FittedEquation,FittedSystem,FittedVAR,FittedVECM,FittedARDL,FittedModel"
VARS = "FittedVAR,FittedVECM"


def _name_errors(name: str, what: str = "Name") -> list[str]:
    name = name.strip()
    if not name:
        return [f"{what}: give the matrix a name"]
    if not name.isidentifier() or keyword.iskeyword(name):
        return [f"{what}: {name!r} is not a Python name (letters, digits and _, not starting with a digit)"]
    if name in GIVEN or name in ALLOWED_BUILTINS or name.startswith("_"):
        return [f"{what}: {name!r} is taken (Matrix, np and the built-ins keep their meaning); choose another"]
    return []


def _objects(text: str) -> list[str]:
    """The session objects an expression names: every name but Matrix, np and the built-ins."""
    return sorted(expression_names(text) - set(GIVEN) - set(ALLOWED_BUILTINS))


def _expression_call(expression: str, name: str, recorded_by: str) -> Any:
    from ..calls import make_call
    from ..codec import Ref

    return make_call(MATRIX_EXPRESSION, output=name.strip(), recorded_by=recorded_by, expression=expression.strip(),
                     inputs={n: Ref(n) for n in _objects(expression)})


@command(menu="Model/Matrix expression...", id="pymetrix:define_matrix")
@dataclass(frozen=True)
class DefineMatrix(Command):
    """A named matrix from an expression in pymetrix's matrix syntax, using the session's objects by name:
    ``(X.T @ X).inv() @ X.T @ y``, ``Matrix.cov(fit)``, ``V.chol(upper=True)``, ``A.kron(Matrix.identity(3))``,
    ``A.svd()`` (a decomposition, shown as a table for each factor).

    The expression may use ``Matrix``, ``np`` and the session's objects; the usual functions are methods (``A.inv()``,
    ``A.pinv()``, ``A.power(2)``, ``A.det()``) or ``Matrix``'s (``Matrix.lyapunov(F, Q)``, ``Matrix.hcat(A, B)``). It
    is checked before it runs: one expression of names, calls, operators, indexing and literals; imports, ``lambda``,
    attributes starting with ``_`` and anything that reads or writes files are refused. The step records
    ``name = expression`` and is recomputed when the objects it uses change.
    """

    topics: ClassVar[tuple[str, ...]] = TOPICS
    live_expression: ClassVar[str] = "expression"  # the dialog checks it as it is typed (live_check)

    expression: Annotated[str, Widget("text", help="e.g. (X.T @ X).inv() @ X.T @ y, Matrix.cov(fit), V.chol(upper=True), A.svd()")]
    name: str = "M"

    def validate(self, session: Session) -> list[str]:
        errors = _name_errors(self.name)
        try:
            check_expression(self.expression, session.objects)
        except ExpressionError as exc:
            errors.append(str(exc))
        return errors

    def reads(self, session: Session, versions: dict[str, int]) -> set[str]:
        return set(_objects(self.expression)) & set(versions)

    def to_call(self) -> Any:
        return _expression_call(self.expression, self.name, self.command_id)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return f"{self.name.strip()} = {self.expression.strip()}"


def live_check(expression: str, name: str, session: Session) -> Any:
    """The Expression… dialog's check as it is typed (``pymetrix.matrix_expr.preview``): the name's problem or the
    expression's, else what it gives when it evaluates cheaply. Nothing is recorded or changed."""
    from ..matrix_expr import ExpressionPreview, preview

    found = preview(expression, session.objects)
    errors = _name_errors(name)
    if errors and found.ok:
        return ExpressionPreview(problem=errors[0])
    return found


SOURCES = ("coefficients", "covariance", "lag matrix", "error covariance", "series", "design")


@command(menu="Model/Matrix from data or a result...", id="pymetrix:matrix_from")
@dataclass(frozen=True)
class MatrixFrom(Command):
    """A named matrix from a result or a dataset: a fit's coefficients (a column) or the covariance matrix of its
    estimates, a VAR's lag matrix or the covariance matrix of its errors, a dataset's series over a span (rows named by
    period), or a formula's regressors X and dependent variable y on a dataset (two matrices).

    Recorded as the line that makes it, e.g. ``V = Matrix.cov(fit)``, ``A1 = Matrix.var_lag(var, 1)``,
    ``Y = Matrix.from_dataset(data, ['c', 'y'], span=('2000Q1', '2019Q4'))`` or ``X, y = Matrix.design('c ~ y',
    data)``, and recomputed when the result or the data changes.
    """

    topics: ClassVar[tuple[str, ...]] = TOPICS

    source: Annotated[Literal["coefficients", "covariance", "lag matrix", "error covariance", "series", "design"],
                      Widget("choice", label="Matrix of", help="what the matrix holds")] = "coefficients"
    result: Annotated[str, Widget("object", source=RESULTS, help="a fitted equation, system, VAR or model",
                                  show_if=("source", ("coefficients", "covariance")))] = ""
    var: Annotated[str, Widget("object", source=VARS, label="VAR", help="a fitted VAR or VECM",
                               show_if=("source", ("lag matrix", "error covariance")))] = ""
    lag: Annotated[int, Widget("number", min=1, max=48, help="which lag's coefficients: 1 for the first",
                               show_if=("source", ("lag matrix",)))] = 1
    data: Annotated[str, Widget("object", source="Dataset", show_if=("source", ("series", "design")))] = "data"
    series: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the columns, in this order; none: every series",
                                  show_if=("source", ("series",)))] = ""
    formula: Annotated[str, Widget("formula", help="e.g. c ~ y + c[-1]: the regressors X and the dependent variable y",
                                   show_if=("source", ("design",)))] = ""
    start: Annotated[str, Widget("period", show_if=("source", ("series", "design")))] = ""
    end: Annotated[str, Widget("period", show_if=("source", ("series", "design")))] = ""
    name: str = "M"
    dependent: Annotated[str, Widget("text", label="Name of y", help="the dependent variable's matrix (Save as names the regressors')",
                                     show_if=("source", ("design",)))] = "y"

    def _span(self) -> str:
        if not (self.start.strip() or self.end.strip()):
            return ""
        return f", span={(self.start.strip() or None, self.end.strip() or None)!r}"

    def _names(self) -> list[str]:
        return [s.strip() for s in self.series.split(",") if s.strip()]

    def expression(self) -> str:
        """The expression the step runs, and records after ``name =``."""
        if self.source == "coefficients":
            return f"Matrix.coef({self.result})"
        if self.source == "covariance":
            return f"Matrix.cov({self.result})"
        if self.source == "lag matrix":
            return f"Matrix.var_lag({self.var}, {int(self.lag)})"
        if self.source == "error covariance":
            return f"Matrix.var_sigma({self.var})"
        if self.source == "series":
            names = self._names()
            return f"Matrix.from_dataset({self.data}{f', {names!r}' if names else ''}{self._span()})"
        return f"Matrix.design({self.formula.strip()!r}, {self.data}{self._span()})"

    def _input(self) -> str:
        return self.result if self.source in ("coefficients", "covariance") else self.var if self.source in ("lag matrix", "error covariance") else self.data

    def validate(self, session: Session) -> list[str]:
        if self.source not in SOURCES:
            return [f"Matrix of: one of {', '.join(SOURCES)}"]
        errors = _name_errors(self.name)
        what = {"coefficients": "a fitted result", "covariance": "a fitted result", "lag matrix": "a VAR",
                "error covariance": "a VAR", "series": "a dataset", "design": "a dataset"}[self.source]
        source = self._input().strip()
        if not source:
            errors.append(f"choose {what}")
        elif source not in session.objects:
            errors.append(f"no object named {source!r}")
        elif not source.isidentifier():
            errors.append(f"{source!r} is not a name a script can use")
        if self.source == "series" and source in session.objects:
            have = set(getattr(getattr(session.objects[source], "frame", None), "columns", ()))
            missing = [n for n in self._names() if n not in have]
            if missing:
                errors.append(f"{source} has no series {', '.join(missing)}")
        if self.source == "design":
            if "~" not in self.formula:
                errors.append("write the formula as y ~ x1 + x2")
            errors += _name_errors(self.dependent, "Name of y")
            if self.dependent.strip() == self.name.strip():
                errors.append("the regressors and the dependent variable need different names")
        if self.source == "lag matrix" and int(self.lag) < 1:
            errors.append("the lag is 1 or more")
        return errors

    def reads(self, session: Session, versions: dict[str, int]) -> set[str]:
        return {self._input()} & set(versions)

    def to_call(self) -> Any:
        """The recorded call (a matrix expression); the design matrices, two results from one line, are recorded as
        their code only."""
        if self.source == "design":
            raise SpecificationError("the design matrices are two results: recorded as their code")
        return _expression_call(self.expression(), self.name, self.command_id)

    def run(self, session: Session) -> Any:
        if self.source == "design":
            X, y = evaluate(self.expression(), session.objects, pair=True)
            session.objects[self.name.strip()] = X
            session.objects[self.dependent.strip()] = y
            return X
        return self.to_call().run(session)

    def to_code(self) -> str:
        if self.source == "design":
            return f"{self.name.strip()}, {self.dependent.strip()} = {self.expression()}"
        return f"{self.name.strip()} = {self.expression()}"


# -- values in a grid -------------------------------------------------------------------------------------------------

_NUMBER = re.compile(r"^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?$")


def parse_number(text: str) -> float:
    """A number as typed or pasted: ``1.5``, ``-2``, ``1e-3``, a decimal comma (``1,5``), a Unicode minus, ``nan``
    and ``inf``; thousands separators are not read (``1,234`` is 1.234)."""
    t = str(text).strip().replace("−", "-").replace(" ", "").replace(" ", "")
    if not t:
        raise SpecificationError("empty")
    if t.lower() in ("nan", "inf", "+inf", "-inf"):
        return float(t)
    if not _NUMBER.match(t):
        raise SpecificationError(f"not a number: {text!r}")
    return float(t.replace(",", "."))


def _split(line: str, sep: str | None) -> list[str]:
    return line.split(sep) if sep is not None else line.split()


def parse_grid(text: str) -> tuple[list[list[float]], list[str] | None, list[str] | None]:
    """Cells pasted from Excel or typed: rows on lines; cells separated by tabs (Excel's copy), else semicolons, else
    commas (unless spaces separate the numbers: ``1 2,5``), else spaces. A decimal comma is read where the separator
    is not a comma. A first row or column that is
    not numbers holds the column or row names. Returns (values, row names or None, column names or None)."""
    lines = [ln for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if ln.strip()]
    if not lines:
        raise SpecificationError("nothing to read: paste or type the numbers, a row per line")
    sep = "\t" if any("\t" in ln for ln in lines) else ";" if any(";" in ln for ln in lines) else "," if any("," in ln for ln in lines) else None
    if sep == "," and any(len(c.strip().split()) > 1 for ln in lines for c in ln.split(",")):
        sep = None  # cells separated by spaces, with decimal commas: 1 2,5
    cells = [[c.strip() for c in _split(ln, sep)] for ln in lines]
    if sep is not None:  # trailing empty cells (a spreadsheet's selection) are not columns
        width = max(max((i + 1 for i, c in enumerate(row) if c), default=0) for row in cells)
        cells = [row[:width] + [""] * (width - len(row)) for row in cells]

    def numeric(cell: str) -> bool:
        try:
            parse_number(cell)
            return True
        except ValueError:
            return False

    cols = None
    if len(cells) > 1 and not all(numeric(c) for c in cells[0][1:] or cells[0]):
        cols, cells = cells[0], cells[1:]
    rows = None
    if cells and all(not numeric(row[0]) for row in cells) and len(cells[0]) > 1:
        rows = [row[0] for row in cells]
        cells = [row[1:] for row in cells]
        if cols is not None:
            cols = cols[1:]
    if cols is not None and len(cols) > len(cells[0]) and not cols[0]:  # the corner cell
        cols = cols[1:]
    width = len(cells[0])
    values: list[list[float]] = []
    for i, row in enumerate(cells, 1):
        if len(row) != width:
            raise SpecificationError(f"row {i} has {len(row)} cells; the first has {width}")
        out = []
        for j, cell in enumerate(row, 1):
            if not cell:
                raise SpecificationError(f"row {i}, column {j} is empty")
            try:
                out.append(parse_number(cell))
            except ValueError:
                raise SpecificationError(f"row {i}, column {j}: {cell!r} is not a number") from None
        values.append(out)
    if cols is not None and len(cols) != width:
        raise SpecificationError(f"{len(cols)} column names for {width} columns")
    return values, rows, cols


def format_number(v: float) -> str:
    """A number as a matrix literal writes it: whole numbers without a point, others as Python's shortest repr."""
    if v != v:
        return "np.nan"
    if v in (float("inf"), float("-inf")):
        return "np.inf" if v > 0 else "-np.inf"
    return str(int(v)) if float(v).is_integer() and abs(v) < 1e15 else repr(float(v))


def matrix_literal(values: list[list[float]], rows: list[str] | None = None, cols: list[str] | None = None) -> str:
    """``Matrix([[...], ...], rows=[...], cols=[...])``: the names left out when numbered 1, 2, ... (the default)."""
    body = ", ".join("[" + ", ".join(format_number(v) for v in row) + "]" for row in values)
    parts = [f"[{body}]"]
    n, k = len(values), len(values[0]) if values else 0
    if rows and list(rows) != [str(i) for i in range(1, n + 1)]:
        parts.append(f"rows={list(rows)!r}")
    if cols and list(cols) != [str(i) for i in range(1, k + 1)]:
        parts.append(f"cols={list(cols)!r}")
    return f"Matrix({', '.join(parts)})"


def _names_list(text: str) -> list[str]:
    return [p.strip() for p in text.split(",")] if text.strip() else []


@command(menu="Model/Enter matrix...", id="pymetrix:enter_matrix")
@dataclass(frozen=True)
class EnterMatrix(Command):
    """A named matrix from numbers typed or pasted into a grid (from Excel, or text with tabs, semicolons, commas or
    spaces between the cells), with names for its rows and columns.

    Model > Matrix > Enter values opens the grid: set the numbers of rows and columns, type or paste the numbers
    (Ctrl+V; a first row or column of names is read as the names), and double-click a row or column header to rename
    it. Recorded as the matrix written out, ``M = Matrix([[1, 0.5], [0.5, 1]], rows=['a', 'b'], cols=['a', 'b'])``.
    """

    topics: ClassVar[tuple[str, ...]] = TOPICS

    values: Annotated[str, Widget("multiline", help="the numbers: a row per line, cells separated by tabs, commas or spaces")]
    rows: Annotated[str, Widget("text", help="row names, separated by commas; empty: numbered 1, 2, ...")] = ""
    cols: Annotated[str, Widget("text", help="column names, separated by commas; empty: numbered 1, 2, ...")] = ""
    name: str = "M"

    def grid(self) -> tuple[list[list[float]], list[str], list[str]]:
        values, rows, cols = parse_grid(self.values)
        if rows is not None or cols is not None:
            raise SpecificationError("the values are numbers only: give the names under Rows and Columns")
        rows, cols = _names_list(self.rows), _names_list(self.cols)
        n, k = len(values), len(values[0])
        for what, names, size in (("row", rows, n), ("column", cols, k)):
            if names and len(names) != size:
                raise SpecificationError(f"{len(names)} {what} names for {size} {what}s")
            if any(not x for x in names):
                raise SpecificationError(f"a {what} name is empty")
            if len(set(names)) != len(names):
                raise SpecificationError(f"{what} names must differ: {next(x for x in names if names.count(x) > 1)!r} is used twice")
        return values, rows, cols

    def expression(self) -> str:
        values, rows, cols = self.grid()
        return matrix_literal(values, rows or None, cols or None)

    def validate(self, session: Session) -> list[str]:
        errors = _name_errors(self.name)
        try:
            self.grid()
        except SpecificationError as exc:
            errors.append(str(exc))
        return errors

    def reads(self, session: Session, versions: dict[str, int]) -> set[str]:
        return set()

    def to_call(self) -> Any:
        return _expression_call(self.expression(), self.name, self.command_id)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return f"{self.name.strip()} = {self.expression()}"


# -- a matrix window's actions ----------------------------------------------------------------------------------------

# (label, expression with {m} for the matrix, suffix of the new matrix's name); "-" a separator
MATRIX_ACTIONS: tuple[tuple[str, str, str], ...] = (
    ("Transpose", "{m}.T", "T"), ("Inverse", "{m}.inv()", "inv"), ("Pseudo-inverse", "{m}.pinv()", "pinv"),
    ("Cholesky factor (lower)", "{m}.chol()", "chol"), ("Cholesky factor (upper)", "{m}.chol(upper=True)", "cholU"),
)
DECOMPOSE_ACTIONS: tuple[tuple[str, str, str], ...] = (
    ("Singular values (SVD)", "{m}.svd()", "svd"), ("Eigenvalues", "{m}.eig()", "eig"), ("QR", "{m}.qr()", "qr"),
    ("LU", "{m}.lu()", "lu"), ("Schur", "{m}.schur()", "schur"),
)


def unique_name(base: str, taken: Any) -> str:
    """``base``, or ``base2``, ``base3``, ... : the first not in ``taken``."""
    if base not in taken:
        return base
    n = 2
    while f"{base}{n}" in taken:
        n += 1
    return f"{base}{n}"


def action_step(matrix: str, expression: str, suffix: str, taken: Any, name: str | None = None) -> DefineMatrix:
    """The step a matrix window's action records: ``{matrix}_{suffix} = <expression>``."""
    code = expression.format(m=matrix)
    return DefineMatrix(code, name=name or unique_name(f"{matrix}_{suffix}", taken))


def power_step(matrix: str, n: int, taken: Any, name: str | None = None) -> DefineMatrix:
    suffix = f"pow{n}" if n >= 0 else f"pow_m{-n}"
    return DefineMatrix(f"{matrix}.power({int(n)})", name=name or unique_name(f"{matrix}_{suffix}", taken))


def lyapunov_step(matrix: str, q: str, taken: Any, name: str | None = None) -> DefineMatrix:
    return DefineMatrix(f"Matrix.lyapunov({matrix}, {q})", name=name or unique_name(f"{matrix}_lyap", taken))


def result_matrix_steps(name: str, obj: Any, taken: Any) -> list[tuple[str, MatrixFrom]]:
    """The "As matrix" choices for a result: its coefficients and covariance matrix; for a VAR also each lag's
    matrix and the errors' covariance. Empty for objects that have none."""
    out: list[tuple[str, MatrixFrom]] = []
    params = getattr(obj, "params", None)
    if params is not None and hasattr(params, "names"):
        out.append(("Coefficients", MatrixFrom("coefficients", result=name, name=unique_name(f"{name}_b", taken))))
        if getattr(params, "cov", None) is not None:
            out.append(("Covariance of the estimates", MatrixFrom("covariance", result=name, name=unique_name(f"{name}_V", taken))))
    coefs = getattr(obj, "coefs", None)
    if coefs is not None and hasattr(obj, "variables") and hasattr(obj, "sigma"):
        for k in range(1, len(coefs) + 1):
            out.append((f"Lag {k} matrix", MatrixFrom("lag matrix", var=name, lag=k, name=unique_name(f"{name}_A{k}", taken))))
        out.append(("Error covariance", MatrixFrom("error covariance", var=name, name=unique_name(f"{name}_Sigma", taken))))
    return out
