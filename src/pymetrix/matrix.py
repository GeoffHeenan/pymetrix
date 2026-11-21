"""Matrices with names: matrix algebra that keeps track of which row and column is which.

    X, y = Matrix.design("c ~ y + c[-1]", data)        # the regressors and the dependent variable, named
    b = (X.T @ X).inv() @ X.T @ y                       # rows named const, y, c[-1]
    V = Matrix.cov(fit)                                 # a result's covariance matrix, rows and columns named
    V["y", "y"] ** 0.5                                  # a standard error, by name
    A1 = Matrix.var_lag(var_fit, 1)                     # a VAR's first lag matrix (rows: equations, columns: variables)
    S = Matrix.lyapunov(F, Q)                           # X = F X F' + Q: a VAR's unconditional variance

Products check that the names meet: ``A @ B`` needs A's columns to be B's rows, in the same order, unless one of them
is unnamed (numbered from 1). ``+``, ``-``, ``*``, ``/`` and ``**`` work element by element on matrices with the same
names, or with a number (``A.power(n)`` is the matrix power; ``A ** n`` on a square matrix with a whole n still gives
it, with a DeprecationWarning, until it becomes element by element as in NumPy); so do NumPy's element-wise functions (``np.log(m)``, ``np.exp``, ``np.sqrt``, ``abs``), which keep the
names. The usual functions are methods:

- ``inv``, ``pinv``, ``solve``, ``power``, ``det``, ``trace``, ``rank``, ``norm``, ``cond``, ``kron``, ``vec``, ``vech``,
  ``diag``, ``T``, ``outer``, ``inner``, ``sum``, ``mean``;
- factors: ``chol`` (lower, or ``upper=True``), ``svd`` (U, s, V with ``A = U diag(s) V'``), ``qr``, ``lu`` (``A = P L U``), ``schur``
  (``A = Q T Q'``), ``qz`` (``A = Q AA Z'``, ``B = Q BB Z'``), ``eig`` (complex eigenvalues for a general matrix,
  real ones for a symmetric one) and ``eigh``;
- functions of a matrix: ``expm``, ``logm``, ``sqrtm``; subspaces ``null``, ``orth`` and ``perp`` (the orthogonal
  complement, as for a VECM's alpha-perp and beta-perp);
- the econometric staples ``Matrix.commutation``, ``Matrix.duplication``, ``Matrix.elimination``, ``Matrix.unvech``
  and the solvers ``Matrix.lyapunov`` (discrete or continuous), ``Matrix.sylvester`` and ``Matrix.riccati`` (discrete
  algebraic Riccati);
- assembly: ``Matrix.hcat``, ``Matrix.vcat`` (names must meet, and may not clash), ``set`` (an element or a block, by
  name, as a new matrix), ``Matrix.from_dataset`` and ``to_dataset`` (rows named by period).

Factors name their new dimension (``s1``, ``s2``, ... for singular values; ``λ1``, ... for eigenvalues; ``q1``, ... for
QR; ``l1``, ... for LU; ``u1``, ... for Schur vectors; ``null1``, ``orth1``, ``perp1``, ...), so that the factors
multiply back with names that meet. A matrix may hold complex numbers (eigenvalues, a logarithm or square root that
has no real value); everything else is real. A matrix shows as a table in its window, and as NumPy (``values``) or
pandas (``to_frame``) whenever wanted.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from .errors import SpecificationError

__all__ = ["Matrix", "SVD", "QR", "LU", "Eig", "Schur", "QZ"]


def _numbered(n: int) -> tuple[str, ...]:
    return tuple(str(i) for i in range(1, n + 1))


def _seq(prefix: str, n: int) -> tuple[str, ...]:
    return tuple(f"{prefix}{i}" for i in range(1, n + 1))


def _names_of(n: int | Sequence[str]) -> tuple[str, ...]:
    return _numbered(n) if isinstance(n, (int, np.integer)) else tuple(str(x) for x in n)


def _real_if_close(values: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Real values when the imaginary parts are negligible (relative to the largest value), complex otherwise."""
    if not np.iscomplexobj(values):
        return values
    scale = max(1.0, float(np.abs(values).max())) if values.size else 1.0
    return values.real.copy() if values.size == 0 or np.abs(values.imag).max() <= tol * scale else values


def _scalar(value: Any) -> float | complex:
    value = complex(value) if np.iscomplexobj(value) else float(value)
    return value.real if isinstance(value, complex) and value.imag == 0 else value


class SVD(NamedTuple):
    """``A = U @ s.diag() @ V.T``: U's rows are A's rows, V's rows A's columns; both have columns s1, s2, ..."""

    U: Matrix
    s: Matrix
    V: Matrix


class QR(NamedTuple):
    """``A = Q @ R``, Q with orthonormal columns q1, q2, ... and R upper triangular."""

    Q: Matrix
    R: Matrix


class LU(NamedTuple):
    """``A = P @ L @ U``: P a permutation, L unit lower triangular, U upper triangular."""

    P: Matrix
    L: Matrix
    U: Matrix


class Eig(NamedTuple):
    """Eigenvalues (a column, largest first in size) and eigenvectors (the columns, in the same order)."""

    values: Matrix
    vectors: Matrix


class Schur(NamedTuple):
    """``A = Q @ T @ Q.T`` (``Q.T`` conjugated when complex), Q orthogonal (unitary) and T quasi-upper triangular."""

    T: Matrix
    Q: Matrix


class QZ(NamedTuple):
    """``A = Q @ AA @ Z.T`` and ``B = Q @ BB @ Z.T`` (the generalized Schur form)."""

    AA: Matrix
    BB: Matrix
    Q: Matrix
    Z: Matrix


@dataclass(frozen=True, eq=False)
class Matrix:
    """A two-dimensional array of numbers with names for its rows and columns."""

    values: np.ndarray
    rows: tuple[str, ...] = ()
    cols: tuple[str, ...] = ()
    title: str = ""

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        values = values.astype(complex if np.iscomplexobj(values) else float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)  # a vector is a column
        if values.ndim != 2:
            raise SpecificationError("a matrix has two dimensions: rows and columns")
        rows = tuple(str(r) for r in self.rows) or _numbered(values.shape[0])
        cols = tuple(str(c) for c in self.cols) or _numbered(values.shape[1])
        if len(rows) != values.shape[0] or len(cols) != values.shape[1]:
            raise SpecificationError(f"{values.shape[0]} x {values.shape[1]} values with {len(rows)} row and {len(cols)} column names")
        for what, names in (("row", rows), ("column", cols)):
            if len(set(names)) != len(names):
                raise SpecificationError(f"{what} names must differ: {[n for n in names if names.count(n) > 1][0]!r} is used twice")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "cols", cols)

    # -- making matrices ----------------------------------------------------------------------------------------------

    @classmethod
    def from_frame(cls, frame: pd.DataFrame | pd.Series, title: str = "") -> Matrix:
        """A matrix from a pandas table (or a series, as a column), named by its index and columns."""
        if isinstance(frame, pd.Series):
            frame = frame.to_frame(frame.name or "value")
        return cls(frame.to_numpy(float), tuple(map(str, frame.index)), tuple(map(str, frame.columns)), title)

    @classmethod
    def identity(cls, names: int | Sequence[str]) -> Matrix:
        """The identity matrix, of size n or with these names for both its rows and columns."""
        names = _numbered(names) if isinstance(names, int) else tuple(names)
        return cls(np.eye(len(names)), names, names, "identity")

    @classmethod
    def zeros(cls, rows: int | Sequence[str], cols: int | Sequence[str]) -> Matrix:
        rows = _numbered(rows) if isinstance(rows, int) else tuple(rows)
        cols = _numbered(cols) if isinstance(cols, int) else tuple(cols)
        return cls(np.zeros((len(rows), len(cols))), rows, cols)

    @classmethod
    def from_dataset(cls, data: Any, names: str | Sequence[str] | None = None, span: Any = None, title: str = "") -> Matrix:
        """Series of a dataset as the columns of a matrix, its rows named by period (``None``: every series; ``span``:
        the active sample when ``None``, as commands use, or ``"all"`` or a span). Missing values stay missing."""
        from .data.axes import TimeAxis

        if len(data.axes) != 1 or not isinstance(data.axes[0], TimeAxis):
            raise SpecificationError("a matrix takes a dataset with one time axis; select an entity first for a panel")
        names = [names] if isinstance(names, str) else (list(data.frame.columns) if names is None else list(names))
        frame = data.to_frame(names, span)
        return cls(frame.to_numpy(float), tuple(str(i) for i in frame.index), tuple(map(str, frame.columns)), title)

    @classmethod
    def coef(cls, result: Any) -> Matrix:
        """A result's estimates as a column vector, its rows named by the parameters."""
        p = _params(result)
        return cls(np.asarray(p.values, float).reshape(-1, 1), tuple(p.names), ("estimate",), "estimates")

    @classmethod
    def cov(cls, result: Any) -> Matrix:
        """A result's covariance matrix of the estimates, rows and columns named by the parameters."""
        p = _params(result)
        if p.cov is None:
            raise SpecificationError(f"{type(result).__name__} has no covariance matrix of its estimates")
        return cls(np.asarray(p.cov, float), tuple(p.names), tuple(p.names), "covariance of the estimates")

    @classmethod
    def var_lag(cls, result: Any, lag: int) -> Matrix:
        """A VAR's coefficient matrix at ``lag`` (1 = the first): rows are the equations, columns the variables."""
        coefs = np.asarray(result.coefs, float)
        if not 1 <= lag <= len(coefs):
            raise SpecificationError(f"the VAR has lags 1 to {len(coefs)}")
        names = tuple(result.variables)
        return cls(coefs[lag - 1], names, tuple(f"{v}[-{lag}]" for v in names), f"lag {lag} coefficients")

    @classmethod
    def var_sigma(cls, result: Any) -> Matrix:
        """A VAR's covariance matrix of its errors."""
        return cls.from_frame(result.sigma, "error covariance")

    @classmethod
    def design(cls, formula: str, data: Any, span: Any = None) -> tuple[Matrix, Matrix]:
        """The regressors X and the dependent variable y of a formula on the data's sample, rows named by period."""
        from .toolkit import design

        d = design(formula, data, span=span)
        rows = tuple(str(i) for i in d.index)
        return (cls(np.asarray(d.X, float), rows, tuple(d.names), "regressors"),
                cls(np.asarray(d.y, float).reshape(-1, 1), rows, (str(formula.split("~")[0].strip()),), "dependent variable"))

    # -- looking at it ------------------------------------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape  # type: ignore[return-value]

    @property
    def T(self) -> Matrix:  # noqa: N802 - the usual name
        return Matrix(self.values.T, self.cols, self.rows, f"{self.title}'" if self.title else "")

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        """The matrix as a pandas table (or one of its window's views by name, as for every result)."""
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return pd.DataFrame(self.values, index=list(self.rows), columns=list(self.cols))

    def to_dataset(self) -> Any:
        """A matrix whose rows are periods (as ``from_dataset`` and ``design`` name them) as a Dataset, a series for
        each column."""
        from .data.dataset import Dataset

        if self.is_complex:
            raise SpecificationError("a dataset holds real numbers; this matrix is complex")
        try:
            periods = [pd.Period(r) for r in self.rows]
        except (ValueError, TypeError):
            periods = []
        freqs = {p.freqstr for p in periods}
        if not periods or len(periods) != len(self.rows) or len(freqs) != 1:
            raise SpecificationError(f"the rows are not periods of one frequency: {', '.join(self.rows[:4])}"
                                     " (rename them with .named(rows=...))")
        index = pd.PeriodIndex(periods, name="time")
        if not index.is_unique:
            raise SpecificationError("the rows name the same period twice")
        return Dataset.from_frame(pd.DataFrame(self.values, index=index, columns=list(self.cols)))

    @property
    def is_complex(self) -> bool:
        return bool(np.iscomplexobj(self.values))

    def table(self, decimals: int | None = None) -> Any:
        from .output import Table

        values = np.round(self.values, decimals) if decimals is not None else self.values
        cells = values.tolist()
        if self.is_complex:  # complex numbers as text, e.g. 0.5+0.866j; real ones stay numbers
            cells = [[v.real if v.imag == 0 else f"{v.real:.6g}{v.imag:+.6g}j" for v in row] for row in cells]
        return Table(list(self.cols), cells, index=list(self.rows), title=self.title or f"{self.shape[0]} x {self.shape[1]} matrix")

    def to_latex(self, names: bool = False, digits: int = 6) -> str:
        """The matrix as LaTeX: a plain ``bmatrix`` (amsmath) of its numbers, or with ``names=True`` a ``tabular``
        with the column names on top and the row names down the left (underscores and the like escaped), for a
        matrix whose names matter. Numbers to ``digits`` significant digits; complex ones as ``a+bi``."""
        def number(v: Any) -> str:
            if isinstance(v, complex):
                if v.imag == 0:
                    return number(v.real)
                return f"{v.real:.{digits}g}{v.imag:+.{digits}g}i"
            return "\\text{NaN}" if np.isnan(v) and not names else ("NaN" if np.isnan(v) else f"{v:.{digits}g}")

        cells = [[number(v) for v in row] for row in self.values.tolist()]
        if not names:
            body = " \\\\\n".join("  " + " & ".join(row) for row in cells)
            return f"\\begin{{bmatrix}}\n{body}\n\\end{{bmatrix}}"
        head = " & ".join(["", *(_latex_text(c) for c in self.cols)])
        body = " \\\\\n".join(" & ".join([_latex_text(r), *row]) for r, row in zip(self.rows, cells))
        return f"\\begin{{tabular}}{{l{'r' * self.shape[1]}}}\n\\hline\n{head} \\\\\n\\hline\n{body} \\\\\n\\hline\n\\end{{tabular}}"

    def __repr__(self) -> str:
        head =f"<Matrix {self.shape[0]} x {self.shape[1]}{': ' + self.title if self.title else ''}>"
        return head + "\n" + self.to_frame().to_string(max_rows=20, max_cols=10)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return self.values if dtype is None else self.values.astype(dtype)

    def _labels(self, key: Any, names: tuple[str, ...], what: str) -> list[int]:
        if isinstance(key, slice):
            return list(range(len(names)))[key]
        keys = [key] if isinstance(key, (str, int, np.integer)) else list(key)
        out = []
        for k in keys:
            if isinstance(k, str):
                if k not in names:
                    raise SpecificationError(f"no {what} named {k!r}; the {what}s are {', '.join(names[:12])}{'…' if len(names) > 12 else ''}")
                out.append(names.index(k))
            else:
                out.append(int(k))
        return out

    def __getitem__(self, key: Any) -> Any:
        """``m["y", "c"]`` is a number; ``m[["a", "b"], :]`` or ``m[:, "c"]`` a smaller matrix (names or positions)."""
        r, c = key if isinstance(key, tuple) else (key, slice(None))
        ri, ci = self._labels(r, self.rows, "row"), self._labels(c, self.cols, "column")
        if isinstance(r, (str, int, np.integer)) and isinstance(c, (str, int, np.integer)):
            return _scalar(self.values[ri[0], ci[0]])
        return Matrix(self.values[np.ix_(ri, ci)], tuple(self.rows[i] for i in ri), tuple(self.cols[i] for i in ci))

    def set(self, rows: Any, cols: Any, value: Any) -> Matrix:
        """A new matrix with an element or a block replaced (``m.set("y", "c", 0.5)``, ``m.set(["a", "b"], :, B)``
        with ``slice(None)`` for every column). A named block must have the names of the rows and columns it goes
        into; a number fills the block."""
        ri, ci = self._labels(rows, self.rows, "row"), self._labels(cols, self.cols, "column")
        values = self.values.copy()
        if isinstance(value, (int, float, complex, np.number)):
            block = np.full((len(ri), len(ci)), value)
        else:
            new = _as_matrix(value)
            if new.shape != (len(ri), len(ci)):
                raise SpecificationError(f"the block is {new.shape[0]} x {new.shape[1]}; the place for it is {len(ri)} x {len(ci)}")
            here = self[ri, ci] if (ri and ci) else Matrix(np.zeros((len(ri), len(ci))))
            here._meet(new, here.rows, new.rows, "the block's rows and the rows it goes into")
            here._meet(new, here.cols, new.cols, "the block's columns and the columns it goes into")
            block = new.values
        if np.iscomplexobj(block) and not np.iscomplexobj(values):
            values = values.astype(complex)
        values[np.ix_(ri, ci)] = block
        return Matrix(values, self.rows, self.cols, self.title)

    # -- assembly -----------------------------------------------------------------------------------------------------

    @classmethod
    def hcat(cls, *parts: Any) -> Matrix:
        """Matrices side by side: their rows must meet, and their columns may not share a name."""
        return _cat([_as_matrix(p) for p in parts], across=True)

    @classmethod
    def vcat(cls, *parts: Any) -> Matrix:
        """Matrices one above the other: their columns must meet, and their rows may not share a name."""
        return _cat([_as_matrix(p) for p in parts], across=False)

    # -- arithmetic ---------------------------------------------------------------------------------------------------

    def _unnamed(self, names: tuple[str, ...]) -> bool:
        return names == _numbered(len(names))

    def _meet(self, other: Matrix, mine: tuple[str, ...], theirs: tuple[str, ...], what: str) -> None:
        if len(mine) != len(theirs):
            raise SpecificationError(f"{what}: {len(mine)} against {len(theirs)}")
        if mine != theirs and not (self._unnamed(mine) or self._unnamed(theirs)):
            raise SpecificationError(f"{what} do not match: {', '.join(mine[:6])} against {', '.join(theirs[:6])}"
                                     " (reorder with m[names, names], or rename with .named(...))")

    def __matmul__(self, other: Any) -> Matrix:
        other = _as_matrix(other)
        self._meet(other, self.cols, other.rows, "the left matrix's columns and the right matrix's rows")
        return Matrix(self.values @ other.values, self.rows, other.cols)

    def __rmatmul__(self, other: Any) -> Matrix:
        return _as_matrix(other).__matmul__(self)

    def _elementwise(self, other: Any, op: Any, symbol: str, reflected: bool = False) -> Matrix:
        if isinstance(other, (int, float, complex, np.number)):
            return Matrix(op(other, self.values) if reflected else op(self.values, other), self.rows, self.cols)
        other = _as_matrix(other)
        if other.shape != self.shape:
            raise SpecificationError(f"{symbol} needs matrices of the same size: {self.shape} against {other.shape}")
        self._meet(other, self.rows, other.rows, "the rows")
        self._meet(other, self.cols, other.cols, "the columns")
        rows = other.rows if self._unnamed(self.rows) else self.rows  # the names, if either has them
        cols = other.cols if self._unnamed(self.cols) else self.cols
        values = op(other.values, self.values) if reflected else op(self.values, other.values)
        return Matrix(values, rows, cols)

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        """NumPy's element-wise functions keep the names: ``np.log(m)``, ``np.exp(m)``, ``np.maximum(m, 0)``, and an
        array on the left of ``+`` or ``@`` gives a named matrix (its names checked as for two matrices). Reductions,
        ``out=`` and functions with several outputs are left to NumPy's arrays (``m.values``)."""
        if method != "__call__" or ufunc.nout != 1 or any(v is not None for v in kwargs.values()):
            return NotImplemented
        if ufunc is np.matmul:
            return _as_matrix(inputs[0]) @ _as_matrix(inputs[1])
        if ufunc.nin == 1:
            return Matrix(ufunc(inputs[0].values), inputs[0].rows, inputs[0].cols)
        if ufunc.nin == 2:
            a, b = inputs
            if isinstance(a, Matrix):
                return a._elementwise(b, ufunc, ufunc.__name__)
            return b._elementwise(a, ufunc, ufunc.__name__, reflected=True)
        return NotImplemented

    def log(self) -> Matrix:
        """Natural logarithms, element by element (``logm`` is the matrix logarithm)."""
        return np.log(self)

    def exp(self) -> Matrix:
        """Exponentials, element by element (``expm`` is the matrix exponential)."""
        return np.exp(self)

    def sqrt(self) -> Matrix:
        """Square roots, element by element (``sqrtm`` is the matrix square root)."""
        return np.sqrt(self)

    def abs(self) -> Matrix:
        return np.abs(self)

    def __abs__(self) -> Matrix:
        return np.abs(self)

    def __add__(self, other: Any) -> Matrix:
        return self._elementwise(other, np.add, "+")

    def __radd__(self, other: Any) -> Matrix:
        return self._elementwise(other, np.add, "+")

    def __sub__(self, other: Any) -> Matrix:
        return self._elementwise(other, np.subtract, "-")

    def __rsub__(self, other: Any) -> Matrix:
        return self._elementwise(other, np.subtract, "-", reflected=True)

    def __mul__(self, other: Any) -> Matrix:
        return self._elementwise(other, np.multiply, "*")

    def __rmul__(self, other: Any) -> Matrix:
        return self._elementwise(other, np.multiply, "*")

    def __truediv__(self, other: Any) -> Matrix:
        return self._elementwise(other, np.divide, "/")

    def __rtruediv__(self, other: Any) -> Matrix:
        return self._elementwise(other, np.divide, "/", reflected=True)

    def __neg__(self) -> Matrix:
        return Matrix(-self.values, self.rows, self.cols)

    def __pow__(self, power: Any) -> Matrix:
        """Element by element, as NumPy's ``**`` (``A.power(n)`` is the matrix power). For now a square matrix to a
        whole power still gives the matrix power, with a DeprecationWarning: it will become element by element."""
        if float(power).is_integer() and self.shape[0] == self.shape[1] and self.shape[0] > 1:
            warnings.warn("A ** n on a square matrix is the matrix power for now, but is deprecated: it will become element "
                          "by element, as in NumPy. Use A.power(n) for the matrix power, or A * A or np.square(A) element "
                          "by element", DeprecationWarning, stacklevel=2)
            return self.power(int(power))
        return Matrix(self.values ** power, self.rows, self.cols)

    def power(self, n: int) -> Matrix:
        """The matrix power of a square matrix: ``A.power(2)`` is ``A @ A``, ``A.power(0)`` the identity and
        ``A.power(-2)`` is ``A.inv().power(2)`` (``numpy.linalg.matrix_power``). A negative power is named as the
        inverse is (rows by the columns, columns by the rows)."""
        self._square("the matrix power")
        if not float(n).is_integer():
            raise SpecificationError(f"the matrix power takes a whole number, not {n}; ** is element by element")
        n = int(n)
        try:
            values = np.linalg.matrix_power(self.values, n)
        except np.linalg.LinAlgError:
            raise SpecificationError("the matrix is singular: it has no negative powers") from None
        return Matrix(values, *((self.cols, self.rows) if n < 0 else (self.rows, self.cols)))

    matrix_power = power

    def named(self, rows: Sequence[str] | None = None, cols: Sequence[str] | None = None) -> Matrix:
        """The same numbers with other names."""
        return Matrix(self.values, tuple(rows) if rows is not None else self.rows, tuple(cols) if cols is not None else self.cols, self.title)

    # -- functions ----------------------------------------------------------------------------------------------------

    def _square(self, what: str) -> None:
        if self.shape[0] != self.shape[1]:
            raise SpecificationError(f"{what} needs a square matrix; this one is {self.shape[0]} x {self.shape[1]}")

    def inv(self) -> Matrix:
        """The inverse: its rows are named by the columns, and its columns by the rows."""
        self._square("the inverse")
        try:
            values = np.linalg.inv(self.values)
        except np.linalg.LinAlgError:
            raise SpecificationError("the matrix is singular: it has no inverse (pinv gives the pseudo-inverse)") from None
        return Matrix(values, self.cols, self.rows)

    def pinv(self) -> Matrix:
        return Matrix(np.linalg.pinv(self.values), self.cols, self.rows)

    def solve(self, b: Any) -> Matrix:
        """x with ``self @ x = b``, more accurate than ``self.inv() @ b``."""
        b = _as_matrix(b)
        self._square("solve")
        self._meet(b, self.rows, b.rows, "the matrix's rows and the right-hand side's rows")
        return Matrix(np.linalg.solve(self.values, b.values), self.cols, b.cols)

    def det(self) -> float | complex:
        self._square("the determinant")
        return _scalar(np.linalg.det(self.values))

    def trace(self) -> float | complex:
        self._square("the trace")
        return _scalar(np.trace(self.values))

    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self.values))

    def chol(self, upper: bool = False) -> Matrix:
        """The lower triangular L with ``L @ L.T`` equal to the matrix (symmetric and positive definite), or with
        ``upper=True`` the upper triangular R with ``R.T @ R`` equal to it (MATLAB's ``chol``). Named as the matrix."""
        self._square("the Cholesky factor")
        try:
            L = np.linalg.cholesky(self.values)
            return Matrix(L.conj().T if upper else L, self.rows, self.cols)
        except np.linalg.LinAlgError:
            raise SpecificationError("the matrix is not positive definite: it has no Cholesky factor") from None

    def eig(self, symmetric: bool | None = None) -> Eig:
        """Eigenvalues (a column ``eigenvalue``, rows λ1, λ2, ..., largest first in size) and eigenvectors (columns
        λ1, λ2, ..., rows named as the matrix's rows), so that ``A @ v = λ v``. ``symmetric=None`` (the default)
        uses ``numpy.linalg.eigh`` for a symmetric matrix (real values, orthonormal vectors) and
        ``numpy.linalg.eig`` otherwise; ``True`` insists on a symmetric matrix, ``False`` uses the general method.
        The general method's results are real when every imaginary part is below 1e-12 (relative to the largest
        eigenvalue) and complex otherwise; a conjugate pair keeps NumPy's order."""
        self._square("eigenvalues")
        is_symmetric = bool(np.allclose(self.values, self.values.conj().T))
        if symmetric and not is_symmetric:
            raise SpecificationError("symmetric=True needs a symmetric matrix; this one is not (symmetric=False for the general method)")
        if symmetric or (symmetric is None and is_symmetric):
            values, vectors = np.linalg.eigh(self.values)
        else:
            values, vectors = np.linalg.eig(self.values)
            if np.iscomplexobj(values) and _real_if_close(values).dtype.kind == "f":
                values, vectors = values.real, vectors.real
        order = np.argsort(-np.abs(values), kind="stable")
        names = _seq("λ", len(values))
        return Eig(Matrix(values[order].reshape(-1, 1), names, ("eigenvalue",)), Matrix(vectors[:, order], self.rows, names))

    def eigh(self) -> Eig:
        """The eigenvalues and orthonormal eigenvectors of a symmetric matrix (``eig(symmetric=True)``)."""
        return self.eig(symmetric=True)

    def svd(self) -> SVD:
        """The thin singular value decomposition ``A = U diag(s) V'``: ``U`` (rows: A's rows), ``s`` (a column,
        largest first) and ``V`` (rows: A's columns), all named s1, s2, ... for the k = min(m, n) singular values."""
        U, s, Vt = np.linalg.svd(self.values, full_matrices=False)
        names = _seq("s", len(s))
        return SVD(Matrix(U, self.rows, names), Matrix(s.reshape(-1, 1), names, ("singular value",)), Matrix(Vt.conj().T, self.cols, names))

    def qr(self) -> QR:
        """The thin QR decomposition ``A = Q R``: Q's rows are A's rows, R's columns A's columns, both named q1, q2,
        ... between."""
        Q, R = np.linalg.qr(self.values, mode="reduced")
        names = _seq("q", Q.shape[1])
        return QR(Matrix(Q, self.rows, names), Matrix(R, names, self.cols))

    def lu(self) -> LU:
        """``A = P L U`` with partial pivoting (``scipy.linalg.lu``). P's rows are A's rows and its columns the same
        names in pivot order, which name L's rows; L's columns and U's rows are l1, l2, ...; U's columns are A's."""
        from scipy import linalg

        P, L, U = linalg.lu(self.values)
        pivot = tuple(self.rows[i] for i in np.argmax(P, axis=0))  # the row of A that P puts in each place
        names = _seq("l", L.shape[1])
        return LU(Matrix(P, self.rows, pivot), Matrix(L, pivot, names), Matrix(U, names, self.cols))

    def schur(self, complex: bool = False) -> Schur:  # noqa: A002 - scipy's word
        """The Schur decomposition ``A = Q T Q'`` (``scipy.linalg.schur``): real, with 2 x 2 blocks on T's diagonal
        for complex pairs, or ``complex=True`` for triangular T. Q's rows are A's rows; T and Q's columns are u1, u2,
        ... (the Schur vectors)."""
        from scipy import linalg

        self._square("the Schur decomposition")
        T, Q = linalg.schur(self.values, output="complex" if complex else "real")
        names = _seq("u", T.shape[0])
        return Schur(Matrix(T, names, names), Matrix(Q, self.rows, names))

    def qz(self, B: Any) -> QZ:
        """The generalized Schur (QZ) decomposition of the pencil (A, B) (``scipy.linalg.qz``, real):
        ``A = Q AA Z'`` and ``B = Q BB Z'``. B must have A's names (or none); Q's rows are their rows, Z's rows
        their columns, and AA and BB have rows q1, ... and columns z1, ...."""
        from scipy import linalg

        B = _as_matrix(B)
        self._square("the QZ decomposition")
        if B.shape != self.shape:
            raise SpecificationError(f"the QZ decomposition needs two matrices of the same size: {self.shape} against {B.shape}")
        self._meet(B, self.rows, B.rows, "the two matrices' rows")
        self._meet(B, self.cols, B.cols, "the two matrices' columns")
        rows = B.rows if self._unnamed(self.rows) else self.rows
        cols = B.cols if self._unnamed(self.cols) else self.cols
        AA, BB, Q, Z = linalg.qz(self.values, B.values, output="complex" if (self.is_complex or B.is_complex) else "real")
        q, z = _seq("q", self.shape[0]), _seq("z", self.shape[0])
        return QZ(Matrix(AA, q, z), Matrix(BB, q, z), Matrix(Q, rows, q), Matrix(Z, cols, z))

    # -- functions of a matrix ----------------------------------------------------------------------------------------

    def _function(self, name: str, fn: Any) -> Matrix:
        self._square(f"the matrix {name}")
        return Matrix(_real_if_close(np.asarray(fn(self.values))), self.rows, self.cols)

    def expm(self) -> Matrix:
        """The matrix exponential (``scipy.linalg.expm``), named as the matrix (as ``A.power(2)`` is)."""
        from scipy import linalg

        return self._function("exponential", linalg.expm)

    def logm(self) -> Matrix:
        """The principal matrix logarithm (``scipy.linalg.logm``): real when it has a real value, complex otherwise."""
        from scipy import linalg

        return self._function("logarithm", lambda a: linalg.logm(a))

    def sqrtm(self) -> Matrix:
        """The principal square root S with ``S @ S = A`` (``scipy.linalg.sqrtm``): real when it has a real value."""
        from scipy import linalg

        return self._function("square root", linalg.sqrtm)

    def norm(self, kind: Any = "fro") -> float:
        """A matrix norm: ``"fro"`` (Frobenius, the default), ``"nuc"`` (nuclear), ``1`` and ``"inf"`` (the largest
        column and row sums of absolute values), ``2`` (the largest singular value), as ``numpy.linalg.norm``."""
        kinds = {"fro": "fro", "nuc": "nuc", "inf": np.inf, "-inf": -np.inf, 1: 1, -1: -1, 2: 2, -2: -2, np.inf: np.inf, -np.inf: -np.inf}
        if kind not in kinds:
            raise SpecificationError(f"no norm {kind!r}: fro, nuc, 1, 2, inf, -1, -2 or -inf")
        return float(np.linalg.norm(self.values, kinds[kind]))

    def cond(self, kind: Any = 2) -> float:
        """The condition number in a norm (the 2-norm by default: largest over smallest singular value)."""
        kinds = {"fro": "fro", "nuc": "nuc", "inf": np.inf, 1: 1, -1: -1, 2: 2, -2: -2, np.inf: np.inf}
        if kind not in kinds:
            raise SpecificationError(f"no norm {kind!r}: fro, nuc, 1, 2, inf, -1 or -2")
        return float(np.linalg.cond(self.values, kinds[kind]))

    def null(self, rcond: float | None = None) -> Matrix:
        """An orthonormal basis of the null space (x with ``A @ x = 0``): rows named by A's columns, columns null1,
        null2, .... ``rcond``: singular values below it times the largest count as zero (default as SciPy's)."""
        from scipy import linalg

        N = linalg.null_space(self.values, rcond=rcond) if rcond is not None else linalg.null_space(self.values)
        return Matrix(N, self.cols, _seq("null", N.shape[1]))

    def orth(self, rcond: float | None = None) -> Matrix:
        """An orthonormal basis of the column space: rows named by A's rows, columns orth1, orth2, ...."""
        from scipy import linalg

        O = linalg.orth(self.values, rcond=rcond) if rcond is not None else linalg.orth(self.values)  # noqa: E741
        return Matrix(O, self.rows, _seq("orth", O.shape[1]))

    def perp(self, rcond: float | None = None) -> Matrix:
        """The orthogonal complement A⊥ of an m x r matrix of rank r: m x (m - r), with ``A⊥' A = 0`` and
        ``[A A⊥]`` of full rank m, as for a VECM's alpha-perp and beta-perp. Normalised to orthonormal columns
        (``A⊥' A⊥ = I``), the left null space from the SVD, as MATLAB's ``null(A')``; Johansen (1995) defines A⊥ only
        up to an invertible r x r factor, which cancels in the expressions it enters (such as the long-run impact
        ``β⊥ (α⊥' Γ β⊥)^-1 α⊥'``). Rows are A's rows; columns perp1, perp2, ...."""
        N = self.T.null(rcond).values
        return Matrix(N, self.rows, _seq("perp", N.shape[1]))

    # -- sums and products --------------------------------------------------------------------------------------------

    def _reduce(self, fn: Any, label: str, axis: int | None, dtype: Any, out: Any) -> Any:
        if out is not None or dtype is not None:
            raise SpecificationError(f"{label} of a matrix takes no out= or dtype=; use m.values for NumPy's options")
        if axis is None:
            return _scalar(fn(self.values))
        if axis in (0, -2):  # down the rows: one row, the columns kept
            return Matrix(fn(self.values, axis=0).reshape(1, -1), (label,), self.cols)
        if axis in (1, -1):  # across the columns: one column, the rows kept
            return Matrix(fn(self.values, axis=1).reshape(-1, 1), self.rows, (label,))
        raise SpecificationError("axis is None (everything), 0 (each column, down the rows) or 1 (each row)")

    def sum(self, axis: int | None = None, dtype: Any = None, out: Any = None) -> Any:
        """The sum of everything (a number), of each column (``axis=0``: a row named ``sum``) or of each row
        (``axis=1``: a column named ``sum``)."""
        return self._reduce(np.sum, "sum", axis, dtype, out)

    def mean(self, axis: int | None = None, dtype: Any = None, out: Any = None) -> Any:
        """The mean of everything, of each column (``axis=0``) or of each row (``axis=1``), named ``mean``."""
        return self._reduce(np.mean, "mean", axis, dtype, out)

    def _vector(self, what: str) -> tuple[np.ndarray, tuple[str, ...]]:
        if self.shape[1] == 1:
            return self.values[:, 0], self.rows
        if self.shape[0] == 1:
            return self.values[0, :], self.cols
        raise SpecificationError(f"{what} takes vectors (one column or one row); this matrix is {self.shape[0]} x {self.shape[1]}")

    def outer(self, other: Any) -> Matrix:
        """The outer product ``a b'`` of two vectors: rows named by a's names, columns by b's."""
        a, ra = self._vector("the outer product")
        b, rb = _as_matrix(other)._vector("the outer product")
        return Matrix(np.outer(a, b), ra, rb)

    def inner(self, other: Any) -> float | complex:
        """The inner product ``a' b`` of two vectors with the same names, or the sum of the element-by-element
        products (``trace(A' B)``) of two matrices with the same names. Not conjugated."""
        other = _as_matrix(other)
        if 1 in self.shape and 1 in other.shape:
            a, ra = self._vector("the inner product")
            b, rb = other._vector("the inner product")
            self._meet(other, ra, rb, "the two vectors' names")
            return _scalar(a @ b)
        return _scalar(np.sum(self._elementwise(other, np.multiply, "the inner product").values))

    # -- vech and the matrices of matrix calculus ---------------------------------------------------------------------

    def vech(self) -> Matrix:
        """The lower triangle (with the diagonal) of a square matrix, column by column, into one column; each row
        named "row,column"."""
        self._square("vech")
        n = self.shape[0]
        idx = [(i, j) for j in range(n) for i in range(j, n)]
        return Matrix(np.array([self.values[i, j] for i, j in idx]).reshape(-1, 1),
                      tuple(f"{self.rows[i]},{self.cols[j]}" for i, j in idx), ("vech",))

    @classmethod
    def unvech(cls, v: Any, names: Sequence[str] | None = None) -> Matrix:
        """The symmetric matrix whose ``vech`` is v (n(n+1)/2 numbers). Named by ``names``, or by the names in v's
        rows when v came from ``vech`` of a matrix with the same row and column names, or numbered."""
        x, labels = _as_matrix(v)._vector("unvech")
        n = int(round((np.sqrt(8 * len(x) + 1) - 1) / 2))
        if n * (n + 1) // 2 != len(x):
            raise SpecificationError(f"vech of an n x n matrix has n(n+1)/2 numbers; {len(x)} is not one of those")
        names = tuple(names) if names is not None else _vech_names(labels, n)
        if len(names) != n:
            raise SpecificationError(f"{len(names)} names for a {n} x {n} matrix")
        out = np.zeros((n, n), dtype=x.dtype)
        k = 0
        for j in range(n):
            for i in range(j, n):
                out[i, j] = out[j, i] = x[k]
                k += 1
        return cls(out, names, names)

    @classmethod
    def commutation(cls, m: int | Sequence[str], n: int | Sequence[str]) -> Matrix:
        """The commutation matrix K (mn x mn) with ``K @ A.vec() = A.T.vec()`` for an m x n matrix A. Given names
        (A's rows and columns) instead of sizes, K is named so that the product's names meet: columns as
        ``A.vec()``'s rows, rows as ``A.T.vec()``'s."""
        rows, cols = _names_of(m), _names_of(n)
        p, q = len(rows), len(cols)
        K = np.zeros((p * q, p * q))
        for i in range(p):
            for j in range(q):
                K[j + i * q, i + j * p] = 1.0
        if isinstance(m, (int, np.integer)) and isinstance(n, (int, np.integer)):
            return cls(K, title="commutation")
        return cls(K, tuple(f"{c},{r}" for r in rows for c in cols), tuple(f"{r},{c}" for c in cols for r in rows), "commutation")

    @classmethod
    def duplication(cls, n: int | Sequence[str]) -> Matrix:
        """The duplication matrix D (n² x n(n+1)/2) with ``D @ A.vech() = A.vec()`` for a symmetric n x n A. Given
        A's names instead of n, its columns are named as ``vech``'s rows and its rows as ``vec``'s."""
        names = _names_of(n)
        k = len(names)
        D = np.zeros((k * k, k * (k + 1) // 2))
        col = 0
        for j in range(k):
            for i in range(j, k):
                D[i + j * k, col] = D[j + i * k, col] = 1.0
                col += 1
        if isinstance(n, (int, np.integer)):
            return cls(D, title="duplication")
        return cls(D, tuple(f"{r},{c}" for c in names for r in names),
                   tuple(f"{names[i]},{names[j]}" for j in range(k) for i in range(j, k)), "duplication")

    @classmethod
    def elimination(cls, n: int | Sequence[str]) -> Matrix:
        """The elimination matrix L (n(n+1)/2 x n²) with ``L @ A.vec() = A.vech()`` for any n x n A; named, given
        names, as the transpose of ``duplication``."""
        k = len(_names_of(n))
        L = np.zeros((k * (k + 1) // 2, k * k))
        row = 0
        for j in range(k):
            for i in range(j, k):
                L[row, i + j * k] = 1.0
                row += 1
        D = cls.duplication(n)
        return cls(L, D.cols, D.rows, "elimination")

    # -- matrix equations ---------------------------------------------------------------------------------------------

    @classmethod
    def lyapunov(cls, A: Any, Q: Any, continuous: bool = False) -> Matrix:
        """X with ``X = A X A' + Q`` (discrete, the default; e.g. the unconditional variance of ``y = A y[-1] + e``
        with ``var(e) = Q``, by ``scipy.linalg.solve_discrete_lyapunov``), or with ``A X + X A' + Q = 0``
        (``continuous=True``). X is named as Q. A's rows must meet Q's; its columns are taken to be the same
        variables (a VAR's lag matrix has columns such as ``y[-1]``). Explosive A is not refused: X then solves the
        equation but is no variance."""
        from scipy import linalg

        A, Q = _as_matrix(A), _as_matrix(Q)
        A._square("the Lyapunov equation's A")
        Q._square("the Lyapunov equation's Q")
        A._meet(Q, A.rows, Q.rows, "A's rows and Q's rows")
        try:
            X = linalg.solve_continuous_lyapunov(A.values, -Q.values) if continuous else linalg.solve_discrete_lyapunov(A.values, Q.values)
        except (np.linalg.LinAlgError, ValueError) as exc:
            raise SpecificationError(f"the Lyapunov equation has no unique solution ({exc})") from None
        if not np.all(np.isfinite(X)):
            raise SpecificationError("the Lyapunov equation has no unique solution")
        return cls(_real_if_close(X), Q.rows, Q.cols, "Lyapunov solution")

    @classmethod
    def sylvester(cls, A: Any, B: Any, C: Any) -> Matrix:
        """X with ``A X + X B = C`` (``scipy.linalg.solve_sylvester``), named as C. C's rows must meet A's rows and
        C's columns B's columns; A's columns and B's rows are taken to be the same names (A X and X B add up)."""
        from scipy import linalg

        A, B, C = _as_matrix(A), _as_matrix(B), _as_matrix(C)
        A._square("the Sylvester equation's A")
        B._square("the Sylvester equation's B")
        C._meet(A, C.rows, A.rows, "C's rows and A's rows")
        C._meet(B, C.cols, B.cols, "C's columns and B's columns")
        X = linalg.solve_sylvester(A.values, B.values, C.values)
        if not np.all(np.isfinite(X)):
            raise SpecificationError("the Sylvester equation has no unique solution (A and -B share an eigenvalue)")
        return cls(_real_if_close(X), C.rows, C.cols, "Sylvester solution")

    @classmethod
    def riccati(cls, A: Any, B: Any, Q: Any, R: Any) -> Matrix:
        """The stabilising X of the discrete algebraic Riccati equation
        ``X = A' X A - A' X B (R + B' X B)^-1 B' X A + Q`` (``scipy.linalg.solve_discrete_are``): the cost-to-go of
        the linear-quadratic regulator, or, with ``A = T'``, ``B = Z'``, ``Q = R Q R'`` and ``R = H``, the
        steady-state prediction covariance of the Kalman filter. A (n x n), B (n x k), Q (n x n), R (k x k); B's
        rows must meet A's rows, Q's A's, and R's B's columns. X is named as Q."""
        from scipy import linalg

        A, B, Q, R = (_as_matrix(m) for m in (A, B, Q, R))
        for m, what in ((A, "A"), (Q, "Q"), (R, "R")):
            m._square(f"the Riccati equation's {what}")
        A._meet(B, A.rows, B.rows, "A's rows and B's rows")
        A._meet(Q, A.rows, Q.rows, "A's rows and Q's rows")
        B._meet(R, B.cols, R.rows, "B's columns and R's rows")
        try:
            X = linalg.solve_discrete_are(A.values, B.values, Q.values, R.values)
        except (np.linalg.LinAlgError, ValueError) as exc:
            raise SpecificationError(f"the Riccati equation has no stabilising solution ({exc})") from None
        return cls(_real_if_close(X), Q.rows, Q.cols, "Riccati solution")

    def kron(self, other: Any) -> Matrix:
        """The Kronecker product: rows and columns named "a:b" from the two matrices' names."""
        other = _as_matrix(other)
        return Matrix(np.kron(self.values, other.values), tuple(f"{a}:{b}" for a in self.rows for b in other.rows),
                      tuple(f"{a}:{b}" for a in self.cols for b in other.cols))

    def vec(self) -> Matrix:
        """The columns stacked into one column, each row named "row,column"."""
        return Matrix(self.values.reshape(-1, order="F").reshape(-1, 1), tuple(f"{r},{c}" for c in self.cols for r in self.rows), ("vec",))

    def diag(self) -> Matrix:
        """A square matrix's diagonal as a column; a column's values on the diagonal of a square matrix."""
        if self.shape[1] == 1 and self.shape[0] > 1:
            return Matrix(np.diag(self.values[:, 0]), self.rows, self.rows)
        self._square("the diagonal")
        return Matrix(np.diag(self.values).reshape(-1, 1), self.rows, ("diagonal",))


def _as_matrix(value: Any) -> Matrix:
    if isinstance(value, Matrix):
        return value
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return Matrix.from_frame(value)
    return Matrix(np.asarray(value, float))


def _cat(parts: list[Matrix], across: bool) -> Matrix:
    """Side by side (``across``) or one above the other: the shared dimension must meet, the other may not clash."""
    if not parts:
        raise SpecificationError("nothing to put together")
    shared = [p.rows if across else p.cols for p in parts]
    first = parts[0]
    names = next((s for s in shared if not first._unnamed(s)), shared[0])
    for part, s in zip(parts, shared):
        first._meet(part, names, s, "the matrices' rows" if across else "the matrices' columns")
    others = [p.cols if across else p.rows for p in parts]
    if all(first._unnamed(o) for o in others):
        joined = _numbered(sum(len(o) for o in others))
    else:
        joined = tuple(n for o in others for n in o)
        seen: set[str] = set()
        for n in joined:
            if n in seen:
                what = "column" if across else "row"
                raise SpecificationError(f"the {what} names clash: {n!r} is in two of the matrices (rename with .named(...))")
            seen.add(n)
    values = np.hstack([p.values for p in parts]) if across else np.vstack([p.values for p in parts])
    return Matrix(values, names, joined) if across else Matrix(values, joined, names)


_LATEX = {"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "_": "\\_", "{": "\\{", "}": "\\}",
          "~": "\\textasciitilde{}", "^": "\\textasciicircum{}"}


def _latex_text(text: str) -> str:
    return "".join(_LATEX.get(ch, ch) for ch in str(text))


def _vech_names(labels: tuple[str, ...], n: int) -> tuple[str, ...]:
    """The names of the matrix whose ``vech`` has these row names ("row,column"), if they are of that form."""
    first = labels[0]
    half = (len(first) - 1) // 2
    base = first[:half]
    if len(first) == 2 * half + 1 and first == f"{base},{base}":
        names = tuple(lab[: -(len(base) + 1)] for lab in labels[:n])
        if all(lab.endswith("," + base) for lab in labels[:n]) and len(set(names)) == n:
            expected = tuple(f"{names[i]},{names[j]}" for j in range(n) for i in range(j, n))
            if expected == labels:
                return names
    return _numbered(n)


def _params(result: Any) -> Any:
    params = getattr(result, "params", None)
    if params is None or not hasattr(params, "names"):
        raise SpecificationError(f"{type(result).__name__} has no named estimates")
    return params
