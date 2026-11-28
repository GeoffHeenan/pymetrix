"""Least squares without a formula: ``pymetrix.ols(y, X)`` on a dependent series and a table of regressors.

    fit = pymetrix.ols(y, X)                              # y a Series, X a DataFrame (or numpy arrays)
    fit = pymetrix.ols(y, X, cov="hac", hac_lags=4)       # the standard errors Equation.fit(method=OLS(...)) offers
    fit = pymetrix.ols(y, X, cov=Cluster("firm"), clusters=firms)

It is for code that runs many small regressions (loops, simulations, model building), where writing each as a formula
and packing the data into a ``Dataset`` costs more than the regression. The result is the ``FittedEquation`` that
``Equation("y ~ x1 + x2").fit(data, method=OLS(...))`` returns on the same data, with the same numbers (the two paths
share the least-squares code), views, tests and provenance; its equation is that formula, written from the names.

Provenance is as complete as for a formula fit: the fingerprint of the data used, the rows used and the reason each
other row was not (missing values), the method and backend. ``vintage_request`` is ``{"policy": "direct"}``: the data
were passed directly to the call, not taken from a dataset. In a session the call is recorded as Python code; the
arrays are not saved with the project unless they are themselves objects of the session (or the project's data).
"""

from __future__ import annotations

import functools
import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from .._api import api
from ..errors import DataError, SpecificationError
from .covariance import Cluster
from .design import LinearDesign

__all__ = ["ols"]

_NAME = re.compile(r"^[^\W\d]\w*$")


@api("pymetrix:ols")
def ols(
    y: Any,
    X: Any,
    cov: str | Cluster = "classic",
    constant: bool = True,
    weights: Any = None,
    clusters: Any = None,
    hac_lags: int | None = None,
    hac_correction: bool = True,
    missing: Literal["drop", "raise"] = "drop",
    names: Sequence[str] | None = None,
) -> Any:
    """Least squares of ``y`` on the columns of ``X``, without a formula or a ``Dataset``: for code running many
    regressions. Returns the ``FittedEquation`` that ``Equation("y ~ x1 + x2").fit(data, method=OLS(cov=...))``
    gives on the same data, with the same numbers, views, tests and provenance.

    ``y`` is a Series (or a 1-d array, named ``y``); ``X`` a DataFrame, a Series, a mapping of names to columns, or a
    2-d array (columns named by ``names``, else ``x1``, ``x2``...). Pandas inputs must share their index; a
    ``PeriodIndex`` (or dates with a frequency) makes the data a time series, as in a dataset, and rows are put in
    index order. ``constant`` adds the intercept ``const``; ``weights`` is a series of weights (weighted least
    squares, as ``fit(weights=...)``); ``missing`` drops (or refuses) rows with a missing value.

    ``cov`` is as for ``OLS``: ``"classic"``, ``"white"`` (HC1), ``"hc0"``, ``"hc2"``, ``"hc3"``, ``"hac"`` (Newey-West, time series; ``hac_lags``
    and ``hac_correction`` as ``OLS``), or ``Cluster("a")`` / ``Cluster("a", "b")`` (one- or two-way), the names
    being columns of ``clusters`` (a Series or DataFrame of group labels) or levels of the index; ``cov="cluster"``
    clusters by every column of ``clusters``.

    Provenance records ``vintage_request={"policy": "direct"}``: the data were passed directly. In a session the
    call is recorded as code; the arrays are saved with a project only if you save them there."""
    from .._capture import capture
    from .methods import OLS, _ols_on_design

    cov = _cluster_spec(cov, clusters)
    if not isinstance(cov, Cluster) and cov not in ("classic", "white", "hc0", "hc2", "hc3", "hac"):
        raise SpecificationError(f"unknown covariance {cov!r}; use 'classic', 'white', 'hc0', 'hc2', 'hc3', 'hac', 'cluster' or Cluster(...)")
    method = OLS(cov=cov, hac_lags=hac_lags, hac_correction=hac_correction)
    if missing not in ("drop", "raise"):
        raise SpecificationError("missing must be 'drop' or 'raise'")
    raw = _Inputs(y, X, names, weights, clusters)
    if cov == "hac" and raw.kind != "time_series":
        raise SpecificationError("HAC covariance needs a single time series (give y a PeriodIndex); use Cluster(...) for panels")
    eq = _equation(raw.y_name, tuple(raw.x_names), bool(constant))
    Xall = np.column_stack([np.ones(len(raw.y)), raw.X]) if constant else raw.X
    keep = np.isfinite(raw.y) & np.isfinite(Xall).all(axis=1)
    if raw.w is not None:
        keep &= np.isfinite(raw.w)
    dropped = int(len(keep) - keep.sum())
    if dropped and missing == "raise":
        raise SpecificationError(f"{dropped} observations in the sample have missing values; use missing='drop'")
    if not keep.any():
        raise SpecificationError("no observations left in the sample")
    full = bool(keep.all())
    index = raw.index if full else raw.index[keep]
    y_used = raw.y if full else raw.y[keep]
    design = LinearDesign(y_used, Xall if full else Xall[keep], list(eq.coefficient_names()), index,
                          None if raw.w is None else (raw.w if full else raw.w[keep]), None, [], dropped, 0, y_used)
    groups = lambda: raw.groups(cov, keep)  # noqa: E731
    with capture() as caught:
        fit = _ols_on_design(method, eq, design, groups, raw.kind == "time_series", lambda: None, raw.w_name)
    object.__setattr__(fit, "provenance", _provenance(eq, method, raw, keep, dropped, fit, caught))
    return fit


def _cluster_spec(cov: Any, clusters: Any) -> Any:
    if isinstance(cov, str) and cov == "cluster":
        if clusters is None:
            raise SpecificationError("cov='cluster' needs clusters= (a Series or DataFrame of group labels)")
        names = [clusters.name] if isinstance(clusters, pd.Series) else list(getattr(clusters, "columns", []))
        if not names or any(n is None for n in names):
            raise SpecificationError("clusters= must be a named Series or a DataFrame, so its groups can be named")
        return Cluster(*map(str, names))
    if clusters is not None and not isinstance(cov, Cluster):
        raise SpecificationError(f"clusters= is for clustered standard errors; give cov=Cluster(...) or cov='cluster' (cov is {cov!r})")
    return cov


@functools.lru_cache(maxsize=512)
def _equation(y_name: str, x_names: tuple[str, ...], constant: bool) -> Any:
    """The linear-form equation of the names (``y ~ x1 + x2``, ``- 1`` without a constant), checked to be exactly
    those series: the same equation a formula fit would have. Equations are immutable, so one is made per set of names."""
    from ..expr import ExpressionError, Name
    from .equation import Equation

    if constant and "const" in x_names:
        raise SpecificationError("X has a column named 'const': give constant=False to use it as the intercept, or rename it")
    if y_name in x_names:
        raise SpecificationError(f"{y_name!r} is both the dependent variable and a regressor")
    if len(set(x_names)) != len(x_names):
        raise SpecificationError(f"repeated regressor names in {list(x_names)}")
    for name in (y_name, *x_names):
        if not _NAME.match(name):
            raise SpecificationError(f"{name!r} is not a series name pymetrix can use (letters, digits and _, not starting "
                                     "with a digit); rename it, e.g. X.rename(columns={...})")
    rhs = " + ".join(x_names) if x_names else ("1" if constant else "")
    if not rhs:
        raise SpecificationError("the equation has no regressors")
    text = f"{y_name} ~ {rhs}" + ("" if constant or not x_names else " - 1")
    try:
        eq = Equation(text)
    except ExpressionError as e:
        raise SpecificationError(f"the names {[y_name, *x_names]} cannot be written as an equation ({e}); rename the series") from None
    plain = all(isinstance(t.expr, Name) and t.expr.lag == 0 and t.expr.id == t.label for t in eq.terms)
    if not plain or [t.label for t in eq.terms] != list(x_names) or str(eq.lhs) != y_name or eq.intercept != constant:
        raise SpecificationError(f"the names {[y_name, *x_names]} clash with pymetrix's functions or keywords; rename the series")
    return eq


class _Inputs:
    """The arrays of a direct call, in index order: y, X, weights and cluster groups, with their names and index."""

    def __init__(self, y: Any, X: Any, names: Sequence[str] | None, weights: Any, clusters: Any) -> None:
        index = None
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise DataError("y must be one series")
            y = y.iloc[:, 0]
        if isinstance(y, pd.Series):
            index = y.index
            self.y_name = "y" if y.name is None else str(y.name)
            yv = y.to_numpy(dtype=float, na_value=np.nan)
        else:
            self.y_name = "y"
            yv = np.asarray(y, dtype=float)
            if yv.ndim == 2 and yv.shape[1] == 1:
                yv = yv[:, 0]
        if yv.ndim != 1:
            raise DataError("y must be one series (a Series or a 1-d array)")
        n = len(yv)
        self.x_names, Xv, x_index = _regressors(X, names, n)
        if index is None:
            index = x_index
        elif x_index is not None and not index.equals(x_index):
            raise DataError("y and X must have the same index; align them first (e.g. X = X.loc[y.index])")
        if index is None:
            index = pd.RangeIndex(n)
        self.w_name, self.w = None, None
        if weights is not None:
            self.w_name = str(getattr(weights, "name", None) or "weights")
            self.w = _column(weights, index, n, "weights")
        self.clusters = None if clusters is None else _cluster_frame(clusters, index, n)
        index = _time_index(index)
        if not index.is_unique:
            raise DataError("each observation must appear once in the index")
        if not index.is_monotonic_increasing:  # rows in index order, as a dataset keeps them
            index, order = index.sort_values(return_indexer=True)
            yv, Xv = yv[order], Xv[order]
            self.w = None if self.w is None else self.w[order]
            self.clusters = None if self.clusters is None else self.clusters.iloc[order]
        self.y, self.X, self.index = yv, Xv, index
        self.kind, self.time = _kind(index)

    def groups(self, cov: Any, keep: np.ndarray) -> list[np.ndarray]:
        if not isinstance(cov, Cluster):
            return []
        out = []
        for level in cov.levels:
            if self.clusters is not None and level in self.clusters.columns:
                values = self.clusters[level].to_numpy()
            elif level in [n for n in self.index.names if n is not None]:
                values = self.index.get_level_values(level).to_numpy()
            else:
                raise SpecificationError(f"cluster {level!r} is neither a column of clusters= nor a level of the index")
            out.append(values[keep])
        return out

    def raw(self) -> dict[str, np.ndarray]:
        """Each variable's values as given (before missing rows were dropped)."""
        out = {self.y_name: self.y}
        out.update({name: self.X[:, j] for j, name in enumerate(self.x_names)})
        return out


def _regressors(X: Any, names: Sequence[str] | None, n: int) -> tuple[list[str], np.ndarray, pd.Index | None]:
    index = None
    if isinstance(X, pd.Series):
        X = X.to_frame("x1" if X.name is None else X.name)
    if isinstance(X, pd.DataFrame):
        if names is not None:
            raise SpecificationError("names= is for arrays; a DataFrame's columns are its names")
        index = X.index
        x_names = [str(c) for c in X.columns]
        values = X.to_numpy(dtype=float, na_value=np.nan) if X.shape[1] else np.empty((len(X), 0))
    elif isinstance(X, Mapping):
        x_names = [str(k) for k in X]
        cols = [np.asarray(v, dtype=float) for v in X.values()]
        if any(c.ndim != 1 for c in cols):
            raise DataError("each regressor in a mapping must be one column")
        values = np.column_stack(cols) if cols else np.empty((n, 0))
    else:
        arr = np.asarray(X)
        if arr.dtype.names:  # a structured array: its field names
            x_names = list(arr.dtype.names)
            values = np.column_stack([np.asarray(arr[f], dtype=float) for f in x_names])
        else:
            values = np.asarray(arr, dtype=float)
            if values.ndim == 1:
                values = values[:, None]
            if values.ndim != 2:
                raise DataError("X must be a table of regressors: a DataFrame or a 2-d array")
            x_names = [f"x{j + 1}" for j in range(values.shape[1])]
        if names is not None:
            x_names = [str(v) for v in names]
            if len(x_names) != values.shape[1]:
                raise DataError(f"names= gives {len(x_names)} names for {values.shape[1]} columns")
    if values.shape[0] != n:
        raise DataError(f"y has {n} observations and X {values.shape[0]}")
    return x_names, np.ascontiguousarray(values), index


def _column(value: Any, index: pd.Index, n: int, what: str) -> np.ndarray:
    if isinstance(value, pd.Series):
        if not value.index.equals(index):
            raise DataError(f"{what} must have the same index as y and X")
        return value.to_numpy(dtype=float, na_value=np.nan)
    out = np.asarray(value, dtype=float)
    if out.shape != (n,):
        raise DataError(f"{what} must be one value per observation ({n})")
    return out


def _cluster_frame(value: Any, index: pd.Index, n: int) -> pd.DataFrame:
    if isinstance(value, pd.Series):
        value = value.to_frame("clusters" if value.name is None else value.name)
    if isinstance(value, pd.DataFrame):
        if not value.index.equals(index):
            raise DataError("clusters must have the same index as y and X")
        frame = value.copy()
        frame.columns = [str(c) for c in frame.columns]
        return frame.reset_index(drop=True)
    arr = np.asarray(value)
    if arr.shape[0] != n:
        raise DataError(f"clusters must give a group for each observation ({n})")
    return pd.DataFrame({"clusters": arr}) if arr.ndim == 1 else pd.DataFrame(arr, columns=[f"clusters{j + 1}" for j in range(arr.shape[1])])


def _time_index(index: pd.Index) -> pd.Index:
    """Dates with a frequency become periods, as ``as_dataset`` makes them; an unnamed time index is called "time"."""
    if isinstance(index, pd.DatetimeIndex):
        from ..data.coerce import _HOW, _period_freq

        freq = index.freqstr or (pd.infer_freq(index) if len(index) >= 3 else None)
        if not freq:
            raise DataError(f"the dates have no regular frequency, so pymetrix cannot tell the periods; {_HOW}")
        index = index.to_period(_period_freq(freq)).rename(index.name or "time")
    elif isinstance(index, pd.PeriodIndex) and index.name is None:
        index = index.rename("time")
    return index


def _kind(index: pd.Index) -> tuple[str, str | None]:
    if isinstance(index, pd.PeriodIndex):
        return "time_series", index.name
    if isinstance(index, pd.MultiIndex):
        periods = [name for name, level in zip(index.names, index.levels) if isinstance(level, pd.PeriodIndex)]
        if len(periods) == 1:
            return ("panel" if index.nlevels == 2 else "multi"), periods[0]
        return ("cross_section" if index.nlevels == 1 else "multi"), None
    return "cross_section", None


# -- provenance: the fields Equation.fit records, computed from the arrays ----------------------------------------------


def _fingerprint(columns: Sequence[str], index: pd.Index, values: np.ndarray) -> str:
    """``pymetrix._capture.fingerprint`` of the frame these make (the same hash), without building it."""
    digest = hashlib.sha256()
    digest.update("|".join(map(str, columns)).encode())
    digest.update("|".join(map(str, index)).encode())
    digest.update(np.ascontiguousarray(values, dtype=float).tobytes())
    return digest.hexdigest()[:16]


def _runs(index: pd.PeriodIndex) -> list[list[str]]:
    """Runs of consecutive periods, as ``pymetrix._capture`` records them."""
    ordinals = np.sort(index.asi8)
    if not len(ordinals):
        return []
    breaks = np.flatnonzero(np.diff(ordinals) != 1) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks - 1, [len(ordinals) - 1]])
    freq = index.freq
    return [[str(pd.Period(ordinal=int(ordinals[a]), freq=freq)), str(pd.Period(ordinal=int(ordinals[b]), freq=freq))]
            for a, b in zip(starts, ends)]


class _Axis:
    def __init__(self, name: str) -> None:
        self.name = name


class _Data:  # what pymetrix._capture._row_runs reads of a dataset
    def __init__(self, time: str | None) -> None:
        self.time_axis = _Axis(time) if time is not None else None


def _provenance(eq: Any, method: Any, raw: _Inputs, keep: np.ndarray, dropped: int, fit: Any, caught: list[str]) -> Any:
    from .._capture import _LISTED, _label, _row_runs
    from .._serial import to_jsonable
    from .._version import __version__
    from ..adapters import native
    from ..metadata import Provenance

    used = fit.resid.index
    variables = sorted({raw.y_name, *raw.x_names})
    values = raw.raw()
    span = None
    rows = None
    if raw.time is not None and len(used):
        periods = used.get_level_values(raw.time) if isinstance(used, pd.MultiIndex) else used
        span = (str(periods.min()), str(periods.max()))
        rows = _runs(used) if isinstance(used, pd.PeriodIndex) else _row_runs(_Data(raw.time), used)
    resolved = {"hac_lags": int(fit.stats["hac_lags"])} if "hac_lags" in fit.stats else None
    dropped_rows = None
    if dropped:
        observed = np.ones(len(keep), dtype=bool)
        for column in [*(values[v] for v in variables), *([raw.w] if raw.w is not None else [])]:
            observed &= ~np.isnan(column)
        reasons: dict[str, int] = {}
        listed = []
        for pos in np.flatnonzero(~keep):
            reason = "lag" if observed[pos] else "missing"  # (a row whose values are all there but infinite, as a dataset says)
            reasons[reason] = reasons.get(reason, 0) + 1
            if len(listed) < _LISTED:
                listed.append([_label(raw.index[pos]), reason])
        dropped_rows = {"by_reason": reasons, "rows": listed, "more": max(0, dropped - _LISTED)}
    used_values = np.column_stack([values[v][keep] for v in variables])
    return Provenance(
        data_vintages={v: None for v in variables},
        spec=to_jsonable(eq),
        method=to_jsonable(method),
        sample=span,
        nobs=len(used),
        dropped=dropped,
        package_version=__version__,
        seed=None,
        data_fingerprint=_fingerprint(variables, used, used_values),
        backend=native.backend_record(f"pymetrix.{type(method).__name__}"),
        warnings=tuple(caught),
        resolved=resolved,
        sample_source="all observations",
        sample_definition=None,
        rows=rows,
        presample=(),
        dropped_rows=dropped_rows,
        vintage_request={"policy": "direct"},
    )
