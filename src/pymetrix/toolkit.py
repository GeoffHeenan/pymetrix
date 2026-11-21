"""Building blocks for writing your own estimators and solution methods.

These are the pieces pymetrix's own estimators are made of, kept stable as
part of the Extension API. A custom estimator written with them works like a
built-in one: in ``Equation.fit``, the Estimate dialog (once registered),
scripts, projects, group loops and ``Results`` tables.

Data to arrays::

    d = design("c ~ y + c[-1]", data, span=("2000Q1", None))   # d.y, d.X, d.names, d.index
    index, M = arrays(data, ["c", "y"])                          # plain columns as a matrix

Inference::

    beta, u = least_squares(d.y, d.X)
    V = covariance("hac", d.X, u, d.df_resid)                    # classic, white, hac, Cluster(...)
    V = sandwich(H, scores, kind="white")                        # for M- and ML estimators
    H = hessian(f, x); J = jacobian(g, x)                        # numerical derivatives

Results::

    return result(eq, data, d, self, beta, V, u)                 # a FittedEquation, with statistics

A worked estimator (ridge regression) and solver (damped Newton) are in
``examples/custom_estimator.py`` and ``examples/custom_solver.py``; see also
the help topic "Writing your own estimator".
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .errors import SpecificationError
from .estimation.covariance import Cluster, default_hac_lags
from .estimation.covariance import covariance as _covariance
from .estimation.design import cov_groups, instrument_series, linear_design, ls_stats, provenance
from .estimation.numdiff import hessian, jacobian
from .estimation.params import point
from .models.solvers import ModelError, PeriodSystem
from .plugins import register
from .statespace import SystemMatrices, kalman_filter, kalman_smoother, run

if TYPE_CHECKING:
    from .data import Dataset
    from .estimation import Equation, FittedEquation

__all__ = [
    "Cluster", "Design", "ModelError", "PeriodSystem", "SystemMatrices", "arrays", "covariance", "default_hac_lags", "design",
    "fit_stats", "gradient", "hessian", "jacobian", "kalman_filter", "kalman_smoother", "least_squares", "register", "result",
    "run", "sandwich", "to_series",
]


# ---------------------------------------------------------------------------------------------
# data to arrays


@dataclass(frozen=True, eq=False)
class Design:
    """An equation's data as arrays, after sample selection and missing values.

    ``y`` (n,), ``X`` (n, k) with columns ``names`` (``const`` first when the equation has one),
    ``index`` (the rows used), ``w`` (weights or None), ``Z`` (excluded instruments or None, columns
    ``z_names``), ``dropped`` (rows dropped for missing values) and ``absorbed`` (degrees of freedom
    used by fixed effects, already removed from ``y`` and ``X``).
    """

    y: np.ndarray
    X: np.ndarray
    names: list[str]
    index: pd.Index
    w: np.ndarray | None
    Z: np.ndarray | None
    z_names: list[str]
    dropped: int
    absorbed: int
    equation: Any
    cov_data: Any = None  # the dataset, for cluster groups

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def k(self) -> int:
        return self.X.shape[1]

    @property
    def df_resid(self) -> int:
        return self.n - self.k - self.absorbed

    def frame(self) -> pd.DataFrame:
        """``y`` and the regressors as a DataFrame on the rows used."""
        out = pd.DataFrame(self.X, index=self.index, columns=self.names)
        out.insert(0, str(self.equation.lhs), self.y)
        return out

    def groups(self, cov: Any) -> list[np.ndarray]:
        """Cluster codes for a ``Cluster(...)`` covariance, aligned with the rows used."""
        return cov_groups(cov, self.cov_data, self.index)


def design(equation: str | Equation, data: Dataset, *, span: Any = None, where: str | None = None, weights: str | None = None,
           missing: str = "drop", instruments: Sequence[str] = (), params: Sequence[str] = ()) -> Design:
    """The data of a linear-form equation as arrays: the formula language, lags within entity,
    fixed effects (``fe()``), categoricals (``C()``), sample and missing values, done for you."""
    from .estimation import Equation

    eq = equation if isinstance(equation, Equation) else Equation(equation, params=list(params))
    if eq.form != "linear":
        raise SpecificationError("design() is for linear-form equations; for explicit forms, evaluate the residual expression")
    z = instrument_series(instruments, data) if instruments else None
    d = linear_design(eq, data, span, where, weights, missing, instruments=z)
    return Design(d.y, d.X, list(d.names), d.index, d.w, d.Z, list(d.z_names), d.dropped, d.absorbed, eq, data)


def arrays(data: Dataset, names: Sequence[str], *, span: Any = None, missing: str = "drop") -> tuple[pd.Index, np.ndarray]:
    """Series (or expressions such as ``dlog(y)``) as the columns of a matrix, on the rows where
    all are observed (``missing="keep"`` keeps every row, with NaN)."""
    # expressions are evaluated on all the data, so lags at the start of the span use earlier periods
    columns = {n: data[n].to_numpy(float) if n in data.names else data.eval(n).to_numpy(float) for n in names}
    M = np.column_stack([columns[n] for n in names]) if names else np.zeros((len(data.frame), 0))
    keep = data.sample_mask(span)
    if missing != "keep":
        keep &= np.isfinite(M).all(axis=1)
    return data.frame.index[keep], M[keep]


def to_series(values: Any, index: pd.Index, name: str = "") -> pd.Series:
    """Values as a Series on ``index``, ready for ``data.with_series(name=...)``."""
    return pd.Series(np.asarray(values, dtype=float), index=index, name=name or None)


# ---------------------------------------------------------------------------------------------
# inference


def least_squares(y: np.ndarray, X: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(Weighted) least squares: coefficients and residuals."""
    if w is not None:
        sw = np.sqrt(w)
        beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    else:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta, y - X @ beta


def covariance(kind: Any, X: np.ndarray, u: np.ndarray, df_resid: int | None = None, *, groups: Sequence[np.ndarray] = (),
               hac_lags: int | None = None) -> np.ndarray:
    """Least-squares coefficient covariance: ``classic``, ``white`` (HC1), ``hac`` (Newey-West) or
    ``Cluster(...)`` (pass ``groups=design.groups(cov)``). ``X`` is the regressors, or the Jacobian of
    the residuals for nonlinear least squares."""
    df = len(u) - X.shape[1] if df_resid is None else df_resid
    return _covariance(kind, X, u, df, groups=groups, hac_lags=hac_lags)


def sandwich(H: np.ndarray, scores: np.ndarray, kind: str = "white", hac_lags: int | None = None) -> np.ndarray:
    """``H^-1 S H^-1`` for M- and ML estimators: ``H`` is the Hessian of the objective (or of the
    negative log likelihood), ``scores`` the per-observation gradients (n, k). ``kind``: ``white``
    (S = scores' scores), ``hac`` (Newey-West S) or ``classic`` (H^-1 alone)."""
    Hinv = np.linalg.inv(H)
    if kind == "classic":
        return Hinv
    S = scores.T @ scores
    if kind == "hac":
        lags = default_hac_lags(len(scores)) if hac_lags is None else hac_lags
        for lag in range(1, lags + 1):
            gamma = scores[lag:].T @ scores[:-lag]
            S += (1 - lag / (lags + 1)) * (gamma + gamma.T)
    elif kind != "white":
        raise SpecificationError("kind must be 'classic', 'white' or 'hac'")
    return Hinv @ S @ Hinv


def gradient(f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
    """Central-difference gradient of a scalar function."""
    return jacobian(lambda p: np.atleast_1d(f(p)), np.asarray(x, dtype=float))[0]


def fit_stats(y: np.ndarray, resid: np.ndarray, k: int, *, w: np.ndarray | None = None, df_resid: int | None = None,
              intercept: bool = True, time_series: bool = True) -> dict[str, float]:
    """The usual least-squares statistics: R², adjusted R², standard error, log likelihood, AIC, BIC,
    Durbin-Watson."""
    df = len(resid) - k if df_resid is None else df_resid
    return ls_stats(np.asarray(y, float), np.asarray(resid, float), w, k, df, intercept, time_series, within=False)


# ---------------------------------------------------------------------------------------------
# results


def result(equation: Equation, data: Dataset, d: Design, method: Any, coefficients: Any, cov: np.ndarray | None,
           resid: np.ndarray, *, names: Sequence[str] | None = None, fitted: np.ndarray | None = None,
           stats: Mapping[str, float] | None = None, extra: Mapping[str, Any] | None = None,
           backend: str = "") -> FittedEquation:
    """A ``FittedEquation`` from your estimates: coefficients (in ``names`` order, default the design's),
    their covariance, residuals on the design's rows. Statistics default to the least-squares ones;
    provenance (data vintages, sample, method) is recorded. ``backend`` names a library you used."""
    from .estimation import FittedEquation

    names = list(names or d.names)
    coefficients = np.asarray(coefficients, dtype=float)
    resid = np.asarray(resid, dtype=float)
    y_obs = d.y
    fitted = y_obs - resid if fitted is None else np.asarray(fitted, dtype=float)
    if stats is None:
        stats = fit_stats(y_obs, resid, len(names), w=d.w, df_resid=d.df_resid, intercept="const" in names,
                          time_series=data.kind == "time_series")
    prov = provenance(equation, method, data, d.index, d.dropped)
    if backend:
        prov = replace(prov, method={**prov.method, "backend": backend})
    return FittedEquation(
        equation=equation,
        params=point(names, coefficients, cov),
        method=method,
        resid=pd.Series(resid, index=d.index, name="resid"),
        fitted=pd.Series(fitted, index=d.index, name="fitted"),
        stats=dict(stats),
        provenance=prov,
        design=pd.DataFrame(d.X, index=d.index, columns=d.names),
        extra=dict(extra or {}),
    )
