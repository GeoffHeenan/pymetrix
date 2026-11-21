"""Smooth local projections (Barnichon and Brownlees 2019): impulse responses that are a penalised spline in the horizon.

    lp = LocalProjection("100*log(gdp)", "mp_shock", horizons=20, lags=4, controls=["infl"])
    smooth = lp.fit_smooth(data)                      # the penalty chosen by cross-validation
    smooth.irf; smooth.plot(); lp.fit_smooth(data, penalty=0)   # no penalty: the ordinary local projection

All horizons' regressions are estimated together, on the periods where every horizon's response is observed:
y_(t+h) = a_h + beta(h) x_t + g_h' w_t + e_(t,h), where the controls' coefficients g_h and the constants are free at
each horizon and the response beta(h) = sum_k b_k B_k(h) is a cubic B-spline in h on evenly spaced knots, one a horizon, so that
with no penalty it is any response at all and the estimates are the ordinary local projection's.
The penalty lambda * sum (Delta^r b)^2 (r-th differences, ``order`` 2 by default) shrinks the response towards a
polynomial of degree r - 1 in h; lambda is chosen by ``folds``-fold cross-validation over blocks of consecutive
periods, from a grid, unless ``penalty`` is given. Standard errors are Newey-West's (h + 1 lags of the stacked scores,
grouped by period) for the penalised estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pymetrix._lazy import stats

from .errors import SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset
    from .projections import LocalProjection


def spline_basis(horizons: int) -> np.ndarray:
    """Cubic B-splines on evenly spaced knots (one a horizon, extended beyond both ends), evaluated at 0, ..., H:
    (H + 1) x (H + 3), of rank H + 1. Evenly spaced, so that coefficients changing linearly give a straight line, as
    the difference penalty needs (Eilers and Marx's P-splines)."""
    from scipy.interpolate import BSpline

    x = np.arange(horizons + 1, dtype=float)
    knots = np.arange(-3.0, horizons + 4.0)
    return BSpline.design_matrix(x, knots, 3).toarray()


def _split_terms(rhs: str) -> list[str]:
    """The top-level terms of a sum (``+`` outside parentheses)."""
    out, depth, cur = [], 0, ""
    for ch in rhs:
        depth += ch == "("
        depth -= ch == ")"
        if ch == "+" and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    out.append(cur.strip())
    return [t for t in out if t]


@dataclass(frozen=True, eq=False)
class FittedSmoothLocalProjection:
    """A smooth local projection: the response by horizon with standard errors, and the penalty."""

    spec: Any
    irf: pd.DataFrame  # by horizon: response, se, bands, and the ordinary LP's response on the same sample
    penalty: float
    order: int
    nobs: int
    cv: pd.Series | None = field(default=None, repr=False)  # cross-validation error by penalty

    stage = "estimate"

    def summary(self) -> Table:
        f = self.irf
        rows = [[float(r.response), float(r.se), float(r.lower), float(r.upper), float(r.ordinary)] for r in f.itertuples()]
        how = "chosen by cross-validation" if self.cv is not None else "given"
        return Table(["response", "std err", "lower 90%", "upper 90%", "ordinary LP"], rows, index=list(f.index), index_name="horizon",
                     title=f"Smooth local projection: {self.spec.response} to {self.spec.shock}",
                     notes=(f"Cubic B-spline in the horizon, penalty {self.penalty:.4g} ({how}) on differences of order {self.order}; "
                            f"{self.nobs} periods with every horizon observed.",
                            "Newey-West standard errors for the penalised estimator; 'ordinary LP' is the unpenalised estimate on the same periods."))

    def plot(self) -> Any:
        from .figures import Band, Figure, Line, Panel, Rule

        f = self.irf
        return Figure((Panel((Band(f["lower"], f["upper"], "90% band"), Line(f["response"], "smooth"),
                              Line(f["ordinary"], "ordinary LP", style="--"), Rule(0.0)),
                             title=f"Response of {self.spec.response} to {self.spec.shock}", xlabel="horizon"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.irf

    def __repr__(self) -> str:
        return f"<FittedSmoothLocalProjection {self.spec.response} to {self.spec.shock}, penalty {self.penalty:.3g}>"


def _stacked(spec: LocalProjection, data: Dataset, span: Any) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, pd.Index]:
    """The responses by horizon (T, H + 1), the shock and controls (T, 1 + m) on the common sample, and its periods."""
    H = spec.horizons
    lhs, rhs = {}, None
    for h in range(H + 1):
        left, right = spec.formula(h).split(" ~ ", 1)
        lhs[h] = data.eval(left)
        if rhs is None:
            rhs = _split_terms(right)
    cols = {"__shock__": data.eval(rhs[0])}
    for j, term in enumerate(rhs[1:]):
        cols[f"__w{j}"] = data.eval(term)
    frame = pd.concat([pd.DataFrame({f"__y{h}": s for h, s in lhs.items()}), pd.DataFrame(cols)], axis=1)
    frame = frame[data.sample_mask(span)].dropna()
    if len(frame) < 3 * (len(rhs) + 2):
        raise SpecificationError(f"{len(frame)} periods with every horizon observed: too few for a smooth local projection")
    Y = frame[[f"__y{h}" for h in range(H + 1)]].to_numpy(float)
    W = frame[[c for c in frame.columns if c.startswith("__w")]].to_numpy(float)
    x = frame["__shock__"].to_numpy(float)
    return Y, [x, W], frame.index


def _design(x: np.ndarray, W: np.ndarray, B: np.ndarray, rows: np.ndarray, H: int) -> tuple[np.ndarray, int]:
    """The stacked design for periods ``rows``: per-horizon constants and controls, then the spline in the shock."""
    n, m = len(rows), W.shape[1]
    free = 1 + m
    X = np.zeros((n * (H + 1), (H + 1) * free + B.shape[1]))
    for h in range(H + 1):
        r = slice(h * n, (h + 1) * n)
        X[r, h * free] = 1.0
        X[r, h * free + 1:(h + 1) * free] = W[rows]
        X[r, (H + 1) * free:] = x[rows, None] * B[h][None, :]
    return X, (H + 1) * free


def _solve(X: np.ndarray, y: np.ndarray, P: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """The penalised least squares estimate; without a penalty the spline's coefficients are not unique (two more than
    the horizons), the response is, and the minimum-norm solution gives it. Solved as least squares on X stacked over
    the penalty's square root, not through X'X, whose condition number is the square of X's."""
    A = X.T @ X + lam * P
    w, V = np.linalg.eigh(P)
    R = np.sqrt(lam * np.clip(w, 0, None))[:, None] * V.T  # R'R = lam P
    theta = np.linalg.lstsq(np.vstack([X, R]), np.concatenate([y, np.zeros(len(R))]), rcond=None)[0]
    return theta, A


def fit_smooth(spec: LocalProjection, data: Dataset, span: Any = None, penalty: float | None = None, order: int = 2,
               folds: int = 5, alpha: float = 0.10) -> FittedSmoothLocalProjection:
    if spec.state or spec.instruments or data.kind == "panel":
        raise SpecificationError("smooth local projections here are for time series, without a state or instruments")
    H = spec.horizons
    if H < order:
        raise SpecificationError(f"a penalty of order {order} needs more than {order} horizons")
    Y, (x, W), index = _stacked(spec, data, span)
    T = len(index)
    B = spline_basis(H)
    K = B.shape[1]
    D = np.diff(np.eye(K), n=order, axis=0)
    rows_all = np.arange(T)
    X, start = _design(x, W, B, rows_all, H)
    y = Y.T.reshape(-1)  # horizon-major, as the design
    P = np.zeros((X.shape[1], X.shape[1]))
    P[start:, start:] = D.T @ D
    cv = None
    if penalty is None:
        grid = np.concatenate([[0.0], np.geomspace(1e-3, 1e4, 30)]) * (np.trace(X.T @ X) / X.shape[1])
        edges = np.linspace(0, T, folds + 1).astype(int)
        errors = []
        for lam in grid:
            err = 0.0
            for f in range(folds):
                test = np.arange(edges[f], edges[f + 1])
                train = np.setdiff1d(rows_all, test)
                Xtr, _ = _design(x, W, B, train, H)
                ytr = Y[train].T.reshape(-1)
                theta, _ = _solve(Xtr, ytr, P, lam)
                Xte, _ = _design(x, W, B, test, H)
                err += float(np.sum((Y[test].T.reshape(-1) - Xte @ theta) ** 2))
            errors.append(err)
        cv = pd.Series(errors, index=pd.Index(grid, name="penalty"), name="cross-validation error")
        penalty = float(grid[int(np.argmin(errors))])
    theta, A = _solve(X, y, P, float(penalty))
    e = y - X @ theta
    # Newey-West on scores grouped by period (the period's errors across horizons are correlated and overlap)
    S = np.zeros((T, X.shape[1]))
    for h in range(H + 1):
        r = slice(h * T, (h + 1) * T)
        S += X[r] * e[r, None]
    L = H + 1
    meat = S.T @ S
    for lag in range(1, L + 1):
        w = 1 - lag / (L + 1)
        G = S[lag:].T @ S[:-lag]
        meat += w * (G + G.T)
    Ainv = np.linalg.pinv(A)
    cov = Ainv @ meat @ Ainv
    Vb = cov[start:, start:]
    resp = B @ theta[start:]
    se = np.sqrt(np.einsum("hk,kl,hl->h", B, Vb, B))
    ordinary, _ = _solve(X, y, P, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    irf = pd.DataFrame({"response": resp * spec.scale, "se": se * spec.scale, "lower": (resp - z * se) * spec.scale,
                        "upper": (resp + z * se) * spec.scale, "ordinary": (B @ ordinary[start:]) * spec.scale},
                       index=pd.RangeIndex(H + 1, name="horizon"))
    return FittedSmoothLocalProjection(spec, irf, float(penalty), int(order), T, cv)
