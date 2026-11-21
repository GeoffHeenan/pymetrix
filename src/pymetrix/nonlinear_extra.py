"""Threshold VARs and bilinear models.

    tvar = threshold_var(data, ["gy", "pi", "r"], lags=2, threshold="gy", delay=1)   # two regimes split by gy[-1]
    tvar.summary(); tvar.regime_table(); tvar.test(draws=200)                         # Tsay's sup-LR, bootstrapped
    bilinear(data, "y")                     # y_t = mu + phi y_{t-1} + b y_{t-1} e_{t-1} + e_t (Granger and Andersen 1978)

A threshold VAR (Tsay 1998) fits a separate VAR in each of two regimes, regime 1 when the threshold variable d periods
back is at or below gamma; gamma minimises log |Sigma(gamma)| over the middle of its values (``trim`` cut from each
end), the residual covariance pooled over both regimes. The LR statistic against the linear VAR is
T (log |Sigma_linear| - log |Sigma(gamma)|); gamma is unidentified under linearity, so its p-value comes from a
residual bootstrap of the linear VAR (Hansen 1996), searching gamma again in each draw.

The bilinear BL(1, 0, 1, 1) model is estimated by conditional least squares: the residuals by the recursion
e_t = y_t - mu - phi y_{t-1} - b y_{t-1} e_{t-1} from e_1 = 0, which is Gaussian maximum likelihood conditional on the
start. With b = 0 it is least squares on an AR(1).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import optimize

from pymetrix._lazy import stats as sps

from ._api import api
from ._random import generator as make_rng
from .errors import SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["FittedThresholdVAR", "bilinear", "threshold_var"]


def _var_design(Y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """(Y_t for t = p..T-1, [1, Y_{t-1}, ..., Y_{t-p}])."""
    T = len(Y)
    X = np.column_stack([np.ones(T - p)] + [Y[p - j:T - j] for j in range(1, p + 1)])
    return Y[p:], X


def _design(Y: np.ndarray, m: int, p: int) -> tuple[np.ndarray, np.ndarray]:
    """The VAR(p) regression on rows m, ..., T - 1 (m >= p, so that the delay's presample is dropped too)."""
    y, X = _var_design(Y, m)
    return y, X[:, :1 + Y.shape[1] * p]


def _fit_regimes(y: np.ndarray, X: np.ndarray, low: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(coefficients of regime 1, of regime 2, residuals), each regime by least squares."""
    U = np.empty_like(y)
    B = []
    for mask in (low, ~low):
        b = np.linalg.lstsq(X[mask], y[mask], rcond=None)[0]
        U[mask] = y[mask] - X[mask] @ b
        B.append(b)
    return B[0], B[1], U


def _search(y: np.ndarray, X: np.ndarray, z: np.ndarray, trim: float) -> tuple[float, float]:
    """(gamma, log |Sigma(gamma)|) minimising the pooled residual covariance's log determinant."""
    k = X.shape[1]
    lo, hi = int(np.floor(trim * len(z))), int(np.ceil((1 - trim) * len(z)))
    candidates = np.unique(np.sort(z)[lo:hi])
    best, gamma = np.inf, np.nan
    for g in candidates:
        low = z <= g
        if min(low.sum(), (~low).sum()) <= k + 1:
            continue
        _, _, U = _fit_regimes(y, X, low)
        ld = np.linalg.slogdet(U.T @ U / len(U))[1]
        if ld < best:
            best, gamma = ld, float(g)
    if not np.isfinite(best):
        raise SpecificationError("no threshold leaves enough observations in each regime; lower trim or lags")
    return gamma, float(best)


@dataclass(frozen=True, eq=False)
class FittedThresholdVAR:
    """A two-regime threshold VAR."""

    variables: tuple[str, ...]
    lags: int
    threshold_variable: str
    delay: int
    gamma: float
    coefs_low: np.ndarray = field(repr=False)  # (1 + K p, K): constant, then lags; columns are equations
    coefs_high: np.ndarray = field(repr=False)
    sigma: np.ndarray = field(repr=False)
    regime: pd.Series = field(repr=False)  # 1 or 2 by period
    LR: float = 0.0
    trim: float = 0.15
    Y: np.ndarray = field(default=None, repr=False)  # the variables over the whole sample, presample included
    z: np.ndarray = field(default=None, repr=False)  # the threshold variable, aligned with the estimation sample

    stage = "estimate"

    def _names(self) -> list[str]:
        return ["const"] + [f"{v}[-{j}]" for j in range(1, self.lags + 1) for v in self.variables]

    def regime_table(self) -> Table:
        rows, index = [], []
        for name, b in (("regime 1", self.coefs_low), ("regime 2", self.coefs_high)):
            for r, reg in enumerate(self._names()):
                index.append(f"{name}: {reg}")
                rows.append([float(x) for x in b[r]])
        return Table(list(self.variables), rows, index=index, index_name="", title="Threshold VAR coefficients (columns: equations)")

    def summary(self) -> Table:
        n1 = int((self.regime == 1).sum())
        rows = [["threshold variable", f"{self.threshold_variable}[-{self.delay}]"], ["threshold", self.gamma],
                ["periods in regime 1 (at or below)", n1], ["periods in regime 2", int(len(self.regime) - n1)],
                ["LR against the linear VAR", self.LR]]
        notes = ("Regimes by least squares; the threshold minimises the pooled residual covariance's log determinant, searched over "
                 f"the middle {1 - 2 * self.trim:.0%} of the threshold variable's values. The LR statistic's p-value needs "
                 "test(draws=...): the threshold is unidentified under linearity.",)
        return Table(["", "value"], rows, title=f"Threshold VAR: {', '.join(self.variables)}", notes=notes)

    @api()
    def test(self, draws: int = 200, seed: int = 0) -> Any:
        """Tsay's (1998) sup-LR test of linearity, its p-value by a residual bootstrap of the linear VAR, the threshold
        searched again in every draw (Hansen 1996)."""
        from .stats.base import TestResult

        Yfull, m, p, K = self.Y, max(self.lags, self.delay), self.lags, len(self.variables)
        y, X = _design(Yfull, m, p)
        b_lin = np.linalg.lstsq(X, y, rcond=None)[0]
        U = y - X @ b_lin
        T = len(y)
        rng = make_rng(seed)
        zi = self.variables.index(self.threshold_variable) if self.threshold_variable in self.variables else None
        stats = []
        for _ in range(int(draws)):
            e = U[rng.integers(0, T, T)]
            Ys = np.empty_like(Yfull)
            Ys[:m] = Yfull[:m]
            for t in range(m, len(Yfull)):
                x = np.r_[1.0, Ys[t - p:t][::-1].reshape(-1)]
                Ys[t] = x @ b_lin + e[t - m]
            ys, Xs = _design(Ys, m, p)
            zs = Ys[m - self.delay:len(Ys) - self.delay, zi] if zi is not None else self.z
            u = ys - Xs @ np.linalg.lstsq(Xs, ys, rcond=None)[0]
            try:
                _, ld = _search(ys, Xs, zs, self.trim)
            except SpecificationError:
                continue
            stats.append(T * (np.linalg.slogdet(u.T @ u / T)[1] - ld))
        stats = np.array(stats)
        p_value = float((np.sum(stats >= self.LR) + 1) / (len(stats) + 1))
        return TestResult("Threshold VAR sup-LR (Tsay)", "the VAR is linear (one regime)", float(self.LR), None, p_value,
                          "bootstrap", details={"draws": len(stats), "threshold": self.gamma})

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.regime_table().to_frame()

    def __repr__(self) -> str:
        return f"<FittedThresholdVAR {', '.join(self.variables)}: threshold {self.threshold_variable}[-{self.delay}] = {self.gamma:.4g}>"


@api("pymetrix:threshold_var")
def threshold_var(data: Dataset, variables: Sequence[str], lags: int = 1, threshold: str | None = None, delay: int = 1,
                  trim: float = 0.15, span: Any = None) -> FittedThresholdVAR:
    """A two-regime threshold VAR (Tsay 1998), split by ``threshold`` (default: the first variable) ``delay`` periods back."""
    variables = [variables] if isinstance(variables, str) else list(variables)
    threshold = threshold or variables[0]
    if data.kind != "time_series":
        raise SpecificationError("a threshold VAR needs a single time series")
    missing = [v for v in [*variables, threshold] if v not in data]
    if missing:
        raise SpecificationError(f"{', '.join(missing)} are not series of the data")
    if not 1 <= delay or lags < 1:
        raise SpecificationError("lags and delay must be at least 1")
    if not 0 < trim < 0.5:
        raise SpecificationError("trim is between 0 and 0.5")
    frame = data.frame[list(dict.fromkeys([*variables, threshold]))].astype(float)
    if span is not None:
        frame = frame[data.sample_mask(span)]
    frame = frame.dropna()
    m = max(lags, delay)
    Y = frame[variables].to_numpy()
    zfull = frame[threshold].to_numpy()
    y, X = _design(Y, m, lags)
    z = zfull[m - delay:len(zfull) - delay]
    gamma, ld = _search(y, X, z, trim)
    low = z <= gamma
    b1, b2, U = _fit_regimes(y, X, low)
    T = len(y)
    e_lin = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    LR = T * (np.linalg.slogdet(e_lin.T @ e_lin / T)[1] - ld)
    regime = pd.Series(np.where(low, 1, 2), index=frame.index[m:], name="regime")
    return FittedThresholdVAR(tuple(variables), lags, threshold, delay, gamma, b1, b2, U.T @ U / T, regime, float(LR), trim, Y, z)


# bilinear


def _bilinear_resid(y: np.ndarray, mu: float, phi: float, b: float) -> np.ndarray:
    e = np.zeros(len(y))
    for t in range(1, len(y)):
        e[t] = y[t] - mu - phi * y[t - 1] - b * y[t - 1] * e[t - 1]
    return e[1:]


@api("pymetrix:bilinear")
def bilinear(data: Dataset, variable: str, span: Any = None) -> Any:
    """The bilinear model y_t = mu + phi y_{t-1} + b y_{t-1} e_{t-1} + e_t by conditional least squares (Gaussian
    likelihood conditional on e_1 = 0); standard errors from the Gauss-Newton information."""
    from .count_series import SeriesModel, _series

    s = _series(data, variable, span)
    y = s.to_numpy()
    X = np.column_stack([np.ones(len(y) - 1), y[:-1]])
    mu0, phi0 = np.linalg.lstsq(X, y[1:], rcond=None)[0]

    def resid(th: np.ndarray) -> np.ndarray:
        e = _bilinear_resid(y, *th)
        return e if np.all(np.isfinite(e)) and np.max(np.abs(e)) < 1e8 else np.full(len(y) - 1, 1e8)

    found = optimize.least_squares(resid, np.array([mu0, phi0, 0.0]), method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14)
    theta = found.x
    e = resid(theta)
    n = len(e)
    s2 = e @ e / n
    J = found.jac
    cov = s2 * np.linalg.pinv(J.T @ J)
    ll = float(-n / 2 * (np.log(2 * np.pi) + np.log(s2) + 1))
    fitted = pd.Series(np.r_[np.nan, y[1:] - e], index=s.index, name="conditional mean")
    return SeriesModel("Bilinear BL(1, 0, 1, 1)", variable, ("mu", "phi", "b"), theta, cov, ll, n, fitted,
                       ("Conditional least squares from e_1 = 0 (Gaussian ML given the start); Gauss-Newton standard errors. "
                        "|phi| and b^2 var(e) must keep the process stable.",))
