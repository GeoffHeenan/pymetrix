"""Spatial econometrics for a cross-section: spatial weights, Moran's I and the LM tests, and the spatial lag and
spatial error models by maximum likelihood (Ord 1975; Anselin 1988).

    W = knn_weights(data, ["lon", "lat"], k=5)          # or any n x n array (rows in the data's order)
    spatial_diagnostics(data, "crime", ["income", "housing"], W)   # Moran's I on OLS residuals, LM lag, LM error, robust
    spatial_regression(data, "crime", ["income", "housing"], W, model="lag")     # y = rho W y + X b + e
    spatial_regression(data, "crime", ["income", "housing"], W, model="error")   # y = X b + u, u = lambda W u + e

Weights are row-standardised unless ``row_standardise=False``. The likelihoods are concentrated in b and sigma^2 and
maximised over rho (or lambda) on the interval between the reciprocals of W's smallest and largest eigenvalues, the
log-determinant log|I - rho W| = sum log(1 - rho omega_i) from W's eigenvalues; standard errors come from
Anselin's (1988) analytic information matrix, as spreg's do. Moran's I of the residuals uses its exact moments under normality
(Cliff and Ord); the LM tests are Anselin's (1988), and their robust forms Anselin, Bera, Florax and Yoon's (1996).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import optimize

from pymetrix._lazy import stats as sps

from ._api import api
from .errors import SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["SpatialModel", "knn_weights", "spatial_diagnostics", "spatial_regression"]


def knn_weights(data: Dataset, coordinates: Sequence[str], k: int = 5) -> np.ndarray:
    """Binary k-nearest-neighbour weights from the ``coordinates`` columns (Euclidean distance), rows in the data's order."""
    P = data.frame[list(coordinates)].to_numpy(float)
    n = len(P)
    if not 1 <= k < n:
        raise SpecificationError("k must be between 1 and the number of observations less one")
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    W = np.zeros((n, n))
    nearest = np.argsort(D, axis=1)[:, :k]
    W[np.repeat(np.arange(n), k), nearest.ravel()] = 1.0
    return W


def _weights(W: Any, n: int, row_standardise: bool) -> np.ndarray:
    W = np.asarray(W.to_numpy() if hasattr(W, "to_numpy") else W, float)
    if W.shape != (n, n):
        raise SpecificationError(f"the weights must be {n} x {n}, a row and column for each observation")
    if np.any(np.diag(W) != 0):
        raise SpecificationError("the weights' diagonal must be zero: no observation is its own neighbour")
    if row_standardise:
        rs = W.sum(axis=1, keepdims=True)
        if np.any(rs == 0):
            raise SpecificationError("an observation has no neighbours: row-standardising needs at least one each")
        W = W / rs
    return W


def _arrays(data: Dataset, y: str, x: Sequence[str], W: Any, row_standardise: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    missing = [v for v in [y, *x] if v not in data]
    if missing:
        raise SpecificationError(f"{', '.join(missing)} are not series of the data")
    frame = data.frame[[y, *x]].astype(float)
    if frame.isna().to_numpy().any():
        raise SpecificationError("spatial models need complete data: the weights refer to every observation")
    n = len(frame)
    X = np.column_stack([np.ones(n), frame[list(x)].to_numpy()])
    return frame[y].to_numpy(), X, _weights(W, n, row_standardise), ["const", *x]


@api("pymetrix:spatial_diagnostics")
def spatial_diagnostics(data: Dataset, y: str, x: Sequence[str], W: Any, row_standardise: bool = True) -> Table:
    """Moran's I of the OLS residuals and Anselin's LM tests for a spatial lag and a spatial error, with robust forms."""
    x = [x] if isinstance(x, str) else list(x)
    Y, X, Wm, _ = _arrays(data, y, x, W, row_standardise)
    n, k = X.shape
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    e = Y - X @ b
    s2 = e @ e / n
    M = np.eye(n) - X @ np.linalg.pinv(X)
    S0 = Wm.sum()
    I = n / S0 * (e @ Wm @ e) / (e @ e)
    MW = M @ Wm
    EI = n / S0 * np.trace(MW) / (n - k)
    VI = (n / S0) ** 2 * (np.trace(MW @ M @ Wm.T) + np.trace(MW @ MW) + np.trace(MW) ** 2) / ((n - k) * (n - k + 2)) - EI**2
    zI = (I - EI) / np.sqrt(VI)
    T = np.trace(Wm.T @ Wm + Wm @ Wm)
    WXb = Wm @ X @ b
    J = (WXb @ M @ WXb + T * s2) / (n * s2)
    d_err = (e @ Wm @ e) / s2
    d_lag = (e @ Wm @ Y) / s2
    lm_err = d_err**2 / T
    lm_lag = d_lag**2 / (n * J)
    rlm_lag = (d_lag - d_err) ** 2 / (n * J - T)
    rlm_err = (d_err - T / (n * J) * d_lag) ** 2 / (T * (1 - T / (n * J)))
    rows = [["Moran's I (residuals)", float(I), float(zI), float(2 * sps.norm.sf(abs(zI)))],
            ["LM lag", float(lm_lag), np.nan, float(sps.chi2.sf(lm_lag, 1))],
            ["robust LM lag", float(rlm_lag), np.nan, float(sps.chi2.sf(rlm_lag, 1))],
            ["LM error", float(lm_err), np.nan, float(sps.chi2.sf(lm_err, 1))],
            ["robust LM error", float(rlm_err), np.nan, float(sps.chi2.sf(rlm_err, 1))]]
    notes = (f"OLS of {y} on {', '.join(x)}; {n} observations; {'row-standardised ' if row_standardise else ''}weights. Moran's I with its exact "
             "moments under normality (z); the LM statistics are chi-squared(1). Where both LM tests reject, the larger robust one "
             "suggests the model (Anselin et al. 1996).",)
    return Table(["test", "statistic", "z", "p-value"], rows, title=f"Spatial diagnostics: {y}", notes=notes)


@dataclass(frozen=True, eq=False)
class SpatialModel:
    """A spatial lag or spatial error model estimated by maximum likelihood."""

    model: str
    y_name: str
    names: tuple[str, ...]
    params: np.ndarray  # b, then rho or lambda
    cov: np.ndarray = field(repr=False)
    sigma2: float = 0.0
    loglike: float = 0.0
    nobs: int = 0
    resid: np.ndarray = field(default=None, repr=False)

    stage = "estimate"

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def summary(self) -> Table:
        z = self.params / self.se
        rows = [[float(b), float(s), float(t), float(2 * sps.norm.sf(abs(t)))] for b, s, t in zip(self.params, self.se, z)]
        label = "Spatial lag (SAR)" if self.model == "lag" else "Spatial error (SEM)"
        k = len(self.params) - 1
        aic = -2 * self.loglike + 2 * (k + 2)
        notes = (f"Method: {label} by maximum likelihood", f"Observations: {self.nobs}", f"Error variance (sigma^2): {self.sigma2:.6g}",
                 f"Log likelihood: {self.loglike:.4f}", f"Akaike info criterion: {aic:.4f}")
        return Table(["estimate", "std err", "z", "p-value"], rows, index=list(self.names), index_name="parameter",
                     title=f"{label}: {self.y_name}", notes=notes)

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return pd.DataFrame({"estimate": self.params, "se": self.se}, index=list(self.names))

    def __repr__(self) -> str:
        return f"<SpatialModel {self.model} {self.y_name}: {self.names[-1]} = {self.params[-1]:.4f}>"


@api("pymetrix:spatial_regression")
def spatial_regression(data: Dataset, y: str, x: Sequence[str], W: Any, model: Literal["lag", "error"] = "lag",
                       row_standardise: bool = True) -> SpatialModel:
    """The spatial lag (y = rho W y + X b + e) or spatial error (u = lambda W u + e) model by maximum likelihood."""
    x = [x] if isinstance(x, str) else list(x)
    if model not in ("lag", "error"):
        raise SpecificationError("model is 'lag' or 'error'")
    Y, X, Wm, names = _arrays(data, y, x, W, row_standardise)
    n, k = X.shape
    omega = np.linalg.eigvals(Wm)
    real = omega.real
    lo, hi = 1 / real.min(), 1 / real.max()
    I = np.eye(n)

    def logdet(r: float) -> float:
        return float(np.sum(np.log(np.abs(1 - r * omega))))

    def concentrated(r: float) -> tuple[float, np.ndarray, float]:
        if model == "lag":
            Ay = Y - r * (Wm @ Y)
            b = np.linalg.lstsq(X, Ay, rcond=None)[0]
            e = Ay - X @ b
        else:
            B = I - r * Wm
            b = np.linalg.lstsq(B @ X, B @ Y, rcond=None)[0]
            e = B @ (Y - X @ b)
        s2 = e @ e / n
        return float(-n / 2 * (np.log(2 * np.pi) + np.log(s2) + 1) + logdet(r)), b, float(s2)

    found = optimize.minimize_scalar(lambda r: -concentrated(r)[0], bounds=(lo + 1e-6, hi - 1e-6), method="bounded", options={"xatol": 1e-10})
    r = float(found.x)
    ll, b, s2 = concentrated(r)

    # Anselin's (1988) information matrix in (b, rho or lambda, sigma^2), evaluated at the estimates
    if model == "lag":
        WA = Wm @ np.linalg.inv(I - r * Wm)
        WAXb = WA @ X @ b
        info = np.zeros((k + 2, k + 2))
        info[:k, :k] = X.T @ X / s2
        info[:k, k] = info[k, :k] = X.T @ WAXb / s2
        info[k, k] = np.trace(WA @ WA) + np.trace(WA.T @ WA) + WAXb @ WAXb / s2
        info[k, k + 1] = info[k + 1, k] = np.trace(WA) / s2
        info[k + 1, k + 1] = n / (2 * s2**2)
    else:
        B = I - r * Wm
        WB = Wm @ np.linalg.inv(B)
        BX = B @ X
        info = np.zeros((k + 2, k + 2))
        info[:k, :k] = BX.T @ BX / s2
        info[k, k] = np.trace(WB @ WB) + np.trace(WB.T @ WB)
        info[k, k + 1] = info[k + 1, k] = np.trace(WB) / s2
        info[k + 1, k + 1] = n / (2 * s2**2)
    cov = np.linalg.inv(info)[:k + 1, :k + 1]
    e = (Y - r * (Wm @ Y) - X @ b) if model == "lag" else (I - r * Wm) @ (Y - X @ b)
    return SpatialModel(model, y, (*names, "rho" if model == "lag" else "lambda"), np.r_[b, r], cov, s2, ll, n, e)
