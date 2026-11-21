"""Vector ARMA models by conditional maximum likelihood.

    fit = varma(data, ["gy", "pi"], p=1, q=1)
    fit.summary(); fit.irf(20).plot()

y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + e_t + M_1 e_{t-1} + ... + M_q e_{t-q}, Gaussian e_t with covariance Sigma.
The residuals come from the recursion with presample residuals zero, starting after the first p observations; the
likelihood, concentrated in Sigma, is -T/2 log |e'e / T|, maximised from the VAR(p) least-squares estimates with M = 0.
With q = 0 it is least squares. A VARMA is identified only up to common factors: keep p and q small, and check that the
estimated MA polynomial is invertible (reported). Responses are the moving-average representation's, identified
recursively (Cholesky) or to unit shocks.
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
from .errors import SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["FittedVARMA", "varma"]


def _unpack(theta: np.ndarray, K: int, p: int, q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = theta[:K]
    A = theta[K:K + K * K * p].reshape(p, K, K)
    M = theta[K + K * K * p:].reshape(q, K, K)
    return c, A, M


def _residuals(Y: np.ndarray, theta: np.ndarray, p: int, q: int) -> np.ndarray:
    """e_t for t = p, ..., T - 1: the AR part at once, then the MA recursion (presample residuals zero)."""
    T, K = Y.shape
    c, A, M = _unpack(theta, K, p, q)
    R = Y[p:] - c
    for i in range(p):
        R = R - Y[p - 1 - i:T - 1 - i] @ A[i].T
    if not q:
        return R
    E = np.zeros((T - p + q, K))  # q leading zeros: the presample residuals
    Mt = [M[j].T for j in range(q)]
    for t in range(T - p):
        e = R[t].copy()
        for j in range(q):
            e -= E[q + t - 1 - j] @ Mt[j]
        E[q + t] = e
    return E[q:]


def _companion_roots(B: np.ndarray) -> np.ndarray:
    m, K, _ = B.shape
    if m == 0:
        return np.zeros(0)
    C = np.zeros((K * m, K * m))
    C[:K] = np.hstack(list(B))
    C[K:, :-K] = np.eye(K * (m - 1))
    return np.abs(np.linalg.eigvals(C))


@dataclass(frozen=True, eq=False)
class FittedVARMA:
    """A VARMA(p, q) estimated by conditional maximum likelihood."""

    variables: tuple[str, ...]
    p: int
    q: int
    c: np.ndarray
    A: np.ndarray  # (p, K, K)
    M: np.ndarray  # (q, K, K)
    sigma: np.ndarray
    cov: np.ndarray = field(repr=False)
    loglike: float = 0.0
    nobs: int = 0
    resid: pd.DataFrame = field(default=None, repr=False)

    stage = "estimate"

    def _names(self) -> list[str]:
        K, v = len(self.variables), self.variables
        names = [f"const {v[i]}" for i in range(K)]
        names += [f"AR{l + 1} {v[i]} <- {v[j]}" for l in range(self.p) for i in range(K) for j in range(K)]
        names += [f"MA{l + 1} {v[i]} <- {v[j]}" for l in range(self.q) for i in range(K) for j in range(K)]
        return names

    @property
    def values(self) -> np.ndarray:
        return np.r_[self.c, self.A.reshape(-1), self.M.reshape(-1)]

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.cov), 0, None))

    def summary(self) -> Table:
        z = self.values / self.se
        rows = [[float(v), float(s), float(t), float(2 * sps.norm.sf(abs(t)))] for v, s, t in zip(self.values, self.se, z)]
        ar, ma = _companion_roots(self.A), _companion_roots(-self.M)
        notes = ("Method: conditional ML", f"Observations: {self.nobs}", f"Log likelihood: {self.loglike:.4f}",
                 f"Largest AR root modulus {ar.max() if len(ar) else 0:.4f} (below 1: stationary); largest MA root modulus "
                 f"{ma.max() if len(ma) else 0:.4f} (below 1: invertible).")
        return Table(["estimate", "std err", "z", "p-value"], rows, index=self._names(), index_name="parameter",
                     title=f"VARMA({self.p}, {self.q}): {', '.join(self.variables)}", notes=notes)

    def ma_representation(self, periods: int) -> np.ndarray:
        """Psi_0 = I, Psi_h = M_h + sum_i A_i Psi_{h-i} (M_h = 0 beyond q): (periods + 1, K, K)."""
        K = len(self.variables)
        Psi = np.zeros((periods + 1, K, K))
        Psi[0] = np.eye(K)
        for h in range(1, periods + 1):
            Psi[h] = self.M[h - 1] if h <= self.q else 0
            for i in range(min(self.p, h)):
                Psi[h] = Psi[h] + self.A[i] @ Psi[h - 1 - i]
        return Psi

    def irf(self, periods: int = 20, orthogonal: bool = True) -> Any:
        from .var import IRF

        Psi = self.ma_representation(periods)
        values = Psi @ np.linalg.cholesky(self.sigma) if orthogonal else Psi
        return IRF(values, self.variables, self.variables, None, None, False,
                   "Cholesky (VARMA order)" if orthogonal else "reduced-form (unit) shocks")

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return pd.DataFrame({"estimate": self.values, "se": self.se}, index=self._names())

    def __repr__(self) -> str:
        return f"<FittedVARMA({self.p}, {self.q}) {', '.join(self.variables)}: log likelihood {self.loglike:.4f}>"


@api("pymetrix:varma")
def varma(data: Dataset, variables: Sequence[str], p: int = 1, q: int = 1, span: Any = None) -> FittedVARMA:
    """A VARMA(p, q) of ``variables`` by conditional maximum likelihood, from the VAR(p) least-squares estimates."""
    variables = [variables] if isinstance(variables, str) else list(variables)
    if data.kind != "time_series":
        raise SpecificationError("a VARMA needs a single time series")
    if p < 0 or q < 0 or p + q == 0:
        raise SpecificationError("p and q are non-negative, and not both zero")
    missing = [v for v in variables if v not in data]
    if missing:
        raise SpecificationError(f"{', '.join(missing)} are not series of the data")
    frame = data.frame[variables].astype(float)
    if span is not None:
        frame = frame[data.sample_mask(span)]
    frame = frame.dropna()
    Y = frame.to_numpy()
    T, K = Y.shape
    if T < 10 * K * (p + q + 1):
        raise SpecificationError("too few observations for this VARMA")
    if p:
        X = np.column_stack([np.ones(T - p)] + [Y[p - i:T - i] for i in range(1, p + 1)])
        B = np.linalg.lstsq(X, Y[p:], rcond=None)[0]
        c0 = B[0]
        A0 = np.stack([B[1 + K * i:1 + K * (i + 1)].T for i in range(p)])
    else:
        c0, A0 = Y.mean(axis=0), np.zeros((0, K, K))
    theta0 = np.r_[c0, A0.reshape(-1), np.zeros(K * K * q)]
    n = T - p

    def negll(th: np.ndarray) -> float:
        E = _residuals(Y, th, p, q)
        if not np.all(np.isfinite(E)) or np.max(np.abs(E)) > 1e8:
            return 1e300
        sign, ld = np.linalg.slogdet(E.T @ E / n)
        return float(n / 2 * ld) if sign > 0 else 1e300

    if q:
        found = optimize.minimize(negll, theta0, method="BFGS", options={"gtol": 1e-7, "maxiter": 10_000})
        theta = optimize.minimize(negll, found.x, method="BFGS", options={"gtol": 1e-9, "maxiter": 10_000}).x  # a restart, to settle
    else:
        theta = theta0
    E = _residuals(Y, theta, p, q)
    sigma = E.T @ E / n
    k = len(theta)

    def ll_full(th: np.ndarray) -> float:
        return -negll(th) - n * K / 2 * (1 + np.log(2 * np.pi))

    H = np.empty((k, k))
    h = 1e-4 * np.maximum(np.abs(theta), 0.1)
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = h[i], h[j]
            H[i, j] = H[j, i] = (ll_full(theta + ei + ej) - ll_full(theta + ei - ej) - ll_full(theta - ei + ej)
                                 + ll_full(theta - ei - ej)) / (4 * h[i] * h[j])
    cov = np.linalg.pinv(-H)
    c, A, M = _unpack(theta, K, p, q)
    return FittedVARMA(tuple(variables), p, q, c, A, M, sigma, cov, float(ll_full(theta)), n,
                       pd.DataFrame(E, index=frame.index[p:], columns=variables))
