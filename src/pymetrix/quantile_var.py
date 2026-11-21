"""Quantile VARs (White, Kim and Manganelli 2015; Chavleishvili and Manganelli 2024), recursive form.

    qv = QuantileVAR(["gdp", "fci"], lags=2, quantile=0.1).fit(data)
    qv.summary(); qv.irf(12).plot()                  # responses of the system at its 10% quantile
    qv.fevd(12)                                      # quantile variance decomposition (Ando, Greenwood-Nimmo and Shin)

Each equation is a quantile regression at ``quantile`` of a variable on the variables ordered before it in the same
period and on p lags of all of them, so the system is recursive like a Cholesky VAR: A_0 y_t = c + A_1 y_t-1 + ... with
A_0 lower triangular with ones on its diagonal. The quantile regressions are linear programmes, solved by scipy's
HiGHS. Responses (Montes-Rojas 2019) are those of this quantile system to a one-unit structural shock in each
equation: the path of the tau-quantiles if every later period stays at its tau-quantile. Comparing quantiles shows
whether bad outcomes respond differently from typical ones.

Variance decompositions follow quantile connectedness (Ando, Greenwood-Nimmo and Shin 2022): the quantile system's
moving-average weights with the covariance of its quantile residuals, here orthogonalised by Cholesky. There is no
historical decomposition: the quantile residuals are not centred (at the 0.1 quantile nine in ten are positive), so
splitting the data into a "baseline" quantile path and shock contributions mixes location with shocks, and the
literature has no established version.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import optimize

from ._api import api
from .errors import SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["FittedQuantileVAR", "QuantileVAR", "quantile_regression"]


def quantile_regression(y: np.ndarray, X: np.ndarray, q: float) -> np.ndarray:
    """argmin_b sum_t rho_q(y_t - x_t'b), as the linear programme min q 1'u + (1 - q) 1'v s.t. X b + u - v = y, u, v >= 0."""
    n, k = X.shape
    c = np.r_[np.zeros(k), np.full(n, q), np.full(n, 1 - q)]
    A = np.hstack([X, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * k + [(0, None)] * (2 * n)
    res = optimize.linprog(c, A_eq=A, b_eq=y, bounds=bounds, method="highs")
    if not res.success:
        raise SpecificationError(f"the quantile regression failed: {res.message}")
    return res.x[:k]


@dataclass(frozen=True, eq=False)
class QuantileVAR:
    """A recursive quantile VAR at one quantile for every equation."""

    variables: Sequence[str]
    lags: int = 1
    quantile: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", tuple(self.variables))
        if not 0 < self.quantile < 1:
            raise SpecificationError("the quantile is between 0 and 1")
        if self.lags < 1:
            raise SpecificationError("a quantile VAR needs at least one lag")

    @api("pymetrix:quantile_var.fit")
    def fit(self, data: Dataset, span: Any = None) -> FittedQuantileVAR:
        if data.kind != "time_series":
            raise SpecificationError("a quantile VAR needs a single time series")
        missing = [v for v in self.variables if v not in data]
        if missing:
            raise SpecificationError(f"{', '.join(missing)} are not series of the data")
        frame = data.frame[list(self.variables)].astype(float)
        if span is not None:
            frame = frame[data.sample_mask(span)]
        frame = frame.dropna()
        Yall = frame.to_numpy()
        T0, K = Yall.shape
        p = self.lags
        Y = Yall[p:]
        L = np.column_stack([Yall[p - l:T0 - l] for l in range(1, p + 1)])
        T = len(Y)
        A0 = np.eye(K)
        c = np.empty(K)
        A = np.zeros((p, K, K))
        for i in range(K):
            X = np.column_stack([np.ones(T), Y[:, :i], L])
            b = quantile_regression(Y[:, i], X, self.quantile)
            c[i] = b[0]
            A0[i, :i] = -b[1:1 + i]
            lagb = b[1 + i:]
            for l in range(p):
                A[l, i] = lagb[K * l:K * (l + 1)]
        return FittedQuantileVAR(self, c, A0, A, frame.index[p:], T, frame)


@dataclass(frozen=True, eq=False)
class FittedQuantileVAR:
    """An estimated recursive quantile VAR: A_0 y_t = c + sum_l A_l y_t-l at the quantile."""

    spec: QuantileVAR
    intercepts: np.ndarray
    A0: np.ndarray = field(repr=False)
    A: np.ndarray = field(repr=False)  # (lags, K, K)
    index: pd.Index = field(repr=False)
    nobs: int = 0
    sample_rows: pd.DataFrame | None = field(default=None, repr=False)  # the data fitted, with the first lags observations (not `observations`: that is the tidy-table verb)

    stage = "estimate"

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self.spec.variables)

    @api()
    def irf(self, periods: int = 20) -> Any:
        """Responses of the quantile system to a one-unit structural shock in each equation."""
        from .var import IRF, _ma

        A0inv = np.linalg.inv(self.A0)
        reduced = np.stack([A0inv @ self.A[l] for l in range(self.spec.lags)])
        values = _ma(reduced, periods) @ A0inv
        return IRF(values, self.variables, self.variables, None, None, False,
                   f"recursive quantile VAR at the {self.spec.quantile:g} quantile (unit shocks)")

    def residuals(self) -> pd.DataFrame:
        """The reduced-form quantile residuals u_t = y_t - A_0^-1 (c + sum_l A_l y_t-l): each variable less the
        quantile system's prediction (at quantile 0.5 about as many positive as negative; at 0.1 mostly positive)."""
        if self.sample_rows is None:
            raise SpecificationError("this quantile VAR was fitted before residuals were kept: fit it again")
        p = self.spec.lags
        Yall = self.sample_rows.to_numpy(dtype=float)
        T = len(Yall) - p
        A0inv = np.linalg.inv(self.A0)
        rhs = self.intercepts + sum(Yall[p - 1 - l:p - 1 - l + T] @ self.A[l].T for l in range(p))
        return pd.DataFrame(Yall[p:] - rhs @ A0inv.T, index=self.index, columns=list(self.variables))

    @api()
    def fevd(self, periods: int = 20) -> Any:
        """Forecast-error variance decomposition of the quantile system (Ando, Greenwood-Nimmo and Shin's quantile
        connectedness, orthogonalised as Diebold and Yilmaz 2009): the moving-average weights of the system at the
        quantile, Psi_h, and the covariance Sigma of its reduced-form quantile residuals (about their mean, divided
        by T), orthogonalised by its Cholesky factor in the VAR's order: theta_h = Psi_h chol(Sigma). Each row adds
        up to one. When the structural quantile residuals are uncorrelated, chol(Sigma) is the responses' impact
        A_0^-1 scaled by their standard deviations, so the shocks are ``irf``'s; at the median of a Gaussian VAR the
        decomposition is close to the least-squares VAR's Cholesky decomposition."""
        from .var import FEVD, _ma
        from .var_decompositions import fevd_shares

        u = self.residuals().to_numpy()
        u = u - u.mean(axis=0)
        sigma = u.T @ u / len(u)
        A0inv = np.linalg.inv(self.A0)
        reduced = np.stack([A0inv @ self.A[l] for l in range(self.spec.lags)])
        theta = _ma(reduced, periods - 1) @ np.linalg.cholesky(sigma)
        return FEVD(fevd_shares(theta), self.variables, self.variables,
                    f"Cholesky of the quantile residuals' covariance, quantile VAR at {self.spec.quantile:g}")

    def summary(self) -> Table:
        K, p = len(self.variables), self.spec.lags
        rows = []
        for i, v in enumerate(self.variables):
            rows.append([f"{v}: const", float(self.intercepts[i])])
            rows += [[f"{v}: {self.variables[j]} (same period)", float(-self.A0[i, j])] for j in range(i)]
            rows += [[f"{v}: {w}[-{l + 1}]", float(self.A[l, i, m])] for l in range(p) for m, w in enumerate(self.variables)]
        notes = (f"Quantile regressions at the {self.spec.quantile:g} quantile, recursive in the order {', '.join(self.variables)}; "
                 f"{self.nobs} observations. Standard errors are not reported: compare quantiles, or bootstrap.",)
        return Table(["", "coefficient"], rows, title=f"Quantile VAR at {self.spec.quantile:g}: {', '.join(self.variables)}", notes=notes)

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.summary().to_frame()

    def __repr__(self) -> str:
        return f"<FittedQuantileVAR {', '.join(self.variables)} at {self.spec.quantile:g}>"
