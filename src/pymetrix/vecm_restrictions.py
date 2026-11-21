"""Likelihood-ratio tests of restrictions on a VECM's cointegrating vectors and adjustment coefficients (Johansen 1991,
1995; Johansen and Juselius 1990, 1992).

    vecm.test_beta([[1, 0], [-1, 0], [0, 1]])     # beta = H phi: here the first two variables enter as y1 - y2
    vecm.test_weak_exogeneity(["li"])              # alpha's row for li is zero: li does not adjust to the relations

Both re-solve Johansen's reduced-rank regression under the restriction. With S00, S01 and S11 the moment matrices of
the residuals R0 (differences) and R1 (lagged levels, with any restricted constant or trend) after partialling out
the lagged differences and unrestricted deterministic terms, the unrestricted eigenvalues solve
|lambda S11 - S10 S00^-1 S01| = 0. Under beta = H phi they solve |lambda H'S11H - H'S10 S00^-1 S01 H| = 0; under
alpha = A psi (B = A-perp, the weakly exogenous directions) the same problem conditional on B'R0. The statistic is
T sum_{i <= r} log((1 - lambda*_i) / (1 - lambda_i)), chi-squared with r(p1 - s) and r m degrees of freedom.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg

from pymetrix._lazy import stats as sps

from .errors import SpecificationError

__all__ = ["ReducedRank", "reduced_rank", "test_beta", "test_weak_exogeneity"]


@dataclass(frozen=True, eq=False)
class ReducedRank:
    """Johansen's reduced-rank regression on a VECM's sample."""

    R0: np.ndarray
    R1: np.ndarray
    S00: np.ndarray
    S01: np.ndarray
    S11: np.ndarray
    eigenvalues: np.ndarray
    vectors: np.ndarray  # columns: the eigenvectors, normalised v' S11 v = 1, by decreasing eigenvalue
    rows: tuple[str, ...]  # the names of R1's columns: the variables, then a restricted constant or trend

    @property
    def T(self) -> int:
        return len(self.R0)


def _eig(M: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generalised eigenvalues of M v = lambda S v (S positive definite), decreasing, with v' S v = 1."""
    w, v = linalg.eigh((M + M.T) / 2, (S + S.T) / 2)
    order = np.argsort(w)[::-1]
    return w[order], v[:, order]


def reduced_rank(fitted: Any) -> ReducedRank:
    """The moment matrices of a fitted VECM's sample (statsmodels' conventions for the deterministic cases)."""
    spec = fitted.spec
    if spec.exog or spec.exog_coint or spec.seasons:
        raise SpecificationError("restriction tests here take VECMs without exogenous terms or seasonal dummies")
    y = np.asarray(fitted.results.model.endog, float)  # the levels, the presample included
    p = spec.lags
    dy = np.diff(y, axis=0)
    T = len(dy) - p
    Z0 = dy[p:]
    Z1 = y[p:-1]
    lagged = [dy[p - j:len(dy) - j] for j in range(1, p + 1)]
    det = spec.deterministic
    t = np.arange(p + 1, p + 1 + T, dtype=float)  # statsmodels' trend counts from the first observation
    names = list(fitted.variables)
    if det == "ci":
        Z1 = np.column_stack([Z1, np.ones(T)])
        names.append("const")
    elif det == "coli":
        Z1 = np.column_stack([Z1, t])
        names.append("trend")
    unrestricted = list(lagged)
    if det in ("co", "coli", "colo"):
        unrestricted.append(np.ones((T, 1)))
    if det == "colo":
        unrestricted.append(t[:, None])
    if unrestricted:
        Z2 = np.column_stack(unrestricted)
        P = Z2 @ np.linalg.pinv(Z2)
        R0, R1 = Z0 - P @ Z0, Z1 - P @ Z1
    else:
        R0, R1 = Z0, Z1
    S00, S01, S11 = R0.T @ R0 / T, R0.T @ R1 / T, R1.T @ R1 / T
    lam, v = _eig(S01.T @ np.linalg.solve(S00, S01), S11)
    return ReducedRank(R0, R1, S00, S01, S11, np.clip(lam, 0, 1 - 1e-15), v, tuple(names))


def _matrix(M: Any, rows: int, what: str) -> np.ndarray:
    M = np.atleast_2d(np.asarray(M, float))
    if M.shape[0] != rows:
        M = M.T if M.shape[1] == rows else M
    if M.shape[0] != rows:
        raise SpecificationError(f"{what} needs {rows} rows, one for each of the cointegrating relations' terms")
    return M


def _result(name: str, null: str, stat: float, df: int, details: dict[str, Any]) -> Any:
    from .stats.base import TestResult

    return TestResult(name, null, float(stat), df, float(sps.chi2.sf(stat, df)), "chi2", details=details)


def test_beta(fitted: Any, H: Any) -> Any:
    """Johansen's test that every cointegrating vector lies in the column space of ``H`` (beta = H phi), p1 x s with p1
    the variables (plus a restricted constant or trend): chi-squared(r (p1 - s)). The restricted beta is in details."""
    rr = reduced_rank(fitted)
    r = fitted.rank
    H = _matrix(H, len(rr.rows), "H")
    s = H.shape[1]
    if np.linalg.matrix_rank(H) < s:
        raise SpecificationError("H's columns must be linearly independent")
    if s < r:
        raise SpecificationError(f"H has {s} columns but there are {r} cointegrating relations: give at least {r}")
    lam_star, phi = _eig(H.T @ rr.S01.T @ np.linalg.solve(rr.S00, rr.S01) @ H, H.T @ rr.S11 @ H)
    lam_star = np.clip(lam_star, 0, 1 - 1e-15)
    stat = rr.T * np.sum(np.log((1 - lam_star[:r]) / (1 - rr.eigenvalues[:r])))
    beta = H @ phi[:, :r]
    beta = beta @ np.linalg.inv(beta[:r, :r]) if abs(np.linalg.det(beta[:r, :r])) > 1e-12 else beta
    frame = pd.DataFrame(beta, index=list(rr.rows), columns=[f"ce{i + 1}" for i in range(r)])
    return _result("LR test of restrictions on beta", "beta = H phi", max(stat, 0.0), r * (len(rr.rows) - s),
                   {"restricted beta": frame, "eigenvalues": lam_star[:r].tolist(), "unrestricted eigenvalues": rr.eigenvalues[:r].tolist()})


def test_weak_exogeneity(fitted: Any, variables: Sequence[str]) -> Any:
    """Johansen's test that the ``variables`` are weakly exogenous for the cointegrating vectors: their rows of alpha
    are zero (alpha = A psi). Chi-squared(r m), m the number of variables."""
    names = list(fitted.variables)
    variables = [variables] if isinstance(variables, str) else list(variables)
    unknown = [v for v in variables if v not in names]
    if unknown:
        raise SpecificationError(f"{', '.join(unknown)} are not variables of the VECM ({', '.join(names)})")
    K, r, m = len(names), fitted.rank, len(variables)
    if K - m < r:
        raise SpecificationError(f"with {m} weakly exogenous variables only {K - m} can adjust, fewer than the rank {r}")
    rr = reduced_rank(fitted)
    B = np.zeros((K, m))
    for j, v in enumerate(variables):
        B[names.index(v), j] = 1.0
    A = np.delete(np.eye(K), [names.index(v) for v in variables], axis=1)
    Sbb = B.T @ rr.S00 @ B
    proj = rr.S00 @ B @ np.linalg.solve(Sbb, B.T)  # conditioning on B'R0
    Saa_b = A.T @ (rr.S00 - proj @ rr.S00) @ A
    Sa1_b = A.T @ (rr.S01 - proj @ rr.S01)
    S11_b = rr.S11 - rr.S01.T @ B @ np.linalg.solve(Sbb, B.T @ rr.S01)
    lam_star, _ = _eig(Sa1_b.T @ np.linalg.solve(Saa_b, Sa1_b), S11_b)
    lam_star = np.clip(lam_star, 0, 1 - 1e-15)
    stat = rr.T * np.sum(np.log((1 - lam_star[:r]) / (1 - rr.eigenvalues[:r])))
    return _result("LR test of weak exogeneity", f"{', '.join(variables)} weakly exogenous (alpha rows zero)", max(stat, 0.0), r * m,
                   {"eigenvalues": lam_star[:r].tolist(), "unrestricted eigenvalues": rr.eigenvalues[:r].tolist()})
