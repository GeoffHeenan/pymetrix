"""Structural VARs identified by restrictions on A and B, estimated by maximum likelihood (Amisano and Giannini 1997).

    var.identify(AB(A=[[1, 0, 0], ["*", 1, 0], ["*", "*", 1]], B=[["*", 0, 0], [0, "*", 0], [0, 0, "*"]]))
    var.identify(AB(B=[["*", 0, "*"], ["*", "*", "*"], ["*", "*", "*"]], long_run=[["*", 0, 0], ["*", "*", "*"], ["*", "*", "*"]]))
    var.structural()        # A and B with standard errors, the log likelihood and the over-identification test

The reduced-form residuals u_t and structural shocks e_t (independent, unit variance) are related by A u_t = B e_t,
so the impact matrix is A^-1 B. Entries given as numbers are fixed; ``"*"`` marks a free one. ``long_run`` restricts
the long-run effects (I - A_1 - ... - A_p)^-1 A^-1 B: zeros there, with A the identity, are linear restrictions on
B, which combine short- and long-run identification (Galí 1999). The free parameters maximise

    log L = -T/2 [K log 2 pi + log |B|^2 - log |A|^2 + tr(A' B'^-1 B^-1 A Sigma)]

given the least-squares Sigma. The order condition (no more free parameters than K(K + 1)/2) and the rank condition
(the Jacobian of vech(Sigma) in the free parameters of full column rank, at random values) are checked. With fewer
free parameters than K(K + 1)/2, a likelihood-ratio test of the over-identifying restrictions is reported. Columns
are signed so that the impact matrix has a positive diagonal, where B's fixed entries allow it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize

from pymetrix._lazy import stats as sps

from ._random import generator as make_rng
from .errors import SpecificationError
from .output import Table

__all__ = ["AB", "StructuralEstimate"]

FREE = "*"


@dataclass(frozen=True)
class AB:
    """Identification by restrictions on A and B (A u = B e); ``"*"`` marks a free entry, numbers are fixed. A is the
    identity and B entirely free unless given; ``long_run`` sets zeros (or other fixed values) of the long-run
    effects, with A the identity."""

    A: Sequence[Sequence[Any]] | None = None
    B: Sequence[Sequence[Any]] | None = None
    long_run: Sequence[Sequence[Any]] | None = None
    starts: int = 10
    seed: int = 0


def _pattern(M: Any, K: int, default: str, what: str) -> tuple[np.ndarray, np.ndarray]:
    """(values, free mask) of a K x K pattern; ``default`` is "identity" or "free"."""
    if M is None:
        if default == "identity":
            return np.eye(K), np.zeros((K, K), bool)
        return np.zeros((K, K)), np.ones((K, K), bool)
    rows = [list(r) for r in M]
    if len(rows) != K or any(len(r) != K for r in rows):
        raise SpecificationError(f"{what} must be {K} x {K}, one row and column for each variable")
    free = np.array([[isinstance(x, str) and x.strip() == FREE for x in r] for r in rows])
    vals = np.zeros((K, K))
    for i, r in enumerate(rows):
        for j, x in enumerate(r):
            if not free[i, j]:
                try:
                    vals[i, j] = float(x)
                except (TypeError, ValueError):
                    raise SpecificationError(f"{what}[{i}][{j}] is {x!r}: a number, or '*' for a free entry") from None
    return vals, free


@dataclass(frozen=True, eq=False)
class _Parametrisation:
    """theta -> (A, B): free entries of A, then the free entries of B through b = b0 + N gamma (the long-run
    restrictions' null space)."""

    A0: np.ndarray
    A_free: np.ndarray
    B0: np.ndarray
    B_free: np.ndarray
    b0: np.ndarray
    N: np.ndarray

    @property
    def size(self) -> int:
        return int(self.A_free.sum()) + self.N.shape[1]

    def matrices(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        na = int(self.A_free.sum())
        A = self.A0.copy()
        A[self.A_free] = theta[:na]
        B = self.B0.copy()
        B[self.B_free] = self.b0 + self.N @ theta[na:]
        return A, B

    def free_values(self, theta: np.ndarray) -> np.ndarray:
        """The free entries of A and B (in that order) given theta: a linear map, J theta + c."""
        na = int(self.A_free.sum())
        return np.concatenate([theta[:na], self.b0 + self.N @ theta[na:]])

    def free_jacobian(self) -> np.ndarray:
        na = int(self.A_free.sum())
        nb = int(self.B_free.sum())
        J = np.zeros((na + nb, self.size))
        J[:na, :na] = np.eye(na)
        J[na:, na:] = self.N
        return J


def _parametrise(scheme: AB, K: int, lags: np.ndarray) -> _Parametrisation:
    A0, A_free = _pattern(scheme.A, K, "identity", "A")
    B0, B_free = _pattern(scheme.B, K, "free", "B")
    nb = int(B_free.sum())
    if scheme.long_run is None:
        return _Parametrisation(A0, A_free, B0, B_free, np.zeros(nb), np.eye(nb))
    if A_free.any() or not np.allclose(A0, np.eye(K)):
        raise SpecificationError("long-run restrictions here take A as the identity: restrict B and long_run")
    L0, L_free = _pattern(scheme.long_run, K, "free", "long_run")
    Xi = np.linalg.inv(np.eye(K) - lags.sum(axis=0))  # long-run effects Xi B
    rows, rhs = [], []
    where = np.argwhere(B_free)  # row-major, as B[B_free] orders them
    for i, j in np.argwhere(~L_free):
        r = np.zeros(nb)
        for n, (k, jj) in enumerate(where):
            if jj == j:
                r[n] = Xi[i, k]
        rows.append(r)
        rhs.append(L0[i, j] - Xi[i] @ np.where(B_free[:, j], 0.0, B0[:, j]))
    R, c = np.array(rows), np.array(rhs)
    b0 = np.linalg.lstsq(R, c, rcond=None)[0]
    if not np.allclose(R @ b0, c, atol=1e-10):
        raise SpecificationError("the long-run restrictions contradict the fixed entries of B")
    _, s, Vt = np.linalg.svd(R)
    rank = int((s > 1e-10 * max(s.max(), 1)).sum())
    return _Parametrisation(A0, A_free, B0, B_free, b0, Vt[rank:].T)


def _loglike(par: _Parametrisation, theta: np.ndarray, sigma: np.ndarray, T: int) -> float:
    A, B = par.matrices(theta)
    sa, la = np.linalg.slogdet(A)
    sb, lb = np.linalg.slogdet(B)
    if sa == 0 or sb == 0:
        return -1e300
    W = np.linalg.solve(B, A)
    K = len(sigma)
    return float(-T / 2 * (K * np.log(2 * np.pi) + 2 * lb - 2 * la + np.trace(W.T @ W @ sigma)))


def _vech_jacobian(par: _Parametrisation, theta: np.ndarray) -> np.ndarray:
    K = par.A0.shape[0]
    idx = np.tril_indices(K)

    def implied(th: np.ndarray) -> np.ndarray:
        A, B = par.matrices(th)
        P = np.linalg.solve(A, B)
        return (P @ P.T)[idx]

    J = np.empty((len(idx[0]), par.size))
    for j in range(par.size):
        h = 1e-6
        up, dn = theta.copy(), theta.copy()
        up[j] += h
        dn[j] -= h
        J[:, j] = (implied(up) - implied(dn)) / (2 * h)
    return J


def _check_identified(par: _Parametrisation, K: int, rng: np.random.Generator) -> None:
    m = K * (K + 1) // 2
    if par.size > m:
        raise SpecificationError(f"{par.size} free parameters but Sigma has only {m} distinct entries: not identified "
                                 "(restrict more entries of A, B or the long run)")
    for _ in range(3):
        theta = rng.normal(size=par.size)
        A, B = par.matrices(theta)
        if abs(np.linalg.det(A)) > 1e-6 and abs(np.linalg.det(B)) > 1e-6:
            break
    if np.linalg.matrix_rank(_vech_jacobian(par, theta), tol=1e-7) < par.size:
        raise SpecificationError("the restrictions fail the rank condition: some parameters are not identified")


def _sign(par: _Parametrisation, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip columns of B (shock signs) so the impact matrix's diagonal is positive, where B's fixed entries are zero."""
    P = np.linalg.solve(A, B)
    B = B.copy()
    for j in range(B.shape[1]):
        fixed_nonzero = np.any((~par.B_free[:, j]) & (par.B0[:, j] != 0))
        if P[j, j] < 0 and not fixed_nonzero:
            B[:, j] = -B[:, j]
    return A, B


def _hessian(f: Any, x: np.ndarray) -> np.ndarray:
    n = len(x)
    H = np.empty((n, n))
    h = 1e-4 * np.maximum(np.abs(x), 1e-2)
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i], ej[j] = h[i], h[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


def _start(par: _Parametrisation, sigma: np.ndarray, rng: np.random.Generator, k: int) -> np.ndarray:
    """A starting theta: near A = I and B = chol(Sigma) (projected on the free entries), then perturbed."""
    target_B = np.linalg.cholesky(sigma)
    na = int(par.A_free.sum())
    a = np.eye(par.A0.shape[0])[par.A_free]
    b = target_B[par.B_free]
    gamma = np.linalg.lstsq(par.N, b - par.b0, rcond=None)[0] if par.N.shape[1] else np.zeros(0)
    theta = np.concatenate([a, gamma])
    if k:
        theta = theta + rng.normal(scale=0.5 * np.sqrt(np.mean(np.diag(sigma))), size=len(theta)) * np.r_[np.full(na, 0.5), np.ones(len(gamma))]
    return theta


@dataclass(frozen=True, eq=False)
class StructuralEstimate:
    """A and B estimated by maximum likelihood, with standard errors and the over-identification test."""

    variables: tuple[str, ...]
    A: np.ndarray
    B: np.ndarray
    A_se: np.ndarray = field(repr=False)
    B_se: np.ndarray = field(repr=False)
    loglike: float = 0.0
    LR: float | None = None
    df: int = 0
    nobs: int = 0
    converged: bool = True

    stage = "estimate"

    @property
    def impact(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.B)

    def summary(self) -> Table:
        rows, index = [], []
        for name, M, S in (("A", self.A, self.A_se), ("B", self.B, self.B_se)):
            for i, j in np.argwhere(~np.isnan(S)):
                z = M[i, j] / S[i, j] if S[i, j] > 0 else np.nan
                rows.append([float(M[i, j]), float(S[i, j]), float(z), float(2 * sps.norm.sf(abs(z)))])
                index.append(f"{name}[{self.variables[i]}, {self.variables[j] if name == 'A' else f'shock {j + 1}'}]")
        notes = ["Method: A u = B e, by maximum likelihood", f"Observations: {self.nobs}", f"Log likelihood: {self.loglike:.4f}"]
        if self.LR is not None:
            notes.append(f"Over-identifying restrictions: LR = {self.LR:.4f}, chi-squared({self.df}), "
                         f"p = {sps.chi2.sf(self.LR, self.df):.4f}.")
        else:
            notes.append("Exactly identified: no restrictions to test.")
        if not self.converged:
            notes.append("The optimiser did not converge.")
        return Table(["estimate", "std err", "z", "p-value"], rows, index=index, index_name="parameter",
                     title="Structural VAR (A-B model)", notes=tuple(notes))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.summary().to_frame()

    def __repr__(self) -> str:
        return f"<StructuralEstimate A-B, log likelihood {self.loglike:.4f}>"


def estimate_ab(scheme: AB, sigma: np.ndarray, lags: np.ndarray, T: int, variables: Sequence[str],
                standard_errors: bool = True) -> StructuralEstimate:
    """Maximum likelihood estimates of A and B given the reduced-form covariance ``sigma`` and lag matrices (without
    ``standard_errors``, for the impact matrix alone, as the bootstrap needs)."""
    K = len(sigma)
    rng = make_rng(scheme.seed)
    par = _parametrise(scheme, K, lags)
    _check_identified(par, K, rng)
    f = lambda th: -_loglike(par, th, sigma, T) / T  # noqa: E731
    best = None
    for k in range(max(scheme.starts, 1)):
        found = optimize.minimize(f, _start(par, sigma, rng, k), method="BFGS", options={"gtol": 1e-10, "maxiter": 5000})
        if best is None or found.fun < best.fun - 1e-12:
            best = found
        A, B = par.matrices(best.x)
        P = np.linalg.solve(A, B)
        if par.size == K * (K + 1) // 2 and np.allclose(P @ P.T, sigma, rtol=1e-8, atol=1e-12):
            break  # just identified and Sigma reproduced: the global maximum
    if standard_errors or not best.success:  # the bootstrap's draws need no polish once BFGS has converged
        polish = optimize.minimize(f, best.x, method="Nelder-Mead", options={"xatol": 1e-11, "fatol": 1e-14, "maxiter": 40000})
        theta = polish.x if polish.fun <= best.fun else best.x
    else:
        polish = best
        theta = best.x
    A, B = par.matrices(theta)
    A, B = _sign(par, A, B)
    # re-express theta after the sign flips (B's columns), for the standard errors
    na = int(par.A_free.sum())
    gamma = np.linalg.lstsq(par.N, B[par.B_free] - par.b0, rcond=None)[0] if par.N.shape[1] else np.zeros(0)
    theta = np.concatenate([A[par.A_free], gamma])
    cov_theta = np.full((par.size, par.size), np.nan)
    if standard_errors:
        try:
            cov_theta = np.linalg.inv(-_hessian(lambda th: _loglike(par, th, sigma, T), theta))
        except np.linalg.LinAlgError:
            pass
    J = par.free_jacobian()
    se_free = np.sqrt(np.clip(np.diag(J @ cov_theta @ J.T), 0, None))
    A_se = np.full((K, K), np.nan)
    B_se = np.full((K, K), np.nan)
    A_se[par.A_free] = se_free[:na]
    B_se[par.B_free] = se_free[na:]
    ll = _loglike(par, theta, sigma, T)
    m = K * (K + 1) // 2
    unrestricted = -T / 2 * (K * np.log(2 * np.pi) + np.linalg.slogdet(sigma)[1] + K)
    LR = 2 * (unrestricted - ll) if par.size < m else None
    P = np.linalg.solve(A, B)
    converged = bool(np.allclose(P @ P.T, sigma, rtol=1e-4, atol=1e-8)) if LR is None else bool(best.success or polish.success)
    return StructuralEstimate(tuple(variables), A, B, A_se, B_se, float(ll), None if LR is None else float(max(LR, 0.0)),
                              m - par.size, T, converged)
