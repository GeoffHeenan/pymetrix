"""Parameter uncertainty for Markov-switching VARs: the maximum-likelihood estimates' asymptotic covariance and draws
from their asymptotic normal distribution, behind the bands of ``FittedMSVAR.irf``, ``fevd`` and
``historical_decomposition``.

The parameters are put on an unrestricted scale: the intercepts and lag coefficients as they are, each covariance
matrix by its Cholesky factor with the log of its diagonal (so every draw is a valid covariance matrix), and each row
of the transition matrix by the logits of its first M - 1 entries against the last (so every draw is a valid
transition matrix). The covariance of the estimates on that scale is the inverse of the observed information, minus
the Hessian of the exact log likelihood (the Hamilton filter's, the first regime from the chain's ergodic
distribution) at the estimates, by central differences. Draws are theta_d = theta_hat + chol(V) z_d, z_d standard
normal, mapped back (a transition row with a probability at 0 or 1, below 1e-6, is on the boundary of the
parameter space, where the likelihood is flat in its logits: it is held at its estimates); a function of the parameters (a response, a variance share, a decomposition) gets bands from
the pointwise quantiles of its values over the draws. This is the parametric (Krinsky-Robb) analogue of the delta
method, without its linearisation, and the frequentist counterpart of the posterior draws a Bayesian VAR's bands use.
It does not refit the model (a bootstrap would rerun EM from several starts each draw, and draws of the regime
labels could switch); the labels are those of the estimates, which is sound when the regimes are well separated
(when they are not, the information matrix is close to singular and the bands say so by their width, or are refused).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._random import generator as make_rng
from .errors import SpecificationError

if TYPE_CHECKING:
    from .ms_var import FittedMSVAR

__all__ = ["boundary_rows", "loglike_batch", "pack", "parameter_covariance", "parameter_draws", "unpack"]


def _layout(fit: FittedMSVAR) -> tuple[int, int, int, int, int]:
    """(M, K, K p, regimes with their own lag coefficients, regimes with their own covariance)."""
    M, K = fit.spec.regimes, len(fit.variables)
    nB = M if fit.spec.switching == "all" else 1
    nS = 1 if fit.spec.switching == "intercept" else M
    return M, K, K * fit.spec.lags, nB, nS


def pack(fit: FittedMSVAR) -> np.ndarray:
    """The estimates on the unrestricted scale: intercepts, lag coefficients (once if common), Cholesky factors
    (lower triangle by rows, log diagonal; once if common) and the transition matrix's logits."""
    _, K, _, nB, nS = _layout(fit)
    tri = np.tril_indices(K)
    parts = [fit.intercepts.ravel(), fit.coefs[:nB].reshape(-1)]
    for S in fit.sigmas[:nS]:
        L = np.linalg.cholesky(S)
        L[np.diag_indices(K)] = np.log(np.diag(L))
        parts.append(L[tri])
    lp = np.log(np.maximum(fit.transition, 1e-300))
    parts.append((lp[:, :-1] - lp[:, -1:]).ravel())
    return np.concatenate(parts)


def unpack(theta: np.ndarray, fit: FittedMSVAR) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draws (N, n) back to intercepts (N, M, K), lag coefficients (N, M, K, K p), Cholesky factors (N, M, K, K) and
    transition matrices (N, M, M)."""
    theta = np.atleast_2d(theta)
    N = len(theta)
    M, K, kx, nB, nS = _layout(fit)
    i = 0

    def take(size: int) -> np.ndarray:
        nonlocal i
        out = theta[:, i:i + size]
        i += size
        return out

    C = take(M * K).reshape(N, M, K)
    B = np.broadcast_to(take(nB * K * kx).reshape(N, nB, K, kx), (N, M, K, kx))
    tri = np.tril_indices(K)
    L = np.zeros((N, nS, K, K))
    for j in range(nS):
        L[:, j][:, tri[0], tri[1]] = take(len(tri[0]))
    d = np.arange(K)
    L[:, :, d, d] = np.exp(L[:, :, d, d])
    L = np.broadcast_to(L, (N, M, K, K))
    z = np.concatenate([take(M * (M - 1)).reshape(N, M, M - 1), np.zeros((N, M, 1))], axis=2)
    z = np.exp(z - z.max(axis=2, keepdims=True))
    return C, B, L, z / z.sum(axis=2, keepdims=True)


def _data(fit: FittedMSVAR) -> tuple[np.ndarray, np.ndarray]:
    if fit.sample_rows is None:
        raise SpecificationError("this Markov-switching VAR was fitted before bands were available: fit it again")
    from .ms_var import _design

    Yall = fit.sample_rows.to_numpy(dtype=float)
    if fit.spec.lags:
        return _design(Yall, fit.spec.lags)
    return Yall, np.zeros((len(Yall), 0))


def loglike_batch(theta: np.ndarray, fit: FittedMSVAR, chunk: int | None = None) -> np.ndarray:
    """The exact log likelihood (Hamilton filter, the first regime from the ergodic distribution) at each row of
    ``theta`` (N, n), the rows run together in chunks."""
    from .ms_var_em import loglike_theta

    return loglike_theta(theta, fit, *_data(fit), chunk)


def _free(fit: FittedMSVAR) -> np.ndarray:
    """Which of ``pack``'s parameters are drawn: all but the logits of a transition row with a probability at the
    boundary (below 1e-6), where the likelihood is flat in the logit and has no asymptotic normal distribution."""
    M = fit.spec.regimes
    free = np.ones(len(pack(fit)), dtype=bool)
    nP = M * (M - 1)
    rows = np.repeat(fit.transition.min(axis=1) < 1e-6, M - 1)
    free[len(free) - nP:] = ~rows
    return free


def boundary_rows(fit: FittedMSVAR) -> list[int]:
    """The transition rows held at their estimates in the draws (a probability below 1e-6)."""
    return [int(i) for i in np.flatnonzero(fit.transition.min(axis=1) < 1e-6)]


def parameter_covariance(fit: FittedMSVAR) -> np.ndarray:
    """The estimates' asymptotic covariance on the unrestricted scale (``pack``'s order): the inverse of minus the
    Hessian of the exact log likelihood at the estimates, by central differences. A transition row with a probability
    at the boundary (below 1e-6) is held at its estimates: its rows and columns are zero."""
    theta = pack(fit)
    free = np.flatnonzero(_free(fit))
    n = len(free)
    h = 1e-4 * np.maximum(np.abs(theta[free]), 1.0)
    E = np.zeros((n, len(theta)))
    E[np.arange(n), free] = h
    points = [theta]
    for i in range(n):
        points += [theta + E[i], theta - E[i]]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for i, j in pairs:
        points += [theta + E[i] + E[j], theta + E[i] - E[j], theta - E[i] + E[j], theta - E[i] - E[j]]
    f = loglike_batch(np.array(points), fit)
    f0 = f[0]
    H = np.empty((n, n))
    for i in range(n):
        H[i, i] = (f[1 + 2 * i] - 2 * f0 + f[2 + 2 * i]) / h[i] ** 2
    base = 1 + 2 * n
    for k, (i, j) in enumerate(pairs):
        pp, pm, mp, mm = f[base + 4 * k:base + 4 * k + 4]
        H[i, j] = H[j, i] = (pp - pm - mp + mm) / (4 * h[i] * h[j])
    info = -(H + H.T) / 2
    eig = np.linalg.eigvalsh(info)
    if not np.all(np.isfinite(eig)) or eig.min() <= 1e-8 * max(eig.max(), 1.0):
        raise SpecificationError("the log likelihood is flat (or not at a maximum) in some direction at the estimates (regimes "
                                 "that barely differ, or a regime hardly ever in effect), so its parameters have no asymptotic "
                                 "covariance and there are no bands: try fewer regimes or switching parameters")
    V = np.zeros((len(theta), len(theta)))
    V[np.ix_(free, free)] = np.linalg.inv(info)
    return V


def parameter_draws(fit: FittedMSVAR, draws: int, seed: int = 0) -> dict[str, np.ndarray]:
    """``draws`` parameter sets from the estimates' asymptotic normal distribution on the unrestricted scale, mapped
    back: ``intercepts`` (N, M, K), ``coefs`` (N, M, K, K p), ``sigmas`` (N, M, K, K), ``transition`` (N, M, M)."""
    if int(draws) < 2:
        raise SpecificationError(f"bands need at least 2 draws; got {draws}")
    V = fit.parameter_covariance()
    theta = pack(fit)
    free = np.flatnonzero(_free(fit))
    z = make_rng(seed).standard_normal((int(draws), len(free)))
    theta_d = np.tile(theta, (int(draws), 1))
    theta_d[:, free] += z @ np.linalg.cholesky(V[np.ix_(free, free)]).T
    C, B, L, P = unpack(theta_d, fit)
    return {"intercepts": C, "coefs": np.array(B), "sigmas": L @ np.transpose(L, (0, 1, 3, 2)), "transition": P}


def quantile_bands(values: np.ndarray, coverage: float) -> tuple[np.ndarray, np.ndarray]:
    """The pointwise (1 - coverage) / 2 and (1 + coverage) / 2 quantiles over the draws (axis 0)."""
    lo, hi = np.quantile(values, [(1 - coverage) / 2, (1 + coverage) / 2], axis=0)
    return lo, hi


def check_coverage(coverage: float) -> None:
    if not 0 < coverage < 1:
        raise SpecificationError(f"coverage is between 0 and 1, e.g. 0.68; got {coverage}")


def band_source(fit: FittedMSVAR, draws: int) -> str:
    rows = boundary_rows(fit)
    held = f" (transition rows {', '.join(map(str, rows))} at the boundary held fixed)" if rows else ""
    return f"{int(draws)} draws of the parameters from the estimates' asymptotic normal distribution{held}"
