"""Zero and sign restrictions together (Arias, Rubio-Ramírez and Waggoner 2018, "Inference based on structural vector
autoregressions identified with sign and zero restrictions: theory and applications", Econometrica 86(2), 685-720).

    var.identify(SignRestrictions({"monetary": {"r": "+", "pi": "-"}}, zeros={"monetary": ["gy"]}))
    var.identify(SignRestrictions({"supply": {"gy": "+"}}, long_run_zeros={"demand": ["gy"]}))

Notation as ARW's: the structural VAR ``y_t' A0 = x_t' A+ + e_t'`` has reduced form ``B = A+ A0^-1`` and
``Sigma = (A0 A0')^-1``; with ``h(Sigma)`` the upper Cholesky factor (``h' h = Sigma``) and Q orthogonal,
``A0 = h(Sigma)^-1 Q``, so the impact responses are ``L0 = h(Sigma)' Q`` (pymetrix's impact matrix) and the long-run
responses ``Linf = (I - A_1 - ... - A_p)^-1 L0``. A zero restriction on shock j says ``Z_j F q_j = 0`` with
``F = [h(Sigma)'; (I - sum A)^-1 h(Sigma)']`` and ``Z_j`` selecting the restricted rows.

The draw (ARW's Algorithm 2): the columns are drawn in order of decreasing number of zeros ``z_j``, which must satisfy
``z_j <= n - j`` (equality throughout: exact identification). Column j is ``N_j N_j' x_j / |N_j' x_j|`` with x_j
standard normal and N_j an orthonormal basis of the null space of ``[Z_j F; q_1'; ...; q_(j-1)']``; then each column
is signed so that its first sign restriction holds, and the draw is kept if every sign holds.

This does not draw Q uniformly given the zeros; ARW's importance weight makes the kept draws come from the
normal-generalised-normal posterior conditional on the zeros (the uniform-normal-inverse-Wishart one, when there are
none): ``w = |det A0|^-(2n+m+1) / v``, where v is the volume element of the map from (A0, A+), restricted to the
manifold the zeros define, to the draw's own coordinates (B, vech Sigma, w_1, ..., w_n), ``w_j = N_j' q_j`` on its
sphere; m is the number of regressors. v is computed numerically, as ARW do, with two shortcuts that change nothing:
A+ enters the zeros and (Sigma, w) only through the sum of its lag blocks, so the other directions of A+ factor out
of v as ``|det A0|^-(m - n)`` (and, with impact zeros only, the lag sum factors out as well, ``|det A0|^-n``), which
leaves a numerical Jacobian in 2 n^2 (or n^2) coordinates. Without zeros the weights are constant (checked in the
tests). Draws are resampled in proportion to the weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .errors import SpecificationError

__all__ = ["zero_rows", "generation_order", "draw_rotation", "log_weight", "zero_sign_draws", "resample"]


def _lag_sum(A: np.ndarray) -> np.ndarray:
    return np.asarray(A).sum(axis=0)


def zero_rows(scheme: Any, sigma: np.ndarray, A: np.ndarray, variables: Sequence[str]) -> list[np.ndarray]:
    """For each shock (the scheme's shock order, then the unidentified ones), the rows ``Z_j F`` its column must be
    orthogonal to (z_j x n)."""
    n = len(variables)
    index = {v: i for i, v in enumerate(variables)}
    P0 = np.linalg.cholesky(sigma)
    out = []
    linf = None
    for shock in scheme.shock_names(variables):
        rows = [P0[index[v]] for v in scheme.zeros.get(shock, ())]
        if scheme.long_run_zeros.get(shock):
            if linf is None:
                I_A = np.eye(n) - _lag_sum(A)
                if np.linalg.cond(I_A) > 1e12:
                    raise SpecificationError("long-run zero restrictions need a VAR without unit roots: I - A(1) is singular")
                linf = np.linalg.solve(I_A, P0)
            rows += [linf[index[v]] for v in scheme.long_run_zeros[shock]]
        out.append(np.array(rows).reshape(len(rows), n))
    return out


def generation_order(counts: Sequence[int]) -> list[int]:
    """Columns in order of decreasing number of zeros (ARW's ordering), checking z_j <= n - j."""
    n = len(counts)
    order = sorted(range(n), key=lambda k: -counts[k])
    for pos, k in enumerate(order):
        if counts[k] > n - 1 - pos:
            raise SpecificationError(
                f"too many zero restrictions: with {n} variables, the shock with the most zeros may have {n - 1}, the next "
                f"{n - 2}, and so on (ordered by their number); these would over-identify the VAR")
    return order


def _null(R: np.ndarray, n: int) -> np.ndarray:
    if R.shape[0] == 0:
        return np.eye(n)
    _, s, vt = np.linalg.svd(R)
    rank = int((s > 1e-12 * max(1.0, s[0])).sum())
    return vt[rank:].T


def draw_rotation(rows: Sequence[np.ndarray], order: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    """ARW's draw of Q column by column in the null space of each column's zeros and the columns before it."""
    n = len(rows)
    Q = np.zeros((n, n))
    for pos, k in enumerate(order):
        R = np.vstack([rows[k], Q[:, list(order[:pos])].T])
        N = _null(R, n)
        y = N.T @ rng.standard_normal(n)
        Q[:, k] = N @ y / np.linalg.norm(y)
    return Q


# the importance weight


class _Map:
    """The map from (A0, s) (s the scaled lag sum of A+, when there are long-run zeros) to the draw's coordinates, and
    the zero restrictions, both vectorised over a batch of points for numerical differentiation."""

    def __init__(self, scheme_rows: dict[int, tuple[list[int], list[int]]], order: Sequence[int], n: int, lags: int,
                 long_run: bool, bases: list[np.ndarray]):
        self.rows, self.order, self.n, self.root_p, self.long_run, self.bases = scheme_rows, list(order), n, np.sqrt(lags), long_run, bases

    def split(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        n = self.n
        A0 = x[:, : n * n].reshape(-1, n, n)
        return A0, (x[:, n * n:].reshape(-1, n, n) if self.long_run else None)

    def _parts(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        n = self.n
        A0, s = self.split(x)
        A0inv = np.linalg.inv(A0)
        sigma = np.swapaxes(A0inv, 1, 2) @ A0inv
        sigma = (sigma + np.swapaxes(sigma, 1, 2)) / 2
        P0 = np.linalg.cholesky(sigma)
        Q = np.swapaxes(P0, 1, 2) @ A0
        linf = bs = None
        if self.long_run:
            bs = s @ A0inv
            C = self.root_p * bs  # the lag sum of the reduced-form coefficients, sum_l B_l (= A(1)')
            linf = np.linalg.solve(np.eye(n) - np.swapaxes(C, 1, 2), P0)
        return sigma, P0, Q, linf, bs

    def _F(self, k: int, P0: np.ndarray, linf: np.ndarray | None) -> np.ndarray:
        impact, longrun = self.rows[k]
        parts = [P0[:, impact, :]]
        if longrun:
            parts.append(linf[:, longrun, :])  # type: ignore[index]
        return np.concatenate(parts, axis=1)

    def constraints(self, x: np.ndarray) -> np.ndarray:
        _, P0, Q, linf, _ = self._parts(x)
        out = [np.einsum("bij,bj->bi", self._F(k, P0, linf), Q[:, :, k]) for k in range(self.n)]
        return np.concatenate(out, axis=1)

    def coordinates(self, x: np.ndarray) -> np.ndarray:
        n = self.n
        sigma, P0, Q, linf, bs = self._parts(x)
        iu = np.triu_indices(n)
        out = [sigma[:, iu[0], iu[1]]]
        if self.long_run:
            out.insert(0, bs.reshape(len(x), -1))  # type: ignore[union-attr]
        for pos, k in enumerate(self.order):
            E = self.bases[pos]
            if E.shape[1] == 0:
                continue
            R = np.concatenate([self._F(k, P0, linf), np.swapaxes(Q[:, :, self.order[:pos]], 1, 2)], axis=1)
            if R.shape[1]:
                RRt = R @ np.swapaxes(R, 1, 2)
                proj = np.eye(n) - np.swapaxes(R, 1, 2) @ np.linalg.solve(RRt, R)
            else:
                proj = np.broadcast_to(np.eye(n), (len(x), n, n))
            N = proj @ E  # a basis of the null space that moves smoothly with x: E projected, then orthonormalised
            vals, vecs = np.linalg.eigh(np.swapaxes(N, 1, 2) @ N)
            N = N @ (vecs / np.sqrt(vals)[:, None, :]) @ np.swapaxes(vecs, 1, 2)
            out.append(np.einsum("bji,bj->bi", N, Q[:, :, k]))
        return np.concatenate(out, axis=1)


def _jacobian(f: Any, x0: np.ndarray, step: float = 1e-6) -> np.ndarray:
    D = len(x0)
    h = step * np.maximum(1.0, np.abs(x0))
    X = np.vstack([x0 + np.diag(h), x0 - np.diag(h)])
    Y = f(X)
    return ((Y[:D] - Y[D:]) / (2 * h)[:, None]).T


def log_weight(rows: Sequence[np.ndarray], scheme_rows: dict[int, tuple[list[int], list[int]]], order: Sequence[int],
               sigma: np.ndarray, A: np.ndarray, Q: np.ndarray, _long_run: bool | None = None) -> float:
    """ARW's log importance weight of the draw Q at (A, Sigma), up to a constant: -(2n+m+1) log|det A0| - log v, with
    the factors of v in A+ taken out analytically (see the module notes): -(2n+1) log|det A0| - log v(A0) with impact
    zeros only, -(3n+1) log|det A0| - log v(A0, s) with long-run zeros."""
    n = len(rows)
    long_run = any(lr for _, lr in scheme_rows.values()) if _long_run is None else _long_run  # (forced only in the tests)
    P0 = np.linalg.cholesky(sigma)
    A0 = np.linalg.solve(P0.T, Q)
    x0 = A0.reshape(-1)
    lags = len(A)
    if long_run:
        C = _lag_sum(A).T  # sum_l B_l
        x0 = np.concatenate([x0, (C @ A0 / np.sqrt(lags)).reshape(-1)])
    bases = []
    for pos, k in enumerate(order):
        R = np.vstack([rows[k], Q[:, list(order[:pos])].T])
        bases.append(_null(R, n))
    m = _Map(scheme_rows, order, n, lags, long_run, bases)
    J = _jacobian(m.coordinates, x0)
    Jc = _jacobian(m.constraints, x0)
    T = _null(Jc, len(x0)) if Jc.shape[0] else np.eye(len(x0))
    G = J @ T
    logv = 0.5 * np.linalg.slogdet(G.T @ G)[1]
    logdet = np.log(abs(np.linalg.det(A0)))
    return float(-((3 if long_run else 2) * n + 1) * logdet - logv)


def _sign_ok(scheme: Any, variables: Sequence[str], responses: np.ndarray, Q: np.ndarray) -> bool:
    """Signs each restricted column by its first sign restriction (in place) and checks them all."""
    index = {v: i for i, v in enumerate(variables)}
    for k, shock in enumerate(scheme.shock_names(variables)):
        restriction = [(index[v], s) for v, s in scheme.signs.get(shock, {}).items()]
        if not restriction:
            continue
        i0, s0 = restriction[0]
        flip = 1.0 if s0 * responses[0, i0, k] > 0 else -1.0
        if not all(np.all(flip * s * responses[:, i, k] > 0) for i, s in restriction):
            return False
        if flip < 0:
            Q[:, k] = -Q[:, k]
            responses[:, :, k] = -responses[:, :, k]
    return True


def _scheme_rows(scheme: Any, variables: Sequence[str]) -> dict[int, tuple[list[int], list[int]]]:
    index = {v: i for i, v in enumerate(variables)}
    return {k: ([index[v] for v in scheme.zeros.get(shock, ())], [index[v] for v in scheme.long_run_zeros.get(shock, ())])
            for k, shock in enumerate(scheme.shock_names(variables))}


def zero_sign_draws(scheme: Any, sigma: np.ndarray, A: np.ndarray, variables: Sequence[str], tries: int,
                    rng: np.random.Generator, want: int | None = None) -> tuple[list[np.ndarray], list[float], int]:
    """Up to ``tries`` draws of Q at (A, Sigma) (stopping once ``want`` are kept): the impact matrices that satisfy the
    zeros and signs, their log importance weights, and how many were drawn."""
    from .var import _ma

    rows = zero_rows(scheme, sigma, A, variables)
    order = generation_order([len(r) for r in rows])
    srows = _scheme_rows(scheme, variables)
    P0 = np.linalg.cholesky(sigma)
    phi = _ma(A, scheme.horizon)
    kept, logw = [], []
    drawn = 0
    while drawn < tries and (want is None or len(kept) < want):
        drawn += 1
        Q = draw_rotation(rows, order, rng)
        responses = phi @ (P0 @ Q)
        if _sign_ok(scheme, variables, responses, Q):
            kept.append(P0 @ Q)
            logw.append(log_weight(rows, srows, order, sigma, A, Q))
    return kept, logw, drawn


def resample(items: Sequence[Any], logw: Sequence[float], size: int, rng: np.random.Generator) -> tuple[list[Any], float]:
    """``size`` items drawn with replacement in proportion to exp(logw); and the effective sample size."""
    lw = np.asarray(logw, float)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    ess = float(1.0 / np.sum(w**2))
    pick = rng.choice(len(items), size=size, replace=True, p=w)
    return [items[i] for i in pick], ess
