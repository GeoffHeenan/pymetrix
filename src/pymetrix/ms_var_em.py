"""The fitting engine of ``MSVAR.fit``: a Hamilton filter and Kim smoother vectorised over time, starting regime paths
from the data, short EM runs to choose among them, and a quasi-Newton polish on the exact likelihood.

**Filter and smoother without a loop over time.** With d_t the regime densities at t (scaled by their largest, the
scale kept in logs), the unnormalised forward probabilities are alpha_t = alpha_0 G_1 ... G_t, G_t = P diag(d_t),
and the backward ones beta_t = G_t+1 ... G_T 1. Both are prefix products of small positive matrices, computed in
log2(T) rounds of batched matrix products (a Hillis-Steele scan; each product rescaled by its sum, the scales kept in
logs), so there is no cancellation and no Python loop over periods. Filtered probabilities are alpha_t normalised,
smoothed ones alpha_t beta_t normalised, and the transitions' smoothed probabilities alpha_t(i) P_ij d_t+1(j)
beta_t+1(j) normalised: the same numbers as the period-by-period recursions (Hamilton 1989; Kim 1994) up to rounding.
The likelihood alone is a tree reduction of the same products, batched over many parameter sets (for the polish's
gradient and the bands' Hessian).

**Starting values.** EM climbs to the nearest local maximum, and a random soft split of the sample (each period's
regime weights drawn uniformly) gives regimes that look alike, which EM then keeps alike: it stops where the regimes
coincide (the likelihood of the linear VAR) or nearly so. Starting regime paths that already differ come from the
data instead: (1) each variable split at its quantiles, raw and as a centred rolling mean (switching levels); (2) the
residuals of the linear VAR split by the rolling mean of their squares, per variable and summed over the standardised
residuals (switching variances); (3) k-means on the standardised [y_t, y_t-1] and on the standardised residuals
(k-means++ seeding from the fit's seed); (4) a one-variable Markov-switching fit of each variable (the same EM,
switching intercept and variance), its smoothed probabilities; and (5) ``starts - 1`` random soft splits, as before.
Each start gets a short EM run (``SCREEN`` iterations); one that has collapsed (two regimes' parameters within 1e-4
of each other, scaled, or a regime with less than half an observation of smoothed probability) is replaced by a
perturbed copy of the best start (its probabilities mixed with a random split) and screened again. The ``KEEP``
best distinct starts by likelihood run EM to convergence and are polished; the fit is the best of them.

**Polish.** EM's update of P ignores that the first regime is drawn from P's own ergodic distribution, so EM's fixed
point is not quite the maximum of the exact likelihood. BFGS maximises the exact likelihood over all the parameters
on the unrestricted scale of ``ms_var_bands.pack`` (Cholesky factors with log diagonals, transition logits), with
the gradient by central differences evaluated as one batch.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .ms_var import MSVAR

SCREEN = 30  # EM iterations for screening a start
KEEP = 3  # starts run to convergence
EM_TOL = 1e-8  # EM stops when the log likelihood changes by less than this (relative); the polish finishes
POLISH_MARGIN = 1.0  # converged starts this far below the best are not polished

__all__ = ["EM_TOL", "KEEP", "POLISH_MARGIN", "SCREEN", "forward_backward", "loglike_from_logdens", "screen_and_fit", "start_paths"]


# --- filter and smoother ---------------------------------------------------------------------------------------------

def _ergodic(P: np.ndarray) -> np.ndarray:
    """Ergodic distributions of transition matrices (..., M, M)."""
    M = P.shape[-1]
    A = np.swapaxes(np.eye(M) - P + np.ones((M, M)), -1, -2)
    return np.linalg.solve(A, np.ones(P.shape[:-1])[..., None])[..., 0]


def _scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = X.sum(axis=(-1, -2))
    s = np.where(s > 0, s, 1.0)
    return X / s[..., None, None], np.log(s)


def _prefix_products(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prefix products G_0 G_1 ... G_t along axis -3 of positive matrices (..., n, M, M), each scaled to sum to one:
    (products, log scales)."""
    X, logs = _scale(G)
    n = X.shape[-3]
    shift = 1
    while shift < n:
        Y, ls = _scale(X[..., :-shift, :, :] @ X[..., shift:, :, :])
        X = np.concatenate([X[..., :shift, :, :], Y], axis=-3)
        logs = np.concatenate([logs[..., :shift], logs[..., :-shift] + logs[..., shift:] + ls], axis=-1)
        shift *= 2
    return X, logs


def _dens(logd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    top = logd.max(axis=-1, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    return np.exp(logd - top), top[..., 0]


def forward_backward(logd: np.ndarray, P: np.ndarray, smooth: bool = True) -> dict[str, Any]:
    """Filtered, predicted and (with ``smooth``) smoothed probabilities (T, M), the transitions' smoothed
    probabilities summed over t (M, M), and the log likelihood, from log densities (T, M) and P (M, M); the first
    regime from P's ergodic distribution."""
    T, M = logd.shape
    d, top = _dens(logd)
    xi0 = _ergodic(P)
    a0 = xi0 * d[0]
    G = P[None] * d[1:, None, :]  # (T-1, M, M)
    if T > 1:
        F, logs = _prefix_products(G)
        alpha = np.vstack([a0[None], np.einsum("i,tij->tj", a0, F)])
    else:
        alpha, logs = a0[None], np.zeros(0)
    tot = alpha.sum(axis=1)
    filt = alpha / np.maximum(tot, 1e-300)[:, None]
    ll = float(np.log(max(tot[-1], 1e-300)) + (logs[-1] if T > 1 else 0.0) + top.sum())
    pred = np.vstack([xi0[None], filt[:-1] @ P])
    out = {"filt": filt, "pred": pred, "ll": ll}
    if not smooth:
        return out
    if T > 1:  # beta_t = G_t+1 ... G_T-1 1: prefix products of the reversed, transposed sequence
        R, _ = _prefix_products(np.swapaxes(G[::-1], 1, 2))
        beta = np.vstack([R.sum(axis=1)[::-1], np.ones((1, M))])  # beta_t for t = 0..T-2, then beta_T-1 = 1
    else:
        beta = np.ones((1, M))
    sm = filt * beta
    sm /= np.maximum(sm.sum(axis=1, keepdims=True), 1e-300)
    trans = np.zeros((M, M))
    if T > 1:
        joint = filt[:-1, :, None] * G * beta[1:, None, :]
        joint /= np.maximum(joint.sum(axis=(1, 2), keepdims=True), 1e-300)
        trans = joint.sum(axis=0)
    out.update(smooth=sm, trans=trans)
    return out


def loglike_from_logdens(logd: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Exact log likelihoods (N,) from log densities (N, T, M) and transition matrices (N, M, M), by a tree
    reduction of the products alpha_0 G_1 ... G_T-1 over time."""
    d, top = _dens(logd)
    N, T, M = d.shape
    a = _ergodic(P) * d[:, 0]
    G = P[:, None] * d[:, 1:, None, :]
    X, logs = _scale(G)
    total = logs.sum(axis=1) if T > 1 else np.zeros(N)
    while X.shape[1] > 1:
        if X.shape[1] % 2:
            X = np.concatenate([X, np.broadcast_to(np.eye(M), (N, 1, M, M))], axis=1)
        X, ls = _scale(X[:, 0::2] @ X[:, 1::2])
        total = total + ls.sum(axis=1)
    if T > 1:
        a = np.einsum("ni,nij->nj", a, X[:, 0])
    return np.log(np.maximum(a.sum(axis=1), 1e-300)) + total + top.sum(axis=1)


def log_densities(Y: np.ndarray, means: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """(T, M) log normal densities; means (M, T, K), sigmas (M, K, K)."""
    T, K = Y.shape
    L = np.linalg.cholesky(sigmas)
    z = np.einsum("mij,mtj->mti", np.linalg.inv(L), Y[None] - means)  # (M, T, K)
    return (-0.5 * np.sum(z**2, axis=2) - np.log(np.diagonal(L, axis1=1, axis2=2)).sum(axis=1)[:, None]
            - K / 2 * np.log(2 * np.pi)).T


# --- EM ----------------------------------------------------------------------------------------------------------------

def _means(C: np.ndarray, B: np.ndarray, Xl: np.ndarray) -> np.ndarray:
    return C[:, None, :] + np.einsum("tj,mkj->mtk", Xl, B)


def em(spec: MSVAR, Y: np.ndarray, Xl: np.ndarray, w: np.ndarray, iters: int, tol: float,
       P: np.ndarray | None = None) -> dict[str, Any] | None:
    """EM from smoothed weights ``w`` (T, M) for at most ``iters`` iterations, until the log likelihood changes by
    less than ``tol`` (relative). None when a covariance is singular."""
    M = w.shape[1]
    S = None
    if P is None:
        P = np.full((M, M), 0.1 / (M - 1))
        np.fill_diagonal(P, 0.9)
    ll_old = -np.inf
    it = 0
    for it in range(1, iters + 1):
        try:
            C, B, S = spec._mstep(Y, Xl, w, S)
            if spec.switching != "all":  # one more pass with the new covariances (conditional maximisation)
                C, B, S = spec._mstep(Y, Xl, w, S)
            logd = log_densities(Y, _means(C, B, Xl), S)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(S)):
            return None
        r = forward_backward(logd, P)
        w = r["smooth"]
        P = r["trans"] / np.maximum(r["trans"].sum(axis=1, keepdims=True), 1e-300)
        ll = r["ll"]
        if abs(ll - ll_old) < tol * (1 + abs(ll)):
            break
        ll_old = ll
    if not np.isfinite(ll):
        return None
    return {"C": C, "B": B, "S": S, "P": P, "ll": ll, "w": w, "iterations": it}


def collapsed(spec: MSVAR, r: dict[str, Any], Y: np.ndarray) -> bool:
    """Two regimes with (nearly) the same parameters, or a regime with under half an observation of probability."""
    if r["w"].sum(axis=0).min() < 0.5:
        return True
    scale = np.maximum(Y.std(axis=0), 1e-12)
    M = len(r["C"])
    sd = np.sqrt(np.maximum(np.diagonal(r["S"], axis1=1, axis2=2), 0))
    for i in range(M):
        for j in range(i + 1, M):
            gap = max(np.max(np.abs(r["C"][i] - r["C"][j]) / scale), np.max(np.abs(r["B"][i] - r["B"][j]), initial=0.0),
                      np.max(np.abs(np.log(np.maximum(sd[i], 1e-300) / np.maximum(sd[j], 1e-300)))))
            if gap < 1e-4:
                return True
    return False


# --- starting values ---------------------------------------------------------------------------------------------------

def _soft(labels: np.ndarray, M: int) -> np.ndarray:
    w = np.full((len(labels), M), 0.1 / (M - 1))
    w[np.arange(len(labels)), labels] = 0.9
    return w


def _quantile_labels(x: np.ndarray, M: int) -> np.ndarray:
    return np.digitize(x, np.quantile(x, np.linspace(0, 1, M + 1)[1:-1]))


def _rolling(x: np.ndarray, width: int) -> np.ndarray:
    """Centred moving average (shorter windows at the ends)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    T = len(x)
    lo = np.clip(np.arange(T) - width // 2, 0, T)
    hi = np.clip(np.arange(T) + width - width // 2, 0, T)
    return (c[hi] - c[lo]) / (hi - lo)


def _kmeans(X: np.ndarray, M: int, rng: np.random.Generator, rounds: int = 50) -> np.ndarray:
    """Lloyd's k-means with k-means++ seeding; the labels."""
    centres = [X[rng.integers(len(X))]]
    for _ in range(1, M):
        d2 = np.min([((X - c) ** 2).sum(axis=1) for c in centres], axis=0)
        centres.append(X[rng.choice(len(X), p=d2 / d2.sum())] if d2.sum() > 0 else X[rng.integers(len(X))])
    C = np.array(centres)
    labels = np.zeros(len(X), int)
    for _ in range(rounds):
        new = np.argmin(((X[:, None] - C[None]) ** 2).sum(axis=2), axis=1)
        if _ and (new == labels).all():
            break
        labels = new
        C = np.array([X[labels == j].mean(axis=0) if (labels == j).any() else C[j] for j in range(M)])
    return labels


def _standardise(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)


def start_paths(spec: MSVAR, Y: np.ndarray, Xl: np.ndarray, starts: int, rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    """Starting weights (T, M), each named by where it comes from. The first is the old default (the first variable
    split at its quantiles); the random soft splits come last."""
    T, K = Y.shape
    M = spec.regimes
    width = max(3, min(9, T // 30))
    out: list[tuple[str, np.ndarray]] = [("quantiles of the first variable", _soft(_quantile_labels(Y[:, 0], M), M))]
    for k in range(K):
        if k:
            out.append((f"quantiles of variable {k}", _soft(_quantile_labels(Y[:, k], M), M)))
        out.append((f"rolling mean of variable {k}", _soft(_quantile_labels(_rolling(Y[:, k], width), M), M)))
    Z = np.column_stack([np.ones(T), Xl])
    U = Y - Z @ np.linalg.lstsq(Z, Y, rcond=None)[0]
    Us = U / np.maximum(U.std(axis=0), 1e-12)
    if spec.switching != "intercept":
        out.append(("rolling variance of the residuals", _soft(_quantile_labels(_rolling((Us**2).sum(axis=1), width), M), M)))
        if K > 1:
            out += [(f"rolling variance of residual {k}", _soft(_quantile_labels(_rolling(Us[:, k] ** 2, width), M), M)) for k in range(K)]
    out.append(("k-means on the data and its lags", _soft(_kmeans(_standardise(np.column_stack([Y, Xl])), M, rng), M)))
    out.append(("k-means on the residuals", _soft(_kmeans(np.column_stack([Us, np.log(Us**2 + 1e-3)]) if spec.switching != "intercept" else Us, M, rng), M)))
    if K > 1:  # one-variable Markov-switching fits
        from .ms_var import MSVAR

        for k in range(K):
            one = MSVAR([f"y{k}"], lags=spec.lags, regimes=M, switching="intercept" if spec.switching == "intercept" else "intercept_variance")
            Xk = Xl[:, k::K] if spec.lags else Xl
            best = None
            for _, w in start_paths(one, Y[:, [k]], Xk, 1, rng):
                r = em(one, Y[:, [k]], Xk, w, SCREEN, 1e-8)
                if r is not None and (best is None or r["ll"] > best["ll"]):
                    best = r
            if best is not None:
                out.append((f"one-variable Markov-switching fit of variable {k}", 0.8 * best["w"] + 0.2 / M))
    out += [("random split", rng.dirichlet(np.ones(M), size=T)) for _ in range(max(starts, 1) - 1)]
    return out


# --- polish ------------------------------------------------------------------------------------------------------------

def _shape(spec: MSVAR, r: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(spec=spec, variables=tuple(range(r["C"].shape[1])), intercepts=r["C"], coefs=r["B"], sigmas=r["S"],
                           transition=r["P"])


def loglike_theta(theta: np.ndarray, shape: Any, Y: np.ndarray, Xl: np.ndarray, chunk: int | None = None) -> np.ndarray:
    """Exact log likelihoods at the rows of ``theta`` (N, n) (on ``ms_var_bands.pack``'s scale), the rows run
    together in chunks; ``shape`` is a fit (or anything with its ``spec`` and ``variables``)."""
    from .ms_var_bands import unpack

    T, K = Y.shape
    theta = np.atleast_2d(theta)
    M = shape.spec.regimes
    chunk = chunk or max(1, int(4e6 // (M * T * max(K, M) * max(K, 2))))
    out = np.empty(len(theta))
    for a in range(0, len(theta), chunk):
        C, B, L, P = unpack(theta[a:a + chunk], shape)
        resid = Y[None, None] - C[:, :, None, :] - np.einsum("tj,nmkj->nmtk", Xl, B)
        z = np.einsum("nmij,nmtj->nmti", np.linalg.inv(L), resid)
        logd = (-0.5 * np.sum(z**2, axis=3) - np.log(np.diagonal(L, axis1=2, axis2=3)).sum(axis=2)[:, :, None]
                - K / 2 * np.log(2 * np.pi)).transpose(0, 2, 1)
        out[a:a + chunk] = loglike_from_logdens(logd, P)
    return out


def polish(spec: MSVAR, r: dict[str, Any], Y: np.ndarray, Xl: np.ndarray) -> dict[str, Any]:
    """BFGS on the exact likelihood from EM's estimates; then the probabilities at the optimum."""
    from scipy import optimize

    from .ms_var_bands import pack, unpack

    shape = _shape(spec, r)
    theta0 = pack(shape)
    n = len(theta0)
    free = np.ones(n, bool)
    nP = spec.regimes * (spec.regimes - 1)
    free[n - nP:] = ~np.repeat(r["P"].min(axis=1) < 1e-8, spec.regimes - 1)  # a row on the boundary stays there
    idx = np.flatnonzero(free)

    def f(z: np.ndarray) -> tuple[float, np.ndarray]:
        h = 1e-6 * np.maximum(np.abs(z), 1.0)
        th = np.tile(theta0, (2 * len(idx) + 1, 1))
        th[:, idx] = z
        th[1 + np.arange(len(idx)), idx] += h
        th[1 + len(idx) + np.arange(len(idx)), idx] -= h
        v = loglike_theta(th, shape, Y, Xl)
        if not np.isfinite(v[0]):
            return 1e300, np.zeros(len(idx))
        g = (v[1:1 + len(idx)] - v[1 + len(idx):]) / (2 * h)
        return -float(v[0]), np.where(np.isfinite(g), -g, 0.0)

    found = optimize.minimize(f, theta0[idx], jac=True, method="BFGS", options={"gtol": 1e-6, "maxiter": 500})
    theta = theta0.copy()
    theta[idx] = found.x
    if not -found.fun >= r["ll"] - 1e-12:  # never worse than EM's point
        theta = theta0
    C, B, L, P = (np.array(x[0]) for x in unpack(theta, shape))
    S = L @ np.swapaxes(L, -1, -2)
    fb = forward_backward(log_densities(Y, _means(C, B, Xl), S), P)
    return {**r, "C": C, "B": B, "S": S, "P": P, "ll": fb["ll"], "filt": fb["filt"], "smooth": fb["smooth"]}


# --- the whole fit ----------------------------------------------------------------------------------------------------

def screen_and_fit(spec: MSVAR, Y: np.ndarray, Xl: np.ndarray, starts: int, max_iter: int, tol: float,
                   rng: np.random.Generator) -> dict[str, Any] | None:
    """The best fit from the starts of ``start_paths``: each screened by a short EM run (collapsed ones replaced by a
    perturbation of the best so far), the ``KEEP`` best distinct ones run to convergence and polished."""
    paths = start_paths(spec, Y, Xl, starts, rng)
    screen = min(SCREEN, max(max_iter, 1))
    screened: list[tuple[float, int, dict[str, Any]]] = []
    fallback = None
    retries = 0
    queue = list(enumerate(w for _, w in paths))
    while queue:
        i, w = queue.pop(0)
        r = em(spec, Y, Xl, w, screen, 0.0)
        if r is None:
            continue
        if collapsed(spec, r, Y):
            if fallback is None or r["ll"] > fallback["ll"]:
                fallback = r
            if retries < len(paths) and screened:  # restart from a perturbation of the best start so far
                retries += 1
                best_w = max(screened, key=lambda x: x[0])[2]["w"]
                queue.append((len(paths) + retries, 0.6 * best_w + 0.4 * rng.dirichlet(np.ones(spec.regimes), size=len(Y))))
            continue
        screened.append((r["ll"], i, r))
    screened.sort(key=lambda x: (-x[0], x[1]))
    chosen: list[dict[str, Any]] = []
    for ll, _, r in screened:
        if all(abs(ll - c["ll"]) > 1e-6 * (1 + abs(ll)) for c in chosen):
            chosen.append(r)
        if len(chosen) == KEEP:
            break
    if not chosen and fallback is not None:
        chosen = [fallback]
    fits = [f for f in (em(spec, Y, Xl, r["w"], max(max_iter - screen, 1), max(tol, EM_TOL), P=r["P"]) for r in chosen) if f is not None]
    if not fits:
        return None
    fits.sort(key=lambda f: -f["ll"])
    best = None
    polished: list[float] = []
    for full in fits:
        # the polish moves the likelihood by far less than POLISH_MARGIN; starts that EM took to the same maximum
        # (log likelihoods within 1e-4) are polished once
        if full["ll"] < fits[0]["ll"] - POLISH_MARGIN or any(abs(full["ll"] - q) < 1e-4 for q in polished):
            continue
        polished.append(full["ll"])
        full["iterations"] += screen
        full = polish(spec, full, Y, Xl)
        if best is None or full["ll"] > best["ll"]:
            best = full
    return best
