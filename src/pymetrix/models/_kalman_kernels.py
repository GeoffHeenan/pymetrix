"""The compiled kernels of ``pymetrix.models._kalman_compiled`` (imported only when numba is installed): module-level
functions, so that numba caches them on disk and later processes (chains in other processes too) load them at once."""

import numpy as np
from numba import njit

_jit = njit(cache=True, nogil=True)


@_jit
def chol(F, c):  # lower Cholesky factor of F (its lower triangle read); False where F is not positive definite
    p = F.shape[0]
    for j in range(p):
        s = F[j, j]
        for k in range(j):
            s -= c[j, k] * c[j, k]
        if not s > 0.0:
            return False
        d = np.sqrt(s)
        c[j, j] = d
        for i in range(j + 1, p):
            s = F[i, j]
            for k in range(j):
                s -= c[i, k] * c[j, k]
            c[i, j] = s / d
        for i in range(j):
            c[i, j] = 0.0
    return True

@_jit
def tri_inv(c, out):  # the inverse of a lower-triangular matrix, by forward substitution on the identity
    p = c.shape[0]
    for j in range(p):
        for i in range(p):
            out[i, j] = 0.0
        out[j, j] = 1.0 / c[j, j]
        for i in range(j + 1, p):
            s = 0.0
            for k in range(j, i):
                s -= c[i, k] * out[k, j]
            out[i, j] = s / c[i, i]

@_jit
def masked(Z, H, obs):  # the full-size recursion's matrices for a period with missing observations
    p = Z.shape[0]
    Zt = Z.copy()
    Ht = H.copy()
    for i in range(p):
        if not obs[i]:
            Zt[i, :] = 0.0
            Ht[i, :] = 0.0
            Ht[:, i] = 0.0
            Ht[i, i] = 1.0
    return Zt, Ht

@_jit
def converged(P_next, P, trace_next, trace, n, tol):
    if abs(trace_next - trace) > n * tol * max(1.0, trace_next):
        return False
    big, change = 0.0, 0.0
    for i in range(n):
        for j in range(n):
            big = max(big, abs(P_next[i, j]))
            change = max(change, abs(P_next[i, j] - P[i, j]))
    return change <= tol * max(1.0, big)

@_jit
def variance_pass(T, RQR, Z, H, P0, observed, tol):
    """``kalman_score._filter``'s variance recursion: per period the state it uses, and per state P, G = P Z',
    the inverse Cholesky factor of F, the filtered variance, the period updated; ``fail`` the first period whose
    innovation variance is not positive definite (-1: none)."""
    n_t, p = observed.shape
    n = T.shape[0]
    state = np.empty(n_t, np.int64)
    Ps = np.empty((n_t, n, n))
    Gs = np.empty((n_t, n, p))
    Lis = np.empty((n_t, p, p))
    Pfs = np.empty((n_t, n, n))
    updated = np.empty(n_t, np.int64)
    c = np.zeros((p, p))
    TT = np.ascontiguousarray(T.T)
    ZT = np.ascontiguousarray(Z.T)
    P = P0.copy()
    trace = np.trace(P)
    S = 0
    frozen = -1
    for t in range(n_t):
        complete = True
        for i in range(p):
            if not observed[t, i]:
                complete = False
        if frozen >= 0 and complete:
            state[t] = frozen
            continue
        frozen = -1
        if complete:
            G = P.dot(ZT)
            F = Z.dot(G) + H
        else:
            Zt, Ht = masked(Z, H, observed[t])
            G = P.dot(np.ascontiguousarray(Zt.T))
            F = Zt.dot(G) + Ht
        if not chol(F, c):
            return state, Ps[:S], Gs[:S], Lis[:S], Pfs[:S], updated[:S], t
        tri_inv(c, Lis[S])
        B = Lis[S].dot(np.ascontiguousarray(G.T))
        Pf = P - B.T.dot(B)
        P_next = T.dot(Pf).dot(TT) + RQR
        state[t] = S
        Ps[S] = P
        Gs[S] = G
        Pfs[S] = Pf
        updated[S] = t
        trace_next = np.trace(P_next)
        if complete and converged(P_next, P, trace_next, trace, n, tol):
            frozen = S
        S += 1
        P = P_next
        trace = trace_next
    return state, Ps[:S], Gs[:S], Lis[:S], Pfs[:S], updated[:S], -1

@_jit
def means(L, push, state):
    """a(t+1) = L(state t) a(t) + push(t) from a(0) = 0: the predicted means a(0) ... a(n_t - 1)."""
    n_t, n = push.shape
    a = np.zeros((n_t, n))
    at = np.zeros(n)
    for t in range(n_t):
        a[t] = at
        Ls = L[state[t]]
        nxt = np.empty(n)
        for i in range(n):
            s = push[t, i]
            for j in range(n):
                s += Ls[i, j] * at[j]
            nxt[i] = s
        at = nxt
    return a

@_jit
def backward_means(L, ze, state):
    """abar(t) = L(state t)' abar(t+1) + ze(t) from abar(n_t) = 0: abar(0) ... abar(n_t)."""
    n_t, n = ze.shape
    abar = np.zeros((n_t + 1, n))
    for t in range(n_t - 1, -1, -1):
        Ls = L[state[t]]
        for i in range(n):
            s = ze[t, i]
            for j in range(n):
                s += Ls[j, i] * abar[t + 1, j]
            abar[t, i] = s
    return abar

@_jit
def backward_variance(L, C):
    """Pbar(s) = L(s)' Pbar(s+1) L(s) + C(s) from zero, state by state backwards: Pbar(s+1) for each state, and
    Pbar(0)."""
    S, n, _ = L.shape
    Pnext = np.zeros((S, n, n))
    Pn = np.zeros((n, n))
    for s in range(S - 1, -1, -1):
        Pnext[s] = Pn
        Ls = L[s]
        Pn = np.ascontiguousarray(Ls.T).dot(Pn.dot(Ls)) + C[s]
    return Pnext, Pn

@_jit
def loglike(y, observed, T, c0, RQR, Z, d, H, a0, P0, presample, tol):
    """The log likelihood by the Kalman filter from mean ``a0`` and variance ``P0`` (missing observations exact,
    the converged-variance shortcut as above), and the smallest innovation variance observed; ``fail`` the first
    period whose innovation variance is not positive definite (-1: none)."""
    n_t, p = y.shape
    n = T.shape[0]
    log2pi = np.log(2 * np.pi)
    TT = np.ascontiguousarray(T.T)
    ZT = np.ascontiguousarray(Z.T)
    c = np.zeros((p, p))
    Li = np.zeros((p, p))
    P = P0.copy()
    a = a0.copy()
    trace = np.trace(P)
    frozen = False
    G = np.zeros((n, p))
    Finv = np.zeros((p, p))
    logdet = 0.0
    Fdiag = np.zeros(p)
    ll = 0.0
    smallest = np.inf
    v = np.zeros(p)
    for t in range(n_t):
        complete = True
        seen = 0
        for i in range(p):
            if observed[t, i]:
                seen += 1
            else:
                complete = False
        if not (frozen and complete):
            frozen = False
            if complete:
                Zt = Z
                G = P.dot(ZT)
                F = Z.dot(G) + H
            else:
                Zt, Ht = masked(Z, H, observed[t])
                G = P.dot(np.ascontiguousarray(Zt.T))
                F = Zt.dot(G) + Ht
            if not chol(F, c):
                return ll, smallest, t
            tri_inv(c, Li)
            Finv = np.ascontiguousarray(Li.T).dot(Li)
            logdet = 0.0
            for i in range(p):
                logdet += 2.0 * np.log(c[i, i])
            for i in range(p):
                Fdiag[i] = F[i, i]
            B = Li.dot(np.ascontiguousarray(G.T))
            Pf = P - B.T.dot(B)
            P_next = T.dot(Pf).dot(TT) + RQR
            trace_next = np.trace(P_next)
            if complete and converged(P_next, P, trace_next, trace, n, tol):
                frozen = True
            P = P_next
            trace = trace_next
        for i in range(p):
            if observed[t, i]:
                smallest = min(smallest, Fdiag[i])
                s = y[t, i] - d[i]
                for j in range(n):
                    s -= Z[i, j] * a[j]
                v[i] = s
            else:
                v[i] = 0.0
        e = Finv.dot(v)
        if t >= presample:
            q = 0.0
            for i in range(p):
                q += v[i] * e[i]
            ll -= 0.5 * (seen * log2pi + logdet + q)
        af = a + G.dot(e)
        a = T.dot(af) + c0
    return ll, smallest, -1
