"""The multivariate GARCH recursions, run over every period at once (linear filters), against the plain loop over
periods written out here: BEKK (scalar, diagonal and with a free coefficient for each element), the full VECH, and DCC
with and without the asymmetric term."""

import importlib

import numpy as np

mg = importlib.import_module("pymetrix_finance.mgarch")


def _returns(T=400, N=3, seed=0):
    rng = np.random.default_rng(seed)
    C = np.linalg.cholesky(np.array([[1.0, 0.4, 0.2], [0.4, 1.5, -0.3], [0.2, -0.3, 0.8]])[:N, :N])
    return 0.01 * rng.standard_normal((T, N)) @ C.T


def test_bekk_against_the_loop():
    e = _returns()
    T, N = e.shape
    S = e.T @ e / T
    a, b = np.array([0.25, 0.3, 0.2]), np.array([0.95, 0.93, 0.96])
    rng = np.random.default_rng(1)
    free_a, free_b = rng.uniform(0.02, 0.08, (N, N)), rng.uniform(0.85, 0.9, (N, N))
    for A, B in ((0.05, 0.9), (np.outer(a, a), np.outer(b, b)), (free_a, free_b)):
        H, h = np.empty((T, N, N)), S.copy()
        for t in range(T):
            H[t] = h
            h = (1 - A - B) * S + A * np.outer(e[t], e[t]) + B * h
        np.testing.assert_allclose(mg.bekk_path(e, S, A, B), H, rtol=1e-12, atol=1e-20)


def test_vech_against_the_loop():
    for N in (2, 3):
        e = _returns(N=N, seed=N)
        T, m = len(e), N * (N + 1) // 2
        S = e.T @ e / T
        rng = np.random.default_rng(N)
        A, B = rng.uniform(0.0, 0.04, (m, m)), 0.9 * np.eye(m) + rng.uniform(0.0, 0.01, (m, m))
        iu = np.triu_indices(N)
        H, h = mg.vech_path(e, S, A, B), S[iu].copy()
        assert H.shape == (T, N, N)
        for t in range(T):
            Ht = np.zeros((N, N))
            Ht[iu] = h
            np.testing.assert_allclose(H[t], Ht + np.triu(Ht, 1).T, rtol=1e-12)
            h = (np.eye(m) - A - B) @ S[iu] + A @ np.outer(e[t], e[t])[iu] + B @ h


def test_dcc_against_the_loop():
    z = _returns(T=500, seed=4) / 0.01
    Qbar = z.T @ z / len(z)
    n = np.minimum(z, 0.0)
    Nbar = n.T @ n / len(n)
    for a, b, g in ((0.05, 0.9, 0.0), (0.03, 0.9, 0.04)):
        Q = Qbar.copy()
        R = np.empty((len(z), 3, 3))
        for t in range(len(z)):
            d = 1 / np.sqrt(np.diag(Q))
            R[t] = Q * np.outer(d, d)
            Q = (1 - a - b) * Qbar + a * np.outer(z[t], z[t]) + b * Q + g * (np.outer(n[t], n[t]) - Nbar)
        np.testing.assert_allclose(mg.dcc_path(z, a, b, Qbar, g, Nbar if g else None), R, rtol=1e-12, atol=1e-14)
        R_linear, Q_next = mg.dcc_path_linear(z, a, b, Qbar, g, Nbar if g else None)
        np.testing.assert_allclose(R_linear, R, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(Q_next, Q, rtol=1e-12)
