"""Third-order perturbation, as Dynare's ``stoch_simul(order=3)`` (its k-order solver).

    sol3 = rbc.solve(order=3)            # a template, a Dynare or IRIS model; or third_order(model, shocks, shock_sd)
    sol3.ghxxx, sol3.ghxxu, sol3.ghxuu, sol3.ghuuu, sol3.ghxss, sol3.ghuss   # Dynare's names for the third-order terms
    sol3.mean()                          # the ergodic mean of the pruned process
    sol3.simulate(200, seed=1)           # pruned (Andreasen et al. 2018); pruning=False for the plain rule
    sol3.irf(20)                         # from the stochastic steady state; replications=500: generalised responses
    sol3.moments()                       # mean, standard deviation, skewness, kurtosis: a long pruned simulation

The decision rule, in deviations from the steady state, with ``s`` the predetermined variables last period (the
states), ``u`` the shocks and uncertainty at the model's shock sizes:

    x(t) = ghx s + ghu u + 1/2 (ghxx s s + 2 ghxu s u + ghuu u u + ghs2)
           + 1/6 (ghxxx s s s + 3 ghxxu s s u + 3 ghxuu s u u + ghuuu u u u + 3 ghxss s + 3 ghuss u + ghsss)

(products are Kronecker products). The terms beyond second order are the cubic ones and two corrections for risk:
``ghxss`` and ``ghuss`` make the responses to states and shocks depend on how uncertain the future is. ``ghsss`` is
zero: shocks are taken to be symmetric (normal), as in Dynare.

How it is solved: the decision rule is a function ``g(s, u, sigma)`` with ``sigma`` scaling future shocks. The
model's expected residual is differentiated one, two and three times in ``(s, u, sigma)`` (the chain rule through
next period's variables, ``x(t+1) = g(g_s(s, u, sigma), sigma e(t+1), sigma)``), with exact symbolic derivatives of
the equations up to the third. Each order's unknown derivatives are found block by block (Schmitt-Grohe and Uribe
2004 and Kim, Kim, Schaumburg and Sims 2008 for the second order; Levintal 2017 for higher ones): for each number
``c`` of derivatives in ``sigma``, first those in the states only, a generalised Sylvester equation
``A X + A0 X (P_s kron ... kron P_s) = R`` solved on the Schur form of the state transition one Kronecker factor at a
time, then those with shocks, a linear system each. Odd numbers of ``sigma`` derivatives are zero. The expectation over next period's shocks is
exact: the residual's derivatives are polynomials of degree three or less in the shocks, and they are averaged over
``2k`` points that match the shocks' mean, covariance and (zero) third moments.

Simulations are pruned by default: the second- and third-order terms act on the lower-order parts of the state,
which keeps paths stable and gives the mean, variance and so on that the pruned process has (Andreasen,
Fernandez-Villaverde and Rubio-Ramirez 2018). Checked against Dynare 6.0's own solution and simulations
(tests/test_third_order.py).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import factorial
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg

from .._random import generator as make_rng
from ..output import Table
from .solvers import ModelError

__all__ = ["ThirdOrderSolution", "third_order"]


@dataclass(frozen=True, eq=False)
class ThirdOrderSolution:
    """A third-order approximation of a model's decision rules: see the module docstring."""

    variables: tuple[str, ...]  # every variable (with auxiliary ones for longer lags and leads)
    endogenous: tuple[str, ...]  # the model's own
    states: tuple[str, ...]  # the variables that appear lagged
    shocks: tuple[str, ...]
    ghx: np.ndarray  # n x states
    ghu: np.ndarray  # n x shocks
    ghxx: np.ndarray  # n x states^2
    ghxu: np.ndarray  # n x states*shocks
    ghuu: np.ndarray  # n x shocks^2
    ghs2: np.ndarray  # n
    ghxxx: np.ndarray  # n x states^3
    ghxxu: np.ndarray  # n x states^2*shocks
    ghxuu: np.ndarray  # n x states*shocks^2
    ghuuu: np.ndarray  # n x shocks^3
    ghxss: np.ndarray  # n x states
    ghuss: np.ndarray  # n x shocks
    ghsss: np.ndarray  # n (zero for symmetric shocks)
    sigma: np.ndarray  # covariance of the shocks
    steady: Any  # SteadyState
    first: Any  # the first-order LinearSolution
    derivatives: tuple[np.ndarray, ...] = ()  # g's derivatives in (states, shocks, sigma), full symmetric tensors

    # -- evaluation -------------------------------------------------------------------------------------------------

    def _index(self, names: Sequence[str]) -> list[int]:
        return [self.variables.index(v) for v in names]

    def _parts(self, sf: np.ndarray, ss_: np.ndarray, sr: np.ndarray, u: np.ndarray, sigma2: float = 1.0
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One pruned step (Andreasen et al. 2018): the first-, second- and third-order parts of x(t) from those parts
        of the states and the shocks; each argument is (paths x size)."""
        xf = sf @ self.ghx.T + u @ self.ghu.T
        xs = (ss_ @ self.ghx.T + 0.5 * _kron(sf, sf) @ self.ghxx.T + _kron(sf, u) @ self.ghxu.T
              + 0.5 * _kron(u, u) @ self.ghuu.T + 0.5 * sigma2 * self.ghs2)
        xr = (sr @ self.ghx.T + _kron(sf, ss_) @ self.ghxx.T + _kron(ss_, u) @ self.ghxu.T
              + _kron(sf, sf, sf) @ self.ghxxx.T / 6 + 0.5 * _kron(sf, sf, u) @ self.ghxxu.T
              + 0.5 * _kron(sf, u, u) @ self.ghxuu.T + _kron(u, u, u) @ self.ghuuu.T / 6
              + sigma2 * (0.5 * sf @ self.ghxss.T + 0.5 * u @ self.ghuss.T) + sigma2**1.5 * self.ghsss / 6)
        return xf, xs, xr

    def _rule(self, s: np.ndarray, u: np.ndarray, sigma2: float = 1.0) -> np.ndarray:
        """The decision rule itself (no pruning): x(t) from the whole state; (paths x size) arrays."""
        return (s @ self.ghx.T + u @ self.ghu.T
                + 0.5 * (_kron(s, s) @ self.ghxx.T + 2 * _kron(s, u) @ self.ghxu.T + _kron(u, u) @ self.ghuu.T + sigma2 * self.ghs2)
                + (_kron(s, s, s) @ self.ghxxx.T + 3 * _kron(s, s, u) @ self.ghxxu.T + 3 * _kron(s, u, u) @ self.ghxuu.T
                   + _kron(u, u, u) @ self.ghuuu.T + 3 * sigma2 * (s @ self.ghxss.T + u @ self.ghuss.T) + sigma2**1.5 * self.ghsss) / 6)

    def _run(self, U: np.ndarray, pruning: bool, start: tuple[np.ndarray, ...] | None = None) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        """Paths (periods x paths x n, deviations) for shocks U (periods x paths x k), from ``start`` (the state's
        parts: (first, second, third) with pruning, the whole state without), and the state at the end."""
        st = self._index(self.states)
        T, m = U.shape[0], U.shape[1]
        ns = len(st)
        out = np.empty((T, m, len(self.variables)))
        if pruning:
            sf, ss_, sr = (np.zeros((m, ns)) for _ in range(3)) if start is None else (np.array(a, dtype=float) for a in start)
            for t in range(T):
                xf, xs, xr = self._parts(sf, ss_, sr, U[t])
                out[t] = xf + xs + xr
                sf, ss_, sr = xf[:, st], xs[:, st], xr[:, st]
            return out, (sf, ss_, sr)
        s = np.zeros((m, ns)) if start is None else np.array(start[0], dtype=float)
        for t in range(T):
            x = self._rule(s, U[t])
            out[t] = x
            s = x[:, st]
        return out, (s,)

    def _draws(self, shape: tuple[int, ...], rng: Any) -> np.ndarray:
        k = len(self.shocks)
        w, V = np.linalg.eigh(self.sigma) if k else (np.zeros(0), np.zeros((0, 0)))
        root = V * np.sqrt(np.clip(w, 0.0, None))
        return rng.standard_normal((*shape, k)) @ root.T

    def _levels(self, x: np.ndarray) -> pd.DataFrame:
        keep = self._index(self.endogenous)
        out = pd.DataFrame(x[:, keep], columns=list(self.endogenous))
        return out + pd.Series({v: self.steady.values[v] for v in self.endogenous})

    def _stochastic_steady(self, pruning: bool) -> tuple[np.ndarray, ...]:
        """Where the model rests when shocks are expected but none occur: the rule iterated with no shocks."""
        k = len(self.shocks)
        state = None
        previous = None
        for _ in range(2000):
            x, state = self._run(np.zeros((1, 1, k)), pruning, state)
            if previous is not None and np.allclose(x, previous, atol=1e-13, rtol=0):
                break
            previous = x
        return state  # type: ignore[return-value]

    # -- results ----------------------------------------------------------------------------------------------------

    def mean(self) -> pd.Series:
        """The ergodic mean of the pruned third-order process, in levels. It is the pruned second-order mean: the
        third-order part has mean zero when shocks are symmetric (Andreasen et al. 2018)."""
        st = self._index(self.states)
        A = self.ghx[st, :]
        V = linalg.solve_discrete_lyapunov(A, self.ghu[st, :] @ self.sigma @ self.ghu[st, :].T) if st else np.zeros((0, 0))
        drift = 0.5 * (self.ghxx @ V.reshape(-1) + self.ghuu @ self.sigma.reshape(-1) + self.ghs2) + self.ghsss / 6
        Es = np.linalg.solve(np.eye(len(st)) - A, drift[st]) if st else np.zeros(0)
        Ex = self.ghx @ Es + drift
        return pd.Series({v: self.steady.values[v] + Ex[self.variables.index(v)] for v in self.endogenous}, name="mean")

    def simulate(self, periods: int, seed: int | None = None, shocks: np.ndarray | None = None, burn: int = 0,
                 pruning: bool = True) -> pd.DataFrame:
        """A stochastic simulation in levels from the steady state (``shocks``: a periods x shocks array to use instead
        of draws; ``burn`` periods dropped first). Pruned unless ``pruning=False``: the plain third-order rule can
        explode once shocks are large."""
        U = np.asarray(shocks, dtype=float) if shocks is not None else self._draws((periods + burn,), make_rng(seed))
        x, _ = self._run(U[:, None, :], pruning)
        return self._levels(x[burn:, 0])

    def irf(self, periods: int = 20, shock_sd: Mapping[str, float] | None = None, state: Mapping[str, float] | None = None,
            replications: int = 0, seed: int | None = 0, pruning: bool = True) -> Any:
        """Responses to one-standard-deviation shocks (or ``shock_sd``), as differences from the path without the
        shock; an ``IRF``. They start from the stochastic steady state, or from ``state`` (levels of some of the
        states; the others at the stochastic steady state). With ``replications`` > 0 they are generalised responses
        (Koop, Pesaran and Potter 1996): the mean difference over that many draws of the other shocks, the same
        draws with and without the impulse; with 0, no other shocks occur. Unlike first-order responses they depend
        on the size and sign of the shock and on the starting point. With pruning, a given start's distance from the
        steady state is put in the first-order part of the state, and the path without the impulse starts there too."""
        from ..var import IRF

        k = len(self.shocks)
        sizes = np.array([float((shock_sd or {}).get(s, np.sqrt(self.sigma[i, i]))) for i, s in enumerate(self.shocks)])
        base = self._stochastic_steady(pruning)
        if state:
            unknown = [v for v in state if v not in self.states]
            if unknown:
                raise ModelError(f"not states of the model: {', '.join(unknown)} (states: {', '.join(self.states)})")
            # a given state: its distance from the steady state is put in the first-order part, the rest kept
            total = sum(base)
            first = total[0].copy()
            for v, level in state.items():
                first[self.states.index(v)] = float(level) - self.steady.values[v]
            base = (first[None, :], np.zeros_like(first)[None, :], np.zeros_like(first)[None, :]) if pruning else (first[None, :],)
        m = max(int(replications), 1)
        U = self._draws((periods + 1, m), make_rng(seed)) if replications else np.zeros((periods + 1, m, k))
        start = tuple(np.repeat(b, m, axis=0) for b in base)
        calm, _ = self._run(U, pruning, start)
        out = np.empty((periods + 1, len(self.endogenous), k))
        keep = self._index(self.endogenous)
        for j in range(k):
            hit = U.copy()
            hit[0, :, j] += sizes[j]
            x, _ = self._run(hit, pruning, start)
            out[:, :, j] = (x - calm).mean(axis=1)[:, keep]
        kind = f"generalised, {replications} replications" if replications else "no other shocks"
        start = "from the given state" if state else "from the stochastic steady state"
        return IRF(out, list(self.endogenous), list(self.shocks), identification=f"model, third order ({kind}, {start})")

    def moments(self, periods: int = 100_000, seed: int | None = 0, burn: int = 1000, pruning: bool = True) -> Table:
        """Moments of the variables from a long simulation (pruned unless ``pruning=False``): mean, standard deviation,
        skewness, excess kurtosis and first-order autocorrelation; the mean in closed form beside it."""
        sim = self.simulate(periods, seed=seed, burn=burn, pruning=pruning)
        x = sim.to_numpy()
        mu, sd = x.mean(axis=0), x.std(axis=0)
        z = (x - mu) / np.where(sd > 0, sd, 1.0)
        skew, kurt = (z**3).mean(axis=0), (z**4).mean(axis=0) - 3.0
        ac = (z[1:] * z[:-1]).mean(axis=0)
        exact = self.mean() if pruning else pd.Series(np.nan, index=list(self.endogenous))
        rows = [[float(exact[v]), mu[i], sd[i], skew[i], kurt[i], ac[i]] for i, v in enumerate(self.endogenous)]
        return Table(["mean (closed form)", "mean", "std. dev.", "skewness", "excess kurtosis", "autocorrelation"], rows,
                     index=list(self.endogenous), title="Moments, third order (simulated)",
                     notes=[f"{periods:,} periods after {burn:,} dropped, {'pruned' if pruning else 'not pruned'}, seed {seed}"])

    def rules(self) -> Table:
        """The decision rules with uncertainty at the model's shock sizes: the coefficient of each product of states
        and shocks, the constant (the correction for risk) first; corrections for risk fold into the constant and the
        linear terms."""
        ns, k = len(self.states), len(self.shocks)
        labels = [f"{s}[-1]" for s in self.states] + list(self.shocks)
        nv = ns + k + 1
        coef: dict[tuple[int, ...], np.ndarray] = {}
        for p, G in enumerate(self.derivatives, start=1):
            for idx in itertools.combinations_with_replacement(range(nv), p):
                perms = factorial(p)
                for c in set(idx):
                    perms //= factorial(idx.count(c))
                key = tuple(i for i in idx if i < nv - 1)
                coef[key] = coef.get(key, 0.0) + G[(slice(None), *idx)] * perms / factorial(p)
        coef.setdefault((), np.zeros(len(self.variables)))
        keys = sorted(coef, key=lambda key: (len(key), key))
        keep_rows = self._index(self.endogenous)
        cols = [key for key in keys if np.any(np.abs(coef[key][keep_rows]) > 1e-12) or key == ()]
        names = ["constant" if not key else "*".join(labels[i] for i in key) for key in cols]
        rows = [[float(coef[key][r]) for key in cols] for r in keep_rows]
        return Table(names, rows, index=list(self.endogenous), title="Decision rules, third order (deviations from the steady state)",
                     notes=["x = constant + linear + quadratic + cubic terms in last period's states and the shocks; the constant "
                            "and the linear terms include the corrections for risk (ghs2/2, ghxss/2, ghuss/2)"])

    def __repr__(self) -> str:
        return f"<ThirdOrderSolution: {len(self.endogenous)} variables, {len(self.states)} states, {len(self.shocks)} shocks>"

    def _repr_html_(self) -> str:
        return self.rules()._repr_html_()


# -- the equations' derivatives --------------------------------------------------------------------------------------

def _derivatives(residuals: list[Any], variables: tuple[str, ...], shocks: tuple[str, ...], resolve: Any
                 ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    """The equations' first, second (dense) and third derivatives (sparse: equation, three positions, value, every
    ordering) with respect to z = [x(t+1), x(t), x(t-1), u(t)] at the steady state, by symbolic differentiation."""
    from ..expr import Name
    from ..expr.diff import diff
    from ..expr.evaluate import evaluate

    n, k = len(variables), len(shocks)
    N = 3 * n + k
    block = {1: 0, 0: n, -1: 2 * n}

    def position(ref: Any) -> int | None:
        if ref.id in shocks:
            return 3 * n + shocks.index(ref.id)
        if ref.id in variables:
            return block[ref.lag] + variables.index(ref.id)
        return None

    f1 = np.zeros((len(residuals), N))
    f2 = np.zeros((len(residuals), N, N))
    third: list[tuple[int, int, int, int, float]] = []
    for e, r in enumerate(residuals):
        refs = sorted({(ref.id, ref.lag) for ref in r.refs() if position(ref) is not None},
                      key=lambda ref: position(Name(*ref)))  # type: ignore[arg-type]
        pos = [position(Name(*ref)) for ref in refs]
        for a, ra in enumerate(refs):
            d1 = diff(r, Name(*ra))
            f1[e, pos[a]] = float(evaluate(d1, resolve))
            for b in range(a, len(refs)):
                d2 = diff(d1, Name(*refs[b]))
                v2 = float(evaluate(d2, resolve))
                f2[e, pos[a], pos[b]] = f2[e, pos[b], pos[a]] = v2
                for c in range(b, len(refs)):
                    v3 = float(evaluate(diff(d2, Name(*refs[c])), resolve))
                    if v3:
                        for i, j, l in set(itertools.permutations((pos[a], pos[b], pos[c]))):
                            third.append((e, i, j, l, v3))
    arr = np.array(third, dtype=float).reshape(-1, 5)
    f3 = (arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2].astype(int), arr[:, 3].astype(int), arr[:, 4])
    return f1, f2, f3


# -- tensor helpers --------------------------------------------------------------------------------------------------

def _kron(*rows: np.ndarray) -> np.ndarray:
    """Row by row Kronecker products of (paths x size) arrays."""
    out = rows[0]
    for r in rows[1:]:
        out = (out[:, :, None] * r[:, None, :]).reshape(len(out), -1)
    return out


def _modes(X: np.ndarray, M: np.ndarray, axes: Sequence[int]) -> np.ndarray:
    """X with M applied along each of ``axes`` (X[..., i, ...] -> sum_i X[..., i, ...] M[i, j])."""
    for ax in axes:
        X = np.moveaxis(np.tensordot(X, M, axes=([ax], [0])), -1, ax)
    return X


def _sym3(T: np.ndarray) -> np.ndarray:
    """T(i, j, l) + T(i, l, j) + T(j, l, i) for T symmetric in its first two indices (after the equation's)."""
    return T + T.transpose(0, 1, 3, 2) + T.transpose(0, 3, 1, 2)


def _kron_sylvester(D: np.ndarray, T: np.ndarray, F: np.ndarray, power: int, scale: complex = 1.0) -> np.ndarray:
    """Y + scale D Y (T kron ... kron T) = F with T upper triangular (``power`` factors), one factor at a time:
    for the first index i, Y_i + scale t_ii D Y_i T^(power-1) = F_i - scale D (sum_{i'<i} t_i'i Y_i') T^(power-1)."""
    n = D.shape[0]
    if power == 0:
        return np.linalg.solve(np.eye(n) + scale * D, F)
    ns = T.shape[0]
    rest = ns ** (power - 1)
    Fr = F.reshape(n, ns, rest)
    Y = np.zeros((ns, n * rest), dtype=complex)  # the first index first, so that the sums below are products
    for i in range(ns):
        rhs = Fr[:, i]
        if i:
            S = (T[:i, i] @ Y[:i]).reshape(n, rest)
            if power > 1:
                S = _modes(S.reshape((n,) + (ns,) * (power - 1)), T, range(1, power)).reshape(n, rest)
            rhs = rhs - scale * (D @ S)
        Y[i] = _kron_sylvester(D, T, rhs, power - 1, scale * T[i, i]).reshape(-1)
    return Y.reshape(ns, n, rest).transpose(1, 0, 2).reshape(n, -1)


# -- the solution ----------------------------------------------------------------------------------------------------

class _Expansion:
    """The expected residual's derivatives in v = (s, u, sigma) given the decision rule's (see the module docstring)."""

    def __init__(self, f1: np.ndarray, f2: np.ndarray, f3: tuple[np.ndarray, ...], L: list[int], n: int, k: int,
                 Sigma: np.ndarray) -> None:
        self.f1, self.f2, self.f3 = f1, f2, f3
        self.L, self.n, self.k, self.ns = L, n, k, len(L)
        self.nv = self.ns + k + 1
        # points with the shocks' mean (zero), covariance and third moments (zero): exact for cubic polynomials
        if k:
            w, V = np.linalg.eigh(Sigma)
            root = V * np.sqrt(np.clip(w, 0.0, None))
            self.nodes = [sign * np.sqrt(k) * root[:, i] for i in range(k) for sign in (1.0, -1.0)]
        else:
            self.nodes = [np.zeros(0)]
        nv, ns = self.nv, self.ns
        self.EL = np.zeros((n, nv))
        self.EL[L, np.arange(ns)] = 1.0
        self.Eu = np.zeros((k, nv))
        self.Eu[:, ns:ns + k] = np.eye(k)

    def residual(self, G: list[np.ndarray], p: int) -> np.ndarray:
        """E[d^p f / dv^p] with the rule's derivatives G[0..p-1] (G[p-1] partly known: the rest zero). Only next
        period's variables depend on the shocks then; what is linear in them is averaged before it is contracted."""
        n, ns, k, nv, L = self.n, self.ns, self.k, self.nv, self.L
        G1 = G[0]
        ghx = G1[:, :ns]
        f1, f2 = self.f1, self.f2
        m = len(self.nodes)
        out = np.zeros((f1.shape[0],) + (nv,) * p)
        X_sum = np.zeros((n,) + (nv,) * p)  # next period's variables' p-th derivatives, summed over the points
        cross = np.zeros((f1.shape[0],) + (nv,) * p)  # f_zz (z_vv kron z_v), before symmetrising
        for eps in self.nodes:
            M = np.zeros((nv, nv))
            M[:ns] = G1[L]
            M[ns:ns + k, nv - 1] = eps
            M[nv - 1, nv - 1] = 1.0
            Zv = np.vstack([G1 @ M, G1, self.EL, self.Eu])
            if p == 1:
                out += f1 @ Zv
                continue
            G2 = G[1]
            Xvv = _modes(G2, M, (1, 2))
            if p == 2:
                X_sum += Xvv
                out += (f2 @ Zv).transpose(0, 2, 1) @ Zv
                continue
            Xvv = Xvv + np.tensordot(ghx, G2[L], axes=1)
            G3 = G[2]
            X_sum += _modes(G3, M, (1, 2, 3)) + _sym3(np.tensordot(G2[:, :ns, :] @ M, G2[L], axes=([1], [0])).transpose(0, 2, 3, 1))
            H = f2[:, :2 * n] @ Zv  # n x 2n x nv: only x(t+1) and x(t) have second derivatives in v
            cross += np.tensordot(H[:, :n], Xvv, axes=([1], [0])) + np.tensordot(H[:, n:], G2, axes=([1], [0]))
            e, i, j, l, val = self.f3
            for eq in np.unique(e):
                sel = e == eq
                W = (val[sel, None] * Zv[i[sel]])[:, :, None] * Zv[j[sel]][:, None, :]
                out[eq] += (W.reshape(W.shape[0], -1).T @ Zv[l[sel]]).reshape(nv, nv, nv)
        if p == 1:
            return out / m
        Gp = G[p - 1]
        X = X_sum / m + np.tensordot(ghx, Gp[L], axes=1)
        out = out / m + np.tensordot(f1[:, :n], X, axes=1) + np.tensordot(f1[:, n:2 * n], Gp, axes=1)
        if p == 3:
            out += _sym3((cross / m).transpose(0, 2, 3, 1))
        return out


def _solve_order(ex: _Expansion, G: list[np.ndarray], p: int, A: np.ndarray, A0: np.ndarray, PL: np.ndarray) -> np.ndarray:
    """The p-th derivatives of the rule, block by block in the number of sigma derivatives (see the module docstring)."""
    n, ns, k, nv = ex.n, ex.ns, ex.k, ex.nv
    lu = linalg.lu_factor(A)
    D = linalg.lu_solve(lu, A0)
    T, U = linalg.schur(PL.astype(complex), output="complex") if ns else (np.zeros((0, 0)), np.zeros((0, 0)))
    S, sig = list(range(ns)), [nv - 1]
    kinds = np.array([0] * ns + [1] * k + [2])
    grids = np.meshgrid(*([kinds] * p), indexing="ij")
    count_u = sum((g == 1).astype(int) for g in grids)
    count_sig = sum((g == 2).astype(int) for g in grids)
    Gp = np.zeros((n,) + (nv,) * p)
    work = [*G, Gp]
    for c in range(0, p + 1, 2):  # odd numbers of sigma derivatives are zero: the shocks are symmetric
        a = p - c
        R = ex.residual(work, p)
        lists = [list(range(n))] + [S] * a + [sig] * c
        sub = R[np.ix_(*lists)].reshape(n, ns**a)
        if a == 0:
            X = -np.linalg.solve(A + A0, sub)
        else:  # X + D X PL^(kron a) = E, on the Schur form PL = U T U*
            E = -linalg.lu_solve(lu, sub)
            F = _modes(E.reshape((n,) + (ns,) * a).astype(complex), U, range(1, a + 1)).reshape(n, -1)
            Y = _kron_sylvester(D, T, F, a)
            X = _modes(Y.reshape((n,) + (ns,) * a), U.conj().T, range(1, a + 1)).real.reshape(n, -1)
        Xt = X.reshape((n,) + (ns,) * a + (1,) * c)
        for perm in set(itertools.permutations(range(p))):
            Gp[np.ix_(*([lists[0]] + [lists[1 + q] for q in perm]))] = Xt.transpose([0] + [1 + q for q in perm])
        if a and k:  # the blocks with shocks: A X = -R
            R = ex.residual(work, p)
            mask = ((count_sig == c) & (count_u > 0)).reshape(-1)
            flat = Gp.reshape(n, -1)
            flat[:, mask] = -linalg.lu_solve(lu, R.reshape(n, -1)[:, mask])
            Gp = flat.reshape(Gp.shape)
            work[-1] = Gp
    return Gp


def third_order(model: Any, shocks: Sequence[str], shock_sd: Mapping[str, float] | None = None, steady: Any = None,
                guess: Mapping[str, float] | None = None, shock_corr: Mapping[Any, float] | None = None) -> ThirdOrderSolution:
    """The third-order solution of ``model`` (a Model) with these shocks and sizes (and correlations, ``shock_corr``,
    as Dynare's ``Sigma_e``): see the module docstring."""
    from .covariance import shock_covariance
    from .linear import _residual_system, linearize

    shocks = tuple(shocks)
    ss = steady if steady is not None else model.steady_state(guess=guess)
    if getattr(ss, "growth", None):
        raise ModelError("third-order solutions of models with growth are not available yet; write the model in ratios")
    lin = linearize(model, ss, shocks)
    first = lin.solve()
    residuals, variables, endo, _, resolve = _residual_system(model, ss, shocks)
    n, k = len(variables), len(shocks)
    P, Q = first.P, first.Q
    L = [j for j in range(n) if np.any(lin.A2[:, j] != 0)]
    ns = len(L)
    sizes = np.array([float((shock_sd or {}).get(s, 1.0)) for s in shocks])
    Sigma = np.diag(sizes**2) if not shock_corr else shock_covariance(shocks, sizes, shock_corr)
    f1, f2, f3 = _derivatives(residuals, variables, shocks, resolve)
    A = lin.A0 @ P + lin.A1
    try:
        linalg.lu_factor(A, check_finite=True)
        if np.linalg.cond(A) > 1e14:
            raise np.linalg.LinAlgError
    except (ValueError, np.linalg.LinAlgError):
        raise ModelError("A0 P + A1 is singular: the higher-order terms are not determined") from None
    if np.linalg.cond(A + lin.A0) > 1e14:
        raise ModelError("A0 P + A0 + A1 is singular: the corrections for risk are not determined")
    ex = _Expansion(f1, f2, f3, L, n, k, Sigma)
    nv = ns + k + 1
    G1 = np.zeros((n, nv))
    G1[:, :ns], G1[:, ns:ns + k] = P[:, L], Q
    PL = P[np.ix_(L, L)]
    G2 = _solve_order(ex, [G1], 2, A, lin.A0, PL)
    G3 = _solve_order(ex, [G1, G2], 3, A, lin.A0, PL)
    S, U, s = slice(0, ns), slice(ns, ns + k), nv - 1

    def flat(G: np.ndarray, *blocks: Any) -> np.ndarray:
        return G[(slice(None), *blocks)].reshape(n, -1)

    return ThirdOrderSolution(
        variables, tuple(endo), tuple(variables[j] for j in L), shocks, P[:, L], Q,
        flat(G2, S, S), flat(G2, S, U), flat(G2, U, U), G2[:, s, s].copy(),
        flat(G3, S, S, S), flat(G3, S, S, U), flat(G3, S, U, U), flat(G3, U, U, U),
        flat(G3, S, s, s), flat(G3, U, s, s), G3[:, s, s, s].copy(), Sigma, ss, first, (G1, G2, G3))
