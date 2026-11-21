"""Second-order perturbation (Schmitt-Grohe and Uribe 2004), as Dynare's ``stoch_simul(order=2)``.

    sol2 = rbc.solve(order=2)            # a template, a Dynare or IRIS model; or second_order(model, shocks, shock_sd)
    sol2.ghxx, sol2.ghxu, sol2.ghuu, sol2.ghs2   # Dynare's names for the second-order terms
    sol2.mean()                          # the ergodic mean: the steady state plus the risk correction
    sol2.simulate(200, seed=1)           # a stochastic simulation, pruned (Kim et al. 2008)
    sol2.irf(20)                         # responses from the stochastic steady state
    sol2.particle_filter(data, measurement)   # likelihood and filtered states (pymetrix.models.particle)

The decision rule, in deviations from the steady state, with ``s`` the predetermined variables last period (the
states), ``u`` the shocks and ``sigma`` scaling uncertainty (1 for the model's shock sizes):

    x(t) = ghx s + ghu u + 1/2 ghxx (s kron s) + ghxu (s kron u) + 1/2 ghuu (u kron u) + 1/2 ghs2

``ghs2`` is the correction for risk: the effect of future uncertainty on today's decisions (precaution, risk
premia); certainty equivalence holds to first order but not here. The second-order terms solve, from the model's
second derivatives (exact, symbolic), first a generalised Sylvester equation for ``ghxx`` (solved column by
column on the Schur form of the state transition, so that it scales), then linear systems for ``ghxu``, ``ghuu``
and ``ghs2``. Simulations are pruned: the second-order terms act on the first-order part of the state only, which
keeps paths stable (Dynare's ``pruning``); the mean is that of the pruned process.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg

from ..output import Table
from .solvers import ModelError
from .._random import generator as make_rng

__all__ = ["SecondOrderSolution", "second_order"]


@dataclass(frozen=True, eq=False)
class SecondOrderSolution:
    """A second-order approximation of a model's decision rules: see the module docstring."""

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
    sigma: np.ndarray  # covariance of the shocks
    steady: Any  # SteadyState
    first: Any  # the first-order LinearSolution

    def _index(self, names: Sequence[str]) -> list[int]:
        return [self.variables.index(v) for v in names]

    def _step(self, sf: np.ndarray, ss_: np.ndarray, u: np.ndarray, sigma2: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """One pruned step: (first-order part, second-order part) of x(t) from those parts of the states and shocks."""
        xf = self.ghx @ sf + self.ghu @ u
        xs = (self.ghx @ ss_ + 0.5 * self.ghxx @ np.kron(sf, sf) + self.ghxu @ np.kron(sf, u)
              + 0.5 * self.ghuu @ np.kron(u, u) + 0.5 * sigma2 * self.ghs2)
        return xf, xs

    def mean(self) -> pd.Series:
        """The ergodic mean of the pruned second-order process, in levels: the steady state plus the correction for
        risk (``E[x] = (I - ghx_s)^-1 (1/2 ghxx vec(V) + 1/2 ghuu vec(Sigma) + 1/2 ghs2)``, V the first-order
        variance of the states)."""
        st = self._index(self.states)
        A = self.ghx[st, :]  # states on states
        V = linalg.solve_discrete_lyapunov(A, self.ghu[st, :] @ self.sigma @ self.ghu[st, :].T) if st else np.zeros((0, 0))
        drift = 0.5 * (self.ghxx @ V.reshape(-1) + self.ghuu @ self.sigma.reshape(-1) + self.ghs2)
        Es = np.linalg.solve(np.eye(len(st)) - A, drift[st]) if st else np.zeros(0)
        Ex = self.ghx @ Es + drift
        return pd.Series({v: self.steady.values[v] + Ex[self.variables.index(v)] for v in self.endogenous}, name="mean")

    def simulate(self, periods: int, seed: int | None = None, shocks: np.ndarray | None = None, burn: int = 0) -> pd.DataFrame:
        """A pruned stochastic simulation in levels, from the steady state (``shocks``: a periods x shocks array to use
        instead of draws; ``burn`` periods dropped first)."""
        rng = make_rng(seed)
        k = len(self.shocks)
        chol = np.linalg.cholesky(self.sigma) if k else np.zeros((0, 0))
        U = shocks if shocks is not None else rng.standard_normal((periods + burn, k)) @ chol.T
        st = self._index(self.states)
        sf, ss_ = np.zeros(len(st)), np.zeros(len(st))
        rows = []
        for t in range(len(U)):
            xf, xs = self._step(sf, ss_, U[t])
            sf, ss_ = xf[st], xs[st]
            rows.append(xf + xs)
        out = pd.DataFrame(np.array(rows)[burn:, self._index(self.endogenous)], columns=list(self.endogenous))
        return out + pd.Series({v: self.steady.values[v] for v in self.endogenous})

    def irf(self, periods: int = 20, shock_sd: Mapping[str, float] | None = None) -> Any:
        """Responses to one-standard-deviation shocks (or ``shock_sd``) from the stochastic steady state (where the
        model rests when shocks are expected but none occur), as differences from staying there; an ``IRF``.
        Unlike first-order responses they depend on the size and sign of the shock."""
        from ..var import IRF

        st = self._index(self.states)
        sizes = np.array([float((shock_sd or {}).get(s, np.sqrt(self.sigma[i, i]))) for i, s in enumerate(self.shocks)])
        # the stochastic steady state: iterate the pruned rule with no shocks until it settles
        sf, ss_ = np.zeros(len(st)), np.zeros(len(st))
        for _ in range(2000):
            xf, xs = self._step(sf, ss_, np.zeros(len(self.shocks)))
            new = xs[st]
            if np.allclose(new, ss_, atol=1e-12, rtol=0):
                break
            sf, ss_ = xf[st], new
        base_f, base_s = sf, ss_
        out = np.empty((periods + 1, len(self.endogenous), len(self.shocks)))
        keep = self._index(self.endogenous)
        for j in range(len(self.shocks)):
            u = np.zeros(len(self.shocks))
            u[j] = sizes[j]
            f, s2 = base_f.copy(), base_s.copy()
            calm_f, calm_s = base_f.copy(), base_s.copy()
            for h in range(periods + 1):
                xf, xs = self._step(f, s2, u if h == 0 else np.zeros(len(self.shocks)))
                cf, cs = self._step(calm_f, calm_s, np.zeros(len(self.shocks)))
                out[h, :, j] = (xf + xs - cf - cs)[keep]
                f, s2, calm_f, calm_s = xf[st], xs[st], cf[st], cs[st]
        return IRF(out, list(self.endogenous), list(self.shocks), identification="model, second order (from the stochastic steady state)")

    def rules(self) -> Table:
        """The decision rules, Dynare style: the constant (half ghs2), then each term, for the model's variables."""
        cols = ["constant", *[f"{s}[-1]" for s in self.states], *self.shocks]
        st, k = self.states, self.shocks
        cols += [f"{a}[-1]*{b}[-1]" for i, a in enumerate(st) for j, b in enumerate(st) if j >= i]
        cols += [f"{a}[-1]*{u}" for a in st for u in k]
        cols += [f"{u}*{v}" for i, u in enumerate(k) for j, v in enumerate(k) if j >= i]
        rows = []
        n_s, n_k = len(st), len(k)
        for v in self.endogenous:
            r = self.variables.index(v)
            row = [0.5 * self.ghs2[r], *self.ghx[r], *self.ghu[r]]
            G = self.ghxx[r].reshape(n_s, n_s)
            row += [0.5 * (G[i, j] + (G[j, i] if j != i else 0.0)) if j != i else 0.5 * G[i, i] for i in range(n_s) for j in range(n_s) if j >= i]
            row += list(self.ghxu[r])
            U = self.ghuu[r].reshape(n_k, n_k)
            row += [0.5 * (U[i, j] + U[j, i]) if j != i else 0.5 * U[i, i] for i in range(n_k) for j in range(n_k) if j >= i]
            rows.append(row)
        keep = [c for c in range(len(cols)) if any(abs(r[c]) > 1e-12 for r in rows)]
        return Table([cols[c] for c in keep], [[r[c] for c in keep] for r in rows], index=list(self.endogenous),
                     title="Decision rules, second order (deviations from the steady state)",
                     notes=["x = constant + linear terms + quadratic terms; the constant is the correction for risk (ghs2/2)"])

    def particle_filter(self, data: Any, measurement: Any, span: Any = None, particles: int = 2000, seed: int | None = 0,
                        resample: float = 0.5, proposal: str = "bootstrap", burn: int | None = None, presample: int = 0,
                        quantiles: Sequence[float] = (0.05, 0.5, 0.95), measurement_sd: Mapping[str, float] | None = None,
                        smooth_paths: int = 0, candidates: int | None = 200) -> Any:
        """The particle filter on ``data`` through ``measurement`` (each observed series with a measurement error): the
        log likelihood estimate and the filtered variables, following the pruned rules as ``simulate`` does;
        ``smooth_paths`` > 0 adds the particle smoother's paths (a ``ParticleFilter``; see ``pymetrix.models.particle``)."""
        from .particle import particle_filter

        return particle_filter(self, data, measurement, None, span, particles, seed, resample, proposal, burn, presample,
                               quantiles, None, measurement_sd, smooth_paths, candidates)

    def __repr__(self) -> str:
        return f"<SecondOrderSolution: {len(self.endogenous)} variables, {len(self.states)} states, {len(self.shocks)} shocks>"

    def _repr_html_(self) -> str:
        return self.rules()._repr_html_()


def _hessians(residuals: list[Any], variables: tuple[str, ...], shocks: tuple[str, ...], resolve: Any) -> list[np.ndarray]:
    """For each equation, the second derivatives with respect to z = [x(t+1), x(t), x(t-1), u(t)] at the steady
    state (exact: symbolic differentiation)."""
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

    out = []
    for r in residuals:
        H = np.zeros((N, N))
        refs = [ref for ref in r.refs() if position(ref) is not None]
        for a, ra in enumerate(refs):
            first = diff(r, Name(ra.id, ra.lag))
            for rb in refs[a:]:
                value = float(evaluate(diff(first, Name(rb.id, rb.lag)), resolve))
                if value:
                    i, j = position(ra), position(rb)
                    H[i, j] = H[j, i] = value
        out.append(H)
    return out


def second_order(model: Any, shocks: Sequence[str], shock_sd: Mapping[str, float] | None = None, steady: Any = None,
                 guess: Mapping[str, float] | None = None, shock_corr: Mapping[Any, float] | None = None) -> SecondOrderSolution:
    """The second-order solution of ``model`` (a Model) with these shocks and sizes (and correlations, ``shock_corr``:
    the risk terms take the full covariance, as Dynare's ``Sigma_e``): see the module docstring."""
    from .covariance import shock_covariance
    from .linear import _residual_system, linearize

    shocks = tuple(shocks)
    ss = steady if steady is not None else model.steady_state(guess=guess)
    if getattr(ss, "growth", None):
        raise ModelError("second-order solutions of models with growth are not available yet; write the model in ratios")
    lin = linearize(model, ss, shocks)
    first = lin.solve()
    residuals, variables, endo, _, resolve = _residual_system(model, ss, shocks)
    n, k = len(variables), len(shocks)
    P, Q = first.P, first.Q
    L = [j for j in range(n) if np.any(lin.A2[:, j] != 0)]
    ns = len(L)
    PL, QL = P[np.ix_(L, L)], Q[L, :]
    sizes = np.array([float((shock_sd or {}).get(s, 1.0)) for s in shocks])
    Sigma = np.diag(sizes**2) if not shock_corr else shock_covariance(shocks, sizes, shock_corr)
    H = _hessians(residuals, variables, shocks, resolve)

    EL = np.zeros((n, ns))
    EL[L, np.arange(ns)] = 1.0
    # how z = [x(t+1), x(t), x(t-1), u(t)] moves with the states s and shocks u (first order)
    Ms = np.vstack([P @ P[:, L], P[:, L], EL, np.zeros((k, ns))])
    Mu = np.vstack([P @ Q, Q, np.zeros((n, k)), np.eye(k)])

    def contract(Ma: np.ndarray, Mb: np.ndarray) -> np.ndarray:  # f_zz (Ma kron Mb), equation by equation
        return np.array([(Ma.T @ Hi @ Mb).reshape(-1) for Hi in H]) if H else np.zeros((0, Ma.shape[1] * Mb.shape[1]))

    A = lin.A0 @ P + lin.A1
    try:
        lu = linalg.lu_factor(A)
    except (ValueError, np.linalg.LinAlgError):
        raise ModelError("A0 P + A1 is singular: the second-order terms are not determined") from None

    # ghxx: A X + A0 X (PL kron PL) = -f_zz(Ms kron Ms); with D = A^-1 A0: X + D X C = E
    D = linalg.lu_solve(lu, lin.A0)
    E = -linalg.lu_solve(lu, contract(Ms, Ms))
    ghxx = _sylvester(D, PL, E, ns)
    G = ghxx.reshape(n, ns, ns)
    # ghxu: A X = -(f_zz(Ms kron Mu) + A0 ghxx (PL kron QL))
    cross = np.einsum("ipq,pa,qb->iab", G, PL, QL).reshape(n, ns * k)
    ghxu = -linalg.lu_solve(lu, contract(Ms, Mu) + lin.A0 @ cross)
    # ghuu: A X = -(f_zz(Mu kron Mu) + A0 ghxx (QL kron QL))
    quad = np.einsum("ipq,pa,qb->iab", G, QL, QL).reshape(n, k * k)
    ghuu = -linalg.lu_solve(lu, contract(Mu, Mu) + lin.A0 @ quad)
    # ghs2: (A0 P + A0 + A1) X = -(A0 ghuu vec(Sigma) + f_{x+x+}(Q kron Q) vec(Sigma))
    Mq = np.vstack([Q, np.zeros((2 * n + k, k))])
    risk = lin.A0 @ ghuu @ Sigma.reshape(-1) + contract(Mq, Mq) @ Sigma.reshape(-1)
    try:
        ghs2 = -np.linalg.solve(A + lin.A0, risk)
    except np.linalg.LinAlgError:
        raise ModelError("A0 P + A0 + A1 is singular: the correction for risk is not determined") from None
    return SecondOrderSolution(variables, tuple(endo), tuple(variables[j] for j in L), shocks, P[:, L], Q, ghxx, ghxu, ghuu,
                               ghs2, Sigma, ss, first)


def _sylvester(D: np.ndarray, PL: np.ndarray, E: np.ndarray, ns: int) -> np.ndarray:
    """X + D X (PL kron PL) = E for X (n x ns^2), column by column on the complex Schur form PL = U T U*: with
    Y = X (U kron U) and T kron T upper triangular, (I + t_jj D) y_j = e_j - D sum_{i<j} t_ij y_i."""
    n = D.shape[0]
    if ns == 0:
        return np.zeros((n, 0))
    T, U = linalg.schur(PL.astype(complex), output="complex")
    W = np.kron(U, U)
    TT = np.kron(T, T)
    F = E.astype(complex) @ W
    Y = np.zeros((n, ns * ns), dtype=complex)
    I = np.eye(n)
    DY = np.zeros((n, ns * ns), dtype=complex)  # D y_i, kept for the sums
    for j in range(ns * ns):
        rhs = F[:, j] - DY[:, :j] @ TT[:j, j]
        Y[:, j] = np.linalg.solve(I + TT[j, j] * D, rhs)
        DY[:, j] = D @ Y[:, j]
    X = Y @ W.conj().T
    return np.real_if_close(X, tol=1e6).real
