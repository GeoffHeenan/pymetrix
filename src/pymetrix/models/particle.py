"""The particle filter: the likelihood and filtered states of a model solved to second order (or to first, for checks).

    sol2 = rbc.solve(order=2)                                    # a SecondOrderSolution (pruned, see perturbation.py)
    pf = sol2.particle_filter(data, measurement, particles=5000, seed=1)
    pf.loglike, pf.loglike_path                                  # the log likelihood estimate, and by period
    pf.filtered                                                  # the model's variables from data up to each period (means)
    pf.quantiles                                                 # their weighted quantiles (5%, 50%, 95% by default)
    pf.ess, pf.resampled                                         # the effective sample size by period; when it resampled
    fit = rbc.estimate(data, order=2, filter="particle", particles=2000)   # estimation on the particle filter's likelihood
    post = fit.mcmc(draws=2000)                                  # particle marginal Metropolis-Hastings (pseudo-marginal)

**The state-space form.** A second-order solution is simulated pruned (Kim, Kim, Schaumburg and Sims 2008;
Andreasen, Fernández-Villaverde and Rubio-Ramírez 2018): each particle carries the first-order part ``sf`` and the
second-order part ``ss`` of the states, and

    xf(t) = ghx sf(t-1) + ghu u(t)
    xs(t) = ghx ss(t-1) + 1/2 ghxx (sf kron sf) + ghxu (sf kron u) + 1/2 ghuu (u kron u) + 1/2 ghs2
    x(t)  = xf(t) + xs(t)    (deviations from the steady state)

exactly as ``SecondOrderSolution.simulate`` does, so a path the filter follows is one the simulation can produce.
The observed series are the measurement equations evaluated exactly on each particle (in the data's units, with
their deterministic trends; ``transforms`` make the observed series from raw data first, as for the Kalman filter;
lags such as ``y[-1]`` are carried on each particle), plus Gaussian measurement errors: every observed series needs
one (``MeasurementMap(..., errors=...)``, or estimated as ``sd_<series>``), since the bootstrap filter weighs
particles by the measurement density. A first-order solution (``LinearSolution.particle_filter``) gives the same filter
with the second-order terms left out: with linear measurement equations it estimates the Kalman filter's likelihood,
a check on both.

**The filter.** Bootstrap (sampling importance resampling, Gordon, Salmond and Smith 1993): particles move by the
model with drawn shocks and are weighted by how likely each makes the period's observations (series missing in a
period are left out of its weight). The log likelihood is the sum over periods of the log of the average weight; it is
an unbiased estimate of the likelihood itself (its log is biased down by about half its variance). Particles are
resampled (systematic resampling) when the effective sample size ``1 / sum(w^2)`` falls below ``resample`` times
their number (0.5 by default; 1 resamples every period). ``proposal="guided"`` draws each particle's shocks from
the conditional normal implied by the measurement equations linearised at the steady state, given that particle's
predicted observations (the guided filter of Herbst and Schorfheide 2015, ch. 8), with weights corrected exactly, so
the estimate stays unbiased; it helps when measurement errors are small relative to the shocks. The filter starts
from the first-order stationary distribution of the states (and the ergodic mean of the second-order part), then runs
``burn`` periods without data (default: none at first order, 50 at second order, so that the second-order part
settles into its own distribution) and as many more as the measurement equations' lags need.

**Random numbers.** Every draw comes from ``pymetrix._random.generator(seed)``, and each period takes the same
draws whatever the parameters (normal draws for the shocks and one uniform for resampling, drawn whether or not it
resamples): common random numbers, so the likelihood moves smoothly enough with the parameters for a mode search.
It is still noisy, and not smooth where a resampling decision or a resampled particle changes: increase
``particles`` until ``loglike_sd`` (the standard deviation over seeds, ``loglike_spread``) is well below one.
Estimation uses this likelihood at a fixed seed for the mode; ``FittedModel.mcmc`` then draws fresh random numbers
at each proposal and keeps the current draw's estimate (pseudo-marginal, Andrieu, Doucet and Holenstein 2010), so its
draws are from the exact posterior whatever the noise, though the chain mixes worse as the noise grows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg

from ..errors import SpecificationError
from ..output import Table
from .solvers import ModelError
from .._random import generator as make_rng, get_random_generator

__all__ = ["ParticleFilter", "particle_filter", "PROPOSALS"]

PROPOSALS = ("bootstrap", "guided")
_LOG2PI = float(np.log(2 * np.pi))


@dataclass(frozen=True, eq=False)
class ParticleFilter:
    """A particle filter's result: see the module docstring."""

    loglike: float  # the log likelihood estimate (periods after the presample)
    loglike_path: pd.Series  # each period's contribution
    filtered: Any  # Dataset: the model's variables (levels), each observed series as fitted (<series>_fitted), reported series
    quantiles: pd.DataFrame  # columns (quantile, variable): weighted quantiles of the filtered variables
    ess: pd.Series  # effective sample size in each period, before any resampling
    resampled: pd.Series  # True where the particles were resampled
    particles: int
    order: int  # the solution's order: 1 or 2
    proposal: str = "bootstrap"
    observables: tuple[str, ...] = ()
    seed: int | None = None
    provenance: Any = None
    smoothed: Any = None  # Dataset: the particle smoother's means (data of every period), as ``filtered``; None without it
    smoothed_quantiles: pd.DataFrame | None = None  # columns (quantile, variable): quantiles over the smoother's paths
    smoothed_shocks: pd.DataFrame | None = None  # the shocks' smoothed means
    smooth_paths: int = 0  # paths the smoother drew (0: not smoothed)

    def summary(self) -> Table:
        ess = self.ess.to_numpy(float)
        rows = [["log likelihood", f"{self.loglike:.4f}"], ["particles", f"{self.particles:,}"],
                ["solution", "second order, pruned" if self.order == 2 else "first order"],
                ["proposal", self.proposal], ["periods", str(len(self.ess))],
                ["effective sample size", f"mean {np.mean(ess):,.0f}, lowest {np.min(ess):,.0f}"],
                ["periods resampled", f"{int(self.resampled.sum())}"], ["seed", str(self.seed)],
                ["smoothed", f"{self.smooth_paths:,} paths (backward simulation)" if self.smooth_paths else "no (filtered means only)"]]
        return Table(["value"], [[r[1]] for r in rows], index=[r[0] for r in rows], title="Particle filter",
                     notes=[f"Observed: {', '.join(self.observables)}",
                            "The log likelihood is an estimate: run again with other seeds (loglike_spread) to see its noise"])

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        """The filtered means, or one of the result's views (``"Quantiles"``, ``"Diagnostics"``, ...) as a frame."""
        if view is not None:
            from ..views import _views_to_frame

            return _views_to_frame(self, view)
        return self.filtered.frame

    def __repr__(self) -> str:
        return f"<ParticleFilter: {self.particles} particles, order {self.order}, log likelihood {self.loglike:.3f}>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


# --- the pruned decision rules on particles -------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class _Rules:
    variables: tuple[str, ...]
    endogenous: tuple[str, ...]
    st: np.ndarray  # positions of the states among the variables
    ghx: np.ndarray
    ghu: np.ndarray
    ghxx: np.ndarray | None
    ghxu: np.ndarray | None
    ghuu: np.ndarray | None
    ghs2: np.ndarray | None
    sigma: np.ndarray
    steady: Mapping[str, float]
    order: int


def _steady(values: Mapping[str, float], variables: Sequence[str]) -> dict[str, float]:
    steady = {k: float(v) for k, v in values.items()}
    for v in variables:  # auxiliary variables equal their variable in the steady state
        steady.setdefault(v, steady.get(v.rsplit("_", 1)[0], 0.0))
    return steady


def rules_of(solution: Any, shock_sd: Mapping[str, float] | None = None, shock_corr: Mapping[Any, float] | None = None) -> _Rules:
    """The decision rules of a ``SecondOrderSolution`` (its own shock covariance) or a ``LinearSolution`` (with
    ``shock_sd`` and ``shock_corr``)."""
    from .perturbation import SecondOrderSolution

    if isinstance(solution, SecondOrderSolution):
        if shock_sd or shock_corr:
            raise SpecificationError("a second-order solution has its shock sizes built in (its correction for risk depends on "
                                     "them): solve it again with others instead of giving shock_sd or shock_corr")
        st = np.array(solution._index(solution.states), dtype=int)
        return _Rules(tuple(solution.variables), tuple(solution.endogenous), st, solution.ghx, solution.ghu, solution.ghxx,
                      solution.ghxu, solution.ghuu, solution.ghs2, np.asarray(solution.sigma, float),
                      _steady(solution.steady.values, solution.variables), 2)
    if not hasattr(solution, "P") or not hasattr(solution, "Q"):
        raise SpecificationError("the particle filter runs on a SecondOrderSolution or a first-order LinearSolution")
    from .covariance import normalise, shock_covariance

    shocks = tuple(solution.shocks)
    sizes = np.array([float((shock_sd or {}).get(s, 1.0)) for s in shocks])
    corr = {k: v for k, v in normalise(shocks, shock_corr).items() if v != 0}
    sigma = shock_covariance(shocks, sizes, corr) if corr else np.diag(sizes**2)
    lagged = np.flatnonzero(np.any(solution.P != 0, axis=0))
    if getattr(solution.model.steady, "growth", None):
        raise ModelError("the particle filter is for models without growth trends: write the model in ratios")
    return _Rules(tuple(solution.variables), tuple(solution.endogenous), lagged, solution.P[:, lagged], solution.Q, None, None,
                  None, None, sigma, _steady(solution.model.steady.values, solution.variables), 1)


def _psd_factor(V: np.ndarray) -> np.ndarray:
    """A factor F with F F' = V for a positive semi-definite V."""
    if V.size == 0:
        return V
    w, U = np.linalg.eigh(0.5 * (V + V.T))
    return U * np.sqrt(np.clip(w, 0.0, None))


class _Engine:
    """The filter's moving parts for one solution and measurement map: particles' steps and predicted observations."""

    def __init__(self, rules: _Rules, measurement: Any) -> None:
        self.r = rules
        n = len(rules.variables)
        self.n, self.k, self.ns = n, rules.ghu.shape[1], len(rules.st)
        missing = sorted(measurement.variables() - set(rules.variables))
        if missing:
            raise ModelError(f"{', '.join(missing)} are not variables of the model")
        if any(r != 0 for r in (getattr(measurement, "error_corr", None) or {}).values()):
            raise ModelError("the particle filter weighs each series' measurement error on its own: correlated measurement errors "
                             "(error_corr) are not supported here; use the Kalman filter")
        self.measurement = measurement
        params = dict(measurement.params)
        self.exprs = [m.rhs.substitute(params) for m in measurement.measurements]
        self.reports = [(r.observable, r.rhs.substitute(params)) for r in measurement.reports]
        lags = [-ref.lag for e in [*self.exprs, *(e for _, e in self.reports)] for ref in e.refs()]
        self.L = max(lags, default=0)
        self.position = {v: i for i, v in enumerate(rules.variables)}
        keys = [(v, -lag) for lag in range(self.L + 1) for v in rules.variables]
        Z, d = measurement.linearize(keys, rules.steady) if self.exprs else (np.zeros((0, len(keys))), np.zeros(0))
        self.Z, self.d = Z, d
        self.Z0 = Z[:, :n]  # on the current period's variables
        self.sbar = np.array([rules.steady[v] for v in rules.variables])
        self.linear = self._is_linear()
        A = rules.ghx[rules.st, :] if self.ns else np.zeros((0, 0))
        if self.ns and float(np.max(np.abs(np.linalg.eigvals(A)))) >= 1 - 1e-9:
            raise ModelError("the particle filter starts from the stationary distribution of the states: this solution has unit or "
                             "explosive roots")
        Bu = rules.ghu[rules.st, :]
        self.V = linalg.solve_discrete_lyapunov(A, Bu @ rules.sigma @ Bu.T) if self.ns else np.zeros((0, 0))
        self.Es = np.zeros(self.ns)
        if rules.order == 2 and self.ns:
            drift = 0.5 * (rules.ghxx @ self.V.reshape(-1) + rules.ghuu @ rules.sigma.reshape(-1) + rules.ghs2)
            self.Es = np.linalg.solve(np.eye(self.ns) - A, drift[rules.st])
        self.chol = _psd_factor(rules.sigma)

    def _is_linear(self) -> bool:
        """Whether the measurement equations are linear in the model's variables (then observe by matrices)."""
        if not self.exprs:
            return True
        for probe in (0.37, -0.61):  # two fixed, irregular points (not random draws)
            lags = [0.1 * np.sin(probe * (1 + np.arange(self.n) + 7 * j))[None, :] for j in range(self.L + 1)]
            exact = self._evaluate(self.exprs, lags)
            approx = self._linear(lags)
            if not np.allclose(exact, approx, rtol=1e-9, atol=1e-9 * (1 + np.max(np.abs(approx)))):
                return False
        return True

    def _linear(self, lags: Sequence[np.ndarray]) -> np.ndarray:
        n = self.n
        out = np.broadcast_to(self.d, (lags[0].shape[0], len(self.d))).copy()
        for j, X in enumerate(lags):
            out += X @ self.Z[:, j * n:(j + 1) * n].T
        return out

    def _evaluate(self, exprs: Sequence[Any], lags: Sequence[np.ndarray]) -> np.ndarray:
        from ..expr import evaluate

        N = lags[0].shape[0]
        sbar, position = self.sbar, self.position

        def resolve(name: str, lag: int) -> np.ndarray:
            i = position[name]
            return sbar[i] + lags[-lag][:, i]

        if not exprs:
            return np.zeros((N, 0))
        return np.column_stack([np.broadcast_to(np.asarray(evaluate(e, resolve), float), (N,)) for e in exprs])

    def observe(self, lags: Sequence[np.ndarray]) -> np.ndarray:
        """Predicted observations (particles x observables), before trends and measurement errors."""
        return self._linear(lags) if self.linear else self._evaluate(self.exprs, lags)

    def step(self, sf: np.ndarray, ss: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One pruned step for every particle: the first- and second-order parts of x(t) (particles x variables)."""
        r = self.r
        xf = sf @ r.ghx.T + u @ r.ghu.T
        if r.order == 1:
            return xf, None  # type: ignore[return-value]
        N = sf.shape[0]
        xs = ss @ r.ghx.T + 0.5 * r.ghs2
        if self.ns:
            xs += 0.5 * np.einsum("ni,nj->nij", sf, sf).reshape(N, -1) @ r.ghxx.T
            xs += np.einsum("ni,nj->nij", sf, u).reshape(N, -1) @ r.ghxu.T
        xs += 0.5 * np.einsum("ni,nj->nij", u, u).reshape(N, -1) @ r.ghuu.T
        return xf, xs


def _systematic(W: np.ndarray, u: float) -> np.ndarray:
    N = len(W)
    c = np.cumsum(W)
    c[-1] = 1.0
    return np.minimum(np.searchsorted(c, (u + np.arange(N)) / N), N - 1)


def _weighted_quantiles(X: np.ndarray, W: np.ndarray, qs: Sequence[float]) -> np.ndarray:
    """Weighted quantiles of each column of X (particles x columns): quantiles x columns."""
    order = np.argsort(X, axis=0)
    xs = np.take_along_axis(X, order, axis=0)
    cw = np.cumsum(W[order], axis=0)
    out = np.empty((len(qs), X.shape[1]))
    for j in range(X.shape[1]):
        idx = np.minimum(np.searchsorted(cw[:, j], qs), X.shape[0] - 1)
        out[:, j] = xs[idx, j]
    return out


def run(engine: _Engine, y: np.ndarray, errors: np.ndarray, trends: np.ndarray, particles: int, seed: Any,
        resample: float = 0.5, proposal: str = "bootstrap", burn: int | None = None, presample: int = 0,
        keep: Sequence[float] | None = None, smooth_paths: int = 0, candidates: int | None = None) -> dict[str, Any]:
    """The filter on observations ``y`` (periods x observables, NaN where missing) with measurement-error standard
    deviations ``errors`` and deterministic ``trends`` (periods x observables). The log likelihood, each period's
    contribution, the effective sample sizes and resampling; with ``keep`` (quantiles), the filtered means and those
    quantiles of the model's variables, observed and reported series; with ``smooth_paths``, the particle smoother's
    (``_smooth``) means and quantiles too, from that many paths, each drawn among ``candidates`` particles a period."""
    r = engine.r
    N, (T, p) = int(particles), y.shape
    n, k, ns = engine.n, engine.k, engine.ns
    if not 0 < resample <= 1:
        raise SpecificationError("resample is the share of the particles the effective sample size may fall to: above 0, up to 1")
    if proposal not in PROPOSALS:
        raise SpecificationError(f"proposal is one of {', '.join(map(repr, PROPOSALS))}, not {proposal!r}")
    errors = np.asarray(errors, float)
    if np.any(~(errors > 0)):
        raise ModelError("the particle filter needs a measurement error for every observed series (MeasurementMap(..., errors=...), "
                         "or estimate sd_<series>)")
    h = errors**2
    rng = make_rng(seed)
    burn = (50 if r.order == 2 else 0) if burn is None else int(burn)
    # the start: the first-order stationary distribution of the states, the second-order part at its ergodic mean
    sf = rng.standard_normal((N, ns)) @ _psd_factor(engine.V).T if ns else np.zeros((N, 0))
    ss = np.broadcast_to(engine.Es, (N, ns)).copy() if r.order == 2 else None
    lags = [np.zeros((N, n)) for _ in range(engine.L)]  # x(t-1), x(t-2), ...
    for _ in range(burn + engine.L):
        u = rng.standard_normal((N, k)) @ engine.chol.T
        xf, xs = engine.step(sf, ss, u)
        x = xf if xs is None else xf + xs
        sf, ss = xf[:, r.st], (None if xs is None else xs[:, r.st])
        lags = [x, *lags[:-1]] if engine.L else lags
    logW = np.full(N, -np.log(N))
    store: list[tuple[Any, ...]] | None = [] if smooth_paths else None
    if store is not None:  # the start, before the first period: equal weights
        store.append((sf.copy(), None if ss is None else ss.copy(), [X.copy() for X in lags], logW.copy(), None))
    total, path, ess, resampled = 0.0, np.zeros(T), np.zeros(T), np.zeros(T, bool)
    out_means, out_q = [], []
    names_r = [name for name, _ in engine.reports]
    endo = [engine.position[v] for v in r.endogenous]
    for t in range(T):
        e = rng.standard_normal((N, k))  # drawn every period, whatever happens: common random numbers
        v = float(rng.uniform())
        obs = np.isfinite(y[t])
        if proposal == "guided" and obs.any() and k:
            # the shocks' conditional normal given this period's data, with the measurement linearised (in e-space)
            B = engine.Z0[obs] @ r.ghu @ engine.chol  # observables x shocks
            Hinv = 1.0 / h[obs]
            S = np.linalg.inv(np.eye(k) + (B.T * Hinv) @ B)
            Sf = np.linalg.cholesky(0.5 * (S + S.T))
            xf0, xs0 = engine.step(sf, ss, np.zeros((N, k)))
            x0 = xf0 if xs0 is None else xf0 + xs0
            a = engine.observe([x0, *lags])[:, obs] + trends[t, obs]
            m = ((y[t, obs] - a) * Hinv) @ B @ S  # particles x shocks
            z = e
            e = m + z @ Sf.T
            # log prior of e minus log proposal of e (the normalising constants differ by log det Sf)
            correction = -0.5 * np.sum(e**2, axis=1) + 0.5 * np.sum(z**2, axis=1) + float(np.sum(np.log(np.diag(Sf))))
        else:
            correction = 0.0
        u = e @ engine.chol.T
        xf, xs = engine.step(sf, ss, u)
        x = xf if xs is None else xf + xs
        current = [x, *lags]
        yhat = engine.observe(current) + trends[t]
        if obs.any():
            resid = y[t, obs] - yhat[:, obs]
            logg = -0.5 * np.sum(resid**2 / h[obs], axis=1) - 0.5 * float(np.sum(np.log(h[obs]) + _LOG2PI)) + correction
            logw = logW + logg
            top = float(np.max(logw))
            if not np.isfinite(top):
                return {"loglike": -np.inf}
            inc = top + float(np.log(np.sum(np.exp(logw - top))))
            logW = logw - inc
        else:
            inc = 0.0
        path[t] = inc
        if t >= presample:
            total += inc
        W = np.exp(logW)
        W /= W.sum()
        ess[t] = 1.0 / float(np.sum(W**2))
        if keep is not None:
            reported = engine._evaluate([e_ for _, e_ in engine.reports], current) if engine.reports else np.zeros((N, 0))
            levels = np.column_stack([engine.sbar[endo] + x[:, endo], yhat, reported])
            out_means.append(W @ levels)
            out_q.append(_weighted_quantiles(levels, W, keep))
        sf, ss = xf[:, r.st], (None if xs is None else xs[:, r.st])
        if engine.L:
            lags = [x, *lags[:-1]]
        if store is not None:  # each particle's state, weight and the shocks that made it (before any resampling)
            store.append((sf.copy(), None if ss is None else ss.copy(), [X.copy() for X in lags], logW.copy(), u.copy()))
        if ess[t] < resample * N:
            idx = _systematic(W, v)
            sf = sf[idx]
            ss = None if ss is None else ss[idx]
            lags = [X[idx] for X in lags]
            logW = np.full(N, -np.log(N))
            resampled[t] = True
    result = {"loglike": total, "path": path, "ess": ess, "resampled": resampled}
    if keep is not None:
        result["names"] = ([r.variables[i] for i in endo] + [f"{o}_fitted" for o in engine.measurement.observables] + names_r)
        result["means"] = np.array(out_means)
        result["quantiles"] = np.array(out_q)  # periods x quantiles x columns
    if store is not None:
        result.update(_smooth(engine, y, h, trends, store, int(smooth_paths), candidates, rng, keep))
    return result


def _step_all(engine: _Engine, sf: np.ndarray, ss: np.ndarray | None, lags: list[np.ndarray], u: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[np.ndarray], list[np.ndarray]]:
    """One period from the states (sf, ss, lags) with shocks u: x(t), the next states, and [x(t), lags...] for the
    measurement equations."""
    r = engine.r
    xf, xs = engine.step(sf, ss, u)
    x = xf if xs is None else xf + xs
    current = [x, *lags]
    return x, xf[:, r.st], (None if xs is None else xs[:, r.st]), ([x, *lags[:-1]] if engine.L else lags), current


def _smooth(engine: _Engine, y: np.ndarray, h: np.ndarray, trends: np.ndarray, store: list[tuple[Any, ...]], M: int,
            candidates: int | None, rng: Any, keep: Sequence[float] | None) -> dict[str, Any]:
    """The particle smoother: backward simulation (Godsill, Doucet and West 2004) in the space of the shocks.

    The pruned state's transition is deterministic given the shocks (the second-order part, lagged copies for the
    measurement equations, and fewer shocks than states all make it degenerate), so a backward kernel through a
    transition density does not exist. Written in the shocks instead, the model's unknowns are the start ``s(-1)`` and
    the shocks ``u(0..T-1)``, independent a priori, and the smoothing distribution is
    ``p(s(-1), u | y) ~ p(s(-1)) prod N(u(t)) prod g(y(t) | x(t))`` with ``x(t)`` a function of the start and the
    shocks to ``t``. Backward simulation then draws, for a path whose shocks from ``t`` on are fixed, the particle at
    ``t - 1`` with probability proportional to its filter weight times the likelihood of the later data along the path
    re-simulated from that particle's state with those shocks: ``w(t-1, i) prod_{s >= t} g(y(s) | x(s; state_i,
    u(t..s)))`` (the backward simulation of non-Markovian models, Lindsten and Schön 2013, section 4); the drawn
    particle's own shock at ``t - 1`` joins the path. A path is the start drawn last and its shocks, re-simulated: a
    path the model can produce, its smoothed shocks with it. The re-simulation makes each path cost about
    ``candidates T^2 / 2`` particle steps; it stops early, exactly, once every candidate has reached the same state
    (to rounding), when the rest of the path weighs them alike. Each period's candidates are drawn from the filter's
    weighted particles (all of them when there are no more than ``candidates``), independently for each path."""
    T = len(y)
    N = store[0][0].shape[0]
    k = engine.k
    C = N if candidates is None or candidates >= N else int(candidates)
    shocks = np.zeros((M, T, k))
    logW = store[-1][3]
    W = np.exp(logW - np.max(logW))
    W /= W.sum()
    pick = _multinomial(W, M, rng)
    shocks[:, T - 1] = store[-1][4][pick]
    chosen_start = None
    obs_mask = np.isfinite(y)
    for t in range(T - 1, -1, -1):  # draw the particle at t - 1 given the path's shocks from t on (store index t)
        sf0, ss0, lags0, logw0, u0 = store[t]
        if C == N:
            rows = np.tile(np.arange(N), M)
            base = np.tile(logw0, M)
        else:
            w0 = np.exp(logw0 - np.max(logw0))
            w0 /= w0.sum()
            rows = _multinomial(w0, M * C, rng)  # each path's own candidates
            base = np.zeros(M * C)
        sf = sf0[rows]
        ss = None if ss0 is None else ss0[rows]
        lags = [X[rows] for X in lags0]
        acc = base.copy()
        for tau in range(t, T):
            u = np.repeat(shocks[:, tau], C, axis=0)
            _, sf, ss, lags, current = _step_all(engine, sf, ss, lags, u)
            obs = obs_mask[tau]
            if obs.any():
                yhat = engine.observe(current)[:, obs] + trends[tau, obs]
                acc -= 0.5 * np.sum((y[tau, obs] - yhat) ** 2 / h[obs], axis=1)
            if tau < T - 1 and _settled(sf, ss, lags, M, C):
                break
        acc = acc.reshape(M, C)
        acc -= acc.max(axis=1, keepdims=True)
        probs = np.exp(acc)
        probs /= probs.sum(axis=1, keepdims=True)
        draw = (probs.cumsum(axis=1) < rng.uniform(size=(M, 1))).sum(axis=1)
        drawn = rows.reshape(M, C)[np.arange(M), np.minimum(draw, C - 1)]
        if t > 0:
            shocks[:, t - 1] = u0[drawn] if u0 is not None else 0.0
        else:
            chosen_start = drawn
    # each path re-simulated from its start with its shocks
    sf0, ss0, lags0, _, _ = store[0]
    sf, ss, lags = sf0[chosen_start], (None if ss0 is None else ss0[chosen_start]), [X[chosen_start] for X in lags0]
    r = engine.r
    endo = [engine.position[v] for v in r.endogenous]
    levels = []
    for t in range(T):
        x, sf, ss, lags, current = _step_all(engine, sf, ss, lags, shocks[:, t])
        yhat = engine.observe(current) + trends[t]
        reported = engine._evaluate([e_ for _, e_ in engine.reports], current) if engine.reports else np.zeros((M, 0))
        levels.append(np.column_stack([engine.sbar[endo] + x[:, endo], yhat, reported]))
    paths = np.array(levels)  # periods x paths x columns
    out = {"smoothed_means": paths.mean(axis=1), "smoothed_paths": M, "smoothed_shocks": shocks.mean(axis=0),
           "smoothed_candidates": C}
    if keep:
        out["smoothed_quantiles"] = np.moveaxis(np.quantile(paths, list(keep), axis=1), 0, 1)  # periods x quantiles x columns
    return out


def _multinomial(W: np.ndarray, n: int, rng: Any) -> np.ndarray:
    c = np.cumsum(W)
    c[-1] = 1.0
    return np.minimum(np.searchsorted(c, rng.uniform(size=n)), len(W) - 1)


def _settled(sf: np.ndarray, ss: np.ndarray | None, lags: list[np.ndarray], M: int, C: int) -> bool:
    """Whether every path's candidates have reached the same state (to rounding): the rest of the path then weighs them
    alike, and re-simulating it changes nothing."""
    if C == 1:
        return True
    for X in (sf, ss, *lags):
        if X is None or X.size == 0:
            continue
        Y = X.reshape(M, C, -1)
        if np.max(np.abs(Y - Y[:, :1])) > 1e-13 * (1.0 + float(np.max(np.abs(Y)))):
            return False
    return True


def particle_filter(solution: Any, data: Any, measurement: Any, shock_sd: Mapping[str, float] | None = None,
                    span: Any = None, particles: int = 2000, seed: int | None = 0, resample: float = 0.5,
                    proposal: str = "bootstrap", burn: int | None = None, presample: int = 0,
                    quantiles: Sequence[float] = (0.05, 0.5, 0.95), shock_corr: Mapping[Any, float] | None = None,
                    measurement_sd: Mapping[str, float] | None = None, smooth_paths: int = 0,
                    candidates: int | None = 200) -> ParticleFilter:
    """The particle filter of ``solution`` (a ``SecondOrderSolution``, or a first-order ``LinearSolution`` with
    ``shock_sd``) on ``data`` through ``measurement`` (a ``MeasurementMap``, or model variables observed as they are,
    with ``measurement_sd``): see the module docstring. ``smooth_paths``: paths the particle smoother draws (backward
    simulation in the shocks, each period's particle among ``candidates`` drawn from the filter's, all with None), for
    ``smoothed``, ``smoothed_quantiles`` and ``smoothed_shocks``; 0 (the default) filters only. A ``ParticleFilter``."""
    from .._capture import fingerprint
    from .._version import __version__
    from ..data import Dataset
    from ..metadata import Provenance
    from .measurement import MeasurementMap

    if measurement is None:
        raise ModelError("give measurement equations (a MeasurementMap) or the model variables observed")
    if not isinstance(measurement, MeasurementMap):
        measurement = MeasurementMap.direct(list(measurement), measurement_sd)
    elif measurement_sd:
        measurement = measurement.with_errors(measurement_sd)
    if not measurement.observables:
        raise ModelError("the measurement map observes no series: the particle filter needs measurement equations for observed series")
    if int(particles) < 10:
        raise SpecificationError("particles is the number of particles: at least 10")
    if presample < 0:
        raise SpecificationError("presample is a number of periods, at least 0")
    if int(smooth_paths) < 0 or (candidates is not None and int(candidates) < 1):
        raise SpecificationError("smooth_paths is a number of paths, at least 0 (0: no smoother), and candidates at least 1")
    rules = rules_of(solution, shock_sd, shock_corr)
    engine = _Engine(rules, measurement)
    observed = measurement.observed(data, span)
    y = observed.to_numpy(float)
    trends = measurement.trend_values(observed.index) if measurement.trend_exprs else np.zeros(y.shape)
    errors = np.array([float(measurement.errors.get(o, 0.0)) for o in measurement.observables])
    qs = tuple(float(q) for q in quantiles)
    if any(not 0 <= q <= 1 for q in qs):
        raise SpecificationError("quantiles are between 0 and 1")
    out = run(engine, y, errors, trends, int(particles), seed, resample, proposal, burn, int(presample), qs, int(smooth_paths),
              None if candidates is None else int(candidates))
    if not np.isfinite(out["loglike"]):
        raise ModelError("every particle makes the data impossible (zero weight): check the measurement errors and the data")
    index = observed.index
    means = pd.DataFrame(out["means"], index=index, columns=out["names"])
    cols = pd.MultiIndex.from_product([list(qs), out["names"]], names=["quantile", "variable"])
    qframe = pd.DataFrame(out["quantiles"].reshape(len(index), -1), index=index, columns=cols)
    sources = [s for s in measurement.data_series() if s in data.frame.columns]
    resample_text = "every period" if resample >= 1 else f"when the effective sample size falls below {resample:g} of the particles"
    provenance = Provenance(
        data_vintages={s: data.variables[s].vintage for s in sources if s in data.variables},
        spec={"measurement": list(measurement.equations), "shocks": list(getattr(solution, "shocks", ()))},
        method={"type": "particle filter", "proposal": proposal, "particles": int(particles),
                "solution": "second-order perturbation (pruned)" if rules.order == 2 else "first order",
                "resampling": f"systematic, {resample_text}", "burn": (50 if rules.order == 2 else 0) if burn is None else int(burn),
                "presample": int(presample), "measurement": "exact" if not engine.linear else "linear",
                **({"smoother": f"backward simulation in the shocks, {int(smooth_paths)} paths, "
                                f"{out['smoothed_candidates']} candidate particles a period"} if smooth_paths else {})},
        sample=(str(index[0]), str(index[-1])), nobs=len(index), dropped=0, seed=seed, package_version=__version__,
        data_fingerprint=fingerprint(data.frame.loc[index, sources]) if sources else "",
        random_generator=get_random_generator() if seed is not None else None)
    smoothed = s_quantiles = s_shocks = None
    if smooth_paths:
        smoothed = Dataset.from_frame(pd.DataFrame(out["smoothed_means"], index=index, columns=out["names"]))
        s_shocks = pd.DataFrame(out["smoothed_shocks"], index=index, columns=list(getattr(solution, "shocks", ())) or None)
        if qs:
            s_quantiles = pd.DataFrame(out["smoothed_quantiles"].reshape(len(index), -1), index=index, columns=cols)
    return ParticleFilter(float(out["loglike"]), pd.Series(out["path"], index=index, name="loglike"), Dataset.from_frame(means),
                          qframe, pd.Series(out["ess"], index=index, name="ess"),
                          pd.Series(out["resampled"], index=index, name="resampled"), int(particles), rules.order, proposal,
                          tuple(measurement.observables), seed, provenance, smoothed, s_quantiles, s_shocks, int(smooth_paths))


def loglike_spread(solution: Any, data: Any, measurement: Any, seeds: Sequence[int] = range(10), **kw: Any) -> pd.Series:
    """The particle filter's log likelihood with each of ``seeds`` (other arguments as ``particle_filter``): its
    spread is the noise of the estimate; increase ``particles`` until the standard deviation is well below one."""
    return pd.Series({int(s): particle_filter(solution, data, measurement, seed=int(s), quantiles=(), **kw).loglike for s in seeds},
                     name="loglike")


@dataclass(frozen=True, eq=False)
class PseudoMarginal:
    """The log posterior with the particle filter's likelihood on fresh random numbers at every call (a chain's own
    stream): what particle marginal Metropolis-Hastings evaluates at each proposal, keeping the current draw's value."""

    lik: Any  # estimation._Likelihood
    stream: Any  # a SeedSequence
    _state: dict = field(default_factory=dict, repr=False)

    def forget(self) -> None:
        self.lik.forget()
        self._state.pop("rng", None)  # the chain's likelihood draws start again: the same chain here or in another process

    def __call__(self, x: Any) -> float:
        rng = self._state.get("rng")
        if rng is None:
            rng = self._state["rng"] = make_rng(self.stream)
        self.lik.__dict__["_pf_seed"] = int(rng.integers(2**63 - 1))
        try:
            return self.lik.log_posterior(x)
        finally:
            self.lik.__dict__.pop("_pf_seed", None)

    def __getstate__(self) -> dict[str, Any]:
        return {"lik": self.lik, "stream": self.stream, "_state": {}}

    def __setstate__(self, state: dict[str, Any]) -> None:
        for k, v in state.items():
            object.__setattr__(self, k, v)
