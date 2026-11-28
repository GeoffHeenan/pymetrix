"""The mixed (random-parameters) logit: McFadden's choice model with coefficients that vary across decision makers,
estimated by simulated maximum likelihood (McFadden and Train 2000; Train 2009).

    mixed_logit(trips, "chosen", ["cost", "time"], random={"time": "normal"}, alternative_constants=True)
    mixed_logit(panel, "chosen", ["price", "quality"], random={"price": "lognormal", "quality": "normal"},
                panel="person", correlated=True, draws=1000)

Decision maker n's utility from alternative j in choice situation t is U_ntj = x_ntj'beta_n + e_ntj with e
extreme value, so given beta_n the probabilities are the conditional logit's, L_nt(beta_n). A random coefficient is
normal, beta = b + s eta, or lognormal, beta = exp(b + s eta) (positive: enter the negative of a price), with eta
standard normal; uncorrelated by default, or correlated, the vector of b + L eta with L lower triangular (the
Cholesky factor of the covariance of the underlying normals). Coefficients without a distribution are fixed, as
are alternative constants and case characteristics. With ``panel`` (the decision maker), the choices of one
decision maker share one draw of beta_n, so the likelihood of a person is the integral of the product of their
choice probabilities; without it each case is a decision maker of its own.

The integral is simulated with R draws per decision maker, L_n = (1/R) sum_r prod_t L_nt(beta_n^r). The draws are
Halton sequences by default (deterministic: the radical inverses in the first primes, one prime per random
coefficient, the first 100 points dropped, R consecutive points per decision maker in the order of their ids, as
Train describes and xlogit does), turned normal by the inverse normal distribution; or pseudo-random normal draws
from pymetrix's random generator and ``seed``. The simulated log likelihood is maximised by BFGS with its analytic
gradient, then Newton steps on a central-difference Hessian; standard errors are from its inverse or clustered by
decision maker. The estimates depend on the draws: with too few draws they are biased (Train 2009, chapter 10), so
check them with more draws. The number of draws, their kind and the points dropped are recorded in
``provenance.resolved``.

Each decision maker's conditional mean of a random coefficient, E[beta_n | their choices] =
sum_r beta^r prod_t L_nt(beta^r) / sum_r prod_t L_nt(beta^r) (Revelt and Train 2000; Train 2009, chapter 11), is in
``individual``.

Written natively: xlogit (MIT, the one light library) matches this code on the same draws (tests/test_mixed_logit.py)
but has no correlated coefficients, seeds NumPy's global generator, assumes every case has the same alternatives and
has had no release since 2023; Biogeme needs Python 3.12, JAX and PyMC and is not under an open-source licence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import optimize

from . import _choice, _micro
from ._api import api
from .errors import PyMetrixError, SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["FittedMixedLogit", "halton_draws", "mixed_logit"]

_DISTRIBUTIONS = {"normal": "n", "n": "n", "lognormal": "ln", "ln": "ln"}
_HALTON_DROP = 100
_CHUNK = 4_000_000  # cells (cases x alternatives x draws) simulated at once


def halton_draws(people: int, draws: int, dimensions: int, drop: int = _HALTON_DROP) -> np.ndarray:
    """Standard normal draws from Halton sequences (people, draws, dimensions): dimension d uses the radical inverse
    in the d-th prime from 0, the first ``drop`` points dropped, ``draws`` consecutive points per person."""
    from scipy.stats import norm, qmc

    if dimensions == 0:
        return np.zeros((people, draws, 0))
    u = qmc.Halton(d=dimensions, scramble=False).random(people * draws + drop)[drop:]
    return norm.ppf(u).reshape(people, draws, dimensions)


@dataclass(frozen=True, eq=False)
class _Sim:
    """The simulation: cases sorted by decision maker, ``starts`` the first case of each, the draws (people, R, Kr)."""

    X: np.ndarray
    available: np.ndarray
    chosen: np.ndarray
    person: np.ndarray       # each case's decision maker (0 .. people - 1), nondecreasing
    starts: np.ndarray       # each decision maker's first case
    rand: np.ndarray         # the design's columns with random coefficients
    lognormal: np.ndarray    # (Kr,) bool
    correlated: bool
    eta: np.ndarray          # (people, R, Kr)

    @property
    def p(self) -> int:
        return self.X.shape[2]

    @property
    def fixed(self) -> np.ndarray:
        return np.setdiff1d(np.arange(self.p), self.rand)

    def chol(self, theta: np.ndarray) -> np.ndarray:
        Kr = len(self.rand)
        extra = theta[self.p:]
        if not self.correlated:
            return np.diag(extra)
        L = np.zeros((Kr, Kr))
        L[np.tril_indices(Kr)] = extra
        return L

    def coefficients(self, theta: np.ndarray, people: slice = slice(None)) -> tuple[np.ndarray, np.ndarray]:
        """The random coefficients' draws (people, R, Kr) and their derivatives in the underlying normal."""
        u = theta[self.rand][None, None, :] + self.eta[people] @ self.chol(theta).T
        beta = np.where(self.lognormal, np.exp(np.minimum(u, 700)), u)
        return beta, np.where(self.lognormal, beta, 1.0)

    def blocks(self) -> list[tuple[int, int]]:
        """Decision makers in blocks of about ``_CHUNK`` cells."""
        A, R = self.X.shape[1], self.eta.shape[1]
        per = max(1, _CHUNK // max(1, A * R))
        out, first = [], 0
        P = len(self.starts)
        while first < P:
            last = int(np.searchsorted(self.starts, self.starts[first] + per, side="left"))
            last = max(last, first + 1)
            out.append((first, min(last, P)))
            first = last
        return out

    def cases(self, first: int, last: int) -> slice:
        end = self.starts[last] if last < len(self.starts) else len(self.chosen)
        return slice(int(self.starts[first]), int(end))

    def probabilities(self, theta: np.ndarray, first: int, last: int) -> np.ndarray:
        """Each case's probabilities given each draw (cases, A, R)."""
        cs = self.cases(first, last)
        beta, _ = self.coefficients(theta, slice(first, last))
        X = self.X[cs]
        fixed = self.fixed
        V = (X[:, :, fixed] @ theta[fixed])[:, :, None] + np.matmul(X[:, :, self.rand], beta[self.person[cs] - first].transpose(0, 2, 1))
        avail = self.available[cs][:, :, None]
        V = np.where(avail, V, -np.inf)
        V = V - V.max(axis=1, keepdims=True)
        e = np.where(avail, np.exp(V), 0.0)
        return e / e.sum(axis=1, keepdims=True)


def simulated_loglike(theta: np.ndarray, sim: _Sim, gradient: bool = True) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """The simulated log likelihood, its gradient, each decision maker's score (rows) and the weights of their draws
    given their choices (people, R)."""
    P_all = len(sim.starts)
    R, Kr, p = sim.eta.shape[1], len(sim.rand), sim.p
    k = len(theta)
    scores = np.zeros((P_all, k))
    weights = np.zeros((P_all, R))
    ll = 0.0
    fixed = sim.fixed
    tri = np.tril_indices(Kr)
    for first, last in sim.blocks():
        cs = sim.cases(first, last)
        P = sim.probabilities(theta, first, last)
        n = P.shape[0]
        local = sim.person[cs] - first
        lstarts = sim.starts[first:last] - sim.starts[first]
        with np.errstate(divide="ignore"):
            logp = np.log(P[np.arange(n), sim.chosen[cs], :])                       # (cases, R)
        logS = np.add.reduceat(logp, lstarts, axis=0)                               # (people, R)
        top = logS.max(axis=1, keepdims=True)
        lik = np.exp(logS - top)
        ll += float((top[:, 0] + np.log(lik.mean(axis=1))).sum())
        W = lik / lik.sum(axis=1, keepdims=True)
        weights[first:last] = W
        if not gradient:
            continue
        G = -P
        G[np.arange(n), sim.chosen[cs], :] += 1.0
        X = sim.X[cs]
        Wc = W[local]
        s = np.zeros((last - first, k))
        if len(fixed):
            GW = np.matmul(G, Wc[:, :, None])                                       # (cases, A, 1)
            s[:, fixed] = np.add.reduceat((GW * X[:, :, fixed]).sum(axis=1), lstarts, axis=0)
        if Kr:
            Gr = np.add.reduceat(np.matmul(G.transpose(0, 2, 1), X[:, :, sim.rand]), lstarts, axis=0) * W[:, :, None]   # (people, R, Kr)
            _, dbeta = sim.coefficients(theta, slice(first, last))
            T = Gr * dbeta
            s[:, sim.rand] = T.sum(axis=1)
            eta = sim.eta[first:last]
            if sim.correlated:
                s[:, p:] = np.einsum("prm,prk->pmk", T, eta)[:, tri[0], tri[1]]
            else:
                s[:, p:] = (T * eta).sum(axis=1)
        scores[first:last] = s
    return ll, scores.sum(axis=0), scores, weights


# -- the result -------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FittedMixedLogit:
    """A mixed logit by simulated maximum likelihood: the fixed coefficients and the random coefficients' means
    (of the underlying normal for a lognormal one), then their standard deviations (``sd.x``) or, correlated, the
    Cholesky factor of their covariance (``chol.x.y``)."""

    outcome: str
    names: tuple[str, ...]
    values: np.ndarray
    cov: np.ndarray = field(repr=False)
    random: dict[str, str] = field(default_factory=dict)  # regressor: "normal" or "lognormal"
    correlated: bool = False
    draws: int = 0
    draw_type: str = "halton"
    loglike: float = 0.0
    loglike_logit: float = 0.0  # the conditional logit (every coefficient fixed)
    loglike_null: float = 0.0
    cases: int = 0
    people: int = 0
    dropped_cases: int = 0
    nobs: int = 0
    cov_type: str = "hessian"
    case: str = ""
    alternative: str = ""
    panel: str | None = None
    probabilities: pd.Series | None = field(default=None, repr=False)
    individual: pd.DataFrame | None = field(default=None, repr=False)  # each decision maker's conditional mean of each random coefficient
    iterations: int = 0
    converged: bool = True
    provenance: Any = field(default=None, repr=False)
    design: Any = field(default=None, repr=False)

    stage = "estimate"

    @property
    def params(self) -> Any:
        from .estimation.params import PointEstimate

        return PointEstimate(self.names, np.asarray(self.values, float), np.asarray(self.cov, float))

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    @property
    def stats(self) -> dict[str, float]:
        k = len(self.values)
        return {"nobs": float(self.nobs), "cases": float(self.cases), "people": float(self.people), "draws": float(self.draws),
                "loglike": self.loglike, "loglike_logit": self.loglike_logit, "loglike_null": self.loglike_null,
                "pseudo_r2": 1 - self.loglike / self.loglike_null if self.loglike_null else float("nan"),
                "aic": -2 * self.loglike + 2 * k, "bic": -2 * self.loglike + k * np.log(self.people)}

    def _moments(self, theta: np.ndarray) -> np.ndarray:
        """Each random coefficient's mean, median and standard deviation (of the coefficient itself), then the
        correlations of the underlying normals (below the diagonal)."""
        sim: _Sim = self.design["sim"]
        L = sim.chol(theta)
        S = L @ L.T
        b = theta[sim.rand]
        sd = np.sqrt(np.diag(S))
        ln = sim.lognormal
        mean = np.where(ln, np.exp(b + sd**2 / 2), b)
        median = np.where(ln, np.exp(b), b)
        spread = np.where(ln, mean * np.sqrt(np.maximum(np.expm1(sd**2), 0.0)), sd)
        out = [mean, median, spread]
        if self.correlated and len(b) > 1:
            with np.errstate(invalid="ignore", divide="ignore"):
                R = S / np.outer(sd, sd)
            i, j = np.tril_indices(len(b), -1)
            out.append(R[i, j])
        return np.concatenate(out)

    def random_coefficients(self) -> Table:
        """Each random coefficient's distribution: mean, median and standard deviation of the coefficient itself (for a
        lognormal one, from the underlying normal's), and, correlated, the correlations of the underlying normals;
        delta-method standard errors."""
        from .estimation.numdiff import jacobian

        sim: _Sim = self.design["sim"]
        theta, V = np.asarray(self.values, float), np.asarray(self.cov, float)
        values = self._moments(theta)
        J = jacobian(self._moments, theta)
        se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", J, V, J), 0.0))
        rn = [self.design["names"][c] for c in sim.rand]
        index = [f"{w}: {r}" for w in ("mean", "median", "s.d.") for r in rn]
        if self.correlated and len(rn) > 1:
            i, j = np.tril_indices(len(rn), -1)
            index += [f"correlation: {rn[a]}, {rn[b]}" for a, b in zip(i, j)]
        notes = ["The distribution of each random coefficient across decision makers"
                 + (" (for a lognormal one, the mean exp(b + s^2/2), median exp(b) and s.d. mean * sqrt(exp(s^2) - 1) of the coefficient)"
                    if sim.lognormal.any() else "") + "; delta-method standard errors."]
        if self.correlated:
            notes.append("Correlations are between the underlying normals.")
        return _micro.coefficient_table(index, values, se, f"Random coefficients: {self.outcome}", notes, index_name="moment")

    def marginal_effects(self, at: Literal["average"] = "average", alpha: float = 0.05) -> Any:
        """Derivatives and elasticities of the choice probabilities, averaged over the cases, each integrated over the
        random coefficients with the estimation's draws: dP_j/dx_k = E_beta[beta L_j (delta_jk - L_k)] and the
        elasticity x_k / P_j times that (Train 2009, chapter 6); with delta-method standard errors
        (``ConditionalLogitEffects``)."""
        from .clogit_effects import _cases, _split

        if at != "average":
            raise SpecificationError("the mixed logit's effects are averaged over the cases (at='average'): at the means there is no one draw of the coefficients")
        d = self.design
        if not d["regressors"] and not d["case_variables"]:
            raise SpecificationError("the model has only alternative constants: there are no variables to take effects of")
        sim: _Sim = d["sim"]
        c = _cases(d)
        order = d["order"]  # the simulation's cases, in the effects' case order
        r_names = list(d["regressors"])
        rand_of = {r_names.index(d["names"][col]): i for i, col in enumerate(sim.rand)}
        n, A = c.available.shape
        eye = np.eye(A)
        both = c.available[:, :, None] & c.available[:, None, :]
        x = np.nan_to_num(c.x)

        def effects(theta: np.ndarray) -> dict[str, np.ndarray]:
            beta_fixed, _, gamma = _split(theta[:sim.p], d)
            D = {name: np.zeros((A, A)) for name in r_names}
            E = {name: np.zeros((A, A)) for name in r_names}
            Dz = {name: np.zeros(A) for name in d["case_variables"]}
            Ez = {name: np.zeros(A) for name in d["case_variables"]}
            for first, last in sim.blocks():
                cs = sim.cases(first, last)
                Pr = sim.probabilities(theta, first, last)                         # (cases, A, R)
                beta, _ = sim.coefficients(theta, slice(first, last))
                local = sim.person[cs] - first
                idx = order[cs]
                Pbar = Pr.mean(axis=2)
                for m, name in enumerate(r_names):
                    b = beta[local][:, :, rand_of[m]] if m in rand_of else np.full((Pr.shape[0], Pr.shape[2]), beta_fixed[m])
                    # E_beta[beta L_j (delta_jk - L_k)], (cases, j, k)
                    dd = np.einsum("cr,cjr,jk->cjk", b, Pr, eye) - np.einsum("cr,cjr,ckr->cjk", b, Pr, Pr)
                    dd /= Pr.shape[2]
                    D[name] += dd.sum(axis=0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        e = dd * x[m][idx][:, None, :] / Pbar[:, :, None]
                    E[name] += np.where(both[idx], e, 0.0).sum(axis=0)
                for v, name in enumerate(d["case_variables"]):
                    g = gamma[v]
                    dz = (Pr * (g[None, :, None] - np.einsum("car,a->cr", Pr, g)[:, None, :])).mean(axis=2)
                    Dz[name] += dz.sum(axis=0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        e = c.z[idx][:, [v]] * dz / Pbar
                    Ez[name] += np.where(c.available[idx], e, 0.0).sum(axis=0)
            out = {}
            for name in r_names:
                out[f"d:{name}"] = D[name] / n
                with np.errstate(invalid="ignore"):
                    out[f"e:{name}"] = E[name] / both.sum(axis=0)
            for name in d["case_variables"]:
                out[f"d:{name}"] = Dz[name] / n
                with np.errstate(invalid="ignore"):
                    out[f"e:{name}"] = Ez[name] / c.available.sum(axis=0)
            return out

        return _choice.effects_result(effects, np.asarray(self.values, float), np.asarray(self.cov, float), d, at, alpha, n,
                                      self.outcome, self.provenance, "mixed logit")

    def summary(self) -> Table:
        st = self.stats
        dist = ", ".join(f"{k} {v}" for k, v in self.random.items())
        notes = [f"Mixed logit (simulated maximum likelihood) of the choice {self.outcome}; {self.cases} cases ({self.case}), "
                 f"{self.nobs} alternatives ({self.alternative})"
                 + (f", {self.people} decision makers ({self.panel}), each with one draw of the coefficients for all their choices." if self.panel else "."),
                 f"Random coefficients: {dist}" + ("; correlated (chol.x.y: the Cholesky factor of the covariance of the underlying normals)."
                                                    if self.correlated else "; uncorrelated (sd.x: their standard deviations)."),
                 "For a lognormal coefficient, x and sd.x are the mean and standard deviation of its logarithm." if "lognormal" in self.random.values() else "",
                 f"{self.draws} {'Halton' if self.draw_type == 'halton' else 'pseudo-random'} draws per decision maker.",
                 f"Log likelihood: {self.loglike:.4f}", f"Log likelihood (conditional logit): {self.loglike_logit:.4f}",
                 f"Pseudo R-squared against equal shares within each choice set: {st['pseudo_r2']:.4f}.",
                 "Standard errors from the inverse Hessian." if self.cov_type == "hessian" else
                 f"Standard errors clustered by {self.panel or self.case}."]
        if self.dropped_cases:
            notes.append(f"{self.dropped_cases} cases without exactly one chosen alternative dropped.")
        if not self.converged:
            notes.append(f"The maximisation did not converge in {self.iterations} steps.")
        return _micro.coefficient_table(self.names, self.values, self.se, f"Mixed logit: {self.outcome}", [n for n in notes if n])

    def plot(self) -> Any:
        from .figures import Estimates, Figure, Panel

        b, s = np.asarray(self.values), self.se
        idx = list(self.names)
        return Figure((Panel((Estimates(pd.Series(b, index=idx), pd.Series(b - 1.96 * s, index=idx), pd.Series(b + 1.96 * s, index=idx), "estimate"),),
                             title=f"Mixed logit: {self.outcome}"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.summary().to_frame()

    def __repr__(self) -> str:
        return f"<FittedMixedLogit {self.outcome}: {dict(zip(self.names, np.round(self.values, 4)))}>"


# -- estimation -------------------------------------------------------------------------------------------------------

def _normalise(theta: np.ndarray, V: np.ndarray, sim: _Sim) -> tuple[np.ndarray, np.ndarray]:
    """Standard deviations, or the Cholesky factor's diagonal, made positive (the likelihood does not change)."""
    Kr, p = len(sim.rand), sim.p
    sign = np.ones(len(theta))
    if sim.correlated:
        L = sim.chol(theta)
        flip = np.where(np.diag(L) < 0, -1.0, 1.0)  # a column of L times -1 leaves L L' as it is
        i, j = np.tril_indices(Kr)
        sign[p:] = flip[j]
    else:
        sign[p:] = np.where(theta[p:] < 0, -1.0, 1.0)
    return theta * sign, V * np.outer(sign, sign)


@api("pymetrix:mixed_logit")
def mixed_logit(data: Dataset, choice: str, regressors: Sequence[str] = (), random: Mapping[str, str] | None = None,
                case: str | None = None, alternative: str | None = None, panel: str | None = None, alternative_constants: bool = False,
                case_variables: Sequence[str] = (), base: Any = None, correlated: bool = False, draws: int = 500,
                draw_type: Literal["halton", "random"] = "halton", seed: int = 0,
                cov: Literal["hessian", "cluster"] = "hessian") -> FittedMixedLogit:
    """A mixed (random-parameters) logit by simulated maximum likelihood, for choice data in long form as
    ``choice_logit`` takes it. ``random`` maps regressors to "normal" or "lognormal" (others are fixed);
    ``correlated`` estimates the Cholesky factor of their covariance; ``panel`` names the decision maker whose
    choices share one draw. ``draws`` per decision maker, Halton (deterministic) or pseudo-random (``seed``).
    ``cov``: the inverse Hessian, or clustered by decision maker (by case without a panel)."""
    from ._capture import capture

    regressors = _micro.names(regressors)
    random = dict(random or {})
    if not random:
        raise SpecificationError("give random= ({regressor: 'normal' or 'lognormal'}): with no random coefficient it is the conditional logit")
    bad = [k for k in random if k not in regressors]
    if bad:
        raise SpecificationError(f"random coefficients are for regressors (attributes of the alternatives); not among them: {', '.join(bad)}")
    wrong = {k: v for k, v in random.items() if v not in _DISTRIBUTIONS}
    if wrong:
        raise SpecificationError(f"a random coefficient is 'normal' or 'lognormal', not {wrong}")
    random = {k: "lognormal" if _DISTRIBUTIONS[v] == "ln" else "normal" for k, v in random.items()}
    if int(draws) < 1:
        raise SpecificationError("draws is a positive whole number")
    if draw_type not in ("halton", "random"):
        raise SpecificationError(f"draw_type is 'halton' or 'random', not {draw_type!r}")
    if cov not in ("hessian", "cluster"):
        raise SpecificationError(f"cov is 'hessian' or 'cluster', not {cov!r}")
    draws = int(draws)
    with capture() as caught:
        choices = _choice.read_choices(data, choice, regressors, case, alternative, alternative_constants, case_variables, base,
                                       "mixed logit", extra=[panel] if panel else [])
        n = len(choices.chosen)
        if panel:
            pid = choices.per_case(panel)
            check = pd.Series(choices.extra[panel][choices.used]).groupby(choices.design["case"]).nunique()
            if (check > 1).any():
                raise SpecificationError(f"{panel} must be the same for every alternative of a case")
            codes, people_ids = pd.factorize(pd.Series(pid), sort=True)
        else:
            codes, people_ids = np.arange(n), pd.Index(choices.case_ids)
        order = np.argsort(codes, kind="stable")       # cases sorted by decision maker
        codes_sorted = codes[order]
        starts = np.flatnonzero(np.r_[True, codes_sorted[1:] != codes_sorted[:-1]])
        people = len(starts)
        rand_cols = np.array([choices.names.index(k) for k in regressors if k in random])
        rand_names = [k for k in regressors if k in random]
        lognormal = np.array([random[k] == "lognormal" for k in rand_names])
        Kr = len(rand_cols)
        if draw_type == "halton":
            eta = halton_draws(people, draws, Kr)
        else:
            from ._random import generator as make_rng

            eta = make_rng(seed).standard_normal((people, draws, Kr))
        sim = _Sim(choices.X[order], choices.available[order], choices.chosen[order], codes_sorted, starts, rand_cols, lognormal, correlated, eta)
        beta0, ll_logit = _choice.fit_logit(choices.X, choices.available, choices.chosen)
        start = beta0.copy()
        for c, ln in zip(rand_cols, lognormal):
            if ln:
                start[c] = np.log(max(abs(beta0[c]), 1e-3))
        if correlated:
            L0 = np.zeros((Kr, Kr))
            L0[np.diag_indices(Kr)] = 0.1
            extra0 = L0[np.tril_indices(Kr)]
        else:
            extra0 = np.full(Kr, 0.1)
        theta0 = np.concatenate([start, extra0])

        def loglike(t: np.ndarray) -> tuple[float, np.ndarray]:
            ll, g, _, _ = simulated_loglike(t, sim)
            return ll, g

        def negative(t: np.ndarray) -> tuple[float, np.ndarray]:
            ll, g = loglike(t)
            return (-ll, -g) if np.isfinite(ll) else (np.inf, np.zeros_like(t))

        res = optimize.minimize(negative, theta0, jac=True, method="BFGS", options={"gtol": 1e-6, "maxiter": 2000})
        theta, H, ok = _choice.newton_polish(res.x, loglike, tol=1e-6, max_iter=5)
        ll, grad, scores, W = simulated_loglike(theta, sim)
        if not np.all(np.isfinite(theta)) or not np.isfinite(ll):
            raise PyMetrixError("the mixed logit did not converge")
        V = _choice.covariance(H, scores, cov)
        theta, V = _normalise(theta, V, sim)
        ll, grad, scores, W = simulated_loglike(theta, sim)
    names = list(choices.names)
    if correlated:
        i, j = np.tril_indices(Kr)
        names += [f"chol.{rand_names[a]}.{rand_names[b]}" for a, b in zip(i, j)]
    else:
        names += [f"sd.{r}" for r in rand_names]
    beta_draws, _ = sim.coefficients(theta)
    individual = pd.DataFrame(np.einsum("pr,prk->pk", W, beta_draws), index=pd.Index(people_ids, name=panel or choices.case), columns=rand_names)
    probs = np.zeros(choices.available.shape)
    for first, last in sim.blocks():
        probs[order[sim.cases(first, last)]] = sim.probabilities(theta, first, last).mean(axis=2)
    used = choices.frame.iloc[np.sort(choices.used)]
    spec = {"choice": choice, "regressors": regressors, "random": random, "case": case, "alternative": alternative, "panel": panel,
            "alternative_constants": alternative_constants, "case_variables": _micro.names(case_variables), "base": base,
            "correlated": correlated, "draws": draws, "draw_type": draw_type, "seed": seed, "cov": cov}
    resolved = {"draws": draws, "draw_type": draw_type, "case": choices.case, "alternative": choices.alternative, "base": choices.design["base"],
                "decision_makers": people}
    if draw_type == "halton":
        resolved.update({"halton_primes": [int(q) for q in _primes(Kr)], "halton_drop": _HALTON_DROP})
    prov = _micro.provenance(spec, data, used, [choice, *regressors, *_micro.names(case_variables)], caught,
                             {"type": "mixed logit", "estimation": "simulated maximum likelihood"}, converged=ok,
                             seed=seed if draw_type == "random" else None, resolved=resolved)
    # the simulation's case s is case order[s] of choices (and of the effects, whose cases are in the same order)
    design = {**choices.design, "sim": sim, "names": list(choices.names), "order": order}
    null = -float(np.log(choices.available.sum(axis=1)).sum())
    return FittedMixedLogit(choice, tuple(names), theta, V, random, bool(correlated), draws, draw_type, ll, ll_logit, null, n, people,
                            choices.dropped, int(choices.available.sum()), cov, choices.case, choices.alternative, panel,
                            _choice.probabilities_series(probs, choices), individual, int(res.nit), ok, prov, design)


def _primes(k: int) -> list[int]:
    out: list[int] = []
    n = 2
    while len(out) < k:
        if all(n % q for q in out):
            out.append(n)
        n += 1
    return out
