"""The nested logit: McFadden's choice model with alternatives grouped in nests whose unobserved utilities are
correlated, so that the independence of irrelevant alternatives holds within a nest but not across nests.

    nested_logit(trips, "chosen", ["cost", "time"], nests="group", alternative_constants=True)
    nested_logit(trips, "chosen", ["cost"], nests={"public": ["bus", "train"], "private": ["car"]}, dissimilarity="equal")

Written in the form consistent with random utility maximisation (McFadden 1978; the "RU2" normalisation of Hensher
and Greene 2002): utilities enter divided by their nest's dissimilarity (log-sum) parameter lambda_m, so with nests
B_m and V_ij = x_ij'beta (+ alternative constants and case characteristics, as the conditional logit's),

    IV_im = log sum_{k in B_m} exp(V_ik / lambda_m)                                    (the inclusive value)
    P_ij  = exp(V_ij / lambda_m) / exp(IV_im)  x  exp(lambda_m IV_im) / sum_l exp(lambda_l IV_il),   j in B_m.

With every lambda_m = 1 this is the conditional logit exactly. The model is consistent with utility maximisation
for every x when 0 < lambda_m <= 1 (and locally for values a little above 1); lambda_m near 0 means the alternatives
in nest m are near-perfect substitutes. A nest with a single alternative has no lambda in this form (it cancels), so
it is fixed at 1. Estimation is full-information maximum likelihood (FIML), the coefficients and the lambdas
together, with analytic gradients, quasi-Newton steps and Newton steps at the end on a central-difference Hessian;
standard errors are from the inverse Hessian, or clustered by case (a sandwich). The likelihood-ratio test of every
lambda = 1 (the conditional logit, fitted too) is a test of IIA against this nesting.

Written natively: the likelihood is in closed form, statsmodels has no nested logit, and Biogeme (which has one, and
agrees: tests/test_nested_logit.py) needs Python 3.12, JAX and PyMC, takes wide data through its own expression
language, and is not under an open-source licence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import optimize

from pymetrix._lazy import stats as sps

from . import _choice, _micro
from ._api import api
from .errors import PyMetrixError, SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["FittedNestedLogit", "nested_logit", "nested_loglike"]


# -- the likelihood ---------------------------------------------------------------------------------------------------

def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = a.max(axis=axis, keepdims=True)
    safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.exp(a - safe).sum(axis=axis, keepdims=True)) + safe
    return np.squeeze(out, axis=axis)


def _parts(beta: np.ndarray, lam: np.ndarray, X: np.ndarray, available: np.ndarray, nest_of: np.ndarray) -> dict[str, np.ndarray]:
    """Utilities, scaled utilities, inclusive values and probabilities (joint, within the nest, of the nest)."""
    V = X @ beta
    la = lam[nest_of]
    s = np.where(available, V / la, -np.inf)
    M = len(lam)
    IV = np.column_stack([_logsumexp(np.where(nest_of == m, s, -np.inf), axis=1) for m in range(M)])
    top = lam * IV
    D = _logsumexp(top, axis=1)
    with np.errstate(invalid="ignore"):
        logP = np.where(available, s - IV[:, nest_of] + top[:, nest_of] - D[:, None], -np.inf)
        within = np.where(available, np.exp(s - IV[:, nest_of]), 0.0)
        Q = np.exp(top - D[:, None])
    return {"V": V, "s": s, "IV": IV, "logP": logP, "P": np.exp(logP), "within": within, "Q": np.nan_to_num(Q), "la": la}


def nested_loglike(theta: np.ndarray, X: np.ndarray, available: np.ndarray, chosen: np.ndarray, nest_of: np.ndarray,
                   lam_param: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """The nested logit's log likelihood at ``theta`` (the coefficients, then the free lambdas), with its gradient and
    each case's score. ``nest_of`` gives each alternative's nest; ``lam_param`` each nest's lambda's position among
    the free lambdas (-1: fixed at 1)."""
    p = X.shape[2]
    lam = np.ones(len(lam_param))
    free = lam_param >= 0
    lam[free] = theta[p + lam_param[free]]
    if np.any(lam <= 0):
        return -np.inf, np.zeros(len(theta)), np.zeros((len(chosen), len(theta)))
    n = len(chosen)
    r = np.arange(n)
    f = _parts(theta[:p], lam, X, available, nest_of)
    ll_i = f["logP"][r, chosen]
    m_j = nest_of[chosen]
    lam_j = lam[m_j]
    same = nest_of[None, :] == m_j[:, None]
    gV = np.where(same, (lam_j - 1)[:, None] / lam_j[:, None] * f["within"], 0.0) - f["P"]
    gV[r, chosen] += 1 / lam_j
    scores = [np.einsum("na,nap->np", gV, X)]
    q = int(lam_param.max()) + 1 if free.any() else 0
    if q:
        s_fin = np.where(available, f["s"], 0.0)
        G = np.zeros((n, q))
        for m in np.flatnonzero(free):
            in_m = nest_of == m
            sbar = (f["within"][:, in_m] * s_fin[:, in_m]).sum(axis=1)
            IV = np.where(np.isfinite(f["IV"][:, m]), f["IV"][:, m], 0.0)
            g = -f["Q"][:, m] * (IV - sbar)
            mine = m_j == m
            g[mine] += (-s_fin[r, chosen] / lam[m] + IV - (lam[m] - 1) * sbar / lam[m])[mine]
            G[:, lam_param[m]] += g
        scores.append(G)
    S = np.column_stack(scores)
    return float(ll_i.sum()), S.sum(axis=0), S


# -- the result -------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FittedNestedLogit:
    """A nested logit by FIML in the form consistent with random utility maximisation (RU2): coefficients, then
    each nest's dissimilarity parameter lambda (one shared, with ``dissimilarity="equal"``)."""

    outcome: str
    names: tuple[str, ...]
    values: np.ndarray
    cov: np.ndarray = field(repr=False)
    nests: dict[Any, str] = field(default_factory=dict)  # alternative: nest
    dissimilarity: str = "free"
    loglike: float = 0.0
    loglike_logit: float = 0.0  # every lambda 1: the conditional logit
    loglike_null: float = 0.0   # all coefficients zero, every lambda 1: equal probabilities within each choice set
    iia_test: Any = None        # the likelihood-ratio test of every lambda = 1 (TestResult), or None
    cases: int = 0
    dropped_cases: int = 0
    nobs: int = 0
    cov_type: str = "hessian"
    case: str = ""
    alternative: str = ""
    probabilities: pd.Series | None = field(default=None, repr=False)
    fixed_nests: tuple[str, ...] = ()  # nests with one alternative: lambda fixed at 1
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
    def lambdas(self) -> pd.Series:
        """Each nest's dissimilarity parameter (1 for a nest of one alternative)."""
        d = self.design
        lam = np.ones(len(d["nest_names"]))
        free = d["lam_param"] >= 0
        lam[free] = np.asarray(self.values)[d["p"] + d["lam_param"][free]]
        return pd.Series(lam, index=pd.Index(d["nest_names"], name="nest"), name="lambda")

    @property
    def stats(self) -> dict[str, float]:
        k = len(self.values)
        return {"nobs": float(self.nobs), "cases": float(self.cases), "loglike": self.loglike, "loglike_logit": self.loglike_logit,
                "loglike_null": self.loglike_null, "pseudo_r2": 1 - self.loglike / self.loglike_null if self.loglike_null else float("nan"),
                "aic": -2 * self.loglike + 2 * k, "bic": -2 * self.loglike + k * np.log(self.cases)}

    def nest_table(self) -> Table:
        """The nests, their alternatives and their lambdas, with a Wald test of lambda = 1 for each free one."""
        d = self.design
        rows, index = [], []
        se = self.se
        for m, nest in enumerate(d["nest_names"]):
            members = ", ".join(str(a) for a, n in self.nests.items() if n == nest)
            pos = d["lam_param"][m]
            if pos < 0:
                rows.append([members, 1.0, np.nan, np.nan, np.nan])
            else:
                lam, s = float(self.values[d["p"] + pos]), float(se[d["p"] + pos])
                z = (lam - 1) / s
                rows.append([members, lam, s, z, float(2 * sps.norm.sf(abs(z)))])
            index.append(str(nest))
        notes = ["lambda is the nest's dissimilarity (log-sum) parameter, in the form consistent with utility maximisation (RU2); "
                 "z and p test lambda = 1."]
        if self.fixed_nests:
            notes.append(f"Nests of one alternative ({', '.join(self.fixed_nests)}) have no lambda in this form: fixed at 1.")
        return Table(["alternatives", "lambda", "std err", "z (lambda = 1)", "p-value"], rows, index=index, index_name="nest",
                     title=f"Nests: {self.outcome}", notes=tuple(notes))

    def marginal_effects(self, at: Literal["average", "mean"] = "average", alpha: float = 0.05) -> Any:
        """Derivatives and elasticities of the choice probabilities in each attribute of each alternative and each
        characteristic of the case, averaged over the cases or at the means, with delta-method standard errors
        (``ConditionalLogitEffects``, as after the conditional logit)."""
        from .clogit_effects import _at_means, _cases, _split

        if at not in ("average", "mean"):
            raise SpecificationError(f"at is 'average' (over the cases) or 'mean' (at the means), not {at!r}")
        d = self.design
        if not d["regressors"] and not d["case_variables"]:
            raise SpecificationError("the model has only alternative constants: there are no variables to take effects of")
        c = _cases(d)
        base = c if at == "average" else _at_means(c)
        p, nest_of, lam_param = d["p"], d["nest_of"], d["lam_param"]
        A = c.available.shape[1]
        eye = np.eye(A)
        same = (nest_of[:, None] == nest_of[None, :]).astype(float)
        both = base.available[:, :, None] & base.available[:, None, :]

        def effects(theta: np.ndarray) -> dict[str, np.ndarray]:
            beta, alpha_, gamma = _split(theta[:p], d)
            lam = np.ones(len(lam_param))
            free = lam_param >= 0
            lam[free] = theta[p + lam_param[free]]
            la = lam[nest_of]
            V = np.einsum("r,rna->na", beta, np.nan_to_num(base.x)) + alpha_[None, :] + base.z @ gamma
            f = _parts(np.ones(1), lam, V[:, :, None], base.available, nest_of)
            P, W = f["P"], f["within"]
            # dlog P_j / dV_k = delta_jk / lambda_j + [same nest] (lambda_j - 1) / lambda_j P_k|nest - P_k
            dlog = eye[None] / la[None, :, None] + same[None] * ((la - 1) / la)[None, :, None] * W[:, None, :] - P[:, None, :]
            dPdV = P[:, :, None] * dlog
            out = {}
            for m, name in enumerate(d["regressors"]):
                out[f"d:{name}"] = (beta[m] * dPdV).mean(axis=0)
                E = beta[m] * np.nan_to_num(base.x[m])[:, None, :] * dlog
                with np.errstate(invalid="ignore"):
                    out[f"e:{name}"] = np.where(both, E, 0.0).sum(axis=0) / both.sum(axis=0)
            for v, name in enumerate(d["case_variables"]):
                dz = dPdV @ gamma[v]                                     # (case, j)
                out[f"d:{name}"] = dz.mean(axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    e = np.where(base.available, base.z[:, [v]] * dz / P, 0.0)
                    out[f"e:{name}"] = e.sum(axis=0) / base.available.sum(axis=0)
            return out

        return _choice.effects_result(effects, np.asarray(self.values, float), np.asarray(self.cov, float), d, at, alpha,
                                      c.available.shape[0], self.outcome, self.provenance, "nested logit")

    def summary(self) -> Table:
        st = self.stats
        nest_text = "; ".join(f"{n}: {', '.join(str(a) for a, m in self.nests.items() if m == n)}" for n in self.design["nest_names"])
        notes = [f"Nested logit (FIML, RU2) of the choice {self.outcome}; {self.cases} cases ({self.case}), {self.nobs} alternatives ({self.alternative}).",
                 f"Nests: {nest_text}.",
                 "lambda: the nests' dissimilarity parameters (1: the conditional logit; between 0 and 1: consistent with utility maximisation)"
                 + (", one shared by every nest." if self.dissimilarity == "equal" else ".")]
        if self.fixed_nests:
            notes.append(f"Nests of one alternative ({', '.join(self.fixed_nests)}): lambda fixed at 1 (it cancels in this form).")
        if self.iia_test is not None:
            t = self.iia_test
            notes.append(f"LR test of IIA (every lambda = 1, the conditional logit): {t.statistic:.4f} on {t.df} df, p = {t.p_value:.4g}.")
        notes += [f"Log likelihood: {self.loglike:.4f}", f"Log likelihood (conditional logit): {self.loglike_logit:.4f}",
                  f"Pseudo R-squared against equal shares within each choice set: {st['pseudo_r2']:.4f}.",
                  "Standard errors from the inverse Hessian." if self.cov_type == "hessian" else f"Standard errors clustered by {self.case}."]
        if self.dropped_cases:
            notes.append(f"{self.dropped_cases} cases without exactly one chosen alternative dropped.")
        if not self.converged:
            notes.append(f"The maximisation did not converge in {self.iterations} steps.")
        return _micro.coefficient_table(self.names, self.values, self.se, f"Nested logit: {self.outcome}", notes)

    def plot(self) -> Any:
        from .figures import Estimates, Figure, Panel

        b, s = np.asarray(self.values), self.se
        idx = list(self.names)
        return Figure((Panel((Estimates(pd.Series(b, index=idx), pd.Series(b - 1.96 * s, index=idx), pd.Series(b + 1.96 * s, index=idx), "estimate"),),
                             title=f"Nested logit: {self.outcome}"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.summary().to_frame()

    def __repr__(self) -> str:
        return f"<FittedNestedLogit {self.outcome}: {dict(zip(self.names, np.round(self.values, 4)))}>"


# -- estimation -------------------------------------------------------------------------------------------------------

def _nest_map(choices: _choice.Choices, nests: str | Mapping[Any, Sequence[Any]]) -> dict[Any, str]:
    alts = choices.alternatives
    if isinstance(nests, str):
        col = pd.DataFrame({"a": choices.frame[choices.alternative].to_numpy(object), "n": choices.extra[nests]})
        per = col.groupby("a", sort=False)["n"].nunique()
        if (per > 1).any():
            bad = ", ".join(str(a) for a in per.index[per > 1])
            raise SpecificationError(f"{nests} must give each alternative one nest; it varies for {bad}")
        first = col.drop_duplicates("a").set_index("a")["n"]
        return {a: str(first[a]) for a in alts}
    out: dict[Any, str] = {}
    for nest, members in dict(nests).items():
        for a in ([members] if isinstance(members, str) else members):
            if a in out:
                raise SpecificationError(f"alternative {a!r} is in two nests ({out[a]} and {nest})")
            out[a] = str(nest)
    unknown = [a for a in out if a not in alts]
    if unknown:
        raise SpecificationError(f"nests name alternatives that are not in the data: {', '.join(map(repr, unknown))} (alternatives: {alts})")
    missing = [a for a in alts if a not in out]
    if missing:
        raise SpecificationError(f"every alternative needs a nest; {', '.join(map(repr, missing))} have none")
    return out


def _maximise(theta0: np.ndarray, X: np.ndarray, available: np.ndarray, chosen: np.ndarray, nest_of: np.ndarray,
              lam_param: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Quasi-Newton (BFGS) on log lambda, then Newton steps on lambda: estimates, Hessian, iterations, converged."""
    p = X.shape[2]

    def loglike(t: np.ndarray) -> tuple[float, np.ndarray]:
        ll, g, _ = nested_loglike(t, X, available, chosen, nest_of, lam_param)
        return ll, g

    def negative(u: np.ndarray) -> tuple[float, np.ndarray]:
        t = u.copy()
        t[p:] = np.exp(u[p:])
        ll, g = loglike(t)
        g = g.copy()
        g[p:] *= t[p:]
        return (-ll, -g) if np.isfinite(ll) else (np.inf, np.zeros_like(u))

    u0 = theta0.copy()
    u0[p:] = np.log(theta0[p:])
    res = optimize.minimize(negative, u0, jac=True, method="BFGS", options={"gtol": 1e-8, "maxiter": 2000})
    theta = res.x.copy()
    theta[p:] = np.exp(res.x[p:])
    theta, H, ok = _choice.newton_polish(theta, loglike)
    return theta, H, int(res.nit), ok


@api("pymetrix:nested_logit")
def nested_logit(data: Dataset, choice: str, regressors: Sequence[str] = (), nests: str | Mapping[Any, Sequence[Any]] = "",
                 case: str | None = None, alternative: str | None = None, alternative_constants: bool = False,
                 case_variables: Sequence[str] = (), base: Any = None, dissimilarity: Literal["free", "equal", "one"] = "free",
                 cov: Literal["hessian", "cluster"] = "hessian") -> FittedNestedLogit:
    """A nested logit by full-information maximum likelihood for choice data in long form (one row per case and
    alternative), as ``choice_logit`` takes it. ``nests`` is a series (or axis) giving each row's nest, constant for an
    alternative, or a mapping {nest: [alternatives]}. ``dissimilarity``: "free" (one lambda per nest), "equal" (one
    lambda shared by the nests) or "one" (every lambda 1: the conditional logit). The form is McFadden's,
    consistent with utility maximisation (RU2); nests of one alternative have lambda fixed at 1. ``cov``: the
    inverse Hessian, or clustered by case."""
    from ._capture import capture

    if not nests:
        raise SpecificationError("give nests= (a series naming each alternative's nest, or {nest: [alternatives]})")
    if dissimilarity not in ("free", "equal", "one"):
        raise SpecificationError(f"dissimilarity is 'free', 'equal' or 'one', not {dissimilarity!r}")
    if cov not in ("hessian", "cluster"):
        raise SpecificationError(f"cov is 'hessian' or 'cluster', not {cov!r}")
    with capture() as caught:
        choices = _choice.read_choices(data, choice, regressors, case, alternative, alternative_constants, case_variables, base,
                                       "nested logit", extra=[nests] if isinstance(nests, str) else [])
        nest_map = _nest_map(choices, nests)
        nest_names = list(dict.fromkeys(nest_map[a] for a in choices.alternatives))
        if len(nest_names) < 2:
            raise SpecificationError("a nested logit needs at least two nests (with one, lambda is not identified)")
        nest_of = np.array([nest_names.index(nest_map[a]) for a in choices.alternatives])
        sizes = np.bincount(nest_of, minlength=len(nest_names))
        fixed = tuple(nest_names[m] for m in range(len(nest_names)) if sizes[m] == 1)
        lam_param = np.full(len(nest_names), -1)
        if dissimilarity != "one":
            movable = [m for m in range(len(nest_names)) if sizes[m] > 1]
            lam_param[movable] = 0 if dissimilarity == "equal" else np.arange(len(movable))
        q = int(lam_param.max()) + 1 if (lam_param >= 0).any() else 0
        X, available, chosen = choices.X, choices.available, choices.chosen
        p = X.shape[2]
        beta0, ll_logit = _choice.fit_logit(X, available, chosen)
        if q:
            theta, H, iterations, ok = _maximise(np.concatenate([beta0, np.ones(q)]), X, available, chosen, nest_of, lam_param)
        else:
            theta, H, ok = _choice.newton_polish(beta0, lambda t: nested_loglike(t, X, available, chosen, nest_of, lam_param)[:2])
            iterations = 0
        ll, grad, scores = nested_loglike(theta, X, available, chosen, nest_of, lam_param)
        if not np.all(np.isfinite(theta)) or not np.isfinite(ll):
            raise PyMetrixError("the nested logit did not converge")
        V = _choice.covariance(H, scores, cov)
    names = list(choices.names)
    if q:
        names += ["lambda"] if dissimilarity == "equal" else [f"lambda:{nest_names[m]}" for m in range(len(nest_names)) if lam_param[m] >= 0]
    iia = None
    if q:
        from .stats.base import TestResult

        stat = max(2 * (ll - ll_logit), 0.0)
        iia = TestResult("LR test of IIA (nested against conditional logit)", "every dissimilarity parameter is 1 (the conditional logit: IIA)",
                         float(stat), q, float(sps.chi2.sf(stat, q)), f"chi-squared({q})",
                         details={"log likelihood (nested)": ll, "log likelihood (conditional logit)": ll_logit})
    P = _parts(theta[:p], np.where(lam_param >= 0, theta[p + np.maximum(lam_param, 0)] if q else 1.0, 1.0), X, available, nest_of)["P"]
    used = choices.frame.iloc[np.sort(choices.used)]
    spec = {"choice": choice, "regressors": _micro.names(regressors), "nests": nests if isinstance(nests, str) else {str(k): list(v) for k, v in dict(nests).items()},
            "case": case, "alternative": alternative, "alternative_constants": alternative_constants,
            "case_variables": _micro.names(case_variables), "base": base, "dissimilarity": dissimilarity, "cov": cov}
    prov = _micro.provenance(spec, data, used, [choice, *_micro.names(regressors), *_micro.names(case_variables)], caught,
                             {"type": "nested logit", "estimation": "full-information maximum likelihood (RU2 normalisation)"},
                             converged=ok, resolved={"case": choices.case, "alternative": choices.alternative, "base": choices.design["base"],
                                                     "nests": {str(a): n for a, n in nest_map.items()}, "fixed_nests": list(fixed)})
    design = {**choices.design, "p": p, "nest_of": nest_of, "lam_param": lam_param, "nest_names": nest_names}
    null = -float(np.log(available.sum(axis=1)).sum())
    return FittedNestedLogit(choice, tuple(names), theta, V, nest_map, dissimilarity, ll, ll_logit, null, iia, len(chosen), choices.dropped,
                             int(available.sum()), cov, choices.case, choices.alternative, _choice.probabilities_series(P, choices), fixed,
                             iterations, ok, prov, design)
