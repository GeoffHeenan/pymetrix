"""Conditional logit: Chamberlain's (1980) fixed-effects logit for panels, and McFadden's (1974) conditional logit for
choices among alternatives. Both maximise the same likelihood.

    conditional_logit(panel, "employed", ["kids", "married"])                  # unit fixed effects conditioned away
    choice_logit(trips, "chosen", ["cost", "time"], case="trip", alternative="mode",
                 alternative_constants=True, case_variables=["income"])       # McFadden's model of the chosen mode

Within a group g (a unit's periods, or a case's alternatives) with observations t, outcomes y_t in {0, 1} and
k_g = sum_t y_t, the probability of the observed outcomes given their sum is

    P(y | k_g) = exp(sum_t y_t x_t' beta) / sum_{d: sum_t d_t = k_g} exp(sum_t d_t x_t' beta),

which does not depend on a group's own intercept (the fixed effect). The denominator is computed by Howard's
recursion over the group's observations (f(t, k) = f(t-1, k) + exp(x_t' beta) f(t-1, k-1)), with its first and second
derivatives, so the score and the Hessian are exact. With one chosen alternative per case (k_g = 1) the denominator
is sum_t exp(x_t' beta): McFadden's choice probabilities, computed directly. Groups whose outcomes are all 0 or all 1
carry no information and are dropped. Standard errors are from the inverse Hessian, or clustered by group (a sandwich
of the groups' scores).

For choice data, ``alternative_constants`` adds a constant for each alternative but the first (the base), and each
of ``case_variables`` (which do not vary across a case's alternatives) enters interacted with the alternatives'
indicators, again with the base left out. Choice sets may differ across cases: each case lists the alternatives it
could choose from.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import linalg

from . import _micro
from ._api import api
from .errors import PyMetrixError, SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["FittedConditionalLogit", "choice_logit", "conditional_logit", "conditional_loglike"]


# -- the likelihood ---------------------------------------------------------------------------------------------------

def _group_terms(X: np.ndarray, k: int, beta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """log of sum over subsets of size k of exp(sum d_t x_t'beta), with its gradient and Hessian in beta."""
    n, p = X.shape
    eta = X @ beta
    shift = eta.max() if k == 1 else np.sort(eta)[-k:].mean()  # the k largest dominate: their product starts near 1
    e = np.exp(eta - shift)
    if k == 1:
        s = e.sum()
        w = e / s
        g = w @ X
        H = (X * w[:, None]).T @ X - np.outer(g, g)
        return float(np.log(s) + shift), g, H
    # Howard's recursion on scaled values: f[j] = sum over subsets of size j of the first t observations
    f = np.zeros(k + 1)
    f[0] = 1.0
    G = np.zeros((k + 1, p))
    Hh = np.zeros((k + 1, p, p))
    logscale = 0.0
    for t in range(n):
        x = X[t]
        xx = np.outer(x, x)
        top = min(t + 1, k)
        for j in range(top, 0, -1):
            Hh[j] = Hh[j] + e[t] * (Hh[j - 1] + np.outer(x, G[j - 1]) + np.outer(G[j - 1], x) + xx * f[j - 1])
            G[j] = G[j] + e[t] * (G[j - 1] + x * f[j - 1])
            f[j] = f[j] + e[t] * f[j - 1]
        m = np.abs(f).max()
        if m > 1e200 or m < 1e-200:
            f, G, Hh = f / m, G / m, Hh / m
            logscale += np.log(m)
    F = f[k]
    g = G[k] / F
    H = Hh[k] / F - np.outer(g, g)
    return float(np.log(F) + logscale + k * shift), g, H


def conditional_loglike(beta: np.ndarray, groups: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """The conditional log likelihood of (X_g, y_g) groups at ``beta``, with its gradient, Hessian and each group's
    score (rows)."""
    p = len(beta)
    ll, grad, hess = 0.0, np.zeros(p), np.zeros((p, p))
    scores = np.zeros((len(groups), p))
    for i, (X, y) in enumerate(groups):
        k = int(round(y.sum()))
        logden, g, H = _group_terms(X, k, beta)
        ll += float(y @ X @ beta) - logden
        s = y @ X - g
        scores[i] = s
        grad += s
        hess -= H
    return ll, grad, hess, scores


def _maximise(groups: list[tuple[np.ndarray, np.ndarray]], p: int, tol: float = 1e-10, max_iter: int = 200) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, int, bool]:
    """Newton's method with step halving."""
    beta = np.zeros(p)
    ll, grad, hess, scores = conditional_loglike(beta, groups)
    for it in range(1, max_iter + 1):
        try:
            step = linalg.solve(-hess, grad, assume_a="pos")
        except (linalg.LinAlgError, ValueError):
            step = np.linalg.lstsq(-hess, grad, rcond=None)[0]
        t = 1.0
        while True:
            new = beta + t * step
            ll_new, g_new, h_new, s_new = conditional_loglike(new, groups)
            if np.isfinite(ll_new) and ll_new >= ll - 1e-12 * (1 + abs(ll)):
                break
            t /= 2
            if t < 1e-10:
                return beta, ll, hess, scores, it, False
        beta, ll, grad, hess, scores = new, ll_new, g_new, h_new, s_new
        if np.max(np.abs(t * step)) < tol * (1 + np.max(np.abs(beta))) or abs(grad @ step) < tol:
            return beta, ll, hess, scores, it, True
    return beta, ll, hess, scores, max_iter, False


# -- the result -------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FittedConditionalLogit:
    """A conditional logit: Chamberlain's fixed-effects logit (``kind="panel"``) or McFadden's choice model (``"choice"``)."""

    kind: str
    outcome: str
    names: tuple[str, ...]
    values: np.ndarray
    cov: np.ndarray = field(repr=False)
    loglike: float = 0.0
    loglike_null: float = 0.0  # all coefficients zero: equal probabilities within each group
    groups: int = 0
    dropped_groups: int = 0
    nobs: int = 0
    cov_type: str = "hessian"
    group: str = ""
    alternative: str | None = None
    probabilities: pd.Series | None = field(default=None, repr=False)  # each row's fitted probability (choice data)
    iterations: int = 0
    converged: bool = True
    provenance: Any = field(default=None, repr=False)
    design: Any = field(default=None, repr=False)  # the rows used, for marginal effects (pymetrix.clogit_effects)

    stage = "estimate"
    next_steps = (("Marginal effects…", "ClogitMarginalEffects", "fit", "estimate"),)

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
        return {"nobs": float(self.nobs), "groups": float(self.groups), "loglike": self.loglike, "loglike_null": self.loglike_null,
                "pseudo_r2": 1 - self.loglike / self.loglike_null if self.loglike_null else float("nan"),
                "aic": -2 * self.loglike + 2 * k, "bic": -2 * self.loglike + k * np.log(self.nobs)}

    def odds_ratios(self) -> pd.DataFrame:
        b, s = np.asarray(self.values), self.se
        return pd.DataFrame({"odds ratio": np.exp(b), "lower": np.exp(b - 1.959964 * s), "upper": np.exp(b + 1.959964 * s)},
                            index=pd.Index(self.names, name="parameter"))

    def marginal_effects(self, at: Literal["average", "mean"] = "average", fixed_effect: float | None = None, alpha: float = 0.05,
                         dummy: bool = False, changes: Any = None) -> Any:
        """Marginal effects and elasticities of the choice probabilities (choice data), or odds ratios (and, with
        ``fixed_effect``, effects at that value of the fixed effect) for the panel logit; discrete changes for 0/1
        variables (``dummy=True``) and for ``changes={name: delta}``: ``clogit_effects``."""
        from .clogit_effects import clogit_effects

        return clogit_effects(self, at=at, fixed_effect=fixed_effect, alpha=alpha, dummy=dummy, changes=changes)

    def summary(self) -> Table:
        st = self.stats
        if self.kind == "panel":
            notes = [f"Chamberlain's conditional (fixed-effects) logit of {self.outcome}, conditioning on each {self.group}'s number of ones.",
                     f"{self.groups} groups, {self.nobs} observations; {self.dropped_groups} groups with all-0 or all-1 outcomes dropped."]
        else:
            notes = [f"McFadden's conditional logit of the choice {self.outcome}; {self.groups} cases ({self.group}), {self.nobs} alternatives"
                     + (f", alternatives {self.alternative}" if self.alternative else "") + ".",
                     f"Pseudo R-squared against equal shares within each choice set: {st['pseudo_r2']:.4f}."]
            if self.dropped_groups:
                notes.append(f"{self.dropped_groups} cases without exactly one chosen alternative dropped.")
        notes += [f"Log likelihood: {self.loglike:.4f}", f"Log likelihood (all zero): {self.loglike_null:.4f}",
                  "Standard errors from the inverse Hessian." if self.cov_type == "hessian" else f"Standard errors clustered by {self.group}."]
        if not self.converged:
            notes.append(f"Newton's method did not converge in {self.iterations} steps.")
        title = "Conditional logit (fixed effects)" if self.kind == "panel" else "Conditional logit (McFadden)"
        return _micro.coefficient_table(self.names, self.values, self.se, f"{title}: {self.outcome}", notes)

    def plot(self) -> Any:
        from .figures import Estimates, Figure, Panel

        b, s = np.asarray(self.values), self.se
        idx = list(self.names)
        return Figure((Panel((Estimates(pd.Series(b, index=idx), pd.Series(b - 1.96 * s, index=idx), pd.Series(b + 1.96 * s, index=idx), "estimate"),),
                             title=f"Conditional logit: {self.outcome}"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.summary().to_frame()

    def __repr__(self) -> str:
        return f"<FittedConditionalLogit {self.kind} {self.outcome}: {dict(zip(self.names, np.round(self.values, 4)))}>"


def _fit(frame: pd.DataFrame, outcome: str, X: np.ndarray, gid: np.ndarray, names: list[str], cov: str, exactly_one: bool) -> dict[str, Any]:
    y = frame[outcome].to_numpy(float)
    if not np.isin(y, (0.0, 1.0)).all():
        raise SpecificationError(f"{outcome} must be 0 or 1")
    groups, kept, dropped = [], [], 0
    order = pd.Series(np.arange(len(y))).groupby(gid, sort=True)
    for _, rows in order:
        r = rows.to_numpy()
        k = y[r].sum()
        if (exactly_one and k != 1) or (not exactly_one and (k == 0 or k == len(r))):
            dropped += 1
            continue
        groups.append((X[r], y[r]))
        kept.append(r)
    if not groups:
        raise SpecificationError("no group has variation in the outcome")
    used = np.concatenate(kept)
    Xu = X[used]
    if np.linalg.matrix_rank(Xu - pd.DataFrame(Xu).groupby(gid[used]).transform("mean").to_numpy()) < X.shape[1]:
        raise SpecificationError("the regressors are collinear, or do not vary within groups (they are absorbed by the fixed effects)")
    beta, ll, hess, scores, it, ok = _maximise(groups, X.shape[1])
    if not np.all(np.isfinite(beta)):
        raise PyMetrixError("the conditional logit did not converge")
    inv = np.linalg.inv(-hess)
    if cov == "cluster":
        G = len(groups)
        V = inv @ (scores.T @ scores) @ inv * G / (G - 1)
    else:
        V = inv
    ll0 = conditional_loglike(np.zeros(X.shape[1]), groups)[0]
    return {"beta": beta, "cov": V, "ll": ll, "ll0": ll0, "groups": len(groups), "dropped": dropped, "nobs": len(used), "used": used,
            "iterations": it, "converged": ok, "group_list": groups}


@api("pymetrix:conditional_logit")
def conditional_logit(data: Dataset, outcome: str, regressors: Sequence[str], group: str | None = None,
                      cov: Literal["hessian", "cluster"] = "hessian") -> FittedConditionalLogit:
    """Chamberlain's fixed-effects logit: the 0/1 ``outcome`` on ``regressors``, conditioning on each group's number of
    ones, which removes the group's own intercept. ``group`` is the panel's unit axis unless named (a series or an
    axis); ``cov`` the inverse Hessian or errors clustered by group."""
    from ._capture import capture

    regressors = _micro.names(regressors)
    if not regressors:
        raise SpecificationError("the conditional logit needs at least one regressor")
    if group is None:
        if not data.entity_axes:
            raise SpecificationError("give group= (the series or axis whose fixed effects are conditioned away)")
        group = data.entity_axes[0].name
    if cov not in ("hessian", "cluster"):
        raise SpecificationError(f"cov is 'hessian' or 'cluster', not {cov!r}")
    frame = _micro.columns(data, [outcome, *regressors, group], "conditional logit")
    X = _micro.numeric(frame, regressors, "conditional logit")
    gid = pd.factorize(frame[group])[0]
    with capture() as caught:
        out = _fit(frame, outcome, X, gid, regressors, cov, exactly_one=False)
    used = frame.iloc[out["used"]]
    spec = {"outcome": outcome, "regressors": regressors, "group": group, "cov": cov}
    prov = _micro.provenance(spec, data, used, [outcome, *regressors], caught,
                             {"type": "conditional logit", "estimation": "conditional maximum likelihood (Chamberlain)"}, converged=out["converged"])
    design = {"x": X[out["used"]], "case": gid[out["used"]]}
    return FittedConditionalLogit("panel", outcome, tuple(regressors), out["beta"], out["cov"], out["ll"], out["ll0"], out["groups"], out["dropped"],
                                  out["nobs"], cov, group, None, None, out["iterations"], out["converged"], prov, design)


@api("pymetrix:choice_logit")
def choice_logit(data: Dataset, choice: str, regressors: Sequence[str] = (), case: str | None = None, alternative: str | None = None,
                 alternative_constants: bool = False, case_variables: Sequence[str] = (), base: Any = None,
                 cov: Literal["hessian", "cluster"] = "hessian") -> FittedConditionalLogit:
    """McFadden's conditional logit for choice data in long form, one row per case and alternative: ``choice`` is 1
    for the chosen alternative, ``regressors`` vary across alternatives (cost, time). ``case`` and ``alternative``
    name the series or axes identifying them (default: the dataset's first two entity axes). With
    ``alternative_constants`` a constant for each alternative but ``base`` (default the first); ``case_variables``
    enter interacted with the alternatives' indicators."""
    from ._capture import capture

    regressors, case_vars = _micro.names(regressors), _micro.names(case_variables)
    axes = [a.name for a in data.entity_axes]
    case = case or (axes[0] if axes else None)
    if case is None:
        raise SpecificationError("give case= (the series or axis identifying each choice situation)")
    if alternative is None and len(axes) > 1:
        alternative = axes[1] if axes[0] == case else axes[0]
    if (alternative_constants or case_vars) and alternative is None:
        raise SpecificationError("alternative constants and case variables need alternative= (the series or axis naming the alternatives)")
    if cov not in ("hessian", "cluster"):
        raise SpecificationError(f"cov is 'hessian' or 'cluster', not {cov!r}")
    frame = _micro.columns(data, [choice, *regressors, *case_vars, case, *([alternative] if alternative else [])], "choice logit")
    blocks, names = [], list(regressors)
    alts, others, b = [], [], None
    if regressors:
        blocks.append(_micro.numeric(frame, regressors, "choice logit"))
    if alternative is not None and (alternative_constants or case_vars):
        alts = sorted(frame[alternative].unique(), key=str)
        b = alts[0] if base is None else base
        if b not in alts:
            raise SpecificationError(f"the base alternative {b!r} is not among {alts}")
        others = [a for a in alts if a != b]
        dummies = np.column_stack([(frame[alternative] == a).to_numpy(float) for a in others])
        if alternative_constants:
            blocks.append(dummies)
            names += [f"{a}" for a in others]
        for v in case_vars:
            values = _micro.numeric(frame, [v], "choice logit")[:, 0]
            if (pd.Series(values).groupby(frame[case].to_numpy()).nunique() > 1).any():
                raise SpecificationError(f"{v} varies across a case's alternatives: put it among the regressors")
            blocks.append(dummies * values[:, None])
            names += [f"{v}:{a}" for a in others]
    if not blocks:
        raise SpecificationError("the choice logit needs regressors, alternative constants or case variables")
    X = np.column_stack(blocks)
    gid = pd.factorize(frame[case])[0]
    with capture() as caught:
        out = _fit(frame, choice, X, gid, names, cov, exactly_one=True)
    used = frame.iloc[out["used"]]
    beta = out["beta"]
    eta = X[out["used"]] @ beta
    ug = gid[out["used"]]
    ex = np.exp(eta - pd.Series(eta).groupby(ug).transform("max").to_numpy())
    prob = ex / pd.Series(ex).groupby(ug).transform("sum").to_numpy()
    spec = {"choice": choice, "regressors": regressors, "case": case, "alternative": alternative, "alternative_constants": alternative_constants,
            "case_variables": case_vars, "base": base, "cov": cov}
    prov = _micro.provenance(spec, data, used, [choice, *regressors, *case_vars], caught,
                             {"type": "conditional logit", "estimation": "maximum likelihood (McFadden)"}, converged=out["converged"])
    u = out["used"]
    design = {"x": X[u][:, :len(regressors)], "z": _micro.numeric(frame, case_vars, "choice logit")[u] if case_vars else np.zeros((len(u), 0)),
              "case": gid[u], "alternative": None if alternative is None else frame[alternative].to_numpy(object)[u],
              "alternatives": list(alts) if alts else (sorted(frame[alternative].unique(), key=str) if alternative is not None else []),
              "base": b, "others": list(others), "constants": bool(alternative_constants), "regressors": list(regressors),
              "case_variables": list(case_vars)}
    return FittedConditionalLogit("choice", choice, tuple(names), beta, out["cov"], out["ll"], out["ll0"], out["groups"], out["dropped"],
                                  out["nobs"], cov, case, alternative, pd.Series(prob, index=used.index, name="probability"), out["iterations"],
                                  out["converged"], prov, design)
