"""What the choice models beyond the conditional logit share (``nested_logit``, ``mixed_logit``): long choice data,
one row per case and alternative, read into arrays of cases by alternatives; the multinomial logit as a start; and
marginal effects in the form ``clogit_effects`` gives them (``ConditionalLogitEffects``).

The design is the conditional logit's (``pymetrix.clogit.choice_logit``): attributes that vary across alternatives
with one coefficient each, a constant for each alternative but the base, and characteristics of the case interacted
with the alternatives' indicators, the base left out. Choice sets may differ across cases.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import linalg

from pymetrix._lazy import stats as sps

from . import _micro
from .errors import PyMetrixError, SpecificationError

if TYPE_CHECKING:
    from .data import Dataset


@dataclass(frozen=True, eq=False)
class Choices:
    """Choice data by case: ``X`` (case, alternative, coefficient) with zeros where an alternative is not available,
    ``available`` (case, alternative), ``chosen`` (each case's chosen column), ``rows`` (each cell's position in
    ``frame``, -1 where not available)."""

    frame: pd.DataFrame
    names: list[str]
    X: np.ndarray
    available: np.ndarray
    chosen: np.ndarray
    alternatives: list[Any]
    case_ids: np.ndarray
    rows: np.ndarray
    design: dict[str, Any]
    dropped: int
    case: str
    alternative: str
    extra: dict[str, np.ndarray] = field(default_factory=dict)  # extra columns, per row of frame

    @property
    def used(self) -> np.ndarray:
        """The positions in ``frame`` of the rows used, case by case."""
        return self.rows[self.available]

    def per_case(self, column: str) -> np.ndarray:
        """``column``'s value for each case (the first available alternative's row)."""
        first = self.rows[np.arange(len(self.rows)), self.available.argmax(axis=1)]
        return self.extra[column][first]


def read_choices(data: Dataset, choice: str, regressors: Sequence[str], case: str | None, alternative: str | None,
                 alternative_constants: bool, case_variables: Sequence[str], base: Any, what: str,
                 extra: Sequence[str] = ()) -> Choices:
    """Long choice data as ``Choices``, with the conditional logit's design; cases without exactly one chosen
    alternative are dropped (counted in ``dropped``). ``extra`` names more columns to read (a nest, a panel)."""
    regressors, case_vars = _micro.names(regressors), _micro.names(case_variables)
    axes = [a.name for a in data.entity_axes]
    case = case or (axes[0] if axes else None)
    if case is None:
        raise SpecificationError("give case= (the series or axis identifying each choice situation)")
    if alternative is None and len(axes) > 1:
        alternative = axes[1] if axes[0] == case else axes[0]
    if alternative is None:
        raise SpecificationError(f"the {what} needs alternative= (the series or axis naming the alternatives)")
    extra = [e for e in extra if e]
    frame = _micro.columns(data, [choice, *regressors, *case_vars, case, alternative, *extra], what)
    if frame.duplicated([case, alternative]).any():
        raise SpecificationError(f"a case lists an alternative twice: each ({case}, {alternative}) pair is one row")
    y = frame[choice].to_numpy(float)
    if not np.isin(y, (0.0, 1.0)).all():
        raise SpecificationError(f"{choice} must be 0 or 1")
    alts = sorted(frame[alternative].unique(), key=str)
    b = alts[0] if base is None else base
    if b not in alts:
        raise SpecificationError(f"the base alternative {b!r} is not among {alts}")
    others = [a for a in alts if a != b]
    blocks, names = [], list(regressors)
    if regressors:
        blocks.append(_micro.numeric(frame, regressors, what))
    dummies = np.column_stack([(frame[alternative] == a).to_numpy(float) for a in others]) if others else np.zeros((len(frame), 0))
    if alternative_constants:
        blocks.append(dummies)
        names += [f"{a}" for a in others]
    for v in case_vars:
        values = _micro.numeric(frame, [v], what)[:, 0]
        if (pd.Series(values).groupby(frame[case].to_numpy()).nunique() > 1).any():
            raise SpecificationError(f"{v} varies across a case's alternatives: put it among the regressors")
        blocks.append(dummies * values[:, None])
        names += [f"{v}:{a}" for a in others]
    if not blocks:
        raise SpecificationError(f"the {what} needs regressors, alternative constants or case variables")
    Xl = np.column_stack(blocks)
    codes, ids = pd.factorize(frame[case], sort=True)
    col = {a: i for i, a in enumerate(alts)}
    acol = np.array([col[a] for a in frame[alternative]])
    n, A = len(ids), len(alts)
    chosen_count = np.bincount(codes, weights=y, minlength=n)
    keep_case = chosen_count == 1
    dropped = int((~keep_case).sum())
    if not keep_case.any():
        raise SpecificationError("no case has exactly one chosen alternative")
    new = np.cumsum(keep_case) - 1
    keep_row = keep_case[codes]
    rows = np.full((int(keep_case.sum()), A), -1)
    rows[new[codes[keep_row]], acol[keep_row]] = np.flatnonzero(keep_row)
    available = rows >= 0
    X = np.zeros((*rows.shape, Xl.shape[1]))
    X[available] = Xl[rows[available]]
    chosen = np.zeros(len(rows), int)
    hit = keep_row & (y == 1)
    chosen[new[codes[hit]]] = acol[hit]
    used = rows[available]
    Xu = Xl[used]
    if np.linalg.matrix_rank(Xu - pd.DataFrame(Xu).groupby(codes[used]).transform("mean").to_numpy()) < Xl.shape[1]:
        raise SpecificationError("the regressors are collinear, or do not vary across a case's alternatives")
    design = {"x": _micro.numeric(frame, regressors, what)[used] if regressors else np.zeros((len(used), 0)),
              "z": _micro.numeric(frame, case_vars, what)[used] if case_vars else np.zeros((len(used), 0)),
              "case": codes[used], "alternative": frame[alternative].to_numpy(object)[used], "alternatives": list(alts), "base": b,
              "others": list(others), "constants": bool(alternative_constants), "regressors": list(regressors), "case_variables": list(case_vars)}
    return Choices(frame, names, X, available, chosen, list(alts), np.asarray(ids)[keep_case], rows, design, dropped, case, alternative,
                   {e: frame[e].to_numpy(object) for e in extra})


# -- the multinomial logit: the start, and the model every dissimilarity parameter of 1 gives -------------------------

def logit_loglike(beta: np.ndarray, X: np.ndarray, available: np.ndarray, chosen: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """McFadden's log likelihood with its gradient, Hessian and each case's score."""
    V = np.where(available, X @ beta, -np.inf)
    V = V - V.max(axis=1, keepdims=True)
    e = np.where(available, np.exp(V), 0.0)
    P = e / e.sum(axis=1, keepdims=True)
    n = len(chosen)
    ll = float(np.log(P[np.arange(n), chosen]).sum())
    xbar = np.einsum("na,nap->np", P, X)
    scores = X[np.arange(n), chosen] - xbar
    hess = -(np.einsum("na,nap,naq->pq", P, X, X) - xbar.T @ xbar)
    return ll, scores.sum(axis=0), hess, scores


def fit_logit(X: np.ndarray, available: np.ndarray, chosen: np.ndarray, tol: float = 1e-12, max_iter: int = 100) -> tuple[np.ndarray, float]:
    """The multinomial (conditional) logit by Newton's method with step halving: estimates and log likelihood."""
    beta = np.zeros(X.shape[2])
    ll, grad, hess, _ = logit_loglike(beta, X, available, chosen)
    for _ in range(max_iter):
        try:
            step = linalg.solve(-hess, grad, assume_a="pos")
        except (linalg.LinAlgError, ValueError):
            step = np.linalg.lstsq(-hess, grad, rcond=None)[0]
        t = 1.0
        while True:
            new = beta + t * step
            ll_new, g_new, h_new, _ = logit_loglike(new, X, available, chosen)
            if np.isfinite(ll_new) and ll_new >= ll - 1e-12 * (1 + abs(ll)):
                break
            t /= 2
            if t < 1e-10:
                return beta, ll
        beta, ll, grad, hess = new, ll_new, g_new, h_new
        if np.max(np.abs(t * step)) < tol * (1 + np.max(np.abs(beta))):
            break
    return beta, ll


def newton_polish(theta: np.ndarray, loglike: Callable[[np.ndarray], tuple[float, np.ndarray]], tol: float = 1e-10,
                  max_iter: int = 20) -> tuple[np.ndarray, np.ndarray, bool]:
    """Newton steps with a central-difference Hessian of the analytic gradient, from a quasi-Newton optimum, until the
    step is below ``tol``: the estimates, the Hessian at them, and whether it converged."""
    from .estimation.numdiff import jacobian

    def hessian(t: np.ndarray) -> np.ndarray:
        H = jacobian(lambda u: loglike(u)[1], t)
        return (H + H.T) / 2

    ll, grad = loglike(theta)
    for _ in range(max_iter):
        H = hessian(theta)
        try:
            step = -np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            return theta, H, False
        if np.max(np.abs(step)) < tol * (1 + np.max(np.abs(theta))):
            return theta + step, H, True  # the Hessian a step away from the optimum: no need to compute it again
        t = 1.0
        while True:
            new = theta + t * step
            ll_new, g_new = loglike(new)
            if np.isfinite(ll_new) and ll_new >= ll - 1e-9 * (1 + abs(ll)):
                break
            t /= 2
            if t < 1e-6:
                return theta, H, bool(np.max(np.abs(grad)) < 1e-4)
        theta, ll, grad = new, ll_new, g_new
    return theta, hessian(theta), False


def covariance(H: np.ndarray, scores: np.ndarray, cov: str, clusters: np.ndarray | None = None) -> np.ndarray:
    """The inverse of minus the Hessian, or a sandwich of the scores summed by cluster (each row a cluster when
    ``clusters`` is None), with the G / (G - 1) correction."""
    try:
        inv = np.linalg.inv(-H)
    except np.linalg.LinAlgError:
        raise PyMetrixError("the Hessian is singular at the estimates: a parameter is not identified") from None
    if cov == "hessian":
        return inv
    if clusters is not None:
        scores = pd.DataFrame(scores).groupby(clusters).sum().to_numpy()
    G = len(scores)
    return inv @ (scores.T @ scores) @ inv * G / (G - 1)


# -- marginal effects ---------------------------------------------------------------------------------------------------

def effects_result(effects: Callable[[np.ndarray], dict[str, np.ndarray]], theta: np.ndarray, V: np.ndarray, design: dict[str, Any],
                   at: str, alpha: float, cases: int, outcome: str, provenance: Any, model: str) -> Any:
    """``ConditionalLogitEffects`` from ``effects(theta)`` ({"d:x": derivatives, "e:x": elasticities}, alternatives by
    alternatives for an attribute, one per alternative for a characteristic), with delta-method standard errors (the
    derivative in the parameters by central differences)."""
    import dataclasses

    from .clogit_effects import _COLUMNS, _INDEX, ConditionalLogitEffects, _jacobian, _rows

    if not 0 < alpha < 1:
        raise SpecificationError("alpha is between 0 and 1")
    q = float(sps.norm.ppf(1 - alpha / 2))
    keys = list(effects(theta))

    def flat(t: np.ndarray) -> np.ndarray:
        got = effects(t)
        return np.concatenate([got[k].ravel() for k in keys])

    values = flat(theta)
    J = _jacobian(flat, theta)
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", J, V, J), 0.0))
    alts = [str(a) for a in design["alternatives"]]
    index = []
    for k in keys:
        kind, name = k.split(":", 1)
        measure = {"d": "derivative", "e": "elasticity"}[kind]
        if name in design["regressors"]:
            index += [(name, measure, j, kk) for j in alts for kk in alts]
        else:
            index += [(name, measure, j, "") for j in alts]
    frame = pd.DataFrame(_rows(values, se, q), index=pd.MultiIndex.from_tuples(index, names=_INDEX), columns=_COLUMNS)
    if provenance is not None:
        provenance = dataclasses.replace(provenance, spec={**dict(provenance.spec), "marginal_effects": {"at": at, "alpha": alpha}},
                                         method={**dict(provenance.method), "marginal effects": "delta method (central differences)"})
    return ConditionalLogitEffects("choice", outcome, at, frame, alternatives=tuple(design["alternatives"]), regressors=tuple(design["regressors"]),
                                   case_variables=tuple(design["case_variables"]), alpha=alpha, cases=cases, provenance=provenance, model=model)


def probabilities_series(P: np.ndarray, choices: Choices) -> pd.Series:
    """Each used row's fitted probability, in the data's index."""
    out = np.full(len(choices.frame), np.nan)
    out[choices.rows[choices.available]] = P[choices.available]
    used = np.sort(choices.used)
    return pd.Series(out[used], index=choices.frame.index[used], name="probability")
