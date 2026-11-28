"""Marginal effects and elasticities after a conditional logit (``pymetrix.clogit``).

    fit = choice_logit(trips, "chosen", ["cost", "time"], alternative_constants=True, case_variables=["income"])
    me = clogit_effects(fit)                  # average over cases; at="mean": at the means of the regressors
    me.table("cost")                          # dP(row)/dcost(column), alternatives by alternatives
    me.table("cost", "elasticity")            # the same as elasticities
    me = clogit_effects(fit, dummy=True, changes={"income": 1})
    me.table("wifi", "discrete change")       # 0/1 attributes: P(row) with wifi of the column 1 less with it 0

McFadden's model gives each alternative j in case i the probability P_ij = exp(V_ij) / sum_k exp(V_ik), with
V_ij = x_ij'beta + alpha_j + z_i'gamma_j (alpha and gamma zero for the base). For an attribute x that varies across
alternatives, with one coefficient beta,

    dP_ij / dx_ik = beta P_ij (delta_jk - P_ik),    elasticity  beta x_ik (delta_jk - P_ik)     (= derivative x_ik / P_ij);

for a characteristic z of the case, with one coefficient per alternative,

    dP_ij / dz_i = P_ij (gamma_j - sum_k P_ik gamma_k),    elasticity  z_i (gamma_j - sum_k P_ik gamma_k).

Average effects average the derivatives over every case, counting zero where j or k was not in the choice set (so
each column of derivatives still sums to zero over the alternatives), and the elasticities over the cases where both
were available. Effects at the means evaluate them once, for a case with every alternative, each attribute at its
mean over the cases that had that alternative and each characteristic at its mean. Standard errors are by the delta
method (the derivative of the effects in the coefficients by central differences).

Discrete changes replace the derivative by a finite change, computed from the probabilities themselves: for a 0/1
variable (``dummy=True`` finds them, as statsmodels' ``get_margeff(dummy=True)``: every value 0 or 1, both present)
the change from 0 to 1, and for any variable given in ``changes={name: delta}`` the change from its value to the
value plus delta. For an attribute, alternative k's value changes alone (own effect on P_ik, cross effects on the
others), the others' kept as observed; for a characteristic of the case, it changes for the case. Averaged over
every case (zero where k is not available, so each column still sums to zero over the alternatives) or at the means
(the variable itself at 0 and 1, or at its mean and the mean plus delta). Delta-method standard errors, as above.

Chamberlain's panel logit conditions the unit's fixed effect away, so the probabilities, and their derivatives,
depend on the unidentified fixed effect: there are no marginal effects to estimate. Its coefficients are log odds
ratios within a unit, reported as odds ratios with delta-method standard errors. ``fixed_effect=a`` adds the effects
on Lambda(a + x'beta), the probability for a unit whose fixed effect is ``a`` (0: Stata's ``pu0``); they are labelled
as such, an illustration at a value chosen by the user, not the sample's effects; with ``dummy`` or ``changes``, the
discrete changes of that probability too.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.special import expit

from pymetrix._lazy import stats as sps

from ._api import api
from .errors import SpecificationError
from .output import Table

__all__ = ["ConditionalLogitEffects", "clogit_effects"]

_COLUMNS = ["estimate", "std err", "z", "p-value", "lower", "upper"]
_INDEX = ["variable", "measure", "probability", "change in"]


def _jacobian(f: Any, theta: np.ndarray) -> np.ndarray:
    from .estimation.numdiff import jacobian

    return jacobian(f, theta)


# -- choice data -------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Cases:
    """The choice data by case: attributes (regressor, case, alternative; nan where not available), characteristics
    (case, variable) and availability (case, alternative)."""

    x: np.ndarray
    z: np.ndarray
    available: np.ndarray


def _cases(design: dict[str, Any]) -> _Cases:
    alts = list(design["alternatives"])
    col = {a: i for i, a in enumerate(alts)}
    case = np.asarray(design["case"])
    codes, rows = np.unique(case, return_inverse=True)
    n, A, r = len(codes), len(alts), design["x"].shape[1]
    x = np.full((r, n, A), np.nan)
    available = np.zeros((n, A), bool)
    j = np.array([col[a] for a in design["alternative"]])
    x[:, rows, j] = design["x"].T
    available[rows, j] = True
    z = np.zeros((n, design["z"].shape[1]))
    z[rows] = design["z"]
    return _Cases(x, z, available)


def _at_means(c: _Cases) -> _Cases:
    with np.errstate(invalid="ignore"):
        x = np.nanmean(c.x, axis=1, keepdims=True)
    return _Cases(np.nan_to_num(x), c.z.mean(axis=0, keepdims=True), np.ones((1, c.available.shape[1]), bool))


def _split(theta: np.ndarray, design: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """beta (r), alpha (A, base 0) and gamma (variables, A, base 0) from the coefficients."""
    alts, others = list(design["alternatives"]), list(design["others"])
    r, A = len(design["regressors"]), len(alts)
    pos = [alts.index(a) for a in others]
    beta = theta[:r]
    k = r
    alpha = np.zeros(A)
    if design["constants"]:
        alpha[pos] = theta[k:k + len(others)]
        k += len(others)
    gamma = np.zeros((len(design["case_variables"]), A))
    for v in range(len(design["case_variables"])):
        gamma[v, pos] = theta[k:k + len(others)]
        k += len(others)
    return beta, alpha, gamma


def choice_probabilities(theta: np.ndarray, design: dict[str, Any], c: _Cases) -> np.ndarray:
    """P (case, alternative), zero where an alternative is not available."""
    beta, alpha, gamma = _split(theta, design)
    V = np.einsum("r,rna->na", beta, np.nan_to_num(c.x)) + alpha[None, :] + c.z @ gamma
    V = np.where(c.available, V, -np.inf)
    V = V - V.max(axis=1, keepdims=True)
    e = np.where(c.available, np.exp(V), 0.0)
    return e / e.sum(axis=1, keepdims=True)


def _choice_effects(theta: np.ndarray, design: dict[str, Any], c: _Cases) -> dict[str, np.ndarray]:
    """Derivatives and elasticities, averaged over the cases given (one case: at the means)."""
    beta, _, gamma = _split(theta, design)
    P = choice_probabilities(theta, design, c)
    A = P.shape[1]
    eye = np.eye(A)
    both = c.available[:, :, None] & c.available[:, None, :]
    out = {}
    for m, name in enumerate(design["regressors"]):
        D = beta[m] * P[:, :, None] * (eye[None] - P[:, None, :])                     # (case, j, k)
        E = beta[m] * c.x[m][:, None, :] * (eye[None] - P[:, None, :])
        out[f"d:{name}"] = D.mean(axis=0)
        with np.errstate(invalid="ignore"):
            out[f"e:{name}"] = np.where(both, E, 0.0).sum(axis=0) / both.sum(axis=0)
    for v, name in enumerate(design["case_variables"]):
        g = gamma[v]
        dev = g[None, :] - (P @ g)[:, None]                                           # (case, j)
        out[f"d:{name}"] = (P * dev).mean(axis=0)
        with np.errstate(invalid="ignore"):
            out[f"e:{name}"] = np.where(c.available, c.z[:, [v]] * dev, 0.0).sum(axis=0) / c.available.sum(axis=0)
    return out


def _is_dummy(values: np.ndarray) -> bool:
    """0/1 only, both present (statsmodels' test for a dummy)."""
    v = values[np.isfinite(values)]
    return v.size > 0 and bool(np.all((v == 0) | (v == 1))) and v.min() == 0 and v.max() == 1


def _discrete_plan(design: dict[str, Any], names: list[str], data: list[np.ndarray], dummy: bool,
                   changes: Mapping[str, float] | None) -> dict[str, float | None]:
    """{variable: delta} for the discrete changes (None: from 0 to 1), the variables named in ``changes`` first and
    then, with ``dummy``, every 0/1 variable among ``names`` (their values in ``data``)."""
    plan: dict[str, float | None] = {}
    for name, delta in dict(changes or {}).items():
        if name not in names:
            raise SpecificationError(f"changes: no variable {name!r} in the model: {', '.join(names)}")
        try:
            d = float(delta)
        except (TypeError, ValueError) as exc:
            raise SpecificationError(f"changes: the change in {name} is a number, not {delta!r}") from exc
        if not np.isfinite(d) or d == 0:
            raise SpecificationError(f"changes: the change in {name} is a nonzero number, not {delta!r}")
        plan[name] = d
    if dummy:
        for name, values in zip(names, data):
            if name not in plan and _is_dummy(values):
                plan[name] = None
    return plan


def _choice_discrete(theta: np.ndarray, design: dict[str, Any], c: _Cases, plan: dict[str, float | None]) -> dict[str, np.ndarray]:
    """Discrete changes in the probabilities, averaged over the cases given (one case: at the means): for an
    attribute (P_j with alternative k's value changed, less P_j) as (j, k); for a characteristic, (j,)."""
    out = {}
    regressors, chars = list(design["regressors"]), list(design["case_variables"])
    for name, delta in plan.items():
        if name in regressors:
            m = regressors.index(name)
            A = c.available.shape[1]
            D = np.empty((A, A))
            for k in range(A):
                lo, hi = c.x.copy(), c.x.copy()
                if delta is None:
                    lo[m, :, k], hi[m, :, k] = 0.0, 1.0
                else:
                    hi[m, :, k] += delta
                # an unavailable k: its value is nan, its probability zero either way, so the change is zero
                P0 = choice_probabilities(theta, design, _Cases(lo, c.z, c.available))
                P1 = choice_probabilities(theta, design, _Cases(hi, c.z, c.available))
                D[:, k] = (P1 - P0).mean(axis=0)
            out[f"c:{name}"] = D
        else:
            v = chars.index(name)
            lo, hi = c.z.copy(), c.z.copy()
            if delta is None:
                lo[:, v], hi[:, v] = 0.0, 1.0
            else:
                hi[:, v] += delta
            P0 = choice_probabilities(theta, design, _Cases(c.x, lo, c.available))
            P1 = choice_probabilities(theta, design, _Cases(c.x, hi, c.available))
            out[f"c:{name}"] = (P1 - P0).mean(axis=0)
    return out


def _change_label(delta: float | None) -> str:
    return "0 to 1" if delta is None else f"x to x {'+' if delta > 0 else '-'} {abs(delta):g}"


# -- the result --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class ConditionalLogitEffects:
    """Marginal effects and elasticities after McFadden's choice logit (``kind="choice"``), or odds ratios (and
    effects at a chosen fixed effect) after Chamberlain's panel logit (``kind="panel"``). ``frame`` holds every
    number: one row per variable, measure, probability and the alternative whose attribute changes."""

    kind: str
    outcome: str
    at: str
    frame: pd.DataFrame
    alternatives: tuple[Any, ...] = ()
    regressors: tuple[str, ...] = ()
    case_variables: tuple[str, ...] = ()
    fixed_effect: float | None = None
    alpha: float = 0.05
    cases: int = 0
    discrete: dict[str, str] = field(default_factory=dict)  # variable: its discrete change ("0 to 1", "x to x + 1")
    provenance: Any = field(default=None, repr=False)
    model: str = "conditional logit"  # or "nested logit", "mixed logit" (pymetrix.nested_logit, pymetrix.mixed_logit)

    stage = "estimate"

    @property
    def estimates(self) -> pd.Series:
        return self.frame["estimate"]

    @property
    def se(self) -> pd.Series:
        return self.frame["std err"]

    def table(self, variable: str, measure: Literal["derivative", "elasticity", "discrete change"] = "derivative",
              values: str = "estimate") -> pd.DataFrame:
        """The effects of ``variable``: for an attribute, alternatives by alternatives (row j, column k: the change in
        P(j) for a change in the attribute of k); for a characteristic of the case, one row per alternative.
        ``measure`` "derivative", "elasticity" or "discrete change" (for the variables in ``discrete``); ``values``
        "estimate" or "std err"."""
        if self.kind != "choice":
            raise SpecificationError("tables by alternative are for the choice logit; the panel logit has odds ratios (.frame)")
        if variable not in (*self.regressors, *self.case_variables):
            raise SpecificationError(f"no variable {variable!r}: {', '.join((*self.regressors, *self.case_variables))}")
        if measure not in ("derivative", "elasticity", "discrete change"):
            raise SpecificationError(f"measure is 'derivative', 'elasticity' or 'discrete change', not {measure!r}")
        if measure == "discrete change" and variable not in self.discrete:
            raise SpecificationError(f"no discrete change for {variable!r}: ask for it with changes={{{variable!r}: delta}}"
                                     + (" (or dummy=True for a 0/1 variable)" if self.kind == "choice" else ""))
        part = self.frame.xs((variable, measure), level=("variable", "measure"))[values]
        if variable in self.case_variables:
            out = part.droplevel("change in").reindex([str(a) for a in self.alternatives]).to_frame(variable)
            out.index.name = "probability"
            return out
        out = part.unstack("change in").reindex(index=[str(a) for a in self.alternatives], columns=[str(a) for a in self.alternatives])
        out.columns.name = f"{variable} of"
        return out

    def _label(self) -> str:
        if self.at == "mean":
            return "at the means"
        return "average over cases" if self.kind == "choice" else "average over the observations"

    def summary(self) -> Table:
        f = self.frame
        if self.kind == "panel":
            rows = f.reset_index()
            index = [f"{r.variable}: {r.measure}" for r in rows.itertuples()]
            notes = ["Chamberlain's logit conditions the fixed effects away, so the probabilities and their derivatives (which depend on them) "
                     "are not identified; the coefficients are log odds ratios within a unit.",
                     "Odds ratios exp(b) with delta-method standard errors exp(b) se(b); z and p-values test b = 0 (an odds ratio of 1); "
                     "the interval is exp of b's."]
            if self.fixed_effect is not None:
                notes.append(f"'effect at fixed effect {self.fixed_effect:g}': dP/dx for a unit whose fixed effect is {self.fixed_effect:g}, "
                             f"P = Lambda({self.fixed_effect:g} + x'b), {self._label()}: an illustration at a chosen value, not the sample's effects. "
                             "Delta-method standard errors.")
            if self.discrete:
                notes.append("Discrete changes at that fixed effect for " + "; ".join(f"{v} ({how})" for v, how in self.discrete.items())
                             + ": Lambda(a + x'b) after the change less before it.")
            return Table(_COLUMNS, f.to_numpy(float).tolist(), index=index, index_name="parameter",
                         title=f"Odds ratios: {self.outcome}", notes=tuple(notes))
        wanted = {v: ("discrete change" if v in self.discrete else "derivative") for v in (*self.regressors, *self.case_variables)}
        keep = [(v, m, p, k) for v, m, p, k in f.index if m == wanted[v] and (k == p or k == "")]
        index = [(f"ΔP({p})/Δ{v}({k})" if k else f"ΔP({p})/Δ{v}") if m == "discrete change" else
                 (f"dP({p})/d{v}({k})" if k else f"dP({p})/d{v}") for v, m, p, k in keep]
        iia = ("under IIA each column's cross elasticities are equal at the means." if self.model == "conditional logit" else
               f"the {self.model} does not assume IIA, so a column's cross elasticities differ.")
        notes = (f"Marginal effects on the choice probabilities{'' if self.model == 'conditional logit' else f' ({self.model})'}, {self._label()} ({self.cases} cases): own effects of each "
                 "attribute and the effects of each characteristic of the case. Cross effects and elasticities are in the tables by variable.",
                 "Delta-method standard errors. Derivatives of an attribute sum to zero over the alternatives; " + iia)
        if self.discrete:
            notes += ("Discrete changes (ΔP) in place of derivatives for " + "; ".join(f"{v} ({how})" for v, how in self.discrete.items())
                      + ": the probability after the change less before it, computed exactly; they sum to zero over the alternatives too.",)
        return Table(_COLUMNS, f.loc[keep].to_numpy(float).tolist(), index=index, index_name="effect",
                     title=f"Marginal effects: {self.outcome}", notes=notes)

    def plot(self) -> Any:
        from .figures import Estimates, Figure, Panel

        f = self.frame
        if self.kind == "panel":
            part = f.xs("odds ratio", level="measure")
            idx = [str(i[0]) for i in part.index]
            title = f"Odds ratios: {self.outcome}"
        else:
            keep = [i for i in f.index if i[1] == "elasticity" and (i[3] == i[2] or i[3] == "")]
            part = f.loc[keep]
            idx = [f"{v} ({p})" for v, _, p, _ in keep]
            title = f"Own elasticities ({self._label()}): {self.outcome}"
        est = pd.Series(part["estimate"].to_numpy(), index=pd.Index(idx, name="effect"))
        lo = pd.Series(part["lower"].to_numpy(), index=est.index)
        hi = pd.Series(part["upper"].to_numpy(), index=est.index)
        return Figure((Panel((Estimates(est, lo, hi, "estimate"),), title=title),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.frame

    def __repr__(self) -> str:
        return f"<ConditionalLogitEffects {self.kind} {self.outcome}, {self._label()}: {len(self.frame)} rows>"


def _rows(values: np.ndarray, se: np.ndarray, q: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        z = values / se
    p = 2 * sps.norm.sf(np.abs(z))
    return np.column_stack([values, se, z, p, values - q * se, values + q * se])


@api("pymetrix:clogit_effects")
def clogit_effects(fit: Any, at: Literal["average", "mean"] = "average", fixed_effect: float | None = None, alpha: float = 0.05,
                   dummy: bool = False, changes: Mapping[str, float] | None = None) -> ConditionalLogitEffects:
    """Marginal effects and elasticities after ``choice_logit``: own and cross derivatives of the probabilities in
    each attribute, and derivatives in each characteristic of the case, averaged over the cases (``at="average"``)
    or at the means (``"mean"``), with delta-method standard errors. After ``conditional_logit`` (fixed effects):
    odds ratios, and with ``fixed_effect`` the effects on the probability of a unit with that fixed effect.

    Discrete changes (measure "discrete change"), beside the derivatives: with ``dummy=True`` for every 0/1 variable
    (all values 0 or 1, as statsmodels' ``get_margeff(dummy=True)``) the change in the probabilities when it goes
    from 0 to 1, and for each variable in ``changes={name: delta}`` the change when it goes from its value to the
    value plus delta. For an attribute, one alternative's value changes at a time (own and cross changes, as the
    derivatives); for a characteristic, the case's. The panel logit has them only at a ``fixed_effect``."""
    from .clogit import FittedConditionalLogit

    if not isinstance(fit, FittedConditionalLogit):
        raise SpecificationError("clogit_effects takes a conditional_logit or choice_logit result")
    if fit.design is None:
        raise SpecificationError("this fit keeps no design (it was saved, or made by an older version): estimate it again")
    if at not in ("average", "mean"):
        raise SpecificationError(f"at is 'average' (over the cases) or 'mean' (at the means), not {at!r}")
    if not 0 < alpha < 1:
        raise SpecificationError("alpha is between 0 and 1")
    q = float(sps.norm.ppf(1 - alpha / 2))
    theta, V = np.asarray(fit.values, float), np.asarray(fit.cov, float)
    design = fit.design
    if fit.kind == "panel":
        names = list(fit.names)
        X = np.asarray(design["x"], float)
        plan = _discrete_plan(design, names, list(X.T), dummy, changes)
        if plan and fixed_effect is None:
            raise SpecificationError("the panel logit's probabilities depend on the fixed effects, so discrete changes need a fixed_effect "
                                     "to evaluate them at (as the derivatives do)")
        frame, cases = _panel(fit, theta, V, at, fixed_effect, q, plan), int(fit.groups)
        extra: dict[str, Any] = {"discrete": {v: _change_label(d) for v, d in plan.items()}}
    else:
        if fixed_effect is not None:
            raise SpecificationError("fixed_effect is for the panel logit; the choice logit has no fixed effects")
        if design.get("alternative") is None:
            raise SpecificationError("effects by alternative need the alternatives: estimate with alternative=")
        if not design["regressors"] and not design["case_variables"]:
            raise SpecificationError("the model has only alternative constants: there are no variables to take effects of")
        c = _cases(design)
        base = c if at == "average" else _at_means(c)
        regressors, chars = list(design["regressors"]), list(design["case_variables"])
        plan = _discrete_plan(design, regressors + chars, [*c.x, *c.z.T], dummy, changes)

        def effects(t: np.ndarray) -> dict[str, np.ndarray]:
            return {**_choice_effects(t, design, base), **_choice_discrete(t, design, base, plan)}

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
            measure = {"d": "derivative", "e": "elasticity", "c": "discrete change"}[kind]
            if name in design["regressors"]:
                index += [(name, measure, j, kk) for j in alts for kk in alts]
            else:
                index += [(name, measure, j, "") for j in alts]
        frame = pd.DataFrame(_rows(values, se, q), index=pd.MultiIndex.from_tuples(index, names=_INDEX), columns=_COLUMNS)
        cases = c.available.shape[0]
        extra = {"alternatives": tuple(design["alternatives"]), "regressors": tuple(design["regressors"]),
                 "case_variables": tuple(design["case_variables"]), "discrete": {v: _change_label(d) for v, d in plan.items()}}
    prov = fit.provenance
    if prov is not None:
        settings = {"at": at, "fixed_effect": fixed_effect, "alpha": alpha, "dummy": bool(dummy), "changes": dict(changes or {})}
        prov = dataclasses.replace(prov, spec={**dict(prov.spec), "marginal_effects": settings},
                                   method={**dict(prov.method), "marginal effects": "delta method (central differences)"})
    return ConditionalLogitEffects(fit.kind, fit.outcome, at, frame, fixed_effect=None if fixed_effect is None else float(fixed_effect),
                                   alpha=alpha, cases=cases, provenance=prov, **extra)


def _panel(fit: Any, theta: np.ndarray, V: np.ndarray, at: str, fixed_effect: float | None, q: float,
           plan: dict[str, float | None] | None = None) -> pd.DataFrame:
    names = list(fit.names)
    se = np.sqrt(np.diag(V))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = theta / se
    odds = np.column_stack([np.exp(theta), np.exp(theta) * se, z, 2 * sps.norm.sf(np.abs(z)), np.exp(theta - q * se), np.exp(theta + q * se)])
    index = [(n, "odds ratio", "", "") for n in names]
    blocks = [odds]
    if fixed_effect is not None:
        a = float(fixed_effect)
        if not np.isfinite(a):
            raise SpecificationError("fixed_effect is a number (the value of the fixed effect to evaluate the probability at)")
        X = np.asarray(fit.design["x"], float)
        base = X if at == "average" else X.mean(axis=0, keepdims=True)

        def effects(t: np.ndarray) -> np.ndarray:
            p = expit(a + base @ t)
            return (p * (1 - p)).mean() * t  # dP/dx_m = b_m P (1 - P), averaged over the rows (one row: at the means)

        values = effects(theta)
        J = _jacobian(effects, theta)
        blocks.append(_rows(values, np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", J, V, J), 0.0)), q))
        index += [(n, f"effect at fixed effect {a:g}", "", "") for n in names]
        for name, delta in (plan or {}).items():
            m = names.index(name)
            lo, hi = base.copy(), base.copy()
            if delta is None:
                lo[:, m], hi[:, m] = 0.0, 1.0
            else:
                hi[:, m] += delta

            def change(t: np.ndarray, lo: np.ndarray = lo, hi: np.ndarray = hi) -> np.ndarray:
                return np.array([(expit(a + hi @ t) - expit(a + lo @ t)).mean()])

            J = _jacobian(change, theta)
            blocks.append(_rows(change(theta), np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", J, V, J), 0.0)), q))
            index.append((name, f"discrete change at fixed effect {a:g}", "", ""))
    return pd.DataFrame(np.vstack(blocks), index=pd.MultiIndex.from_tuples(index, names=_INDEX), columns=_COLUMNS)
