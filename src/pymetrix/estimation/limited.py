"""Limited dependent variables by maximum likelihood, and marginal effects.

    Equation("visits ~ age + income", family=NegativeBinomial()).fit(data, method=ML())
    Equation("visits ~ age + income", family=ZeroInflatedPoisson(inflate="age")).fit(data, method=ML())
    Equation("rating ~ x1 + x2", family=OrderedProbit()).fit(data, method=ML())        # cut points, no constant
    Equation("mode ~ cost + time", family=Multinomial()).fit(data, method=ML())         # against the first outcome
    Equation("hours ~ educ + exper", family=Tobit(left=0)).fit(data, method=ML())        # censored at zero
    Equation("lwage ~ educ + exper", ).fit(data, method=Heckman("inlf ~ educ + kids"))  # sample selection
    fit.marginal_effects()                                                               # average marginal effects

The likelihoods are written here and maximised by BFGS, then polished by Newton steps with a numerical Hessian.
Parameters with a range (the negative binomial's alpha, Tobit's sigma, ordered cut points, Heckman's rho and
sigma) are estimated on an unrestricted scale and reported on their own, with delta-method standard errors.
Covariances: "classic" (inverse Hessian), "white" (sandwich) or Cluster(...), from numerical scores.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import expit, gammaln, log_ndtr, logsumexp, ndtr

from pymetrix._lazy import stats as sps

from ..errors import SpecificationError
from ..forms import Widget
from .covariance import Cluster
from .families import (
    Logit,
    Multinomial,
    NegativeBinomial,
    OrderedLogit,
    OrderedProbit,
    Poisson,
    Probit,
    Tobit,
    ZeroInflatedPoisson,
)
from .numdiff import hessian, jacobian

if TYPE_CHECKING:
    from ..data import Dataset
    from .equation import Equation
    from .fitted import FittedEquation

LIMITED = (NegativeBinomial, ZeroInflatedPoisson, OrderedLogit, OrderedProbit, Multinomial, Tobit)
_TINY = 1e-300


@dataclass
class _Model:
    """One model's likelihood on its data: ``theta`` is the unrestricted vector the optimiser moves."""

    names: list[str]  # reported parameter names
    start: np.ndarray
    loglike: Callable[[np.ndarray], np.ndarray]  # per observation
    report: Callable[[np.ndarray], np.ndarray]  # theta -> reported values
    response: Callable[[np.ndarray, Mapping[str, np.ndarray]], np.ndarray]  # E[y] (n,) or probabilities (n, J)
    inputs: dict[str, np.ndarray]  # "X" (and "Z"): the data the response is computed from
    input_names: dict[str, list[str]]
    categories: list[float] = field(default_factory=list)  # ordered and multinomial outcomes
    fitted: Callable[[np.ndarray], np.ndarray] | None = None  # in-sample fitted values (default: the response)


def _label(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


# -- the models -------------------------------------------------------------------------------------------------------

def _negbin(y: np.ndarray, X: np.ndarray, names: list[str]) -> _Model:
    if np.any(y < 0) or not np.allclose(y, np.round(y)):
        raise SpecificationError("the negative binomial needs a non-negative count dependent variable")
    k = X.shape[1]

    def mu(t: np.ndarray, inp: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.exp(np.clip(inp["X"] @ t[:k], -700, 700))

    def ll(t: np.ndarray) -> np.ndarray:
        r = np.exp(-t[k])  # 1 / alpha
        m = mu(t, {"X": X})
        return gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(r / (r + m)) + y * np.log(m / (r + m) + _TINY)

    start = np.r_[_poisson_start(y, X), np.log(0.5)]
    return _Model(names + ["alpha"], start, ll, lambda t: np.r_[t[:k], np.exp(t[k])], mu, {"X": X}, {"X": names})


def _zip(y: np.ndarray, X: np.ndarray, names: list[str], Z: np.ndarray, z_names: list[str]) -> _Model:
    if np.any(y < 0) or not np.allclose(y, np.round(y)):
        raise SpecificationError("the zero-inflated Poisson needs a non-negative count dependent variable")
    k = X.shape[1]
    zero = y == 0

    def parts(t: np.ndarray, inp: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        eta = np.clip(inp["X"] @ t[:k], -700, 700)
        return eta, inp["Z"] @ t[k:]

    def ll(t: np.ndarray) -> np.ndarray:
        eta, g = parts(t, {"X": X, "Z": Z})
        log_pi, log_1mpi = -np.logaddexp(0, -g), -np.logaddexp(0, g)
        mu = np.exp(eta)
        positive = log_1mpi + y * eta - mu - gammaln(y + 1)
        return np.where(zero, np.logaddexp(log_pi, log_1mpi - mu), positive)

    def mean(t: np.ndarray, inp: Mapping[str, np.ndarray]) -> np.ndarray:
        eta, g = parts(t, inp)
        return (1 - expit(g)) * np.exp(eta)

    start = np.r_[_poisson_start(y, X), np.zeros(Z.shape[1])]
    return _Model(names + [f"inflate: {n}" for n in z_names], start, ll, lambda t: np.asarray(t, float), mean,
                  {"X": X, "Z": Z}, {"X": names, "Z": z_names})


def _ordered(y: np.ndarray, X: np.ndarray, names: list[str], family: Any) -> _Model:
    keep = [i for i, n in enumerate(names) if n != "const"]  # the cut points take the constant's place
    X, names = X[:, keep], [names[i] for i in keep]
    cats = sorted(set(y.tolist()))
    if len(cats) < 2:
        raise SpecificationError("an ordered model needs at least two outcomes")
    J, k = len(cats), X.shape[1]
    code = np.searchsorted(cats, y)
    cdf = expit if isinstance(family, OrderedLogit) else ndtr
    ppf = (lambda p: np.log(p / (1 - p))) if isinstance(family, OrderedLogit) else sps.norm.ppf

    def cuts(t: np.ndarray) -> np.ndarray:
        return t[k] + np.r_[0.0, np.cumsum(np.exp(t[k + 1:]))]

    def probs(t: np.ndarray, inp: Mapping[str, np.ndarray]) -> np.ndarray:
        xb = inp["X"] @ t[:k]
        edges = np.column_stack([np.zeros(len(xb)), cdf(cuts(t)[None, :] - xb[:, None]), np.ones(len(xb))])
        return np.clip(np.diff(edges, axis=1), _TINY, 1.0)

    def ll(t: np.ndarray) -> np.ndarray:
        return np.log(probs(t, {"X": X})[np.arange(len(y)), code])

    shares = np.cumsum(np.bincount(code, minlength=J))[:-1] / len(y)
    c0 = ppf(np.clip(shares, 1e-6, 1 - 1e-6))
    start = np.r_[np.zeros(k), c0[0], np.log(np.maximum(np.diff(c0), 1e-3))]
    values = np.asarray(cats, float)
    return _Model(names + [f"cut{j}" for j in range(1, J)], start, ll, lambda t: np.r_[t[:k], cuts(t)], probs, {"X": X},
                  {"X": names}, list(cats), fitted=lambda t: probs(t, {"X": X}) @ values)


def _multinomial(y: np.ndarray, X: np.ndarray, names: list[str], family: Multinomial) -> _Model:
    cats = sorted(set(y.tolist()))
    if len(cats) < 2:
        raise SpecificationError("a multinomial model needs at least two outcomes")
    base = cats[0] if family.base is None else float(family.base)
    if base not in cats:
        raise SpecificationError(f"the base outcome {family.base!r} is not among the outcomes ({', '.join(map(_label, cats))})")
    others = [c for c in cats if c != base]
    J, k = len(cats), X.shape[1]
    code = np.searchsorted(cats, y)
    base_index = cats.index(base)
    order = [base_index] + [cats.index(c) for c in others]  # columns of eta: the base (zeros), then the others

    def probs(t: np.ndarray, inp: Mapping[str, np.ndarray]) -> np.ndarray:
        B = t.reshape(len(others), k)
        eta = np.column_stack([np.zeros(len(inp["X"]))] + [inp["X"] @ b for b in B])
        logp = eta - logsumexp(eta, axis=1, keepdims=True)
        out = np.empty_like(logp)
        out[:, order] = np.exp(logp)
        return out

    def ll(t: np.ndarray) -> np.ndarray:
        return np.log(np.clip(probs(t, {"X": X})[np.arange(len(y)), code], _TINY, 1.0))

    values = np.asarray(cats, float)
    labels = [f"{_label(c)}: {n}" for c in others for n in names]
    return _Model(labels, np.zeros(len(others) * k), ll, lambda t: np.asarray(t, float), probs, {"X": X}, {"X": names}, list(cats),
                  fitted=lambda t: values[np.argmax(probs(t, {"X": X}), axis=1)])


def _tobit(y: np.ndarray, X: np.ndarray, names: list[str], family: Tobit) -> _Model:
    left = -np.inf if family.left is None else float(family.left)
    right = np.inf if family.right is None else float(family.right)
    if left >= right:
        raise SpecificationError("Tobit's left limit must be below its right limit")
    if np.any(y < left - 1e-12) or np.any(y > right + 1e-12):
        raise SpecificationError(f"the dependent variable has values outside the limits ({left:g}, {right:g})")
    low, high = y <= left, y >= right
    mid = ~(low | high)
    k = X.shape[1]

    def ll(t: np.ndarray) -> np.ndarray:
        xb, s = X @ t[:k], np.exp(t[k])
        out = np.empty(len(y))
        out[mid] = sps.norm.logpdf((y[mid] - xb[mid]) / s) - np.log(s)
        out[low] = log_ndtr((left - xb[low]) / s)
        out[high] = log_ndtr((xb[high] - right) / s)
        return out

    def mean(t: np.ndarray, inp: Mapping[str, np.ndarray]) -> np.ndarray:  # E[y], censored as observed
        xb, s = inp["X"] @ t[:k], np.exp(t[k])
        a = (left - xb) / s if np.isfinite(left) else np.full_like(xb, -np.inf)
        b = (right - xb) / s if np.isfinite(right) else np.full_like(xb, np.inf)
        Pa, Pb = ndtr(a), ndtr(b)
        out = (Pb - Pa) * xb + s * (sps.norm.pdf(a) - sps.norm.pdf(b))
        if np.isfinite(left):
            out = out + left * Pa
        if np.isfinite(right):
            out = out + right * (1 - Pb)
        return out

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    start = np.r_[beta, np.log(max(np.std(y - X @ beta), 1e-6))]
    return _Model(names + ["sigma"], start, ll, lambda t: np.r_[t[:k], np.exp(t[k])], mean, {"X": X}, {"X": names})


def _poisson_start(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    from .ml import _newton

    try:
        return _newton(Poisson(), y, X, np.ones(len(y)), 100, 1e-8)[0]
    except Exception:  # noqa: BLE001 - a start value only
        beta = np.zeros(X.shape[1])
        return beta


def build(family: Any, y: np.ndarray, X: np.ndarray, names: list[str], Z: np.ndarray | None = None,
          z_names: list[str] | None = None) -> _Model:
    if isinstance(family, NegativeBinomial):
        return _negbin(y, X, names)
    if isinstance(family, ZeroInflatedPoisson):
        return _zip(y, X, names, Z if Z is not None else np.ones((len(y), 1)), z_names or ["const"])
    if isinstance(family, (OrderedLogit, OrderedProbit)):
        return _ordered(y, X, names, family)
    if isinstance(family, Multinomial):
        return _multinomial(y, X, names, family)
    if isinstance(family, Tobit):
        return _tobit(y, X, names, family)
    raise SpecificationError(f"no likelihood for {type(family).__name__}")


# -- maximisation and covariance --------------------------------------------------------------------------------------

def _gradient(f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
    h = 1e-5 * np.maximum(np.abs(x), 0.1)
    g = np.empty(len(x))
    for i in range(len(x)):
        e = np.zeros(len(x))
        e[i] = h[i]
        g[i] = (f(x + e) - f(x - e)) / (2 * h[i])
    return g


def maximise(loglike: Callable[[np.ndarray], np.ndarray], start: np.ndarray, w: np.ndarray, max_iter: int = 100,
             tol: float = 1e-10) -> tuple[np.ndarray, float, int]:
    """The maximum of sum(w * loglike(theta)): BFGS from ``start``, then Newton steps with a numerical Hessian."""
    from ..progress import check

    def total(t: np.ndarray) -> float:
        value = float(np.sum(w * loglike(t)))
        return value if np.isfinite(value) else -1e300

    def objective(t: np.ndarray) -> float:
        check()  # Cancel in the app stops here
        return -total(t)

    found = optimize.minimize(objective, np.asarray(start, float), jac=lambda t: -_gradient(total, t), method="BFGS",
                              options={"gtol": 1e-7, "maxiter": max(200, 20 * max_iter)})
    theta, ll, iterations = found.x, total(found.x), int(found.nit)
    for _ in range(max_iter):  # Newton polish: the BFGS result to full precision
        g = _gradient(total, theta)
        H = hessian(total, theta)
        try:
            step = np.linalg.solve(-H, g)
        except np.linalg.LinAlgError:
            break
        t = 1.0
        while t > 1e-6:
            new = theta + t * step
            new_ll = total(new)
            if new_ll >= ll - 1e-12 * abs(ll):
                break
            t /= 2
        else:
            break
        change, theta, ll = new_ll - ll, new, new_ll
        iterations += 1
        if np.max(np.abs(t * step)) < tol * (1 + np.max(np.abs(theta))) and abs(change) < tol * (1 + abs(ll)):
            break
    if not np.isfinite(ll) or ll <= -1e299:
        raise SpecificationError("maximum likelihood failed: the likelihood is not finite at any point tried")
    return theta, ll, iterations


def covariance(kind: Any, loglike: Callable[[np.ndarray], np.ndarray], theta: np.ndarray, w: np.ndarray,
               groups: list[np.ndarray]) -> np.ndarray:
    """``classic``: inverse of minus the Hessian; ``white``: the sandwich; ``Cluster(...)``: scores summed by group."""
    H = hessian(lambda t: float(np.sum(w * loglike(t))), theta)
    try:
        inv_info = np.linalg.inv(-H)
    except np.linalg.LinAlgError:
        raise SpecificationError("the information matrix is singular (perfect prediction, or collinear regressors?)") from None
    if kind == "classic":
        return inv_info
    S = jacobian(lambda t: w * loglike(t), theta)  # (n, p): each observation's score
    if kind == "white":
        return inv_info @ (S.T @ S) @ inv_info
    if isinstance(kind, Cluster):
        if len(groups) != 1:
            raise NotImplementedError("two-way clustering for these models")
        codes = pd.factorize(groups[0])[0]
        G = codes.max() + 1
        sums = np.zeros((G, S.shape[1]))
        np.add.at(sums, codes, S)
        return G / (G - 1) * inv_info @ (sums.T @ sums) @ inv_info
    raise SpecificationError(f"these models take cov='classic', 'white' or Cluster(...), not {kind!r}")


def _reported(model: _Model, theta: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = model.report(theta)
    J = jacobian(model.report, theta)  # delta method: the reported scale's covariance
    return values, J @ cov @ J.T


# -- estimation ------------------------------------------------------------------------------------------------------

def _second_design(eq: Equation, rhs: str, data: Dataset, span: Any, where: Any, weights: Any, missing: str) -> Any:
    from .design import linear_design
    from .equation import Equation as Eq

    return linear_design(Eq(f"{eq.lhs} ~ {rhs}" if rhs.strip() else f"{eq.lhs} ~ 1"), data, span, where, weights, missing)


def estimate(method: Any, eq: Equation, data: Dataset, span: Any, where: Any, weights: Any, missing: str) -> FittedEquation:
    from . import criteria
    from .design import cov_groups, linear_design, provenance
    from .fitted import FittedEquation
    from .params import point

    family = eq.family
    if eq.form != "linear":
        raise SpecificationError(f"{type(family).__name__} equations are linear in their coefficients: y ~ x1 + x2")
    if eq.fixed_effects:
        raise NotImplementedError("fixed effects in nonlinear ML models are not supported (incidental parameters)")
    if method.cov == "hac":
        raise SpecificationError("these models take cov='classic', 'white' or Cluster(...)")
    d = linear_design(eq, data, span, where, weights, missing)
    y, X, names, index = d.y, d.X, list(d.names), d.index
    Z = z_names = None
    if isinstance(family, ZeroInflatedPoisson):
        dz = _second_design(eq, family.inflate, data, span, where, weights, missing)
        common = index.intersection(dz.index)
        rows, zrows = index.get_indexer(common), dz.index.get_indexer(common)
        y, X, index = y[rows], X[rows], common
        Z, z_names = dz.X[zrows], list(dz.names)
        d_w = d.w[rows] if d.w is not None else None
    else:
        d_w = d.w
    w = d_w if d_w is not None else np.ones(len(y))
    model = build(family, y, X, names, Z, z_names)
    theta, ll, iterations = maximise(model.loglike, model.start, w, method.max_iter, method.tol)
    cov_internal = covariance(method.cov, model.loglike, theta, w, cov_groups(method.cov, data, index))
    values, cov = _reported(model, theta, cov_internal)
    null = _null_model(family, y)
    _, ll0, _ = maximise(null.loglike, null.start, w, method.max_iter, method.tol)
    n, p = len(y), len(theta)
    lr = 2 * (ll - ll0)
    fitted = model.fitted(theta) if model.fitted is not None else model.response(theta, model.inputs)
    if isinstance(family, Multinomial):
        resid = (y != fitted).astype(float)  # 1 where the most likely outcome is not the one observed
    else:
        resid = y - fitted
    stats = {"nobs": float(n), "k": float(p), "loglike": float(ll), "ll_null": float(ll0), "pseudo_r2": float(1 - ll / ll0),
             "lr": float(lr), "lr_pvalue": float(sps.chi2.sf(lr, p - len(null.start))), "aic": criteria.aic(ll, p),
             "bic": criteria.bic(ll, p, n), "hq": criteria.hq(ll, p, n), "iterations": float(iterations)}
    notes = _notes(family, model, theta)
    return FittedEquation(
        equation=eq, params=point(model.names, values, cov), method=method,
        resid=pd.Series(resid, index=index, name="resid"), fitted=pd.Series(fitted, index=index, name="fitted"),
        stats=stats, provenance=provenance(eq, method, data, index, d.dropped), design=pd.DataFrame(X, index=index, columns=names),
        weights=weights,
        extra={"converged": True, "notes": notes,
               "limited": {"theta": theta, "cov": cov_internal, "y": y, "Z": Z, "z_names": z_names, "categories": model.categories}})


def _null_model(family: Any, y: np.ndarray) -> _Model:
    one = np.ones((len(y), 1))
    if isinstance(family, ZeroInflatedPoisson):
        return build(family, y, one, ["const"], one, ["const"])
    return build(family, y, one, ["const"])


def _notes(family: Any, model: _Model, theta: np.ndarray) -> list[str]:
    if isinstance(family, NegativeBinomial):
        return ["Negative binomial (NB2): Var(y) = mu + alpha mu^2; alpha = 0 is the Poisson."]
    if isinstance(family, ZeroInflatedPoisson):
        return ["Zero-inflated Poisson: the inflate terms give the log odds of a structural zero."]
    if isinstance(family, (OrderedLogit, OrderedProbit)):
        return [f"Outcomes {', '.join(map(_label, model.categories))}: P(y <= j) = F(cut_j - x b); no constant, the cut points take its place."]
    if isinstance(family, Multinomial):
        return [f"Outcomes {', '.join(map(_label, model.categories))}; coefficients against the base outcome "
                f"{_label(model.categories[0] if family.base is None else family.base)}."]
    if isinstance(family, Tobit):
        limits = [f"left at {family.left:g}" for _ in [0] if family.left is not None] + [f"right at {family.right:g}" for _ in [0] if family.right is not None]
        return [f"Tobit: censored {' and '.join(limits) or 'nowhere'}; fitted values are E[y], censored as observed."]
    return []


def _rebuild(fit: FittedEquation) -> tuple[_Model, np.ndarray, np.ndarray]:
    info = fit.extra["limited"]
    X = fit.design.to_numpy(float)
    model = build(fit.equation.family, np.asarray(info["y"], float), X, list(fit.design.columns), info["Z"], info["z_names"])
    return model, np.asarray(info["theta"], float), np.asarray(info["cov"], float)


# -- prediction ------------------------------------------------------------------------------------------------------

def _inputs(fit: FittedEquation, data: Dataset) -> tuple[dict[str, np.ndarray], pd.Index]:
    """The regressors (and inflate terms) of the fit evaluated on ``data``, on the rows where all are known."""
    from ..expr import evaluate

    def columns(eq: Any, names: Sequence[str]) -> pd.DataFrame:
        exprs = {t.label: t.expr for t in eq.terms}
        out = {}
        for n in names:
            out[n] = pd.Series(1.0, index=data.frame.index) if n == "const" else pd.Series(
                np.asarray(evaluate(exprs[n], data.lagged), float), index=data.frame.index)
        return pd.DataFrame(out)

    frames = {"X": columns(fit.equation, list(fit.design.columns))}
    info = fit.extra["limited"]
    if info.get("z_names"):
        from .equation import Equation as Eq

        family = fit.equation.family
        frames["Z"] = columns(Eq(f"{fit.equation.lhs} ~ {family.inflate or '1'}"), info["z_names"])
    ok = np.logical_and.reduce([f.notna().all(axis=1).to_numpy() for f in frames.values()])
    index = data.frame.index[ok]
    return {k: f.to_numpy(float)[ok] for k, f in frames.items()}, index


def predict(fit: FittedEquation, data: Dataset, span: Any, kind: str) -> pd.Series | pd.DataFrame:
    """E[y] (counts, Tobit), the probabilities of each outcome (ordered, multinomial: a table), or the linear index."""
    model, theta, _ = _rebuild(fit)
    inputs, index = _inputs(fit, data)
    mask = pd.Series(np.asarray(data.sample_mask(span)), index=data.frame.index).reindex(index).to_numpy(bool)
    k = inputs["X"].shape[1]
    if kind == "index":
        if isinstance(fit.equation.family, Multinomial):
            raise SpecificationError("a multinomial model has one index per outcome: use kind='response' for the probabilities")
        return pd.Series(inputs["X"] @ theta[:k], index=index, name="predicted")[mask]
    out = model.response(theta, inputs)
    if out.ndim == 2:
        return pd.DataFrame(out, index=index, columns=[f"P({_label(c)})" for c in model.categories])[mask]
    return pd.Series(out, index=index, name="predicted")[mask]


# -- marginal effects ------------------------------------------------------------------------------------------------

def marginal_effects(fit: FittedEquation, at: Literal["average", "mean"] = "average", alpha: float = 0.05) -> Any:
    """The change in E[y] (counts, Tobit: the censored mean), the probability (logit, probit) or each outcome's
    probability (ordered, multinomial) for a unit change in each regressor: averaged over the sample (``at=
    "average"``) or at the regressors' means. A 0/1 regressor's effect is the change from 0 to 1. Standard errors
    by the delta method. Each term counts on its own: for x and x^2, the effect of each is reported separately."""
    from ..output import Table

    family = fit.equation.family
    if isinstance(family, LIMITED):
        model, theta, cov = _rebuild(fit)
        response, inputs, names = model.response, model.inputs, model.input_names
        categories = model.categories if isinstance(family, (OrderedLogit, OrderedProbit, Multinomial)) else []
    elif isinstance(family, (Logit, Probit, Poisson)):
        if fit.design is None:
            raise SpecificationError("this fit keeps no design matrix")
        theta, cov = np.asarray(fit.params.values, float), np.asarray(fit.params.cov, float)
        inputs, names, categories = {"X": fit.design.to_numpy(float)}, {"X": list(fit.design.columns)}, []

        def response(t: np.ndarray, inp: Mapping[str, np.ndarray]) -> np.ndarray:
            return family.mean(inp["X"] @ t)
    else:
        raise SpecificationError("marginal effects are for logit, probit, Poisson and the limited dependent variable models; "
                                 "in a linear model the coefficients are the marginal effects")
    if at not in ("average", "mean"):
        raise SpecificationError("at='average' (over the sample) or at='mean' (at the regressors' means)")
    variables = list(dict.fromkeys(n for group in names.values() for n in group if n != "const"))
    if not variables:
        raise SpecificationError("the model has no regressors besides the constant")
    base = {k: (v.mean(axis=0, keepdims=True) if at == "mean" else v) for k, v in inputs.items()}
    binary = {v: all(np.isin(inputs[k][:, names[k].index(v)], (0.0, 1.0)).all() for k in inputs if v in names[k]) for v in variables}

    def effects(t: np.ndarray) -> np.ndarray:
        out = []
        for v in variables:
            if binary[v]:
                hi = {k: _set(base[k], names[k], v, 1.0) for k in base}
                lo = {k: _set(base[k], names[k], v, 0.0) for k in base}
                change = response(t, hi) - response(t, lo)
            else:
                h = 1e-5 * max(float(np.mean(np.abs(inputs["X"][:, names["X"].index(v)]))) if v in names["X"] else 1.0, 1e-3)
                up = {k: _shift(base[k], names[k], v, h) for k in base}
                down = {k: _shift(base[k], names[k], v, -h) for k in base}
                change = (response(t, up) - response(t, down)) / (2 * h)
            out.append(np.atleast_1d(np.mean(change, axis=0)))
        return np.concatenate(out)

    value = effects(theta)
    J = jacobian(effects, theta)
    se = np.sqrt(np.maximum(np.diag(J @ cov @ J.T), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = value / se
    q = sps.norm.ppf(1 - alpha / 2)
    rows = [[float(v), float(s), float(zz), float(2 * sps.norm.sf(abs(zz))), float(v - q * s), float(v + q * s)]
            for v, s, zz in zip(value, se, z)]
    if categories:
        index = [f"{v}: P({_label(c)})" for v in variables for c in categories]
    else:
        index = list(variables)
    what = ("each outcome's probability" if categories else "the probability" if isinstance(family, (Logit, Probit))
            else "E[y], censored as observed" if isinstance(family, Tobit) else "E[y]")
    notes = (f"{'Average marginal effects' if at == 'average' else 'Marginal effects at the means'} on {what}; "
             "0/1 regressors: the change from 0 to 1. Delta-method standard errors.",
             *(["Terms built from the same variable (x and x^2) are each varied on their own."]
               if any("^" in v or "**" in v or "*" in v for v in variables) else []))
    return Table(["effect", "std err", "z", "p-value", f"[{alpha / 2:.3g}", f"{1 - alpha / 2:.3g}]"], rows, index=index,
                 index_name="regressor", title="Marginal effects", notes=notes)


def _set(M: np.ndarray, names: Sequence[str], v: str, value: float) -> np.ndarray:
    if v not in names:
        return M
    out = np.array(M, float)
    out[:, list(names).index(v)] = value
    return out


def _shift(M: np.ndarray, names: Sequence[str], v: str, h: float) -> np.ndarray:
    if v not in names:
        return M
    out = np.array(M, float)
    out[:, list(names).index(v)] += h
    return out


# -- Heckman's sample selection --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Heckman:
    """Heckman's sample-selection model, for an outcome observed only where ``selection`` is 1.

    The outcome equation allows for the selection's correlation with the outcome. ``twostep``: a probit of the
    selection, then least squares with the inverse Mills ratio (lambda), with Heckman's corrected covariance;
    ``ml``: full maximum likelihood (rho and sigma estimated).

    Least squares on the selected observations alone is biased when the unobservables that decide selection
    also move the outcome (rho not zero, e.g. wage offers seen only for those who work). The correction rests on
    joint normality of the two errors; it is credible only with an exclusion restriction, a variable in the
    selection equation that is not in the outcome equation. Without one, identification comes from the probit's
    nonlinearity alone, lambda is nearly collinear with the regressors and the estimates are fragile. The
    coefficient on lambda (rho times sigma) measures the selection: a significant one says the uncorrected
    regression would be biased. The outcome coefficients describe the whole population, not only the selected.
    Maximum likelihood is more efficient if normality holds; the two-step is simpler and a useful check.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset, Equation, Heckman

        mroz = Dataset.example("mroz")                           # wages are observed only for the 428 women who worked (inlf = 1)
        eq = Equation("lwage ~ educ + exper")
        fit = eq.fit(mroz, method=Heckman("inlf ~ educ + exper + kidslt6 + nwifeinc"))
        print(fit.summary())                                     # lambda (rho times sigma) measures the selection; kidslt6 and nwifeinc are excluded from the wage equation
        print(eq.fit(mroz, method=Heckman("inlf ~ educ + exper + kidslt6 + nwifeinc", estimator="ml")).summary())  # rho and sigma by maximum likelihood
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("coef, std err, z", ("the outcome equation, for the whole population; two-step standard errors are Heckman's "
                              "corrected ones, ml's from the inverse Hessian (or white)")),
        ("select: ...", ("the probit of the selection equation: effects on the index of being observed, not on the "
                         "outcome")),
        ("lambda", ("two-step: the coefficient on the inverse Mills ratio, rho times sigma; significant means least "
                    "squares on the selected sample is biased")),
        ("rho, sigma", ("ml: the errors' correlation and the outcome error's standard deviation, with standard "
                        "errors; the two-step note gives them as point values")),
        ("Sample", "the selected observations of the outcome equation; the note gives the total used by the probit"),
        ("fit.stats", "rho, sigma, lambda, nobs (every observation) and nobs_selected"),
    )

    label: ClassVar[str] = "Heckman selection"
    topics: ClassVar[tuple[str, ...]] = ("maximum-likelihood",)
    references: ClassVar[tuple[str, ...]] = (
        "Heckman, J. J. (1979). Sample selection bias as a specification error. Econometrica 47(1), 153-161. doi:10.2307/1912352",
    )
    kind: ClassVar[str] = "heckman"

    selection: Annotated[str, Widget("text", help="the selection equation: a 0/1 variable on its regressors, e.g. inlf ~ educ + kids")] = ""
    estimator: Annotated[Literal["twostep", "ml"], Widget("choice", help="two-step (probit, then least squares with lambda) or maximum likelihood")] = "twostep"
    cov: Annotated[str | Cluster, Widget("", help="ml: classic (inverse Hessian) or white (sandwich); the two-step uses "
                                         "Heckman's corrected covariance")] = "classic"
    max_iter: Annotated[int, Widget("", min=1, max=10000, advanced=True,
                                    help="maximum likelihood only: Newton steps after the quasi-Newton search, which may take 20 times as many")] = 100
    tol: Annotated[float, Widget("", min=1e-14, max=1e-2, advanced=True,
                                 help="maximum likelihood only: relative change in the estimates and likelihood that stops it")] = 1e-10

    def estimate(self, eq: Equation, data: Dataset, *, span: Any = None, where: Any = None, weights: Any = None,
                 missing: str = "drop") -> FittedEquation:
        return _heckman(self, eq, data, span, where, weights, missing)


def _heckman(m: Heckman, eq: Equation, data: Dataset, span: Any, where: Any, weights: Any, missing: str) -> FittedEquation:
    from . import criteria
    from .design import linear_design, provenance
    from .equation import Equation as Eq
    from .families import Gaussian
    from .fitted import FittedEquation
    from .ml import _newton, _score_hessian
    from .params import point

    if not isinstance(eq.family, Gaussian) or eq.form != "linear":
        raise SpecificationError("Heckman's outcome equation is a linear regression: y ~ x1 + x2")
    if weights:
        raise SpecificationError("Heckman selection does not take weights")
    if not m.selection or "~" not in m.selection:
        raise SpecificationError("give the selection equation: Heckman(selection='inlf ~ educ + kids')")
    if m.estimator not in ("twostep", "ml"):
        raise SpecificationError("estimator='twostep' or 'ml'")
    sel_eq = Eq(m.selection)
    ds = linear_design(sel_eq, data, span, where, None, missing)
    s = ds.y
    if not np.isin(s, (0.0, 1.0)).all():
        raise SpecificationError(f"the selection variable {sel_eq.lhs} must be 0 or 1")
    Zall, z_names = ds.X, list(ds.names)
    gamma, _, _ = _newton(Probit(), s, Zall, np.ones(len(s)), 100, 1e-10)
    _, h = _score_hessian(Probit(), s, Zall @ gamma)
    V_gamma = np.linalg.inv(-(Zall * h[:, None]).T @ Zall)
    do = linear_design(eq, data, span, where, None, "drop")  # rows where the outcome and its regressors are known
    selected = ds.index[s == 1]
    common = selected.intersection(do.index)
    if len(common) < len(selected):
        raise SpecificationError(f"{len(selected) - len(common)} selected observations lack the outcome or a regressor")
    rows, zrows = do.index.get_indexer(common), ds.index.get_indexer(common)
    y, X, x_names = do.y[rows], do.X[rows], list(do.names)
    Z = Zall[zrows]
    zg = Z @ gamma
    lam = np.exp(sps.norm.logpdf(zg) - log_ndtr(zg))  # inverse Mills ratio
    delta = lam * (lam + zg)
    Xs = np.column_stack([X, lam])
    XtX_inv = np.linalg.inv(Xs.T @ Xs)
    b = XtX_inv @ Xs.T @ y
    e = y - Xs @ b
    n_sel = len(y)
    b_lam = b[-1]
    sigma2 = e @ e / n_sel + b_lam**2 * delta.mean()
    rho = float(np.clip(b_lam / np.sqrt(sigma2), -1, 1))
    names = x_names + ["lambda"] + [f"select: {n}" for n in z_names]
    if m.estimator == "twostep":
        F = (Xs * delta[:, None]).T @ Z
        Q = rho**2 * F @ V_gamma @ F.T
        V_b = sigma2 * XtX_inv @ ((Xs * (1 - rho**2 * delta)[:, None]).T @ Xs + Q) @ XtX_inv
        values = np.r_[b, gamma]
        cov = np.zeros((len(values), len(values)))
        cov[: len(b), : len(b)] = V_b
        cov[len(b):, len(b):] = V_gamma
        fitted = X @ b[:-1]
        stats = {"nobs": float(len(s)), "nobs_selected": float(n_sel), "k": float(len(values)), "rho": rho,
                 "sigma": float(np.sqrt(sigma2)), "lambda": float(b_lam)}
        notes = [f"Heckman two-step: probit of {sel_eq.lhs} on all {len(s)} observations, then least squares with the inverse "
                 f"Mills ratio on the {n_sel} selected; rho {rho:.4f}, sigma {np.sqrt(sigma2):.4f} (Heckman's corrected covariance)."]
        extra = {"notes": notes, "converged": True}
        loglike = None
    else:
        sel_obs = s == 1
        y_all = np.zeros(len(s))
        X_all = np.zeros((len(s), X.shape[1]))
        y_all[zrows], X_all[zrows] = y, X
        k, q = X.shape[1], Zall.shape[1]

        def ll(t: np.ndarray) -> np.ndarray:
            beta, g, r, sg = t[:k], t[k:k + q], np.tanh(t[k + q]), np.exp(t[k + q + 1])
            zgam = Zall @ g
            u = (y_all - X_all @ beta) / sg
            on = log_ndtr((zgam + r * u) / np.sqrt(1 - r**2)) + sps.norm.logpdf(u) - np.log(sg)
            return np.where(sel_obs, on, log_ndtr(-zgam))

        start = np.r_[b[:-1], gamma, np.arctanh(np.clip(rho, -0.95, 0.95)), 0.5 * np.log(sigma2)]
        w = np.ones(len(s))
        theta, loglike, iterations = maximise(ll, start, w, m.max_iter, m.tol)
        cov_internal = covariance(m.cov, ll, theta, w, [])

        def report(t: np.ndarray) -> np.ndarray:
            return np.r_[t[:k + q], np.tanh(t[k + q]), np.exp(t[k + q + 1])]

        values = report(theta)
        Jr = jacobian(report, theta)
        cov = Jr @ cov_internal @ Jr.T
        r_hat, s_hat = values[k + q], values[k + q + 1]
        lam_coef = r_hat * s_hat
        names = x_names + [f"select: {n}" for n in z_names] + ["rho", "sigma"]
        fitted = X @ values[:k]
        p = len(values)
        stats = {"nobs": float(len(s)), "nobs_selected": float(n_sel), "k": float(p), "loglike": float(loglike), "rho": float(r_hat),
                 "sigma": float(s_hat), "lambda": float(lam_coef), "aic": criteria.aic(loglike, p),
                 "bic": criteria.bic(loglike, p, len(s)), "hq": criteria.hq(loglike, p, len(s)), "iterations": float(iterations)}
        notes = [f"Heckman by maximum likelihood: {n_sel} of {len(s)} observations selected; lambda = rho sigma = {lam_coef:.4f}."]
        extra = {"notes": notes, "converged": True}
    return FittedEquation(
        equation=eq, params=point(names, values, cov), method=m,
        resid=pd.Series(y - fitted, index=common, name="resid"), fitted=pd.Series(fitted, index=common, name="fitted"),
        stats=stats, provenance=provenance(eq, m, data, common, int(len(ds.index) - len(common))),
        design=pd.DataFrame(X, index=common, columns=x_names), extra=extra)


from ..plugins import register  # noqa: E402 - after the class it registers

register("methods", Heckman)
