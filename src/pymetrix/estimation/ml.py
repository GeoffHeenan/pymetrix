"""Maximum likelihood for linear-index models: Logit, Probit and Poisson (and, in ``estimation.limited``, the
negative binomial, zero-inflated Poisson, ordered, multinomial and Tobit models).

The equation's ``family`` says which likelihood; ``ML`` says how to estimate
and which covariance to report. Estimation is Newton-Raphson with analytic
score and Hessian, and step halving when the log likelihood would fall.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd
from scipy.special import expit, gammaln, log_ndtr

from pymetrix._lazy import stats as sps

from .._planned import planned
from ..errors import SpecificationError
from ..forms import Widget
from ..plugins import register
from . import criteria
from .covariance import Cluster
from .design import cov_groups, linear_design, provenance
from .families import Gaussian, Logit, Poisson, Probit
from .fitted import FittedEquation
from .params import point

if TYPE_CHECKING:
    from ..data import Dataset
    from .equation import Equation


@dataclass(frozen=True)
class ML:
    """Maximum likelihood: Logit, Probit, Poisson, negative binomial, zero-inflated Poisson, ordered, multinomial and
    Tobit families, ARMA errors and GARCH variance.

    The estimates are consistent and efficient when the family's likelihood is the right one. Logit and probit
    coefficients act on a latent index, so their size is not an effect on the probability: use
    ``fit.marginal_effects()`` for that, and compare signs and ratios, not levels, across the two. The Poisson
    coefficients stay consistent for the conditional mean when the counts are over-dispersed, but then only the
    white (sandwich) standard errors are right. The statistics give the log likelihood, McFadden's pseudo
    R-squared (one minus its ratio to the intercept-only model's) and the LR test that every slope is zero.
    Estimates that grow without bound or a singular information matrix usually mean perfect prediction
    (separation) by a regressor. Fixed effects are not supported, since in short panels they bias the other
    coefficients (the incidental-parameters problem). ARMA errors and GARCH variance have their own optimiser, so
    the optimiser settings below do not apply to them, and their standard errors are classic or white. An
    explicit-form equation (``y ~ a + b*exp(c*x)`` as the index) is fitted by quasi-Newton from ``start``; the
    optimiser choice does not apply to it.
    """

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import ML, Dataset, Equation, Logit, Probit

        mroz = Dataset.example("mroz")                           # inlf: 1 if the woman was in the labour force in 1975
        fit = Equation("inlf ~ educ + exper + kidslt6 + nwifeinc", family=Probit()).fit(mroz, method=ML())
        print(fit.summary())                                     # coefficients act on the latent index; pseudo R-squared and the LR test of all slopes
        print(fit.marginal_effects())                            # the change in the probability of working, averaged over the women
        logit = Equation("inlf ~ educ + exper + kidslt6 + nwifeinc", family=Logit()).fit(mroz, method=ML())
        print(logit.marginal_effects())                          # compare marginal effects, not coefficients, across logit and probit
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("coef, std err, z", ("logit and probit coefficients act on the latent index; classic standard errors "
                              "(inverse Hessian) assume the likelihood is right, white does not")),
        ("Pseudo R-squared", ("McFadden's: one minus the log likelihood over the constant-only one; not comparable "
                              "with a least-squares R-squared")),
        ("Log likelihood (constant only)", ("the model with only a constant: the base for the pseudo R-squared and "
                                            "the LR test")),
        ("LR statistic", ("twice the log-likelihood gain over the constant-only model, chi-squared with as many df as "
                          "slopes; fit.stats['lr_pvalue'] tests that all slopes are zero")),
        ("fit.marginal_effects()", ("average effects on the probability, with delta-method standard errors; for a 0/1 "
                                    "regressor the change from 0 to 1")),
        ("fit.stats", "loglike, ll_null, pseudo_r2, lr, lr_pvalue, iterations, aic, bic and hq"),
    )

    label: ClassVar[str] = "ML"
    topics: ClassVar[tuple[str, ...]] = ('maximum-likelihood',)
    references: ClassVar[tuple[str, ...]] = (
        "Greene, W. H. (2018). Econometric Analysis, 8th ed., chapter 14, \"Maximum likelihood estimation\". Pearson.",
        ("McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. In Zarembka, P. (ed.), Frontiers in Econometrics. Academic Press."),
    )
    kind: ClassVar[str] = "ml"

    cov: Annotated[str | Cluster, Widget("", help="classic: inverse Hessian; white: sandwich, robust to a wrong variance; "
                                         "Cluster(...); opg (logit, probit, Poisson)")] = "classic"
    optimizer: Annotated[Literal["newton", "bhhh"], Widget("choice", advanced=True,
                         help="newton: the Hessian; bhhh: the outer product of the scores (Berndt, Hall, Hall and "
                              "Hausman); linear logit, probit and Poisson only")] = "newton"
    max_iter: Annotated[int, Widget("", min=1, max=10000, advanced=True,
                                    help="iterations of the likelihood search; raise it if a fit fails to converge")] = 100
    tol: Annotated[float, Widget("", min=1e-14, max=1e-2, advanced=True,
                                 help="relative change in the estimates and log likelihood at which the search stops "
                                      "(not explicit-form equations)")] = 1e-10
    start: Annotated[Mapping[str, float] | None, Widget("", advanced=True,
                     help="starting values for explicit-form equations, e.g. b0=0.5, b1=0.1")] = None

    def estimate(self, eq: Equation, data: Dataset, *, span=None, where=None, weights=None, missing="drop") -> FittedEquation:
        if eq.has_error_model:
            from . import timeseries

            if isinstance(self.cov, Cluster):
                raise SpecificationError("ARMA and GARCH fits support cov='classic' or 'white'")
            return timeseries.estimate(self, eq, data, span, where, weights, missing)
        family = eq.family
        if isinstance(family, Gaussian):
            raise SpecificationError("for a Gaussian equation use OLS, which gives the maximum-likelihood coefficients")
        from . import limited

        if isinstance(family, limited.LIMITED):
            return limited.estimate(self, eq, data, span, where, weights, missing)
        if not isinstance(family, (Logit, Probit, Poisson)):
            planned(f"ML for {type(family).__name__}", 5)
        if self.cov == "hac":
            raise SpecificationError("ML supports cov='classic', 'white', 'opg' or Cluster(...)")
        if eq.form != "linear":
            return _ml_explicit(self, eq, data, span, where, weights, missing)
        if eq.fixed_effects:
            raise NotImplementedError("fixed effects in nonlinear ML models are not supported (incidental parameters)")
        if self.optimizer not in ("newton", "bhhh"):
            raise SpecificationError("optimizer is 'newton' or 'bhhh'")
        d = linear_design(eq, data, span, where, weights, missing)
        y, X = d.y, d.X
        _check_outcome(family, y)
        w = d.w if d.w is not None else np.ones(len(y))
        beta, ll, iterations = _newton(family, y, X, w, self.max_iter, self.tol, bhhh=self.optimizer == "bhhh")
        eta = X @ beta
        s, h = _score_hessian(family, y, eta)
        H = (X * (w * h)[:, None]).T @ X
        inv_info = np.linalg.inv(-H)
        cov = _ml_covariance(self.cov, X, w * s, inv_info, cov_groups(self.cov, data, d.index))
        mu = family.mean(eta)
        n, k = X.shape
        ll0 = _null_loglike(family, y, w, eq.intercept)
        lr = 2 * (ll - ll0)
        stats = {
            "nobs": float(n),
            "k": float(k),
            "loglike": float(ll),
            "ll_null": float(ll0),
            "pseudo_r2": float(1 - ll / ll0),
            "lr": float(lr),
            "lr_pvalue": float(sps.chi2.sf(lr, k - int(eq.intercept))),
            "aic": criteria.aic(ll, k),
            "bic": criteria.bic(ll, k, n),
            "hq": criteria.hq(ll, k, n),
            "iterations": float(iterations),
        }
        return FittedEquation(
            equation=eq,
            params=point(d.names, beta, cov),
            method=self,
            resid=pd.Series(y - mu, index=d.index, name="resid"),
            fitted=pd.Series(mu, index=d.index, name="fitted"),
            stats=stats,
            provenance=provenance(eq, self, data, d.index, d.dropped),
            design=pd.DataFrame(X, index=d.index, columns=d.names),
            weights=weights,
            extra={"converged": True},  # Newton's method raises if it does not converge
        )


def _ml_explicit(m: ML, eq: Equation, data: Dataset, span: Any, where: Any, weights: Any, missing: str) -> FittedEquation:
    """An explicit-form equation's right-hand side as the family's index, eta(x; theta): ``y ~ a + b*exp(c*x)`` with
    ``family=Logit()`` (or Probit, Poisson). Quasi-Newton on the log likelihood with its analytic score, from ``start``
    (or each parameter's start, else 0.1), within any bounds; the classic covariance is the inverse of the exact
    Hessian, which includes the index's second derivatives; white, OPG and clustered ones use the scores."""
    from scipy.optimize import minimize

    from ..expr import Coef, diff, evaluate
    from .design import explicit_design
    from .methods import _start

    d = explicit_design(eq, data, span, where, weights, missing)
    names = list(eq.param_names)
    family = eq.family
    y = np.broadcast_to(np.asarray(evaluate(eq.lhs, d.resolve), dtype=float), (d.n,)).copy()
    _check_outcome(family, y)
    w = d.w if d.w is not None else np.ones(d.n)
    grads = [diff(eq.rhs, Coef(p)) for p in names]
    second = [[diff(g, Coef(q)) for q in names] for g in grads]

    def column(expr: Any, theta: np.ndarray) -> np.ndarray:
        return np.broadcast_to(np.asarray(evaluate(expr, d.resolve, dict(zip(names, theta))), dtype=float), (d.n,))

    def negloglike(theta: np.ndarray) -> float:
        value = -float(np.sum(w * _loglike(family, y, column(eq.rhs, theta))))
        return value if np.isfinite(value) else 1e300

    def negscore(theta: np.ndarray) -> np.ndarray:
        s, _ = _score_hessian(family, y, column(eq.rhs, theta))
        return -np.array([np.sum(w * s * column(g, theta)) for g in grads])

    theta0 = _start(eq, m.start)
    if not np.isfinite(negloglike(theta0)) or negloglike(theta0) >= 1e300:
        raise SpecificationError("the likelihood cannot be evaluated at the starting values; set start={...} or Param(start=...)")
    bounds = [p.bounds for p in eq.param_specs]
    bounded = any(np.isfinite(b).any() for b in bounds)
    result = minimize(negloglike, theta0, jac=negscore, method="L-BFGS-B" if bounded else "BFGS",
                      bounds=bounds if bounded else None, options={"maxiter": 20 * m.max_iter, "gtol": 1e-10})
    theta = result.x
    eta = column(eq.rhs, theta)
    s, h = _score_hessian(family, y, eta)
    G = np.column_stack([column(g, theta) for g in grads])  # d eta / d theta
    H = (G * (w * h)[:, None]).T @ G + np.array([[np.sum(w * s * column(second[i][j], theta)) for j in range(len(names))]
                                                  for i in range(len(names))])
    inv_info = np.linalg.inv(-H)
    cov = _ml_covariance(m.cov, G, w * s, inv_info, cov_groups(m.cov, data, d.index))
    ll = -negloglike(theta)
    n, k = d.n, len(names)
    mu = family.mean(eta)
    ll0 = _null_loglike(family, y, w, True)
    lr = 2 * (ll - ll0)
    stats = {"nobs": float(n), "k": float(k), "loglike": float(ll), "ll_null": float(ll0), "pseudo_r2": float(1 - ll / ll0),
             "lr": float(lr), "lr_pvalue": float(sps.chi2.sf(lr, k - 1)) if k > 1 else float("nan"),
             "aic": criteria.aic(ll, k), "bic": criteria.bic(ll, k, n), "hq": criteria.hq(ll, k, n), "iterations": float(result.nit)}
    gradient = float(np.max(np.abs(negscore(theta)))) / max(1.0, abs(ll))
    return FittedEquation(
        equation=eq,
        params=point(names, theta, cov),
        method=m,
        resid=pd.Series(y - mu, index=d.index, name="resid"),
        fitted=pd.Series(mu, index=d.index, name="fitted"),
        stats=stats,
        provenance=provenance(eq, m, data, d.index, d.dropped),
        design=pd.DataFrame(G, index=d.index, columns=names),
        weights=weights,
        extra={"converged": bool(result.success or gradient < 1e-6)},
    )


def _check_outcome(family: Any, y: np.ndarray) -> None:
    if isinstance(family, (Logit, Probit)) and not np.isin(y, (0.0, 1.0)).all():
        raise SpecificationError(f"{type(family).__name__} needs a 0/1 dependent variable")
    if isinstance(family, Poisson) and (np.any(y < 0) or not np.allclose(y, np.round(y))):
        raise SpecificationError("Poisson needs a non-negative count dependent variable")


def _loglike(family: Any, y: np.ndarray, eta: np.ndarray) -> np.ndarray:
    if isinstance(family, Logit):
        return y * eta - np.logaddexp(0, eta)
    if isinstance(family, Probit):
        return log_ndtr((2 * y - 1) * eta)
    return y * eta - np.exp(eta) - gammaln(y + 1)


def _score_hessian(family: Any, y: np.ndarray, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and second derivatives of each observation's log likelihood with respect to eta."""
    if isinstance(family, Logit):
        mu = expit(eta)
        return y - mu, -mu * (1 - mu)
    if isinstance(family, Probit):
        q = 2 * y - 1
        a = q * eta
        r = np.exp(sps.norm.logpdf(a) - log_ndtr(a))  # phi(a) / Phi(a), stable in the tails
        return q * r, -r * (a + r)
    mu = np.exp(eta)
    return y - mu, -mu


def _newton(family: Any, y: np.ndarray, X: np.ndarray, w: np.ndarray, max_iter: int, tol: float,
            bhhh: bool = False) -> tuple[np.ndarray, float, int]:
    """Newton's method with a line search; with ``bhhh``, minus the outer product of the scores in place of the Hessian
    (Berndt, Hall, Hall and Hausman 1974): always negative definite, but slower near the maximum."""
    beta = np.zeros(X.shape[1])
    ll = float(np.sum(w * _loglike(family, y, X @ beta)))
    for it in range(1, max_iter + 1):
        s, h = _score_hessian(family, y, X @ beta)
        g = X.T @ (w * s)
        if bhhh:
            G = X * (w * s)[:, None]
            H = -(G.T @ G)
        else:
            H = (X * (w * h)[:, None]).T @ X
        try:
            step = np.linalg.solve(-H, g)
        except np.linalg.LinAlgError:
            raise SpecificationError("the information matrix is singular (perfect prediction or collinear regressors?)") from None
        t = 1.0
        for _ in range(40):
            new = beta + t * step
            new_ll = float(np.sum(w * _loglike(family, y, X @ new)))
            if np.isfinite(new_ll) and new_ll >= ll - 1e-12 * abs(ll):
                break
            t /= 2
        else:
            raise RuntimeError("ML line search failed")
        beta, change = new, new_ll - ll
        ll = new_ll
        if np.max(np.abs(t * step)) < tol * (1 + np.max(np.abs(beta))) and abs(change) < tol * (1 + abs(ll)):
            return beta, ll, it
    raise RuntimeError(f"ML did not converge in {max_iter} iterations")


def _ml_covariance(kind: Any, X: np.ndarray, scores: np.ndarray, inv_info: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    if kind == "classic":
        return inv_info
    if kind == "white":
        S = X * scores[:, None]
        return inv_info @ (S.T @ S) @ inv_info
    if kind == "opg":  # the outer product of the gradients: the BHHH estimate of the information
        S = X * scores[:, None]
        return np.linalg.inv(S.T @ S)
    if isinstance(kind, Cluster):
        if len(groups) != 1:
            raise NotImplementedError("two-way clustering for ML models")
        codes = pd.factorize(groups[0])[0]
        G = codes.max() + 1
        sums = np.zeros((G, X.shape[1]))
        np.add.at(sums, codes, X * scores[:, None])
        return G / (G - 1) * inv_info @ (sums.T @ sums) @ inv_info
    raise SpecificationError(f"unknown covariance {kind!r}")


def _null_loglike(family: Any, y: np.ndarray, w: np.ndarray, intercept: bool) -> float:
    """Log likelihood of the intercept-only model (or eta = 0 without an intercept)."""
    if not intercept:
        return float(np.sum(w * _loglike(family, y, np.zeros_like(y))))
    ybar = np.sum(w * y) / np.sum(w)
    if isinstance(family, Logit):
        eta0 = np.log(ybar / (1 - ybar))
    elif isinstance(family, Probit):
        eta0 = sps.norm.ppf(ybar)
    else:
        eta0 = np.log(ybar)
    return float(np.sum(w * _loglike(family, y, np.full_like(y, eta0))))


register("methods", ML)
