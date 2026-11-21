"""Regression extras: AR(1) errors by Cochrane-Orcutt and Prais-Winsten, multiplicative heteroskedasticity
(Harvey 1976), the normal-half-normal stochastic frontier (Aigner, Lovell and Schmidt 1977), and the Davidson-MacKinnon
J and Cox tests of non-nested models.

    Equation("c ~ y").fit(data, method=CochraneOrcutt())          # iterated: rho from the residuals, until it settles
    Equation("c ~ y").fit(data, method=PraisWinsten())            # keeps the first observation, scaled by sqrt(1 - rho^2)
    Equation("y ~ x").fit(data, method=MultiplicativeHeteroskedasticity(variance=["z"]))   # var u = exp(g0 + g z)
    Equation("lq ~ lk + ll").fit(data, method=StochasticFrontier())                       # y = x'b + v - u, u half-normal
    j_test(Equation("c ~ y"), Equation("c ~ c[-1]"), data)       # is the first model right, given the second's fit?

Exact maximum likelihood with AR(1) errors is ``Equation(..., errors=ARMA(1, 0))`` with ML.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd
from scipy import optimize

from pymetrix._lazy import stats as sps

from .._api import api
from ..errors import SpecificationError
from ..forms import Widget
from ..plugins import register

if TYPE_CHECKING:
    from ..data import Dataset
    from .equation import Equation
    from .fitted import FittedEquation

__all__ = ["CochraneOrcutt", "MultiplicativeHeteroskedasticity", "PraisWinsten", "StochasticFrontier", "cox_test", "j_test"]


def _linear(eq: Equation, data: Dataset, span: Any, where: Any, weights: Any, missing: str, what: str) -> Any:
    from .design import linear_design

    if weights:
        raise SpecificationError(f"{what} does not take weights")
    if eq.form != "linear" or eq.fixed_effects or eq.categoricals:
        raise SpecificationError(f"{what} needs a linear-form equation without fe() or C()")
    return linear_design(eq, data, span, where, None, missing)


def _fitted(eq: Equation, method: Any, data: Dataset, d: Any, beta: np.ndarray, cov: np.ndarray, u: np.ndarray, extra_stats: dict,
            notes: list[str], df_resid: int) -> FittedEquation:
    from .design import ls_stats, provenance
    from .fitted import FittedEquation
    from .params import point

    n, k = d.X.shape
    stats = ls_stats(d.y, u, None, k, df_resid, eq.intercept, data.kind == "time_series", False)
    stats.update(extra_stats)
    return FittedEquation(eq, point(d.names, beta, cov), method, resid=pd.Series(u, index=d.index, name="resid"),
                          fitted=pd.Series(d.y - u, index=d.index, name="fitted"), stats=stats,
                          provenance=provenance(eq, method, data, d.index, d.dropped),
                          design=pd.DataFrame(d.X, index=d.index, columns=d.names), extra={"notes": notes})


# AR(1) errors


def _ar1(m: Any, eq: Equation, data: Dataset, span: Any, where: Any, weights: Any, missing: str, keep_first: bool) -> FittedEquation:
    what = "Prais-Winsten" if keep_first else "Cochrane-Orcutt"
    d = _linear(eq, data, span, where, weights, missing, what)
    if data.kind != "time_series":
        raise SpecificationError(f"{what} needs a single time series")
    y, X = d.y, d.X
    n, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rho, iterations = 0.0, 0
    for iterations in range(1, m.max_iter + 1):
        u = y - X @ beta
        new = float(u[1:] @ u[:-1] / (u[:-1] @ u[:-1]))
        if abs(new) >= 1:
            raise SpecificationError(f"rho reached {new:.3f}: the errors look nonstationary; difference the series")
        ys, Xs = _transform(y, X, new, keep_first)
        beta = np.linalg.lstsq(Xs, ys, rcond=None)[0]
        done = abs(new - rho) < m.tol
        rho = new
        if done or not m.iterate:
            break
    else:
        raise SpecificationError(f"{what} did not converge in {m.max_iter} iterations")
    ys, Xs = _transform(y, X, rho, keep_first)
    e = ys - Xs @ beta
    df = len(ys) - k
    s2 = e @ e / df
    cov = s2 * np.linalg.inv(Xs.T @ Xs)
    u = y - X @ beta
    notes = [f"{what}: AR(1) errors, rho = {rho:.4f} ({'iterated, ' + str(iterations) + ' rounds' if m.iterate else 'one round'}); "
             f"{'first observation kept, scaled by sqrt(1 - rho^2)' if keep_first else 'first observation dropped'} "
             f"(the transformed regression has {len(ys)} observations, {df} degrees of freedom); "
             "standard errors do not allow for rho being estimated."]
    return _fitted(eq, m, data, d, beta, cov, u, {"rho": rho, "iterations": float(iterations), "sigma2_transformed": float(s2)}, notes, df)


def _transform(y: np.ndarray, X: np.ndarray, rho: float, keep_first: bool) -> tuple[np.ndarray, np.ndarray]:
    ys, Xs = y[1:] - rho * y[:-1], X[1:] - rho * X[:-1]
    if keep_first:
        c = np.sqrt(1 - rho**2)
        ys, Xs = np.r_[c * y[0], ys], np.vstack([c * X[0], Xs])
    return ys, Xs


@dataclass(frozen=True)
class CochraneOrcutt:
    """Least squares with AR(1) errors by Cochrane and Orcutt: rho from the residuals, the equation quasi-differenced
    (the first observation dropped), repeated until rho settles.

    Feasible generalised least squares for errors u_t = rho u_t-1 + e_t. It is efficient, with valid standard
    errors, only if the regressors are strictly exogenous; with a lagged dependent variable among them rho and the
    coefficients are inconsistent (use ML with ARMA errors, or instruments). The standard errors treat rho as
    known. Autocorrelated residuals often signal missing dynamics rather than AR(1) errors, so adding lags may be
    the better model; least squares with hac standard errors keeps the OLS coefficients and corrects only the
    inference. Dropping the first observation costs efficiency in short samples; Prais-Winsten keeps it. The
    fit stops with an error if rho reaches 1 in absolute value: difference the series.
    """

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import CochraneOrcutt, Dataset, Equation, PraisWinsten

        data = Dataset.example("sample_macro")
        fit = Equation("c ~ y").fit(data, method=CochraneOrcutt())
        print(fit.summary())                                     # the notes give rho, the errors' AR(1) coefficient, and the rounds taken
        print(Equation("c ~ y").fit(data, method=PraisWinsten()).summary())  # the same model, keeping the first observation
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("coef, std err, t", ("feasible GLS estimates; the standard errors treat rho as known and need strictly "
                              "exogenous regressors")),
        ("rho", ("in the note: the errors' AR(1) coefficient and the rounds taken; fit.stats['rho'] and "
                 "['iterations'] hold them")),
        ("R-squared, S.E. of regression, Durbin-Watson", ("from the untransformed residuals u = y - Xb, so Durbin-"
                                                          "Watson still shows the AR(1) the model allows for")),
        ("fit.stats['sigma2_transformed']", ("the variance of the AR(1) innovations, from the quasi-differenced "
                                             "regression, which scales the standard errors")),
        ("fit.resid", ("u, the untransformed residuals, which keep the AR(1): residual tests on them find the "
                       "autocorrelation the model already allows for")),
    )

    label: ClassVar[str] = "Cochrane-Orcutt (AR(1) errors)"
    kind: ClassVar[str] = "ls"
    topics: ClassVar[tuple[str, ...]] = ("least-squares",)
    references: ClassVar[tuple[str, ...]] = (
        ("Cochrane, D. and Orcutt, G. H. (1949). Application of least squares regression to relationships containing auto-correlated error terms. Journal of the American Statistical Association 44(245), 32-61. doi:10.1080/01621459.1949.10483290"),
    )

    iterate: Annotated[bool, Widget("checkbox", help="repeat until rho settles (otherwise one round)")] = True
    max_iter: Annotated[int, Widget("number", min=1, max=1000, advanced=True, help="most rounds when iterating; if rho has not settled by then, the fit stops with an error")] = 200
    tol: Annotated[float, Widget("number", min=1e-14, max=1e-2, advanced=True, help="convergence tolerance for rho")] = 1e-10

    def estimate(self, eq: Equation, data: Dataset, *, span: Any = None, where: Any = None, weights: Any = None,
                 missing: str = "drop") -> FittedEquation:
        return _ar1(self, eq, data, span, where, weights, missing, keep_first=False)


@dataclass(frozen=True)
class PraisWinsten:
    """Least squares with AR(1) errors by Prais and Winsten: as Cochrane-Orcutt, but the first observation kept,
    scaled by sqrt(1 - rho^2), so that it is generalised least squares for the AR(1) covariance.

    Feasible generalised least squares for errors u_t = rho u_t-1 + e_t. It is efficient, with valid standard
    errors, only if the regressors are strictly exogenous; with a lagged dependent variable among them rho and the
    coefficients are inconsistent (use ML with ARMA errors, or instruments). The standard errors treat rho as
    known. Autocorrelated residuals often signal missing dynamics rather than AR(1) errors, so adding lags may be
    the better model; least squares with hac standard errors keeps the OLS coefficients and corrects only the
    inference. Keeping the first observation matters most in short samples and with trending regressors. The
    fit stops with an error if rho reaches 1 in absolute value: difference the series.
    """

    maturity: ClassVar[str] = "stable"
    label: ClassVar[str] = "Prais-Winsten (AR(1) errors)"
    kind: ClassVar[str] = "ls"
    topics: ClassVar[tuple[str, ...]] = ("least-squares",)
    references: ClassVar[tuple[str, ...]] = (
        ("Prais, S. J. and Winsten, C. B. (1954). Trend estimators and serial correlation. Cowles Commission Discussion "
         "Paper 383."),
    )

    iterate: Annotated[bool, Widget("checkbox", help="repeat until rho settles (otherwise one round)")] = True
    max_iter: Annotated[int, Widget("number", min=1, max=1000, advanced=True, help="most rounds when iterating; if rho has not settled by then, the fit stops with an error")] = 200
    tol: Annotated[float, Widget("number", min=1e-14, max=1e-2, advanced=True, help="convergence tolerance for rho")] = 1e-10

    def estimate(self, eq: Equation, data: Dataset, *, span: Any = None, where: Any = None, weights: Any = None,
                 missing: str = "drop") -> FittedEquation:
        return _ar1(self, eq, data, span, where, weights, missing, keep_first=True)


# multiplicative heteroskedasticity


@dataclass(frozen=True)
class MultiplicativeHeteroskedasticity:
    """Harvey's (1976) multiplicative heteroskedasticity by maximum likelihood.

    var(u_i) = exp(g0 + z_i'g): the coefficients by weighted least squares given the variance and the variance by
    scoring, alternately. The LR test of g = 0 (homoskedasticity) is in the statistics.

    Heteroskedasticity does not bias least squares but makes it inefficient and its classic standard errors wrong.
    Modelling the variance gives efficient weighted estimates if the variance function is right; if it is wrong
    the coefficients stay consistent but their standard errors, which assume the model, mislead, and least squares
    with white standard errors is the safer choice. The exponential form keeps the variance positive: a variance
    coefficient is the proportional change in the variance per unit of its regressor. The variance coefficients
    and their standard errors are in the notes.
    """

    maturity: ClassVar[str] = "beta"
    label: ClassVar[str] = "Multiplicative heteroskedasticity (Harvey)"
    kind: ClassVar[str] = "ls"
    topics: ClassVar[tuple[str, ...]] = ("least-squares", "maximum-likelihood")
    references: ClassVar[tuple[str, ...]] = (
        ("Harvey, A. C. (1976). Estimating regression models with multiplicative heteroscedasticity. Econometrica 44(3), 461-465. doi:10.2307/1913974"),
    )

    variance: Annotated[Sequence[str], Widget("text", help="the variance's regressors (expressions), e.g. x, log(size)")] = ()
    max_iter: Annotated[int, Widget("number", min=1, max=10_000, advanced=True, help="most rounds of weighted least squares and scoring before the fit stops with an error")] = 500
    tol: Annotated[float, Widget("number", min=1e-14, max=1e-2, advanced=True,
                                 help="stop when no variance coefficient changes by more than this in a round")] = 1e-10

    def estimate(self, eq: Equation, data: Dataset, *, span: Any = None, where: Any = None, weights: Any = None,
                 missing: str = "drop") -> FittedEquation:
        from .design import instrument_series, linear_design

        if not self.variance:
            raise SpecificationError("name the variance's regressors: variance=['z']")
        if weights or eq.form != "linear" or eq.fixed_effects or eq.categoricals:
            raise SpecificationError("multiplicative heteroskedasticity needs a linear-form equation without fe(), C() or weights")
        series = instrument_series(list(self.variance), data)
        d = linear_design(eq, data, span, where, None, missing, instruments=series)
        y, X = d.y, d.X
        Z = np.column_stack([np.ones(len(y)), d.Z])
        n, k = X.shape
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        u = y - X @ beta
        gamma = np.linalg.lstsq(Z, np.log(u**2 + 1e-300), rcond=None)[0]
        gamma[0] += 1.2704  # E log chi-squared(1) = -1.2704: Harvey's consistent start

        def loglike(b: np.ndarray, g: np.ndarray) -> float:
            h = Z @ g
            r = y - X @ b
            return float(-0.5 * np.sum(np.log(2 * np.pi) + h + r**2 * np.exp(-h)))

        for it in range(1, self.max_iter + 1):
            w = np.exp(-Z @ gamma)
            beta = np.linalg.solve((X * w[:, None]).T @ X, (X * w[:, None]).T @ y)
            r = y - X @ beta
            step = np.linalg.solve(Z.T @ Z, Z.T @ (r**2 * w - 1))  # scoring: the information is Z'Z / 2
            gamma = gamma + step
            if np.max(np.abs(step)) < self.tol:
                break
        else:
            raise SpecificationError(f"multiplicative heteroskedasticity did not converge in {self.max_iter} iterations")
        w = np.exp(-Z @ gamma)
        cov_b = np.linalg.inv((X * w[:, None]).T @ X)
        cov_g = 2 * np.linalg.inv(Z.T @ Z)
        r = y - X @ beta
        ll = loglike(beta, gamma)
        b_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        e0 = y - X @ b_ols
        ll0 = float(-n / 2 * (np.log(2 * np.pi) + np.log(e0 @ e0 / n) + 1))  # homoskedastic ML: OLS
        LR = 2 * (ll - ll0)
        q = Z.shape[1] - 1
        names = ["const", *d.z_names]
        notes = ["Multiplicative heteroskedasticity by maximum likelihood: log var(u) = "
                 + " + ".join(f"{g:.4f}{'' if nm == 'const' else ' ' + nm}" for g, nm in zip(gamma, names)),
                 "Variance coefficients' standard errors: " + ", ".join(f"{nm} {s:.4f}" for nm, s in zip(names, np.sqrt(np.diag(cov_g)))),
                 f"LR test of homoskedasticity: {LR:.4f}, chi-squared({q}), p = {sps.chi2.sf(LR, q):.4f}"]
        stats = {"loglike": ll, "lr_homoskedastic": float(LR), "lr_homoskedastic_df": float(q),
                 "lr_homoskedastic_pvalue": float(sps.chi2.sf(LR, q)), "iterations": float(it),
                 **{f"variance.{nm}": float(g) for nm, g in zip(names, gamma)},
                 **{f"variance.{nm}_se": float(s) for nm, s in zip(names, np.sqrt(np.diag(cov_g)))}}
        return _fitted(eq, self, data, d, beta, cov_b, r, stats, notes, n - k)


# stochastic frontier


@dataclass(frozen=True)
class StochasticFrontier:
    """The normal-half-normal stochastic frontier by maximum likelihood (Aigner, Lovell and Schmidt 1977).

    y = x'b + v - u for a production frontier (+ u for a cost frontier), v normal and u >= 0 half-normal inefficiency. Reports
    sigma_v, sigma_u, lambda = sigma_u / sigma_v and the mean technical efficiency E[exp(-u)]; each observation's
    inefficiency E[u | e] (Jondrow, Lovell, Materov and Schmidt 1982) is in ``fit.extra["inefficiency"]``.

    Inefficiency is told apart from noise only by the assumed distributions: it shows up as skewness of the
    least-squares residuals, negative for a production frontier and positive for a cost one. Skewed the wrong
    way, sigma_u goes to zero and the frontier is least squares. The LR test of no inefficiency is one-sided (half
    a chi-squared(1)); a small lambda says that deviations from the frontier are mostly noise. Each observation's
    efficiency is a conditional mean, not a consistent estimate of that unit's own, so rank units with care; and
    the results can change with the distribution chosen for u.
    """

    maturity: ClassVar[str] = "beta"
    label: ClassVar[str] = "Stochastic frontier (half-normal)"
    kind: ClassVar[str] = "ls"
    topics: ClassVar[tuple[str, ...]] = ("maximum-likelihood",)
    references: ClassVar[tuple[str, ...]] = (
        ("Aigner, D., Lovell, C. A. K. and Schmidt, P. (1977). Formulation and estimation of stochastic frontier production function models. Journal of Econometrics 6(1), 21-37. doi:10.1016/0304-4076(77)90052-5"),
        ("Jondrow, J., Lovell, C. A. K., Materov, I. S. and Schmidt, P. (1982). On the estimation of technical inefficiency in the stochastic frontier production function model. Journal of Econometrics 19(2-3), 233-238. doi:10.1016/0304-4076(82)90004-5"),
    )

    cost: Annotated[bool, Widget("checkbox", help="a cost frontier (y = x'b + v + u) rather than a production one")] = False

    def estimate(self, eq: Equation, data: Dataset, *, span: Any = None, where: Any = None, weights: Any = None,
                 missing: str = "drop") -> FittedEquation:
        d = _linear(eq, data, span, where, weights, missing, "a stochastic frontier")
        if not eq.intercept:
            raise SpecificationError("a stochastic frontier needs a constant")
        y, X = d.y, d.X
        n, k = X.shape
        s = -1.0 if self.cost else 1.0  # e = y - x'b = v - s u
        b_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        e = y - X @ b_ols
        m2, m3 = np.mean(e**2), np.mean(e**3)
        skew_ok = s * m3 < 0
        su = (s * -m3 / (np.sqrt(2 / np.pi) * (4 / np.pi - 1))) ** (1 / 3) if skew_ok else 0.1 * np.sqrt(m2)
        sv2 = max(m2 - (1 - 2 / np.pi) * su**2, 0.05 * m2)
        b0 = b_ols.copy()
        b0[list(d.names).index("const")] += s * su * np.sqrt(2 / np.pi)  # corrected OLS: the frontier's intercept

        def negll(theta: np.ndarray) -> float:
            b, lsig, llam = theta[:k], theta[k], theta[k + 1]
            sig, lam = np.exp(lsig), np.exp(llam)
            r = y - X @ b
            return -float(np.sum(np.log(2) - np.log(sig) + sps.norm.logpdf(r / sig) + sps.norm.logcdf(-s * r * lam / sig)))

        theta0 = np.r_[b0, 0.5 * np.log(sv2 + su**2), np.log(su / np.sqrt(sv2))]
        found = optimize.minimize(negll, theta0, method="BFGS", options={"gtol": 1e-9, "maxiter": 10_000})
        polish = optimize.minimize(negll, found.x, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 50_000})
        theta = polish.x if polish.fun <= found.fun else found.x
        H = _hessian(negll, theta)
        cov_theta = np.linalg.pinv(H)
        b = theta[:k]
        sig, lam = np.exp(theta[k]), np.exp(theta[k + 1])
        su, sv = sig * lam / np.sqrt(1 + lam**2), sig / np.sqrt(1 + lam**2)
        r = y - X @ b
        # JLMS: u | e is N(mu*, s*^2) truncated at zero, mu* = -s e su^2 / sig^2, s* = su sv / sig
        mu_star, s_star = -s * r * su**2 / sig**2, su * sv / sig
        z = mu_star / s_star
        inefficiency = mu_star + s_star * np.exp(sps.norm.logpdf(z) - sps.norm.logcdf(z))
        efficiency = np.exp(-mu_star + 0.5 * s_star**2) * np.exp(sps.norm.logcdf(z - s_star) - sps.norm.logcdf(z))  # E[exp(-u) | e]
        ll = -negll(theta)
        ll_ols = float(-n / 2 * (np.log(2 * np.pi) + np.log(e @ e / n) + 1))
        LR = max(2 * (ll - ll_ols), 0.0)  # sigma_u = 0 is on the boundary: half of chi-squared(1)
        notes = [f"Stochastic {'cost' if self.cost else 'production'} frontier, normal-half-normal, by maximum likelihood.",
                 f"sigma_v = {sv:.4f}, sigma_u = {su:.4f}, lambda = {lam:.4f}; mean efficiency E[exp(-u)|e] = {efficiency.mean():.4f}",
                 f"LR test of no inefficiency (sigma_u = 0): {LR:.4f}, p = {0.5 * sps.chi2.sf(LR, 1):.4f} (half chi-squared(1))"]
        if not skew_ok:
            notes.append("The OLS residuals are skewed the wrong way for a frontier: expect sigma_u near zero.")
        stats = {"loglike": ll, "sigma_v": float(sv), "sigma_u": float(su), "lambda": float(lam), "sigma": float(sig),
                 "lambda_se": float(lam * np.sqrt(cov_theta[k + 1, k + 1])), "mean_efficiency": float(efficiency.mean()),
                 "lr_no_inefficiency": float(LR), "lr_no_inefficiency_pvalue": float(0.5 * sps.chi2.sf(LR, 1))}
        fit = _fitted(eq, self, data, d, b, cov_theta[:k, :k], r, stats, notes, n - k)
        fit.extra["inefficiency"] = pd.Series(inefficiency, index=d.index, name="inefficiency")
        fit.extra["efficiency"] = pd.Series(efficiency, index=d.index, name="efficiency")
        return fit


def _hessian(f: Any, x: np.ndarray) -> np.ndarray:
    n = len(x)
    H = np.empty((n, n))
    h = 1e-4 * np.maximum(np.abs(x), 1e-2)
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i], ej[j] = h[i], h[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


for _cls in (CochraneOrcutt, PraisWinsten, MultiplicativeHeteroskedasticity, StochasticFrontier):
    register("methods", _cls)


# non-nested tests


def _common(eq1: Equation, eq2: Equation, data: Dataset, span: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from .design import linear_design

    d1 = linear_design(eq1, data, span, None, None, "drop")
    d2 = linear_design(eq2, data, span, None, None, "drop")
    index = d1.index.intersection(d2.index)
    if not len(index):
        raise SpecificationError("the two models share no observations")
    i1, i2 = d1.index.get_indexer(index), d2.index.get_indexer(index)
    if not np.allclose(d1.y[i1], d2.y[i2], equal_nan=True):
        raise SpecificationError("non-nested tests compare models of the same dependent variable")
    return d1.y[i1], d1.X[i1], d2.X[i2]


@api("pymetrix:j_test")
def j_test(eq1: Equation, eq2: Equation, data: Dataset, span: Any = None) -> Any:
    """Davidson and MacKinnon's (1981) J test that the first model is right: the t statistic on the second model's
    fitted values added to the first. Run it both ways; both, or neither, may reject."""
    from ..stats.base import TestResult

    y, X1, X2 = _common(eq1, eq2, data, span)
    f2 = X2 @ np.linalg.lstsq(X2, y, rcond=None)[0]
    X = np.column_stack([X1, f2])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    df = len(y) - X.shape[1]
    t = b[-1] / np.sqrt(e @ e / df * np.linalg.inv(X.T @ X)[-1, -1])
    return TestResult("J test (Davidson-MacKinnon)", f"{eq1.text} is right (against {eq2.text})", float(t), df,
                      float(2 * sps.t.sf(abs(t), df)), "t", details={"observations": len(y)})


@api("pymetrix:cox_test")
def cox_test(eq1: Equation, eq2: Equation, data: Dataset, span: Any = None) -> Any:
    """The Cox (1961) test, in Pesaran's (1974) form for linear regressions, that the first model is right against
    the second: a standard normal statistic, large negative values rejecting."""
    from ..stats.base import TestResult

    y, X1, X2 = _common(eq1, eq2, data, span)
    n = len(y)

    def fit(X: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b = np.linalg.lstsq(X, target, rcond=None)[0]
        return X @ b, target - X @ b

    f1, e1 = fit(X1, y)
    _, e2 = fit(X2, y)
    s1, s2 = e1 @ e1 / n, e2 @ e2 / n
    _, e21 = fit(X2, f1)  # model 1's fitted values regressed on model 2's regressors
    s21 = s1 + e21 @ e21 / n
    c01 = n / 2 * np.log(s2 / s21)
    _, e_aux = fit(X1, e21)
    v01 = s1 / s21**2 * (e_aux @ e_aux)
    q = c01 / np.sqrt(v01)
    return TestResult("Cox test (Pesaran)", f"{eq1.text} is right (against {eq2.text})", float(q), None,
                      float(2 * sps.norm.sf(abs(q))), "normal", details={"observations": n, "c01": float(c01)})
