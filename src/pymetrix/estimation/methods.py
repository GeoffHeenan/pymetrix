"""Estimation methods: configuration objects that say *how* to estimate.

Each method implements ``estimate(equation, data, *, span, where, weights,
missing) -> FittedEquation``. ``OLS`` and ``IV`` handle both equation forms
(explicit-form equations by nonlinear least squares / nonlinear 2SLS); ``ML``
lives in ``ml.py``. The remaining classes are defined so specifications can
be written against them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ..errors import SpecificationError
from ..forms import Widget
from ..plugins import register
from ..expr import BinOp, Coef, ExpressionError, diff, evaluate
from .covariance import Cluster, covariance
from .priors import Beta as _Beta, Gamma as _Gamma, InvGamma as _InvGamma, InvGamma1 as _InvGamma1, Normal as _Normal, Uniform as _Uniform
from .design import (
    ExplicitDesign,
    check_hac,
    cov_groups,
    explicit_design,
    instrument_series,
    linear_design,
    ls_stats,
    provenance,
)
from .fitted import FittedEquation
from .params import point

if TYPE_CHECKING:
    from ..data import Dataset
    from .equation import Equation


_START = Widget("", advanced=True, help="starting values for explicit-form equations, e.g. b0=0.5, b1=0.1")
_RESTRICTIONS = Widget("text", help="linear restrictions on the coefficients, imposed in estimation (restricted least squares), "
                                    "e.g. y + x[-1] = 1, const = 0; separate several with commas; empty = none")


def _restrictions_field(value: Any) -> tuple[str, ...] | None:
    from .restrictions import normalise

    return normalise(value)


@dataclass(frozen=True)
class Calibrated:
    """Marks parameters that were set, not estimated."""


@dataclass(frozen=True)
class OLS:
    """Least squares. Explicit-form equations are estimated by nonlinear least squares.

    The coefficients are consistent when the regressors are uncorrelated with the error (exogeneity); an omitted
    variable correlated with a regressor, simultaneity or measurement error in a regressor biases them, and only
    instruments (IV, GMM) put that right. The standard errors change the inference, not the estimates: classic
    assumes homoskedastic, serially uncorrelated errors; white allows heteroskedasticity; hac (Newey-West) allows
    autocorrelation as well, in time series; Cluster(...) allows any correlation within groups, and is needed when
    the regressor of interest varies only between groups. R-squared measures fit in the sample, not whether the
    coefficients are causal. Explicit-form equations are solved numerically from the starting values, so check
    that the estimates do not depend on them.

    ``restrictions=["y + c[-1] = 1"]`` imposes linear restrictions R b = r on a linear-form equation's coefficients
    (restricted least squares): b - (X'X)^-1 R' [R (X'X)^-1 R']^-1 (R b - r), the estimate the restrictions
    substituted into the equation would give, with that substitution's covariance (of the chosen kind) and one
    residual degree of freedom more per restriction. Coefficients are named as the summary names them.
    """

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import OLS, Dataset, Equation

        data = Dataset.example("sample_macro")
        fit = Equation("c ~ y").fit(data)                        # OLS is the default method
        print(fit.summary())                                     # y: the marginal propensity to consume out of output
        fit = Equation("c ~ y").fit(data, method=OLS(cov="hac"))
        print(fit.summary())                                     # the same coefficients; Newey-West standard errors allow for autocorrelation
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("coef, std err, t", ("standard errors as cov sets them: classic assumes homoskedastic, uncorrelated errors; "
                              "the Method line names any other (White, HAC with its lags, clustered)")),
        ("R-squared, Adjusted R-squared", ("the share of the dependent variable's variance fitted in the sample; high "
                                           "values say nothing about whether the coefficients are causal")),
        ("S.E. of regression", ("the estimated standard deviation of the errors, in the units of the dependent "
                                "variable")),
        ("Akaike info criterion, Schwarz criterion", ("lower is better, comparing models of the same dependent "
                                                      "variable on the same sample; Schwarz penalises extra "
                                                      "parameters most")),
        ("Durbin-Watson", ("time series only: near 2 without first-order autocorrelation, well below 2 with positive; "
                           "invalid with a lagged dependent variable (use Breusch-Godfrey)")),
        ("fit.stats", ("the same numbers for code: r2, r2_adj, se_regression, ssr, loglike, aic, bic, hq, "
                       "durbin_watson, nobs, df_resid")),
    )

    label: ClassVar[str] = "OLS"
    topics: ClassVar[tuple[str, ...]] = ('least-squares',)
    references: ClassVar[tuple[str, ...]] = (
        "Greene, W. H. (2018). Econometric Analysis, 8th ed., chapters 2 (\"The linear regression model\"), 3 (\"Least squares regression\") and 4 (\"Estimating the regression model by least squares\"). Pearson.",
        ("White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. Econometrica 48(4), 817-838. doi:10.2307/1912934"),
        ("Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. Econometrica 55(3), 703-708. doi:10.2307/1913610"),
    )
    kind: ClassVar[str] = "ls"

    cov: str | Cluster = "classic"  # "classic", "white", "hac" or Cluster(...)
    hac_lags: int | None = None
    start: Annotated[Mapping[str, float] | None, _START] = None  # NLS starting values
    hac_correction: Annotated[bool, Widget("checkbox", advanced=True,
                              help="HAC: multiply by n / (n - k), as EViews and Stata's newey; untick for statsmodels' default")] = True
    restrictions: Annotated[str | Sequence[str] | None, _RESTRICTIONS] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "restrictions", _restrictions_field(self.restrictions))

    def estimate(self, eq: Equation, data: Dataset, *, span=None, where=None, weights=None, missing="drop") -> FittedEquation:
        _require_gaussian(eq, "OLS")
        if eq.form == "linear":
            return _ols_linear(self, eq, data, span, where, weights, missing)
        return _ols_explicit(self, eq, data, span, where, weights, missing)


@dataclass(frozen=True)
class IV:
    """Two-stage least squares.

    ``endog`` names the endogenous regressors (linear form: term labels such
    as ``"y"``; explicit form: variables whose current value is endogenous).
    ``instruments`` lists the excluded instruments as expressions. Every other
    regressor, and the constant, instruments itself.

    The estimates are consistent when the excluded instruments are relevant (they move the endogenous regressors,
    given the exogenous ones) and valid (uncorrelated with the error, so rightly left out of the equation). At
    least as many excluded instruments as endogenous regressors are needed; more over-identify the equation. For
    the linear form the notes give each endogenous regressor's first-stage F of the excluded instruments: below
    about 10 the instruments are weak, 2SLS is biased towards OLS and its standard errors understate the
    uncertainty. Its diagnostics add the Sargan test of the over-identifying restrictions (it can detect invalid
    instruments only if some are valid) and a Durbin-Wu-Hausman test of whether the instrumented regressors are
    endogenous at all.

    ``restrictions=[...]`` imposes linear restrictions on the coefficients as for OLS, with (Xhat'Xhat)^-1 in place
    of (X'X)^-1 (Xhat: the regressors projected on the instruments); the first stage is unchanged.
    """

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import IV, Dataset, Equation

        card = Dataset.example("card")                           # Card (1995): growing up near a college shifts schooling
        eq = Equation("lwage ~ educ + exper + black")
        fit = eq.fit(card, method=IV(endog=["educ"], instruments=["nearc4"]))
        print(fit.summary())                                     # educ: the return to a year of schooling; the notes give the first-stage F
        over = eq.fit(card, method=IV(endog=["educ"], instruments=["nearc4", "nearc2"]))
        over.test("sargan")                                      # two instruments for one regressor: Sargan tests the extra restriction
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("coef, std err, t", ("two-stage least-squares estimates; the default classic standard errors assume "
                              "homoskedastic errors (cov='white' or 'hac' otherwise)")),
        ("Instruments", ("every instrument, the exogenous regressors and the constant included: the excluded ones "
                         "must at least match the endogenous regressors")),
        ("First-stage F (name)", ("F of the excluded instruments in each endogenous regressor's first stage; below "
                                  "about 10 they are weak and 2SLS is biased towards OLS")),
        ("R-squared", ("can be negative under IV, as the residuals use the actual endogenous regressors; it is no "
                       "guide to the model")),
        ("fit.diagnostics()", ("first-stage F, Sargan (over-identified only), Durbin-Wu-Hausman (a small p-value: the "
                               "regressors are endogenous, OLS inconsistent) and Jarque-Bera")),
        ("fit.test(\"sargan\")", ("chi-squared with (excluded instruments - endogenous) df; a small p-value says some "
                                  "instrument is invalid, a large one cannot prove all are valid")),
    )

    label: ClassVar[str] = "IV"
    topics: ClassVar[tuple[str, ...]] = ('instrumental-variables',)
    references: ClassVar[tuple[str, ...]] = (
        ("Basmann, R. L. (1957). A generalized classical method of linear estimation of coefficients in a structural equation. Econometrica 25(1), 77-83. doi:10.2307/1907743"),
        ("Sargan, J. D. (1958). The estimation of economic relationships using instrumental variables. Econometrica 26(3), 393-415. doi:10.2307/1907619"),
        "Hausman, J. A. (1978). Specification tests in econometrics. Econometrica 46(6), 1251-1271. doi:10.2307/1913827",
    )
    kind: ClassVar[str] = "iv"

    endog: Sequence[str]
    instruments: Sequence[str]
    cov: str | Cluster = "classic"
    hac_lags: int | None = None
    start: Annotated[Mapping[str, float] | None, _START] = None
    hac_correction: Annotated[bool, Widget("checkbox", advanced=True,
                              help="HAC: multiply by n / (n - k), as EViews and Stata's newey; untick for statsmodels' default")] = True
    restrictions: Annotated[str | Sequence[str] | None, _RESTRICTIONS] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "restrictions", _restrictions_field(self.restrictions))

    def estimate(self, eq: Equation, data: Dataset, *, span=None, where=None, weights=None, missing="drop") -> FittedEquation:
        _require_gaussian(eq, "IV")
        if not self.endog or not self.instruments:
            raise SpecificationError("IV needs endog=[...] (endogenous regressors) and instruments=[...] (excluded instruments)")
        if eq.form == "linear":
            return _iv_linear(self, eq, data, span, where, weights, missing)
        return _iv_explicit(self, eq, data, span, where, weights, missing)


# Defined for specifications; implementation follows the build plan.


_Prior = _Normal | _Beta | _Gamma | _InvGamma | _InvGamma1 | _Uniform  # a single equation's priors (a VAR's prior is its own object)


@dataclass(frozen=True)
class Bayes:
    """Bayesian estimation, of VARs (by their priors) and of single equations (by coefficient priors).

    For VARs: ``prior=Minnesota(...)``, with exact posterior draws, or ``prior=NormalWishart(...)``, sampled by Gibbs.
    For single equations (Gaussian, logit, probit, Poisson): ``prior={"x": Normal(0.5, 0.1), "sigma": InvGamma1(1, 1)}``
    by coefficient name, flat where none is given; ``sampler="auto"`` is Gibbs where every step is exact (normal or flat
    coefficient priors, an inverse gamma or none on a Gaussian error) and random-walk Metropolis-Hastings otherwise
(``sampler="rwmh"`` always), and ``sampler="nuts"`` the No-U-Turn sampler with analytic gradients (``metric``
    its mass matrix: ``"diag"``, ``"dense"`` or ``"mode"``, as ``FittedModel.mcmc``);
    ``chains`` chains of ``draws`` each after ``burn``. See pymetrix.estimation.bayes_equation.

    A single equation's priors can also be typed, as the Estimate dialog gives them: ``prior="x=Normal(0.5, 0.1);
    sigma=InvGamma1(1, 1)"`` (by mean and standard deviation; read into the dictionary of priors)."""

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Bayes, Dataset, Equation
        from pymetrix.estimation.priors import Normal

        data = Dataset.example("sample_macro")
        prior = {"y": Normal(0.6, 0.05)}                         # by mean and standard deviation; const, with none, gets a flat one
        fit = Equation("c ~ y").fit(data, method=Bayes(prior=prior, draws=1000, burn=500))
        print(fit.summary())                                     # posterior means and HPD intervals; R-hat near 1 and a large ESS mean the chains agree
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("mean, s.d., median", ("posterior summaries of each coefficient and of sigma, the error standard deviation; "
                                "the means are fit.params (sigma's is fit.stats['sigma'])")),
        ("90% HPD low, 90% HPD high", ("the shortest interval holding 90% of the posterior draws; "
                                       "fit.posterior(coverage) gives other levels")),
        ("R-hat", ("compares the chains: near 1 they agree; above 1.1 a note warns, and more draws or other priors "
                   "are needed")),
        ("ESS", ("the effective number of independent draws behind each summary; a small one means a large Monte "
                 "Carlo error")),
        ("MCSE", ("the Monte Carlo standard error of the posterior mean: s.d. / sqrt(ESS), how far the mean of these "
                  "draws may be from the posterior mean itself; fit.stats['mcse_<name>'] and fit.stats['ess_<name>'] "
                  "give it and the ESS by parameter")),
        ("Method, Priors", ("the sampler (Gibbs, Metropolis-Hastings or NUTS), chains, draws and burn-in, and the priors "
                            "used; a coefficient without one has a flat prior")),
        ("fit.draws", ("every kept draw, one row per draw with its chain, for plots and for functions of the "
                       "parameters")),
    )

    label: ClassVar[str] = "Bayes"
    topics: ClassVar[tuple[str, ...]] = ("bayesian-equations",)
    references: ClassVar[tuple[str, ...]] = (
        "Koop, G. (2003). Bayesian Econometrics. Wiley.",
        ("Litterman, R. B. (1986). Forecasting with Bayesian vector autoregressions: five years of experience. Journal of Business and Economic Statistics 4(1), 25-38. doi:10.2307/1391384"),
        ("Kadiyala, K. R. and Karlsson, S. (1997). Numerical methods for estimation and inference in Bayesian VAR models. Journal of Applied Econometrics 12(2), 99-132. doi:10.1002/(sici)1099-1255(199703)12:2<99::aid-jae429>3.0.co;2-a"),
    )
    kind: ClassVar[str] = "bayes"

    prior: Annotated[str | Mapping[str, _Prior] | None, Widget("text", help="priors by coefficient, mean and standard deviation, e.g. x=Normal(0.5, 0.1); "
                                              "sigma=InvGamma1(1, 1) (Normal, Beta, Gamma, InvGamma, InvGamma1, Uniform); "
                                              "empty = flat")] = None
    draws: Annotated[int, Widget("number", min=100, max=1_000_000, step=500, help="posterior draws kept per chain after the burn-in; more lower the Monte Carlo error of the summaries")] = 5000
    sampler: Annotated[Literal["auto", "rwmh", "nuts"], Widget("choice", advanced=True,
                       help="auto: Gibbs where every step is exact, else random-walk Metropolis-Hastings; nuts: the No-U-Turn "
                            "sampler (gradients), whose burn-in adapts its step size and mass matrix")] = "auto"
    seed: Annotated[int | None, Widget("number", min=0, max=2**31 - 1, advanced=True, help="random seed; empty = a fresh one")] = 0
    chains: Annotated[int, Widget("number", min=1, max=16, help="independent chains, for R-hat")] = 2  # single equations
    burn: Annotated[int, Widget("number", min=0, max=1_000_000, step=500, advanced=True,
                                help="draws discarded at the start of each chain")] = 1000  # single equations
    metric: Annotated[Literal["diag", "dense", "mode"], Widget("choice", advanced=True, show_if=("sampler", ("nuts",)),
                      help="NUTS: the mass matrix; dense undoes correlations between coefficients (adapted in the "
                           "burn-in), mode keeps the inverse Hessian at the mode")] = "diag"  # single equations

    def __post_init__(self) -> None:
        if isinstance(self.prior, str):  # typed priors: x=Normal(0.5, 0.1); ... (or nothing: flat)
            from .priors import parse

            object.__setattr__(self, "prior", parse(self.prior) or None)

    def estimate(self, eq: Equation, data: Dataset, *, span=None, where=None, weights=None, missing="drop") -> FittedEquation:
        from .bayes_equation import estimate

        return estimate(self, eq, data, span, where, weights, missing)


# shared pieces


def _require_gaussian(eq: Equation, method: str) -> None:
    from .families import Gaussian

    if not isinstance(eq.family, Gaussian):
        raise SpecificationError(f"{method} needs a Gaussian equation; use method=ML() for family={type(eq.family).__name__}()")
    if eq.has_error_model:
        raise SpecificationError("ARMA errors and GARCH variance are estimated by maximum likelihood; use method=ML()")


def _weights(w: np.ndarray | None, n: int) -> np.ndarray:
    return np.sqrt(w) if w is not None else np.ones(n)


def _check_rank(M: np.ndarray, names: Sequence[str], what: str) -> None:
    if np.linalg.matrix_rank(M) < M.shape[1]:
        raise SpecificationError(f"{what} are perfectly collinear: {list(names)}")


def _start(eq: Equation, start: Mapping[str, float] | None) -> np.ndarray:
    start = dict(start or {})
    return np.array([start.get(p.name, p.start if p.start is not None else 0.1) for p in eq.param_specs])


class _Nonlinear:
    """Residuals and Jacobian of an explicit-form equation over a design."""

    def __init__(self, eq: Equation, d: ExplicitDesign):
        self.names = list(eq.param_names)
        self.expr = BinOp("-", eq.lhs, eq.rhs)
        self.grads = [diff(self.expr, Coef(p)) for p in self.names]
        self.d = d
        self.sw = _weights(d.w, d.n)

    def f(self, theta: np.ndarray) -> np.ndarray:
        r = evaluate(self.expr, self.d.resolve, dict(zip(self.names, theta)))
        return np.broadcast_to(r, (self.d.n,)) * self.sw

    def jac(self, theta: np.ndarray) -> np.ndarray:
        vals = dict(zip(self.names, theta))
        cols = [np.broadcast_to(evaluate(g, self.d.resolve, vals), (self.d.n,)) for g in self.grads]
        return np.column_stack(cols) * self.sw[:, None]

    def lhs(self, eq: Equation) -> np.ndarray:
        return np.broadcast_to(evaluate(eq.lhs, self.d.resolve), (self.d.n,))


def _solve_nonlinear(eq: Equation, start: np.ndarray, fun, jac, n_resid: int):
    lower = np.array([p.bounds[0] for p in eq.param_specs])
    upper = np.array([p.bounds[1] for p in eq.param_specs])
    bounded = bool(np.isfinite(lower).any() or np.isfinite(upper).any())
    if not np.all(np.isfinite(fun(start))):
        raise ExpressionError("the equation cannot be evaluated at the starting values; set start={...} or Param(start=...)")
    result = least_squares(
        fun, start, jac=jac, bounds=(lower, upper) if bounded else (-np.inf, np.inf),
        method="trf" if bounded or n_resid < len(start) else "lm", xtol=1e-12, ftol=1e-12, gtol=1e-12,
    )
    if not result.success:
        raise RuntimeError(f"nonlinear estimation did not converge: {result.message}")
    return result.x


# OLS


def _hac_lags(m: Any, n: int) -> dict[str, float]:
    """The HAC truncation lag actually used (automatic unless given), as a statistic of the fit."""
    from .covariance import default_hac_lags

    if getattr(m, "cov", None) != "hac":
        return {}
    return {"hac_lags": float(m.hac_lags if m.hac_lags else default_hac_lags(n))}


def _ols_linear(m: OLS, eq: Equation, data: Dataset, span, where, weights, missing) -> FittedEquation:
    check_hac(m.cov, data)
    d = linear_design(eq, data, span, where, weights, missing)
    return _ols_on_design(m, eq, d, lambda: cov_groups(m.cov, data, d.index), data.kind == "time_series",
                          lambda: provenance(eq, m, data, d.index, d.dropped), weights)


def _ols_on_design(m: OLS, eq: Equation, d: Any, groups: Any, time_series: bool, prov: Any, weights: str | None) -> FittedEquation:
    """Least squares on a linear design: shared by ``Equation.fit`` and ``pymetrix.ols`` (arrays passed directly), so
    both give the same numbers. ``groups`` and ``prov`` are functions giving the cluster groups and the provenance."""
    sw = _weights(d.w, len(d.y))
    yt, Xt = d.y * sw, d.X * sw[:, None]
    n, k = Xt.shape
    _check_rank(Xt, d.names, "regressors")
    beta = np.linalg.lstsq(Xt, yt, rcond=None)[0]
    R = _restriction_matrices(m, d.names)
    q = len(R[2]) if R else 0
    if R:
        beta, M = _restrict(beta, Xt, R)
    ut = yt - Xt @ beta
    df_resid = n - k - d.absorbed + q
    cov = covariance(m.cov, Xt, ut, df_resid, groups=groups(), hac_lags=m.hac_lags, hac_correction=getattr(m, "hac_correction", True))
    if R:
        cov = M @ cov @ M.T
    resid = pd.Series(ut / sw, index=d.index, name="resid")
    return FittedEquation(
        equation=eq,
        params=point(d.names, beta, cov),
        method=m,
        resid=resid,
        fitted=pd.Series(d.y_observed - resid.to_numpy(), index=d.index, name="fitted"),
        stats=_hac_lags(m, len(ut)) | ls_stats(d.y, ut / sw, d.w, k - q, df_resid, eq.intercept, time_series, bool(eq.fixed_effects)),
        provenance=prov(),
        design=pd.DataFrame(Xt, index=d.index, columns=d.names),
        weights=weights,
        **_restriction_extra(R),
    )


def _restriction_matrices(m: Any, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """``(R, r, texts)`` of the method's restrictions on the coefficients ``names``, or None without restrictions."""
    if not getattr(m, "restrictions", None):
        return None
    from .restrictions import linear_restrictions

    return linear_restrictions(m.restrictions, names)


def _restrict(beta: np.ndarray, X: np.ndarray, R: tuple[np.ndarray, np.ndarray, list[str]]) -> tuple[np.ndarray, np.ndarray]:
    """The restricted estimate and the map M of the covariance (``M V M'``), with bread ``(X'X)^-1`` (X: the weighted
    regressors, or for 2SLS their projection on the instruments)."""
    from .restrictions import restricted

    return restricted(beta, np.linalg.inv(X.T @ X), R[0], R[1])


def _restriction_extra(R: tuple[np.ndarray, np.ndarray, list[str]] | None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The fit's ``extra`` with the restrictions imposed (keyword arguments of FittedEquation; none without them)."""
    out = dict(extra or {})
    if R:
        out["restrictions"] = list(R[2])
        out["notes"] = [*out.get("notes", []), f"Restrictions imposed: {', '.join(R[2])}"]
    return {"extra": out} if out else {}


def _no_restrictions(m: Any, eq: Equation) -> None:
    if getattr(m, "restrictions", None):
        raise SpecificationError(f"restrictions {list(m.restrictions)} apply to linear-form equations; for an explicit-form "
                                 "equation, substitute them into the equation (e.g. write b2 as 1 - b1)")


def _ols_explicit(m: OLS, eq: Equation, data: Dataset, span, where, weights, missing) -> FittedEquation:
    _no_restrictions(m, eq)
    check_hac(m.cov, data)
    d = explicit_design(eq, data, span, where, weights, missing)
    nl = _Nonlinear(eq, d)
    theta = _solve_nonlinear(eq, _start(eq, m.start), nl.f, nl.jac, d.n)
    ut = nl.f(theta)
    X = -nl.jac(theta)  # derivative of the fitted value with respect to each parameter
    df_resid = d.n - len(theta)
    cov = covariance(m.cov, X, ut, df_resid, groups=cov_groups(m.cov, data, d.index), hac_lags=m.hac_lags, hac_correction=getattr(m, "hac_correction", True))
    lhs = nl.lhs(eq)
    resid = pd.Series(ut / nl.sw, index=d.index, name="resid")
    return FittedEquation(
        equation=eq,
        params=point(nl.names, theta, cov),
        method=m,
        resid=resid,
        fitted=pd.Series(lhs - resid.to_numpy(), index=d.index, name="fitted"),
        stats=_hac_lags(m, len(ut)) | ls_stats(lhs, ut / nl.sw, d.w, len(theta), df_resid, True, data.kind == "time_series", False),
        provenance=provenance(eq, m, data, d.index, d.dropped),
        design=pd.DataFrame(X, index=d.index, columns=nl.names),
        weights=weights,
    )


# IV


def _first_stage(Xt: np.ndarray, Zt: np.ndarray, Z_exog: np.ndarray, endog_idx: list[int], q: int, absorbed: int) -> tuple[dict[int, float], np.ndarray]:
    """Partial F of the excluded instruments for each endogenous regressor, and first-stage residuals."""
    n, L = Zt.shape
    df = n - L - absorbed
    F, V = {}, []
    for j in endog_idx:
        x = Xt[:, j]
        e_u = x - Zt @ np.linalg.lstsq(Zt, x, rcond=None)[0]
        e_r = x - Z_exog @ np.linalg.lstsq(Z_exog, x, rcond=None)[0] if Z_exog.shape[1] else x
        ssr_u, ssr_r = e_u @ e_u, e_r @ e_r
        F[j] = float(((ssr_r - ssr_u) / q) / (ssr_u / df))
        V.append(e_u)
    return F, np.column_stack(V)


def _iv_linear(m: IV, eq: Equation, data: Dataset, span, where, weights, missing) -> FittedEquation:
    check_hac(m.cov, data)
    labels = [t.label for t in eq.terms]
    unknown = [e for e in m.endog if e not in labels]
    if unknown:
        raise SpecificationError(f"endog {unknown} must name regressors of the equation: {labels}")
    d = linear_design(eq, data, span, where, weights, missing, instruments=instrument_series(m.instruments, data))
    endog_idx = [d.names.index(e) for e in m.endog]
    exog_idx = [j for j in range(len(d.names)) if j not in endog_idx]
    sw = _weights(d.w, len(d.y))
    yt, Xt = d.y * sw, d.X * sw[:, None]
    Zx = Xt[:, exog_idx]
    Zt = np.column_stack([Zx, d.Z * sw[:, None]])  # type: ignore[operator]
    z_names = [d.names[j] for j in exog_idx] + d.z_names
    n, k = Xt.shape
    if Zt.shape[1] < k:
        raise SpecificationError(f"the equation is underidentified: {Zt.shape[1]} instruments for {k} regressors")
    _check_rank(Zt, z_names, "instruments")
    Xhat = Zt @ np.linalg.lstsq(Zt, Xt, rcond=None)[0]
    _check_rank(Xhat, d.names, "instrumented regressors")
    beta = np.linalg.lstsq(Xhat, yt, rcond=None)[0]
    R = _restriction_matrices(m, d.names)
    q = len(R[2]) if R else 0
    if R:
        beta, M = _restrict(beta, Xhat, R)
    ut = yt - Xt @ beta
    df_resid = n - k - d.absorbed + q
    cov = covariance(m.cov, Xhat, ut, df_resid, groups=cov_groups(m.cov, data, d.index), hac_lags=m.hac_lags, hac_correction=getattr(m, "hac_correction", True))
    if R:
        cov = M @ cov @ M.T
    F, V = _first_stage(Xt, Zt, Zx, endog_idx, len(d.z_names), d.absorbed)
    resid = pd.Series(ut / sw, index=d.index, name="resid")
    return FittedEquation(
        equation=eq,
        params=point(d.names, beta, cov),
        method=m,
        resid=resid,
        fitted=pd.Series(d.y_observed - resid.to_numpy(), index=d.index, name="fitted"),
        stats=_hac_lags(m, len(ut)) | ls_stats(d.y, ut / sw, d.w, k - q, df_resid, eq.intercept, data.kind == "time_series", bool(eq.fixed_effects)),
        provenance=provenance(eq, m, data, d.index, d.dropped, extra_variables=_instrument_variables(m)),
        design=pd.DataFrame(Xt, index=d.index, columns=d.names),
        weights=weights,
        **_restriction_extra(R, {
            "instruments": pd.DataFrame(Zt, index=d.index, columns=z_names),
            "endog": list(m.endog),
            "first_stage_F": {d.names[j]: F[j] for j in endog_idx},
            "first_stage_resid": V,
        }),
    )


def _instrument_variables(m: IV) -> list[str]:
    from ..expr import parse_expression

    return sorted(set().union(*(parse_expression(t).variables() for t in m.instruments)))


def _iv_explicit(m: IV, eq: Equation, data: Dataset, span, where, weights, missing) -> FittedEquation:
    _no_restrictions(m, eq)
    check_hac(m.cov, data)
    current = {r.id for r in eq.rhs.refs() if r.lag == 0}
    unknown = [e for e in m.endog if e not in current]
    if unknown:
        raise SpecificationError(f"endog {unknown} must be current-period variables on the right-hand side")
    d = explicit_design(eq, data, span, where, weights, missing, instruments=instrument_series(m.instruments, data))
    nl = _Nonlinear(eq, d)
    exog_refs = sorted((r for r in eq.rhs.refs() if not (r.lag == 0 and r.id in m.endog)), key=str)
    Z = np.column_stack([np.ones(d.n), *(d.arrays[(r.id, r.lag)] for r in exog_refs), d.Z]) * nl.sw[:, None]
    z_names = ["const", *(str(r) for r in exog_refs), *d.z_names]
    k = len(nl.names)
    if Z.shape[1] < k:
        raise SpecificationError(f"the equation is underidentified: {Z.shape[1]} instruments for {k} parameters")
    _check_rank(Z, z_names, "instruments")
    Q = np.linalg.qr(Z)[0]
    theta = _solve_nonlinear(eq, _start(eq, m.start), lambda t: Q.T @ nl.f(t), lambda t: Q.T @ nl.jac(t), Z.shape[1])
    ut = nl.f(theta)
    X = -nl.jac(theta)
    Xhat = Q @ (Q.T @ X)
    df_resid = d.n - k
    cov = covariance(m.cov, Xhat, ut, df_resid, groups=cov_groups(m.cov, data, d.index), hac_lags=m.hac_lags, hac_correction=getattr(m, "hac_correction", True))
    lhs = nl.lhs(eq)
    resid = pd.Series(ut / nl.sw, index=d.index, name="resid")
    return FittedEquation(
        equation=eq,
        params=point(nl.names, theta, cov),
        method=m,
        resid=resid,
        fitted=pd.Series(lhs - resid.to_numpy(), index=d.index, name="fitted"),
        stats=_hac_lags(m, len(ut)) | ls_stats(lhs, ut / nl.sw, d.w, k, df_resid, True, data.kind == "time_series", False),
        provenance=provenance(eq, m, data, d.index, d.dropped, extra_variables=_instrument_variables(m)),
        design=pd.DataFrame(X, index=d.index, columns=nl.names),
        weights=weights,
        extra={"instruments": pd.DataFrame(Z, index=d.index, columns=z_names), "endog": list(m.endog)},
    )


register("methods", OLS)
register("methods", IV)
register("methods", Bayes)
