"""How pymetrix defines its fit statistics: one function per definition, and a registry naming each one.

Every result computes its statistics with these functions, so a number called ``aic`` means the same thing on
every result. Where a result follows another convention by design (a VAR's lag-selection criteria) its key says
so through ``definitions(result)``; where a library's value differs and is kept for comparison it is stored
under ``<key>_backend`` with its own definition. ``docs/design/definitions.md`` has the formulas and what each
matches in other packages.

    definitions(fit)          # {"aic": "aic.total", "bic": "bic.total", "r2": "r2.centred", ...}
    DEFINITIONS["aic.total"]  # Definition(name, formula, matches, notes)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["DEFINITIONS", "Definition", "aic", "bic", "definitions", "gaussian_loglike", "hq", "per_observation", "r2", "r2_adjusted",
           "se_regression"]


@dataclass(frozen=True)
class Definition:
    id: str
    name: str
    formula: str
    matches: str = ""  # packages that report the same number
    notes: str = ""


def _defs(*items: Definition) -> dict[str, Definition]:
    return {d.id: d for d in items}


DEFINITIONS: dict[str, Definition] = _defs(
    Definition("loglike.gaussian", "Gaussian log likelihood (concentrated)", "-n/2 · (ln 2π + ln(SSR/n) + 1)",
               "statsmodels OLS llf, Stata e(ll), R logLik(lm), EViews"),
    Definition("loglike.exact", "log likelihood", "the model's own log likelihood at the estimates",
               "statsmodels llf, arch loglikelihood"),
    Definition("k.estimated", "parameters", "the number of estimated parameters reported in params; for least squares the "
               "coefficients, not the error variance (concentrated out)", "statsmodels df_model + const, Stata e(rank)",
               "R's AIC(lm) also counts the error variance (k + 1)."),
    Definition("aic.total", "Akaike information criterion", "-2 · loglike + 2k", "statsmodels, Stata estat ic",
               "Totals, not per observation. EViews reports aic/n; R's AIC(lm) uses k + 1."),
    Definition("bic.total", "Schwarz (Bayesian) information criterion", "-2 · loglike + k · ln n", "statsmodels, Stata estat ic",
               "EViews reports bic/n."),
    Definition("hq.total", "Hannan-Quinn criterion", "-2 · loglike + 2k · ln ln n", "", "EViews reports hq/n."),
    Definition("aic.per_observation", "Akaike criterion per observation", "(-2 · loglike + 2k) / n", "EViews"),
    Definition("aic.lutkepohl", "Akaike criterion (Lütkepohl)", "ln|Σ_ml| + 2 · K · m / T, m regressors per equation",
               "statsmodels VAR, EViews and Stata varsoc lag selection", "Ranks lag orders as aic.total does for a fixed T: "
               "aic.total / T = aic.lutkepohl + K(1 + ln 2π)."),
    Definition("bic.lutkepohl", "Schwarz criterion (Lütkepohl)", "ln|Σ_ml| + ln T · K · m / T", "statsmodels VAR"),
    Definition("hq.lutkepohl", "Hannan-Quinn criterion (Lütkepohl)", "ln|Σ_ml| + 2 ln ln T · K · m / T", "statsmodels VAR"),
    Definition("fpe.lutkepohl", "final prediction error", "((T + m) / (T − m))^K · |Σ_ml|", "statsmodels VAR"),
    Definition("aic.k_plus_variance", "Akaike criterion counting the error variance", "-2 · loglike + 2(k + 1)",
               "statsmodels ARDL and ETS", "Kept as aic_backend for comparison: it is aic.total + 2."),
    Definition("bic.k_plus_variance", "Schwarz criterion counting the error variance", "-2 · loglike + (k + 1) · ln n",
               "statsmodels ARDL and ETS", "Kept as bic_backend: it is bic.total + ln n."),
    Definition("aic.gaussian_quasi", "information criterion of a Gaussian quasi-likelihood", "-2 · loglike.gaussian + 2k (AIC) or + k ln n (BIC)",
               "", "For estimators that are not Gaussian maximum likelihood (quantile regression): useful only to compare "
               "fits of the same estimator."),
    Definition("r2.centred", "R-squared", "1 − SSR / Σ(y − ȳ)² (weighted when weights are given)", "statsmodels rsquared, Stata, EViews",
               "When the equation has a constant."),
    Definition("r2.uncentred", "uncentred R-squared", "1 − SSR / Σy²", "statsmodels rsquared without a constant",
               "When the equation has no constant."),
    Definition("r2.within", "within R-squared", "1 − SSR / Σ(y − ȳ_entity)²", "Stata xtreg fe (within), linearmodels rsquared_within"),
    Definition("r2.pseudo_mcfadden", "McFadden's pseudo R-squared", "1 − loglike / loglike_null", "statsmodels prsquared, Stata"),
    Definition("r2.adjusted", "adjusted R-squared", "1 − (n − c) / df_resid · (1 − R²), c = 1 with a constant, 0 without",
               "statsmodels rsquared_adj, Stata, EViews"),
    Definition("df_resid", "residual degrees of freedom", "n − k (less the fixed effects absorbed)", "statsmodels df_resid"),
    Definition("se_regression", "standard error of the regression", "√(SSR / df_resid)", "EViews S.E. of regression, Stata root MSE"),
    Definition("durbin_watson", "Durbin-Watson statistic", "Σ(u_t − u_{t−1})² / Σu_t²", "statsmodels durbin_watson, EViews"),
    Definition("ssr", "sum of squared residuals", "Σ w·u²"),
    Definition("nobs", "observations", "the rows estimated on, after missing values, lags and any presample"),
    Definition("backend", "as reported by the library", "the library's own value; see the result's provenance.backend"),
)

# a key's definition unless the result says otherwise (results set ``stat_definitions``)
DEFAULT_KEYS: dict[str, str] = {
    "loglike": "loglike.exact", "k": "k.estimated", "aic": "aic.total", "bic": "bic.total", "hq": "hq.total", "r2": "r2.centred",
    "r2_adj": "r2.adjusted", "r2_within": "r2.within", "pseudo_r2": "r2.pseudo_mcfadden", "df_resid": "df_resid",
    "se_regression": "se_regression", "durbin_watson": "durbin_watson", "ssr": "ssr", "nobs": "nobs",
    "aic_total": "aic.total", "bic_total": "bic.total", "hq_total": "hq.total",
    "aic_backend": "aic.k_plus_variance", "bic_backend": "bic.k_plus_variance",
}


def gaussian_loglike(ssr: float, n: int) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):  # an exact fit has ssr = 0
        return float(-n / 2 * (np.log(2 * np.pi) + np.log(ssr / n) + 1))


def aic(loglike: float, k: float) -> float:
    """``aic.total``: -2 · loglike + 2k."""
    return float(-2 * loglike + 2 * k)


def bic(loglike: float, k: float, n: float) -> float:
    """``bic.total``: -2 · loglike + k · ln n."""
    return float(-2 * loglike + k * np.log(n))


def hq(loglike: float, k: float, n: float) -> float:
    """``hq.total``: -2 · loglike + 2k · ln ln n."""
    return float(-2 * loglike + 2 * k * np.log(np.log(n)))


def per_observation(value: float, n: float) -> float:
    """A criterion per observation, as EViews reports it."""
    return float(value / n)


def r2(ssr: float, tss: float) -> float:
    return float(1 - ssr / tss) if tss > 0 else float("nan")


def r2_adjusted(r2_value: float, n: float, df_resid: float, centred: bool) -> float:
    return float(1 - (n - (1 if centred else 0)) / df_resid * (1 - r2_value))


def se_regression(ssr: float, df_resid: float) -> float:
    return float(np.sqrt(ssr / df_resid))


def definitions(result: Any) -> dict[str, str]:
    """The definition each of ``result.stats``' keys follows (``DEFINITIONS`` explains them)."""
    stats = getattr(result, "stats", None)
    if not isinstance(stats, Mapping):
        return {}
    own = dict(getattr(type(result), "stat_definitions", {}) or {})
    out = {}
    for key in stats:
        if key in own:
            out[key] = own[key]
        elif key in DEFAULT_KEYS:
            out[key] = DEFAULT_KEYS[key]
    if "r2" in out and out["r2"] == "r2.centred" and not _centred(result):
        out["r2"] = "r2.uncentred"
    if "loglike" in stats and out.get("loglike") == "loglike.exact" and _least_squares(result):
        out["loglike"] = "loglike.gaussian"
    if _quasi(result):  # quantile, robust regression and shrinkage: a Gaussian quasi-likelihood of their residuals
        for key, value in (("loglike", "loglike.gaussian"), ("aic", "aic.gaussian_quasi"), ("bic", "aic.gaussian_quasi")):
            if key in out:
                out[key] = value
    return out


def _quasi(result: Any) -> bool:
    return getattr(getattr(result, "method", None), "kind", "") in ("quantile", "robust", "shrinkage")


def _centred(result: Any) -> bool:
    eq = getattr(result, "equation", None)
    return bool(getattr(eq, "intercept", True)) if eq is not None else True


def _least_squares(result: Any) -> bool:
    return getattr(getattr(result, "method", None), "kind", "") in ("ls", "iv")
