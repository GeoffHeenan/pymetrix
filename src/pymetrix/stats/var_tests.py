"""Tests on fitted VARs: Granger causality, residual autocorrelation, normality."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np

from pymetrix._lazy import stats as sps

from ..errors import SpecificationError
from ..forms import Widget
from ..var import FittedVAR
from ..vecm import FittedVECM
from .base import Test, TestResult, register


def _from_statsmodels(name: str, null: str, res: object) -> TestResult:
    df = res.df  # type: ignore[attr-defined]
    dist = "F" if isinstance(df, tuple) else "chi2"
    return TestResult(name, null, float(res.test_statistic), df, float(res.pvalue), dist)  # type: ignore[attr-defined]


def _names(value: str | Sequence[str]) -> list[str]:
    """A list of names, or one string of comma-separated names."""
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    return list(value)


@register
@dataclass(frozen=True)
class Granger(Test):
    """Do lags of ``cause`` help predict ``effect``? Wald test on the VAR, reported as F.

    The null is that every lag of the causing variables has a zero coefficient in the equations of the caused ones:
    given the past of the other variables, they do not help predict them. For a VAR the Wald statistic over the
    number of restrictions is reported as F, with the chi-squared form in the details; for a VECM statsmodels
    computes the test. Granger causality is about prediction, not structural causation, and it is relative to the
    variables in the system: an omitted variable can create or hide it. With integrated variables in a VAR in levels
    the statistic need not have its usual distribution, so treat its p-value with caution, or test in a VECM when the
    variables are cointegrated.
    """

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import VAR, Dataset

        data = Dataset.example("sample_macro").derive(dlc="dlog(c)", dli="dlog(i)")
        var = VAR(["dlc", "dli"], lags=2).fit(data)
        var.test("granger", cause="dli", effect="dlc")           # F test that both lags of investment growth are zero in the consumption equation
        var.test("granger", cause="dlc", effect="dli")           # and the other direction
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("statistic", ("for a VAR the Wald statistic over the number of restrictions, read as F; for a VECM "
                       "statsmodels' test")),
        ("df", "(lags of the causing variables tested, residual df of the whole system)"),
        ("p-value", ("a small one rejects that the causing variables' lags help predict nothing: predictive, not "
                     "structural, causality")),
        ("Wald chi2, Wald p-value", "the chi-squared form of the same test; close to the F form in large samples"),
    )

    cause: Annotated[str | Sequence[str], Widget("text", help="the variables whose lags are tested")]
    effect: Annotated[str | Sequence[str], Widget("text", help="the variables they may help predict")]

    name = "granger"
    topics = ("var",)
    references = (
        ("Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. Econometrica 37(3), 424-438. doi:10.2307/1912791"),
    )
    null = "the causing variables do not Granger-cause the caused variables"
    applies_to = (FittedVAR, FittedVECM)

    def run(self, var: FittedVAR | FittedVECM) -> TestResult:
        variables = list(var.variables)
        cause, effect = _names(self.cause), _names(self.effect)
        unknown = [v for v in cause + effect if v not in variables]
        if unknown:
            raise SpecificationError(f"{unknown} are not variables of the model: {variables}")
        if set(cause) & set(effect):
            raise SpecificationError("cause and effect must be different variables")
        if isinstance(var, FittedVECM):
            res = var.results.test_granger_causality(caused=[variables.index(v) for v in effect], causing=[variables.index(v) for v in cause])
            return _from_statsmodels(f"Granger causality: {', '.join(cause)} -> {', '.join(effect)}", self.null, res)
        regressors = var.regressors
        names = list(var.params.names)
        rows = [names.index(f"{e}: {c}[-{l}]") for l in range(1, var.lags + 1) for c in cause for e in effect]
        p = var.params
        b = p.values[rows]
        V = p.cov[np.ix_(rows, rows)]  # type: ignore[index]
        wald = float(b @ np.linalg.solve(V, b))
        q = len(rows)
        df2 = len(variables) * (len(var.arrays.index) - len(regressors))
        F = wald / q
        return TestResult(
            f"Granger causality: {', '.join(cause)} -> {', '.join(effect)}", self.null, F, (q, df2),
            float(sps.f.sf(F, q, df2)), "F", details={"Wald chi2": wald, "Wald p-value": float(sps.chi2.sf(wald, q))},
        )


@register
@dataclass(frozen=True)
class Portmanteau(Test):
    """Multivariate portmanteau (Ljung-Box type) test of residual autocorrelation up to ``lags``.

    The null is that the residuals of every equation are uncorrelated with each other's and their own past up to the
    given lag. For a VAR the statistic is chi-squared with K^2 (lags - p) degrees of freedom, K the number of
    variables and p the VAR's order, so the lag must exceed the order, and the approximation is good only well beyond
    it; for a VECM statsmodels computes the test. The adjusted statistic is closer to its distribution in small
    samples. A rejection suggests adding lags to the VAR.
    """

    maturity: ClassVar[str] = "stable"
    lags: Annotated[int, Widget("", help="residual autocorrelations tested; more than the VAR order, best well beyond it")] = 10
    adjusted: Annotated[bool, Widget("checkbox", help="small-sample adjusted statistic")] = False

    name = "portmanteau"
    topics = ("var",)
    references = (
        ("Hosking, J. R. M. (1980). The multivariate portmanteau statistic. Journal of the American Statistical Association 75(371), 602-608. doi:10.1080/01621459.1980.10477520"),
        "Lütkepohl, H. (2005). New Introduction to Multiple Time Series Analysis, chapter 4, \"VAR order selection and checking the model adequacy\". Springer. doi:10.1007/978-3-540-27752-1",
    )
    null = "no residual autocorrelation up to the given lag"
    applies_to = (FittedVAR, FittedVECM)

    def run(self, var: FittedVAR | FittedVECM) -> TestResult:
        if isinstance(var, FittedVECM):
            res = var.results.test_whiteness(nlags=self.lags, adjusted=self.adjusted)
            return _from_statsmodels("Portmanteau" + (" (adjusted)" if self.adjusted else ""), self.null, res)
        U = var.estimate.U
        T, K = U.shape
        if self.lags <= var.lags:
            raise SpecificationError(f"use more lags than the VAR order ({var.lags})")
        Uc = U - U.mean(axis=0)
        C0inv = np.linalg.inv(Uc.T @ Uc / T)
        stat = 0.0
        for j in range(1, self.lags + 1):
            Cj = Uc[j:].T @ Uc[:-j] / T
            term = np.trace(Cj.T @ C0inv @ Cj @ C0inv)
            stat += term / (T - j) if self.adjusted else term
        stat *= T**2 if self.adjusted else T
        df = K * K * (self.lags - var.lags)
        return TestResult("Portmanteau" + (" (adjusted)" if self.adjusted else ""), self.null, float(stat), df, float(sps.chi2.sf(stat, df)), "chi2")


@register
@dataclass(frozen=True)
class VARNormality(Test):
    """Multivariate Jarque-Bera test on Cholesky-standardised residuals (Lütkepohl)."""

    maturity: ClassVar[str] = "stable"
    name = "normality"
    topics = ("var",)
    references = (
        "Lütkepohl, H. (2005). New Introduction to Multiple Time Series Analysis, chapter 4, \"VAR order selection and checking the model adequacy\". Springer. doi:10.1007/978-3-540-27752-1",
        ("Jarque, C. M. and Bera, A. K. (1987). A test for normality of observations and regression residuals. International Statistical Review 55(2), 163-172. doi:10.2307/1403192"),
    )
    null = "the residuals are multivariate normal"
    applies_to = (FittedVAR, FittedVECM)

    def run(self, var: FittedVAR | FittedVECM) -> TestResult:
        if isinstance(var, FittedVECM):
            return _from_statsmodels("VECM residual normality (Jarque-Bera)", self.null, var.results.test_normality())
        U = var.estimate.U
        T, K = U.shape
        Uc = U - U.mean(axis=0)
        W = np.linalg.inv(np.linalg.cholesky(Uc.T @ Uc / T)) @ Uc.T
        b1 = (W**3).sum(axis=1) / T
        b2 = (W**4).sum(axis=1) / T - 3
        skew, kurt = T * b1 @ b1 / 6, T * b2 @ b2 / 24
        stat = float(skew + kurt)
        return TestResult("VAR residual normality (Jarque-Bera)", self.null, stat, 2 * K, float(sps.chi2.sf(stat, 2 * K)), "chi2",
                          details={"skewness chi2": float(skew), "kurtosis chi2": float(kurt)})
