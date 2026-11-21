"""Tests on fitted equations: coefficient restrictions and residual diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
import pandas as pd

from pymetrix._lazy import stats as sps

from .._random import generator as make_rng
from ..errors import SpecificationError
from ..estimation import FittedEquation
from ..expr import Coef, ExpressionError, diff, evaluate
from ..forms import Widget
from ..output import Table
from .base import Test, TestResult, register


def _split_top_level(text: str) -> list[str]:
    parts, depth, current = [], 0, ""
    for ch in text:
        depth += ch in "(["
        depth -= ch in ")]"
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    return [p.strip() for p in [*parts, current] if p.strip()]


@register
@dataclass(frozen=True)
class Wald(Test):
    """Wald test of one or more restrictions, e.g. ``"b1 + b2 = 1, b3 = 0"``.

    Nonlinear restrictions use the delta method. Coefficients are referred to by name, exactly as restricted least
    squares names them: ``const``, a term's label (``dlog(y)``, ``x[-1]``, ``log(P)``), an explicit-form parameter or
    a column of ``pymetrix.ols(y, X)``; a name may be put in backticks (`` `dlog(y)` ``), and spaces inside it are
    optional.

    The null is that all the restrictions hold. The statistic uses the fit's own coefficient covariance (robust or
    HAC if the fit used one) and is chi-squared with as many degrees of freedom as restrictions; when the fit has
    residual degrees of freedom the F form (the statistic over the number of restrictions, against the residual degrees of freedom) is in the
    details and is more conservative in small samples. With nonlinear restrictions the result depends on how they
    are written (b1 / b2 = 1 and b1 = b2 can give different answers), so prefer the linear form when there is one.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset, Equation

        card = Dataset.example("card")
        fit = Equation("lwage ~ educ + exper + black + south").fit(card)
        fit.test("wald", "black = 0, south = 0")                 # joint: chi-squared with 2 degrees of freedom, the F form in the details
        fit.test("wald", "educ = 2 * exper")                     # coefficients are named by their terms
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("statistic", ("chi-squared with as many df as restrictions, from the fit's own covariance (robust or HAC if "
                       "it used one)")),
        ("p-value", "a small one rejects that all the restrictions hold jointly"),
        ("F, F p-value", ("the statistic over the restrictions against the residual df; more conservative in small "
                          "samples, and missing for fits without residual df")),
        ("Null hypothesis", ("the restrictions as written; a nonlinear restriction's result depends on how it is "
                             "written")),
    )

    restriction: Annotated[str, Widget("text", help="restrictions on the coefficients, e.g. b1 + b2 = 1; separate several with commas")]

    name = "wald"
    topics = ("diagnostic-tests",)
    references = (
        ("Wald, A. (1943). Tests of statistical hypotheses concerning several parameters when the number of observations is large. Transactions of the American Mathematical Society 54(3), 426-482. doi:10.1090/s0002-9947-1943-0012401-3"),
        ("Gregory, A. W. and Veall, M. R. (1985). Formulating Wald tests of nonlinear restrictions. Econometrica 53(6), 1465-1468. doi:10.2307/1913221"),
    )
    null = "the restrictions hold"
    applies_to = (FittedEquation,)

    def run(self, fit: FittedEquation) -> TestResult:
        p = fit.params
        if p.cov is None:
            raise SpecificationError("the fit has no covariance matrix")
        names = list(p.names)
        values = p.as_dict()
        g, R = [], []
        from ..estimation.restrictions import parse_restrictions

        for part in _split_top_level(self.restriction):
            if "=" not in part:
                raise ExpressionError(f"restriction {part!r} needs '='")
        for part, expr in parse_restrictions(self.restriction, names):  # the parser restricted least squares uses
            g.append(float(evaluate(expr, lambda *_: np.nan, values)))
            R.append([float(evaluate(diff(expr, Coef(n)), lambda *_: np.nan, values)) for n in names])
        g_arr, R_arr = np.array(g), np.array(R)
        stat = float(g_arr @ np.linalg.solve(R_arr @ p.cov @ R_arr.T, g_arr))
        q = len(g)
        details = {}
        df_resid = fit.stats.get("df_resid")
        if df_resid:
            details["F"] = stat / q
            details["F p-value"] = float(sps.f.sf(stat / q, q, df_resid))
        return TestResult("Wald", f"{self.restriction}", stat, q, float(sps.chi2.sf(stat, q)), "chi2", details=details)


def _time_series_residuals(fit: FittedEquation, test: str) -> np.ndarray:
    if fit.resid is None:
        raise SpecificationError(f"{test} needs residuals; calibrated equations have none")
    if isinstance(fit.resid.index, pd.MultiIndex):
        raise SpecificationError(f"{test} needs a single time series")
    return fit.resid.to_numpy(float)


def _innovations(fit: FittedEquation) -> np.ndarray:
    """Standardised residuals when the fit has an error model (ARMA, GARCH), else residuals."""
    std = fit.extra.get("std_resid")
    return (std if std is not None else fit.resid).to_numpy(float)  # type: ignore[union-attr]


@register
@dataclass(frozen=True)
class LjungBox(Test):
    """Ljung-Box Q test of residual autocorrelation (``squared=True``: of squared residuals,
    the McLeod-Li test). Degrees of freedom are reduced by the number of ARMA terms.

    The null is that the residuals (standardised residuals when the fit has an ARMA or GARCH error model) are not
    autocorrelated up to the given lag; Q is chi-squared and rejects when large. A rejection means dynamics are left
    in the residuals: add lags or ARMA terms. The number of lags trades off: many dilute the power against
    correlation at short lags, few miss seasonal correlation. For least-squares residuals with a lagged dependent
    variable among the regressors, prefer Breusch-Godfrey. The squared version detects volatility clustering; a
    rejection there points to an ARCH or GARCH model of the variance."""

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import Dataset, Equation

        data = Dataset.example("sample_macro")
        fit = Equation("c ~ y").fit(data)
        fit.test("ljung_box", lags=8)                            # Q on the first 8 autocorrelations of the residuals; a small p-value: dynamics left over
        fit.test("ljung_box", lags=8, squared=True)              # McLeod-Li: the squared residuals, for volatility clustering
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("statistic", ("Q from the first lags autocorrelations of the residuals (standardised after ARMA or GARCH); "
                       "large values mean dynamics left in them")),
        ("df", "the lags, less the ARMA terms of the fit (not for squared residuals)"),
        ("p-value", "a small one rejects no autocorrelation: add lags or ARMA terms"),
        ("McLeod-Li (squared residuals)", ("with squared=True: a rejection points to volatility clustering (ARCH, "
                                           "GARCH)")),
    )

    lags: Annotated[int, Widget("", help="autocorrelations tested jointly; about ln(T), a few seasonal cycles for seasonal data")] = 10
    squared: Annotated[bool, Widget("checkbox", help="test squared residuals (McLeod-Li: volatility clustering)")] = False

    name = "ljung_box"
    topics = ("diagnostic-tests",)
    references = (
        ("Ljung, G. M. and Box, G. E. P. (1978). On a measure of lack of fit in time series models. Biometrika 65(2), 297-303. doi:10.1093/biomet/65.2.297"),
        ("McLeod, A. I. and Li, W. K. (1983). Diagnostic checking ARMA time series models using squared-residual autocorrelations. Journal of Time Series Analysis 4(4), 269-273. doi:10.1111/j.1467-9892.1983.tb00373.x"),
    )
    null = "no autocorrelation up to the given lag"
    applies_to = (FittedEquation,)

    def run(self, fit: FittedEquation) -> TestResult:
        _time_series_residuals(fit, self.name)
        x = _innovations(fit)
        if self.squared:
            x = x**2
        x = x - x.mean()
        n = len(x)
        denom = x @ x
        rho = np.array([x[k:] @ x[:-k] / denom for k in range(1, self.lags + 1)])
        stat = float(n * (n + 2) * np.sum(rho**2 / (n - np.arange(1, self.lags + 1))))
        df = self.lags - (0 if self.squared else int(fit.extra.get("model_df", 0)))
        if df < 1:
            raise SpecificationError(f"use more than {self.lags} lags: the ARMA terms leave no degrees of freedom")
        name = "McLeod-Li (squared residuals)" if self.squared else "Ljung-Box Q"
        return TestResult(name, self.null, stat, df, float(sps.chi2.sf(stat, df)), "chi2")


@register
@dataclass(frozen=True)
class BreuschGodfrey(Test):
    """Breusch-Godfrey LM test for serial correlation in the residuals, up to ``lags``.

    The null is no serial correlation up to the given lag; the alternative is autoregressive or moving-average errors
    of that order. The residuals are regressed on the regressors and their own lags (missing initial lags set to
    zero), and n R-squared is chi-squared with as many degrees of freedom as lags. Unlike Durbin-Watson it stays
    valid with a lagged dependent variable among the regressors. A rejection means the usual standard errors are
    wrong and, with a lagged dependent variable, the estimates are inconsistent too: add dynamics to the equation, or,
    if it is otherwise right, use HAC standard errors. It applies to unweighted least-squares fits.

    Convention: the auxiliary regression has exactly the equation's regressors (Breusch 1978, Godfrey 1978; Greene,
    Econometric Analysis, 8th ed., chapter 20), all n observations with the missing initial lags of the residuals set to zero,
    and R-squared is measured about the residuals' mean. With a constant in the equation statsmodels gives the same
    number. Without one, statsmodels' ``acorr_breusch_godfrey`` adds a constant to the auxiliary regression, which
    changes the statistic; ``auxiliary_constant=True`` does the same (it changes nothing when the equation has a
    constant)."""

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import Dataset, Equation

        data = Dataset.example("sample_macro")
        fit = Equation("c ~ y + c[-1]").fit(data)
        fit.test("breusch_godfrey", lags=4)                      # n R-squared, chi-squared(4); valid with the lagged dependent variable c[-1]
        fit.test("breusch_godfrey", lags=1)                      # against first-order serial correlation only
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("statistic", "n times the R-squared of the residuals on the regressors and their own lags"),
        ("df", "the lags tested: the order of the AR or MA errors under the alternative"),
        ("p-value", ("a small one rejects no serial correlation: the usual standard errors are wrong, and with a "
                     "lagged dependent variable so are the estimates")),
    )

    lags: Annotated[int, Widget("", help="order of serial correlation tested against; at least the seasonal period for quarterly or monthly data")] = 4
    auxiliary_constant: Annotated[bool, Widget("checkbox", advanced=True,
                                  help="add a constant to the auxiliary regression when the equation has none, as "
                                       "statsmodels' acorr_breusch_godfrey does; no effect with a constant")] = False

    name = "breusch_godfrey"
    topics = ("diagnostic-tests",)
    references = (
        ("Breusch, T. S. (1978). Testing for autocorrelation in dynamic linear models. Australian Economic Papers 17(31), 334-355. doi:10.1111/j.1467-8454.1978.tb00635.x"),
        ("Godfrey, L. G. (1978). Testing against general autoregressive and moving average error models when the regressors include lagged dependent variables. Econometrica 46(6), 1293-1301. doi:10.2307/1913829"),
        "Greene, W. H. (2018). Econometric Analysis, 8th ed., chapter 20 (\"Serial correlation\"). Pearson.",
    )
    null = "no serial correlation up to the given lag"
    applies_to = (FittedEquation,)

    def run(self, fit: FittedEquation) -> TestResult:
        u = _time_series_residuals(fit, self.name)
        _, X, _ = _ols_parts(fit, self.name)
        n = len(u)
        lagged = np.column_stack([np.concatenate([np.zeros(k), u[:-k]]) for k in range(1, self.lags + 1)])
        Z = np.column_stack([X, lagged])
        if self.auxiliary_constant and not _has_constant(X):
            Z = np.column_stack([np.ones(n), Z])
        e = u - Z @ np.linalg.lstsq(Z, u, rcond=None)[0]
        r2 = 1 - (e @ e) / np.sum((u - u.mean()) ** 2)
        stat = float(n * r2)
        return TestResult("Breusch-Godfrey LM", self.null, stat, self.lags, float(sps.chi2.sf(stat, self.lags)), "chi2")


def _has_constant(X: np.ndarray) -> bool:
    """Whether a column of the regressors is a nonzero constant (an intercept, however it is written)."""
    return bool(X.shape[0]) and bool(np.any((np.ptp(X, axis=0) == 0) & (X[0] != 0)))


@register
@dataclass(frozen=True)
class JarqueBera(Test):
    """Jarque-Bera test of normal residuals, from their skewness and kurtosis."""

    maturity: ClassVar[str] = "stable"
    name = "jarque_bera"
    topics = ("diagnostic-tests",)
    references = (
        ("Jarque, C. M. and Bera, A. K. (1980). Efficient tests for normality, homoscedasticity and serial independence of regression residuals. Economics Letters 6(3), 255-259. doi:10.1016/0165-1765(80)90024-5"),
        ("Jarque, C. M. and Bera, A. K. (1987). A test for normality of observations and regression residuals. International Statistical Review 55(2), 163-172. doi:10.2307/1403192"),
    )
    null = "residuals are normally distributed"
    applies_to = (FittedEquation,)

    def run(self, fit: FittedEquation) -> TestResult:
        if fit.resid is None:
            raise SpecificationError("Jarque-Bera needs residuals")
        u = _innovations(fit)
        skew = float(sps.skew(u))
        kurt = float(sps.kurtosis(u, fisher=False))
        stat = len(u) / 6 * (skew**2 + (kurt - 3) ** 2 / 4)
        return TestResult("Jarque-Bera", self.null, stat, 2, float(sps.chi2.sf(stat, 2)), "chi2", details={"skewness": skew, "kurtosis": kurt})


# helpers for least-squares diagnostics


def _ols_parts(fit: FittedEquation, test: str, linear: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(y, X, u) of an unweighted least-squares fit."""
    from ..estimation import OLS

    if not isinstance(fit.method, OLS):
        raise SpecificationError(f"{test} applies to least-squares (OLS) fits")
    if fit.weights:
        raise NotImplementedError(f"{test} for weighted fits")
    if linear and (fit.equation.form != "linear" or fit.equation.fixed_effects):
        raise SpecificationError(f"{test} needs a linear-form equation without fixed effects")
    X = fit.design.to_numpy(float)  # type: ignore[union-attr]
    u = fit.resid.to_numpy(float)  # type: ignore[union-attr]
    return X @ fit.params.values + u, X, u


def _aux_r2(y: np.ndarray, Z: np.ndarray) -> tuple[float, int]:
    """Centred R-squared of y on Z, and the rank of Z."""
    e = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(1 - (e @ e) / np.sum((y - y.mean()) ** 2)), int(np.linalg.matrix_rank(Z))


def _ssr(y: np.ndarray, X: np.ndarray) -> float:
    e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(e @ e)


def _f_result(name: str, null: str, ssr_r: float, ssr_u: float, q: int, df: int, **details: float) -> TestResult:
    F = ((ssr_r - ssr_u) / q) / (ssr_u / df)
    return TestResult(name, null, float(F), (q, df), float(sps.f.sf(F, q, df)), "F", details=details)


@register
@dataclass(frozen=True)
class White(Test):
    """White's heteroskedasticity test, with cross terms."""

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import OLS, Dataset, Equation

        mroz = Dataset.example("mroz")
        fit = Equation("hours ~ educ + exper + kidslt6 + nwifeinc").fit(mroz)
        fit.test("white")                                        # squared residuals on the regressors, their squares and cross products
        robust = Equation("hours ~ educ + exper + kidslt6 + nwifeinc").fit(mroz, method=OLS(cov="white"))  # after a rejection: robust standard errors
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("statistic", ("n times the R-squared of the squared residuals on the regressors, their squares and cross "
                       "products")),
        ("df", ("the auxiliary regression's regressors less the constant; it grows fast with the regressors and costs "
                "power")),
        ("p-value", ("a small one rejects homoskedasticity: the classic standard errors are wrong; use cov='white' or "
                     "model the variance")),
    )

    name = "white"
    topics = ("diagnostic-tests",)
    references = (
        ("White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. Econometrica 48(4), 817-838. doi:10.2307/1912934"),
    )
    null = "homoskedastic errors"
    applies_to = (FittedEquation,)

    def run(self, fit: FittedEquation) -> TestResult:
        _, X, u = _ols_parts(fit, self.name)
        i0, i1 = np.triu_indices(X.shape[1])
        r2, rank = _aux_r2(u**2, X[:, i0] * X[:, i1])
        stat, df = len(u) * r2, rank - 1
        return TestResult("White", self.null, stat, df, float(sps.chi2.sf(stat, df)), "chi2")


@register
@dataclass(frozen=True)
class BreuschPagan(Test):
    """Breusch-Pagan test, Koenker's studentised form, on the regressors."""

    maturity: ClassVar[str] = "stable"
    name = "breusch_pagan"
    topics = ("diagnostic-tests",)
    references = (
        ("Breusch, T. S. and Pagan, A. R. (1979). A simple test for heteroscedasticity and random coefficient variation. Econometrica 47(5), 1287-1294. doi:10.2307/1911963"),
        ("Koenker, R. (1981). A note on studentizing a test for heteroscedasticity. Journal of Econometrics 17(1), 107-112. doi:10.1016/0304-4076(81)90062-2"),
    )
    null = "homoskedastic errors"
    applies_to = (FittedEquation,)

    def run(self, fit: FittedEquation) -> TestResult:
        _, X, u = _ols_parts(fit, self.name)
        r2, rank = _aux_r2(u**2, X)
        stat, df = len(u) * r2, rank - 1
        return TestResult("Breusch-Pagan (Koenker)", self.null, stat, df, float(sps.chi2.sf(stat, df)), "chi2")


@register
@dataclass(frozen=True)
class ARCH(Test):
    """Engle's LM test for ARCH effects: do squared residuals depend on their own lags?

    The null is no ARCH effects: the squared residuals (standardised residuals when the fit has an ARMA or GARCH error
    model) are not predictable from their own lags. The statistic is the number of observations times the R-squared
    of that regression, chi-squared with as many degrees of freedom as lags. A rejection suggests modelling the
    variance with an ARCH or GARCH model, or at least heteroskedasticity-robust standard errors; on a GARCH fit it
    checks whether the model has captured the volatility clustering. A misspecified mean equation can also produce a
    rejection, so check the residuals' autocorrelation first."""

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        import numpy as np
        import pandas as pd
        from pymetrix import GARCH, Dataset, Equation

        rng = np.random.default_rng(1)                           # simulated returns from a known GARCH(1, 1)
        r, h = np.zeros(1500), np.ones(1500)
        for t in range(1, 1500):                                 # omega 0.1, alpha 0.1, beta 0.85
            h[t] = 0.1 + 0.1 * r[t - 1] ** 2 + 0.85 * h[t - 1]
            r[t] = np.sqrt(h[t]) * rng.standard_normal()
        data = Dataset.from_frame(pd.DataFrame({"r": r}, index=pd.period_range("2020-01-01", periods=1500, freq="D", name="time")))
        ols = Equation("r ~ 1").fit(data)
        ols.test("arch", lags=4)                                 # do squared residuals depend on their own lags?
        garch = Equation("r ~ 1", variance=GARCH(1, 1)).fit(data)
        garch.test("arch", lags=4)                               # on the standardised residuals: has the GARCH captured the clustering?
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("statistic", "the observations used times the R-squared of the squared residuals on their own lags"),
        ("df", "the lags of squared residuals tested"),
        ("p-value", ("a small one rejects no ARCH effects: model the variance; on a GARCH fit, a large one says the "
                     "clustering is captured")),
    )

    lags: Annotated[int, Widget("", help="lags of the squared residuals tested; a few detect persistent volatility, many dilute power")] = 4

    name = "arch"
    topics = ("diagnostic-tests",)
    references = (
        ("Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. Econometrica 50(4), 987-1007. doi:10.2307/1912773"),
    )
    null = "no ARCH effects up to the given lag"
    applies_to = (FittedEquation,)

    def run(self, fit: FittedEquation) -> TestResult:
        _time_series_residuals(fit, self.name)
        e2 = _innovations(fit) ** 2
        p = self.lags
        Z = np.column_stack([np.ones(len(e2) - p)] + [e2[p - k : len(e2) - k] for k in range(1, p + 1)])
        r2, _ = _aux_r2(e2[p:], Z)
        stat = (len(e2) - p) * r2
        return TestResult("ARCH LM", self.null, stat, p, float(sps.chi2.sf(stat, p)), "chi2")


@register
@dataclass(frozen=True)
class RESET(Test):
    """Ramsey RESET: adds powers 2..power of the fitted values.

    The null is that the functional form is right: powers of the fitted values add nothing to the regression. The F
    test of their joint significance rejects when something is missing, such as a nonlinearity, an omitted variable
    correlated with the powers, or the wrong transformation (levels rather than logs), but it does not say which. It
    has little power against omitted variables unrelated to the fitted values. It applies to linear-form
    least-squares fits without fixed effects or weights.
    """

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import Dataset, Equation

        mroz = Dataset.example("mroz")                           # lwage is missing for women not working, so those rows drop out
        fit = Equation("lwage ~ educ + exper").fit(mroz)
        fit.test("reset")                                        # F test of the squared and cubed fitted values: a small p-value says the form is wrong
        Equation("lwage ~ educ + exper + expersq").fit(mroz).test("reset", power=2)  # squares only
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("statistic", "F of the added powers of the fitted values"),
        ("df", "(powers added, residual df of the augmented regression)"),
        ("p-value", ("a small one rejects the functional form, without saying what is wrong: a nonlinearity, an "
                     "omitted variable or the transformation")),
    )

    power: Annotated[int, Widget("number", min=2, max=6, help="highest power of the fitted values added (3 = squares and cubes)")] = 3

    name = "reset"
    topics = ("diagnostic-tests",)
    references = (
        ("Ramsey, J. B. (1969). Tests for specification errors in classical linear least-squares regression analysis. Journal of the Royal Statistical Society, Series B (Methodological) 31(2), 350-371. doi:10.1111/j.2517-6161.1969.tb00796.x"),
    )
    null = "no omitted nonlinearity"
    applies_to = (FittedEquation,)

    def run(self, fit: FittedEquation) -> TestResult:
        y, X, u = _ols_parts(fit, self.name, linear=True)
        yhat = y - u
        extra = np.column_stack([yhat**p for p in range(2, self.power + 1)])
        q = extra.shape[1]
        return _f_result("RESET", self.null, float(u @ u), _ssr(y, np.column_stack([X, extra])), q, len(y) - X.shape[1] - q)


@register
@dataclass(frozen=True)
class NeuralNetwork(Test):
    """Lee, White and Granger's (1993) neural-network test for neglected nonlinearity: hidden units with random weights.

    The null is that the conditional mean is linear in the regressors. The standardised regressors pass through
    hidden logistic units with random weights, and the leading principal components of those units are added to the
    regression: the F test of their joint significance is reported, with n R-squared (chi-squared) in the details. It
    has power against a broad range of nonlinearities, but because the weights are random the p-value depends on the
    seed: check a borderline result with a few seeds. It applies to linear-form least-squares fits.
    """

    maturity: ClassVar[str] = "beta"
    units: Annotated[int, Widget("number", min=2, max=50, help="random hidden units (logistic of a random combination of the regressors)")] = 10
    components: Annotated[int, Widget("number", min=1, max=10, help="their principal components added (avoiding collinearity)")] = 2
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, help="random seed for the hidden units' weights")] = 0

    name = "neural_network"
    topics = ("diagnostic-tests",)
    references = (
        ("Lee, T.-H., White, H. and Granger, C. W. J. (1993). Testing for neglected nonlinearity in time series models: a comparison of neural network methods and alternative tests. Journal of Econometrics 56(3), 269-290. doi:10.1016/0304-4076(93)90122-l"),
    )
    null = "the regression is linear in the regressors (no neglected nonlinearity)"
    applies_to = (FittedEquation,)

    def run(self, fit: FittedEquation) -> TestResult:
        """The regressors (less the constant) standardised; ``units`` hidden units psi(g'x), g uniform on [-2, 2] and psi
        the logistic, centred; their first ``components`` principal components added to the regression: an F test,
        and n R^2 of the residuals on everything (chi-squared) in the details."""
        y, X, u = _ols_parts(fit, self.name, linear=True)
        n = len(y)
        varying = [j for j in range(X.shape[1]) if np.std(X[:, j]) > 0]
        if not varying:
            raise SpecificationError("the neural-network test needs regressors other than the constant")
        W = X[:, varying]
        W = (W - W.mean(axis=0)) / W.std(axis=0)
        Wc = np.column_stack([np.ones(n), W])
        gamma = make_rng(self.seed).uniform(-2, 2, (Wc.shape[1], self.units))
        hidden = 1 / (1 + np.exp(-Wc @ gamma))
        hidden = hidden - hidden.mean(axis=0)
        U, s, _ = np.linalg.svd(hidden, full_matrices=False)
        q = min(self.components, int((s > 1e-10 * s[0]).sum()))
        extra = U[:, :q] * s[:q]
        full = np.column_stack([X, extra])
        e = u - full @ np.linalg.lstsq(full, u, rcond=None)[0]
        r2 = 1 - (e @ e) / (u @ u)  # u has mean zero when X has a constant
        return _f_result("Neural network (Lee-White-Granger)", self.null, float(u @ u), _ssr(y, full), q, n - X.shape[1] - q,
                         **{"n R2": float(n * r2), "n R2 p-value": float(sps.chi2.sf(n * r2, q)), "units": self.units,
                            "components": q, "seed": self.seed})


@register
@dataclass(frozen=True)
class Chow(Test):
    """Chow breakpoint test: do the coefficients change from period ``at``?

    The null is that all the coefficients are the same before and after the given period. The F statistic compares
    the sum of squared residuals of the full-sample fit with that of separate fits to the two regimes, assuming the
    error variance is the same in both; a variance that changes too distorts it. The date must come from outside the
    data: choosing it where the series seems to break makes the test reject too often, and for an unknown date the
    Quandt-Andrews test is the right one. Each regime needs more observations than coefficients.
    """

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import Dataset, Equation

        nile = Dataset.example("nile")                           # annual flow at Aswan, 1871-1970
        fit = Equation("volume ~ 1").fit(nile)
        fit.test("chow", at="1899")                              # a date from outside the data: the first Aswan dam was begun in 1898
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("statistic", "F comparing the full-sample sum of squared residuals with the two regimes' separate fits"),
        ("df", "(coefficients, observations - 2 x coefficients)"),
        ("p-value", ("a small one rejects equal coefficients before and after the date; valid only for a date chosen "
                     "from outside the data")),
    )

    at: Annotated[str, Widget("period", help="first period of the second regime, chosen from outside the data, not by eye")]

    name = "chow"
    topics = ("diagnostic-tests",)
    references = (
        ("Chow, G. C. (1960). Tests of equality between sets of coefficients in two linear regressions. Econometrica 28(3), 591-605. doi:10.2307/1910133"),
    )
    null = "no structural break at the given period"
    applies_to = (FittedEquation,)

    def run(self, fit: FittedEquation) -> TestResult:
        y, X, u = _ols_parts(fit, self.name, linear=True)
        index = fit.resid.index  # type: ignore[union-attr]
        if not isinstance(index, pd.PeriodIndex):
            raise SpecificationError("the Chow test needs a single time series")
        after = np.asarray(index >= pd.Period(self.at, freq=index.freq))
        n, k = X.shape
        if min(after.sum(), (~after).sum()) <= k:
            raise SpecificationError(f"each side of {self.at} needs more than {k} observations")
        ssr_u = _ssr(y[~after], X[~after]) + _ssr(y[after], X[after])
        return _f_result("Chow breakpoint", f"no structural break at {self.at}", float(u @ u), ssr_u, k, n - 2 * k)


# IV diagnostics


def _iv_parts(fit: FittedEquation, test: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from ..estimation import IV

    if not isinstance(fit.method, IV):
        raise SpecificationError(f"{test} applies to IV fits")
    if fit.weights:
        raise NotImplementedError(f"{test} for weighted fits")
    X = fit.design.to_numpy(float)  # type: ignore[union-attr]
    u = fit.resid.to_numpy(float)  # type: ignore[union-attr]
    Z = fit.extra["instruments"].to_numpy(float)
    return X @ fit.params.values + u, X, u, Z


@register
@dataclass(frozen=True)
class FirstStage(Test):
    """Instrument strength: partial F of the excluded instruments in each first stage."""

    maturity: ClassVar[str] = "stable"
    name = "first_stage"
    topics = ("diagnostic-tests",)
    references = (
        ("Staiger, D. and Stock, J. H. (1997). Instrumental variables regression with weak instruments. Econometrica 65(3), 557-586. doi:10.2307/2171753"),
        "Stock, J. H. and Yogo, M. (2005). Testing for weak instruments in linear IV regression. In Andrews, D. W. K. and Stock, J. H. (eds.), Identification and Inference for Econometric Models: Essays in Honor of Thomas Rothenberg, 80-108. Cambridge University Press. doi:10.1017/cbo9780511614491.006",
    )
    null = "the excluded instruments are irrelevant (rule of thumb: F below 10 signals weak instruments)"
    applies_to = (FittedEquation,)

    def run(self, fit: FittedEquation) -> TestResult:
        _iv_parts(fit, self.name)
        F = fit.extra.get("first_stage_F")
        if not F:
            raise SpecificationError("first-stage statistics are available for linear-form IV fits")
        Z = fit.extra["instruments"]
        q = len(fit.method.instruments)  # type: ignore[attr-defined]
        df = int(fit.stats["df_resid"]) + len(fit.params) - Z.shape[1]
        weakest = min(F, key=F.get)
        return TestResult(
            f"First-stage F ({weakest})", self.null, F[weakest], (q, df), float(sps.f.sf(F[weakest], q, df)), "F",
            details={f"F ({name})": value for name, value in F.items()},
        )


@register
@dataclass(frozen=True)
class HansenJ(Test):
    """Hansen's J test of overidentifying restrictions, from an IV-GMM fit."""

    maturity: ClassVar[str] = "beta"
    name = "hansen_j"
    topics = ("diagnostic-tests",)
    references = (
        ("Hansen, L. P. (1982). Large sample properties of generalized method of moments estimators. Econometrica 50(4), 1029-1054. doi:10.2307/1912775"),
    )
    null = "the instruments are valid (the overidentifying restrictions hold)"
    applies_to = (FittedEquation,)

    def run(self, fit: FittedEquation) -> TestResult:
        if "hansen_j" not in fit.stats:
            raise SpecificationError("Hansen's J comes from IV-GMM fits (method=IVGMM(...))")
        return TestResult("Hansen J", self.null, fit.stats["hansen_j"], fit.extra.get("j_df"), fit.stats["hansen_j_pvalue"], "chi2")


@register
@dataclass(frozen=True)
class Sargan(Test):
    """Sargan test of overidentifying restrictions."""

    maturity: ClassVar[str] = "beta"
    name = "sargan"
    topics = ("diagnostic-tests",)
    references = (
        ("Sargan, J. D. (1958). The estimation of economic relationships using instrumental variables. Econometrica 26(3), 393-415. doi:10.2307/1907619"),
    )
    null = "the instruments are valid (uncorrelated with the error)"
    applies_to = (FittedEquation,)

    def run(self, fit: FittedEquation) -> TestResult:
        _, X, u, Z = _iv_parts(fit, self.name)
        df = Z.shape[1] - X.shape[1]
        if df < 1:
            raise SpecificationError("the Sargan test needs more instruments than regressors (the equation is exactly identified)")
        r2, _ = _aux_r2(u, Z)
        stat = len(u) * r2
        return TestResult("Sargan", self.null, stat, df, float(sps.chi2.sf(stat, df)), "chi2")


@register
@dataclass(frozen=True)
class Endogeneity(Test):
    """Durbin-Wu-Hausman test (regression form): are the instrumented regressors endogenous?"""

    maturity: ClassVar[str] = "beta"
    name = "endogeneity"
    topics = ("diagnostic-tests",)
    references = (
        "Durbin, J. (1954). Errors in variables. Review of the International Statistical Institute 22(1/3), 23-32. doi:10.2307/1401917",
        ("Wu, D.-M. (1973). Alternative tests of independence between stochastic regressors and disturbances. Econometrica 41(4), 733-750. doi:10.2307/1914093"),
        "Hausman, J. A. (1978). Specification tests in econometrics. Econometrica 46(6), 1251-1271. doi:10.2307/1913827",
    )
    null = "the regressors treated as endogenous are exogenous (OLS is consistent)"
    applies_to = (FittedEquation,)

    def run(self, fit: FittedEquation) -> TestResult:
        y, X, _, _ = _iv_parts(fit, self.name)
        V = fit.extra.get("first_stage_resid")
        if V is None:
            raise SpecificationError("the endogeneity test is available for linear-form IV fits")
        q = V.shape[1]
        df = int(fit.stats["df_resid"]) - q
        return _f_result("Durbin-Wu-Hausman", self.null, _ssr(y, X), _ssr(y, np.column_stack([X, V])), q, df)


def diagnostics(fit: FittedEquation) -> Table:
    """The default checks for the fit's method, as one table."""
    from ..estimation import IV, OLS

    time_series = fit.resid is not None and isinstance(fit.resid.index, pd.PeriodIndex)
    checks: list[Test] = []
    if fit.equation.has_error_model:
        checks = [LjungBox(), LjungBox(squared=True), ARCH(), JarqueBera()]
    elif isinstance(fit.method, OLS) and not fit.weights:
        if time_series:
            checks.append(BreuschGodfrey())
        checks += [BreuschPagan(), JarqueBera()]
        if fit.equation.form == "linear" and not fit.equation.fixed_effects:
            checks.append(RESET())
    elif isinstance(fit.method, IV) and not fit.weights and "first_stage_F" in fit.extra:
        checks.append(FirstStage())
        if fit.extra["instruments"].shape[1] > len(fit.params):
            checks.append(Sargan())
        checks += [Endogeneity(), JarqueBera()]
    else:
        checks.append(JarqueBera())
    rows = []
    for t in checks:
        r = t.run(fit)
        rows.append([r.name, r.statistic, r.p_value, r.null])
    return Table(["test", "statistic", "p-value", "null"], rows, title="Diagnostics")
