"""Tests on data: unit roots (ADF, KPSS) and cointegration (Engle-Granger), via statsmodels.

ADF and Engle-Granger can instead simulate their critical values and p-value for the sample at hand
(``critical_values="simulated"``), with the statistic computed by ``pymetrix.stats.adf_arrays``, which reproduces
statsmodels' and is fast enough to compute it for thousands of random walks.

Johansen tests return a table of ranks rather than one statistic: see
``pymetrix.vecm.johansen`` and ``Dataset.johansen``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, ClassVar, Literal

import numpy as np

from ..errors import SpecificationError
from ..adapters import require
from ..adapters import arch as arch_adapter
from ..adapters import statsmodels as statsmodels_adapter
from ..adapters.base import AdapterSpec, BackendOptions, check_backend_options
from ..data import Dataset
from ..forms import Widget, advanced
from .base import Test, TestResult, register


def _series(data: Dataset, name: str, test: str) -> np.ndarray:
    if data.kind != "time_series":
        raise SpecificationError(f"{test} needs a single time series; select one entity first")
    return data[name].dropna().to_numpy(float)


CRITICAL_VALUES = ("mackinnon", "simulated")
_CRITICAL_HELP = ("mackinnon: MacKinnon's response surfaces (asymptotic, poor in short samples); simulated: the null "
                  "simulated for this sample's length, trend and lag rule")
_DRAWS_HELP = "random walks simulated with critical values = simulated; the p-value's simulation error falls as 1/sqrt(draws)"
_SEED_HELP = "random seed of the simulated critical values; the same seed repeats them"


def _check_simulation_settings(test: Any) -> None:
    if test.critical_values not in CRITICAL_VALUES:
        raise SpecificationError(f"critical_values is 'mackinnon' or 'simulated', not {test.critical_values!r}")
    if test.critical_values == "simulated":
        if test.backend_options:
            raise SpecificationError("backend_options are statsmodels' and are not used with simulated critical values; "
                                     "leave them empty")
        if isinstance(test.draws, bool) or int(test.draws) != test.draws or int(test.draws) < 10:
            raise SpecificationError(f"draws is a whole number of at least 10, not {test.draws!r}")


def _simulated_result(test: Any, data: Dataset, title: str, statistic: float, sims: np.ndarray, details: dict[str, Any],
                      null_text: str, warning: str = "") -> TestResult:
    """A test result whose critical values and p-value come from ``sims``, with the draws and seed in the provenance."""
    from .._random import get_random_generator
    from ..adapters.native import backend_record
    from . import adf_arrays
    from .base import _test_provenance

    summary = adf_arrays.simulated_summary(statistic, sims)
    details = {**details, "p-value MCSE": summary["mcse"], "draws": summary["draws"], "seed": int(test.seed),
               "note": (f"critical values and p-value simulated for this sample: {null_text}, each tested exactly as the "
                        "data are; not MacKinnon's asymptotic values")}
    if warning:
        details["warning"] = warning
    prov = _test_provenance(data, test)
    if prov is not None:
        simulation = {"draws": summary["draws"], "seed": int(test.seed), "random_generator": get_random_generator(), "null": null_text}
        prov = replace(prov, seed=int(test.seed), method={**(prov.method or {}), "simulation": simulation},
                       backend=backend_record("pymetrix.stats.adf_arrays"))
    return TestResult(title, test.null, float(statistic), None, summary["p_value"], "simulated for this sample",
                      critical_values=summary["critical_values"], details=details, provenance=prov)


def _adf_max_lag(n: int, ntrend: int, max_lags: Any, used: int = 0) -> int:
    """The largest lag searched: the one given (checked to leave a residual degree of freedom), or the default rule of
    ``adf_arrays.default_max_lag``, which keeps short samples from a regression with none."""
    from . import adf_arrays

    given = _int_or(max_lags, ())
    return adf_arrays.default_max_lag(n, ntrend, used) if given is None else adf_arrays.check_lag(given, n, ntrend)


def _int_or(value: Any, words: tuple[str, ...]) -> Any:
    """Settings typed in a form arrive as text: '4' -> 4; 'aic' stays; '' -> None."""
    if value in (None, ""):
        return None
    if isinstance(value, str) and value.strip().lower() in words:
        return value.strip().lower()
    return int(value)


@register
@dataclass(frozen=True)
class ADF(Test):
    """Augmented Dickey-Fuller test.

    The null is a unit root; the alternative is stationarity around the deterministic terms chosen in Trend. The
    statistic is the t-ratio on the lagged level in a regression of the first difference on it, the deterministic
    terms and lagged differences; it rejects when large and negative, against MacKinnon's p-values and critical
    values, which depend on the trend. Match the terms to the alternative: ct for a trending series (with a constant
    alone a trend-stationary series looks like a unit root), c for one that moves around a level, n only for a
    series with a known zero mean. Too few lags leave serial correlation that distorts the size; too many cost power,
    which is low anyway against roots near one in short samples. Not rejecting is weak evidence for a unit root:
    cross-check with KPSS (whose null is stationarity) or the GLS-based Ng-Perron and ERS tests, and remember that an
    unmodelled break in level or trend also makes ADF fail to reject (see the Perron test).

    MacKinnon's values are asymptotic and the lag search distorts them further in short samples. With
    ``critical_values="simulated"`` they are simulated for this sample instead: ``draws`` Gaussian random walks of the
    series' length (seeded by ``seed``), each tested exactly as the data are (the same trend terms, lag rule and lag
    search); the critical values are the 1, 5 and 10% quantiles of the draws' statistics and the p-value is the share
    at or below the statistic, with its Monte Carlo standard error.
    """

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import Dataset

        data = Dataset.example("sample_macro").derive(ly="log(y)", dly="dlog(y)")
        data.test("adf", "ly", trend="ct")                       # log output: a unit root against stationarity around a trend
        data.test("adf", "dly")                                  # its growth rate, around a level; reject when the statistic is below the critical value
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("statistic", ("the t-ratio on the lagged level; the more negative, the stronger the evidence against a unit "
                       "root")),
        ("p-value", ("MacKinnon's, for the trend chosen, or simulated for this sample (critical values = simulated, "
                     "with its Monte Carlo standard error); a small one rejects the unit root")),
        ("critical value 1%, 5%, 10%", ("reject the unit root when the statistic is below one; they are further out "
                                        "with ct than c")),
        ("lags used", ("lagged differences in the regression, chosen by aic, bic or t-stat or fixed; too few leave "
                       "serial correlation that distorts the test")),
        ("observations", "used in the test regression, after differencing and lags"),
    )

    wraps: ClassVar[str] = "statsmodels.tsa.stattools.adfuller"

    variable: str
    lags: Annotated[str | int, advanced(help="lagged differences absorbing serial correlation: a number, or aic / bic / t-stat to choose")] = "aic"
    trend: Annotated[Literal["c", "ct", "ctt", "n"], Widget("", help="terms under the alternative: c constant, ct adds a trend (trending series), ctt a quadratic, n none")] = "c"
    max_lags: Annotated[str | int, advanced(help="largest lag aic / bic / t-stat try; empty = 12*(n/100)^(1/4), capped in short samples to keep 4 residual degrees of freedom; unused with a fixed lag")] = ""
    backend_options: Annotated[BackendOptions, Widget("", advanced=True, help="other statsmodels arguments")] = field(default_factory=dict)
    critical_values: Annotated[Literal["mackinnon", "simulated"], Widget("", help=_CRITICAL_HELP)] = "mackinnon"
    draws: Annotated[int, advanced(help=_DRAWS_HELP)] = 2000
    seed: Annotated[int, advanced(help=_SEED_HELP)] = 0

    adapter_spec: ClassVar[AdapterSpec] = statsmodels_adapter.ADF

    name = "adf"
    topics = ("unit-roots",)
    references = (
        ("Dickey, D. A. and Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. Journal of the American Statistical Association 74(366a), 427-431. doi:10.1080/01621459.1979.10482531"),
        ("Said, S. E. and Dickey, D. A. (1984). Testing for unit roots in autoregressive-moving average models of unknown order. Biometrika 71(3), 599-607. doi:10.1093/biomet/71.3.599"),
        ("MacKinnon, J. G. (1996). Numerical distribution functions for unit root and cointegration tests. Journal of Applied Econometrics 11(6), 601-618. doi:10.1002/(sici)1099-1255(199611)11:6<601::aid-jae417>3.0.co;2-t"),
    )
    null = "the series has a unit root"
    applies_to = (Dataset,)

    def __post_init__(self) -> None:
        check_backend_options(self)
        _check_simulation_settings(self)

    def run(self, data: Dataset) -> TestResult:
        if self.critical_values == "simulated":
            return self._run_simulated(data)
        from . import adf_arrays

        stattools = require("statsmodels.tsa.stattools")
        x = _series(data, self.variable, "ADF")
        lags = _int_or(self.lags, ("aic", "bic", "t-stat"))
        ntrend = 0 if self.trend == "n" else len(self.trend)
        with statsmodels_adapter.ADAPTER.run(self.adapter_spec, self) as run, warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # statsmodels 0.15 warns about its future return type
            if isinstance(lags, int):
                fixed = adf_arrays.check_lag(lags, len(x), ntrend)
                out = run.call("test", stattools.adfuller, x, maxlag=fixed, regression=self.trend, autolag=None)
            else:
                auto = {"aic": "AIC", "bic": "BIC", "t-stat": "t-stat"}[lags or "aic"]
                out = run.call("test", stattools.adfuller, x, maxlag=_adf_max_lag(len(x), ntrend, self.max_lags),
                               regression=self.trend, autolag=auto)
        stat, p, used, nobs, crit = out[:5]  # with autolag (or store=True) statsmodels returns more
        adf_arrays.degenerate_statistic(float(stat), f"ADF ({self.variable})")
        return TestResult(
            f"ADF ({self.variable})", self.null, float(stat), None, float(p), "MacKinnon",
            critical_values={k: float(v) for k, v in crit.items()},
            details={"lags used": used, "observations": nobs}, backend_call=run.record,
        )

    def _run_simulated(self, data: Dataset) -> TestResult:
        from . import adf_arrays

        x = _series(data, self.variable, "ADF")
        if len(x) and x.max() == x.min():
            raise SpecificationError(f"{self.variable} is constant")
        lags = _int_or(self.lags, ("aic", "bic", "t-stat"))
        rule = ({"lags": lags, "autolag": None, "max_lags": None} if isinstance(lags, int)
                else {"lags": None, "autolag": lags or "aic", "max_lags": _int_or(self.max_lags, ())})
        fit = adf_arrays.adf(x, self.trend, **rule)
        adf_arrays.degenerate_statistic(float(fit.statistic[0]), f"ADF ({self.variable})")
        sims = adf_arrays.simulate_adf(len(x), self.trend, **rule, draws=self.draws, seed=self.seed)
        used, nobs = int(fit.lags[0]), int(fit.nobs[0])
        return _simulated_result(self, data, f"ADF ({self.variable})", float(fit.statistic[0]), sims,
                                 {"lags used": used, "observations": nobs},
                                 f"{int(self.draws)} Gaussian random walks of {len(x)} observations, trend {self.trend}")


@register
@dataclass(frozen=True)
class KPSS(Test):
    """KPSS test: the null is stationarity (the reverse of ADF).

    The null is that the series is stationary around a level (c) or a linear trend (ct); the alternative is a unit
    root. The statistic is the sum of squared partial sums of the residuals from the deterministic terms, scaled by a
    Newey-West (Bartlett) long-run variance, and it rejects when large. The p-value is interpolated in KPSS's table,
    so it lies between 0.01 and 0.10; a note says when it is at an edge. The lag of the long-run variance matters: too
    few with persistent but stationary data make the test reject too often, too many reduce its power. Read it with
    ADF: ADF not rejecting while KPSS rejects points to a unit root; both rejecting suggests that neither describes
    the data well, for instance because of a break.
    """

    maturity: ClassVar[str] = "stable"
    wraps: ClassVar[str] = "statsmodels.tsa.stattools.kpss"
    example: ClassVar[str] = """
        from pymetrix import Dataset

        data = Dataset.example("sample_macro").derive(dly="dlog(y)")
        data.test("kpss", "dly")                 # null: output growth is stationary around a level
        data.test("kpss", "y", trend="ct")       # null: output is stationary around a linear trend
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("statistic", "the KPSS statistic; large values are evidence against stationarity"),
        ("p-value", "interpolated in KPSS's table, so it lies between 0.01 and 0.10; a note says when it is at an edge"),
        ("critical values", "10%, 5%, 2.5% and 1%: reject stationarity when the statistic exceeds one"),
        ("lags used", "the Bartlett lags of the long-run variance; the result is sensitive to them in short samples"),
    )

    variable: str
    trend: Annotated[Literal["c", "ct"], Widget("", help="the null: c stationary around a level, ct around a linear trend")] = "c"
    lags: Annotated[str | int, advanced(help="Bartlett lags for the long-run variance: a number, auto (from the data) or legacy, 12(n/100)^(1/4)")] = "auto"
    backend_options: Annotated[BackendOptions, Widget("", advanced=True, help="other statsmodels arguments")] = field(default_factory=dict)

    adapter_spec: ClassVar[AdapterSpec] = statsmodels_adapter.KPSS

    name = "kpss"
    topics = ("unit-roots",)
    references = (
        ("Kwiatkowski, D., Phillips, P. C. B., Schmidt, P. and Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root: how sure are we that economic time series have a unit root? Journal of Econometrics 54(1-3), 159-178. doi:10.1016/0304-4076(92)90104-y"),
    )
    null = "the series is stationary (around a level, or a trend with trend='ct')"
    applies_to = (Dataset,)

    def __post_init__(self) -> None:
        check_backend_options(self)

    def run(self, data: Dataset) -> TestResult:
        stattools = require("statsmodels.tsa.stattools")
        x = _series(data, self.variable, "KPSS")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with statsmodels_adapter.ADAPTER.run(self.adapter_spec, self) as run:
                stat, p, lags, crit = run.call("test", stattools.kpss, x, regression=self.trend,
                                               nlags=_int_or(self.lags, ("auto", "legacy")) or "auto")
        details: dict[str, Any] = {"lags used": lags}
        if any("outside of the range" in str(w.message) for w in caught):
            details["note"] = "p-value is at the edge of the tabulated range (0.01 to 0.10)"
        return TestResult(f"KPSS ({self.variable})", self.null, float(stat), None, float(p), "KPSS (interpolated)",
                          critical_values={k: float(v) for k, v in crit.items()}, details=details, backend_call=run.record)


@register
@dataclass(frozen=True)
class EngleGranger(Test):
    """Engle-Granger (augmented) cointegration test: the first variable is regressed on the others.

    The first step regresses the first variable on the others and the deterministic terms by least squares; the
    second runs an ADF regression, without deterministic terms, on its residuals. The null is no cointegration (the
    residuals have a unit root); a large negative statistic rejects it in favour of a stationary combination. The
    p-values and critical values are MacKinnon's for residual-based tests: they depend on the number of variables and
    the trend, and are further out than ordinary ADF values because the coefficients are estimated. The result can
    depend on which variable comes first, and the test cannot say how many relations there are: with three or more
    variables, Johansen's tests count them. After a rejection, estimate the relation by FMOLS or DOLS for valid
    standard errors, or fit an error-correction model.

    MacKinnon's values are asymptotic, and with 13 to 30 observations they are far off. With
    ``critical_values="simulated"`` they are simulated for this sample instead: ``draws`` sets of independent Gaussian
    random walks, one per variable, of the sample's length (seeded by ``seed``), each tested exactly as the data are
    (the same trend terms in the levels regression, the same ADF regression on the residuals and the same lag
    search). Columns named in ``exog`` that are not I(1), such as impulse or break dummies, enter the levels
    regression of the data and of every draw as given. The critical values are the 1, 5 and 10% quantiles of the
    draws' statistics and the p-value is the share at or below the statistic, with its Monte Carlo standard error.
    """

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import Dataset

        data = Dataset.example("sample_macro").derive(lc="log(c)", ly="log(y)")
        data.test("engle_granger", "lc, ly")                     # ADF on the residuals of lc on ly; a large negative statistic: cointegrated
        data.test("engle_granger", "ly, lc")                     # the other normalisation, which can give a different answer
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("statistic", ("ADF statistic on the residuals of the first variable on the others; more negative is stronger "
                       "evidence of cointegration")),
        ("p-value", ("MacKinnon's for residual-based tests, for this number of variables and trend, or simulated for "
                     "this sample (critical values = simulated, with its Monte Carlo standard error); a small one "
                     "rejects no cointegration")),
        ("critical value 1%, 5%, 10%", ("reject no cointegration when the statistic is below one; further out than "
                                        "ordinary ADF values")),
        ("Engle-Granger (y on x)", "the title gives the normalisation: reversing the order can change the result"),
    )

    wraps: ClassVar[str] = "statsmodels.tsa.stattools.coint"

    variables: Annotated[str, Widget("", help="the series, comma-separated: the first is regressed on the others and the residuals tested")]
    trend: Annotated[Literal["c", "ct", "ctt", "n"], Widget("", help="terms in the levels regression: c constant, ct adds a trend, ctt a quadratic, n none")] = "c"
    autolag: Annotated[Literal["aic", "bic", "t-stat"], advanced(help="how the residuals' ADF lags are chosen: aic, bic, or t-stat (drop the last until significant)")] = "aic"
    max_lags: Annotated[str | int, advanced(help="largest lag of the residuals' ADF regression searched; empty = 12*(n/100)^(1/4), capped in short samples to keep 4 residual degrees of freedom after the levels regression")] = ""
    backend_options: Annotated[BackendOptions, Widget("", advanced=True, help="other statsmodels arguments")] = field(default_factory=dict)
    critical_values: Annotated[Literal["mackinnon", "simulated"], Widget("", help=_CRITICAL_HELP)] = "mackinnon"
    draws: Annotated[int, advanced(help=_DRAWS_HELP)] = 2000
    seed: Annotated[int, advanced(help=_SEED_HELP)] = 0
    exog: Annotated[str, advanced(help=("fixed regressors in the levels regression that are not I(1), comma-separated "
                                        "(impulse or break dummies); they enter every simulated regression as given. "
                                        "Needs critical values = simulated"))] = ""

    adapter_spec: ClassVar[AdapterSpec] = statsmodels_adapter.ENGLE_GRANGER

    name = "engle_granger"
    topics = ("unit-roots",)
    references = (
        ("Engle, R. F. and Granger, C. W. J. (1987). Co-integration and error correction: representation, estimation, and testing. Econometrica 55(2), 251-276. doi:10.2307/1913236"),
        ("MacKinnon, J. G. (1996). Numerical distribution functions for unit root and cointegration tests. Journal of Applied Econometrics 11(6), 601-618. doi:10.1002/(sici)1099-1255(199611)11:6<601::aid-jae417>3.0.co;2-t"),
    )
    null = "no cointegration"
    applies_to = (Dataset,)

    def __post_init__(self) -> None:
        check_backend_options(self)
        _check_simulation_settings(self)
        if self.exog.strip() and self.critical_values != "simulated":
            raise SpecificationError("exog needs critical_values='simulated': MacKinnon's values do not allow for "
                                     "extra regressors in the levels regression")

    def run(self, data: Dataset) -> TestResult:
        names = [s.strip() for s in self.variables.split(",") if s.strip()]
        if len(names) < 2:
            raise SpecificationError("give two or more variables; the first is the dependent variable")
        if data.kind != "time_series":
            raise SpecificationError("Engle-Granger needs a single time series")
        if self.critical_values == "simulated":
            return self._run_simulated(data, names)
        stattools = require("statsmodels.tsa.stattools")
        frame = data.frame[names].dropna()
        used = len(names) - 1 + (0 if self.trend == "n" else len(self.trend))  # the levels regression's columns
        maxlag = _adf_max_lag(len(frame), 0, self.max_lags, used)
        with statsmodels_adapter.ADAPTER.run(self.adapter_spec, self) as run, warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            stat, p, crit = run.call("test", stattools.coint, frame[names[0]].to_numpy(float), frame[names[1:]].to_numpy(float),
                                     trend=self.trend, maxlag=maxlag, autolag=self.autolag)
        from . import adf_arrays

        adf_arrays.degenerate_statistic(float(stat), f"Engle-Granger ({names[0]} on {', '.join(names[1:])})")
        return TestResult(f"Engle-Granger ({names[0]} on {', '.join(names[1:])})", self.null, float(stat), None, float(p), "MacKinnon",
                          critical_values=dict(zip(("1%", "5%", "10%"), map(float, crit))), backend_call=run.record)

    def _run_simulated(self, data: Dataset, names: list[str]) -> TestResult:
        from . import adf_arrays

        exog = [s.strip() for s in self.exog.split(",") if s.strip()]
        missing = [n for n in names + exog if n not in data.frame.columns]
        if missing:
            raise SpecificationError(f"not in the data: {', '.join(missing)}")
        if set(exog) & set(names):
            raise SpecificationError("a fixed regressor (exog) cannot also be one of the variables tested")
        frame = data.frame[names + exog].dropna()
        y, X = frame[names[0]].to_numpy(float), frame[names[1:]].to_numpy(float)
        E = frame[exog].to_numpy(float) if exog else None
        n = len(frame)
        design = np.column_stack([X, adf_arrays.trend_terms(n, self.trend)] + ([E] if exog else []))
        if n <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
            raise SpecificationError("the levels regression's columns are collinear or too many for the sample "
                                     "(does an exog column repeat the constant or the trend?)")
        rule = {"autolag": self.autolag, "max_lags": _adf_max_lag(n, 0, self.max_lags, design.shape[1])}
        fit = adf_arrays.engle_granger(y, X, self.trend, **rule, exog=E)
        adf_arrays.degenerate_statistic(float(fit.statistic[0]), f"Engle-Granger ({names[0]} on {', '.join(names[1:])})")
        sims = adf_arrays.simulate_engle_granger(n, len(names), self.trend, **rule, exog=E, draws=self.draws, seed=self.seed)
        used, stat = int(fit.lags[0]), float(fit.statistic[0])
        details: dict[str, Any] = {"lags used": used, "observations": n}
        if exog:
            details["fixed regressors"] = ", ".join(exog)
        warning = ""
        if stat == -np.inf:
            warning = "the series are (almost) perfectly collinear: the residuals are zero and the test is not informative"
        null_text = (f"{int(self.draws)} draws of {len(names)} independent Gaussian random walks of {n} observations, trend "
                     f"{self.trend}" + (f", with {', '.join(exog)} fixed in the levels regression" if exog else ""))
        return _simulated_result(self, data, f"Engle-Granger ({names[0]} on {', '.join(names[1:])})", stat, sims, details,
                                 null_text, warning)


@register
@dataclass(frozen=True)
class PhillipsOuliaris(Test):
    """Phillips-Ouliaris residual-based cointegration tests (Zt, Za, Pu, Pz), computed by arch.

    Zt and Za correct the Dickey-Fuller statistics of the cointegrating regression's residuals for serial
    correlation nonparametrically (a kernel long-run variance), with no lag choice; Pu and Pz are variance-ratio
    and trace statistics. Pz does not depend on which variable is normalised as dependent.

    The null is no cointegration; Zt and Za reject when small (large and negative), Pu and Pz when large, as the
    details say. Their appeal over Engle-Granger is that no lag length need be chosen; the price is that kernel
    corrections of this kind can reject too often in small samples when the errors have a large negative moving-average
    component, and that the kernel and bandwidth then matter. As with Engle-Granger, Zt, Za and Pu can depend on which
    variable is first, and none of them counts the cointegrating relations. With the bandwidth left empty, arch
    chooses it, and arch 9.0 chooses it differently from earlier versions (Newey and West's rule as R's sandwich has
    it); the details record arch's version and the bandwidth used."""

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import Dataset

        data = Dataset.example("sample_macro").derive(lc="log(c)", ly="log(y)")
        data.test("phillips_ouliaris", "lc, ly")                 # Zt: rejects no cointegration when small; no lag length to choose
        data.test("phillips_ouliaris", "lc, ly", test_type="Pz") # Pz: rejects when large, whichever variable comes first
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("statistic", "Zt or Za, or Pu or Pz, as the title says; the null is no cointegration"),
        ("rejects for", "small values for Zt and Za, large for Pu and Pz: read the critical values in that direction"),
        ("p-value, critical value 1%, 5%, 10%", ("from response surfaces for this number of variables and trend; a "
                                                 "small p-value rejects no cointegration")),
        ("kernel, bandwidth", ("the long-run variance estimate behind the correction; check that a different "
                               "bandwidth does not change the conclusion")),
        ("arch version", ("the arch release that computed the test: arch 9.0 changed the automatic bandwidth, so "
                          "with it left empty the statistic depends on the version; give a bandwidth to repeat a result")),
        ("observations", "the rows with every variable present"),
    )

    variables: Annotated[str, Widget("", help="the series, comma-separated; the first is the dependent variable (Pz is the same in any order)")]
    trend: Annotated[Literal["c", "ct", "ctt", "n"], Widget("", help="terms in the cointegrating regression: c constant, ct adds a trend, ctt a quadratic, n none")] = "c"
    test_type: Annotated[Literal["Zt", "Za", "Pu", "Pz"], Widget(kind="", help="Zt and Za: corrected Dickey-Fuller on the residuals; Pu: variance ratio; Pz: trace, the same whichever variable is first")] = "Zt"
    kernel: Annotated[Literal["bartlett", "parzen", "qs"], advanced(help="kernel for the long-run variance (qs: quadratic spectral)")] = "bartlett"
    bandwidth: Annotated[str | int, advanced(help="kernel bandwidth; empty = chosen from the data")] = ""
    backend_options: Annotated[BackendOptions, Widget("", advanced=True, help="further arguments for arch's phillips_ouliaris, e.g. force_int; not those set above")] = field(default_factory=dict)

    adapter_spec: ClassVar[AdapterSpec] = arch_adapter.PHILLIPS_OULIARIS

    name = "phillips_ouliaris"
    topics = ("unit-roots",)
    references = (
        ("Phillips, P. C. B. and Ouliaris, S. (1990). Asymptotic properties of residual based tests for cointegration. Econometrica 58(1), 165-193. doi:10.2307/2938339"),
    )
    null = "no cointegration"
    applies_to = (Dataset,)

    def __post_init__(self) -> None:
        check_backend_options(self)

    def run(self, data: Dataset) -> TestResult:
        coint = require("arch.unitroot.cointegration")
        names = [s.strip() for s in self.variables.split(",") if s.strip()]
        if len(names) < 2:
            raise SpecificationError("give two or more variables; the first is the dependent variable")
        if data.kind != "time_series":
            raise SpecificationError("Phillips-Ouliaris needs a single time series")
        frame = data.frame[names].dropna()
        bandwidth = _int_or(self.bandwidth, ())
        with arch_adapter.ADAPTER.run(self.adapter_spec, self) as run:
            r = run.call("test", coint.phillips_ouliaris, frame[names[0]].to_numpy(float), frame[names[1:]].to_numpy(float),
                         trend=self.trend, test_type=self.test_type, kernel=self.kernel, bandwidth=bandwidth)
            stat, p = float(r.stat), float(r.pvalue)
        crit = {f"{int(q)}%": float(v) for q, v in sorted(r.critical_values.items())}
        return TestResult(f"Phillips-Ouliaris {self.test_type} ({names[0]} on {', '.join(names[1:])})", self.null, stat, None,
                          p, "Phillips-Ouliaris (MacKinnon-style response surfaces)", critical_values=crit,
                          details={"kernel": r.kernel, "bandwidth": float(r.bandwidth), "observations": len(frame),
                                   "rejects for": "large values" if self.test_type in ("Pu", "Pz") else "small values",
                                   "arch version": run.record.version},
                          backend_call=run.record)
