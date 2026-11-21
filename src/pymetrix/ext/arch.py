"""Reference plugin: volatility models through the arch package.

EGARCH, TARCH, FIGARCH and APARCH, and GARCH with skewed-t or GED errors (or
``backend="arch"``). Bundled with pymetrix for now; written against the
Extension API only, so it can move to its own package (``pymetrix-arch``).

Our design matrix becomes arch's least-squares mean model (``LS`` without its
own constant: the regressors already carry ours); arch estimates; results come
back as a ``FittedEquation`` with the conditional volatility and standardised
residuals in ``extra``. Forecasts refit nothing: the model is rebuilt on the
data up to the forecast origin with the estimated parameters fixed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd

from ..errors import SpecificationError
from .._version import __version__
from ..adapters import arch as specs
from ..adapters import require
from ..adapters.base import AdapterSpec, BackendOptions, check_backend_options, with_backend
from ..estimation.design import LinearDesign, provenance
from ..estimation.fitted import FittedEquation
from ..estimation.params import point
from ..estimation.timeseries import Dist
from ..data import Dataset
from ..forms import Widget, advanced
from ..plugins import PluginInfo, register
from ..stats.base import Test, TestResult
from ..stats.base import register as register_test
from ..stats.data_tests import _int_or, _series

if TYPE_CHECKING:
    from ..estimation.equation import Equation

PLUGIN = PluginInfo(
    name="pymetrix-arch",
    version=__version__,
    requires={"arch": ">=7"},
    license="BSD-3-Clause",
    maintainer="Geoffrey Heenan",
    description="EGARCH, TARCH, FIGARCH and APARCH volatility models; skewed-t and GED errors; unit-root tests",
    install="pip install pymetrix[arch]",
)

_ARCH = Widget("number", min=0, max=10, help="lags of the shocks (ARCH terms)")
_ASYM = Widget("number", min=0, max=10, help="asymmetric terms: extra response to negative shocks")
_GARCH = Widget("number", min=0, max=10, help="lags of the conditional variance (GARCH terms)")


_BACKEND_OPTIONS = Widget("", advanced=True, help="other arch arguments: fit options such as tol=1e-8 (model.x for the "
                                                  "volatility process)")


class _ArchVariance:
    """Shared behaviour: estimated and forecast by arch."""

    adapter_spec: ClassVar[AdapterSpec]
    requires: ClassVar[tuple[str, ...]] = ("arch",)
    libraries: ClassVar[tuple[str, ...]] = ("arch",)
    docs: ClassVar[str] = "https://bashtage.github.io/arch/univariate/volatility.html"
    topics: ClassVar[tuple[str, ...]] = ("garch",)

    def __post_init__(self) -> None:
        check_backend_options(self)

    def arch_volatility(self, **model_options: Any) -> Any:
        raise NotImplementedError

    def fit(self, method: Any, eq: Equation, data: Dataset, design: LinearDesign) -> FittedEquation:
        return fit_volatility(method, eq, data, design)

    def forecast(self, fit: FittedEquation, data: Dataset, span: tuple[Any, Any], dynamic: bool, alpha: float) -> Any:
        return forecast(fit, data, span, dynamic, alpha)


@dataclass(frozen=True)
class EGARCH(_ArchVariance):
    """Exponential GARCH (Nelson): the log variance responds to the size and sign of shocks."""

    maturity: ClassVar[str] = "stable"
    label: ClassVar[str] = "EGARCH"
    wraps: ClassVar[str] = "arch.univariate.EGARCH"
    adapter_spec: ClassVar[AdapterSpec] = specs.EGARCH
    references: ClassVar[tuple[str, ...]] = (
        ("Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: a new approach. Econometrica 59(2), 347-370. doi:10.2307/2938260"),
    )

    arch: Annotated[int, _ARCH] = 1
    asym: Annotated[int, _ASYM] = 1
    garch: Annotated[int, _GARCH] = 1
    dist: Dist = "normal"

    backend_options: Annotated[BackendOptions, _BACKEND_OPTIONS] = field(default_factory=dict)

    def arch_volatility(self, **model_options: Any) -> Any:
        return require("arch.univariate").EGARCH(p=self.arch, o=self.asym, q=self.garch, **model_options)


@dataclass(frozen=True)
class TARCH(_ArchVariance):
    """Threshold GARCH (Zakoian): the conditional standard deviation responds to absolute shocks."""

    maturity: ClassVar[str] = "stable"
    label: ClassVar[str] = "TARCH"
    wraps: ClassVar[str] = "arch.univariate.GARCH"
    adapter_spec: ClassVar[AdapterSpec] = specs.TARCH
    references: ClassVar[tuple[str, ...]] = (
        ("Zakoian, J.-M. (1994). Threshold heteroskedastic models. Journal of Economic Dynamics and Control 18(5), 931-955. doi:10.1016/0165-1889(94)90039-6"),
    )

    arch: Annotated[int, _ARCH] = 1
    asym: Annotated[int, _ASYM] = 1
    garch: Annotated[int, _GARCH] = 1
    dist: Dist = "normal"

    backend_options: Annotated[BackendOptions, _BACKEND_OPTIONS] = field(default_factory=dict)

    def arch_volatility(self, **model_options: Any) -> Any:
        return require("arch.univariate").GARCH(p=self.arch, o=self.asym, q=self.garch, power=1.0, **model_options)


@dataclass(frozen=True)
class FIGARCH(_ArchVariance):
    """Fractionally integrated GARCH (long memory in volatility). ``arch`` and ``garch`` are 0 or 1."""

    maturity: ClassVar[str] = "beta"
    label: ClassVar[str] = "FIGARCH"
    wraps: ClassVar[str] = "arch.univariate.FIGARCH"
    adapter_spec: ClassVar[AdapterSpec] = specs.FIGARCH
    references: ClassVar[tuple[str, ...]] = (
        ("Baillie, R. T., Bollerslev, T. and Mikkelsen, H. O. (1996). Fractionally integrated generalized autoregressive conditional heteroskedasticity. Journal of Econometrics 74(1), 3-30. doi:10.1016/s0304-4076(95)01749-6"),
    )

    arch: Annotated[int, Widget("number", min=0, max=1, help="ARCH term: 0 or 1")] = 1
    garch: Annotated[int, Widget("number", min=0, max=1, help="GARCH term: 0 or 1")] = 1
    power: Annotated[float, Widget("number", min=0.5, max=4, step=0.5, help="power of the shocks (2 = variance, 1 = standard deviation)")] = 2.0
    truncation: Annotated[int, Widget("number", min=10, max=100000, advanced=True, help="lags kept in the fractional filter")] = 1000
    dist: Dist = "normal"

    backend_options: Annotated[BackendOptions, _BACKEND_OPTIONS] = field(default_factory=dict)

    def arch_volatility(self, **model_options: Any) -> Any:
        return require("arch.univariate").FIGARCH(p=self.arch, q=self.garch, power=self.power, truncation=self.truncation,
                                                  **model_options)


@dataclass(frozen=True)
class APARCH(_ArchVariance):
    """Asymmetric power ARCH (Ding, Granger and Engle); the power ``delta`` is estimated unless given."""

    maturity: ClassVar[str] = "beta"
    label: ClassVar[str] = "APARCH"
    wraps: ClassVar[str] = "arch.univariate.APARCH"
    adapter_spec: ClassVar[AdapterSpec] = specs.APARCH
    references: ClassVar[tuple[str, ...]] = (
        ("Ding, Z., Granger, C. W. J. and Engle, R. F. (1993). A long memory property of stock market returns and a new model. Journal of Empirical Finance 1(1), 83-106. doi:10.1016/0927-5398(93)90006-d"),
    )

    arch: Annotated[int, _ARCH] = 1
    asym: Annotated[int, _ASYM] = 1
    garch: Annotated[int, _GARCH] = 1
    delta: Annotated[float | None, Widget("text", advanced=True, help="fixed power; empty = estimate")] = None
    dist: Dist = "normal"

    backend_options: Annotated[BackendOptions, _BACKEND_OPTIONS] = field(default_factory=dict)

    def arch_volatility(self, **model_options: Any) -> Any:
        return require("arch.univariate").APARCH(p=self.arch, o=self.asym, q=self.garch, delta=self.delta, **model_options)


def _volatility(spec: Any, run: Any = None) -> Any:
    if hasattr(spec, "arch_volatility"):
        return run.call("model", spec.arch_volatility) if run is not None else spec.arch_volatility()
    u = require("arch.univariate")  # core GARCH with backend="arch"
    return run.call("model", u.GARCH, p=spec.arch, o=spec.asym, q=spec.garch) if run is not None else u.GARCH(p=spec.arch, o=spec.asym, q=spec.garch)


def _distribution(name: str) -> Any:
    u = require("arch.univariate")
    return {"normal": u.Normal, "t": u.StudentsT, "skewt": u.SkewStudent, "ged": u.GeneralizedError}[name]()


def _model(y: np.ndarray, X: np.ndarray, names: list[str], spec: Any, run: Any = None) -> Any:
    u = require("arch.univariate")
    if names == ["const"]:  # a constant mean forecasts without future regressor values
        mean = u.ConstantMean(pd.Series(y, name="y"))
    elif X.shape[1]:
        mean = u.LS(pd.Series(y, name="y"), pd.DataFrame(X, columns=names), constant=False)
    else:
        mean = u.ZeroMean(pd.Series(y, name="y"))
    mean.volatility = _volatility(spec, run)
    mean.distribution = _distribution(spec.dist)
    return mean


def _cov_type(cov: Any) -> str:
    if cov == "classic":
        return "classic"
    if cov == "white":
        return "robust"
    raise SpecificationError("arch-backed volatility models support cov='classic' (inverse Hessian) or 'white' (Bollerslev-Wooldridge)")


def fit_volatility(method: Any, eq: Equation, data: Dataset, d: LinearDesign) -> FittedEquation:
    spec = eq.variance
    adapter_spec = getattr(spec, "adapter_spec", specs.GARCH)
    # arch's warnings (convergence, data scale) are kept with the result and still shown, no longer silenced
    with specs.ADAPTER.run(adapter_spec, spec) as run:
        model = _model(d.y, d.X, d.names, spec, run)
        res = run.call("fit", model.fit, cov_type=_cov_type(method.cov), options={"maxiter": max(method.max_iter, 100)})
    if res.convergence_flag != 0:
        message = getattr(getattr(res, "optimization_result", None), "message", "")
        run.message(f"arch: the optimiser did not converge (convergence_flag {res.convergence_flag}){f': {message}' if message else ''}")
    names = ["const" if n == "mu" else n for n in res.params.index]
    sigma = np.asarray(res.conditional_volatility, dtype=float)
    resid = np.asarray(res.resid, dtype=float)
    prov = provenance(eq, method, data, d.index, d.dropped)
    prov = with_backend(prov, run.record, plugin=f"{PLUGIN.name} {PLUGIN.version}")
    stats = {"nobs": float(res.nobs), "k": float(len(names)), "loglike": float(res.loglikelihood), "aic": float(res.aic), "bic": float(res.bic)}
    if res.convergence_flag != 0:
        stats["convergence_flag"] = float(res.convergence_flag)
    return FittedEquation(
        equation=eq,
        params=point(names, res.params.to_numpy(float), res.param_cov.to_numpy(float)),
        method=method,
        resid=pd.Series(resid, index=d.index, name="resid"),
        fitted=pd.Series(d.y - resid, index=d.index, name="fitted"),
        stats=stats,
        provenance=prov,
        design=pd.DataFrame(d.X, index=d.index, columns=d.names),
        extra={
            "converged": int(getattr(res, "convergence_flag", 0)) == 0,
            "volatility": pd.Series(sigma, index=d.index, name="volatility"),
            "std_resid": pd.Series(resid / sigma, index=d.index, name="std_resid"),
            "regressors": d.names,
        },
        backend_result=res,
    )


def forecast(fit: FittedEquation, data: Dataset, span: tuple[Any, Any], dynamic: bool, alpha: float,
             simulations: int = 2000, seed: int = 0) -> Dataset:
    """Static (one step ahead) or dynamic forecasts with the estimated parameters held fixed."""
    from ..data import Dataset
    from ..expr import evaluate
    from ..metadata import DatasetMeta

    eq = fit.equation
    params = pd.Series(np.asarray(fit.params.values, float))  # arch's own order: the mean, then the variance and distribution
    names = fit.extra["regressors"]
    if any(n != "const" for n in names):
        raise NotImplementedError("forecasts for arch-backed volatility models need a mean of a constant only (e.g. 'r ~ 1')")
    freq = data.frame.index.freq
    from ..data.sample import as_span
    span = as_span(span)  # '1990Q1:2019Q4' or a pair
    start, end = pd.Period(str(span[0]), freq=freq), pd.Period(str(span[1]), freq=freq)
    first = pd.Period(fit.provenance.sample[0], freq=freq)  # type: ignore[union-attr]
    if start <= first:
        raise SpecificationError(f"the forecast span must start after the first estimation period, {first}")
    periods = pd.period_range(start, end, freq=freq, name=data.time_axis.name)  # type: ignore[union-attr]
    history_end = start - 1 if dynamic else end - 1
    hist = pd.period_range(first, history_end, freq=freq, name=data.time_axis.name)  # type: ignore[union-attr]
    history = Dataset(data.frame.reindex(hist), data.axes, data.variables)
    y = np.asarray(evaluate(eq.lhs, history.lagged), dtype=float)
    y = np.broadcast_to(y, (len(hist),))
    if not np.all(np.isfinite(y)):
        raise SpecificationError("the data before the forecast origin has gaps")
    X = np.ones((len(y), len(names)))
    variance = fit.equation.variance
    with specs.ADAPTER.run(getattr(variance, "adapter_spec", specs.GARCH), variance) as run:  # the model as fitted, backend options too
        model = _model(y, X, names, variance, run)
    fixed = model.fix(params.to_numpy())
    h = len(periods) if dynamic else 1
    origin = len(y) - 1 if dynamic else len(y) - len(periods)
    method = "analytic"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fc = fixed.forecast(horizon=h, start=origin, reindex=False, method=method)
        except ValueError:  # e.g. EGARCH or TARCH beyond one step: simulate from the model's own distribution
            method = "simulation"
            fc = fixed.forecast(horizon=h, start=origin, reindex=False, method=method, simulations=simulations,
                                random_state=np.random.RandomState(seed))
    if dynamic:
        mean, var = fc.mean.iloc[-1].to_numpy(), fc.variance.iloc[-1].to_numpy()
    else:
        mean, var = fc.mean.iloc[:, 0].to_numpy(), fc.variance.iloc[:, 0].to_numpy()
    dist = model.distribution
    dist_params = params[len(params) - dist.num_params:].to_numpy() if dist.num_params else None
    lo, hi = dist.ppf(np.array([alpha / 2, 1 - alpha / 2]), dist_params)
    se = np.sqrt(var)
    actual = np.asarray(evaluate(eq.lhs, data.lagged), dtype=float)
    actual = pd.Series(np.broadcast_to(actual, (len(data.frame),)), index=data.frame.index).reindex(periods).to_numpy()
    frame = pd.DataFrame({"forecast": mean, "se": se, "variance": var, "lower": mean + lo * se, "upper": mean + hi * se, "actual": actual}, index=periods)
    kind = "Dynamic" if dynamic else "Static (one-step-ahead)"
    meta = DatasetMeta(title=f"{kind} forecast: {eq.text}",
                       description=f"arch {method} forecasts; intervals reflect shock uncertainty only, not parameter uncertainty.")
    return Dataset(frame, data.axes, meta=meta)


# unit-root tests


def _arch_result(name: str, null: str, res: Any, **details: Any) -> TestResult:
    """Read inside the adapter's run: arch computes when the statistic is first read."""
    return TestResult(name, null, float(res.stat), None, float(res.pvalue), "MacKinnon / arch",
                      critical_values={k: float(v) for k, v in res.critical_values.items()},
                      details={"lags used": res.lags, "observations": res.nobs, **details})


_BACKEND_OPTIONS = Widget("", advanced=True, help="other arch arguments")


class _ArchTest(Test):
    """An arch unit-root test: its ``backend_options`` checked, and the run's record moved into provenance."""

    def __post_init__(self) -> None:
        check_backend_options(self)


@dataclass(frozen=True)
class PhillipsPerron(_ArchTest):
    """Phillips-Perron unit-root test (arch): ADF regression with a long-run variance correction."""

    maturity: ClassVar[str] = "stable"
    wraps: ClassVar[str] = "arch.unitroot.PhillipsPerron"
    docs: ClassVar[str] = "https://bashtage.github.io/arch/unitroot/unitroot.html"
    topics: ClassVar[tuple[str, ...]] = ("unit-roots",)
    references: ClassVar[tuple[str, ...]] = (
        ("Phillips, P. C. B. and Perron, P. (1988). Testing for a unit root in time series regression. Biometrika 75(2), 335-346. doi:10.1093/biomet/75.2.335"),
    )
    requires: ClassVar[tuple[str, ...]] = ("arch",)

    variable: str
    trend: Literal["c", "ct", "n"] = "c"
    test_type: Annotated[Literal["tau", "rho"], advanced(help="t-statistic (tau) or normalised bias (rho)")] = "tau"
    lags: Annotated[str | int, advanced(help="Newey-West lags; empty = 12*(n/100)^(1/4)")] = ""
    backend_options: Annotated[BackendOptions, _BACKEND_OPTIONS] = field(default_factory=dict)

    adapter_spec: ClassVar[AdapterSpec] = specs.PHILLIPS_PERRON

    name = "phillips_perron"
    null = "the series has a unit root"
    applies_to = (Dataset,)

    def run(self, data: Dataset) -> TestResult:
        unitroot = require("arch.unitroot")
        x = _series(data, self.variable, "Phillips-Perron")
        with specs.ADAPTER.run(self.adapter_spec, self) as run:
            res = run.call("test", unitroot.PhillipsPerron, x, lags=_int_or(self.lags, ()), trend=self.trend, test_type=self.test_type)
            out = _arch_result(f"Phillips-Perron ({self.variable})", self.null, res)
        return replace(out, backend_call=run.record)


@dataclass(frozen=True)
class ZivotAndrews(_ArchTest):
    """Zivot-Andrews unit-root test allowing one structural break at an unknown date (arch)."""

    maturity: ClassVar[str] = "stable"
    wraps: ClassVar[str] = "arch.unitroot.ZivotAndrews"
    docs: ClassVar[str] = "https://bashtage.github.io/arch/unitroot/unitroot.html"
    topics: ClassVar[tuple[str, ...]] = ("unit-roots",)
    references: ClassVar[tuple[str, ...]] = (
        ("Zivot, E. and Andrews, D. W. K. (1992). Further evidence on the great crash, the oil-price shock, and the unit-root hypothesis. Journal of Business and Economic Statistics 10(3), 251-270. doi:10.1080/07350015.1992.10509904"),
    )
    requires: ClassVar[tuple[str, ...]] = ("arch",)

    variable: str
    trend: Literal["c", "t", "ct"] = "c"
    trim: Annotated[float, advanced(help="share of the sample excluded at each end when searching for the break", min=0.05, max=0.3, step=0.05)] = 0.15
    lags: Annotated[str | int, advanced(help="fixed lags; empty = choose by method")] = ""
    method: Annotated[Literal["aic", "bic", "t-stat"], advanced(help="how lags are chosen: information criterion or significance of the last lag")] = "aic"
    max_lags: Annotated[str | int, advanced(help="largest lag searched; empty = default")] = ""
    backend_options: Annotated[BackendOptions, _BACKEND_OPTIONS] = field(default_factory=dict)

    adapter_spec: ClassVar[AdapterSpec] = specs.ZIVOT_ANDREWS

    name = "zivot_andrews"
    null = "the series has a unit root (with a single structural break)"
    applies_to = (Dataset,)

    def run(self, data: Dataset) -> TestResult:
        unitroot = require("arch.unitroot")
        s = data[self.variable].dropna()
        with specs.ADAPTER.run(self.adapter_spec, self) as run:
            res = run.call("test", unitroot.ZivotAndrews, s.to_numpy(float), lags=_int_or(self.lags, ()), trend=self.trend,
                           trim=self.trim, max_lags=_int_or(self.max_lags, ()), method=self.method)
            details: dict[str, Any] = {}
            _ = res.stat  # arch computes lazily; this fills the per-break statistics
            stats = np.asarray(specs.zivot_andrews_all_stats(res), dtype=float)  # one statistic per candidate break date
            if np.isfinite(stats).any():
                details["break period"] = str(s.index[int(np.nanargmin(stats))])
            out = _arch_result(f"Zivot-Andrews ({self.variable})", self.null, res, **details)
        return replace(out, backend_call=run.record)


@dataclass(frozen=True)
class DFGLS(_ArchTest):
    """Elliott-Rothenberg-Stock DF-GLS unit-root test (arch): ADF on GLS-detrended data, more powerful."""

    maturity: ClassVar[str] = "stable"
    wraps: ClassVar[str] = "arch.unitroot.DFGLS"
    docs: ClassVar[str] = "https://bashtage.github.io/arch/unitroot/unitroot.html"
    topics: ClassVar[tuple[str, ...]] = ("unit-roots",)
    references: ClassVar[tuple[str, ...]] = (
        ("Elliott, G., Rothenberg, T. J. and Stock, J. H. (1996). Efficient tests for an autoregressive unit root. Econometrica 64(4), 813-836. doi:10.2307/2171846"),
    )
    requires: ClassVar[tuple[str, ...]] = ("arch",)

    variable: str
    trend: Literal["c", "ct"] = "c"
    lags: Annotated[str | int, advanced(help="fixed lags; empty = choose by method")] = ""
    method: Annotated[Literal["aic", "bic", "t-stat"], advanced(help="how lags are chosen: information criterion or significance of the last lag")] = "aic"
    max_lags: Annotated[str | int, advanced(help="largest lag searched; empty = default")] = ""

    backend_options: Annotated[BackendOptions, _BACKEND_OPTIONS] = field(default_factory=dict)

    adapter_spec: ClassVar[AdapterSpec] = specs.DFGLS

    name = "dfgls"
    null = "the series has a unit root"
    applies_to = (Dataset,)

    def run(self, data: Dataset) -> TestResult:
        unitroot = require("arch.unitroot")
        x = _series(data, self.variable, "DF-GLS")
        with specs.ADAPTER.run(self.adapter_spec, self) as run:
            res = run.call("test", unitroot.DFGLS, x, lags=_int_or(self.lags, ()), trend=self.trend,
                           max_lags=_int_or(self.max_lags, ()), method=self.method)
            out = _arch_result(f"DF-GLS ({self.variable})", self.null, res)
        return replace(out, backend_call=run.record)


@dataclass(frozen=True)
class VarianceRatio(_ArchTest):
    """Lo-MacKinlay variance-ratio test of a random walk (arch)."""

    maturity: ClassVar[str] = "stable"
    wraps: ClassVar[str] = "arch.unitroot.VarianceRatio"
    docs: ClassVar[str] = "https://bashtage.github.io/arch/unitroot/unitroot.html"
    topics: ClassVar[tuple[str, ...]] = ("unit-roots",)
    references: ClassVar[tuple[str, ...]] = (
        ("Lo, A. W. and MacKinlay, A. C. (1988). Stock market prices do not follow random walks: evidence from a simple specification test. Review of Financial Studies 1(1), 41-66. doi:10.1093/rfs/1.1.41"),
    )
    requires: ClassVar[tuple[str, ...]] = ("arch",)

    variable: str
    lags: int = 2
    trend: Literal["c", "n"] = "c"
    debiased: Annotated[bool, advanced()] = True
    robust: Annotated[bool, advanced(help="heteroskedasticity-robust statistic")] = True
    overlap: Annotated[bool, advanced(help="use overlapping blocks")] = True

    backend_options: Annotated[BackendOptions, _BACKEND_OPTIONS] = field(default_factory=dict)

    adapter_spec: ClassVar[AdapterSpec] = specs.VARIANCE_RATIO

    name = "variance_ratio"
    null = "the series is a random walk"
    applies_to = (Dataset,)

    def run(self, data: Dataset) -> TestResult:
        unitroot = require("arch.unitroot")
        x = _series(data, self.variable, "Variance ratio")
        with specs.ADAPTER.run(self.adapter_spec, self) as run:
            res = run.call("test", unitroot.VarianceRatio, x, lags=self.lags, trend=self.trend, debiased=self.debiased,
                           robust=self.robust, overlap=self.overlap)
            out = _arch_result(f"Variance ratio ({self.variable})", self.null, res, **{"variance ratio": float(res.vr)})
        return replace(out, backend_call=run.record)


for _cls in (EGARCH, TARCH, FIGARCH, APARCH):
    register("variances", _cls, plugin=PLUGIN)
for _cls in (PhillipsPerron, ZivotAndrews, DFGLS, VarianceRatio):
    register_test(_cls, plugin=PLUGIN)
