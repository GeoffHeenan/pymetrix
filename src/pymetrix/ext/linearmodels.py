"""Reference plugin: IV and panel estimators backed by linearmodels.

Bundled with pymetrix for now; written against the Extension API only, so it
can move to its own package (``pymetrix-linearmodels``) unchanged.

Our formula parsing, sample selection and missing-value handling produce the
arrays; linearmodels estimates; results come back as ``FittedEquation``.
Covariance choices map onto linearmodels': "classic" -> unadjusted,
"white" -> robust, "hac" -> kernel (Bartlett), ``Cluster(...)`` -> clustered.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd

from ..errors import SpecificationError
from .._version import __version__
from ..adapters import require
from ..adapters import linearmodels as specs
from ..adapters.base import AdapterSpec, BackendCall, BackendOptions, check_backend_options, with_backend
from ..estimation.covariance import Cluster
from ..estimation.design import LinearDesign, instrument_series, linear_design, provenance
from ..estimation.fitted import FittedEquation
from ..estimation.params import point
from ..forms import Widget
from ..plugins import PluginInfo, register

if TYPE_CHECKING:
    from ..data import Dataset
    from ..estimation.equation import Equation

PLUGIN = PluginInfo(
    name="pymetrix-linearmodels",
    version=__version__,
    requires={"linearmodels": ">=6"},
    license="BSD-3-Clause",
    maintainer="Geoffrey Heenan",
    description="LIML, IV-GMM and panel estimators (random effects, between, first difference, Fama-MacBeth)",
    install="pip install pymetrix[panel]",
)


def _adv(help: str = "", **kw: Any) -> Widget:
    return Widget("", help=help, advanced=True, **kw)


_BACKEND_OPTIONS = Widget("", advanced=True, help="other linearmodels arguments, e.g. fit.cov_type options; model.x for the constructor")


class _Wrapped:
    """What every linearmodels method shares: its adapter spec, and backend_options checked on creation."""

    adapter_spec: ClassVar[AdapterSpec]

    def __post_init__(self) -> None:
        check_backend_options(self)


def _check(eq: Equation, method: str) -> None:
    from ..estimation.families import Gaussian

    if eq.form != "linear":
        raise SpecificationError(f"{method} needs a linear-form equation")
    if not isinstance(eq.family, Gaussian) or eq.has_error_model:
        raise SpecificationError(f"{method} needs a Gaussian equation without ARMA or GARCH terms")


def _cov(cov: Any, hac_lags: int | None, data: Dataset, index: pd.Index, panel: bool) -> tuple[str, dict[str, Any]]:
    if cov == "classic":
        return "unadjusted", {}
    if cov == "white":
        return "robust", {}
    if cov == "hac":
        config: dict[str, Any] = {"kernel": "bartlett"}
        if hac_lags:
            config["bandwidth"] = hac_lags
        return "kernel", config
    if isinstance(cov, Cluster):
        if panel:
            entity, time = data.frame.index.names
            if set(cov.levels) <= {entity, time}:
                return "clustered", {"cluster_entity": entity in cov.levels, "cluster_time": time in cov.levels}
        codes = pd.DataFrame({lvl: pd.factorize(data.values_of(lvl).reindex(index))[0] for lvl in cov.levels})
        codes.index = index if panel else pd.RangeIndex(len(index))
        return "clustered", {"clusters": codes}
    raise SpecificationError(f"unknown covariance {cov!r}")


def _result(eq: Equation, method: Any, data: Dataset, d: LinearDesign, res: Any, index: pd.Index, extra: dict[str, Any],
            stats: dict[str, float], names: list[str], X: np.ndarray, record: BackendCall, variables: Sequence[str] = ()) -> FittedEquation:
    params = res.params.reindex(names)
    cov = res.cov.reindex(index=names, columns=names).to_numpy()
    resid = pd.Series(np.asarray(res.resids, dtype=float), index=index, name="resid")
    values = np.asarray(res.fitted_values, dtype=float).ravel()
    if len(values) == len(index):
        fitted = pd.Series(values, index=index, name="fitted")
    else:  # between estimator: residuals are per entity, so fitted = entity mean - residual
        means = pd.Series(d.y_observed, index=d.index).groupby(level=0).mean()
        fitted = (means.reindex(index) - resid).rename("fitted")
    prov = provenance(eq, method, data, d.index, d.dropped, extra_variables=variables)
    prov = with_backend(prov, record, plugin=f"{PLUGIN.name} {PLUGIN.version}")
    return FittedEquation(eq, point(names, params.to_numpy(float), cov), method, resid, fitted, stats, prov,
                          pd.DataFrame(X, index=index, columns=names), extra=extra, backend_result=res)


# instrumental variables


def _iv_data(m: Any, eq: Equation, data: Dataset, span, where, weights, missing):
    labels = [t.label for t in eq.terms]
    unknown = [e for e in m.endog if e not in labels]
    if unknown:
        raise SpecificationError(f"endog {unknown} must name regressors of the equation: {labels}")
    if eq.fixed_effects:
        raise SpecificationError("fe() terms are not supported with linearmodels IV estimators; use IV()")
    d = linear_design(eq, data, span, where, weights, missing, instruments=instrument_series(m.instruments, data))
    rows = pd.RangeIndex(len(d.y))
    X = pd.DataFrame(d.X, index=rows, columns=d.names)
    exog = X[[n for n in d.names if n not in m.endog]]
    endog = X[list(m.endog)]
    inst = pd.DataFrame(d.Z, index=rows, columns=d.z_names)
    w = pd.Series(d.w, index=rows) if d.w is not None else None
    return d, pd.Series(d.y, index=rows), exog if exog.shape[1] else None, endog, inst, w


def _iv_stats(res: Any, n: int, k: int) -> dict[str, float]:
    stats = {"nobs": float(n), "k": float(k), "df_resid": float(res.df_resid), "r2": float(res.rsquared), "r2_adj": float(res.rsquared_adj),
             "se_regression": float(np.sqrt(res.s2))}
    return stats


def _first_stage(res: Any) -> dict[str, float]:
    try:
        diag = res.first_stage.diagnostics
        return {str(k): float(v) for k, v in diag["f.stat"].items()}
    except Exception:  # first-stage diagnostics are informative, not essential
        return {}


@dataclass(frozen=True)
class LIML(_Wrapped):
    """Limited-information maximum likelihood (or k-class with ``kappa``; Fuller with ``fuller``)."""

    maturity: ClassVar[str] = "stable"
    label: ClassVar[str] = "LIML"
    wraps: ClassVar[str] = "linearmodels.iv.IVLIML"
    docs: ClassVar[str] = "https://bashtage.github.io/linearmodels/"
    topics: ClassVar[tuple[str, ...]] = ('instrumental-variables',)
    references: ClassVar[tuple[str, ...]] = (
        ("Anderson, T. W. and Rubin, H. (1949). Estimation of the parameters of a single equation in a complete system of stochastic equations. Annals of Mathematical Statistics 20(1), 46-63. doi:10.1214/aoms/1177730090"),
        ("Fuller, W. A. (1977). Some properties of a modification of the limited information estimator. Econometrica 45(4), 939-953. doi:10.2307/1912683"),
    )
    kind: ClassVar[str] = "iv"
    requires: ClassVar[tuple[str, ...]] = ("linearmodels",)
    adapter_spec: ClassVar[AdapterSpec] = specs.LIML

    endog: Sequence[str]
    instruments: Sequence[str]
    cov: str | Cluster = "classic"
    hac_lags: int | None = None
    fuller: Annotated[float, _adv("Fuller's alpha (0 = LIML, 1 is common)", min=0, max=10, step=1)] = 0.0
    kappa: Annotated[float | None, _adv("fixed k-class kappa; empty = estimate (LIML)")] = None
    debiased: Annotated[bool, _adv("small-sample (n-k) correction of the covariance")] = True

    backend_options: Annotated[BackendOptions, _BACKEND_OPTIONS] = field(default_factory=dict)

    def estimate(self, eq: Equation, data: Dataset, *, span=None, where=None, weights=None, missing="drop") -> FittedEquation:
        _check(eq, "LIML")
        iv = require("linearmodels.iv")
        d, y, exog, endog, inst, w = _iv_data(self, eq, data, span, where, weights, missing)
        cov_type, config = _cov(self.cov, self.hac_lags, data, d.index, panel=False)
        kappa = None if self.kappa in (None, "") else float(self.kappa)
        with specs.ADAPTER.run(self.adapter_spec, self) as run:
            model = run.call("model", iv.IVLIML, y, exog, endog, inst, weights=w, fuller=self.fuller, kappa=kappa)
            res = run.call("fit", model.fit, cov_type=cov_type, debiased=self.debiased, **config)
        stats = {**_iv_stats(res, len(d.y), len(d.names)), "kappa": float(res.kappa)}
        extra = {"first_stage_F": _first_stage(res), "instrument_names": d.z_names}
        return _result(eq, self, data, d, res, d.index, extra, stats, d.names, d.X, run.record, _instrument_vars(self.instruments))


@dataclass(frozen=True)
class IVGMM(_Wrapped):
    """Efficient GMM with instruments; reports Hansen's J test of overidentifying restrictions."""

    maturity: ClassVar[str] = "stable"
    label: ClassVar[str] = "IV-GMM"
    wraps: ClassVar[str] = "linearmodels.iv.IVGMM"
    docs: ClassVar[str] = "https://bashtage.github.io/linearmodels/"
    topics: ClassVar[tuple[str, ...]] = ('instrumental-variables',)
    references: ClassVar[tuple[str, ...]] = (
        ("Hansen, L. P. (1982). Large sample properties of generalized method of moments estimators. Econometrica 50(4), 1029-1054. doi:10.2307/1912775"),
    )
    kind: ClassVar[str] = "iv"
    requires: ClassVar[tuple[str, ...]] = ("linearmodels",)
    adapter_spec: ClassVar[AdapterSpec] = specs.IVGMM

    endog: Sequence[str]
    instruments: Sequence[str]
    cov: str | Cluster = "white"
    hac_lags: int | None = None
    weight_type: Annotated[Literal["robust", "unadjusted", "kernel", "clustered"], _adv("weighting matrix")] = "robust"
    iter_limit: Annotated[int, _adv("GMM iterations (2 = two-step; large = iterated)", min=1, max=1000)] = 2
    tol: Annotated[float, _adv("convergence tolerance for iterated GMM", min=1e-12, max=1e-2, step=1e-6)] = 1e-4
    debiased: Annotated[bool, _adv("small-sample (n-k) correction of the covariance")] = True

    backend_options: Annotated[BackendOptions, _BACKEND_OPTIONS] = field(default_factory=dict)

    def estimate(self, eq: Equation, data: Dataset, *, span=None, where=None, weights=None, missing="drop") -> FittedEquation:
        _check(eq, "IV-GMM")
        iv = require("linearmodels.iv")
        d, y, exog, endog, inst, w = _iv_data(self, eq, data, span, where, weights, missing)
        cov_type, config = _cov(self.cov, self.hac_lags, data, d.index, panel=False)
        weight_config = {k: v for k, v in config.items() if self.weight_type in ("kernel", "clustered")}
        with specs.ADAPTER.run(self.adapter_spec, self) as run:
            model = run.call("model", iv.IVGMM, y, exog, endog, inst, weights=w, weight_type=self.weight_type, **weight_config)
            res = run.call("fit", model.fit, iter_limit=self.iter_limit, tol=self.tol, cov_type=cov_type, debiased=self.debiased, **config)
        if getattr(res, "iterations", 0) and res.iterations >= self.iter_limit > 2:
            run.message(f"IV-GMM stopped at iter_limit={self.iter_limit} iterations")
        j = res.j_stat
        stats = {**_iv_stats(res, len(d.y), len(d.names)), "hansen_j": float(j.stat), "hansen_j_pvalue": float(j.pval)}
        extra = {"first_stage_F": _first_stage(res), "instrument_names": d.z_names, "j_df": int(j.df)}
        return _result(eq, self, data, d, res, d.index, extra, stats, d.names, d.X, run.record, _instrument_vars(self.instruments))


def _instrument_vars(texts: Sequence[str]) -> list[str]:
    from ..expr import parse_expression

    return sorted(set().union(*(parse_expression(t).variables() for t in texts)))


# panel estimators


def _panel_frame(data: Dataset, d: LinearDesign) -> pd.MultiIndex:
    if data.kind != "panel":
        raise SpecificationError("panel estimators need a panel dataset (one entity axis and a time axis)")
    entity, time = d.index.names
    times = d.index.get_level_values(time)
    stamps = times.to_timestamp() if isinstance(times, pd.PeriodIndex) else times
    return pd.MultiIndex.from_arrays([d.index.get_level_values(entity), stamps], names=[entity, time])


def _panel_fit(m: Any, model_name: str, eq: Equation, data: Dataset, span, where, weights, missing, drop_const: bool, **fit_kw: Any) -> FittedEquation:
    _check(eq, model_name)
    if eq.fixed_effects:
        raise SpecificationError(f"{model_name} handles entity effects itself; remove fe() terms")
    panel = require("linearmodels.panel")
    d = linear_design(eq, data, span, where, weights, missing)
    pindex = _panel_frame(data, d)
    names = [n for n in d.names if not (drop_const and n == "const")]
    X = pd.DataFrame(d.X, index=pindex, columns=d.names)[names]
    y = pd.Series(d.y, index=pindex, name="y")
    w = pd.Series(d.w, index=pindex) if d.w is not None else None
    cov_type, config = _cov(m.cov, getattr(m, "hac_lags", None), data, d.index, panel=True)
    if "clusters" in config:
        config["clusters"].index = pindex
    cls = {"RandomEffects": panel.RandomEffects, "BetweenOLS": panel.BetweenOLS, "FirstDifferenceOLS": panel.FirstDifferenceOLS,
           "FamaMacBeth": panel.FamaMacBeth}[model_name]
    with specs.ADAPTER.run(m.adapter_spec, m) as run:  # refuses settings the estimator cannot honour (FamaMacBeth clusters)
        model = run.call("model", cls, y, X, weights=w)
        res = run.call("fit", model.fit, cov_type=cov_type, debiased=m.debiased, **fit_kw, **config)
    # map linearmodels' (entity, timestamp) index back to the dataset's periods;
    # the between estimator has one residual per entity and keeps the entity index
    rindex = res.resids.index
    lookup = dict(zip(pindex, d.index))
    if isinstance(rindex, pd.MultiIndex) and all(i in lookup for i in rindex):
        index = pd.MultiIndex.from_tuples([lookup[i] for i in rindex], names=d.index.names)
        Xu = X.reindex(rindex).to_numpy()
    else:
        index = pd.Index(rindex, name=d.index.names[0])
        Xu = np.full((len(rindex), len(names)), np.nan)
    stats = {"nobs": float(res.nobs), "k": float(len(names)), "df_resid": float(res.df_resid), "r2": float(res.rsquared)}
    for attr, key in (("rsquared_within", "r2_within"), ("rsquared_between", "r2_between"), ("rsquared_overall", "r2_overall")):
        if hasattr(res, attr):
            stats[key] = float(getattr(res, attr))
    if hasattr(res, "s2"):
        stats["se_regression"] = float(np.sqrt(res.s2))
    if hasattr(res, "theta"):
        stats["theta_mean"] = float(np.asarray(res.theta).mean())
    extra: dict[str, Any] = {}
    if hasattr(res, "variance_decomposition"):
        extra["variance_decomposition"] = {str(k): float(v) for k, v in res.variance_decomposition.items()}
    return _result(eq, m, data, d, res, index, extra, stats, names, Xu, run.record)


@dataclass(frozen=True)
class RandomEffects(_Wrapped):
    """Random-effects (GLS) panel estimator."""

    maturity: ClassVar[str] = "stable"
    label: ClassVar[str] = "Random effects"
    wraps: ClassVar[str] = "linearmodels.panel.RandomEffects"
    docs: ClassVar[str] = "https://bashtage.github.io/linearmodels/"
    topics: ClassVar[tuple[str, ...]] = ('panel-data',)
    references: ClassVar[tuple[str, ...]] = (
        ("Balestra, P. and Nerlove, M. (1966). Pooling cross section and time series data in the estimation of a dynamic model: the demand for natural gas. Econometrica 34(3), 585-612. doi:10.2307/1909771"),
        "Wooldridge, J. M. (2010). Econometric Analysis of Cross Section and Panel Data, 2nd ed., chapter 10, \"Basic linear unobserved effects panel data models\". MIT Press.",
    )
    kind: ClassVar[str] = "panel"
    requires: ClassVar[tuple[str, ...]] = ("linearmodels",)
    adapter_spec: ClassVar[AdapterSpec] = specs.RANDOM_EFFECTS

    cov: str | Cluster = "classic"
    small_sample: Annotated[bool, _adv("small-sample adjustment of the variance components")] = False
    debiased: Annotated[bool, _adv("small-sample (n-k) correction of the covariance")] = True

    backend_options: Annotated[BackendOptions, _BACKEND_OPTIONS] = field(default_factory=dict)

    def estimate(self, eq: Equation, data: Dataset, *, span=None, where=None, weights=None, missing="drop") -> FittedEquation:
        return _panel_fit(self, "RandomEffects", eq, data, span, where, weights, missing, False, small_sample=self.small_sample)


@dataclass(frozen=True)
class Between(_Wrapped):
    """Between estimator: OLS on entity means."""

    maturity: ClassVar[str] = "stable"
    label: ClassVar[str] = "Between"
    wraps: ClassVar[str] = "linearmodels.panel.BetweenOLS"
    docs: ClassVar[str] = "https://bashtage.github.io/linearmodels/"
    topics: ClassVar[tuple[str, ...]] = ('panel-data',)
    references: ClassVar[tuple[str, ...]] = (
        "Wooldridge, J. M. (2010). Econometric Analysis of Cross Section and Panel Data, 2nd ed., chapter 10, \"Basic linear unobserved effects panel data models\". MIT Press.",
    )
    kind: ClassVar[str] = "panel"
    requires: ClassVar[tuple[str, ...]] = ("linearmodels",)
    adapter_spec: ClassVar[AdapterSpec] = specs.BETWEEN

    cov: str | Cluster = "classic"
    reweight: Annotated[bool, _adv("weight entities by their number of observations")] = False
    debiased: Annotated[bool, _adv("small-sample (n-k) correction of the covariance")] = True

    backend_options: Annotated[BackendOptions, _BACKEND_OPTIONS] = field(default_factory=dict)

    def estimate(self, eq: Equation, data: Dataset, *, span=None, where=None, weights=None, missing="drop") -> FittedEquation:
        return _panel_fit(self, "BetweenOLS", eq, data, span, where, weights, missing, False, reweight=self.reweight)


@dataclass(frozen=True)
class FirstDifference(_Wrapped):
    """First-difference estimator (the constant drops out)."""

    maturity: ClassVar[str] = "stable"
    label: ClassVar[str] = "First difference"
    wraps: ClassVar[str] = "linearmodels.panel.FirstDifferenceOLS"
    docs: ClassVar[str] = "https://bashtage.github.io/linearmodels/"
    topics: ClassVar[tuple[str, ...]] = ('panel-data',)
    references: ClassVar[tuple[str, ...]] = (
        "Wooldridge, J. M. (2010). Econometric Analysis of Cross Section and Panel Data, 2nd ed., chapter 10, \"Basic linear unobserved effects panel data models\". MIT Press.",
    )
    kind: ClassVar[str] = "panel"
    requires: ClassVar[tuple[str, ...]] = ("linearmodels",)
    adapter_spec: ClassVar[AdapterSpec] = specs.FIRST_DIFFERENCE

    cov: str | Cluster = "classic"
    debiased: Annotated[bool, _adv("small-sample (n-k) correction of the covariance")] = True

    backend_options: Annotated[BackendOptions, _BACKEND_OPTIONS] = field(default_factory=dict)

    def estimate(self, eq: Equation, data: Dataset, *, span=None, where=None, weights=None, missing="drop") -> FittedEquation:
        return _panel_fit(self, "FirstDifferenceOLS", eq, data, span, where, weights, missing, True)


@dataclass(frozen=True)
class FamaMacBeth(_Wrapped):
    """Fama-MacBeth: cross-section regressions each period, averaged."""

    maturity: ClassVar[str] = "stable"
    label: ClassVar[str] = "Fama-MacBeth"
    wraps: ClassVar[str] = "linearmodels.panel.FamaMacBeth"
    docs: ClassVar[str] = "https://bashtage.github.io/linearmodels/"
    topics: ClassVar[tuple[str, ...]] = ('panel-data',)
    references: ClassVar[tuple[str, ...]] = (
        ("Fama, E. F. and MacBeth, J. D. (1973). Risk, return, and equilibrium: empirical tests. Journal of Political Economy 81(3), 607-636. doi:10.1086/260061"),
    )
    kind: ClassVar[str] = "panel"
    requires: ClassVar[tuple[str, ...]] = ("linearmodels",)
    adapter_spec: ClassVar[AdapterSpec] = specs.FAMA_MACBETH

    cov: str | Cluster = "classic"
    hac_lags: int | None = None
    debiased: Annotated[bool, _adv("small-sample (n-k) correction of the covariance")] = True

    backend_options: Annotated[BackendOptions, _BACKEND_OPTIONS] = field(default_factory=dict)

    def estimate(self, eq: Equation, data: Dataset, *, span=None, where=None, weights=None, missing="drop") -> FittedEquation:
        return _panel_fit(self, "FamaMacBeth", eq, data, span, where, weights, missing, False)


for _cls in (LIML, IVGMM, RandomEffects, Between, FirstDifference, FamaMacBeth):
    register("methods", _cls, plugin=PLUGIN)
