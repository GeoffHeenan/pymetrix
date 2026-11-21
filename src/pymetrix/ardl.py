"""ARDL models and the bounds test for a level relationship (Pesaran, Shin and Smith 2001), via statsmodels.

    fit = ARDL("lc ~ ly + lw", lags=2, order=2).fit(data)          # or select="aic" to choose the orders
    fit.long_run()                                                 # long-run multipliers, with standard errors
    fit.bounds_test()                                              # is there a level (cointegrating) relationship?
    fit.ecm()                                                      # the error-correction form

An autoregressive distributed lag model regresses a series on its own lags and current and
lagged values of other series. It needs no decision on whether the series are I(0) or I(1): the
bounds test of Pesaran, Shin and Smith asks whether a level relationship exists, with critical
values bounded by the all-I(0) and all-I(1) cases. Above the upper bound, a relationship exists;
below the lower, it does not; in between, the test is inconclusive. The long-run multipliers are
the coefficients of that relationship; the error-correction form shows how fast deviations from
it close. Estimation, order selection and the bounds test are computed by statsmodels.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd

from ._api import api
from .errors import SpecificationError
from .adapters import require
from .adapters import statsmodels as statsmodels_adapter
from .adapters.base import AdapterSpec, BackendOptions, check_backend_options, with_backend
from .forms import Widget
from .output import Table
from .results import BackendResult, deprecated_alias

if TYPE_CHECKING:
    from .data import Dataset
    from .stats import TestResult

__all__ = ["ARDL", "FittedARDL"]

_CASE = {"n": 1, "c": 3, "ct": 5}  # unrestricted constant (case 3) and trend (case 5), as is usual


def _bounds(bt: Any) -> tuple[float, pd.DataFrame, pd.Series]:
    """Statistic, critical values and p-values of statsmodels' bounds test (renamed in 0.15:
    stat, crit_vals, p_values became statistic, critical_values, pvalue)."""
    stat = getattr(bt, "statistic", None)
    if stat is None:
        return float(bt.stat), bt.crit_vals, bt.p_values
    return float(stat), bt.critical_values, bt.pvalue


@dataclass(frozen=True)
class ARDL:
    """``formula``: the dependent variable on the distributed-lag regressors (``lc ~ ly + lw``; expressions allowed)."""

    formula: str
    lags: int = 1  # lags of the dependent variable
    order: int | Mapping[str, int] = 1  # lags of each regressor (0 = current value only)
    trend: Literal["n", "c", "ct"] = "c"
    select: Literal["aic", "bic", "hqic"] | None = None  # choose lags and orders by an information criterion
    max_lags: int = 4
    max_order: int = 4
    backend_options: Annotated[BackendOptions, Widget("", advanced=True, help="other statsmodels arguments; model.x for the constructor")] = field(default_factory=dict)

    adapter_spec: ClassVar[AdapterSpec] = statsmodels_adapter.ARDL

    def __post_init__(self) -> None:
        check_backend_options(self)

    def _frames(self, data: Dataset, span: Any) -> tuple[pd.Series, pd.DataFrame]:
        from .estimation import Equation
        from .expr import evaluate

        if data.kind != "time_series":
            raise SpecificationError("ARDL needs a single time series dataset")
        eq = Equation(self.formula)
        if not eq.terms:
            raise SpecificationError("give at least one regressor: lc ~ ly")
        y = pd.Series(np.asarray(evaluate(eq.lhs, data.lagged), float), index=data.frame.index, name=eq.require_dependent())
        X = pd.DataFrame({t.label: np.asarray(evaluate(t.expr, data.lagged), float) for t in eq.terms}, index=data.frame.index)
        mask = np.asarray(data.sample_mask(span))
        both = pd.concat([y, X], axis=1)[mask].dropna()
        if len(both) < 10:
            raise SpecificationError("fewer than ten complete observations")
        if len(pd.period_range(both.index[0], both.index[-1], freq=both.index.freq)) != len(both):
            raise SpecificationError("the sample has gaps; ARDL needs consecutive periods")
        return both.iloc[:, 0], both.iloc[:, 1:]

    @api("pymetrix:ardl.fit")
    def fit(self, data: Dataset, span: Any = None) -> FittedARDL:
        from dataclasses import replace

        from ._capture import capture

        with capture() as caught:
            fitted, record = self._fit(data, span)
        y, X = self._frames(data, span)
        from ._capture import provenance_for

        names = [y.name, *X.columns, *_formula_variables(self.formula)]
        presample = y.index.difference(fitted.resid.index)  # statsmodels holds back the first max(lags, order) rows
        prov = provenance_for(self, data, names, fitted.resid.index, pd.concat([y, X], axis=1), caught, {"type": "ARDL"}, span=span,
                              presample=presample)
        return replace(fitted, provenance=with_backend(prov, record))

    def _fit(self, data: Dataset, span: Any = None) -> tuple[FittedARDL, Any]:
        ardl = require("statsmodels.tsa.ardl")
        y, X = self._frames(data, span)
        y.index = X.index = y.index.to_timestamp()  # statsmodels wants dates
        run = statsmodels_adapter.ADAPTER.run(self.adapter_spec, self)
        with run:
            model, res = self._model(ardl, y, X, run)
        return FittedARDL.from_statsmodels(self, model, res, data.frame.index.freqstr), run.record

    def _model(self, ardl: Any, y: pd.Series, X: pd.DataFrame, run: Any) -> tuple[Any, Any]:
        if self.select:
            sel = run.call("model", ardl.ardl_select_order, y, self.max_lags, X, self.max_order, trend=self.trend, ic=self.select)
            model = sel.model
        else:
            order = dict(self.order) if isinstance(self.order, Mapping) else self.order
            if isinstance(order, dict):
                unknown = sorted(set(order) - set(X.columns))
                if unknown:
                    raise SpecificationError(f"order names {unknown}, which are not regressors: {list(X.columns)}")
                order = {c: order.get(c, 1) for c in X.columns}
            model = run.call("model", ardl.ARDL, y, self.lags, X, order, trend=self.trend)
        return model, run.call("fit", model.fit)


def _inference(names: list[str], result: Any) -> pd.DataFrame:
    return pd.DataFrame({"coef": np.asarray(result.params, float), "std err": np.asarray(result.bse, float),
                         "t": np.asarray(result.tvalues, float), "p>|t|": np.asarray(result.pvalues, float)}, index=names)


@dataclass(frozen=True, eq=False)
class FittedARDL(BackendResult):
    """An estimated ARDL: its own fields hold every number it reports, read from statsmodels when it is made.

    Parameter names are pymetrix expressions: ``c[-1]`` (statsmodels' ``c.L1``), ``y`` (``y.L0``) and, in the
    error-correction form, ``d(y)[-1]`` (``D.y.L1``); the old names still work in ``params[...]``, with a warning."""

    spec: ARDL
    params: Any  # PointEstimate
    inference: pd.DataFrame  # coef, std err, t, p>|t| by parameter
    resid: pd.Series
    fitted: pd.Series
    stats: Mapping[str, float]
    long_run_coefficients: pd.DataFrame | None  # the long-run multipliers: coefficient and std err (delta method)
    ecm_params: Any  # PointEstimate: the error-correction form (the speed of adjustment is on y[-1]); None if undefined
    ecm_inference: pd.DataFrame | None
    order: tuple[int, dict[str, int]]  # the dependent variable's lags, and each regressor's (as chosen, when selected)
    dependent: str  # was target
    bounds: Any = None  # the bounds test at the default case
    ecm_problem: str = ""  # why there is no error-correction form, when there is none
    freq: str = ""
    provenance: Any = None
    backend_result: Any = field(default=None, repr=False, compare=False)  # statsmodels' ARDLResults
    backend_model: Any = field(default=None, repr=False, compare=False)  # statsmodels' ARDL model

    _backend_fields: ClassVar[tuple[str, ...]] = ("backend_result", "backend_model")
    stat_definitions: ClassVar[dict[str, str]] = {"loglike": "loglike.gaussian"}
    result = deprecated_alias("backend_result", "result")
    model = deprecated_alias("backend_model", "model")

    @classmethod
    def from_statsmodels(cls, spec: ARDL, model: Any, result: Any, freq: str) -> FittedARDL:
        """Read everything the result reports from statsmodels' ARDL model and results, once."""
        from .adapters.statsmodels import pymetrix_name
        from .estimation.params import point

        target = str(model.endog_names)
        sm_names = [str(n) for n in result.params.index]
        names = [pymetrix_name(n) for n in sm_names]

        def periods(index: pd.Index) -> pd.PeriodIndex:
            return pd.PeriodIndex(index.to_period(freq)) if isinstance(index, pd.DatetimeIndex) else pd.PeriodIndex(index)

        resid = pd.Series(result.resid.to_numpy(float), index=periods(result.resid.index), name="resid")
        fitted = pd.Series(result.fittedvalues.to_numpy(float), index=periods(result.fittedvalues.index), name="fitted")
        y = fitted + resid
        n, k = int(result.nobs), len(result.params)
        r2 = 1 - float((resid**2).sum() / ((y - y.mean()) ** 2).sum())  # centred: statsmodels' ARDL rsquared is uncentred
        from .estimation import criteria

        llf = float(result.llf)
        stats = {"nobs": float(n), "k": float(k), "r2": r2, "r2_adj": 1 - (1 - r2) * (n - 1) / (n - k),
                 "se_regression": float(np.sqrt(result.sigma2)), "loglike": llf, "aic": criteria.aic(llf, k),
                 "bic": criteria.bic(llf, k, n),
                 # statsmodels counts the error variance as a parameter (k + 1): kept, labelled, for comparison
                 "aic_backend": float(result.aic), "bic_backend": float(result.bic)}
        lags = max(model.ar_lags) if model.ar_lags else 0
        dl = {name: (max(v) if v else 0) for name, v in model.dl_lags.items()}
        long_run = ecm_params = ecm_inference = bounds = None
        ecm_problem = ""
        try:  # the error-correction form: not defined for every specification (statsmodels needs each order >= 1)
            uecm = require("statsmodels.tsa.ardl").UECM.from_ardl(model).fit()
        except ValueError as exc:
            ecm_problem = str(exc)
        else:
            ecm_sm = [str(n) for n in uecm.params.index]
            ecm_names = [pymetrix_name(n) for n in ecm_sm]
            lr_index = [k for k in uecm.ci_params.index if k != target]
            long_run = pd.DataFrame({"coefficient": [-float(uecm.ci_params[k]) for k in lr_index],
                                     "std err": [float(uecm.ci_bse[k]) for k in lr_index]}, index=lr_index)
            ecm_params = point(ecm_names, uecm.params.to_numpy(float), np.asarray(uecm.cov_params(), float), dict(zip(ecm_sm, ecm_names)))
            ecm_inference = _inference(ecm_names, uecm)
            try:
                bounds = _bounds_result(uecm, _CASE[spec.trend], None)
            except Exception:  # noqa: BLE001 - nor is the bounds test
                bounds = None
        return cls(spec, point(names, result.params.to_numpy(float), np.asarray(result.cov_params(), float), dict(zip(sm_names, names))),
                   _inference(names, result), resid, fitted, stats, long_run, ecm_params, ecm_inference, (lags, dl), target, bounds,
                   ecm_problem, freq, None, result, model)

    def _ecm(self, what: str) -> None:
        if self.ecm_params is None:
            raise SpecificationError(f"{what} is not available for this ARDL: {self.ecm_problem}")

    def summary(self) -> Table:
        lags, dl = self.order
        orders = ", ".join(f"{k} {v}" for k, v in dl.items())
        s = self.stats
        notes = [f"ARDL({lags}, {orders}); trend {self.spec.trend}" + (f"; orders chosen by {self.spec.select.upper()}" if self.spec.select else ""),
                 f"Sample: {self.resid.index[0]} to {self.resid.index[-1]}, {int(s['nobs'])} observations",
                 f"R-squared: {s['r2']:.4f}", f"S.E. of regression: {s['se_regression']:.4f}",
                 f"Akaike info criterion: {s['aic']:.3f}", f"Schwarz criterion: {s['bic']:.3f}",
                 "Long-run multipliers: long_run(); level relationship: bounds_test(); error-correction form: ecm()"]
        return Table(list(self.inference.columns), self.inference.to_numpy().tolist(), index=list(self.inference.index),
                     title=f"ARDL: {self.dependent}", notes=notes)

    def long_run(self) -> Table:
        """Long-run multipliers: the level relationship y = sum of coefficients x regressors (and constant)."""
        self._ecm("The long-run relationship")
        frame = self.long_run_coefficients
        return Table(["coefficient", "std err"], frame.to_numpy().tolist(), index=list(frame.index), title=f"Long-run relationship for {self.dependent}",
                     notes=["From the error-correction form; standard errors by the delta method (statsmodels)."])

    def ecm(self) -> Table:
        """The error-correction form: differences on lagged levels (the speed of adjustment) and lagged differences."""
        self._ecm("The error-correction form")
        return Table(list(self.ecm_inference.columns), self.ecm_inference.to_numpy().tolist(), index=list(self.ecm_inference.index),
                     title=f"Error-correction form: d({self.dependent})",
                     notes=[f"The coefficient on {self.dependent}[-1] is the speed of adjustment (negative if the relationship is stable)."])

    def bounds_test(self, case: int | None = None) -> TestResult:
        """Pesaran-Shin-Smith bounds F test of no level relationship. Case 3 (unrestricted constant) with a
        constant, case 5 with a trend, case 1 with neither; or give ``case`` (1 to 5)."""
        default = _CASE[self.spec.trend]
        self._ecm("The bounds test")
        if (case or default) == default and self.bounds is not None:
            return self.bounds
        model = self._require_backend("A bounds test for another case")
        ardl = require("statsmodels.tsa.ardl")
        return _bounds_result(ardl.UECM.from_ardl(model).fit(), case or default, case)

    def plot(self) -> Any:
        from .figures import Figure, Line, Panel

        return Figure((Panel((Line(self.fitted + self.resid, "actual"), Line(self.fitted, "fitted")), title=self.dependent),))

    def __repr__(self) -> str:
        lags, dl = self.order
        return f"<FittedARDL ARDL({lags}, {', '.join(map(str, dl.values()))}): {self.spec.formula}>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()

    def _require_backend(self, what: str) -> Any:  # the model, not the results, is what statsmodels' UECM needs
        if self.backend_model is None:
            return super()._require_backend(what)
        return self.backend_model

    def __setstate__(self, state: dict[str, Any]) -> None:  # results cached in projects saved before the rename
        if "target" in state:
            state["dependent"] = state.pop("target")
        super().__setstate__(state)

    target = deprecated_alias("dependent", "target", why="the dependent variable; target is its old name")


def _bounds_result(uecm: Any, case: int, asked: int | None) -> TestResult:
    from .stats import TestResult

    stat, cv, pv = _bounds(uecm.bounds_test(case=case))
    crit = {f"{int(q) if float(q).is_integer() else q}% {bound}": float(cv.loc[q, side])
            for q in cv.index for bound, side in (("I(0)", "lower"), ("I(1)", "upper"))}
    upper95 = float(cv.loc[95.0, "upper"])
    lower95 = float(cv.loc[95.0, "lower"])
    verdict = ("a level relationship (above the I(1) bound)" if stat > upper95 else
               "no level relationship (below the I(0) bound)" if stat < lower95 else "inconclusive (between the bounds)")
    return TestResult("Bounds test (Pesaran-Shin-Smith)", "no level relationship", stat, None, float(pv["upper"]), "F (bounds)",
                      critical_values=crit, details={"p-value, all I(0)": float(pv["lower"]), "p-value, all I(1)": float(pv["upper"]),
                                                     "case": case, "at 5%": verdict})


def _formula_variables(formula: str) -> list[str]:
    """The series a formula uses (for provenance)."""
    from .estimation import Equation

    return sorted(Equation(formula).variables())
