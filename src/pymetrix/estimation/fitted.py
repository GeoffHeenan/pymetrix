"""FittedEquation: the immutable record of one estimation (or calibration)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from pymetrix._lazy import stats as sps

from .._api import api
from ..errors import SpecificationError
from ..expr import Coef, Expr, diff, evaluate
from ..metadata import Provenance
from ..output import Table
from ..results import BackendResult, deprecated_alias
from .params import Params

if TYPE_CHECKING:
    from ..data import Dataset
    from ..stats import TestResult
    from .equation import Equation


class _Extra(dict):
    """``extra``, where the library's result object used to be kept (``extra["arch"]``, ``extra["linearmodels"]``):
    those keys still work, with a warning, and read ``backend_result``."""

    _LEGACY = ("arch", "linearmodels")

    def __init__(self, values: Mapping[str, Any], owner: Any) -> None:
        super().__init__(values)
        self._owner = owner

    def __missing__(self, key: str) -> Any:
        if key in self._LEGACY and getattr(self._owner, "backend_result", None) is not None:
            import warnings

            warnings.warn(f'extra["{key}"] is deprecated since pymetrix 0.1.0a9; use .backend_result (the library\'s own object, '
                          "which is not saved with projects)", DeprecationWarning, stacklevel=2)
            return self._owner.backend_result
        from ..errors import NotFoundError

        raise NotFoundError(key)

    def __reduce__(self) -> Any:
        return (dict, (dict(self),))  # saved as a plain dict: the owner restores the wrapper


@dataclass(frozen=True, eq=False)
class FittedEquation(BackendResult):
    equation: Equation
    params: Params
    method: Any
    resid: pd.Series | None = None
    fitted: pd.Series | None = None
    stats: Mapping[str, float] = field(default_factory=dict)
    provenance: Provenance | None = None
    design: pd.DataFrame | None = field(default=None, repr=False)  # regressors, or NLS Jacobian
    weights: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)  # method-specific results
    backend_result: Any = field(default=None, repr=False, compare=False)  # the library's own result object (arch, linearmodels), if any

    def __post_init__(self) -> None:
        extra = dict(self.extra or {})
        legacy = [k for k in _Extra._LEGACY if k in extra]  # a wrapper written before backend_result: move the object
        if legacy and self.backend_result is None:
            object.__setattr__(self, "backend_result", extra[legacy[0]])
        for k in legacy:
            extra.pop(k)
        object.__setattr__(self, "extra", _Extra(extra, self))

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        object.__setattr__(self, "extra", _Extra(dict(self.extra or {}), self))

    @property
    def name(self) -> str:
        return self.equation.label

    @property
    def dependent(self) -> str | None:
        return self.equation.dependent

    target = deprecated_alias("dependent", "target", why="the dependent variable; target is its old name")

    def residual_expression(self) -> Expr:
        return self.equation.residual_expression(self.params.as_dict())

    def plot(self) -> Any:
        """Actual, fitted and residuals (time series), or actual against fitted (cross-sections)."""
        from ..figures import fit_figure

        return fit_figure(self)

    def qq_plot(self) -> Any:
        """The residuals against the normal distribution (a Q-Q plot): they lie along the dashed line if normal."""
        from ..errors import SpecificationError
        from ..figures import QQ, Figure, Panel

        if self.resid is None:
            raise SpecificationError("this fit has no residuals")
        return Figure((Panel((QQ(self.resid, "residuals"),), title="Residuals against the normal distribution", legend=False),),
                      width=5.0, height=4.0)

    def algebraic(self, style: Literal["unicode", "latex"] = "unicode", estimates: bool = False, digits: int = 4) -> str:
        """The equation in algebraic notation; ``estimates=True`` puts the estimated values in place of the
        coefficients (``cₜ = 9.632 + 0.6425 yₜ + 0.1456 cₜ₋₁``)."""
        from ..notation import algebraic

        values = dict(zip(self.params.names, map(float, self.params.values))) if estimates else None
        return algebraic(self.equation, style, values=values, digits=digits)

    def summary(self) -> Table:
        p = self.params
        se = p.se
        t = p.values / se
        df = self.stats.get("df_resid")
        pvals = 2 * (sps.t.sf(np.abs(t), df) if df else sps.norm.sf(np.abs(t)))
        rows = [[v, s, tv, pv] for v, s, tv, pv in zip(p.values, se, t, pvals)]
        method = type(self.method).__name__
        family = type(self.equation.family).__name__
        notes = [f"Method: {method}" + ("" if family == "Gaussian" else f" ({family})")
                 + _covariance_note(self.method, self.stats.get("nobs"), self.stats.get("hac_lags"))]
        if self.provenance is not None and self.provenance.warnings:
            notes.append("Warnings: " + "; ".join(self.provenance.warnings))
        if self.provenance is not None:
            sample = f"{self.provenance.sample[0]} to {self.provenance.sample[1]}, " if self.provenance.sample else ""
            notes.append(f"Sample: {sample}{self.provenance.nobs} observations ({self.provenance.dropped} dropped)")
        if "instruments" in self.extra:
            notes.append("Instruments: " + ", ".join(self.extra["instruments"].columns))
        elif "instrument_names" in self.extra:
            notes.append("Excluded instruments: " + ", ".join(self.extra["instrument_names"]))
        notes += [f"First-stage F ({name}): {F:.2f}" for name, F in self.extra.get("first_stage_F", {}).items()]
        notes += list(self.extra.get("notes", []))  # method-specific results, e.g. GMM specification tests
        backend = (self.provenance.method.get("backend") if self.provenance else None)
        if backend:
            notes[0] += f" [{backend}]"
        shown = ("r2", "r2_within", "r2_between", "r2_overall", "r2_adj", "pseudo_r2", "se_regression", "kappa", "theta_mean",
                 "hansen_j", "hansen_j_pvalue", "loglike", "ll_null", "lr", "aic", "bic", "hq", "durbin_watson")
        notes += [f"{STAT_LABELS.get(k, k)}: {self.stats[k]:.4f}" for k in shown if k in self.stats]
        stat = "t" if df else "z"
        from ..notation import get_notation, labelled

        if get_notation() == "algebraic" and not self.equation.fixed_effects and not self.equation.categoricals:
            return Table(["coef", "std err", stat, f"p>|{stat}|"], rows, index=labelled(p.names, self.equation),
                         title=self.equation.algebraic(), notes=notes)
        return Table(["coef", "std err", stat, f"p>|{stat}|"], rows, index=list(p.names), title=self.equation.text, notes=notes)

    def __repr__(self) -> str:
        return f"<FittedEquation {self.equation.text!r}, {type(self.method).__name__}, {len(self.params)} parameters>"

    def to_frame(self, view: str = "Coefficients") -> pd.DataFrame:
        """One of the result's views as a DataFrame: ``Coefficients`` (the default), ``Statistics``, ``Summary``,
        ``Covariance``... (the tabs of its window; case does not matter)."""
        from ..views import _views_to_frame

        return _views_to_frame(self, view)

    @classmethod
    def from_statsmodels(cls, res: Any, data: Any = None, command: str | None = None) -> FittedEquation:
        """A ``FittedEquation`` from a statsmodels regression result (see ``pymetrix.interop.from_statsmodels``)."""
        from ..interop import from_statsmodels

        return from_statsmodels(res, data, command)

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()

    def __str__(self) -> str:
        return str(self.summary())

    def _rhs(self) -> Expr:
        eq = self.equation
        if eq.fixed_effects or eq.categoricals:
            raise NotImplementedError("prediction with fe() or C() terms is not supported yet")
        return eq.rhs if eq.form == "explicit" else eq.linear_rhs()

    def _over(self, data: Dataset, value: Any) -> pd.Series:
        return value if isinstance(value, pd.Series) else pd.Series(value, index=data.frame.index, dtype=float)

    def breaks(self, max_breaks: int = 5, trim: float = 0.15) -> Any:
        """Bai-Perron multiple structural breaks at unknown dates, the number chosen by BIC (``stats.breaks``)."""
        from ..stats.breaks import bai_perron

        return bai_perron(self, max_breaks, trim)

    def marginal_effects(self, at: Literal["average", "mean"] = "average", alpha: float = 0.05) -> Table:
        """Marginal effects of a logit, probit, Poisson or limited dependent variable model (``estimation.limited``):
        averaged over the sample, or at the regressors' means."""
        from .limited import marginal_effects

        return marginal_effects(self, at, alpha)

    def predict(
        self,
        data: Dataset,
        span: tuple[Any, Any] | None = None,
        dynamic: bool = False,
        kind: Literal["response", "index"] = "response",
    ) -> pd.Series:
        """Static: fitted values of the left-hand side (for Logit/Probit/Poisson, the
        probability or mean unless ``kind="index"``). Dynamic: simulate the target variable.
        With ARMA errors or GARCH variance: the forecast mean of the left-hand side."""
        if self.equation.has_error_model:
            if span is None:
                raise SpecificationError("give the span to predict over")
            return self.forecast(data, span, dynamic=dynamic)["forecast"].rename("predicted")
        if "limited" in self.extra:  # counts, ordered, multinomial, Tobit: their own response (probabilities for choices)
            from .limited import predict

            if dynamic:
                raise SpecificationError(f"{type(self.equation.family).__name__} equations predict statically")
            return predict(self, data, span, kind)
        if dynamic:
            from ..models import Model

            target = self.equation.require_dependent()
            solution = Model([self]).solve(data, span=span)
            return solution[target][solution.sample_mask(span)]
        out = self._over(data, evaluate(self._rhs(), data.lagged, self.params.as_dict()))
        if kind == "response":
            out = pd.Series(self.equation.family.mean(out.to_numpy()), index=out.index)
        return out[data.sample_mask(span)].rename("predicted")

    @api()
    def forecast(self, data: Dataset, span: tuple[Any, Any], dynamic: bool = False, alpha: float = 0.05) -> Dataset:
        """Forecasts of the left-hand side with standard errors and a (1 - alpha) interval.

        Static forecasts use actual lagged values; their standard errors include
        residual and parameter uncertainty. Dynamic forecasts simulate the target
        variable; their standard errors need stochastic simulation (stage 2) and are NaN.
        """
        from ..data import Dataset
        from .families import Gaussian

        if self.equation.has_error_model:
            from .timeseries import forecast

            return forecast(self, data, span, dynamic, alpha)
        if not isinstance(self.equation.family, Gaussian):
            raise SpecificationError("use predict() for Logit, Probit and Poisson equations")
        mask = data.sample_mask(span)
        if dynamic:
            point = self.predict(data, span, dynamic=True)
            se = pd.Series(np.nan, index=point.index)
        else:
            rhs = self._rhs()
            values = self.params.as_dict()
            point = self._over(data, evaluate(rhs, data.lagged, values))[mask]
            G = np.column_stack([
                np.broadcast_to(self._over(data, evaluate(diff(rhs, Coef(n)), data.lagged, values))[mask].to_numpy(), (int(mask.sum()),))
                for n in self.params.names
            ])
            cov = self.params.cov if self.params.cov is not None else np.zeros((len(self.params), len(self.params)))
            s2 = self.stats.get("se_regression", 0.0) ** 2
            se = pd.Series(np.sqrt(s2 + np.einsum("ij,jk,ik->i", G, cov, G)), index=point.index)
        df = self.stats.get("df_resid")
        z = sps.t.ppf(1 - alpha / 2, df) if df else sps.norm.ppf(1 - alpha / 2)
        frame = pd.DataFrame({"forecast": point, "se": se, "lower": point - z * se, "upper": point + z * se})
        actual = self._over(data, evaluate(self.equation.lhs, data.lagged)) if not dynamic else data[self.equation.require_dependent()]
        frame["actual"] = actual.reindex(frame.index)
        return Dataset(frame, data.axes)

    # tests

    @api()
    def test(self, name: str, *args: Any, **opts: Any) -> TestResult:
        """Run a registered test on this fit by name, e.g. ``fit.test("breusch_godfrey", lags=4)`` or
        ``fit.test("chow", at="2008Q1")``; ``tests()`` lists the ones that apply."""
        from ..stats import run_test

        return run_test(self, name, *args, **opts)

    def tests(self) -> list[str]:
        """The names of the registered tests that apply to this fit."""
        from ..stats import available_tests

        return available_tests(self)

    def wald(self, restriction: str) -> TestResult:
        from ..stats import Wald, run_test

        return run_test(self, Wald(restriction))

    def recursive_residuals(self) -> Dataset:
        """Brown-Durbin-Evans recursive residuals with CUSUM and CUSUM-of-squares paths and 5% bands."""
        from ..stats.stability import recursive_table

        return recursive_table(self)

    @api()
    def correlogram(self, lags: int = 20, squared: bool = False) -> Table:
        """Correlogram of the residuals (standardised after ARMA or GARCH), or of their squares."""
        from ..stats.timeseries import correlogram_table

        if self.resid is None or isinstance(self.resid.index, pd.MultiIndex):
            raise SpecificationError("a residual correlogram needs residuals from a single time series")
        std = self.extra.get("std_resid")
        x = (std if std is not None else self.resid).to_numpy(float)
        what = ("standardised " if std is not None else "") + ("squared residuals" if squared else "residuals")
        model_df = 0 if squared else int(self.extra.get("model_df", 0))
        return correlogram_table(x**2 if squared else x, lags, f"Correlogram of {what}: {self.equation.text}", model_df)

    def diagnostics(self) -> Table:
        from ..stats import diagnostics

        return diagnostics(self)


STAT_LABELS = {
    "r2": "R-squared", "r2_within": "R-squared (within)", "r2_between": "R-squared (between)",
    "r2_overall": "R-squared (overall)", "r2_adj": "Adjusted R-squared", "pseudo_r2": "Pseudo R-squared",
    "se_regression": "S.E. of regression", "kappa": "Kappa (k-class)", "theta_mean": "Mean theta (random effects)",
    "hansen_j": "Hansen J", "hansen_j_pvalue": "Hansen J p-value", "loglike": "Log likelihood",
    "ll_null": "Log likelihood (constant only)", "lr": "LR statistic", "aic": "Akaike info criterion",
    "bic": "Schwarz criterion", "hq": "Hannan-Quinn criterion", "durbin_watson": "Durbin-Watson", "hac_lags": "HAC lags",
}


def _covariance_note(method: Any, nobs: Any, used: Any = None) -> str:
    """How the standard errors were computed, when not the conventional ones: '; HAC standard errors (...)'."""
    from .covariance import Cluster, default_hac_lags

    cov = getattr(method, "cov", "classic")
    if isinstance(cov, Cluster):
        return f"; standard errors clustered by {' and '.join(map(str, cov.levels))}"
    if cov == "white":  # least squares: HC1; maximum likelihood: the sandwich, with no small-sample factor
        return "; quasi-ML (sandwich) standard errors" if getattr(method, "kind", "") == "ml" else "; White (HC1) standard errors"
    if cov == "hac":
        lags = int(used) if used is not None else getattr(method, "hac_lags", None)
        if lags is None and nobs:
            lags = default_hac_lags(int(nobs))
        plain = "" if getattr(method, "hac_correction", True) else ", without the n / (n - k) correction"
        return (f"; HAC standard errors (Newey-West, Bartlett kernel, {lags} lags{plain})" if lags is not None
                else f"; HAC standard errors (Newey-West{plain})")
    if cov in ("robust", "sandwich"):
        return "; robust (sandwich) standard errors"
    return ""
