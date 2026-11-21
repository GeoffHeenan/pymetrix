"""Quantile regression (Koenker and Bassett 1978), computed by statsmodels' QuantReg.

    Equation("wage ~ educ + exper").fit(data, method=Quantile(q=0.9))     # the 90th percentile
    quantile_process(Equation("wage ~ educ + exper"), data).plot()         # coefficients across quantiles

Least squares models the conditional mean; quantile regression models a conditional quantile,
by minimising the sum of absolute residuals weighted q above and 1 - q below the line. It is
robust to outliers in the dependent variable and shows whether a regressor matters more at
the top or the bottom of the distribution (for example, whether downside risks to growth
depend on financial conditions: "growth at risk"). Standard errors are statsmodels' robust
sandwich by default (one kernel density estimate at zero, with each observation weighted by
the sign of its residual), or the iid form.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd

from .._api import api
from ..errors import SpecificationError
from ..adapters import require
from ..adapters import statsmodels as statsmodels_adapter
from ..adapters.base import AdapterSpec, BackendOptions, check_backend_options, with_backend
from ..forms import Widget

if TYPE_CHECKING:
    from ..data import Dataset
    from .equation import Equation
    from .fitted import FittedEquation

__all__ = ["Quantile", "QuantileProcess", "quantile_process"]


@dataclass(frozen=True)
class Quantile:
    """Quantile regression at quantile ``q`` (0.5: median regression).

    A coefficient is the regressor's effect on the q-th quantile of the outcome given the regressors, not on its
    mean; estimating several quantiles (``quantile_process``) shows whether a regressor shifts the whole
    distribution or stretches one tail. The estimates are robust to outliers in the dependent variable, not to
    outliers in the regressors. The standard errors rest on an estimate of the errors' density at the quantile:
    the kernel and the bandwidth rule set that estimate and change the standard errors only, which become less
    reliable at extreme quantiles in small samples, where few observations lie beyond the line. The pseudo
    R-squared is one minus the ratio of the check loss to that of the quantile alone; it is not comparable with
    a least-squares R-squared.
    """

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import Dataset, Equation, Quantile

        card = Dataset.example("card")
        eq = Equation("lwage ~ educ + exper + black")
        print(eq.fit(card, method=Quantile()).summary())         # median regression: effects on the median log wage
        print(eq.fit(card, method=Quantile(q=0.9)).summary())    # on the 90th percentile: compare educ and black with the median
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("coef, std err, t", ("effects on the q-th conditional quantile of the outcome; compare them across q to see "
                              "which part of the distribution a regressor moves")),
        ("Quantile q; standard errors", ("the note gives q and how the standard errors were made: robust or iid, with "
                                         "the kernel and bandwidth of the density estimate")),
        ("Pseudo R-squared", ("one minus the check loss relative to that of the quantile alone; not comparable with a "
                              "least-squares R-squared")),
        ("Log likelihood, criteria", ("Gaussian values from the residuals, as for least squares; they are not the "
                                      "quantile objective, so do not choose q or models by them")),
        ("fit.stats", "q, pseudo_r2, ssr, se_regression, loglike, aic, bic, hq, nobs and df_resid"),
    )

    label: ClassVar[str] = "Quantile regression"
    kind: ClassVar[str] = "quantile"
    topics: ClassVar[tuple[str, ...]] = ("quantile-regression",)
    references: ClassVar[tuple[str, ...]] = (
        "Koenker, R. and Bassett, G. (1978). Regression quantiles. Econometrica 46(1), 33-50. doi:10.2307/1913643",
    )
    requires: ClassVar[tuple[str, ...]] = ("statsmodels",)
    wraps: ClassVar[str] = "statsmodels.regression.quantile_regression.QuantReg"

    q: Annotated[float, Widget("number", min=0.01, max=0.99, step=0.05, help="the quantile, e.g. 0.5 for the median")] = 0.5
    vcov: Annotated[Literal["robust", "iid"], Widget("choice", help="robust: sandwich weighting each residual by its sign; iid: assumes identically distributed errors")] = "robust"
    kernel: Annotated[Literal["epa", "gau", "cos", "par"], Widget("choice", advanced=True, help="kernel of the error-density estimate behind the standard errors: Epanechnikov, Gaussian, cosine, Parzen")] = "epa"
    bandwidth: Annotated[Literal["hsheather", "bofinger", "chamberlain"], Widget("choice", advanced=True, help="bandwidth rule of that density estimate (Hall-Sheather, Bofinger or Chamberlain); affects the standard errors only")] = "hsheather"
    backend_options: Annotated[BackendOptions, Widget("", advanced=True, help="other statsmodels arguments; model.x for the constructor")] = field(default_factory=dict)

    adapter_spec: ClassVar[AdapterSpec] = statsmodels_adapter.QUANTREG

    def __post_init__(self) -> None:
        check_backend_options(self)
        if not 0 < self.q < 1:
            raise SpecificationError("q must be between 0 and 1")

    def estimate(self, eq: Equation, data: Dataset, *, span: Any = None, where: str | None = None, weights: str | None = None,
                 missing: str = "drop") -> FittedEquation:
        from .design import linear_design, ls_stats, provenance
        from .fitted import FittedEquation
        from .params import point

        if weights:
            raise SpecificationError("quantile regression does not take weights")
        if eq.form != "linear" or eq.fixed_effects:
            raise SpecificationError("quantile regression needs a linear-form equation without fe()")
        qr = require("statsmodels.regression.quantile_regression")
        d = linear_design(eq, data, span, where, None, missing)
        with statsmodels_adapter.ADAPTER.run(self.adapter_spec, self) as run:
            model = run.call("model", qr.QuantReg, d.y, d.X)
            res = run.call("fit", model.fit, q=self.q, vcov=self.vcov, kernel=self.kernel, bandwidth=self.bandwidth)
        beta = np.asarray(res.params, float)
        u = d.y - d.X @ beta
        n, k = d.X.shape
        stats = ls_stats(d.y, u, None, k, n - k, eq.intercept, data.kind == "time_series", False)
        stats.pop("r2", None), stats.pop("r2_adj", None)
        stats.update({"q": self.q, "pseudo_r2": float(res.prsquared), "df_resid": float(n - k)})
        prov = with_backend(provenance(eq, self, data, d.index, d.dropped), run.record)
        return FittedEquation(eq, point(d.names, beta, np.asarray(res.cov_params(), float)), self,
                              resid=pd.Series(u, index=d.index, name="resid"), fitted=pd.Series(d.y - u, index=d.index, name="fitted"),
                              stats=stats, provenance=prov, design=pd.DataFrame(d.X, index=d.index, columns=d.names),
                              extra={"notes": [f"Quantile {self.q:g}; standard errors {self.vcov} ({self.kernel} kernel, {self.bandwidth} bandwidth)"]})


@dataclass(frozen=True, eq=False)
class QuantileProcess:
    """Coefficients across quantiles, with OLS for comparison."""

    fits: dict[float, Any]
    ols: Any
    alpha: float = 0.1

    def table(self) -> Any:
        from ..output import Table

        names = list(self.ols.params.names)
        rows = []
        for q, fit in self.fits.items():
            rows.append([q, *[fit.params[n] for n in names]])
        rows.append(["OLS", *[self.ols.params[n] for n in names]])
        return Table(["quantile", *names], rows, title="Coefficients by quantile")

    def plot(self, coefficients: Sequence[str] | None = None) -> Any:
        """Each coefficient across quantiles with a (1 - alpha) band, and the OLS estimate as a dashed line."""
        from scipy import stats as sps

        from ..figures import Band, Figure, Line, Panel, Rule

        names = list(coefficients) if coefficients else list(self.ols.params.names)
        z = sps.norm.ppf(1 - self.alpha / 2)
        qs = pd.Index(list(self.fits), name="quantile")
        panels = []
        for n in names:
            est = pd.Series([f.params[n] for f in self.fits.values()], index=qs)
            se = pd.Series([f.params.se[list(f.params.names).index(n)] for f in self.fits.values()], index=qs)
            panels.append(Panel((Band(est - z * se, est + z * se, f"{1 - self.alpha:.0%}"), Line(est, "quantile regression"),
                                 Rule(float(self.ols.params[n]), style="--", color="#D55E00")), title=n, xlabel="quantile"))
        return Figure(tuple(panels), columns=2 if len(panels) > 1 else 1)

    @property
    def provenance(self) -> Any:
        """Least squares' provenance (every quantile is fitted on the same sample), with the quantiles as the method and
        each fit's warnings, once, with the quantiles where they arose."""
        from dataclasses import replace

        prov = self.ols.provenance
        if prov is None:
            return None
        where: dict[str, list[str]] = {}
        for q, fit in self.fits.items():
            for text in (fit.provenance.warnings if fit.provenance is not None else ()):
                where.setdefault(text, []).append(f"{q:g}")
        first = next(iter(self.fits.values()), None)
        return replace(prov, method={"type": "quantile process", "quantiles": [float(q) for q in self.fits],
                                     **({"refit": {k: v for k, v in first.provenance.method.items() if k != "q"}}
                                        if first is not None and first.provenance else {})},
                       warnings=(*prov.warnings, *(f"{t} (quantile{'s' if len(qs) > 1 else ''} {', '.join(qs)})" for t, qs in where.items())))

    def __repr__(self) -> str:
        return f"<QuantileProcess {len(self.fits)} quantiles: {self.ols.equation.text}>"


@api("pymetrix:quantile_process")
def quantile_process(eq: Equation, data: Dataset, quantiles: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                     span: Any = None, vcov: str = "robust", alpha: float = 0.1) -> QuantileProcess:
    """Quantile regressions at each of ``quantiles``, with least squares for comparison."""
    from .methods import OLS

    fits = {float(q): eq.fit(data, span=span, method=Quantile(q=q, vcov=vcov)) for q in quantiles}  # type: ignore[arg-type]
    return QuantileProcess(fits, eq.fit(data, span=span, method=OLS()), alpha)


def _register() -> None:
    from ..plugins import register

    register("methods", Quantile, name="quantile")


_register()
