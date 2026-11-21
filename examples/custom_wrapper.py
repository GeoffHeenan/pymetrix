"""Wrapping another library: two statsmodels features added to pymetrix as a plugin.

pymetrix's rule for wrappers: pymetrix's specification, data handling and result types, the library's numbers.
This file shows the two common shapes (``docs/wrappers.md`` explains each step):

1. ``Huber``: an *estimation method* for equations (robust regression, statsmodels' ``RLM``). It gets pymetrix's
   formulas, lags, samples and missing values for free, and returns a ``FittedEquation``, so tests, forecasts,
   backtests, reports and the Estimate dialog all work as for least squares.
2. ``Theta``: a *forecasting model* with its own specification and fitted result (statsmodels' ``ThetaModel``).
   ``spec.fit(data, span=...)`` and ``fit.forecast(data, span)`` are all a backtest needs.

Run it from the repository folder:  python examples/custom_wrapper.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd
from scipy import stats

from pymetrix import Dataset, Equation, toolkit
from pymetrix.adapters import backend, require
from pymetrix.commands import Command, command
from pymetrix.errors import SpecificationError
from pymetrix.forms import Widget
from pymetrix.metadata import Provenance
from pymetrix.plugins import PluginInfo, plugin_context, register

# -- who provides these extensions -------------------------------------------------------------

PLUGIN = PluginInfo(
    name="pymetrix-example-wrappers",
    version="0.1",
    requires={"statsmodels": ">=0.14"},
    license="BSD-3-Clause",
    maintainer="An example",
    description="Huber robust regression and the Theta forecasting method, from statsmodels",
    install="pip install statsmodels",
)


# -- 1. an estimation method --------------------------------------------------------------------


@dataclass(frozen=True)
class Huber:
    """Robust regression with Huber's weights (an M-estimator), which downweights outliers.

    Observations whose residual is more than ``t`` robust standard deviations from zero get less weight, so a few
    large errors (a strike, a data error) move the estimates less than they move least squares."""

    label: ClassVar[str] = "Huber robust regression"  # its name in the Estimate dialog's method list
    kind: ClassVar[str] = "ls"
    wraps: ClassVar[str] = "statsmodels.robust.robust_linear_model.RLM"  # the help page shows its documentation
    docs: ClassVar[str] = "https://www.statsmodels.org/stable/rlm.html"
    topics: ClassVar[tuple[str, ...]] = ("least-squares",)
    requires: ClassVar[tuple[str, ...]] = ("statsmodels",)

    t: Annotated[float, Widget("number", min=0.1, max=10, step=0.1, help="where downweighting starts, in robust standard deviations")] = 1.345
    scale: Annotated[Literal["mad", "huber"], Widget("choice", advanced=True,
                                                     help="the robust scale: median absolute deviation, or Huber's proposal 2")] = "mad"
    max_iter: Annotated[int, Widget("number", min=1, max=1000, advanced=True, help="iterations of reweighted least squares")] = 50

    def estimate(self, eq, data, *, span=None, where=None, weights=None, missing="drop"):
        sm = require("statsmodels.api")  # imported on first use; a missing library gets an install hint
        if weights is not None:
            raise SpecificationError("Huber regression does not take weights")
        d = toolkit.design(eq, data, span=span, where=where, missing=missing)  # formulas, lags, sample: pymetrix's
        scale_est = "mad" if self.scale == "mad" else sm.robust.scale.HuberScale()  # pymetrix's words, the library's objects
        res = sm.RLM(d.y, d.X, M=sm.robust.norms.HuberT(t=self.t)).fit(maxiter=self.max_iter, scale_est=scale_est)
        stats_ = {**toolkit.fit_stats(d.y, res.resid, d.k, df_resid=d.df_resid, intercept="const" in d.names, time_series=True),
                  "scale": float(res.scale), "iterations": float(res.fit_history["iteration"])}
        return toolkit.result(eq, data, d, self, res.params, res.cov_params(), res.resid, stats=stats_,
                              extra={"weights": pd.Series(res.weights, index=d.index, name="weight")},
                              backend=backend("statsmodels"))  # provenance records the library and its version


# -- 2. a forecasting model ---------------------------------------------------------------------


@dataclass(frozen=True)
class Theta:
    """The Theta method (Assimakopoulos and Nikolopoulos 2000): simple exponential smoothing with half the trend of
    the series added, a strong benchmark in forecasting competitions."""

    label: ClassVar[str] = "Theta method"
    wraps: ClassVar[str] = "statsmodels.tsa.forecasting.theta.ThetaModel"
    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow",)
    requires: ClassVar[tuple[str, ...]] = ("statsmodels",)

    series: Annotated[str, Widget("text", help="the series to forecast")]
    theta: Annotated[float, Widget("number", min=0, max=10, step=0.5, help="weight on the trend line; 2 is the classic method")] = 2.0
    deseasonalize: Annotated[bool, Widget("checkbox", help="take out multiplicative seasonal factors first (quarterly or monthly)")] = False

    def fit(self, data: Dataset, span: Any = None) -> FittedTheta:
        theta_model = require("statsmodels.tsa.forecasting.theta")
        if self.series not in data.names:
            raise SpecificationError(f"no series {self.series!r} in the dataset")
        y = data[self.series][np.asarray(data.sample_mask(span))].dropna()
        if len(y) < 8:
            raise SpecificationError("the Theta method needs at least 8 observations")
        res = theta_model.ThetaModel(y.to_numpy(float), period=4 if y.index.freqstr.startswith("Q") else 12,
                                     deseasonalize=self.deseasonalize).fit()
        prov = Provenance(data_vintages={self.series: data.variables[self.series].vintage}, spec={"what": "Theta", "series": self.series},
                          method={"type": "Theta", "theta": self.theta, "backend": backend("statsmodels"), "plugin": f"{PLUGIN.name} {PLUGIN.version}"},
                          sample=(str(y.index[0]), str(y.index[-1])), nobs=len(y), dropped=0, package_version=PLUGIN.version)
        return FittedTheta(self, res, y, prov)


@dataclass(frozen=True, eq=False)
class FittedTheta:
    """An estimated Theta model."""

    spec: Theta  # a backtest re-estimates with spec.fit(data, span=...)
    result: Any = field(repr=False)
    data: pd.Series = field(repr=False)
    provenance: Any = None

    @property
    def target(self) -> str:  # what a backtest scores the forecasts against
        return self.spec.series

    def forecast(self, data: Any = None, span: tuple[Any, Any] | None = None, alpha: float = 0.05) -> Dataset:
        """Forecasts for the periods of ``span`` after the sample: columns forecast, se, lower and upper."""
        last = self.data.index[-1]
        start, end = (pd.Period(str(s), freq=last.freq) for s in span)
        if start <= last:
            raise SpecificationError(f"the Theta method forecasts after the sample, from {last + 1}")
        steps = (end - last).n
        mean = np.asarray(self.result.forecast(steps, theta=self.spec.theta), float)
        band = self.result.prediction_intervals(steps, theta=self.spec.theta, alpha=alpha)
        z = stats.norm.ppf(1 - alpha / 2)
        se = (band["upper"].to_numpy(float) - band["lower"].to_numpy(float)) / (2 * z)  # statsmodels gives the interval
        frame = pd.DataFrame({"forecast": mean, "se": se, "lower": mean - z * se, "upper": mean + z * se},
                             index=pd.period_range(last + 1, end, freq=last.freq, name="time"))
        return Dataset.from_frame(frame.loc[start:end])

    def summary(self):
        """The library's own summary."""
        return self.result.summary()


# -- registering: the method joins the Estimate dialog; the model gets a menu command ----------------------

register("methods", Huber, plugin=PLUGIN)  # now in the Estimate dialog's method list
register("models", Theta, plugin=PLUGIN)  # help page, conformance checks

with plugin_context(PLUGIN):  # the command's project ID is example_wrappers:fit_theta

    @command(menu="Estimate/Theta method...")
    @dataclass(frozen=True)
    class FitTheta(Command):
        """The Theta method: forecast a series by simple exponential smoothing plus half its trend."""

        series: Annotated[str, Widget("series", depends_on="data", help="the series to forecast")]
        data: Annotated[str, Widget("object", source="Dataset", help="the dataset holding it")] = "data"
        theta: Annotated[float, Widget("number", min=0, max=10, step=0.5, help="weight on the trend line; 2 is the classic method")] = 2.0
        name: str = "theta"

        def run(self, session):
            result = Theta(self.series, self.theta).fit(session[self.data])
            session.objects[self.name] = result  # a new object in the workfile
            return result

        def to_code(self):  # what Export script and the project's replay use
            return f"{self.name} = Theta({self.series!r}, theta={self.theta!r}).fit({self.data})"


if __name__ == "__main__":
    from pymetrix import backtest

    data = Dataset.example("sample_macro").with_series(ly="100*log(y)")
    eq = Equation("c ~ y + c[-1]")
    for label, method in (("least squares", None), ("Huber", Huber())):
        fit = eq.fit(data) if method is None else eq.fit(data, method=method)
        print(f"{label}:", {name: round(float(v), 3) for name, v in zip(fit.params.names, fit.params.values)})

    theta = Theta("ly").fit(data, span=(None, "2020Q4"))
    print(theta.forecast(data, ("2021Q1", "2021Q4")).frame.round(2))
    print(backtest(theta, data, ("2015Q1", "2023Q4"), horizon=4, label="Theta").accuracy(["ly"]))
