"""Markov-switching regressions (Hamilton 1989), computed by statsmodels.

    ms = MarkovSwitching("growth ~ 1", regimes=2, ar=4).fit(data)     # Hamilton's recession model
    ms.probabilities()                    # smoothed probability of each regime, period by period
    ms.transition, ms.durations           # how persistent the regimes are
    MarkovSwitching("infl ~ gap", regimes=2, switching_variance=True)  # the slope and the variance switch

The intercept (and, unless ``switching_exog=False``, the coefficients) differ between unobserved
regimes that follow a Markov chain; ``ar`` adds switching autoregressive terms, and
``switching_variance`` lets the error variance differ too. The likelihood has several local
maxima, so the fit tries ``search`` random starts (from ``seed``) as well as the default one. Regimes
are numbered as estimated: read their means before naming them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

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

__all__ = ["FittedMarkovSwitching", "MarkovSwitching"]


@dataclass(frozen=True)
class MarkovSwitching:
    """``formula``: the dependent variable on regressors (``growth ~ 1`` for a switching mean)."""

    formula: str
    regimes: int = 2
    ar: int = 0
    switching_variance: bool = False
    switching_exog: bool = True
    search: int = 20  # random starting values tried, besides the default
    seed: int = 0
    backend_options: Annotated[BackendOptions, Widget("", advanced=True, help="other statsmodels arguments; model.x for the constructor")] = field(default_factory=dict)

    adapter_spec: ClassVar[AdapterSpec] = statsmodels_adapter.MARKOV

    def __post_init__(self) -> None:
        check_backend_options(self)

    @api("pymetrix:markov_switching.fit")
    def fit(self, data: Dataset, span: Any = None) -> FittedMarkovSwitching:
        from dataclasses import replace

        from ._capture import capture
        from ._capture import provenance_for
        from .estimation import Equation

        with capture() as caught:
            fitted, record, complete = self._fit(data, span)
        r, model = fitted.backend_result, fitted.backend_model
        converged = (getattr(r, "mle_retvals", None) or {}).get("converged")
        frame = pd.concat([pd.Series(np.asarray(model.endog).ravel()), pd.DataFrame(np.asarray(model.exog) if model.exog is not None else None)], axis=1)
        prov = provenance_for(self, data, sorted(Equation(self.formula).variables()), fitted.probabilities().index, frame, caught,
                           {"type": "MarkovSwitching"}, None if converged is None else bool(converged), self.seed, span=span,
                           presample=complete.difference(fitted.probabilities().index))  # with ar > 0, the first ar rows
        return replace(fitted, provenance=with_backend(prov, record))

    def _fit(self, data: Dataset, span: Any = None) -> tuple[FittedMarkovSwitching, Any, pd.Index]:
        from .estimation import Equation
        from .expr import evaluate

        mr = require("statsmodels.tsa.regime_switching.markov_regression")
        ma = require("statsmodels.tsa.regime_switching.markov_autoregression")
        if data.kind != "time_series":
            raise SpecificationError("Markov-switching models need a single time series")
        if self.regimes < 2:
            raise SpecificationError("regimes must be 2 or more")
        eq = Equation(self.formula)
        y = pd.Series(np.asarray(evaluate(eq.lhs, data.lagged), float), index=data.frame.index, name=eq.require_dependent())
        X = pd.DataFrame({t.label: np.asarray(evaluate(t.expr, data.lagged), float) for t in eq.terms}, index=data.frame.index)
        frame = pd.concat([y, X], axis=1)[np.asarray(data.sample_mask(span))].dropna()
        if len(frame) < 20:
            raise SpecificationError("fewer than twenty complete observations")
        endog = frame.iloc[:, 0].set_axis(frame.index.to_timestamp())
        exog = frame.iloc[:, 1:].set_axis(endog.index) if X.shape[1] else None
        trend = "c" if eq.intercept else "n"
        with statsmodels_adapter.ADAPTER.run(self.adapter_spec, self) as run:
            if self.ar:
                model = run.call("model", ma.MarkovAutoregression, endog, self.regimes, self.ar, trend=trend, exog=exog,
                                 switching_exog=self.switching_exog, switching_variance=self.switching_variance)
            else:
                model = run.call("model", mr.MarkovRegression, endog, self.regimes, trend=trend, exog=exog,
                                 switching_exog=self.switching_exog, switching_variance=self.switching_variance)
            import inspect

            if "rng" in inspect.signature(model.fit).parameters:  # statsmodels 0.15 and later
                result = run.call("fit", model.fit, search_reps=self.search, rng=self.seed, disp=False)
            else:  # 0.14 draws from NumPy's global generator: seed it, then put it back as it was
                state = np.random.get_state()
                try:
                    np.random.seed(self.seed)
                    result = run.call("fit", model.fit, search_reps=self.search, disp=False)
                finally:
                    np.random.set_state(state)
        return FittedMarkovSwitching.from_statsmodels(self, model, result, data.frame.index.freqstr, tuple(X.columns)), run.record, frame.index


@dataclass(frozen=True, eq=False)
class FittedMarkovSwitching(BackendResult):
    """An estimated Markov-switching model: its own fields hold every number it reports, read from statsmodels
    when it is made. Parameter names keep statsmodels' regime suffix (``const[0]``) with pymetrix's terms: the
    regressors' own names for ``x1, x2, ...`` and ``g[-1]`` for ``ar.L1``."""

    spec: MarkovSwitching
    params: Any  # PointEstimate
    inference: pd.DataFrame  # coef, std err, z, p>|z| by parameter
    smoothed: pd.DataFrame  # the probability of each regime in each period, from all the data
    filtered: pd.DataFrame  # ... from the data up to that period
    transition: pd.DataFrame  # P[i, j]: the probability of moving to regime j from regime i
    durations: pd.Series  # expected periods spent in each regime once entered: 1 / (1 - P[i, i])
    endog: pd.Series  # the dependent variable over the estimation sample
    stats: Mapping[str, float]
    freq: str = ""
    regressors: tuple[str, ...] = ()
    provenance: Any = None
    backend_result: Any = field(default=None, repr=False, compare=False)  # statsmodels' MarkovSwitchingResults
    backend_model: Any = field(default=None, repr=False, compare=False)

    _backend_fields: ClassVar[tuple[str, ...]] = ("backend_result", "backend_model")
    result = deprecated_alias("backend_result", "result")
    model = deprecated_alias("backend_model", "model")

    @classmethod
    def from_statsmodels(cls, spec: MarkovSwitching, model: Any, result: Any, freq: str, regressors: tuple[str, ...]) -> FittedMarkovSwitching:
        from .adapters.statsmodels import pymetrix_name
        from .estimation.params import point

        target = str(model.endog_names)
        rename = {f"x{i + 1}": name for i, name in enumerate(regressors)}
        sm_names = [str(n) for n in model.param_names]
        names = [pymetrix_name(re.sub(r"^(x\d+)(?=\[)", lambda m: rename.get(m.group(1), m.group(1)), n), target) for n in sm_names]

        def periods(index: pd.Index) -> pd.PeriodIndex:
            return pd.PeriodIndex(index.to_period(freq)) if isinstance(index, pd.DatetimeIndex) else pd.PeriodIndex(index)

        labels = [f"regime {i}" for i in range(spec.regimes)]

        def probs(values: Any) -> pd.DataFrame:
            return pd.DataFrame(np.asarray(values), index=periods(values.index), columns=labels)

        smoothed, filtered = probs(result.smoothed_marginal_probabilities), probs(result.filtered_marginal_probabilities)
        P = np.asarray(result.regime_transition)[:, :, 0].T  # statsmodels stores [to, from]
        transition = pd.DataFrame(P, index=pd.Index(labels, name="from"), columns=pd.Index(labels, name="to"))
        durations = pd.Series(np.asarray(result.expected_durations, float), index=labels, name="expected duration")
        endog = pd.Series(np.asarray(model.endog).ravel()[-len(smoothed):], index=smoothed.index, name=target)
        inference = pd.DataFrame({"coef": np.asarray(result.params, float), "std err": np.asarray(result.bse, float),
                                  "z": np.asarray(result.tvalues, float), "p>|z|": np.asarray(result.pvalues, float)}, index=names)
        stats = {"nobs": float(result.nobs), "k": float(len(names)), "loglike": float(result.llf), "aic": float(result.aic),
                 "bic": float(result.bic)}
        return cls(spec, point(names, np.asarray(result.params, float), np.asarray(result.cov_params(), float), dict(zip(sm_names, names))),
                   inference, smoothed, filtered, transition, durations, endog, stats, freq, tuple(regressors), None, result, model)

    @property
    def names(self) -> list[str]:
        """Parameter names, with statsmodels' x1, x2, ... replaced by the regressors' own names."""
        return list(self.params.names)

    def probabilities(self, kind: str = "smoothed") -> pd.DataFrame:
        """The probability of each regime in each period: ``smoothed`` (all data) or ``filtered`` (data to then)."""
        return self.smoothed if kind == "smoothed" else self.filtered

    def summary(self) -> Table:
        durations = ", ".join(f"{k}: {v:.1f}" for k, v in self.durations.items())
        notes = [f"{self.spec.regimes} regimes" + (f", {self.spec.ar} switching AR lags" if self.spec.ar else "")
                 + ("; switching variance" if self.spec.switching_variance else ""),
                 f"Sample: {self.smoothed.index[0]} to {self.smoothed.index[-1]}, {int(self.stats['nobs'])} observations",
                 f"Log likelihood: {self.stats['loglike']:.3f}", f"Akaike info criterion: {self.stats['aic']:.3f}",
                 f"Expected durations: {durations}",
                 "Regimes are numbered as estimated: read their means before naming them."]
        return Table(list(self.inference.columns), self.inference.to_numpy().tolist(), index=self.names,
                     title=f"Markov switching: {self.spec.formula}", notes=notes)

    def plot(self) -> Any:
        """The series, and the smoothed probability of each regime."""
        from .figures import Band, Figure, Line, Panel

        probs = self.smoothed
        panels = [Panel((Line(self.endog, str(self.endog.name)),), title=str(self.endog.name))]
        for col in probs.columns:
            zero = pd.Series(0.0, index=probs.index)
            panels.append(Panel((Band(zero, probs[col], col, alpha=0.4),), title=f"Probability of {col}"))
        return Figure(tuple(panels))

    def __repr__(self) -> str:
        return f"<FittedMarkovSwitching {self.spec.formula}: {self.spec.regimes} regimes>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()
