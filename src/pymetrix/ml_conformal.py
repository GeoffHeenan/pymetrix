"""Conformal prediction intervals for an estimated equation's predictions: split conformal, conformalised quantile
regression, and adaptive conformal inference for series whose errors change over time.

    ci = conformal(fit, data, ("2023Q1", "2024Q4"), alpha=0.1)            # split conformal, 90% intervals
    ci = conformal(fit, data, span, method="cqr")                         # quantile learners' band, calibrated
    ci = conformal(fit, data, span, method="aci", gamma=0.005)            # the level adapts to recent misses
    ci.summary(); ci.frame                                                # forecast, lower, upper, actual, covered

**Split conformal** (Vovk, Gammerman and Shafer 2005; Lei et al. 2018). The fit's sample is split in time: the
model is estimated again on the first part (training) with the same method and settings, and its absolute errors on
the last part (calibration, n periods) are the conformity scores. With k = ceil((n + 1)(1 - alpha)), the interval
is the prediction plus or minus the k-th smallest score (infinite when k > n: alpha below 1 / (n + 1) needs more
calibration periods). If the calibration periods and the new one are exchangeable, the interval covers the outcome
with probability at least 1 - alpha (and at most 1 - alpha + 1 / (n + 1) when the scores have no ties), whatever the
model and however wrong it is. The width is the same in every period.

**Conformalised quantile regression** (Romano, Patterson and Candès 2019): the same split, with two quantile
learners (the alpha/2 and 1 - alpha/2 quantiles: boosted trees, LightGBM or XGBoost with loss quantile) in place of
one; the scores are max(lower - y, y - upper), and the band is widened (or narrowed) by their k-th smallest. The
width follows the learners' estimate of the spread, so it changes from period to period, with the same guarantee.

**Time series are not exchangeable.** Trends, breaks and changing volatility make the calibration errors a poor
guide to later ones, and the guarantee above does not hold: coverage can fall short exactly when it matters.
**Adaptive conformal inference** (Gibbs and Candès 2021) walks through the periods of the span in order: each
period's interval uses the most recent ``window`` scores at a level alpha_t, and once its outcome is known,
alpha_t+1 = alpha_t + gamma (alpha - err_t), err_t = 1 if it was missed, so misses widen later intervals and
hits narrow them. Whatever the data, the share of misses over T periods is within (max(alpha, 1 - alpha) + gamma) /
(gamma T) of alpha (their Proposition 4.1): coverage holds on average over time, not in each period. Periods of the
span without an outcome (true forecasts) keep the last alpha_t and scores.

Intervals are for static predictions: each period's regressors as observed (lags of the dependent variable
included), the one-step-ahead case. Written natively: split conformal is a sorted list and an index, and the
library that does it (MAPIE) changed its interface at 1.0 and does not take pymetrix's time-ordered refit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from ._api import api
from .errors import SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["ConformalIntervals", "conformal", "conformal_quantile"]

QUANTILE_METHODS = ("GradientBoosting", "LightGBM", "XGBoost")  # learners with a quantile loss, for CQR


def conformal_quantile(scores: Any, alpha: float) -> tuple[float, int]:
    """The split-conformal quantile of the scores and its rank k = ceil((n + 1)(1 - alpha)): the k-th smallest score,
    or infinity when k > n. ``alpha`` at 1 or above gives -infinity (an empty interval)."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    if alpha >= 1:
        return -math.inf, 0
    k = math.ceil((n + 1) * (1 - alpha) - 1e-12)  # the tolerance keeps (n + 1)(1 - alpha) = 18.000000000000004 at 18
    return (float(s[k - 1]) if k <= n else math.inf), k


@dataclass(frozen=True, eq=False)
class ConformalIntervals:
    """Conformal prediction intervals over a span: the prediction, its interval, and the outcome where known."""

    frame: pd.DataFrame  # forecast, lower, upper, actual, covered (and alpha_t for the adaptive method)
    method: str  # "split", "cqr" or "aci"
    alpha: float
    quantile: float  # the k-th smallest calibration score (the last one used, for aci)
    rank: int  # k
    scores: pd.Series = field(repr=False)  # the calibration scores, by period
    training: tuple[str, str] = ("", "")
    dependent: str = ""
    title: str = ""
    gamma: float | None = None
    window: int | None = None
    overlap: int = 0  # periods of the span inside the fit's sample (no guarantee there)

    stage = "forecast"

    @property
    def coverage(self) -> float:
        """The share of the span's periods with an outcome whose interval covered it."""
        c = self.frame["covered"].dropna()
        return float(c.mean()) if len(c) else float("nan")

    def summary(self) -> Table:
        label = {"split": "Split conformal", "cqr": "Conformalised quantile regression", "aci": "Adaptive conformal inference"}[self.method]
        known = int(self.frame["covered"].notna().sum())
        width = (self.frame["upper"] - self.frame["lower"]).replace([np.inf, -np.inf], np.nan)
        rows: list[list[Any]] = [["nominal coverage", 1 - self.alpha], ["training periods", f"{self.training[0]} to {self.training[1]}"],
                                 ["calibration periods", len(self.scores)], ["rank k of the score used", self.rank],
                                 [("last score quantile" if self.method == "aci" else "score quantile"), self.quantile],
                                 ["periods with an outcome", known], ["empirical coverage", self.coverage],
                                 ["average width", float(width.mean()) if width.notna().any() else float("inf")]]
        if self.method == "aci":
            rows.insert(1, ["gamma (step of the level)", self.gamma])
            rows.insert(2, ["score window", self.window])
            rows.append(["last level alpha_t", float(self.frame["alpha_t"].iloc[-1])])
        notes = [f"{label}: {self.dependent}, {self.title}." if self.title else f"{label}: {self.dependent}.",
                 "The model is estimated again on the training periods with the same method and settings; its errors on the "
                 "calibration periods (the last part of the sample) set the width."]
        if self.method == "aci":
            notes.append("The level moves after each period with an outcome: alpha_t+1 = alpha_t + gamma (alpha - miss). Over T periods the "
                         "share of misses is within (max(alpha, 1 - alpha) + gamma) / (gamma T) of alpha, whatever the data.")
        else:
            notes.append("Coverage of at least 1 - alpha holds if the calibration periods and the new ones are exchangeable; trends, "
                         "breaks and changing volatility break that: method 'aci' adapts to them.")
        if math.isinf(self.quantile) and self.quantile > 0:
            notes.append(f"Too few calibration periods for alpha {self.alpha:g}: the interval is unbounded (k > n). "
                         "Use a larger alpha or a larger calibration share.")
        if self.overlap:
            notes.append(f"{self.overlap} periods of the span are inside the estimation sample: no guarantee holds there.")
        return Table(["", "value"], rows, title=f"Conformal intervals ({1 - self.alpha:.0%})", notes=tuple(notes))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.frame

    def plot(self) -> Any:
        from .figures import Band, Figure, Line, Panel

        f = self.frame
        layers: list[Any] = [Band(f["lower"].replace(-np.inf, np.nan), f["upper"].replace(np.inf, np.nan), f"{1 - self.alpha:.0%} interval"),
                             Line(f["forecast"], "forecast")]
        if f["actual"].notna().any():
            layers.append(Line(f["actual"], self.dependent or "actual", color="#222222"))
        return Figure((Panel(tuple(layers), title=f"Conformal intervals: {self.dependent}"),))

    def __repr__(self) -> str:
        return f"<ConformalIntervals {self.method} {1 - self.alpha:.0%}: {len(self.frame)} periods, coverage {self.coverage:.3g}>"


def _sample(fit: Any) -> pd.Index:
    from .ext.sklearn import FittedLearner

    return fit.y.index if isinstance(fit, FittedLearner) else fit.resid.index


def _refit(fit: Any, data: Dataset, train: pd.Index, method: Any = None) -> Any:
    """The fit again on the training periods, with ``method`` (default: the fit's own)."""
    from .ext import sklearn as sk

    method = fit.method if method is None else method
    if isinstance(fit, sk.FittedLearner):  # the same model on fewer rows; no cross-validation needed
        adapter = getattr(method, "backend_adapter", sk.specs.ADAPTER)
        with adapter.run(method.adapter_spec, method) as run:
            model = method._model(run)
            model.fit(fit.X.loc[train].to_numpy(float), fit.y.loc[train].to_numpy(float))
        return replace(fit, method=method, model=model, y=fit.y.loc[train], X=fit.X.loc[train], cv_rmse=float("nan"),
                       cv_resid=pd.Series(dtype=float, name="cv error"))
    return fit.equation.fit(data, span=(str(train[0]), str(train[-1])), method=method)


def _predict(fit: Any, data: Dataset, periods: pd.Index) -> np.ndarray:
    span = (str(periods[0]), str(periods[-1]))
    return fit.predict(data, span).reindex(periods).to_numpy(float)


def _lhs(fit: Any, data: Dataset, periods: pd.Index) -> np.ndarray:
    from .estimation.design import as_series
    from .expr import evaluate

    return as_series(evaluate(fit.equation.lhs, data.lagged), data.frame.index).reindex(periods).to_numpy(float)


@api("pymetrix:conformal")
def conformal(fit: Any, data: Dataset, span: tuple[Any, Any], alpha: float = 0.1, method: Literal["split", "cqr", "aci"] = "split",
              calibration: float = 0.25, gamma: float = 0.005, window: int = 0) -> ConformalIntervals:
    """Conformal (1 - alpha) prediction intervals over ``span`` for an estimated equation (a learner, shrinkage or least
    squares): ``method`` ``"split"`` (absolute errors), ``"cqr"`` (conformalised quantile regression, for boosted
    trees, LightGBM and XGBoost) or ``"aci"`` (adaptive conformal inference, for series that are not exchangeable,
    with step ``gamma`` and the latest ``window`` scores, 0: as many as the calibration periods). The last
    ``calibration`` share of the fit's sample calibrates a model estimated again on the periods before it."""
    from .ext.sklearn import FittedLearner

    if not hasattr(fit, "equation") or not hasattr(fit, "predict"):
        raise SpecificationError("conformal intervals are for estimated equations (Equation(...).fit(...))")
    if method not in ("split", "cqr", "aci"):
        raise SpecificationError(f"method is 'split', 'cqr' or 'aci', not {method!r}")
    if not 0 < alpha < 1:
        raise SpecificationError("alpha is between 0 and 1")
    if not 0 < calibration < 1:
        raise SpecificationError("calibration is the share of the sample kept for calibration: between 0 and 1")
    if method == "aci" and not gamma > 0:
        raise SpecificationError("gamma, the step of the adaptive level, is above 0")
    if window < 0:
        raise SpecificationError("window is 0 (all calibration periods) or more")
    if method == "cqr" and not (isinstance(fit, FittedLearner) and type(fit.method).__name__ in QUANTILE_METHODS):
        raise SpecificationError("conformalised quantile regression needs a learner with a quantile loss: boosted trees, LightGBM or XGBoost")
    sample = _sample(fit)
    n = len(sample)
    n_cal = int(round(calibration * n))
    if n_cal < 1 or n - n_cal < 2:
        raise SpecificationError(f"{n} periods cannot be split into training and {n_cal} calibration periods: change the calibration share")
    train, cal = sample[: n - n_cal], sample[n - n_cal:]
    mask = np.asarray(data.sample_mask(span))
    periods = data.frame.index[mask]
    if len(periods) == 0:
        raise SpecificationError("the span has no periods")
    point = _refit(fit, data, train)
    y_cal, y_new = _lhs(fit, data, cal), _lhs(fit, data, periods)
    yhat_cal, yhat = _predict(point, data, cal), _predict(point, data, periods)
    frame = pd.DataFrame({"forecast": yhat, "lower": np.nan, "upper": np.nan, "actual": y_new, "covered": np.nan}, index=periods)
    extra: dict[str, Any] = {}
    if method == "cqr":
        lo_fit = _refit(fit, data, train, replace(fit.method, loss="quantile", quantile=alpha / 2))
        hi_fit = _refit(fit, data, train, replace(fit.method, loss="quantile", quantile=1 - alpha / 2))
        lo_cal, hi_cal = _predict(lo_fit, data, cal), _predict(hi_fit, data, cal)
        scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
        q, k = conformal_quantile(scores, alpha)
        frame["lower"], frame["upper"] = _predict(lo_fit, data, periods) - q, _predict(hi_fit, data, periods) + q
    else:
        scores = np.abs(y_cal - yhat_cal)
        q, k = conformal_quantile(scores, alpha)
        if method == "split":
            frame["lower"], frame["upper"] = yhat - q, yhat + q
        else:
            size = int(window) or n_cal
            pool, level, levels, used = list(scores), alpha, [], []
            for i in range(len(periods)):
                q, k = conformal_quantile(pool[-size:], level)
                levels.append(level)
                used.append(q)
                frame.iloc[i, frame.columns.get_loc("lower")] = yhat[i] - q if level < 1 else yhat[i]
                frame.iloc[i, frame.columns.get_loc("upper")] = yhat[i] + q if level < 1 else yhat[i]
                if np.isfinite(y_new[i]) and np.isfinite(yhat[i]):
                    miss = 1.0 if level >= 1 else float(abs(y_new[i] - yhat[i]) > q)  # alpha_t >= 1: the empty set misses
                    frame.iloc[i, frame.columns.get_loc("covered")] = 1.0 - miss
                    level = level + gamma * (alpha - miss)
                    pool.append(abs(y_new[i] - yhat[i]))
            frame["alpha_t"] = levels
            extra = {"gamma": float(gamma), "window": size}
    if method != "aci":
        known = np.isfinite(y_new) & np.isfinite(yhat)
        frame.loc[known, "covered"] = ((y_new >= frame["lower"].to_numpy()) & (y_new <= frame["upper"].to_numpy()))[known].astype(float)
    overlap = int(periods.isin(sample).sum())
    return ConformalIntervals(frame, method, float(alpha), float(q), int(k), pd.Series(scores, index=cal, name="score"),
                              (str(train[0]), str(train[-1])), fit.equation.require_dependent(), f"{fit.equation.text}, {fit.method.label}",
                              overlap=overlap, **extra)
