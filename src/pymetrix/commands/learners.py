"""Dialogs for machine-learning fits: SHAP values, and conformal prediction intervals (Estimate → Machine learning)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session

_EQUATIONS = "FittedLearner,FittedEquation"


class _Call(Command):
    call_id: ClassVar[str] = ""

    def _arguments(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_call(self) -> Any:
        from ..calls import make_call

        return make_call(self.call_id, output=self.name, recorded_by=self.command_id, **self._arguments())  # type: ignore[attr-defined]

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/SHAP values...")
@dataclass(frozen=True)
class ExplainLearner(_Call):
    """SHAP values: how much each regressor moves each of a learner's predictions away from the average prediction.

    Each prediction is split into a base value (the average prediction) and one contribution per regressor, the
    Shapley values of the regressors; the contributions add up to the prediction exactly. Random forests, boosted
    trees, LightGBM and XGBoost use TreeSHAP, which is exact and needs no sampling. Neural networks and nearest
    neighbours are explained from their predictions, with absent regressors drawn from a background sample of the
    training periods: every subset of regressors is evaluated when there are at most ten, otherwise a seeded sample
    of orderings. The summary ranks the regressors by their mean absolute contribution, beside permutation
    importance in the fit's own window. SHAP values describe what the fitted model does, not causal effects; when
    regressors are correlated, how the credit is shared depends on how an absent regressor is filled in.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset, Equation, RandomForest, shap_values

        data = Dataset.example("sample_macro")
        fit = Equation("dlog(c) ~ dlog(y) + dlog(c)[-1] + dlog(i)[-1]").fit(data, method=RandomForest(trees=50))
        sv = shap_values(fit)               # TreeSHAP: exact, for each period of the sample
        print(sv.summary())                 # mean |SHAP| by regressor
        sv.values.tail()                    # each regressor's contribution, period by period
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Summary", "mean |SHAP|, mean SHAP and its standard deviation by regressor, largest first, and how they were computed"),
        ("Graph", "mean |SHAP| by regressor, and each period's contributions stacked, with the prediction minus the base value"),
        ("Values", "each regressor's contribution in each period, with the base value and the prediction"),
    )
    topics: ClassVar[tuple[str, ...]] = ("shap-values",)
    call_id: ClassVar[str] = "pymetrix:shap_values"

    fit: Annotated[str, Widget("object", source="FittedLearner", help="a fitted learner: random forest, boosted trees, LightGBM, XGBoost, "
                                                                     "neural network or nearest neighbours")]
    data: Annotated[str, Widget("object", source="Dataset", help="the data for periods outside the sample; empty: the fit's own sample")] = ""
    start: Annotated[str, Widget("period", help="the first period explained; empty: the first of the sample or the data")] = ""
    end: Annotated[str, Widget("period", help="the last period explained; empty: the last")] = ""
    explainer: Annotated[Literal["auto", "tree", "exact", "permutation"], Widget(
        "choice", help="auto: TreeSHAP for tree models, else exact up to ten regressors and permutation beyond")] = "auto"
    background: Annotated[int, Widget("number", min=1, max=10000, advanced=True,
                                      help="training periods drawn for absent regressors (not TreeSHAP); all when the sample is no larger")] = 100
    evaluations: Annotated[int, Widget("number", min=0, max=1_000_000, advanced=True,
                                       help="the permutation explainer's model evaluations per period; 0: 500, or more with many regressors")] = 0
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, help="seed of the background sample and the orderings")] = 0
    name: str = "shap"

    def validate(self, session: Session) -> list[str]:
        missing = [n for n in (self.fit, self.data) if n and n not in session.objects]
        return [f"no object named {n!r}" for n in missing]

    def _arguments(self) -> dict[str, Any]:
        from ..codec import Ref

        out: dict[str, Any] = {"fit": Ref(self.fit)}
        if self.data:
            out["data"] = Ref(self.data)
        if self.start or self.end:
            out["span"] = (self.start or None, self.end or None)
        if self.explainer != "auto":
            out["explainer"] = self.explainer
        for key, default in (("background", 100), ("evaluations", 0), ("seed", 0)):
            if getattr(self, key) != default:
                out[key] = getattr(self, key)
        return out


@command(menu="Estimate/Conformal intervals...")
@dataclass(frozen=True)
class ConformalForecast(_Call):
    """Conformal prediction intervals: intervals that cover the outcome with a chosen probability whatever the model, calibrated on its own errors.

    The last part of the estimation sample (Calibration) is held back: the equation is estimated again on the
    periods before it, with the same method and settings, and its errors on the held-back periods set the width.
    Split conformal takes the interval as the prediction plus or minus the k-th smallest absolute error, k =
    ceil((n + 1)(1 - alpha)) of n; it covers with probability at least 1 - alpha if the held-back periods and the new
    ones are exchangeable. Conformalised quantile regression (boosted trees, LightGBM, XGBoost) calibrates the band
    between two quantile learners instead, so the width changes with the regressors.

    Time series are rarely exchangeable: trends, breaks and changing volatility make past errors a poor guide, and
    the coverage can fall short. Adaptive conformal inference (method aci) goes through the periods in order and,
    after each outcome, widens the next interval if it missed and narrows it if it covered; over T periods the share
    of misses is within (max(alpha, 1 - alpha) + gamma) / (gamma T) of alpha whatever the data, though not in each
    period. The intervals are for static predictions: each period's regressors as observed (one step ahead).
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset, Equation, Ridge, conformal

        data = Dataset.example("sample_macro")
        fit = Equation("dlog(c) ~ dlog(y) + dlog(c)[-1]").fit(data, span=(None, "2015Q4"), method=Ridge(alpha=1.0))
        ci = conformal(fit, data, ("2016Q1", "2019Q4"), alpha=0.1)              # split conformal, 90%
        print(ci.summary())                                                     # the score quantile, coverage, width
        conformal(fit, data, ("2016Q1", "2019Q4"), alpha=0.1, method="aci").frame  # the level adapts to misses
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Summary", "the method, the calibration periods, the rank and size of the score quantile, the coverage where outcomes are known, the average width"),
        ("Graph", "the prediction with its interval, and the outcome"),
        ("Intervals", "for each period: prediction, lower, upper, outcome, whether it was covered (and the adaptive level)"),
        ("Scores", "the calibration periods' conformity scores"),
    )
    topics: ClassVar[tuple[str, ...]] = ("conformal-prediction",)
    call_id: ClassVar[str] = "pymetrix:conformal"

    fit: Annotated[str, Widget("object", source=_EQUATIONS, help="an estimated equation: a learner, shrinkage or least squares")]
    start: Annotated[str, Widget("period", help="the first period with an interval")]
    end: Annotated[str, Widget("period", help="the last period with an interval")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    alpha: Annotated[float, Widget("number", min=0.001, max=0.5, step=0.01, help="1 - alpha is the coverage: 0.1 gives 90% intervals")] = 0.1
    method: Annotated[Literal["split", "cqr", "aci"], Widget(
        "choice", help="split: absolute errors; cqr: quantile learners calibrated (boosted trees, LightGBM, XGBoost); "
                       "aci: adaptive, for series that are not exchangeable")] = "split"
    calibration: Annotated[float, Widget("number", min=0.05, max=0.9, step=0.05, help="the share of the sample, at its end, held back to calibrate")] = 0.25
    gamma: Annotated[float, Widget("number", min=0.0001, max=0.5, step=0.001, advanced=True, show_if=("method", ("aci",)),
                                   help="aci: how far the level moves after each period")] = 0.005
    window: Annotated[int, Widget("number", min=0, max=100_000, advanced=True, show_if=("method", ("aci",)),
                                  help="aci: the latest scores used; 0: as many as the calibration periods")] = 0
    name: str = "intervals"

    def validate(self, session: Session) -> list[str]:
        missing = [n for n in (self.fit, self.data) if n not in session.objects]
        return [f"no object named {n!r}" for n in missing] + ([] if self.start and self.end else ["give the first and last periods"])

    def _arguments(self) -> dict[str, Any]:
        from ..codec import Ref

        out: dict[str, Any] = {"fit": Ref(self.fit), "data": Ref(self.data), "span": (self.start, self.end)}
        for key, default in (("alpha", 0.1), ("method", "split"), ("calibration", 0.25)):
            if getattr(self, key) != default:
                out[key] = getattr(self, key)
        if self.method == "aci":
            for key, default in (("gamma", 0.005), ("window", 0)):
                if getattr(self, key) != default:
                    out[key] = getattr(self, key)
        return out
