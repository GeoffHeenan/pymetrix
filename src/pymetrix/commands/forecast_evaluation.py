"""Forecast-evaluation dialogs beside Compare many forecasts: Hansen's test for superior predictive ability against a
benchmark backtest, and the scores of probability forecasts of a 0/1 outcome."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from .forecasting import _LINEX, _LOSS, _TAU, _backtests, _names
from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session

_TOPICS: tuple[str, ...] = ("forecast-workflow",)


@command(menu="Forecast/Superior predictive ability...")
@dataclass(frozen=True)
class SuperiorPredictiveAbility(Command):
    """Hansen's test for superior predictive ability: does any of several backtests forecast better than a benchmark?
    Needs the arch package.

    The null is that no model has lower expected loss than the benchmark, at the horizon chosen, on the targets all the
    backtests forecast. The p-value is Hansen's consistent one, from a stationary bootstrap of the loss differentials;
    the lower and upper bounds are in the details (the upper is White's Reality Check, which a poor model in the set
    makes conservative, the problem Hansen's version corrects). A small p-value says at least one model beats the
    benchmark; it does not say which: the details name the one with the lowest average loss, and Compare many forecasts
    (the model confidence set) separates them. Losses are serially correlated at longer horizons, which the block
    bootstrap allows for; the default mean block length is the larger of the horizon and n^(1/3).
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset, Equation, backtest

        data = Dataset.example("sample_macro")
        origins = ("2017Q1", "2022Q4")      # the same origins for every backtest (Forecast > Backtest)
        ar = backtest(Equation("c ~ c[-1]").fit(data), data, origins, horizon=1)                 # the benchmark
        with_y = backtest(Equation("c ~ c[-1] + y[-1]").fit(data), data, origins, horizon=1)
        with_i = backtest(Equation("c ~ c[-1] + i[-1]").fit(data), data, origins, horizon=1)
        result = ar.spa({"with_y": with_y, "with_i": with_i}, "c", horizon=1, reps=500)      # does either beat it?
        result.p_value, result.details["lowest average loss"]
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("statistic, p-value", "the largest studentised loss advantage over the benchmark, and Hansen's consistent p-value"),
        ("p-value, lower and upper", "the bounds of the p-value; the upper is White's Reality Check"),
        ("lowest average loss", "the model with the lowest average loss"),
    )
    topics: ClassVar[tuple[str, ...]] = _TOPICS

    benchmark: Annotated[str, Widget("object", source="Backtest", help="the backtest to beat, e.g. a random walk or the current model")]
    models: Annotated[str, Widget("object", source="Backtest", multiple=True, help="the competing backtests, over the same origins")]
    variable: Annotated[str, Widget("text", help="the variable they all forecast")]
    horizon: Annotated[int, Widget("number", min=1, max=40, help="which horizon to compare; also the shortest default bootstrap block")] = 1
    loss: Annotated[_LOSS, Widget("choice", help="the loss to compare; CRPS and log score need density forecasts")] = "squared"
    tau: Annotated[float, _TAU] = 0.5
    linex_a: Annotated[float, _LINEX] = 1.0
    name: str = "spa_test"
    reps: Annotated[int, Widget("number", min=100, max=100_000, step=100, advanced=True, help="bootstrap resamples for the p-value")] = 1000
    block: Annotated[int | None, Widget("number", min=1, max=200, advanced=True,
                                        help="the stationary bootstrap's mean block length; empty = automatic")] = None
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, help="random seed for the bootstrap")] = 0

    def validate(self, session: Session) -> list[str]:
        problems = _backtests(session, ", ".join([self.benchmark, *_names(self.models)]))
        if self.benchmark in _names(self.models):
            problems.append("the benchmark should not also be among the models")
        return problems + ([] if self.variable else ["name the variable to compare"])

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:backtest.spa", Ref(self.benchmark), output=self.name, recorded_by=self.command_id,
                         models={n: Ref(n) for n in _names(self.models)}, variable=self.variable, horizon=self.horizon, loss=self.loss,
                         tau=self.tau, a=self.linex_a, reps=self.reps, block=self.block, seed=self.seed)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Forecast/Binary forecast scores...")
@dataclass(frozen=True)
class BinaryForecastScores(Command):
    """Scores of probability forecasts of a 0/1 event (a recession, a default): the Brier score and the base rate's,
    the log score, the area under the ROC curve, and the hit rate, false-alarm rate and Kuipers score of forecasting the
    event when the probability reaches a threshold.

    Both series are in one dataset: the forecast probabilities (between 0 and 1) and what happened (0 or 1), over its
    active sample; periods missing either are dropped. The Brier score is lower for better forecasts; compare it with
    the base rate's (always forecasting the share of events), which a useful forecast must beat. The ROC area is the
    chance that an event got a higher probability than a non-event: 0.5 is no skill. The hit and false-alarm rates
    depend on the threshold, and the Kuipers score is their difference.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        import numpy as np
        import pandas as pd
        from pymetrix import Dataset, binary_scores

        rng = np.random.default_rng(0)
        p = rng.uniform(size=80)                                  # forecast probabilities of a recession
        frame = pd.DataFrame({"p_recession": p, "recession": (rng.uniform(size=80) < p).astype(float)},
                             index=pd.period_range("2005Q1", periods=80, freq="Q"))
        data = Dataset.from_frame(frame)
        binary_scores("p_recession", "recession", threshold=0.5, data=data)   # two series of one dataset
        binary_scores(frame["p_recession"], frame["recession"])               # or two pandas series (or arrays)
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Brier score", "lower is better; compare it with the base rate's"),
        ("log score, ROC area", "higher is better; a ROC area of 0.5 is no skill"),
        ("hit rate, false-alarm rate, Kuipers score", "of forecasting the event when the probability reaches the threshold"),
    )
    topics: ClassVar[tuple[str, ...]] = _TOPICS

    probabilities: Annotated[str, Widget("series", depends_on="data", help="the forecast probabilities of the event, between 0 and 1")]
    outcomes: Annotated[str, Widget("series", depends_on="data", help="what happened: 1 for the event, 0 otherwise")]
    data: Annotated[str, Widget("object", source="Dataset", help="the dataset holding both series; its active sample applies")] = "data"
    threshold: Annotated[float, Widget("number", min=0.01, max=0.99, step=0.05,
                                       help="forecast the event when the probability is at least this (hit and false-alarm rates)")] = 0.5
    name: str = "binary_scores_result"

    def validate(self, session: Session) -> list[str]:
        data = session.objects.get(self.data)
        if data is None:
            return [f"no object named {self.data!r}"]
        names = set(getattr(data, "names", ()))
        return [f"{self.data} has no series {n!r}" for n in (self.probabilities, self.outcomes) if n not in names]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:binary_scores", output=self.name, recorded_by=self.command_id, probabilities=self.probabilities,
                         outcomes=self.outcomes, threshold=self.threshold, data=Ref(self.data))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
