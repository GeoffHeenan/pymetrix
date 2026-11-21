"""The Forecast menu: rounds, judgement, the store of rounds, and backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from ..errors import SpecificationError
from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session

_TOPICS: tuple[str, ...] = ("forecast-workflow",)
_LOSS = Literal["squared", "absolute", "lin-lin", "linex", "crps", "log score"]
_TAU = Widget("number", min=0.01, max=0.99, step=0.05, advanced=True, help="lin-lin loss: the weight on under-forecasts (0.5 is symmetric)")
_LINEX = Widget("number", min=-10, max=10, step=0.5, advanced=True, help="linex loss: a > 0 makes under-forecasts costlier, a < 0 over-forecasts")


def _loss_code(loss: str, tau: float, a: float) -> str:
    return f", {loss!r}" + (f", tau={tau}" if loss == "lin-lin" and tau != 0.5 else "") + (f", a={a}" if loss == "linex" and a != 1.0 else "")


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def _cells(text: str) -> dict[str, float]:
    """``"2026Q4=-0.3, 2027Q1=-0.1"`` -> {"2026Q4": -0.3, "2027Q1": -0.1}."""
    out = {}
    for part in _names(text):
        period, sep, value = part.partition("=")
        if not sep:
            raise SpecificationError(f"write each change as period=value, e.g. 2026Q4=-0.3 (not {part!r})")
        out[period.strip()] = float(value)
    return out


@command(menu="Forecast/Make round...")
@dataclass(frozen=True)
class MakeRound(Command):
    """Make a forecast round from a forecast: an equation's, a VAR's or a model solution's.

    A round records the forecast paths with when they were made, the last period of data used, and notes. For an
    equation's forecast, name the variable it forecasts. For a model solution, list the variables to keep.
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS

    forecast: Annotated[str, Widget("object", source="Dataset,Solution", help="a forecast (from Estimate > Forecast) or a model solution")]
    round_name: Annotated[str, Widget("text", help="how the round is known in the store, e.g. 2026Q3 baseline; one name per round")]
    data: Annotated[str, Widget("object", source="Dataset", help="the data the forecast used (sets the last data period)")] = "data"
    variables: Annotated[str, Widget("text", help="variables to keep; empty = all the forecast's")] = ""
    as_name: Annotated[str, Widget("text", help="for an equation's forecast: the variable it forecasts")] = ""
    notes: Annotated[str, Widget("multiline", help="assumptions, story, sign-off")] = ""
    name: str = "forecast_round"

    def validate(self, session: Session) -> list[str]:
        missing = [n for n in (self.forecast, self.data) if n and n not in session.objects]
        return [f"no object named {n!r}" for n in missing] + ([] if self.round_name.strip() else ["give the round a name"])

    def to_call(self, session: Session) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        data = Ref(self.data) if self.data in session.objects else None  # sets the last data period, when there is one
        return make_call("pymetrix:forecast_round.from_forecast", output=self.name, recorded_by=self.command_id, name=self.round_name,
                         forecast=Ref(self.forecast), variables=_names(self.variables) or None, data=data, notes=self.notes,
                         as_name=self.as_name or None)

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        args = [repr(self.round_name), self.forecast]
        if _names(self.variables):
            args.append(f"variables={_names(self.variables)!r}")
        if self.data:
            args.append(f"data={self.data}")
        if self.notes:
            args.append(f"notes={self.notes!r}")
        if self.as_name:
            args.append(f"as_name={self.as_name!r}")
        return f"{self.name} = ForecastRound.from_forecast({', '.join(args)})"


@command(menu="Forecast/Adjust round...")
@dataclass(frozen=True)
class AdjustRound(Command):
    """Judgement: change a round's path of one variable in some periods, with the reason.

    Write the changes as period=value, separated by commas (2026Q4=-0.3, 2027Q1=-0.1). Add adds them to the path;
    Set replaces it. Every change is logged with the path before it.
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS

    round: Annotated[str, Widget("object", source="ForecastRound", help="the round to adjust; the path before judgement is kept")]
    variable: Annotated[str, Widget("text", help="one of the round's variables; only its path changes, not others linked to it")]
    changes: Annotated[str, Widget("text", help="period=value, ... e.g. 2026Q4=-0.3, 2027Q1=-0.1")]
    reason: Annotated[str, Widget("text", help="why: part of the forecast's record")]
    how: Annotated[Literal["add", "set"], Widget("choice", help="add to the path, or set it")] = "add"

    def validate(self, session: Session) -> list[str]:
        try:
            _cells(self.changes)
        except ValueError as exc:
            return [str(exc)]
        return [] if self.reason.strip() else ["give the reason for the adjustment"]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:forecast_round.adjust", Ref(self.round), output=self.round, recorded_by=self.command_id,
                         variable=self.variable, values=_cells(self.changes), reason=self.reason, how=self.how)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Forecast/Save round...")
@dataclass(frozen=True)
class SaveRound(Command):
    """Save a round in a forecast store: a folder that keeps every round, for comparison and evaluation.

    Each round is written as a Parquet file of its paths and a JSON record of when it was made, the last period of
    data used, its notes and every judgemental adjustment. Keeping each round unchanged is what lets you later compare
    rounds, chart how forecasts of a variable evolved, and judge the forecasts against what happened. Saving the same
    round again does nothing; a different round under an existing name is refused unless Overwrite is ticked, which
    loses the earlier record.
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS

    round: Annotated[str, Widget("object", source="ForecastRound", help="the round to save, under its own name")]
    folder: Annotated[str, Widget("text", help="the store's folder, e.g. forecasts; keep one per forecast process so rounds stay comparable")] = "forecasts"
    overwrite: Annotated[bool, Widget("checkbox", help="replace a different round saved under the same name; its earlier record is lost")] = False

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        from ..forecasting import ForecastStore

        return make_call("pymetrix:forecast_store.save", ForecastStore(self.folder), recorded_by=self.command_id,
                         rnd=Ref(self.round), overwrite=self.overwrite)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Forecast/Compare rounds...")
@dataclass(frozen=True)
class CompareRounds(Command):
    """Two rounds from a store side by side, with the revisions between them.

    For each variable the table shows both rounds' paths and the revision (new less old), period by period. Revisions
    are the record of how the forecast changed: separate those that come from new data (the later round's last data
    period is later) from those that come from judgement or a changed model, which the rounds' notes and adjustment
    logs explain. Persistent revisions in one direction can be a sign that a model or assumption is biased.
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS

    old: Annotated[str, Widget("text", help="name of the earlier round in the store; revisions are measured from it")]
    new: Annotated[str, Widget("text", help="name of the later round; each revision is this round's value less the earlier one's")]
    folder: Annotated[str, Widget("text", help="the forecast store holding both rounds, as given when they were saved")] = "forecasts"
    variables: Annotated[str, Widget("text", help="variables; empty = all both rounds have")] = ""
    name: str = "revisions"

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        from ..forecasting import ForecastStore

        return make_call("pymetrix:forecast_store.compare", ForecastStore(self.folder), output=self.name, recorded_by=self.command_id,
                         old=self.old, new=self.new, variables=_names(self.variables) or None)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Forecast/Backtest...")
@dataclass(frozen=True)
class RunBacktest(Command):
    """Evaluate a forecasting model out of sample: re-estimate it at each origin and forecast ahead.

    Choose a fitted equation, VAR or exponential smoothing model: it is re-estimated on the data up to each origin
    (Start to End) and forecast Horizon periods ahead, and the forecasts are compared with what happened.

    This pseudo out-of-sample exercise mimics real-time forecasting, so it is a fairer guide to forecast accuracy than
    the in-sample fit; but it uses the latest data release throughout, not the data as first published, and so tends
    to flatter them. Forecasts at every horizon from 1 to Horizon are kept, and the accuracy table gives the RMSE, MAE and
    mean error by variable and horizon; a mean error far from zero signals bias. An expanding window uses all data to
    each origin; a rolling window adapts to structural change at the cost of noisier estimates. Use enough origins
    (a few dozen) for the comparisons and tests to mean much.

    For a real-time backtest, give Vintages: for each origin, the dataset as it was then (origin=dataset, separated by
    commas, one for every origin from Start to End). The model is then re-estimated and forecast from the data
    available at each origin, while the errors are still measured against the latest release (Data).
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS

    fit: Annotated[str, Widget("object", source="FittedEquation,FittedVAR,FittedExponentialSmoothing,FittedSARIMA,FittedTheta,FittedSTLForecast,FittedBMA,FittedThresholdAR,FittedLearner", help="the estimate to re-estimate at each origin")]
    start: Annotated[str, Widget("period", help="first period at which the model is re-estimated and forecasts made; leave enough data before it")]
    end: Annotated[str, Widget("period", help="last forecast origin; forecasts whose targets lie beyond the data have no error")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    horizon: Annotated[int, Widget("number", min=1, max=40, help="longest horizon forecast at each origin; every horizon from 1 up to it is kept")] = 8
    window: Annotated[int, Widget("number", min=0, max=1000, advanced=True,
                                  help="estimate on the last N periods to each origin (a rolling window); 0 = all of them (expanding)")] = 0
    name: str = "backtest_result"
    vintages: Annotated[str, Widget("text", advanced=True, help="real time: origin=dataset for every origin, e.g. 2015Q1=v2015q1, 2015Q2=v2015q2 "
                                                 "(the data as they were then); empty = the latest data throughout")] = ""

    def validate(self, session: Session) -> list[str]:
        missing = [n for n in (self.fit, self.data) if n not in session.objects]
        problems = [f"no object named {n!r}" for n in missing] + ([] if self.start and self.end else ["give the first and last origins"])
        try:
            vintages = _vintages(self.vintages)
        except SpecificationError as exc:
            return problems + [f"vintages: {exc}"]
        return problems + [f"vintages: no object named {n!r}" for n in dict.fromkeys(vintages.values()) if n not in session.objects]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:backtest", output=self.name, recorded_by=self.command_id, fitted=Ref(self.fit),
                         data=Ref(self.data), origins=(self.start, self.end), horizon=self.horizon, label=self.fit,
                         window=self.window or None, vintages={k: Ref(v) for k, v in _vintages(self.vintages).items()} or None)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


def _vintages(text: str) -> dict[str, str]:
    """``"2015Q1=v1, 2015Q2=v2"`` -> {"2015Q1": "v1", "2015Q2": "v2"}: each origin's dataset, by name."""
    out = {}
    for part in _names(text):
        origin, sep, name = part.partition("=")
        if not sep or not origin.strip() or not name.strip():
            raise SpecificationError(f"write each vintage as origin=dataset, e.g. 2015Q1=v2015q1 (not {part!r})")
        out[origin.strip()] = name.strip()
    return out


def _backtests(session: Session, text: str, least: int = 2) -> list[str]:
    from ..forecasting import Backtest

    names = _names(text)
    problems = [f"no object named {n!r}" for n in names if n not in session.objects]
    problems += [f"{n} is not a backtest" for n in names if n in session.objects and not isinstance(session.objects[n], Backtest)]
    return problems + ([f"name at least {least} backtests, separated by commas"] if len(names) < least else [])


@command(menu="Forecast/Combine forecasts...")
@dataclass(frozen=True)
class CombineForecasts(Command):
    """Combine the forecasts of several backtests, and evaluate the combination as a backtest of its own.

    Equal weights (the simple average, hard to beat), weights in proportion to 1 / MSE, or regression weights that
    sum to one. Weights are estimated in real time: at each origin only from the outcomes known by then.
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS

    backtests: Annotated[str, Widget("object", source="Backtest", multiple=True, help="two or more backtests over the same origins and variables; only common forecasts are combined")]
    method: Annotated[Literal["equal", "inverse_mse", "regression"], Widget("choice", help="equal (robust, hard to beat), inverse MSE, or regression weights (can be negative; noisy if few outcomes)")] = "equal"
    min_history: Annotated[int, Widget("number", min=1, max=200, advanced=True, help="equal weights until this many outcomes are known")] = 4
    name: str = "combination"

    def validate(self, session: Session) -> list[str]:
        return _backtests(session, self.backtests)

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:combine_forecasts", output=self.name, recorded_by=self.command_id,
                         backtests={n: Ref(n) for n in _names(self.backtests)}, method=self.method, min_history=self.min_history)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Forecast/Compare backtests...")
@dataclass(frozen=True)
class CompareBacktests(Command):
    """Test whether two backtests' forecasts of a variable are equally accurate at one horizon.

    Diebold-Mariano (equal expected loss), Giacomini-White (equal predictive ability given what was known when
    forecasting), Clark-West (the first model is nested in the second: does the larger model forecast better?), or
    encompassing (does the first forecast hold all the information in the second?). Losses: squared or absolute
    errors, or for density forecasts the CRPS or the log score.

    The tests use the forecasts of both backtests for the same targets, so run them over the same origins. For
    Diebold-Mariano the loss differential is the first backtest's loss less the second's: a negative statistic
    favours the first. It treats the forecasts as given, which suits comparing forecasts, not population models; for
    nested models it is undersized, which is what Clark-West corrects. Long-horizon forecasts overlap, so their
    losses are serially correlated; the tests allow for this, but with few origins they have little power, and failing
    to reject is not evidence that the forecasts are equally good.
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS

    first: Annotated[str, Widget("object", source="Backtest", help="the first backtest (for Clark-West, the smaller, nested model)")]
    second: Annotated[str, Widget("object", source="Backtest", help="the backtest to compare it with")]
    variable: Annotated[str, Widget("text", help="the variable both forecast")]
    horizon: Annotated[int, Widget("number", min=1, max=40, help="which horizon to compare; both backtests must have forecast this far ahead")] = 1
    test: Annotated[Literal["diebold-mariano", "giacomini-white", "clark-west", "encompassing"],
                    Widget("choice", help="equal accuracy, conditional ability, nested models, or does the first encompass the second")] = "diebold-mariano"
    loss: Annotated[_LOSS, Widget("choice", help="not used by Clark-West and encompassing; CRPS and log score need density forecasts")] = "squared"
    tau: Annotated[float, _TAU] = 0.5
    linex_a: Annotated[float, _LINEX] = 1.0
    name: str = "comparison"

    def validate(self, session: Session) -> list[str]:
        return _backtests(session, f"{self.first}, {self.second}") + ([] if self.variable else ["name the variable to compare"])

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        record = {"output": self.name, "recorded_by": self.command_id}
        if self.test == "encompassing":
            return make_call("pymetrix:backtest.encompasses", Ref(self.first), other=Ref(self.second), variable=self.variable,
                             horizon=self.horizon, **record)
        return make_call("pymetrix:backtest.compare", Ref(self.first), other=Ref(self.second), variable=self.variable,
                         horizon=self.horizon, loss=self.loss, test=self.test, tau=self.tau, a=self.linex_a, **record)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Forecast/Compare many forecasts...")
@dataclass(frozen=True)
class CompareManyForecasts(Command):
    """The model confidence set (Hansen, Lunde and Nason): which of several backtests' forecasts could be the best,
    at one horizon, with a stationary bootstrap. Needs the arch package.

    Starting from all the models, the procedure drops the worst until the remaining ones cannot be told apart, so the
    set contains the best model with probability 1 - alpha (it uses the range statistic). A large set means the data
    cannot separate the forecasts, not that they are equally good; a model outside the set is significantly worse.
    The table gives each model's average loss and its MCS p-value: a model is in the set when its p-value exceeds
    alpha. Losses are serially correlated at longer horizons, which the block bootstrap allows for; the default mean
    block length is the larger of the horizon and n^(1/3).
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS

    backtests: Annotated[str, Widget("object", source="Backtest", multiple=True, help="the competing backtests; their common forecast targets are compared")]
    variable: Annotated[str, Widget("text", help="the variable they all forecast")]
    horizon: Annotated[int, Widget("number", min=1, max=40, help="which horizon to compare; also the shortest default bootstrap block")] = 1
    loss: Annotated[_LOSS, Widget("choice", help="the loss to compare; CRPS and log score need density forecasts")] = "squared"
    tau: Annotated[float, _TAU] = 0.5
    linex_a: Annotated[float, _LINEX] = 1.0
    alpha: Annotated[float, Widget("number", min=0.01, max=0.5, step=0.05, help="the set holds the best model with probability 1 - alpha")] = 0.1
    name: str = "confidence_set"
    reps: Annotated[int, Widget("number", min=100, max=100_000, step=100, advanced=True, help="bootstrap resamples for the p-values; 1000 is usual, more steadies p-values near alpha")] = 1000
    block: Annotated[int | None, Widget("number", min=1, max=200, advanced=True,
                                        help="the stationary bootstrap's mean block length; empty = automatic")] = None
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, help="random seed for the bootstrap")] = 0

    def validate(self, session: Session) -> list[str]:
        return _backtests(session, self.backtests) + ([] if self.variable else ["name the variable to compare"])

    def _bootstrap(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k, d in (("reps", 1000), ("block", None), ("seed", 0)) if getattr(self, k) != d}

    def run(self, session: Session) -> Any:
        from ..stats.evaluation import model_confidence_set

        losses = {n: session[n].losses(self.variable, self.horizon, self.loss, self.tau, self.linex_a) for n in _names(self.backtests)}
        result = model_confidence_set(losses, self.alpha, self.horizon, **self._bootstrap())
        session.objects[self.name] = result
        return result

    def to_code(self) -> str:
        options = _loss_code(self.loss, self.tau, self.linex_a)
        losses = ", ".join(f"{n!r}: {n}.losses({self.variable!r}, {self.horizon}{options})" for n in _names(self.backtests))
        extra = "".join(f", {k}={v!r}" for k, v in self._bootstrap().items())
        return f"{self.name} = model_confidence_set({{{losses}}}, {self.alpha}, {self.horizon}{extra})"
