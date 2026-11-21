"""Nowcasting dialogs: bridge equations, the mixed-frequency dynamic factor model, the ragged-edge tools and combined nowcasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def _datasets(session: Session, *names: str) -> list[str]:
    return [f"no dataset named {n!r}" for n in names if n and n not in session.objects]


@command(menu="Estimate/Bridge equation...")
@dataclass(frozen=True)
class FitBridge(Command):
    """Regress a quarterly series on monthly indicators averaged within each quarter (a bridge equation).

    The formula names the quarterly series and the indicators, e.g. gy ~ gr + pmi. To nowcast, the current quarter's
    missing months are forecast by an autoregression of each indicator, then averaged: the result's nowcast(quarterly,
    monthly) gives the quarters after the last quarterly figure.

    A bridge equation is a plain regression, so it is easy to read and to explain, but each indicator enters only as
    a quarterly aggregate: what it says about the current quarter depends on how many of its months are already out,
    and the months filled in by autoregressions carry little information. Choose the aggregation to match the
    quarterly series: the mean for rates and indices, the sum for flows measured monthly, the last month for
    end-of-period stocks. Use stationary series (growth rates, survey balances) on both sides, and evaluate the
    nowcast out of sample, as the in-sample fit says little about real-time accuracy.
    """

    topics: ClassVar[tuple[str, ...]] = ("midas",)

    formula: Annotated[str, Widget("formula", help="e.g. gy ~ gr + pmi (quarterly and monthly series names)")]
    low: Annotated[str, Widget("object", source="Dataset", help="the dataset with the quarterly series in the formula; its last figure sets where the nowcast starts")]
    high: Annotated[str, Widget("object", source="Dataset", help="the dataset with the monthly indicators; they may end in different months (a ragged edge)")] = "data"
    how: Annotated[Literal["mean", "sum", "last"], Widget("choice", help="how three months make a quarter: mean for rates and indices, sum for flows, last for stocks")] = "mean"
    fill_lags: Annotated[int, Widget("number", min=1, max=12, advanced=True, help="lags of the autoregressions that forecast each indicator's missing months to complete the quarter")] = 2
    name: str = "bridge"

    def _spec(self) -> Any:
        from ..nowcast import Bridge

        return Bridge(self.formula, self.how, self.fill_lags)

    def validate(self, session: Session) -> list[str]:
        return _datasets(session, self.low, self.high)

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:bridge.fit", self._spec(), output=self.name, recorded_by=self.command_id,
                         low=Ref(self.low), high=Ref(self.high))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/Dynamic factor model (nowcasting)...")
@dataclass(frozen=True)
class FitDynamicFactor(Command):
    """A dynamic factor model of monthly indicators and quarterly series, estimated by EM with the Kalman filter, so
    series that end in different months (the ragged edge) are used as they are.

    Series must be stationary: growth rates or surveys (make them with Data > Generate series). The result's
    nowcast("gy") gives the quarters after the last quarterly figure; update() puts newer data through the same
    model, and news() shows how each new figure moved the nowcast.
    """

    topics: ClassVar[tuple[str, ...]] = ("midas", "state-space")

    monthly: Annotated[str, Widget("text", help="monthly series, e.g. gr, gip, gemp, pmi")]
    high: Annotated[str, Widget("object", source="Dataset", help="the dataset with the monthly series; each may end in a different month")] = "data"
    quarterly: Annotated[str, Widget("text", help="quarterly series to nowcast, e.g. gy")] = ""
    low: Annotated[str, Widget("object", source="Dataset", help="the quarterly data (if any quarterly series)")] = ""
    factors: Annotated[int, Widget("number", min=1, max=5, help="number of common factors; one often captures the business cycle, more fit better but are harder to read")] = 1
    factor_order: Annotated[int, Widget("number", min=1, max=6, help="lags in the VAR the factors follow; more lags give richer dynamics at the cost of parameters")] = 2
    name: str = "dfm"

    def _spec(self) -> Any:
        from ..nowcast import DynamicFactor

        return DynamicFactor(tuple(_names(self.monthly)), tuple(_names(self.quarterly)), self.factors, self.factor_order)

    def validate(self, session: Session) -> list[str]:
        errors = _datasets(session, self.high, self.low)
        if _names(self.quarterly) and not self.low:
            errors.append("choose the quarterly data for the quarterly series")
        return errors + ([] if _names(self.monthly) else ["give the monthly series"])

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:dynamic_factor.fit", self._spec(), output=self.name, recorded_by=self.command_id,
                         low=Ref(self.low) if self.low else None, high=Ref(self.high))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


# -- the ragged edge and combining nowcasts -------------------------------------------------------------------------------
# Data tools for nowcasting (pymetrix.nowcast): what was known on a day, filling or realigning the edge, and combining
# the nowcasts of several models. In Estimate → Mixed frequency, after the bridge and dynamic factor models.

_EDGE_TOPICS: tuple[str, ...] = ("nowcasting", "midas")


def _release_lags(text: str) -> dict[str, int] | None:
    """``"ip=45, pmi=1"`` -> {"ip": 45, "pmi": 1}; empty -> None (each series' own publication lag)."""
    out = {}
    for part in _names(text):
        series, _, days = part.partition("=")
        out[series.strip()] = int(days.strip())
    return out or None


def _release_lag_errors(text: str, names: list[str]) -> list[str]:
    errors = []
    for part in _names(text):
        series, eq, days = (p.strip() for p in part.partition("="))
        if not eq or not series or not days.lstrip("-").isdigit():
            errors.append(f"write each publication lag as series=days (e.g. ip=45, pmi=1), not {part!r}")
        elif int(days) < 0:
            errors.append(f"{series}: a publication lag is days after the period ends, 0 or more")
        elif series not in names:
            errors.append(f"no series {series!r} in the data")
    return errors


def _time_series(session: Session, name: str) -> tuple[Any, list[str]]:
    import pandas as pd

    from ..data import Dataset

    data = session.objects.get(name)
    if not isinstance(data, Dataset):
        return None, [f"no dataset named {name!r}"]
    if not isinstance(data.frame.index, pd.PeriodIndex):
        return None, [f"{name} is not a time series with one series per column: the ragged-edge tools need one"]
    return data, []


def _day_errors(text: str, required: bool) -> list[str]:
    import pandas as pd

    if not text.strip():
        return ["give the day, e.g. 2025-02-15"] if required else []
    try:
        pd.Timestamp(text.strip())
    except ValueError:
        return [f"the day is a date such as 2025-02-15, not {text!r}"]
    return []


_RELEASE_LAGS_HELP = ("publication lags in days after the period ends, for series whose metadata has none or to try "
                      "others: ip=45, pmi=1; empty = each series' own (Variable.release_lag)")


@command(menu="Estimate/Ragged edge...")
@dataclass(frozen=True)
class RaggedEdge(Command):
    """Show where each series ends: its last observation, its publication lag and the day that value came out.

    Indicators are published with different delays (a survey days after the month, industrial production six weeks
    later), so on any day they end in different months: the ragged edge. With a day, the table also says how many
    periods each series is behind the most recent one, as it stood that day. Publication lags come from each
    series' metadata (Variable.release_lag); give others to check what difference they make.
    """

    topics: ClassVar[tuple[str, ...]] = _EDGE_TOPICS

    data: Annotated[str, Widget("object", source="Dataset", help="the indicators (monthly, as a rule), each with its publication lag")] = "data"
    when: Annotated[str, Widget("text", label="On the day", help="a date, e.g. 2025-02-15: the edge as it stood then; empty = the data as they are")] = ""
    name: str = "edge"
    release_lags: Annotated[str, Widget("text", advanced=True, help=_RELEASE_LAGS_HELP)] = ""

    def validate(self, session: Session) -> list[str]:
        data, errors = _time_series(session, self.data)
        return errors or _day_errors(self.when, False) + _release_lag_errors(self.release_lags, list(data.names))

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:dataset.ragged_edge", Ref(self.data), output=self.name, recorded_by=self.command_id,
                         when=self.when.strip() or None, lags=_release_lags(self.release_lags))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/Data as known on a day...")
@dataclass(frozen=True)
class DataAsOf(Command):
    """The data as they were known on a day: every value published later removed, so a nowcast can be made (or
    checked) with only what was known then.

    A value counts as published on the last day of its period plus the series' publication lag (from its metadata,
    or given under Advanced). Run a nowcast on data as of a sequence of days to see how it would have moved as the
    figures came out, a pseudo-real-time evaluation. Values are only removed, not revised: earlier vintages of
    revised data come from the provider (File → Load from connector…, Vintage).
    """

    topics: ClassVar[tuple[str, ...]] = _EDGE_TOPICS

    when: Annotated[str, Widget("text", label="On the day", help="a date, e.g. 2025-02-15")]
    data: Annotated[str, Widget("object", source="Dataset", help="the data with publication lags (Variable.release_lag)")] = "data"
    name: str = "known"
    release_lags: Annotated[str, Widget("text", advanced=True, help=_RELEASE_LAGS_HELP)] = ""

    def validate(self, session: Session) -> list[str]:
        data, errors = _time_series(session, self.data)
        return errors or _day_errors(self.when, True) + _release_lag_errors(self.release_lags, list(data.names))

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:dataset.as_of", Ref(self.data), output=self.name, recorded_by=self.command_id,
                         when=self.when.strip(), lags=_release_lags(self.release_lags))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/Fill the ragged edge...")
@dataclass(frozen=True)
class FillEdge(Command):
    """Fill the missing values at the end of each series with forecasts from an autoregression of the series.

    Each series' AR model (a constant and Lags lags) is estimated on its own history and run forward from its last
    observation to Through (by default the last period of any series), the way bridge equations complete the
    current quarter's months. Values already there are kept. The forecasts carry little information beyond the
    series' own momentum, so the more months are filled, the less the nowcast says.
    """

    topics: ClassVar[tuple[str, ...]] = _EDGE_TOPICS

    data: Annotated[str, Widget("object", source="Dataset", help="the indicators, ending in different periods")] = "data"
    lags: Annotated[int, Widget("number", min=1, max=12, help="lags of each series' autoregression; 2 is usual for monthly data")] = 2
    through: Annotated[str, Widget("period", help="fill to this period, e.g. 2025-03 (the end of the quarter); empty = the last period of any series")] = ""
    series: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the series to fill; empty = all")] = ""
    name: str = "filled"

    def validate(self, session: Session) -> list[str]:
        import pandas as pd

        data, errors = _time_series(session, self.data)
        if errors:
            return errors
        errors = [f"no series {n!r} in {self.data}" for n in _names(self.series) if n not in data.names]
        if self.through.strip():
            try:
                pd.Period(self.through.strip(), freq=data.frame.index.freq)
            except ValueError:
                errors.append(f"{self.through!r} is not a period of {self.data}")
        return errors

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        names = _names(self.series)
        return make_call("pymetrix:dataset.fill_edge", Ref(self.data), output=self.name, recorded_by=self.command_id,
                         lags=self.lags, through=self.through.strip() or None, names=names or None)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/Realign the ragged edge...")
@dataclass(frozen=True)
class RealignEdge(Command):
    """Vertical realignment: shift each series on so that its last value falls in the latest period any series has.

    A series one month behind is moved one month on, so its latest figure is treated as current: the data end
    together, with no forecasts (Altissimo et al. 2010). The series' timing changes, so a model estimated on the
    realigned data should be used with data realigned the same way, the same days after each month.
    """

    topics: ClassVar[tuple[str, ...]] = _EDGE_TOPICS

    data: Annotated[str, Widget("object", source="Dataset", help="the indicators, ending in different periods")] = "data"
    name: str = "realigned"

    def validate(self, session: Session) -> list[str]:
        return _time_series(session, self.data)[1]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:dataset.realign", Ref(self.data), output=self.name, recorded_by=self.command_id)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


def _weights(text: str) -> dict[str, float] | None:
    out = {}
    for part in _names(text):
        model, _, weight = part.partition("=")
        out[model.strip()] = float(weight)
    return out or None


@command(menu="Estimate/Combine nowcasts...")
@dataclass(frozen=True)
class CombineNowcasts(Command):
    """Average the nowcasts of several models: bridge equations and dynamic factor models.

    Each model makes its nowcast as its own window does: a bridge equation from the quarterly and monthly data
    given here (as known on the day, Estimate → Mixed frequency → Data as known on a day…), a dynamic factor model
    for the series named, from the data it was fitted to. The combination is a weighted average, period by
    period (equal weights unless given; they are scaled to add up to one); a period is left out unless every
    model nowcasts it. Inverse mean squared errors from a backtest are a common choice of weights.
    """

    topics: ClassVar[tuple[str, ...]] = _EDGE_TOPICS

    models: Annotated[str, Widget("object", source="FittedBridge,FittedDynamicFactor", multiple=True,
                                  help="fitted bridge equations and dynamic factor models")]
    variable: Annotated[str, Widget("text", help="dynamic factor models: the series to nowcast, e.g. gy")] = ""
    low: Annotated[str, Widget("object", source="Dataset", help="bridge equations: the quarterly data", empty="no bridge equation")] = ""
    high: Annotated[str, Widget("object", source="Dataset", help="bridge equations: the monthly indicators, as known on the day")] = "data"
    weights: Annotated[str, Widget("text", help="model=weight, e.g. bridge=2, dfm=1; empty = equal weights")] = ""
    name: str = "combined"
    through: Annotated[str, Widget("period", advanced=True, help="nowcast through this quarter, e.g. 2025Q2; empty = the quarter of the latest month")] = ""

    def validate(self, session: Session) -> list[str]:
        from ..nowcast import FittedBridge, FittedDynamicFactor

        names = _names(self.models)
        if not names:
            return ["choose the models whose nowcasts to combine"]
        errors = [f"no model named {n!r}" for n in names if n not in session.objects]
        if errors:
            return errors
        kinds = {n: type(session[n]) for n in names}
        errors = [f"{n} is not a bridge equation or a dynamic factor model" for n, k in kinds.items()
                  if not issubclass(k, (FittedBridge, FittedDynamicFactor))]
        if any(issubclass(k, FittedBridge) for k in kinds.values()):
            if not self.low:
                errors.append("choose the quarterly data for the bridge equations")
            errors += _datasets(session, self.low, self.high)
        for n, k in kinds.items():
            if issubclass(k, FittedDynamicFactor):
                spec = session[n].spec
                if not self.variable.strip():
                    errors.append(f"name the series {n} nowcasts (Variable)")
                elif self.variable.strip() not in (*spec.monthly, *spec.quarterly):
                    errors.append(f"{n} has no series {self.variable.strip()!r}; it has {', '.join((*spec.monthly, *spec.quarterly))}")
        try:
            weights = _weights(self.weights) or {}
        except ValueError:
            return errors + ["write weights as model=number, e.g. bridge=2, dfm=1"]
        errors += [f"{m} is not one of the models chosen" for m in weights if m not in names]
        if any(w < 0 for w in weights.values()) or (weights and not any(weights.get(n, 1.0) for n in names)):
            errors.append("weights must be 0 or more, and not all 0")
        return errors

    def to_call(self, session: Session) -> Any:
        from ..calls import make_call, method_call
        from ..codec import Ref
        from ..nowcast import FittedBridge

        through = self.through.strip() or None
        nowcasts = {}
        for n in _names(self.models):
            obj = session[n]
            if isinstance(obj, FittedBridge):
                nowcasts[n] = method_call(type(obj), "nowcast", Ref(n), low=Ref(self.low), high=Ref(self.high), through=through)
            else:
                nowcasts[n] = method_call(type(obj), "nowcast", Ref(n), variable=self.variable.strip(), through=through)
        return make_call("pymetrix:nowcast.combine", output=self.name, recorded_by=self.command_id, nowcasts=nowcasts,
                         weights=_weights(self.weights))

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:  # without the session (the models' kinds unknown): bridges when quarterly data are given
        through = f", through={self.through.strip()!r}" if self.through.strip() else ""
        each = (f"{n}.nowcast({self.low}, {self.high}{through})" if self.low else f"{n}.nowcast({self.variable.strip()!r}{through})"
                for n in _names(self.models))
        weights = f", weights={_weights(self.weights)!r}" if self.weights.strip() else ""
        return f"{self.name} = nowcast.combine({{{', '.join(f'{n!r}: {c}' for n, c in zip(_names(self.models), each))}}}{weights})"
