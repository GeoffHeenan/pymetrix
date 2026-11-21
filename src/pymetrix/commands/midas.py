"""The MIDAS dialog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session


@command(menu="Estimate/MIDAS (mixed frequency)...")
@dataclass(frozen=True)
class FitMIDAS(Command):
    """Regress a low-frequency series (quarterly GDP) on high-frequency indicators (monthly), through lag weights.

    Exponential Almon and Beta weights describe many monthly lags with two parameters each; unrestricted estimates
    every lag. Data short at the end lets you build a nowcast as the quarter's months arrive. The result's
    nowcast() gives the series for every quarter the indicators cover.

    The weighting scheme is the main modelling choice. Exponential Almon and Beta weights let many lags enter with few
    parameters, which keeps the model parsimonious when the low-frequency sample is short, but impose a smooth lag
    shape; unrestricted MIDAS estimates every lag freely and suits a small frequency gap (monthly to quarterly) and
    few lags. Inspect the estimated lag weights: they show which months carry the information. Series should be
    stationary (growth rates). Evaluate nowcasts out of sample with the high-frequency lag set to mimic the data
    available at the time.
    """

    topics: ClassVar[tuple[str, ...]] = ("midas", "temporal-disaggregation")

    formula: Annotated[str, Widget("formula", help="e.g. gdp ~ ip + sales (high-frequency series names)")]
    low: Annotated[str, Widget("object", source="Dataset", help="the low-frequency data (e.g. quarterly)")]
    high: Annotated[str, Widget("object", source="Dataset", help="the high-frequency data (e.g. monthly)")] = "data"
    lags: Annotated[int, Widget("number", min=2, max=60, help="high-frequency lags of each indicator, e.g. 12 months; with unrestricted weights each costs a parameter")] = 12
    weights: Annotated[Literal["exp_almon", "beta", "unrestricted"], Widget("choice", help="lag weights: exp_almon or beta (two parameters each, smooth) or unrestricted (one per lag)")] = "exp_almon"
    ar: Annotated[int, Widget("number", min=0, max=8, help="lags of the low-frequency dependent variable, for persistence; 0 = none")] = 0
    high_lag: Annotated[int, Widget("number", min=0, max=24, advanced=True, help="high-frequency periods missing at the end (nowcasting)")] = 0
    name: str = "midas"

    def _spec(self) -> Any:
        from ..midas import MIDAS

        return MIDAS(self.formula, self.lags, self.weights, self.ar, self.high_lag)

    def validate(self, session: Session) -> list[str]:
        missing = [n for n in (self.low, self.high) if n not in session.objects]
        return [f"no dataset named {n!r}" for n in missing]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:midas.fit", self._spec(), output=self.name, recorded_by=self.command_id,
                         low=Ref(self.low), high=Ref(self.high))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
