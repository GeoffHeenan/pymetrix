"""Dialog for data contributions: Model > Analysis > Data contributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session


@command(menu="Model/Data contributions...")
@dataclass(frozen=True)
class DataContributions(Command):
    """Split an estimated model's Kalman-smoothed variables into what each observed series contributed (and the
    starting point), and, with a horizon, the forecast from the end of the data too: what moved the output gap, the
    GDP release or inflation? The parts add up exactly to the smoothed path and the forecast.

    The smoothed estimates are linear in the data, so each observed series' contribution is well defined: it is how
    far the estimate moves when that series is observed rather than set to what the model expects with no news. A
    large contribution from one series means the model's reading of an unobserved variable such as the output gap
    leans on it, so revisions to that series, or doubts about its measurement, carry over directly. The split
    depends on the estimated model and on which values are missing, and says nothing about causes in the economy:
    it attributes the estimate, not the variable itself. Pass newer data to see how a new release moved the estimate.

    A variable observed without measurement error is known exactly, so its smoothed value comes from its own series
    alone and the other series contribute nothing to it in the sample (they can to its forecast, through the other
    states). In the New Keynesian template, inflation and the policy rate are observed this way, so their smoothed
    values are all "infl" and all "rate", and output, observed through its growth, leans almost entirely on
    "growth". The other series share the work only when the measurement equations have errors.
    """

    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models", "state-space")

    source: Annotated[str, Widget("object", source="FittedModel", help="an estimated model (Model > Estimate model)")]
    horizon: Annotated[int, Widget("number", min=0, max=200, help="periods of forecast from the end of the data to decompose too; 0 = the smoothed history only")] = 0
    name: str = "contributions"
    data: Annotated[str, Widget("object", source="Dataset", empty="empty: the data the model was estimated on", advanced=True,
                                help="newer data, filtered with the estimated model; empty = the estimation's data")] = ""

    def validate(self, session: Session) -> list[str]:
        if self.data and self.data not in session.objects:
            return [f"no dataset named {self.data!r}"]
        return [] if self.source in session.objects else [f"no estimated model named {self.source!r}"]

    def to_call(self, session: Session) -> Any:
        from ..calls import method_call
        from ..codec import Ref

        extra = {"data": Ref(self.data)} if self.data else {}
        return method_call(type(session[self.source]), "contributions", Ref(self.source), output=self.name,
                           recorded_by=self.command_id, horizon=self.horizon, **extra)

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        args = ([f"horizon={self.horizon}"] if self.horizon else []) + ([f"data={self.data}"] if self.data else [])
        return f"{self.name} = {self.source}.contributions({', '.join(args)})"
