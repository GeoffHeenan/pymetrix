"""Dialog for occasionally binding constraints: Model > Occasionally binding constraints (OccBin)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from ..errors import SpecificationError
from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session


def _shocks(text: str) -> dict[str, dict[int, float]]:
    """``ey[1]=-3, ez[2:4]=-1`` -> {"ey": {1: -3.0}, "ez": {2: -1.0, 3: -1.0, 4: -1.0}} (periods counted from 1)."""
    out: dict[str, dict[int, float]] = {}
    for item in [t.strip() for t in re.split(r"[,;\n]", text) if t.strip()]:
        m = re.fullmatch(r"([A-Za-z_]\w*)\s*\[\s*(\d+)\s*(?::\s*(\d+)\s*)?\]\s*=\s*([-+]?[\d.eE+-]+)", item)
        if not m:
            raise SpecificationError(f"give shocks as name[period]=value or name[first:last]=value, e.g. ey[1]=-3 (not {item!r})")
        first, last = int(m.group(2)), int(m.group(3) or m.group(2))
        if first < 1 or last < first:
            raise SpecificationError(f"periods count from 1, first before last: {item!r}")
        for t in range(first, last + 1):
            out.setdefault(m.group(1), {})[t] = float(m.group(4))
    return out


@command(menu="Model/Occasionally binding constraints...")
@dataclass(frozen=True)
class OccasionallyBinding(Command):
    """Solve a model with occasionally binding constraints, such as an effective lower bound on the policy rate, by
    the piecewise-linear method of Guerrieri and Iacoviello (OccBin), as in Dynare.

    The constraints are the model's max and min (e.g. i ~ max(rule, floor); the New Keynesian template has one with
    lower_bound=0). Give the shocks as ey[1]=-3, or ez[2:4]=-1 for several periods. Shocks are surprises, period by
    period, unless anticipated, when the whole path is known from the first period. The result shows when each
    constraint binds, and the paths with and without the constraints; the path without them has the same shocks,
    known in the same way, so with anticipated shocks it is the linear model's perfect-foresight path.

    The method solves the model as linear in each regime (constraint slack or binding) and finds the sequence of
    regimes consistent with the paths it implies, so it captures how a binding constraint changes the response to a
    shock, including the expected duration of the spell at the bound. It is a perfect-foresight approximation: agents
    do not take into account the chance that future shocks will push the economy to the constraint, so
    precautionary effects of the constraint are missed. Compare the constrained and unconstrained paths to see how
    much the constraint amplifies the shock. If the regime sequence does not settle, raise the guesses allowed or try
    a smaller shock.
    """

    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models",)

    source: Annotated[str, Widget("object", source="Template,DynareModel", help="a model with max or min (a template, Dynare or IRIS model)")]
    shocks: Annotated[str, Widget("text", help="e.g. ey[1]=-3 or ez[2:4]=-1 (periods count from 1)")]
    periods: Annotated[int, Widget("number", min=1, max=1000, help="length of the simulated paths shown, from the first shock; long enough for the economy to leave the bound")] = 40
    anticipated: Annotated[bool, Widget("checkbox", help="the whole shock path is known from the first period")] = False
    name: str = "constrained"
    horizon: Annotated[int | None, Widget("number", min=1, max=10_000, advanced=True,
                                          help="periods over which the regimes are solved; empty = automatic (beyond the shown periods)")] = None
    max_iter: Annotated[int, Widget("number", min=1, max=10_000, advanced=True, help="iterations on the guessed sequence of regimes before giving up; raise it if it does not converge")] = 100

    def validate(self, session: Session) -> list[str]:
        if self.source not in session.objects:
            return [f"no model named {self.source!r}"]
        try:
            shocks = _shocks(self.shocks)
        except ValueError as exc:
            return [str(exc)]
        if not shocks:
            return ["give at least one shock, e.g. ey[1]=-3"]
        late = [f"{n}[{t}]" for n, path in shocks.items() for t in path if t > self.periods]
        return [f"{', '.join(late)} after the last period ({self.periods})"] if late else []

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:occbin", output=self.name, recorded_by=self.command_id, obj=Ref(self.source),
                         shocks=_shocks(self.shocks), periods=self.periods, anticipated=self.anticipated,
                         **{k: getattr(self, k) for k, d in (("horizon", None), ("max_iter", 100)) if getattr(self, k) != d})

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Model/OccBin filter...", id="pymetrix:occbin_filter")
@dataclass(frozen=True)
class OccBinFilterCommand(Command):
    """Estimate a model's variables from data when it has occasionally binding constraints, such as an effective
    lower bound on the policy rate: the OccBin filter and smoother (the piecewise Kalman filter of Dynare 6).

    In each period the filter uses the decision rule of the regimes expected from its estimate of the state, solves
    the model again with the shock it estimates, and repeats until the regimes settle. The result has the smoothed
    and filtered variables, the smoothed shocks, when each constraint binds and the likelihood of the data; estimate
    the model with this likelihood under Model > Estimate model, Advanced, OccBin. A series fixed at a bound (a rate at
    its floor) and observed without measurement error is allowed: the likelihood is that of the other series.
    Iterate the smoother (advanced) to work the regimes out again on the smoothed path, as Dynare 6's OccBin smoother:
    the smoothed path then respects the constraints period by period.
    """

    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models",)

    source: Annotated[str, Widget("object", source="Template,DynareModel,FittedModel",
                                  help="a model with max or min (a template, Dynare or IRIS model, or one estimated)")]
    data: Annotated[str, Widget("object", source="Dataset", help="data with the model's observed series")] = "data"
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    name: str = "occbin_filtered"
    horizon: Annotated[int, Widget("number", min=2, max=10_000, advanced=True,
                                   help="periods ahead over which each period's regimes are worked out")] = 40
    presample: Annotated[int, Widget("number", min=0, max=400, advanced=True,
                                     help="periods filtered but left out of the likelihood")] = 0
    iterated: Annotated[bool, Widget("checkbox", advanced=True,
                                     help="work the regimes out again on the smoothed path until they settle (Dynare 6's OccBin "
                                          "smoother); unticked, the smoother keeps the filter's regimes")] = False

    def validate(self, session: Session) -> list[str]:
        missing = [n for n in (self.source, self.data) if n not in session.objects]
        return [f"no object named {n!r}" for n in missing]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        extra = {k: getattr(self, k) for k, d in (("horizon", 40), ("presample", 0)) if getattr(self, k) != d}
        if self.iterated:
            extra["smoother"] = "iterated"
        if self.start or self.end:
            extra["span"] = (self.start or None, self.end or None)
        return make_call("pymetrix:occbin_filter", output=self.name, recorded_by=self.command_id, obj=Ref(self.source),
                         data=Ref(self.data), **extra)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
