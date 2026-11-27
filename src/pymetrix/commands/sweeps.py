"""Dialogs for many runs as one step: Model > Scenarios > Parameter sweep and Model > Scenarios > Compare scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from ..errors import SpecificationError
from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session

_TOPICS = ("forward-looking-models",)
_VALUES_HELP = "values: a list (1.2, 1.5, 2) or a range start:stop:step (1:2:0.25)"


def _values(text: str) -> list[float]:
    """``1.2, 1.5, 2`` or ``1:2:0.25`` (both ends included)."""
    import numpy as np

    text = text.strip()
    if ":" in text:
        parts = [float(p) for p in text.split(":")]
        if len(parts) != 3 or parts[2] <= 0:
            raise SpecificationError(f"a range is start:stop:step with a positive step, not {text!r}")
        start, stop, step = parts
        count = int(np.floor((stop - start) / step + 1e-9)) + 1
        return [round(start + i * step, 12) for i in range(max(count, 0))]
    return [float(v) for v in text.replace(";", ",").split(",") if v.strip()]


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


@command(menu="Model/Parameter sweep...")
@dataclass(frozen=True)
class ParameterSweep(Command):
    """Run a model once for each value of one or two parameters, and show the runs side by side: impulse responses
    overlaid, moments, variance decompositions or steady states.

    Values are a list (1.2, 1.5, 2) or a range (1:2:0.25). With two parameters, every combination is run, or the
    values are paired in order (zip). Shock sizes are parameters too: sd_ei. A run that fails (no steady state, no
    unique stable solution) is listed with its error; the others are kept. Workers above 1 run in parallel, which
    pays when each run takes a while.

    A sweep is a sensitivity analysis: it shows how much a conclusion depends on a calibrated value, such as how
    the inflation response to a cost-push shock changes with the policy rule's reaction to inflation. Sweep over the
    range the evidence allows rather than one point, and watch for values where the runs fail: these mark the edge
    of determinacy or of a steady state, which is itself informative. Each run changes only the parameters named;
    everything else stays as calibrated, so a sweep traces partial effects, not a re-estimated model.

    Some responses barely move, and their lines overlap: in the New Keynesian template, with its serially
    uncorrelated shocks, a smoothed policy rule and a flat Phillips curve, output's response to a demand shock changes
    by under 1% between phi_pi = 1.2 and 3, while its response to a cost-push shock doubles. Response and shock
    choose the pair the graph shows; the table of responses at one horizon shows differences too small to see in it.
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS

    source: Annotated[str, Widget("object", source="Template,DynareModel", help="a template, Dynare or IRIS model")]
    parameter: Annotated[str, Widget("text", help="a parameter or shock size, e.g. phi_pi or sd_ei")]
    values: Annotated[str, Widget("text", help=_VALUES_HELP)]
    parameter2: Annotated[str, Widget("text", help="a second parameter (optional)", group="Second parameter")] = ""
    values2: Annotated[str, Widget("text", help=_VALUES_HELP, group="Second parameter")] = ""
    grid: Annotated[Literal["product", "zip"], Widget("choice", help="every combination, or the values paired in order",
                                                      group="Second parameter")] = "product"
    what: Annotated[Literal["irf", "moments", "fevd", "steady_state"], Widget("choice", help="impulse responses, unconditional moments, variance decompositions, or the steady state")] = "irf"
    periods: Annotated[int, Widget("number", min=1, max=400, help="periods of impulse responses and decompositions")] = 20
    response: Annotated[str, Widget("text", advanced=True, group="Graph", help="the variable whose impulse responses the graph shows, e.g. y; empty = the first")] = ""
    shock: Annotated[str, Widget("text", advanced=True, group="Graph", help="the shock they respond to, e.g. epi; empty = the first")] = ""
    workers: Annotated[int, Widget("number", min=1, max=64, advanced=True, help="processes to run in parallel; each takes about a second to start, so only for slow runs")] = 1
    name: str = "runs"

    def _grid(self) -> dict[str, list[float]]:
        out = {self.parameter.strip(): _values(self.values)}
        if self.parameter2.strip():
            out[self.parameter2.strip()] = _values(self.values2)
        return out

    def validate(self, session: Session) -> list[str]:
        if self.source not in session.objects:
            return [f"no model named {self.source!r}"]
        if not self.parameter.strip():
            return ["name the parameter to sweep"]
        try:
            grid = self._grid()
        except ValueError as exc:
            return [str(exc)]
        if any(not v for v in grid.values()):
            return ["give each parameter some values"]
        if self.parameter2.strip() and self.parameter2.strip() == self.parameter.strip():
            return ["sweep two different parameters"]
        return []

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self.grid != "product" and len(self._grid()) > 1:
            options["grid"] = self.grid
        if self.workers > 1:
            options["workers"] = self.workers
        if self.what in ("irf", "fevd"):
            options["periods"] = self.periods
        if self.what == "irf":
            options.update({k: v.strip() for k, v in (("response", self.response), ("shock", self.shock)) if v.strip()})
        return options

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:sweep", output=self.name, recorded_by=self.command_id, obj=Ref(self.source),
                         values=self._grid(), what=self.what, **self._options())

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Model/Compare scenarios...")
@dataclass(frozen=True)
class CompareScenarios(Command):
    """Model paths under several scenarios, side by side: a table per variable, the differences between the first
    two, and a graph with a line per scenario.

    Tick solutions already computed (Model > Solution > Solve model, one per scenario), or choose an estimated model
    (forecast from the end of its data) or a rational-expectations solution, with scenarios made in code
    (Scenario(...)), to solve them all in one step.
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS

    solutions: Annotated[str, Widget("object", source="Solution", multiple=True, help="solutions to compare (two or more)")] = ""
    source: Annotated[str, Widget("object", source="FittedModel,LinearSolution", help="or: a model to solve under each scenario",
                                  group="Or solve scenarios")] = ""
    scenarios: Annotated[str, Widget("object", source="Scenario", multiple=True, help="the Scenario objects (made in code) to solve the model under; each gives one path",
                                     group="Or solve scenarios")] = ""
    data: Annotated[str, Widget("object", source="Dataset", help="the data (for a solution; an estimated model uses its own)",
                                group="Or solve scenarios")] = ""
    start: Annotated[str, Widget("period", group="Or solve scenarios")] = ""
    end: Annotated[str, Widget("period", group="Or solve scenarios")] = ""
    workers: Annotated[int, Widget("number", min=1, max=64, advanced=True, help="processes to run in parallel")] = 1
    baseline: Annotated[str, Widget("text", help="the scenario (or solution) the others are compared with; empty: one named "
                                         "baseline, else the first")] = ""
    loss: Annotated[str, Widget("text", group="Loss", help="weights of a loss to rank the scenarios by, e.g. pi=1, ygap=0.5 "
                                                           "(empty: none)")] = ""
    desired: Annotated[str, Widget("text", group="Loss", help="the values the loss's gaps are measured from, e.g. pi=2; "
                                                              "the others zero")] = ""
    discount: Annotated[float, Widget("number", min=0.0001, max=1.0, step=0.01, group="Loss",
                                      help="discount factor of the loss per period; 1 = none")] = 1.0
    name: str = "comparison"

    def _weights(self, text: str) -> dict[str, float]:
        from .builtin import _calibration

        return _calibration(text)

    def validate(self, session: Session) -> list[str]:
        sols, scens = _names(self.solutions), _names(self.scenarios)
        try:
            weights, desired = self._weights(self.loss), self._weights(self.desired)
        except ValueError as exc:
            return [str(exc)]
        if desired and not weights:
            return ["give the loss's weights (e.g. pi=1, ygap=0.5) for the desired values"]
        extra = [v for v in desired if v not in weights]
        if extra:
            return [f"desired values for {', '.join(extra)}, which are not in the loss"]
        if self.baseline.strip() and self.baseline.strip() not in (sols or scens):
            return [f"the baseline {self.baseline.strip()!r} is not one of the {'solutions' if sols else 'scenarios'} ticked"]
        if sols and (self.source or scens):
            return ["tick solutions, or choose a model and scenarios, not both"]
        if sols:
            missing = [n for n in sols if n not in session.objects]
            return [f"no object named {n!r}" for n in missing] or ([] if len(sols) > 1 else ["tick two or more solutions"])
        if not self.source or not scens:
            return ["tick two or more solutions, or choose a model and the scenarios to solve"]
        if not (self.start and self.end):
            return ["give the periods to solve (start and end)"]
        return [f"no object named {n!r}" for n in [self.source, *scens, *([self.data] if self.data else [])] if n not in session.objects]

    def _span(self) -> tuple[str, str]:
        return (self.start, self.end)

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        record = {"output": self.name, "recorded_by": self.command_id}
        from ..sweeps import ScenarioSet

        base = self.baseline.strip()
        loss = self._weights(self.loss)
        made = {} if loss else record  # with a loss, the set is made, then the loss set on it (one call, nested)
        if _names(self.solutions):  # the baseline goes first: compare_solutions compares with the first (or "baseline")
            sols = _names(self.solutions)
            order = [base, *[n for n in sols if n != base]] if base else sols
            call = make_call("pymetrix:compare_solutions", **made, **{n: Ref(n) for n in order})
        else:
            call = make_call("pymetrix:scenarios", source=Ref(self.source), scenarios={n: Ref(n) for n in _names(self.scenarios)},
                             span=self._span(), data=Ref(self.data) if self.data else None,
                             workers=self.workers if self.workers > 1 else None, baseline=base or None, **made)
        if not loss:
            return call
        return make_call(ScenarioSet.with_loss.__api__.id, call, **record, loss=loss,
                         desired=self._weights(self.desired) or None, discount=self.discount)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
