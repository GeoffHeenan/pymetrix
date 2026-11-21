"""Dialog for a solved model's impulse responses: Model > Analysis > Impulse responses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session


@command(menu="Model/Impulse responses of a solved model...", id="pymetrix:model_impulse_responses")  # in the menu:
# Model > Analysis > Impulse responses… (menus.py); its own label differs from the VAR dialog's, as help pages need
@dataclass(frozen=True)
class ModelImpulseResponses(Command):
    """Trace how a solved forward-looking model responds to one shock: the path of each variable after the shock, as
    deviations from the steady state, from the period of the shock (horizon 0) to the horizon chosen.

    Choose a rational-expectations solution (Model > Solution > Rational expectations, or a template's or an imported
    model's solve()), the shock and its size: one unit by default; give the shock's standard deviation for a
    one-standard-deviation shock, or a negative size for a shock the other way. The model is linear, so the responses
    scale with the size and change sign with it. A variable in logs responds in log points (times 100 for percent).

    The responses come from the decision rules alone: no data, no estimation, no bands. A stable solution's responses
    die out; a persistent shock (an autoregressive coefficient near one) makes them die out slowly, so lengthen the
    horizon to see them return to zero.
    """

    maturity: ClassVar[str] = "beta"  # as the forward-looking models it serves (docs/features.md)
    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models", "model-solution")
    example: ClassVar[str] = """
        from pymetrix import templates

        nk = templates.new_keynesian().solve()                   # a LinearSolution
        irf = nk.irf(12, shock_sd={"ei": 0.25}, shocks=["ei"])   # a 0.25 shock to ei, 12 periods ahead
        irf.to_frame()                                           # a column "variable <- ei" for each variable
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Graph", "a panel per variable: its response to the shock, horizon by horizon, from the period of the shock"),
        ("Selected horizons", "the responses at selected horizons, a column 'variable <- shock' for each variable"),
        ("All horizons", "every horizon from 0 (the period of the shock) to the horizon chosen"),
    )

    solution: Annotated[str, Widget("object", source="LinearSolution",
                                    help="a solved linear model: Model > Solution > Rational expectations, or solve()")]
    shock: Annotated[str, Widget("shock", depends_on="solution", help="the shock to trace, one of the model's shocks")]
    size: Annotated[float, Widget("number", min=-1e4, max=1e4, step=0.1,
                                  help="the shock's size: 1 for one unit, its standard deviation for a "
                                       "one-standard-deviation shock, negative for a shock the other way")] = 1.0
    periods: Annotated[int, Widget("number", min=1, max=400,
                                   help="periods after the shock to trace; long enough to see the responses die out")] = 20
    name: str = "irf"

    def validate(self, session: Session) -> list[str]:
        sol = session.objects.get(self.solution)
        if sol is None:
            return [f"no solution named {self.solution!r}"]
        if type(sol).__name__ != "LinearSolution":
            return [f"{self.solution} is a {type(sol).__name__}, not a solved linear model: solve it first "
                    "(Model > Solution > Rational expectations)"]
        if self.shock not in sol.shocks:
            return [f"{self.solution} has no shock {self.shock!r}; its shocks are {', '.join(sol.shocks)}"]
        if not math.isfinite(self.size) or self.size == 0:
            return ["give the shock a size other than 0 (1 for one unit)"]
        return []

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:linear_solution.irf", Ref(self.solution), output=self.name, recorded_by=self.command_id,
                         periods=self.periods, shock_sd={self.shock: float(self.size)}, shocks=[self.shock])

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
