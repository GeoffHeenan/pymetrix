"""Dialog for second- and third-order solutions: Model > Second-order solution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session


@command(menu="Model/Second-order solution...")
@dataclass(frozen=True)
class SolveSecondOrder(Command):
    """Solve a model to second order (Schmitt-Grohe and Uribe; Dynare's stoch_simul(order=2)), with its shock sizes.

    Unlike the first-order solution, it reflects risk: the mean of each variable differs from its steady state
    (precautionary saving, risk premia), and responses depend on the size and sign of shocks. The result shows the
    decision rules with their quadratic terms, the steady state against the ergodic mean, and responses from the
    stochastic steady state. Simulations are pruned so that they stay stable.

    Order 3 (Dynare's stoch_simul(order=3)) adds the cubic terms and the corrections for risk to the responses to
    states and shocks; its result adds moments from a long pruned simulation.
    """

    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models",)

    source: Annotated[str, Widget("object", source="Template,DynareModel", help="a template, Dynare or IRIS model")]
    name: str = "second"
    order: Annotated[Literal[2, 3], Widget("choice", help="2 (second order) or 3 (third order: cubic terms, and the effect of "
                                                          "risk on the responses to states and shocks)")] = 2

    def validate(self, session: Session) -> list[str]:
        return [] if self.source in session.objects else [f"no model named {self.source!r}"]

    def to_call(self, session: Session) -> Any:
        from ..calls import method_call
        from ..codec import Ref

        return method_call(type(session[self.source]), "solve", Ref(self.source), output=self.name, recorded_by=self.command_id,
                           order=self.order)

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        return f"{self.name} = {self.source}.solve(order={self.order})"
