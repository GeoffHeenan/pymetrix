"""Dialogs for optimal policy: Model > Policy > Optimal policy and Model > Policy > Optimal simple rule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session

_LOSS_HELP = "weights of the loss, e.g. pi=1, y=0.25 (the instrument too, to penalise its moves)"


def _weights(text: str) -> dict[str, float]:
    from .builtin import _calibration

    return _calibration(text)


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


@command(menu="Model/Optimal policy...")
@dataclass(frozen=True)
class SolveOptimalPolicy(Command):
    """Set the policy instrument optimally, minimising a quadratic loss, under commitment (from the timeless
    perspective) or discretion, in place of the model's policy rule.

    The loss is the discounted sum of weighted squares of the variables named (e.g. pi=1, y=0.25). The equation that
    determines the instrument is dropped; name it if the instrument is not its target. The result shows the loss,
    the responses to shocks and the decision rules (under commitment with the Lagrange multipliers as states).
    """

    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models",)

    source: Annotated[str, Widget("object", source="Template,DynareModel", help="a template, Dynare or IRIS model")]
    instrument: Annotated[str, Widget("text", help="the policy instrument, e.g. i")]
    loss: Annotated[str, Widget("text", help=_LOSS_HELP)]
    kind: Annotated[Literal["commitment", "discretion"], Widget("choice", help="commitment (timeless): policy honours promises about the future; discretion: re-optimised each period")] = "commitment"
    beta: Annotated[float, Widget("number", min=0.0001, max=0.9999, help="the policymaker's discount factor; usually the households' (0.99 for quarterly models)")] = 0.99
    rule: Annotated[str, Widget("text", advanced=True, help="the equation to replace, if the instrument is not its target")] = ""
    name: str = "optimal"

    def validate(self, session: Session) -> list[str]:
        if self.source not in session.objects:
            return [f"no model named {self.source!r}"]
        try:
            weights = _weights(self.loss)
        except ValueError as exc:
            return [str(exc)]
        return [] if weights else ["give the loss weights, e.g. pi=1, y=0.25"]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:optimal_policy", output=self.name, recorded_by=self.command_id, obj=Ref(self.source),
                         instrument=self.instrument.strip(), loss=_weights(self.loss), kind=self.kind, beta=self.beta,
                         rule=self.rule.strip() or None)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Model/Optimal simple rule...")
@dataclass(frozen=True)
class OptimalSimpleRule(Command):
    """Choose the coefficients of the model's policy rule to minimise the weighted unconditional variances
    (Dynare's osr). Coefficients that leave no unique stable solution are ruled out; bounds keep the search in a
    sensible region (e.g. phi_pi above one).

    The loss is the weighted sum of the variables' unconditional variances, so it depends on the model's shock
    variances: the best rule is best for the shocks the model has, and changes if their relative sizes change.
    Unlike fully optimal policy, a simple rule responds only to the variables it names, which makes it easier to
    communicate and less dependent on the model being right. Penalise the instrument (give it a weight) or bound the
    coefficients, or the search may push them to implausibly large values that make policy very aggressive. The
    search is numerical (Nelder-Mead from the model's current coefficients, then a bounded refinement) and may find a
    local minimum, so check that the result is stable when the bounds or the model's coefficients change.
    """

    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models",)

    source: Annotated[str, Widget("object", source="Template,DynareModel", help="a template, Dynare or IRIS model")]
    params: Annotated[str, Widget("text", help="parameters of the policy rule to choose, e.g. phi_pi, phi_y; the others keep their values")]
    loss: Annotated[str, Widget("text", help=_LOSS_HELP)]
    lower: Annotated[str, Widget("text", help="lower bounds, e.g. phi_pi=1.01 (the Taylor principle), phi_y=0", group="Bounds")] = ""
    upper: Annotated[str, Widget("text", help="upper bounds, e.g. phi_pi=10, phi_y=5", group="Bounds")] = ""
    name: str = "best_rule"

    def _bounds(self) -> dict[str, tuple[float, float]]:
        lo, hi = _weights(self.lower), _weights(self.upper)
        return {p: (lo.get(p, float("-inf")), hi.get(p, float("inf"))) for p in _names(self.params) if p in lo or p in hi}

    def validate(self, session: Session) -> list[str]:
        if self.source not in session.objects:
            return [f"no model named {self.source!r}"]
        if not _names(self.params):
            return ["name the coefficients to choose"]
        try:
            weights, _ = _weights(self.loss), self._bounds()
        except ValueError as exc:
            return [str(exc)]
        return [] if weights else ["give the loss weights, e.g. pi=1, y=0.25"]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:osr", output=self.name, recorded_by=self.command_id, obj=Ref(self.source),
                         params=_names(self.params), loss=_weights(self.loss), bounds=self._bounds() or None)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
