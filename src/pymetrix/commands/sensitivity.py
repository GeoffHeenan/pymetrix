"""Dialog for global sensitivity analysis: Model > Analysis > Stability mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session


@command(menu="Model/Stability mapping...")
@dataclass(frozen=True)
class MapStability(Command):
    """Map where in the prior a model has a unique stable solution (Ratto's prior stability mapping, as Dynare's
    dynare_sensitivity): draw the parameters from their priors, solve the model at each draw, and count the draws that
    are determinate, indeterminate, explosive, without a steady state or without a solution.

    Then, for each parameter, the Smirnov (two-sample Kolmogorov-Smirnov) test compares its values in the determinate
    draws with its values in the others. A parameter with a small p-value drives the model in or out of the
    determinate region: the cumulative distributions on the Graph tab show where (for a Taylor rule, determinate draws
    have the inflation response above one). A large share of failing draws means the priors put mass where the model
    cannot be solved: estimation will reject those values, and the posterior is then shaped by that region as much as
    by the data. Shock sizes and measurement errors are not drawn, since they do not change whether the model solves.

    The priors are the model's: a Dynare file's estimated_params, or the priors given to the parameters of a model
    or template. Parameters without a prior keep their values.
    """

    maturity: ClassVar[str] = "experimental"
    topics: ClassVar[tuple[str, ...]] = ("global-sensitivity", "forward-looking-models")
    example: ClassVar[str] = """
        from pymetrix import stability_mapping, templates
        from pymetrix.estimation.priors import Uniform

        nk = templates.new_keynesian(rho=0.0, phi_y=0.0)  # or read_mod("nk.mod"): priors from estimated_params
        st = stability_mapping(nk, draws=500, seed=1, priors={"phi_pi": Uniform(0, 3)})
        st.shares                                         # about 2/3 determinate: phi_pi above one
        st.smirnov                                        # each parameter: determinate draws against the others
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Summary", "the number and share of draws in each class, with binomial standard errors"),
        ("Smirnov tests", "each parameter's Smirnov statistic and p-value, determinate draws against the others"),
        ("Graph", "each parameter's cumulative distribution in the determinate draws and in the others"),
        ("Draws", "every draw's parameter values and class"),
    )

    source: Annotated[str, Widget("object", source="DynareModel,Template,Model,FittedModel",
                                  help="a model whose parameters have priors: a Dynare model with estimated_params, or a model "
                                       "or template with priors on its parameters")]
    draws: Annotated[int, Widget("number", min=10, max=100000, step=100,
                                 help="draws from the priors; the shares' standard errors fall as 1/sqrt(draws)")] = 2000
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, help="the random seed, so the draws can be repeated")] = 0
    pvalue_ks: Annotated[float, Widget("number", min=1e-12, max=0.5, step=0.001, advanced=True,
                                       help="a parameter drives stability when its Smirnov p-value is below this (Dynare's pvalue_ks)")] = 0.001
    name: str = "stability"

    def validate(self, session: Session) -> list[str]:
        obj = session.objects.get(self.source)
        if obj is None:
            return [f"no model named {self.source!r}"]
        if type(obj).__name__ not in ("DynareModel", "Template", "Model", "FittedModel"):
            return [f"{self.source} is a {type(obj).__name__}: choose a model with priors (a Dynare model, a template or a model)"]
        return []

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:stability_mapping", output=self.name, recorded_by=self.command_id, source=Ref(self.source),
                         draws=self.draws, seed=self.seed, pvalue_ks=self.pvalue_ks)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
