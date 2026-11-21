"""The system estimation dialog (SUR, 3SLS)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from ..forms import build
from ..plugins import registry
from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


@command(menu="Estimate/System of equations...")
@dataclass(frozen=True)
class FitSystem(Command):
    """Estimate several equations together: SUR for errors correlated across equations, 3SLS when regressors are
    also endogenous.

    Choose the equations (define them first with Model > Define equation). Identities are not estimated, but their
    left-hand sides count as endogenous. For 2SLS and 3SLS, the instruments are the constant and every regressor
    that does not involve the current value of an endogenous variable, plus those you add. Restrictions refer to
    linear-form coefficients as equation.coefficient (c.y), and to explicit-form parameters by name.

    SUR gains efficiency over equation-by-equation OLS when the errors are correlated across equations and the
    equations have different regressors; with identical regressors and no restrictions it reproduces OLS. 3SLS is
    consistent when regressors are endogenous and efficient when every equation is correctly specified, but a
    misspecified equation contaminates the others' estimates, whereas 2SLS keeps the damage to that equation.
    Compare 2SLS and 3SLS estimates as an informal check. Each equation must be identified: at least as many
    instruments as endogenous regressors on its right-hand side.
    """

    topics: ClassVar[tuple[str, ...]] = ("systems",)

    equations: Annotated[str, Widget("object", source="Equation", multiple=True, help="equations to estimate, and identities")]
    method: Annotated[str, Widget("extension", source="systems", help="SUR for errors correlated across equations; 2SLS or 3SLS when regressors are endogenous")] = "sur"
    method_settings: Annotated[str, Widget("options", depends_on="method", source="systems", group="Method settings")] = ""
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    instruments: Annotated[str, Widget("text", help="extra instruments for 2SLS and 3SLS, e.g. g, r[-1]")] = ""
    restrictions: Annotated[str, Widget("text", help="linear restrictions imposed in estimation, within or across equations, e.g. c.y = i.y, a + b = 1")] = ""
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    name: str = "sys"
    endogenous: Annotated[str, Widget("text", advanced=True, help="further endogenous variables, besides the left-hand sides")] = ""

    def _method(self) -> Any:
        return build(registry("systems").get(self.method), self.method_settings)

    def _system(self, session: Session) -> Any:
        from ..systems import System

        return System([session[n] for n in _names(self.equations)], tuple(_names(self.instruments)),
                      tuple(_names(self.endogenous)), self.restrictions)

    def _span(self) -> tuple[str | None, str | None] | None:
        return (self.start or None, self.end or None) if self.start or self.end else None

    def validate(self, session: Session) -> list[str]:
        missing = [n for n in _names(self.equations) if n not in session.objects]
        if missing:
            return [f"no equation named {n!r}" for n in missing]
        if self.data not in session.objects:
            return [f"no dataset named {self.data!r}"]
        try:
            self._system(session)
            self._method()
        except (ValueError, TypeError) as exc:
            return [str(exc)]
        return []

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Fields, Ref
        from ..systems import System

        system = Fields(System, {"equations": tuple(Ref(n) for n in _names(self.equations)), "instruments": tuple(_names(self.instruments)),
                                 "endogenous": tuple(_names(self.endogenous)), "restrictions": self.restrictions})
        return make_call("pymetrix:system.fit", system, output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                         method=self._method(), span=self._span())

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
