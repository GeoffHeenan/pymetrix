"""Models written as text: Model > Model editor (and its dialog form). The text is the step's own: a project keeps it
as written, in its syntax, and builds the model from it when it opens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session


@command(menu="Model/Model editor...", id="pymetrix:define_model")
@dataclass(frozen=True)
class DefineModel(Command):
    """Write a model as text, in pymetrix's syntax, Dynare's, IRIS's or EViews's, and build it: a template (pymetrix, IRIS or EViews: the
    equations, shocks and their sizes, measurement equations) or a Dynare model, ready to check, solve and estimate.

    Model > Model editor checks the text as it is typed: each problem at its line and column (an unknown name with the
    nearest declared one, an unclosed bracket, an equation for a variable another already determines), and the
    symbols with their roles, values and the equations that use them. The project keeps the text as written, with its
    comments and layout, and builds the model from it when it opens.
    """

    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models",)

    text: Annotated[str, Widget("multiline", help="the model: sections parameters, shocks, exogenous, variables, equations, "
                                                  "measurement (pymetrix), or a .mod file's text (Dynare), .model file's (IRIS) or "
                                                  "an EViews model's text or program")]
    syntax: Annotated[Literal["pymetrix", "dynare", "iris", "eviews"], Widget("choice", help="the language the text is written in")] = "pymetrix"
    name: str = "model"

    def _source(self) -> Any:
        from ..models.source import ModelSource

        return ModelSource(self.text, self.syntax, self.name)

    def validate(self, session: Session) -> list[str]:
        found = self._source().check()
        return [str(d) for d in found.errors[:5]]

    def to_call(self) -> Any:
        from ..calls import method_call
        from ..models.source import ModelSource

        return method_call(ModelSource, "build", self._source(), output=self.name, recorded_by=self.command_id)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        quote = '"""' if '"""' not in self.text else "'''"
        body = self.text if self.text.endswith("\n") else self.text + "\n"
        syntax = f", syntax={self.syntax!r}" if self.syntax != "pymetrix" else ""
        return f"{self.name} = ModelSource({quote}\n{body}{quote}{syntax}, name={self.name!r}).build()"
