"""A saved step whose plugin is not installed (or is disabled): kept in the history, not run, and saved again as it was.

Opening a project replays its steps. A step from a plugin that is not here cannot run, and neither can the steps
that use what it made, but the project still opens: those steps stay in the history marked as not run, with what
to install, and saving the project writes them back unchanged, so nothing is lost for whoever has the plugin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from ..errors import SpecificationError

if TYPE_CHECKING:
    from .session import Session


@dataclass(frozen=True)
class MissingStep:
    """A step that cannot run here: its saved entry, the plugin it needs and why it is missing."""

    entry: dict[str, Any] = field(default_factory=dict)
    plugin_name: str = ""  # as the project names it ("pymetrix-foo"), or from the step's identifier
    plugin_version: str = ""  # the version the project was saved with, if it says
    reason: str = ""

    @property
    def plugin(self) -> str:
        return f"{self.plugin_name} {self.plugin_version}".strip() or "a plugin"

    @property
    def plugin_label(self) -> str:
        return self.plugin

    replayable: ClassVar[bool] = False
    topics: ClassVar[tuple[str, ...]] = ()

    @property
    def command_id(self) -> str:
        return str(self.entry.get("id") or self.entry.get("command") or "")

    @property
    def schema_version(self) -> int:
        return int(self.entry.get("version", 1))

    @property
    def name(self) -> str:
        outputs = self.entry.get("outputs") or ()
        return str(outputs[0]) if outputs else ""

    def run(self, session: Session) -> Any:
        raise SpecificationError(f"the step {self.command_id} needs {self.plugin}: {self.reason}")

    def validate(self, session: Session) -> list[str]:
        return [f"the step {self.command_id} needs {self.plugin}, which is not installed"]

    def to_code(self) -> str:
        call = self.entry.get("call") or {}
        what = call.get("function") or self.command_id
        outputs = ", ".join(self.entry.get("outputs") or ())
        made = f" (it made {outputs})" if outputs else ""
        return f"# not run: {what} needs {self.plugin}{made}; {self.reason}"

    def __getattr__(self, name: str) -> Any:  # a dialog's fields, read by code that looks at steps: none here
        if name.startswith("__") or name == "entry":  # not while being copied or unpickled, before entry is set
            raise AttributeError(name)
        fields_ = (self.__dict__.get("entry") or {}).get("fields") or {}
        if name in fields_:
            return fields_[name]
        raise AttributeError(name)
