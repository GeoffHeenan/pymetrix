"""Typed commands: the single source for forms, menus, script capture and projects.

A command is a dataclass. Its field types and ``Widget`` annotations become a
form (see ``pymetrix.forms``); ``run`` performs it against a
``Session``; ``to_code`` returns the equivalent Python, which the session
appends to its script.

Projects save each step as the command's **ID**, its **schema version** and its
fields. The ID (``"pymetrix:fit_equation"``; a plugin's commands use the plugin's
prefix) never changes, even if the class is renamed: give ``command_id`` explicitly
then. When a command's fields change in a way old projects would not survive
(a field renamed, removed or given a new meaning), raise ``schema_version`` and
register a ``migration`` from the old version, which turns old fields into new.
Adding a field with a default needs neither.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, ClassVar

from ..errors import PyMetrixError, SpecificationError
from ..forms import FieldSpec, Widget, advanced, form_spec, option_string

if TYPE_CHECKING:
    from .session import Session

__all__ = ["COMMANDS", "COMMAND_IDS", "Command", "CommandInfo", "FieldSpec", "MigrationError", "Widget", "advanced", "command",
           "form_spec", "from_json", "migration", "option_string", "to_json", "upgrade"]


class Command:
    """Base class for commands. Subclasses are dataclasses registered with ``@command``."""

    command_id: ClassVar[str] = ""  # stable identifier in projects; set by @command unless given
    schema_version: ClassVar[int] = 1  # raise when old projects need a migration to load

    def validate(self, session: Session) -> list[str]:
        return []

    def run(self, session: Session) -> Any:
        raise NotImplementedError

    def to_code(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class CommandInfo:
    cls: type[Command]
    menu: str | None  # None: recorded and replayed, but not in the menus (typed code)
    id: str = ""
    plugin: str = ""  # the plugin that provides it ("pymetrix" for built-in commands)
    plugin_version: str = ""


COMMANDS: dict[str, CommandInfo] = {}  # by class name
COMMAND_IDS: dict[str, CommandInfo] = {}  # by stable ID: what projects use

_ID = re.compile(r"^[a-z][a-z0-9_]*:[a-z][a-z0-9_]*$")


def _snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


def command(menu: str | None, id: str | None = None):
    """Register a command class under a menu path such as ``"Estimate/Equation..."``
    (``None``: not in the menus, but saved in projects and replayed). ``id`` (or the class's
    ``command_id``) fixes its identifier in projects; the default is ``plugin:snake_name``."""

    def decorate(cls: type[Command]) -> type[Command]:
        from ..plugins import current_plugin, register

        plugin = current_plugin()
        cid = id or cls.__dict__.get("command_id") or f"{plugin.prefix.replace('-', '_')}:{_snake(cls.__name__)}"
        if not _ID.match(cid):
            raise SpecificationError(f"command ID {cid!r} must look like 'plugin:name' (lower case, digits and _)")
        if not isinstance(cls.schema_version, int) or cls.schema_version < 1:
            raise SpecificationError(f"{cls.__name__}.schema_version must be a whole number, 1 or more")
        known = COMMAND_IDS.get(cid)
        if known is not None and (known.cls.__module__, known.cls.__qualname__) != (cls.__module__, cls.__qualname__):
            raise SpecificationError(f"command ID {cid!r} is already used by {known.cls.__module__}.{known.cls.__qualname__}")
        info = CommandInfo(cls, menu, cid, plugin.name, plugin.version)
        cls.command_id = cid
        COMMANDS[cls.__name__] = COMMAND_IDS[cid] = info
        cls.menu = menu  # type: ignore[attr-defined]
        if menu:
            from ..wording import menu_label

            register("commands", cls, name=cls.__name__.lower(), label=menu_label(menu).rstrip(".… "))
        return cls

    return decorate


def existing_series(session: Any, data: str, names: Any, replace: bool) -> list[str]:
    """The error for commands that add series to a dataset: names it already has, unless ``replace`` is ticked."""
    if replace or data not in session.objects:
        return []
    have = set(getattr(session.objects[data], "frame", {}).columns) if hasattr(session.objects[data], "frame") else set()
    clash = [n for n in names if n in have]
    if not clash:
        return []
    listed = ", ".join(clash[:5]) + (" ..." if len(clash) > 5 else "")
    return [f"{data} already has {'a series' if len(clash) == 1 else 'series'} {listed}: choose another name, or tick Replace "
            "(replace=True) to overwrite"]


def _replace_before_schema_2(fields: dict[str, Any]) -> dict[str, Any]:
    """Steps saved before the Replace option replaced existing series silently: they keep doing so."""
    fields.setdefault("replace", True)
    return fields


# migrations: how a step saved at one schema version becomes the next

Migration = Callable[[dict[str, Any]], dict[str, Any]]
MIGRATIONS: dict[tuple[str, int], Migration] = {}


class MigrationError(PyMetrixError, ValueError):
    """A saved step cannot be brought up to the installed command's schema."""


def migration(command_id: str, from_version: int):
    """Register a function that turns a step's fields saved at ``from_version`` into those of
    ``from_version + 1``::

        @migration("pymetrix:fit_each", 1)
        def _(fields):
            fields["selection"] = fields.pop("group")
            return fields
    """

    def decorate(fn: Migration) -> Migration:
        MIGRATIONS[(command_id, from_version)] = fn
        return fn

    return decorate


def upgrade(command_id: str, version: int, saved: dict[str, Any]) -> dict[str, Any]:
    """The fields of a step saved at ``version``, migrated to the installed command's schema and checked."""
    info = COMMAND_IDS.get(command_id)
    if info is None:
        raise MigrationError(f"no command {command_id!r} is installed")
    target = info.cls.schema_version
    if version > target:
        raise MigrationError(f"{command_id} was saved by a newer {info.plugin} (step schema {version}; installed: {target}); "
                             f"upgrade {info.plugin}")
    out = dict(saved)
    for v in range(version, target):
        step = MIGRATIONS.get((command_id, v))
        if step is None:
            raise MigrationError(f"{command_id}: no migration from schema {v} to {v + 1}")
        out = step(dict(out))
    names = {f.name for f in fields(info.cls) if f.init}
    unknown = sorted(set(out) - names)
    if unknown:
        raise MigrationError(f"{command_id} (schema {target}) has no field{'s' * (len(unknown) > 1)} {', '.join(unknown)}; "
                             "the step was saved with fields this version does not know, and no migration handles them")
    return out


def to_json(cmd: Command) -> dict[str, Any]:
    """A command as JSON: ``{"kind": "command", "id", "version", "fields"}``, the fields encoded by ``pymetrix.codec``
    (every field, defaults included). ``from_json`` reads it back, migrating it if it was saved at an older schema."""
    from ..codec import encode

    return {"kind": "command", "id": type(cmd).command_id, "version": type(cmd).schema_version,
            "fields": {f.name: encode(getattr(cmd, f.name)) for f in fields(cmd) if f.init}}  # type: ignore[arg-type]


def from_json(value: dict[str, Any]) -> Command:
    """The command ``to_json`` wrote, brought up to the installed command's schema (see ``upgrade``)."""
    from ..codec import decode

    if value.get("kind", "command") != "command":
        raise MigrationError(f"not a command: kind {value.get('kind')!r}")
    cid = value["id"]
    saved = upgrade(cid, int(value.get("version", 1)), dict(value.get("fields", {})))
    return COMMAND_IDS[cid].cls(**{k: decode(v) for k, v in saved.items()})


for _id in ("pymetrix:add_to_dataset", "pymetrix:generate_series", "pymetrix:derive_series", "pymetrix:derive_each"):
    MIGRATIONS[(_id, 1)] = _replace_before_schema_2
