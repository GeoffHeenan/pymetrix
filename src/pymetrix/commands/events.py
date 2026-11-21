"""Session events: what happens to a session's history, for plugins that keep their own record of the work.

A plugin registers a ``SessionObserver`` in the ``observers`` group; every session, in scripts, notebooks, the app and
the MCP server, then tells it what happens::

    from pymetrix.extension import SessionObserver
    from pymetrix.plugins import register

    register("observers", SessionObserver("labbook", handle))   # handle(event) is called for every event

The events (``SessionEvent.kind``):

- ``step``: a step ran and was recorded (``records``: its records; typed code made of API calls gives several); Redo
  (``session.redo()``) runs an undone step again, and is reported as this, a step run;
- ``failed``: a step raised an error, and was recorded as failed;
- ``cancelled``: a step was stopped by Cancel; it leaves no record in the history (``code``: what it was);
- ``undone``: Undo took back a step (``records``: the records removed, the step and any recalculation after it);
- ``removed``: steps were removed from the history (``records``), e.g. by cleaning up dead ends;
- ``opened``: a project was opened (``path``; ``records``: its history as replayed);
- ``saving``: a project is about to be written (``path``); an observer may update ``session.attachments`` now.

``actor`` says who acted (``Actor``): a person, or an AI agent working through the MCP server, and through which
interface (``app``, ``python`` for scripts and notebooks, ``mcp``). The MCP server marks its steps with ``acting``;
everything else is the person at the machine.

``during`` says what the session was doing: ``work`` (a step asked for), ``recalculation`` (results brought up to
date after a change) or ``opening`` (a project's steps replayed). An observer that raises is reported once, as a
warning, and the session carries on: an observer can never stop the work it observes.
"""

from __future__ import annotations

import getpass
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

__all__ = ["Actor", "SessionEvent", "SessionObserver", "SessionObserverWarning", "acting", "emit", "recorders"]

KINDS = ("step", "failed", "cancelled", "undone", "removed", "opened", "saving")
INTERFACE = "python"  # the interface a person uses: the desktop app sets "app" when it starts


@dataclass(frozen=True)
class Actor:
    """Who took a step: ``kind`` "person" or "agent"; ``name``, the person's login or the agent's client as it names
    itself (blank if unknown); ``via``, the interface: "app", "python" (scripts and notebooks) or "mcp"."""

    kind: str = "person"
    name: str = ""
    via: str = "python"

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "name": self.name, "via": self.via}


def _person() -> Actor:
    try:
        name = getpass.getuser()
    except Exception:  # noqa: BLE001 - no login name (some containers): unknown, not an error
        name = ""
    return Actor("person", name, INTERFACE)


@contextmanager
def acting(session: Any, actor: Actor) -> Iterator[None]:
    """Mark what happens to ``session`` inside the block as done by ``actor`` (the MCP server: an agent)."""
    before = getattr(session, "actor", None)
    session.actor = actor
    try:
        yield
    finally:
        session.actor = before


@dataclass(frozen=True)
class SessionEvent:
    """Something that happened to a session's history (see the module docstring for the kinds)."""

    kind: str
    session: Any = field(repr=False)
    records: tuple[Any, ...] = ()  # the CommandRecords concerned
    code: str = ""  # cancelled: the step's code
    path: str = ""  # opened, saving: the project file
    during: str = "work"  # "work", "recalculation" or "opening"
    detail: Mapping[str, Any] = field(default_factory=dict)  # saving: {"as_copy": bool}; cancelled: {"command": class name}
    time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    actor: Actor = field(default_factory=Actor)  # who acted (extension API 1.5)


@dataclass(frozen=True)
class SessionObserver:
    """A plugin's observer of sessions: ``handle(event)`` is called for every ``SessionEvent``.

    ``records=True`` (the default) says it keeps a record of the work, such as a lab book: projects note, at each
    save, which such observers were active, so a verifier can tell work done without them."""

    name: str
    handle: Callable[[SessionEvent], None]
    records: bool = True


class SessionObserverWarning(UserWarning):
    """A session observer raised an error; the session carried on without it for that event."""


_failed: set[str] = set()


def _observers() -> list[Any]:
    from ..plugins import load, registry

    load("observers")
    return [e for e in registry("observers").entries() if isinstance(e.obj, SessionObserver)]


def emit(session: Any, kind: str, **fields: Any) -> None:
    """Tell every observer that ``kind`` happened to ``session``."""
    entries = _observers()
    if not entries:
        return
    fields.setdefault("actor", getattr(session, "actor", None) or _person())
    event = SessionEvent(kind, session, **fields)
    for entry in entries:
        try:
            entry.obj.handle(event)
        except Exception as exc:  # noqa: BLE001 - an observer must never stop the work it observes
            key = f"{entry.plugin.name}:{entry.obj.name}"
            if key not in _failed:
                _failed.add(key)
                warnings.warn(f"session observer {key} failed and was skipped: {exc}", SessionObserverWarning, stacklevel=3)


def recorders() -> dict[str, str]:
    """The observers keeping a record of the work, as projects note them at each save: name -> "plugin version"."""
    return {e.obj.name: f"{e.plugin.name} {e.plugin.version}".strip() for e in _observers() if e.obj.records}
