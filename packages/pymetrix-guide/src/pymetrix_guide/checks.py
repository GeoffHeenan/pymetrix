"""A step's check: a small declarative condition on the session, never code.

    object fit exists                  an object called fit is in the workspace
    object fit is FittedEquation       ... and is of that type (its class or a class it derives from)
    series ly in data                  the dataset data has a series ly
    step AdjustRound ran               a step of that dialog is in the history, and ran
    step SaveRound ran on june_round   ... and read or wrote june_round

Conditions join with ``and``. Names are object, series and class names: letters, digits, ``_`` and ``.``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = ["Check", "CheckError", "Condition", "parse_check"]

_NAME = r"[A-Za-z_][\w.]*"
_FORMS = (
    ("exists", re.compile(rf"object\s+({_NAME})\s+exists")),
    ("is", re.compile(rf"object\s+({_NAME})\s+is\s+(?:an?\s+)?({_NAME})")),
    ("series", re.compile(rf"series\s+({_NAME})\s+in\s+({_NAME})")),
    ("ran", re.compile(rf"step\s+({_NAME})\s+ran(?:\s+on\s+({_NAME}))?")),
)
GRAMMAR = ("object NAME exists; object NAME is TYPE; series NAME in DATASET; step DIALOG ran; "
           "step DIALOG ran on NAME; joined with 'and'")


class CheckError(ValueError):
    """A check that is not in the grammar."""


@dataclass(frozen=True)
class Condition:
    kind: str  # exists, is, series, ran
    args: tuple[str, ...]

    def holds(self, session: Any) -> bool:
        objects = getattr(session, "objects", {})
        if self.kind == "exists":
            return self.args[0] in objects
        if self.kind == "is":
            name, type_name = self.args
            return name in objects and any(c.__name__ == type_name for c in type(objects[name]).__mro__)
        if self.kind == "series":
            series, dataset = self.args
            names = getattr(objects.get(dataset), "names", None)
            return names is not None and series in names
        dialog, on = self.args[0], self.args[1] if len(self.args) > 1 else ""
        for record in getattr(session, "history", ()):
            if (record.ok and not getattr(record, "skipped", False) and type(record.command).__name__ == dialog
                    and (not on or on in record.reads or on in record.writes)):
                return True
        return False

    def describe(self) -> str:
        a = self.args
        return {"exists": lambda: f"{a[0]} exists",
                "is": lambda: f"{a[0]} is a {a[1]}",
                "series": lambda: f"the series {a[0]} is in {a[1]}",
                "ran": lambda: f"{a[0]} has run" + (f" on {a[1]}" if len(a) > 1 else "")}[self.kind]()


@dataclass(frozen=True)
class Check:
    """Every condition holds."""

    conditions: tuple[Condition, ...]
    text: str = ""

    def holds(self, session: Any) -> bool:
        if session is None:
            return False
        try:
            return all(c.holds(session) for c in self.conditions)
        except Exception:  # noqa: BLE001 - an odd object never breaks the panel; the step is just not done
            return False

    def describe(self) -> str:
        return "; ".join(c.describe() for c in self.conditions)


def parse_check(text: str) -> Check:
    """A check from its text; ``CheckError`` says what the grammar allows."""
    text = " ".join(str(text).split())
    if not text:
        raise CheckError(f"an empty check (a check is: {GRAMMAR})")
    conditions = []
    for part in re.split(r"\s+and\s+", text):
        for kind, pattern in _FORMS:
            m = pattern.fullmatch(part)
            if m:
                conditions.append(Condition(kind, tuple(g for g in m.groups() if g)))
                break
        else:
            raise CheckError(f"{part!r} is not a check (a check is: {GRAMMAR})")
    return Check(tuple(conditions), text)
