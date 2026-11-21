"""Named axes of a Dataset: one optional time axis plus any number of entity axes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CodeList:
    codes: Mapping[str, str]  # "WSM" -> "Samoa"
    groups: Mapping[str, frozenset[str]] = field(default_factory=dict)  # "Pacific" -> {...}


@dataclass(frozen=True)
class TimeAxis:
    name: str = "time"
    freq: str = "Q"  # pandas period frequency; "Y-JUN" for a July-June fiscal year
    calendar: str | None = None


@dataclass(frozen=True)
class EntityAxis:
    name: str  # "country", "household", "vintage", "draw"
    codes: CodeList | None = None


Axis = TimeAxis | EntityAxis
