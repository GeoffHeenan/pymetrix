"""The lab book itself: an append-only list of entries, each holding the hash of the one before.

One entry is a JSON object::

    {"seq": 3, "time": "2026-10-27T09:14:02+00:00", "event": "step", "during": "work",
     "by": {"kind": "agent", "name": "claude-ai 0.1.0", "via": "mcp"},
     "data": {...}, "prev": "<hash of entry 2>", "hash": "<hash of this entry, prev included>"}

``by`` says who acted: a person (their login, through the app or Python) or an AI agent (its MCP client). It is
hashed with the rest of the entry, so it cannot be changed afterwards; entries written before it existed lack it.

``hash`` is the SHA-256 of the entry's canonical JSON (sorted keys, no spaces) without the ``hash`` field. Changing,
removing or reordering an entry breaks every later ``prev``, so editing is detectable even though nothing can
prevent it. The lab book is stored as JSON Lines (one entry a line) in the project.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

__all__ = ["GENESIS", "Chain", "ChainProblem", "entry_hash"]

GENESIS = "0" * 64  # the "previous hash" of the first entry


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")


def entry_hash(entry: dict[str, Any]) -> str:
    """The hash of an entry: SHA-256 of its canonical JSON, without its own ``hash`` field."""
    return hashlib.sha256(_canonical({k: v for k, v in entry.items() if k != "hash"})).hexdigest()


@dataclass(frozen=True)
class ChainProblem:
    """Where a lab book's chain is broken."""

    seq: int | None  # the entry's position (from 0), or None for the file as a whole
    message: str

    def __str__(self) -> str:
        return self.message if self.seq is None else f"entry {self.seq}: {self.message}"


@dataclass
class Chain:
    """The entries of a lab book, appended in order."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def head(self) -> str:
        """The hash of the last entry (``GENESIS`` when empty): what a time-stamp would anchor."""
        return self.entries[-1]["hash"] if self.entries else GENESIS

    def append(self, event: str, data: dict[str, Any], during: str = "work", time: datetime | None = None,
               by: dict[str, str] | None = None) -> dict[str, Any]:
        """Add an entry after the last one, chained to it, and return it. ``by``: who acted (see the module docstring)."""
        entry = {"seq": len(self.entries), "time": (time or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
                 "event": event, "during": during, "data": data, "prev": self.head}
        if by is not None:
            entry["by"] = by
        entry = json.loads(_canonical(entry))  # as it will be read back: JSON values only
        entry["hash"] = entry_hash(entry)
        self.entries.append(entry)
        return entry

    def to_bytes(self) -> bytes:
        return b"".join(_canonical(e) + b"\n" for e in self.entries)

    @classmethod
    def from_bytes(cls, blob: bytes) -> Chain:
        """Read a lab book; lines that are not JSON objects are kept as problems for ``check`` to report."""
        entries = []
        for i, line in enumerate(blob.decode("utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                entry = {"seq": i, "unreadable": line[:80]}
            entries.append(entry if isinstance(entry, dict) else {"seq": i, "unreadable": line[:80]})
        return cls(entries)

    def check(self) -> list[ChainProblem]:
        """Every break in the chain: unreadable lines, numbering, links and hashes. Empty when intact."""
        problems: list[ChainProblem] = []
        prev = GENESIS
        for i, entry in enumerate(self.entries):
            if "unreadable" in entry:
                problems.append(ChainProblem(i, f"is not a lab-book entry: {entry['unreadable']!r}"))
                prev = None  # the next link cannot be checked against it
                continue
            if entry.get("seq") != i:
                problems.append(ChainProblem(i, f"is numbered {entry.get('seq')}: entries were removed, added or reordered"))
            if prev is not None and entry.get("prev") != prev:
                problems.append(ChainProblem(i, "does not follow the entry before it: an entry before it was changed, removed "
                                                "or inserted"))
            if entry.get("hash") != entry_hash(entry):
                problems.append(ChainProblem(i, "has been changed since it was written (its hash does not match)"))
            prev = entry.get("hash")
        return problems
