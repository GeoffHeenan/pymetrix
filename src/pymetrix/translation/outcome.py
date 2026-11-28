"""What a front end makes of a program: the program read (``Source``, ``Statement``), and each statement's ``Outcome``
(its status, reason and ops); ``Untranslated``, raised by a statement's handler that leaves it out or finds an
error; and ``report_rows``, a row per line with text from the outcomes, so every line has exactly one."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .report import ERROR, LEFT, TRANSLATED, ReportRow

__all__ = ["Outcome", "Source", "Statement", "Untranslated", "error", "flatten", "left", "report_rows"]


@dataclass(frozen=True)
class Statement:
    """One statement of a program: its ``text`` (comments out, continued lines joined), on lines ``line`` to
    ``last_line`` (from 1). A front end's own statements may carry more (the EViews plugin's carry the command word,
    its options and offsets); these three are what the report and the script read."""

    text: str
    line: int
    last_line: int


@dataclass(frozen=True)
class Source:
    """A program read: its ``source`` text, its ``lines``, its ``statements`` (each with ``text``, ``line`` and
    ``last_line``) and each line's ``comments`` (the comment's text, by line number), which the script keeps."""

    source: str
    lines: tuple[str, ...]
    statements: tuple[Any, ...]
    comments: dict[int, str] = field(default_factory=dict)


@dataclass
class Outcome:
    """One statement translated: its status, reason and ops; a block's also its body (``children``), its else branch
    (``orelse``, after ``else_outcome``) and its end (``closer``); ``file`` the included program it is in ('' for the
    main one); ``kind`` ``"include"`` for an included program or ``"subroutine"`` for a function defined above."""

    statement: Any
    status: str
    reason: str = ""
    ops: list[Any] = field(default_factory=list)
    file: str = ""
    kind: str = ""
    children: list[Any] = field(default_factory=list)
    orelse: list[Any] = field(default_factory=list)
    else_outcome: Any = None
    closer: Any = None


class Untranslated(Exception):
    """A statement that does not translate: its ``status`` (left out or error) and the ``reason``."""

    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status, self.reason = status, reason


def left(reason: str) -> Untranslated:
    """A statement left out, with the reason (``left_reason`` adds pymetrix's equivalent): raise it."""
    return Untranslated(LEFT, reason)


def error(reason: str) -> Untranslated:
    """A statement that does not read, or names what the program has not made: raise it."""
    return Untranslated(ERROR, reason)


def flatten(outcomes: Iterable[Any]) -> list[Any]:
    """Every outcome, a block's body, else branch and end included, in the program's order."""
    out = []
    for o in outcomes:
        if o is None:
            continue
        out.append(o)
        out += flatten(o.children)
        if o.else_outcome is not None:
            out.append(o.else_outcome)
        out += flatten(o.orelse)
        if o.closer is not None:
            out.append(o.closer)
    return out


def _python(outcome: Any) -> str:
    from .script import op_code

    return "\n".join(op_code(op) for op in outcome.ops)


def report_rows(program: Any, outcomes: Iterable[Any], name: str, file: str = "", row: type = ReportRow) -> list[Any]:
    """A row (``row``, a ``ReportRow`` class) per line with text of ``program`` (a ``Source``; ``file`` '' for the
    main one, named ``name``): a statement's first line with its outcome's status, Python and reason, its other lines
    as continuing it, a comment line as translated (the script keeps it); after an ``include`` outcome, the rows of
    the included program under its name."""
    from .report import NOTE

    by_line: dict[int, tuple[Any, bool]] = {}
    includes: dict[int, Any] = {}
    for o in flatten(outcomes):
        if o.file != file:
            continue
        if o.kind == "include" and o.status in (TRANSLATED, NOTE):
            includes[o.statement.line] = o
        for line in range(o.statement.line, o.statement.last_line + 1):
            by_line.setdefault(line, (o, line == o.statement.line))
    shown = file or name
    rows = []
    for number, raw in enumerate(program.lines, start=1):
        if not raw.strip():
            continue
        found = by_line.get(number)
        if found is None:
            if number in program.comments:
                rows.append(row(number, TRANSLATED, raw, "# " + program.comments[number], "a comment", shown))
            else:
                rows.append(row(number, TRANSLATED, raw, "", "", shown))
            continue
        o, first = found
        if first:
            rows.append(row(number, o.status, raw, _python(o), o.reason, shown))
        else:
            rows.append(row(number, o.status, raw, "", f"continues line {o.statement.line}", shown))
        inc = includes.get(number) if first else None
        if inc is not None and inc.children is not None:
            sub_program = getattr(inc, "program", None)
            if sub_program is not None:
                rows += report_rows(sub_program, inc.children, name, getattr(inc, "included_file", ""), row)
    return rows
