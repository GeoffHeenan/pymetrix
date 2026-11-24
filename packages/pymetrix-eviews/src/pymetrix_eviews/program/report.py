"""The translation's report and result: ``ReportRow``, ``TranslationReport`` and ``ProgramTranslation``."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["STATUSES", "ProgramTranslation", "ReportRow", "TranslationReport"]

TRANSLATED, NOTE, LEFT, ERROR = "translated", "translated with a note", "left out", "error"
STATUSES = (TRANSLATED, NOTE, LEFT, ERROR)


@dataclass(frozen=True)
class ReportRow:
    """One line of the program: its number (from 1), ``status`` (one of ``STATUSES``), the EViews text, the Python it
    became (empty for a line left out, or a line that continues a statement), and the reason, with pymetrix's
    equivalent where there is one."""

    line: int
    status: str
    eviews: str
    python: str = ""
    reason: str = ""
    file: str = ""


@dataclass(frozen=True)
class TranslationReport:
    """One row per line of the program that has text (blank lines have none), in order."""

    rows: tuple[ReportRow, ...]
    program: str = ""

    def counts(self) -> dict[str, int]:
        """How many lines have each status, every status listed."""
        c = Counter(r.status for r in self.rows)
        return {s: c.get(s, 0) for s in STATUSES}

    def __iter__(self) -> Any:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> ReportRow:
        return self.rows[i]

    def line(self, number: int, file: str | None = None) -> ReportRow:
        """The row of a line (from 1) of the program, or of an included one (``file``, as the report names it)."""
        for r in self.rows:
            if r.line == number and (r.file == self.program if file is None else r.file == file):
                return r
        from pymetrix.errors import NotFoundError

        raise NotFoundError(f"line {number} has no row: it is blank or past the end")

    def summary(self) -> Any:
        """The report as a ``Table``: line, status, EViews, Python and reason, with the counts as notes."""
        from pymetrix.output import Table

        counts = self.counts()
        notes = ["; ".join(f"{n} {s}" for s, n in counts.items()),
                 "Every line with text has a row, an included program's under its name: nothing is dropped silently. "
                 "Left out: an option not mapped is named; the rest has no pymetrix equivalent or is display only."]
        rows = [[r.file if r.file != self.program else "", r.line, r.status, r.eviews, r.python, r.reason] for r in self.rows]
        return Table(["file", "line", "status", "EViews", "Python", "reason"], rows,
                     title=f"Translation of {self.program}" if self.program else "Translation", notes=notes)

    def files(self) -> list[str]:
        """The program's name and the included programs', in the order their rows come."""
        out: list[str] = []
        for r in self.rows:
            if r.file not in out:
                out.append(r.file)
        return out

    def to_text(self, rows: Any = None) -> str:
        """The report (or some of its ``rows``) as plain text, a line per row; an included program's rows under a
        line naming it."""
        out = []
        file = self.program
        for r in (self.rows if rows is None else rows):
            if r.file != file:
                file = r.file
                out.append(f"  -- {file or 'the program'}")
            tail = f"  [{r.reason}]" if r.reason else ""
            out.append(f"{r.line:>5}  {r.status:<22}  {r.eviews.strip()}{tail}")
        return "\n".join(out)

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()

    def __repr__(self) -> str:
        c = self.counts()
        return f"<TranslationReport: {len(self.rows)} lines; " + ", ".join(f"{n} {s}" for s, n in c.items() if n) + ">"


@dataclass(frozen=True, eq=False)
class ProgramTranslation:
    """An EViews program translated (``translate_program``): ``script``, readable Python that uses pymetrix's public
    API; ``report``, a row per line; and ``run(session)``, the same work as recorded steps. All three come from one
    representation, so the script and the steps give equal objects."""

    script: str
    report: TranslationReport
    program: str = ""
    ops: Sequence[Any] = field(default=(), repr=False)
    pages: Mapping[str, str] = field(default_factory=dict)
    functions: Sequence[Any] = field(default=(), repr=False)  # the subroutines: (line, Def, file)
    objects: Mapping[str, Any] = field(default_factory=dict, repr=False)  # given to the translation (objects=)
    arguments: Mapping[str, str] = field(default_factory=dict)  # the program's arguments: arg0 -> value
    sources: Mapping[str, str] = field(default_factory=dict, repr=False)  # the text of the program and each included one, by name
    script_lines: Mapping[Any, int] = field(default_factory=dict, repr=False)  # (file, line) -> the script's line (from 1)

    def script_line(self, line: int, file: str | None = None) -> int | None:
        """The script's line (from 1) of a program's line, or of the nearest one before it with a line of its own;
        None before the first."""
        file = self.program if file is None else file
        best = None
        for (f, n), at in self.script_lines.items():
            if f == file and n <= line and (best is None or n > best[0]):
                best = (n, at)
        return None if best is None else best[1]

    def run(self, session: Any = None, attach: bool = True, *, args: Any = None, while_cap: int | None = None,
            run: Any = None) -> dict[str, Any]:
        """Run the translation as steps (steps mode) in ``session`` (a new ``Session`` when None): each data
        statement a recorded call (``make_call``), so History and the session's script hold it as if typed or made
        in a dialog; loops, ``if`` blocks and subroutines run in Python, a condition decided by the results made
        before it; scalars, strings and matrices are Python values, not steps, and are returned (name -> value),
        with the session under ``"session"``. With ``attach``, the readable script and the report are kept in the
        project (``session.attachments["pymetrix-eviews"]["<program>.py"]`` and ``"<program>.txt"``). Stops at the
        first step that fails, raising its error with the line.

        ``args`` replaces the program's arguments (``%0``, ``%1``: strings, or one string of words); ``while_cap`` is
        how many passes a ``while`` loop may make before it stops with an error (default 10,000); ``run`` runs each
        step instead of ``session.run`` (the app's ``AppInterface.run``, which records it and lets it be undone).
        The objects given to the translation (``objects=``) are put in the session when it has none of that name."""
        from .steps import run_steps

        return run_steps(self, session, attach, args=args, while_cap=while_cap, runner=run)

    def summary(self) -> Any:
        return self.report.summary()

    def _repr_html_(self) -> str:
        return self.report._repr_html_()

    def __repr__(self) -> str:
        c = self.report.counts()
        return (f"<ProgramTranslation {self.program or 'program'}: {len(self.report)} lines; "
                + ", ".join(f"{n} {s}" for s, n in c.items() if n) + ">")
