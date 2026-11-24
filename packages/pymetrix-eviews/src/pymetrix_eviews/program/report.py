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

    def line(self, number: int) -> ReportRow:
        """The row of a line (from 1)."""
        for r in self.rows:
            if r.line == number:
                return r
        from pymetrix.errors import NotFoundError

        raise NotFoundError(f"line {number} has no row: it is blank or past the end")

    def summary(self) -> Any:
        """The report as a ``Table``: line, status, EViews, Python and reason, with the counts as notes."""
        from pymetrix.output import Table

        counts = self.counts()
        notes = ["; ".join(f"{n} {s}" for s, n in counts.items()),
                 "Every line with text has a row: nothing is dropped silently. Left out: phase 2 is estimation, tests "
                 "and models; phase 3 control flow; the rest has no pymetrix equivalent or is display only."]
        rows = [[r.line, r.status, r.eviews, r.python, r.reason] for r in self.rows]
        return Table(["line", "status", "EViews", "Python", "reason"], rows,
                     title=f"Translation of {self.program}" if self.program else "Translation", notes=notes)

    def to_text(self) -> str:
        """The report as plain text, a line per row."""
        out = []
        for r in self.rows:
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

    def run(self, session: Any = None, attach: bool = True) -> dict[str, Any]:
        """Run the translation as steps (steps mode) in ``session`` (a new ``Session`` when None): each data
        statement a recorded call (``make_call``), so History and the session's script hold it as if typed or made
        in a dialog; scalars, strings and matrices are Python values, not steps, and are returned (name -> value),
        with the session under ``"session"``. With ``attach``, the readable script and the report are kept in the
        project (``session.attachments["pymetrix-eviews"]["<program>.py"]`` and ``"<program>.txt"``). Stops at the first step that fails,
        raising its error with the line."""
        from .steps import run_steps

        return run_steps(self, session, attach)

    def summary(self) -> Any:
        return self.report.summary()

    def _repr_html_(self) -> str:
        return self.report._repr_html_()

    def __repr__(self) -> str:
        c = self.report.counts()
        return (f"<ProgramTranslation {self.program or 'program'}: {len(self.report)} lines; "
                + ", ".join(f"{n} {s}" for s, n in c.items() if n) + ">")
