"""The translation's report and result: ``ReportRow``, ``TranslationReport`` and ``ProgramTranslation``, on core's
``pymetrix.translation`` (extension API 1.11): the EViews translation says its language, its steps' source and where
the project keeps its script."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pymetrix.translation import ERROR, LEFT, NOTE, STATUSES, TRANSLATED, Translation, TranslationReport
from pymetrix.translation import ReportRow as _ReportRow

__all__ = ["ERROR", "LEFT", "NOTE", "STATUSES", "TRANSLATED", "ProgramTranslation", "ReportRow", "TranslationReport"]

LANGUAGE = "EViews"
ATTACHMENTS = "pymetrix-eviews"  # session.attachments[ATTACHMENTS]["<program>.py"]: the script, saved with the project
RECORDED_BY = "eviews:program"  # the steps' source, as a dialog's ID is for its steps


@dataclass(frozen=True)
class ReportRow(_ReportRow):
    """One line of the program: its number (from 1), ``status`` (one of ``STATUSES``), the EViews text (``source``,
    also ``eviews``), the Python it became (empty for a line left out, or a line that continues a statement), and the
    reason, with pymetrix's equivalent where there is one."""

    @property
    def eviews(self) -> str:
        """The line's EViews text: ``source``."""
        return self.source


@dataclass(frozen=True, eq=False)
class ProgramTranslation(Translation):
    """An EViews program translated (``translate_program``): ``script``, readable Python that uses pymetrix's public
    API; ``report``, a row per line; and ``run(session)``, the same work as recorded steps. All three come from one
    representation, so the script and the steps give equal objects. ``pages`` are the workfile's pages, EViews name
    -> the script's variable."""

    pages: Mapping[str, str] = field(default_factory=dict)

    language: ClassVar[str] = LANGUAGE
    recorded_by: ClassVar[str] = RECORDED_BY
    attachments: ClassVar[str] = ATTACHMENTS

    def step_names(self) -> dict[str, Any]:
        from .. import dates

        return {"eviews_dates": dates}  # EViews's date functions, as the script imports them

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
        return super().run(session, attach, args=args, while_cap=while_cap, run=run)
