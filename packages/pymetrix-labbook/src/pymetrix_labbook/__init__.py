"""A research lab book for pymetrix (``pymetrix-labbook``): an append-only, hash-chained record of the work, kept in
the project and checked by ``verify``. See pymetrix's docs/design/research-audit.md.

Installed, it records every session (scripts, notebooks, the app, the MCP server) with no further set-up: each step
run, failed, cancelled, undone or removed, and each project opened or saved. ``verify(path)``, or
``pymetrix-labbook verify project.pymx`` at a command line, checks the chain, the save log, the steps and a replay.
``brief``, ``note`` and ``objective`` record the project's scope and objectives before the work, the reasoning behind
choices, and what each step addresses.
"""

from __future__ import annotations

from ._version import __version__
from pymetrix.extension import PluginInfo, SessionObserver

PLUGIN = PluginInfo(
    name="pymetrix-labbook",
    version=__version__,
    license="BSD-3-Clause",
    maintainer="Geoffrey Heenan",
    description="A research lab book: an append-only, hash-chained record of every step, kept in the project and "
                "verified by replay",
)

from .brief import brief, commit_brief, current_brief, note, objective, reveal_brief, skip_brief  # noqa: E402
from .chain import Chain, ChainProblem  # noqa: E402
from .plan import current_plan, lock_plan  # noqa: E402
from .tried import specification_curve, tried  # noqa: E402
from .recorder import fingerprint, lab_book, observe  # noqa: E402
from .verify import Verification, verify  # noqa: E402

_registered = False


def register() -> None:
    """The entry point: register the observer that writes the lab book (once)."""
    global _registered
    if _registered:
        return
    from pymetrix.extension import plugin_context, register as add

    import pymetrix.extension as extension
    from pymetrix.extension import AppPanel

    from .app import create_panel
    from .tried import report_note

    with plugin_context(PLUGIN):
        add("observers", SessionObserver("labbook", observe), name="labbook")
        add("panels", AppPanel("Lab book", create_panel, area="right"), name="labbook_panel")  # View → Lab book
        if hasattr(extension, "ReportNote"):  # extension API 1.6: reports carry "1 of 23 fitted"
            add("report_notes", extension.ReportNote("tried", "What was tried", report_note), name="tried")
    _registered = True


__all__ = ["PLUGIN", "Chain", "ChainProblem", "Verification", "brief", "commit_brief", "current_brief", "current_plan",
           "fingerprint", "lab_book", "lock_plan", "note", "objective", "observe", "register", "reveal_brief", "skip_brief",
           "specification_curve", "tried", "verify"]
