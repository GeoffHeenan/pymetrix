"""Guided case studies for pymetrix (``pymetrix-guide``): a trainee follows a workflow step by step in the desktop
app, through the real dialogs, and each step is checked as they go.

A case study is a Markdown file in a course folder with fenced ``pymetrix-step`` blocks (``casestudy``): each names
a dialog, its settings and a declarative check (``checks``), and ``##`` sections before the first step make its
introduction. The **Guide** panel (View → Guide) lists the case studies in the loaded courses, shows the
introduction and then one step at a time, with the stage of the workflow it belongs to, opens the step's dialog
filled in (**Open…**) or runs it (**Show me**), and ticks the step when its check passes. The steps are ordinary recorded steps, so the script, the
project and a lab book capture the trainee's work; the guide keeps no record of its own.

The plugin ships Lab 3 of the example course, the quarterly forecast round, as its first case study (a course in
the ``pymetrix.courses`` group, so it is in the help too).
"""

from __future__ import annotations

import re
from pathlib import Path

from ._version import __version__
from pymetrix.extension import PluginInfo

PLUGIN = PluginInfo(
    name="pymetrix-guide",
    version=__version__,
    api="1.5",  # the oldest it works with: from 1.7 the panel opens a case study's demo (AppInterface.open_demo)
    license="BSD-3-Clause",
    maintainer="Geoffrey Heenan",
    description="Guided case studies: follow a workflow step by step in the desktop app, with each step checked",
)

from .casestudy import (  # noqa: E402
    CaseStudy,
    CaseStudyError,
    CaseStudyWarning,
    IntroPage,
    Step,
    build_command,
    check_case_study,
    parse,
    read,
)
from .checks import Check, CheckError, parse_check  # noqa: E402

_STEP_FENCE = re.compile(r"^\s*(`{3,}|~{3,})\s*pymetrix-step\s*$", re.MULTILINE)
COURSES = Path(__file__).resolve().parent / "courses"
_registered = False


def register() -> None:
    """The entry point: register the Guide panel (once). Qt is imported only when the app builds the panel."""
    global _registered
    if _registered:
        return
    from pymetrix.extension import AppPanel, plugin_context, register as add

    def create(app):  # noqa: ANN001, ANN202 - an AppInterface; returns a QWidget
        from .app import create_panel

        return create_panel(app)

    with plugin_context(PLUGIN):
        add("panels", AppPanel("Guide", create, area="right"), name="guide")
    _registered = True


def course_folders() -> list[Path]:
    """The course folders the plugin ships (each a course of case studies)."""
    return sorted(p for p in COURSES.iterdir() if p.is_dir())


def load_courses() -> None:
    """The ``pymetrix.courses`` entry point: the plugin's case studies, as course material in the help."""
    from pymetrix import help as pymetrix_help

    for folder in course_folders():
        pymetrix_help.load_course(folder)


def case_studies(session: object = None) -> list[CaseStudy]:
    """The case studies found: the plugin's own, those in the loaded courses (Help → Course material…, or a
    ``pymetrix.courses`` package) and those in the session's project courses. A Markdown note is a case study when it
    has a ``pymetrix-step`` block."""
    from pymetrix import help as pymetrix_help

    folders = [*course_folders(), *(c.path for c in pymetrix_help.courses())]
    folders += [Path(f) for f in getattr(session, "courses", []) or []]
    seen: set[Path] = set()
    found = []
    for folder in folders:
        folder = Path(folder).resolve()
        if not folder.is_dir():
            continue
        for file in sorted(folder.rglob("*.md")):
            file = file.resolve()
            if file in seen or ".ipynb_checkpoints" in file.parts:
                continue
            seen.add(file)
            try:
                text = file.read_text(encoding="utf-8")
            except OSError:
                continue
            if _STEP_FENCE.search(text):
                found.append(parse(text, file))
    return found


__all__ = ["PLUGIN", "COURSES", "CaseStudy", "CaseStudyError", "CaseStudyWarning", "Check", "CheckError", "IntroPage", "Step",
           "build_command", "case_studies", "check_case_study", "course_folders", "load_courses", "parse", "parse_check", "read", "register"]
