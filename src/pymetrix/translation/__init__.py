"""Translating programs written for other software into pymetrix: the machinery that is not about any one language
(extension API 1.11; docs/design/r-translator.md, phase 0; the help topic "translation-framework").

A front end (a plugin: the EViews one, later R) reads a program and gives each statement an ``Outcome``: a status,
a reason, and ops of the intermediate representation (``Step``, ``Cmd``, ``Code``, the control-flow ops). From the
outcomes this module makes the three things a translation is, from one representation:

- the readable script (``render``): each line of the program a comment above the Python it became;
- the report (``report_rows``, ``TranslationReport``): a row per line with text, its status (``STATUSES``:
  translated, translated with a note, left out, error) and the reason;
- steps mode (``Translation.run``): the same work as recorded steps in a session, control flow run in Python.

A front end subclasses ``Translation`` to give its language, the steps' source and the project attachments' key,
and registers a panel made by ``translation_panel(PanelSpec(...))``. ``pymetrix.translation.testing`` has the checks
its tests need.

    from pymetrix.translation import Outcome, Source, Statement, Step, Text, Translation, TranslationReport
    from pymetrix.translation import TRANSLATED, report_rows, render

The modules: ``ir`` (the representation), ``report`` (statuses, rows, ``Translation``), ``outcome`` (``Source``,
``Outcome``, ``Untranslated``, ``report_rows``), ``names`` (Python names for the program's objects), ``script``,
``steps``, ``panel`` (Qt, made on first use) and ``testing``.
"""

from __future__ import annotations

from .ir import (Arg, Attr, Cmd, Code, Def, For, If, Invoke, Jump, New, Op, Py, RefTo, SampleSpec, Spread, Step, Text, Var,
                 While, literal, name_code, quote, resolved, step_code)
from .names import allocate_name, reserved_names
from .outcome import Outcome, Source, Statement, Untranslated, error, flatten, left, report_rows
from .panel import COLOURS, PanelSpec, panel_class, translation_panel
from .report import ERROR, LEFT, NOTE, STATUSES, TRANSLATED, ReportRow, Translation, TranslationReport, left_reason
from .script import body_lines, op_code, pymetrix_imports, render, source_comment
from .steps import WHILE_CAP, StepFailed, run_steps, step_call

__all__ = [
    # the intermediate representation
    "Arg", "Attr", "Cmd", "Code", "Def", "For", "If", "Invoke", "Jump", "New", "Op", "Py", "RefTo", "SampleSpec", "Spread",
    "Step", "Text", "Var", "While", "literal", "name_code", "quote", "resolved", "step_code",
    # what a front end makes, and the report
    "ERROR", "LEFT", "NOTE", "STATUSES", "TRANSLATED", "Outcome", "ReportRow", "Source", "Statement", "Translation",
    "TranslationReport", "Untranslated", "allocate_name", "error", "flatten", "left", "left_reason", "report_rows",
    "reserved_names",
    # the script, steps mode and the panel
    "COLOURS", "WHILE_CAP", "PanelSpec", "StepFailed", "body_lines", "op_code", "panel_class", "pymetrix_imports", "render",
    "run_steps", "source_comment", "step_call", "translation_panel",
]
