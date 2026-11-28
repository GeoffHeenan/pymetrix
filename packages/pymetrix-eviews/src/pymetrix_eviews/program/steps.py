"""Steps mode: core's ``pymetrix.translation.steps`` (extension API 1.11), each data or estimation statement a recorded
call (``make_call``) or command, as a dialog's step is; the EViews translation's steps are recorded as
``eviews:program``, and its script and report kept under ``pymetrix-eviews`` (``ProgramTranslation``)."""

from __future__ import annotations

from typing import Any

from pymetrix.translation import StepFailed, run_steps
from pymetrix.translation import step_call as _step_call

from .report import ATTACHMENTS, RECORDED_BY

__all__ = ["ATTACHMENTS", "StepFailed", "run_steps", "step_call"]


def step_call(step: Any, env: Any) -> Any:
    """The ``Call`` of a step, with the program's values (``env``) put into its expressions."""
    return _step_call(step, env, RECORDED_BY)
