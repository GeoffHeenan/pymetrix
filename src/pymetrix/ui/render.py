"""Rendering objects as HTML, shared by the Jupyter and Qt frontends.

Dispatches on the object's type. What a result shows (its views) is defined in ``pymetrix.views``,
independent of any frontend; the names below are re-exported from there so existing imports and
plugins keep working.
"""

from __future__ import annotations

import html as _html
from functools import singledispatch
from typing import Any

from ..data import Dataset
from ..estimation import Equation, FittedEquation
from ..models import Solution
from ..figures import Figure
from ..output import Table
from ..stats import TestResult


@singledispatch
def to_html(obj: Any) -> str:
    if hasattr(obj, "_repr_html_"):
        return obj._repr_html_()
    return f"<pre>{_html.escape(repr(obj))}</pre>"


@to_html.register
def _(obj: Table) -> str:
    return obj._repr_html_()


@to_html.register
def _(obj: Figure) -> str:
    import base64

    return f'<img alt="{_html.escape(repr(obj))}" src="data:image/png;base64,{base64.b64encode(obj.png()).decode()}">'


@to_html.register
def _(obj: TestResult) -> str:
    return obj._repr_html_()


@to_html.register
def _(obj: FittedEquation) -> str:
    return obj.summary()._repr_html_()


@to_html.register
def _(obj: Equation) -> str:
    return f"<pre>{_html.escape(repr(obj))}</pre>"


def _frame_html(frame: Any) -> str:
    return frame.to_html(float_format=lambda v: f"{v:,.3f}")


@to_html.register
def _(obj: Dataset) -> str:
    rows = obj.frame if len(obj) <= 24 else obj.frame.iloc[list(range(8)) + list(range(-8, 0))]
    note = "" if len(obj) <= 24 else f"<p>First and last 8 of {len(obj)} rows.</p>"
    return f"<p>{_html.escape(repr(obj))}</p>" + note + _frame_html(rows)


@to_html.register
def _(obj: Solution) -> str:
    solved = obj.solved()
    cols = list(obj.model.endogenous) if obj.model is not None else solved.names
    return f"<p>{_html.escape(repr(obj))}</p>" + _frame_html(solved.frame[cols])


from ..views import Section, register_view, sections, view_sections  # noqa: E402,F401 - moved to pymetrix.views
