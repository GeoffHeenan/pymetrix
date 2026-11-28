"""The EViews translation panel (``AppPanel("EViews translation", create_panel, area="bottom")``, View → EViews
translation): the report of a program translated with File → Translate EViews program…. The panel is core's
translation panel (``pymetrix.translation``, extension API 1.11), told where the EViews translations are and which
dialog makes them:

- a table, a row per line: its number (and file, for an included program), its status in words and in colour, the
  EViews text, the Python and the reason;
- the program and the script side by side; selecting a row scrolls both to that line;
- a filter for each status, with its count;
- Copy (the selected rows, or the whole report, as text) and Copy script;
- Run as steps: the translation's steps run one at a time through ``AppInterface.run``, so each is recorded and can be
  undone; the run stops at the first step that fails, and that row is marked.
"""

from __future__ import annotations

from typing import Any

from pymetrix.translation import COLOURS, PanelSpec, panel_class

__all__ = ["COLOURS", "SPEC", "create_panel", "panel_class"]


def _translations(session: Any) -> list[tuple[Any, Any]]:
    from .app import translations

    return translations(session)


def _spec() -> PanelSpec:
    from .app import TranslateEViewsProgram

    return PanelSpec("EViews", _translations, TranslateEViewsProgram, "File → Translate EViews program…", "an EViews program",
                     noun="program", prefix="eviews")


SPEC: Any = None  # the panel's PanelSpec, made on first use (the dialog's class is in app.py, which imports this)
COLUMNS = ("line", "status", "EViews", "Python", "reason")


def create_panel(app: Any) -> Any:
    """The panel's widget (``AppPanel.create``): ``app`` is pymetrix's ``AppInterface``."""
    global SPEC
    if SPEC is None:
        SPEC = _spec()
    return panel_class()(app, SPEC)
