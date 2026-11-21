"""Zoom for the desktop app's views: for projectors, screen sharing and screenshots.

The View menu's Zoom in, Zoom out and Actual size (Ctrl+= or Ctrl++, Ctrl+-, Ctrl+0) act on the active window only:
they call its ``zoom_in``, ``zoom_out`` or ``zoom_reset``. A view that can zoom says how with ``make_zoomable``, giving
a function that shows it at a factor (text, figures, row heights); the menus, docks and the rest of the application
keep their size. Zoom is view state: it is never saved in a project.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

LEVELS = (0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0)
ACTUAL = LEVELS.index(1.0)


def make_zoomable(widget: Any, show_at: Callable[[float], None]) -> Any:
    """Give ``widget`` zoom_in, zoom_out, zoom_reset, zoom_to(level) and zoom_factor(); ``show_at(factor)`` shows the
    view at that factor of its normal size."""
    widget.zoom_level = ACTUAL

    def zoom_to(level: int) -> None:
        widget.zoom_level = min(max(int(level), 0), len(LEVELS) - 1)
        show_at(LEVELS[widget.zoom_level])

    widget.zoom_to = zoom_to
    widget.zoom_in = lambda: zoom_to(widget.zoom_level + 1)
    widget.zoom_out = lambda: zoom_to(widget.zoom_level - 1)
    widget.zoom_reset = lambda: zoom_to(ACTUAL)
    widget.zoom_factor = lambda: LEVELS[widget.zoom_level]
    return widget


def scale_font(widget: Any, factor: float) -> None:
    """The widget's font (and its children's, unless they set their own) at ``factor`` of its normal size. The normal
    size is the text size's (View → Text size, ``pymetrix.ui.text_size``): the zoom multiplies it, and a text size
    chosen later keeps the zoom."""
    from .text_size import factor as text_factor

    if not hasattr(widget, "base_point_size"):  # its size at Normal text and actual size
        widget.base_point_size = widget.font().pointSizeF() / text_factor() / getattr(widget, "zoom_shown_at", 1.0)
    widget.zoom_shown_at = factor
    font = widget.font()
    font.setPointSizeF(widget.base_point_size * text_factor() * factor)
    widget.setFont(font)
