"""Charts in the app: drawn in the app's font and on its window colours, sharp on high-resolution screens.

``look(widget)`` gives what ``Figure.draw`` takes for that: the application's UI font family and the widget's palette
as a ``figure_style.Surface``: a dark palette gives a chart on the window's own dark colour with light text; a light
palette the usual white chart. The surface carries the app's text size (View → Text size, ``text_size``), so the
chart's text follows it on screen. ``sharpen(canvas)`` makes a matplotlib canvas draw at
the screen's device pixel ratio now, rather than only once it is first shown; the figure's size in inches is unchanged.
Copied and saved images are drawn again, light, as every exported file is.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..figure_style import FALLBACK_FONT, LIGHT, Surface, known_font, surface_from

__all__ = ["look", "sharpen"]


def look(widget: Any = None) -> tuple[str | None, Surface]:
    """The app's font family and the colours charts are drawn on in ``widget`` (the application's, without one).

    The family is one matplotlib can find: the family Qt really draws the app's text in, else the family the app asks
    for, else matplotlib's own DejaVu Sans. Qt's resolved family is not always a font file's name: ``font().family()``
    can be an alias ("Sans Serif"), and where Qt finds no fonts at all (offscreen Qt on a machine without them) it
    resolves to nothing; matplotlib could find neither, and would log "font not found" for every chart drawn."""
    try:
        from PySide6.QtGui import QFontInfo, QPalette
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return None, LIGHT
    app = QApplication.instance()
    if app is None:
        return None, LIGHT
    palette = widget.palette() if widget is not None else app.palette()
    font = app.font()
    family = known_font(QFontInfo(font).family()) or known_font(font.family()) or FALLBACK_FONT
    surface = surface_from(palette.color(QPalette.ColorRole.Window).name(), palette.color(QPalette.ColorRole.WindowText).name())
    from .text_size import factor

    # a light palette: white, as the chart is copied and saved; the text at the app's text size, on screen only
    return family, replace(surface if surface.dark else LIGHT, text_scale=factor())


def sharpen(canvas: Any) -> float:
    """Draw ``canvas`` at its screen's device pixel ratio (2 on most high-resolution screens): matplotlib then renders
    that many pixels per logical pixel, the figure's dpi scaled and its size in inches kept. Returns the ratio."""
    try:
        ratio = float(canvas.devicePixelRatioF()) or 1.0
    except (AttributeError, RuntimeError):
        return 1.0
    setter = getattr(canvas, "_set_device_pixel_ratio", None)  # matplotlib's own (3.5+): what a screen change calls
    if setter is not None:
        setter(ratio)
    return ratio
