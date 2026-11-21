"""One text size for the whole desktop app: View → Text size (Smaller, Normal, Larger, Largest).

The step scales the application's font from the font it has at Normal (the system's own, as Qt gives it), so every
dialog, panel, table, result window, help page and code editor follows; charts on screen follow through the chart
style's font sizes (``chart_look.look`` gives the scale, ``figure_style.scaled`` applies it), while exported and
copied files keep their fixed sizes. Spacing set through ``pymetrix.ui.metrics`` follows too, and so do the objects'
icons (``pymetrix.ui.icons.view_size``).

The per-window zoom (``pymetrix.ui.zoom``) still works on top: the text size is the base, the zoom multiplies it.

The choice is kept in the app's settings (``QSettings("pymetrix", "desktop")``, key ``view/text_size``), applied as
the main window is built, and applied live when changed: ``apply`` changes the application's font, and every widget
that set a size of its own (a code editor's fixed-width font, a zoomed window, a bold header) by the same ratio, so
nothing has to restart. A widget that draws text in a font Qt does not know as its own (the Immediate console's
document) gives itself a ``text_size_changed(ratio)`` method, which ``apply`` calls. No keyboard shortcut: Ctrl+=, Ctrl++, Ctrl+- and Ctrl+0 are the zoom's, and Ctrl+Shift+= is
Ctrl++ on many keyboards.
"""

from __future__ import annotations

import contextlib
from typing import Any

from ..errors import SpecificationError

__all__ = ["FACTORS", "KEY", "STEPS", "apply", "current", "factor", "fixed_font", "save", "saved", "scaled_point_size"]

STEPS = ("Smaller", "Normal", "Larger", "Largest")
FACTORS = {"Smaller": 0.85, "Normal": 1.0, "Larger": 1.15, "Largest": 1.3}
KEY = "view/text_size"  # in QSettings("pymetrix", "desktop")
_SIZE_RESOLVED = 0x0002  # QFont::SizeResolved: the font's size was set, not inherited
_STATE: dict[str, Any] = {"step": "Normal", "normal": None}  # the step applied now, and the app's font at Normal


def _settings() -> Any:
    from PySide6.QtCore import QSettings

    return QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope, "pymetrix", "desktop")


def saved() -> str:
    """The step kept in the settings ("Normal" when none is, or the settings cannot be read)."""
    try:
        value = _settings().value(KEY, "Normal")
    except Exception:  # noqa: BLE001 - a preference that cannot be read is Normal, never a failure
        return "Normal"
    return value if value in FACTORS else "Normal"


def save(step: str) -> None:
    """Keep ``step`` in the settings for the next start (a setting that cannot be written is let go)."""
    if step not in FACTORS:
        raise SpecificationError(f"no text size {step!r}: one of {', '.join(STEPS)}")
    with contextlib.suppress(Exception):
        settings = _settings()
        settings.setValue(KEY, step)
        settings.sync()


def current() -> str:
    """The step applied now."""
    return _STATE["step"]


def factor(step: str | None = None) -> float:
    """The scale of ``step`` (the one applied now, without one) over Normal: 0.85, 1, 1.15 or 1.3."""
    return FACTORS[step or current()]


def scaled_point_size(points: float) -> float:
    """``points`` (a size at Normal) at the text size applied now."""
    return points * factor()


def fixed_font(least: float = 0) -> Any:
    """The system's fixed-width font (for code), at least ``least`` points at Normal, at the text size applied now."""
    from PySide6.QtGui import QFontDatabase

    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    if font.pointSizeF() > 0:
        font.setPointSizeF(max(font.pointSizeF(), least) * factor())
    return font


def _resized(font: Any, ratio: float) -> Any:
    from PySide6.QtGui import QFont

    out = QFont(font)
    if out.pointSizeF() > 0:
        out.setPointSizeF(out.pointSizeF() * ratio)
    elif out.pixelSize() > 0:
        out.setPixelSize(max(1, round(out.pixelSize() * ratio)))
    return out


def apply(step: str, app: Any = None) -> float:
    """Show the whole app at ``step``, at once: the application's font at that step's scale of its Normal font, every
    widget with a size of its own by the same ratio, and the spacing set through ``metrics`` again. Returns the scale.
    Nothing to do without a QApplication, or at the step applied now."""
    if step not in FACTORS:
        raise SpecificationError(f"no text size {step!r}: one of {', '.join(STEPS)}")
    if app is None:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
    old = current()
    if app is None or step == old:
        return factor(step)
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QWidget

    if old == "Normal" or _STATE["normal"] is None:  # the font at Normal is whatever the app has at Normal
        _STATE["normal"] = QFont(app.font())
    ratio = FACTORS[step] / FACTORS[old]
    own = []  # every widget's own size, worked out before any changes: a widget that sets its children's fonts from
    # its own as its font changes (the statistics panel) then lands on the same sizes whichever is changed first
    for widget in app.allWidgets():
        try:
            font = QWidget.font(widget)  # not widget.font: a console's is a property of its own (qtconsole)
            if widget.testAttribute(Qt.WidgetAttribute.WA_SetFont) and font.resolveMask() & _SIZE_RESOLVED:
                own.append((widget, _resized(font, ratio)))
        except RuntimeError:  # deleted while being listed
            continue
    _STATE["step"] = step
    for widget, font in own:
        with contextlib.suppress(RuntimeError):
            QWidget.setFont(widget, font)
    app.setFont(_resized(_STATE["normal"], FACTORS[step]))
    from PySide6.QtCore import QEvent

    # Qt posts the change to the widgets there are: delivered now, so a window made next (a result drawn again) takes
    # the new size from its parent rather than the old one
    app.sendPostedEvents(None, QEvent.Type.ApplicationFontChange.value)
    for widget in app.allWidgets():  # a widget that draws its text in a font of its own (the console's document)
        follow = getattr(widget, "text_size_changed", None)
        if callable(follow):
            with contextlib.suppress(RuntimeError):
                follow(ratio)
    from . import icons, metrics

    metrics.reapply(app)
    icons.reapply(app)  # the views that show the objects' icons beside text: the icons at the new size
    return FACTORS[step]
