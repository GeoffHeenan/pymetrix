"""The desktop app's colours, in one place.

Text that says something (an error, a warning, a success, a link, secondary "muted" text) takes a named role here
rather than a colour written where it is used, so every window says "error" in the same red. Each role has a value for
a light theme and one for a dark theme, picked from the application palette's lightness (the system's own theme: the
app sets none). Whatever the palette, a role's colour reads at a contrast of at least 4.5:1 (WCAG AA for body text)
against the window and the list/field background: one that would not is darkened or lightened until it does.

Muted text was the palette's placeholder text: on Qt's default light palette that is black at half opacity, about 3.9:1
on the window grey, too faint. :func:`muted` mixes the text colour into the background only as far as 4.5:1 allows.

The other colours (syntax highlighting, the spreadsheet's cell tints, the marks drawn on charts) are not roles: they
are kept here as they were, so no colour is written anywhere else in ``pymetrix.ui``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ROLES", "MIN_CONTRAST", "colour", "contrast", "css", "is_dark", "muted", "muted_css", "name", "readable"]

MIN_CONTRAST = 4.5

# role: (light theme, dark theme). The light values are the darker shades of the ones used before (#d9534f, #2e7d32,
# #1c7ed6, #b8860b), which read at only 3 to 4:1 on a light grey window; the dark values are their light tints.
ROLES = {
    "error": ("#c62828", "#ff8a80"),
    "warning": ("#9a5b00", "#ffc46b"),
    "success": ("#2a6d2f", "#7fd17f"),
    "link": ("#1864ab", "#74c0fc"),
}

# Not roles: colours on a fixed or their own background, unchanged from where they were written before.
PYTHON_SYNTAX = {  # the script editor (editor_qt): dark like One Dark, light like the JetBrains default
    "dark": {"kw": "#c678dd", "builtin": "#56b6c2", "def": "#61afef", "num": "#d19a66", "str": "#98c379", "comment": "#7f848e", "deco": "#e5c07b"},
    "light": {"kw": "#0033b3", "builtin": "#00627a", "def": "#00627a", "num": "#1750eb", "str": "#067d17", "comment": "#8c8c8c", "deco": "#9e880d"},
}
MODEL_SYNTAX = {  # the model editor (model_editor_qt)
    "dark": {"keyword": "#c678dd", "number": "#d19a66", "string": "#98c379", "timing": "#61afef", "comment": "#7f848e"},
    "light": {"keyword": "#7a3e9d", "number": "#b05a00", "string": "#2e7d32", "timing": "#1f5fbf", "comment": "#8a8a8a"},
}
EDITOR_BACKGROUNDS = {  # (dark, light): line backgrounds in the code editors
    "current line": ("#2c313a", "#f2f6fc"),
    "error line": ("#5c2b2b", "#fde2e2"),
    "same word": ("#3e4451", "#c8e1ff"),
}
SHEET_CELLS = {  # (dark, light): the spreadsheet's cell tints
    "edited": ("#5c4b00", "#fff3bf"),
    "outside sample": ("#2b2f36", "#f1f3f5"),
    "formula": ("#5a3d1a", "#ffe8cc"),
    "selected series": ("#1c3a52", "#e7f5ff"),
}
SHEET_BANNER = ("#fff3bf", "#5c4b00")  # (background, text): the spreadsheet's notice, the same on either theme
CHART = {  # marks drawn on matplotlib figures (their own white background, not the app palette)
    "tip face": "#ffffe8", "tip edge": "#888888", "box edge": "#888888", "picked point": "#d9480f", "edited point": "#1c7ed6",
    "arrow": "#1c7ed6", "zero line": "#7f848e",
}
SPLASH_TEXT = "#c8d6e8"  # on the splash screen's own dark picture


def _palette(palette: Any = None) -> Any:
    if palette is not None:
        return palette
    from PySide6.QtGui import QGuiApplication, QPalette

    app = QGuiApplication.instance()
    return app.palette() if app is not None else QPalette()


def is_dark(palette: Any = None) -> bool:
    """True when the palette's window colour is dark (the app follows the system theme)."""
    from PySide6.QtGui import QPalette

    return _palette(palette).color(QPalette.ColorRole.Window).lightness() < 128


def _luminance(c: Any) -> float:
    def channel(v: float) -> float:
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(c.redF()) + 0.7152 * channel(c.greenF()) + 0.0722 * channel(c.blueF())


def contrast(a: Any, b: Any) -> float:
    """The WCAG contrast ratio of two colours (QColor or anything QColor takes), 1 to 21."""
    from PySide6.QtGui import QColor

    la, lb = sorted((_luminance(QColor(a)), _luminance(QColor(b))), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _over(fg: Any, bg: Any) -> Any:
    """``fg`` painted over ``bg``: a translucent colour (Qt's placeholder text is half-opaque) as it is seen."""
    from PySide6.QtGui import QColor

    a = fg.alphaF()
    return QColor(QColor.fromRgbF(*(a * f + (1 - a) * g for f, g in ((fg.redF(), bg.redF()), (fg.greenF(), bg.greenF()), (fg.blueF(), bg.blueF())))).name())


def _backgrounds(palette: Any) -> list:
    from PySide6.QtGui import QPalette

    return [palette.color(QPalette.ColorGroup.Active, role) for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base)]


def readable(fg: Any, backgrounds: list, minimum: float = MIN_CONTRAST) -> Any:
    """``fg``, or the nearest shade of it (darker on a light window, lighter on a dark one: the first background) that
    reads at ``minimum`` against each of ``backgrounds`` (any that even black or white could not reach is skipped)."""
    from PySide6.QtGui import QColor

    colour = QColor(fg)
    lighten = _luminance(QColor(backgrounds[0])) < 0.18
    extreme = QColor("#ffffff" if lighten else "#000000")
    reachable = [b for b in backgrounds if contrast(extreme, b) >= minimum] or backgrounds[:1]
    for _ in range(60):
        if min(contrast(colour, b) for b in reachable) >= minimum:
            break
        colour = colour.lighter(106) if lighten else colour.darker(106)
        if lighten and colour.lightness() >= 250:
            colour = QColor(extreme)
    return QColor(colour.name())  # whole 8-bit channels: the colour a style sheet or a pixel has


def colour(role: str, palette: Any = None) -> Any:
    """The QColor for ``role`` ("error", "warning", "success", "link" or "muted") on the given or the app's palette."""
    pal = _palette(palette)
    if role == "muted":
        return muted(pal)
    light, dark = ROLES[role]
    return readable(dark if is_dark(pal) else light, _backgrounds(pal))


def name(role: str, palette: Any = None) -> str:
    """``role``'s colour as "#rrggbb"."""
    return colour(role, palette).name()


def css(role: str, palette: Any = None) -> str:
    """A style sheet that sets the text colour to ``role``'s: "color: #rrggbb;"."""
    return f"color: {name(role, palette)};"


def muted(palette: Any = None) -> Any:
    """Secondary text: the palette's placeholder text as it is seen on the window, or, where that reads at less than
    4.5:1, the text colour mixed into the window colour only as far as 4.5:1 allows (grey on a light theme, light grey
    on a dark one)."""
    from PySide6.QtGui import QColor, QPalette

    pal = _palette(palette)
    window = pal.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Window)
    text = pal.color(QPalette.ColorGroup.Active, QPalette.ColorRole.WindowText)
    placeholder = _over(pal.color(QPalette.ColorGroup.Active, QPalette.ColorRole.PlaceholderText), window)
    backgrounds = _backgrounds(pal)
    if min(contrast(placeholder, b) for b in backgrounds) >= MIN_CONTRAST:
        return placeholder
    best = QColor(text)
    for step in range(1, 50):  # from the text colour towards the window colour, 2% at a time, while still readable
        t = step / 50
        mixed = QColor.fromRgbF(*((1 - t) * a + t * b for a, b in ((text.redF(), window.redF()), (text.greenF(), window.greenF()), (text.blueF(), window.blueF()))))
        if min(contrast(mixed, b) for b in backgrounds) < MIN_CONTRAST + 0.1:  # a little margin over the line
            break
        best = mixed
    return QColor(best.name())


def muted_css(palette: Any = None) -> str:
    """A style sheet for secondary text: "color: #rrggbb;" with :func:`muted`'s colour."""
    return f"color: {muted(palette).name()};"
