"""The desktop app's spacing, in one place: a few named sizes on a 4-pixel grid, scaled with the text size.

Margins and spacing in ``pymetrix.ui`` take a token here rather than a number written where they are used, so a
dialog's margins, the space above a section heading or a group box, and the indent of a sub-form are the same in
every window. The sizes are at Normal text (View → Text size); at another text size they scale with it
(``px``), and margins and spacing set through ``set_margins``, ``set_spacing`` and ``space_above`` follow a text size
chosen while the window is open (``reapply``). Qt's own spacing between the widgets of a layout (the style's) is
left as the style has it: tokens are for what pymetrix sets itself.

Deliberate exceptions keep their number with a comment saying why where they are set (the result window's seven
buttons at 3 pixels apart, a list entry's padding); ``tests/test_ui_spacing.py`` lists them.
"""

from __future__ import annotations

from typing import Any

__all__ = ["GRID", "TOKENS", "add_space", "px", "reapply", "section_heading", "set_margins", "set_spacing", "space_above", "spaced"]

GRID = 4
TOKENS = {
    "none": 0,
    "hairline": 2,  # half a step: a field and its own message line; a compact bar's top and bottom; rows of a table of figures
    "tight": 4,  # between the parts of one control (a toolbar's buttons, a bar's ends)
    "normal": 8,  # between related items; around a page inside the workspace (the Start window)
    "section": 12,  # above a section heading or a group box; between groups
    "indent": 12,  # a sub-form under the choice that opens it
    "dialog_margin": 12,  # around a dialog's or a page's content (Qt's own default is 11, off the grid)
}
_MARGINS, _SPACING = "pymetrixMargins", "pymetrixSpacing"  # dynamic properties: the tokens a layout or widget was given


def px(token: str) -> int:
    """``token``'s size in pixels at the text size applied now (at Normal, exactly the token's)."""
    from .text_size import factor

    return int(round(TOKENS[token] * factor()))


def _tokens(left: str, top: str | None, right: str | None, bottom: str | None) -> tuple[str, str, str, str]:
    top = left if top is None else top
    return left, top, left if right is None else right, top if bottom is None else bottom


def set_margins(target: Any, left: str, top: str | None = None, right: str | None = None, bottom: str | None = None) -> None:
    """A layout's (or a widget's) contents margins from tokens, CSS-like: one token for all four sides; two for
    across and down; four for left, top, right and bottom."""
    names = _tokens(left, top, right, bottom)
    target.setProperty(_MARGINS, ",".join(names))
    target.setContentsMargins(*(px(n) for n in names))


def set_spacing(layout: Any, token: str, vertical: str | None = None) -> None:
    """A layout's (or an ``ActionRow``'s) spacing from a token; for a grid or form layout, ``vertical`` gives the
    spacing down apart from the spacing across."""
    layout.setProperty(_SPACING, token if vertical is None else f"{token},{vertical}")
    _spacing(layout, token, vertical)


def _spacing(layout: Any, token: str, vertical: str | None) -> None:
    if vertical is None:
        layout.setSpacing(px(token))
    else:
        layout.setHorizontalSpacing(px(token))
        layout.setVerticalSpacing(px(vertical))


def space_above(widget: Any, token: str = "section") -> Any:
    """``widget`` (a section heading, a section's toggle) with ``token``'s room above it: the same above every
    heading, wherever it is. Returns the widget."""
    set_margins(widget, "none", token, "none", "none")
    return widget


def spaced(widget: Any, token: str = "section") -> Any:
    """``widget`` in a holder with ``token``'s room above it: for a button (a section's fold toggle), whose own size
    hint leaves out contents margins. Returns the holder, to add where the widget went."""
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    holder = QWidget()
    layout = QVBoxLayout(holder)
    set_margins(layout, "none", token, "none", "none")
    layout.addWidget(widget)
    return holder


def section_heading(text: str, first: bool = False) -> Any:
    """A section's heading: its words in bold, with a section's room above it (none for the first thing on a page)."""
    from .fit import WrapLabel

    label = WrapLabel(f"<b>{text}</b>")
    label.setObjectName("sectionHeading")
    if not first:
        space_above(label)
    return label


def add_space(layout: Any, token: str = "section") -> None:
    """``token``'s room in a box layout before what is added next (a group box after other content)."""
    layout.addSpacing(px(token))


def reapply(app: Any) -> None:
    """The margins and spacing set through this module again, at the text size applied now (``text_size.apply``)."""
    from PySide6.QtWidgets import QLayout

    for widget in app.allWidgets():
        try:
            targets = [widget]
            layout = widget.layout()
            if layout is not None:
                targets += [layout, *layout.findChildren(QLayout)]
            for target in targets:
                margins = target.property(_MARGINS)
                if margins:
                    target.setContentsMargins(*(px(n) for n in str(margins).split(",")))
                spacing = target.property(_SPACING)
                if spacing and hasattr(target, "setSpacing"):  # a layout, or a widget that passes it on (ActionRow)
                    parts = str(spacing).split(",")
                    _spacing(target, parts[0], parts[1] if len(parts) > 1 else None)
        except RuntimeError:  # deleted while being listed
            continue
