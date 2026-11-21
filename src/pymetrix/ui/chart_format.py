"""The Format… form for what a figure's recorded steps added: an arrow or note (``annotate``), a text box (``text``),
a line at a date or value, a band, a change marker. Opened from the figure's right-click menu or by a double-click on
the item; each change is the step's new keyword arguments, run as a Change a style step (recorded, undone and redone
like any step, replayed in reports and in the figure's matplotlib code).

``fields(call)`` lists the form's fields for a recorded step, filled in with its current values; ``props(call, values)``
turns what the form holds back into the step's changed arguments. Both are plain Python, tested without Qt;
``format_dialog`` is the Qt form over them. Under the form, **Show the code** (for experts) gives the change as
matplotlib's keyword arguments, following the form as it changes; edited there, the keywords typed are what OK runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..errors import SpecificationError
from . import theme

ARROW_STYLES = {"->": "open head", "-|>": "filled head", "<->": "open heads, both ends", "<|-|>": "filled heads, both ends",
                "-": "no head (a line)", "fancy": "fancy", "simple": "simple", "wedge": "wedge"}
BOX_STYLES = {"none": "no box", "round": "rounded", "square": "square", "round4": "very rounded", "sawtooth": "sawtooth"}
LINE_STYLES = {"solid": "solid", "dashed": "dashed", "dashdot": "dash-dot", "dotted": "dotted"}
CHANGE_STYLES = {"arrow": "double-headed arrow", "bracket": "bracket"}
CHANGE_SHOWS = {"both": "change and percent change", "change": "change", "percent": "percent change"}


@dataclass(frozen=True)
class Field:
    """One field of the form: the step's value ``value``; ``kind`` "text", "number", "colour" or "choice"."""

    key: str
    label: str
    kind: str
    value: Any
    choices: dict[str, str] | None = None
    help: str = ""


def _curvature(arrow: dict[str, Any]) -> float:
    found = re.search(r"rad\s*=\s*(-?[\d.]+)", str(arrow.get("connectionstyle", "")))
    return float(found.group(1)) if found else 0.0


def _box_style(bbox: dict[str, Any] | None) -> str:
    if not bbox:
        return "none"
    return str(bbox.get("boxstyle", "square")).split(",")[0].strip() or "square"


def fields(call: Any) -> list[Field]:
    """The form's fields for a recorded step, filled in with how it is drawn now."""
    p, kind = dict(call.props), call.kind
    if kind == "annotate":
        arrow = dict(p.get("arrowprops") or {})
        out = [Field("text", "Words", "text", p.get("text", ""), help="the words at the tail (empty: an arrow alone)"),
               Field("color", "Colour", "colour", p.get("color", arrow.get("color", "")), help="the words' and the arrow's colour"),
               Field("fontsize", "Text size", "number", p.get("fontsize", ""), help="in points")]
        if p.get("arrowprops") is not None:
            out += [Field("arrowstyle", "Head", "choice", arrow.get("arrowstyle", "->"), ARROW_STYLES),
                    Field("linewidth", "Line width", "number", arrow.get("linewidth", arrow.get("lw", "")), help="in points"),
                    Field("mutation_scale", "Head size", "number", arrow.get("mutation_scale", ""),
                          help="the head's size in points (10 is matplotlib's)"),
                    Field("curvature", "Curve", "number", _curvature(arrow),
                          help="0 for straight; 0.3 bends one way, -0.3 the other")]
        return out
    if kind == "text":
        bbox = dict(p.get("bbox") or {})
        return [Field("s", "Words", "text", p.get("s", "")),
                Field("color", "Colour", "colour", p.get("color", "")),
                Field("fontsize", "Text size", "number", p.get("fontsize", ""), help="in points"),
                Field("box", "Box", "choice", _box_style(p.get("bbox")), BOX_STYLES),
                Field("facecolor", "Box fill", "colour", bbox.get("facecolor", "")),
                Field("edgecolor", "Box edge", "colour", bbox.get("edgecolor", ""))]
    if kind in ("axvline", "axhline"):
        return [Field("color", "Colour", "colour", p.get("color", "")),
                Field("linewidth", "Line width", "number", p.get("linewidth", p.get("lw", "")), help="in points"),
                Field("linestyle", "Line style", "choice", p.get("linestyle", p.get("ls", "solid")) or "solid", LINE_STYLES)]
    if kind in ("axvspan", "axhspan"):
        return [Field("color", "Colour", "colour", p.get("color", p.get("facecolor", ""))),
                Field("alpha", "Opacity", "number", p.get("alpha", ""), help="0 (invisible) to 1")]
    if kind == "change":
        return [Field("color", "Colour", "colour", p.get("color", "")),
                Field("style", "Shape", "choice", p.get("style", "arrow"), CHANGE_STYLES),
                Field("show", "Label", "choice", p.get("show", "both"), CHANGE_SHOWS)]
    return []


def _number(text: Any) -> float | None:
    text = str(text).strip()
    if not text:
        return None
    value = float(text)
    return int(value) if value.is_integer() else value


def props(call: Any, values: dict[str, Any]) -> dict[str, Any]:
    """The step's changed keyword arguments for the form's ``values`` (by field key): only what differs from now.
    A number left empty keeps the step's own value (matplotlib's default if it has none)."""
    p, kind = dict(call.props), call.kind
    out: dict[str, Any] = {}

    def put(key: str, value: Any) -> None:
        if value is not None and value != "" and p.get(key) != value:
            out[key] = value

    if kind == "annotate":
        if "text" in values and values["text"] != p.get("text", ""):
            out["text"] = values["text"]
        put("color", str(values.get("color", "")).strip() or None)
        put("fontsize", _number(values.get("fontsize", "")))
        if p.get("arrowprops") is not None:
            arrow = dict(p["arrowprops"])
            arrow["arrowstyle"] = values.get("arrowstyle", arrow.get("arrowstyle", "->"))
            for key in ("linewidth", "mutation_scale"):
                number = _number(values.get(key, ""))
                if number is not None:
                    arrow[key] = number
                    arrow.pop("lw" if key == "linewidth" else "", None)
            curve = _number(values.get("curvature", "")) or 0
            if curve:
                arrow["connectionstyle"] = f"arc3,rad={curve}"
            else:
                arrow.pop("connectionstyle", None)
            colour = str(values.get("color", "")).strip()
            if colour:
                arrow["color"] = colour
            if arrow != dict(p["arrowprops"]):
                out["arrowprops"] = arrow
        return out
    if kind == "text":
        if "s" in values and values["s"] != p.get("s"):
            out["s"] = values["s"]
        put("color", str(values.get("color", "")).strip() or None)
        put("fontsize", _number(values.get("fontsize", "")))
        box = values.get("box", _box_style(p.get("bbox")))
        old = dict(p.get("bbox") or {})
        if box == "none":
            if p.get("bbox"):
                out["bbox"] = None
        else:
            pad = re.search(r"pad\s*=\s*([\d.]+)", str(old.get("boxstyle", "")))
            bbox = {**old, "boxstyle": box + (f",pad={pad.group(1)}" if pad else "")}
            for key in ("facecolor", "edgecolor"):
                colour = str(values.get(key, "")).strip()
                if colour:
                    bbox[key] = colour
            bbox.setdefault("facecolor", "white")
            bbox.setdefault("edgecolor", theme.CHART["box edge"])
            if bbox != old:
                out["bbox"] = bbox
        return out
    if kind in ("axvline", "axhline"):
        put("color", str(values.get("color", "")).strip() or None)
        put("linewidth", _number(values.get("linewidth", "")))
        if values.get("linestyle") and values["linestyle"] != (p.get("linestyle", p.get("ls", "solid")) or "solid"):
            out["linestyle"] = values["linestyle"]
        return out
    if kind in ("axvspan", "axhspan"):
        put("color", str(values.get("color", "")).strip() or None)
        alpha = _number(values.get("alpha", ""))
        if alpha is not None and not 0 <= alpha <= 1:
            raise SpecificationError("opacity is between 0 (invisible) and 1")
        put("alpha", alpha)
        return out
    if kind == "change":
        put("color", str(values.get("color", "")).strip() or None)
        for key in ("style", "show"):
            if values.get(key) and values[key] != p.get(key, "arrow" if key == "style" else "both"):
                out[key] = values[key]
        return out
    return out


TITLES = {"annotate": "Format the arrow or note", "text": "Format the text box", "axvline": "Format the line",
          "axhline": "Format the line", "axvspan": "Format the band", "axhspan": "Format the band",
          "change": "Format the change marker"}


def format_dialog(parent: Any, call: Any) -> Any:
    """The Format… form over ``call`` (Qt): ``dialog.exec()`` then ``dialog.result_props()``, the step's changed
    arguments (empty when nothing changed). ``dialog.widgets`` holds each field's widget, by key."""
    from PySide6 import QtWidgets as W
    from PySide6.QtGui import QColor

    dialog = W.QDialog(parent)
    dialog.setWindowTitle(TITLES.get(call.kind, "Format"))
    layout = W.QFormLayout(dialog)
    from .metrics import set_margins

    set_margins(layout, "dialog_margin")
    dialog.widgets = {}
    for f in fields(call):
        if f.kind == "choice":
            widget = W.QComboBox()
            for code, words in (f.choices or {}).items():
                widget.addItem(words, code)
            index = widget.findData(f.value)
            if index < 0:  # a value the list does not name: kept, under its own code
                widget.addItem(str(f.value), f.value)
                index = widget.count() - 1
            widget.setCurrentIndex(index)
            row: Any = widget
        elif f.kind == "colour":
            widget = W.QLineEdit(str(f.value or ""))
            widget.setPlaceholderText("as drawn")
            pick = W.QPushButton("Choose…")

            def choose(_: bool = False, edit: Any = widget) -> None:
                start = QColor(edit.text()) if QColor.isValidColorName(edit.text()) else QColor("black")
                colour = W.QColorDialog.getColor(start, dialog, "Choose a colour")
                if colour.isValid():
                    edit.setText(colour.name())

            pick.clicked.connect(choose)
            row = W.QWidget()
            line = W.QHBoxLayout(row)
            line.setContentsMargins(0, 0, 0, 0)
            line.addWidget(widget, 1)
            line.addWidget(pick)
        else:
            widget = W.QLineEdit("" if f.value in (None, "") else str(f.value))
            if f.kind == "number":
                widget.setPlaceholderText("as drawn")
            row = widget
        if f.help:
            widget.setToolTip(f.help)
        dialog.widgets[f.key] = widget
        layout.addRow(f.label, row)
    from PySide6.QtCore import Qt as QtNS

    toggle = W.QToolButton()
    toggle.setText("Show the code")
    toggle.setCheckable(True)
    toggle.setAutoRaise(True)
    toggle.setToolButtonStyle(QtNS.ToolButtonStyle.ToolButtonTextBesideIcon)
    toggle.setArrowType(QtNS.ArrowType.RightArrow)
    toggle.setToolTip("the change as matplotlib's keyword arguments, to read or type (for experts)")
    code = W.QPlainTextEdit()
    code.setMaximumHeight(90)
    code.setStyleSheet("font-family: monospace;")
    code.setPlaceholderText("no change yet")
    code.setVisible(False)
    layout.addRow(toggle)
    layout.addRow(code)
    message = W.QLabel("")
    message.setStyleSheet(theme.css("error", dialog.palette()))
    message.setVisible(False)
    layout.addRow(message)
    written = {"text": ""}  # the code as the form last wrote it: typed over, the typing wins
    buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Ok | W.QDialogButtonBox.StandardButton.Cancel)
    layout.addRow(buttons)
    dialog.found = {}

    def values() -> dict[str, Any]:
        return {k: (w.currentData() if isinstance(w, W.QComboBox) else w.text()) for k, w in dialog.widgets.items()}

    def follow() -> None:
        """The code box follows the form, unless what is in it was typed there."""
        if code.toPlainText() != written["text"]:
            return
        try:
            text = keywords(props(call, values()))
        except ValueError:
            text = written["text"]
        written["text"] = text
        code.setPlainText(text)

    def show(on: bool) -> None:
        toggle.setArrowType(QtNS.ArrowType.DownArrow if on else QtNS.ArrowType.RightArrow)
        toggle.setText("Hide the code" if on else "Show the code")
        code.setVisible(on)
        follow()

    toggle.toggled.connect(show)
    for widget in dialog.widgets.values():
        (widget.currentIndexChanged if isinstance(widget, W.QComboBox) else widget.textChanged).connect(lambda *_: follow())

    def accept() -> None:
        try:
            typed = code.toPlainText().strip()
            if typed != written["text"].strip():  # typed in the code box: those keywords, as Change a style reads them
                from ..forms import parse_options

                dialog.found = parse_options(typed)[1]
            else:
                dialog.found = props(call, values())
        except ValueError as exc:
            message.setText(str(exc) if "number" not in str(exc) else "sizes, widths and the curve are numbers")
            message.setVisible(True)
            return
        dialog.accept()

    buttons.accepted.connect(accept)
    buttons.rejected.connect(dialog.reject)
    dialog.values = values
    dialog.result_props = lambda: dialog.found
    dialog.code_toggle, dialog.code_box = toggle, code
    return dialog


def keywords(changed: dict[str, Any]) -> str:
    """The step's changed arguments as matplotlib's keywords, as Change a style reads them: ``color='red', lw=2``."""
    return ", ".join(f"{k}={v!r}" for k, v in changed.items())
