"""Formulas checked as they are typed: a line under each formula field of a command dialog, and OK held while it has
an error.

``check_formula`` (no Qt) parses the text with pymetrix's expression parser, as the step will, and looks up its names
in the datasets the dialog has chosen. Nothing is evaluated: a parse and a look-up, cheap at every key. It gives

- an error (red, OK held) with the parser's reason and where: "Expected ')' at character 6, after “log(y”";
- a hint (grey, OK free) for names the data do not have, with the nearest one: "Not in data: gpd (did you mean gdp?)".
  Only a hint: a model's equations, an identity's new series or a name made by the same step may be meant;
- nothing when the formula is fine (or empty: the dialog's * check says so).

``live_formulas`` wires it into ``command_dialog``: each formula field (kind "formula", the derived-series templates
and the "Rows where" conditions) is checked 300 ms after typing stops, and whenever another field changes (another
dataset), under its own key on the dialog's ``OkGate``.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any

from . import theme

__all__ = ["FormulaCheck", "check_formula", "formula_fields", "live_formulas"]

DELAY_MS = 300
_EXPRESSIONS = {("GenerateSeries", "formula"), ("DeriveSeries", "formula"), ("DeriveEach", "formula")}  # not equations
_PLACEHOLDER = re.compile(r"\{[^{}]*\}")  # {s}, {x}, {1}: a template's stand-ins for each series
_MARK = "Ξ"  # what a stand-in becomes while parsing (as long as it, so the parser's positions still point at the text)
_POSITION = re.compile(r" at position (\d+) in '.*'$", re.S)
_KNOWN = re.compile(r" \(known: [^)]*\)")


@dataclass(frozen=True)
class FormulaCheck:
    """``level``: "ok", "hint" or "error"; ``message`` in words; ``position``: the character (from 0) the error is
    at, when the parser said; ``unknown``: names the data do not have."""

    level: str = "ok"
    message: str = ""
    position: int | None = None
    unknown: tuple[str, ...] = ()

    @property
    def error(self) -> bool:
        return self.level == "error"


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _where(text: str, at: int) -> str:
    """Where the parser stopped, in the reader's terms: "at character 6, after “log(y”"."""
    before = text[:at].rstrip()
    if not before:
        return "at the start"
    shown = before if len(before) <= 16 else "…" + before[-15:]
    return f"at character {at + 1}, after “{shown}”"


def _templated(text: str) -> tuple[str, list[tuple[str, str]]]:
    """``text`` with each template stand-in ({s}, {x}, {1}) replaced by a name of the same length, and the pairs to
    put them back in messages."""
    pairs: list[tuple[str, str]] = []

    def swap(m: re.Match[str]) -> str:
        original = m.group(0)
        mark = _MARK + "_" * (len(original) - 1)
        pairs.append((mark, original))
        return mark

    return _PLACEHOLDER.sub(swap, text), pairs


def _names_in(value: Any) -> list[str]:
    return [p for p in re.split(r"[\s,;]+", str(value or "")) if p]


def check_formula(text: str, mode: str = "auto", names: Any = None, params: Any = (), equation_class: bool = False) -> FormulaCheck:
    """Check ``text`` as the step will read it. ``mode``: "equation" (``y ~ x``, ``y = x``), "expression" or "auto"
    (an equation when it has ``~`` or ``=``). ``names``: the series (and axes) the data have, or None not to look names
    up; ``params``: the parameters' names (not series); ``equation_class``: also build ``pymetrix.Equation`` from it,
    which checks what the parse alone does not (a parameter named but not used, parameters in an identity)."""
    from ..expr import ExpressionError, parse_equation, parse_expression

    raw = str(text or "")
    if not raw.strip():
        return FormulaCheck()
    parsed_text, pairs = _templated(raw)
    params = [p for p in _names_in(params) if p.isidentifier()]
    if mode == "auto":
        mode = "equation" if re.search(r"~|(?<![<>=!])=(?!=)", parsed_text) else "expression"
    try:
        if mode == "equation":
            lhs, _, rhs = parse_equation(parsed_text, params)
            refs = lhs.variables() | rhs.variables()
            if equation_class:
                from ..estimation.equation import Equation

                Equation(parsed_text, params=params)
        else:
            refs = parse_expression(parsed_text, params).variables()
    except (ExpressionError, ValueError) as exc:
        message = _KNOWN.sub("", str(exc))
        found = _POSITION.search(message)
        position = int(found.group(1)) if found else None
        if found:
            message = message[:found.start()] + " " + _where(raw, position)
        message = message.replace(repr(parsed_text), "the formula")
        for mark, original in pairs:
            message = message.replace(mark, original)
        return FormulaCheck("error", _sentence(message.strip()), position)
    if names is None:
        return FormulaCheck()
    known = {str(n) for n in names}
    unknown = sorted(n for n in refs if n not in known and _MARK not in n and n not in params)
    if not unknown:
        return FormulaCheck()
    parts = []
    for name in unknown[:4]:
        near = difflib.get_close_matches(name, sorted(known), n=1, cutoff=0.6)
        parts.append(f"{name} (did you mean {near[0]}?)" if near else name)
    more = f" and {len(unknown) - 4} more" if len(unknown) > 4 else ""
    return FormulaCheck("hint", f"Not in the data: {', '.join(parts)}{more}.", None, tuple(unknown))


def formula_fields(command_cls: Any, specs: Any) -> dict[str, str]:
    """The fields of a command's form checked as typed, and how each is read: "equation", "expression" or "auto"
    (a plugin's formula: either)."""
    ours = (getattr(command_cls, "__module__", "") or "").startswith("pymetrix.")
    out: dict[str, str] = {}
    for spec in specs:
        if (command_cls.__name__, spec.name) in _EXPRESSIONS:
            out[spec.name] = "expression"
        elif spec.kind == "formula":
            out[spec.name] = "equation" if ours else "auto"
        elif spec.kind == "text" and spec.name == "where":  # "Rows where": a condition, e.g. recession == 0
            out[spec.name] = "expression"
    return out


def data_names(session: Any, specs: Any, values: dict[str, Any]) -> list[str] | None:
    """The names a formula can use from the datasets the dialog has chosen (their series and axes), or None when it
    names no dataset (a model's equation): then names are not looked up."""
    found: list[str] | None = None
    for spec in specs:
        if spec.kind != "object" or "Dataset" not in (spec.source or "").split(","):
            continue
        data = session.objects.get(str(values.get(spec.name) or ""))
        frame = getattr(data, "frame", None)
        if frame is None:
            continue
        found = (found or []) + [str(c) for c in frame.columns] + [str(n) for n in frame.index.names if n is not None]
        found += list(getattr(data, "calendar_names", lambda: [])())  # C(quarter), C(month): from the time axis
    return found


def live_formulas(dialog: Any, command_cls: Any, editors: dict[str, Any], specs: Any, gate: Any, session: Any,
                  values: Any) -> None:
    """Check each formula field of ``dialog`` as it is typed (``DELAY_MS`` after the last key), its line under the field
    and its hold on ``gate`` (``OkGate``) while it has an error. ``dialog.check_formulas()`` checks now (tests);
    ``dialog.formula_checks`` the last result of each field."""
    from PySide6.QtCore import QTimer

    fields = {name: mode for name, mode in formula_fields(command_cls, specs).items() if name in editors}
    dialog.formula_checks = {}
    if not fields:
        dialog.check_formulas = lambda: None
        return
    by_name = {s.name: s for s in specs}
    ours = (getattr(command_cls, "__module__", "") or "").startswith("pymetrix.")

    def update() -> None:
        try:
            current = values()
        except RuntimeError:  # the dialog is gone
            return
        names = data_names(session, specs, current)
        for name, mode in fields.items():
            editor, spec = editors[name], by_name[name]
            line = getattr(editor, "message", None)
            shown = spec.visible(current)
            found = (check_formula(str(current.get(name) or ""), mode, names, current.get("params", ()),
                                   equation_class=ours and mode == "equation")
                     if shown else FormulaCheck())
            dialog.formula_checks[name] = found
            label = spec.label + (" (in Advanced settings)" if spec.advanced else "")
            gate.set(f"formula:{name}", f"{label}: {found.message}" if found.error else None)
            if line is None:
                continue
            line.setText(found.message)
            line.setStyleSheet(theme.css("error") if found.error else theme.muted_css())
            line.setVisible(bool(found.message))
            line.setAccessibleDescription(found.message)
            editor.widget.setProperty("invalid", found.error)

    timer = QTimer(dialog)
    timer.setSingleShot(True)
    timer.setInterval(DELAY_MS)
    timer.timeout.connect(update)
    for editor in editors.values():
        editor.connect(timer.start)
    dialog.formula_timer, dialog.check_formulas = timer, update
    update()
