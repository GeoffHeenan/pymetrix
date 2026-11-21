"""A command dialog's checks before OK: what holds OK, what it says, and what a picker with nothing to pick offers.

- ``OkGate``: OK is enabled only when nothing holds it. Each check (required fields, start after end, a formula
  checked as it is typed, a preview) holds or releases OK under its own key, so one check's all-clear never
  re-enables OK while another still holds it; the reasons show under the form and in OK's tooltip.
- ``EmptyPicker``: an object field with no object of its kinds in the session says so in words ("No estimated
  models yet: estimate an equation or a VAR first") with a button to the dialog that makes one, instead of a dropdown
  naming internal classes ("(no FittedEquation,FittedVAR,… available)").
- ``error_box``: a failed step's message, titled after the dialog, in plain words with a next step; the technical
  text under Details (``pymetrix.ui.messages``).
"""

from __future__ import annotations

from typing import Any

from PySide6 import QtCore, QtWidgets

from . import theme

__all__ = ["EmptyPicker", "OkGate", "error_box", "makers_for", "most_recent_model", "picker_words"]


class OkGate:
    """Holds OK while any check has a reason to: ``hold(key, reason)`` and ``release(key)``; ``proxy(key)`` gives an
    object with ``setEnabled`` and ``setToolTip``, for checks written against the OK button itself."""

    def __init__(self, ok: Any, note: Any = None) -> None:
        self.ok, self.note = ok, note
        self.reasons: dict[str, str] = {}

    def hold(self, key: str, reason: str) -> None:
        self.reasons[key] = reason or "not ready"
        self._show()

    def release(self, key: str) -> None:
        self.reasons.pop(key, None)
        self._show()

    def set(self, key: str, reason: str | None) -> None:
        self.hold(key, reason) if reason else self.release(key)

    def held(self) -> bool:
        return bool(self.reasons)

    def reason(self) -> str:
        """The first reason OK is held ("" if it is not)."""
        return next(iter(self.reasons.values()), "")

    def _show(self) -> None:
        try:
            self.ok.setEnabled(not self.reasons)
            self.ok.setToolTip(self.reason())
            if self.note is not None:
                shown = [r for k, r in self.reasons.items() if k == "required"]  # the others show their own line
                self.note.setText(shown[0] if shown else "")
                self.note.setVisible(bool(shown))
        except RuntimeError:  # the dialog is gone
            pass

    def proxy(self, key: str) -> Any:
        gate = self

        class _Proxy:
            """OK as one check sees it: enabling it releases only that check's hold."""

            tip = ""

            def setEnabled(self, on: bool) -> None:  # noqa: N802 - QWidget's name
                gate.set(key, None if on else (self.tip or "not ready"))

            def setToolTip(self, text: str) -> None:  # noqa: N802
                self.tip = text
                if key in gate.reasons and text:
                    gate.hold(key, text)

            def isEnabled(self) -> bool:  # noqa: N802
                return key not in gate.reasons

        return _Proxy()


# --- pickers with nothing to pick ------------------------------------------------------------------------------------

_MODEL_KINDS = ("Template", "DynareModel", "Model", "EViewsXmlModel")  # whole models (not single equations)
_MAKERS: dict[str, tuple[str, ...]] = {  # a kind of object -> the dialogs that make one, best first
    "Dataset": ("LoadExample", "LoadData"),
    "FittedEquation": ("FitEquation",),
    "Equation": ("DefineEquation",),
    "FittedVAR": ("FitVAR",),
    "FittedVECM": ("FitVECM",),
    "FittedBVARSV": ("FitVAR",),
    "FittedMSVAR": ("FitVAR",),
    "FittedQuantileVAR": ("FitVAR",),
    "FittedStateSpace": ("FitStateSpace",),
    "FittedSARIMA": ("FitSARIMA",),
    "Template": ("LoadTemplate",),
    "DynareModel": ("LoadTemplate",),
    "Model": ("DefineEquation",),
    "FittedModel": ("EstimateModel",),
    "ForecastRound": ("MakeRound",),
    "LinearSolution": ("SolveRationalExpectations",),
    "Solution": ("SolveModel",),
}
_PLURALS = {"Dataset": "datasets", "Template": "template models", "DynareModel": "Dynare models", "Model": "models",
            "Equation": "equations", "FittedEquation": "estimated equations", "ForecastRound": "forecast rounds",
            "Selection": "selections", "Backtest": "backtests", "Solution": "model solutions",
            "LinearSolution": "rational-expectations solutions", "FittedModel": "estimated models",
            "EViewsXmlModel": "EViews models", "StateSpaceSystem": "state-space systems"}


def _kinds(source: str) -> list[str]:
    return [k.strip() for k in (source or "").split(",") if k.strip()]


def picker_words(source: str) -> str:
    """The kinds a picker takes, in words: "estimated models", "datasets", "equations or models"."""
    kinds = _kinds(source)
    if kinds and all(k.startswith("Fitted") or k == "ProxySVAR" for k in kinds):
        if all("VAR" in k or "VECM" in k for k in kinds):
            return "estimated VARs"
        return "estimated models" if len(kinds) > 1 else f"estimated {_words(kinds[0][len('Fitted'):])}s"
    if {"FittedEquation", "Equation"} <= set(kinds):  # estimated or defined: equations either way
        kinds = ["Equation" if k == "FittedEquation" else k for k in kinds]
    words = list(dict.fromkeys(_PLURALS.get(k) or (_words(k) + "s") for k in kinds if k not in ("dict",)))
    if not words:
        return "objects"
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " or " + words[-1]


def _words(name: str) -> str:
    import re

    parts = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name).split()
    return " ".join(p if p.isupper() else p.lower() for p in parts) or name


def makers_for(source: str) -> list[Any]:
    """The registered commands that make an object of one of the kinds, best first (no repeats)."""
    from ..commands import COMMANDS

    by_name = {i.cls.__name__: i for i in COMMANDS.values() if i.menu}
    out: list[Any] = []
    for kind in _kinds(source):
        for name in _MAKERS.get(kind, ()):
            info = by_name.get(name)
            if info is not None and info not in out:
                out.append(info)
    return out


def _menu_path(info: Any) -> str:
    from ..help import menu_path  # with its submenu ("Model → Solution → Rational expectations…")

    return menu_path(info.cls) or info.menu.replace("/", " → ").replace("...", "…")


def most_recent_model(session: Any, source: str) -> str:
    """The newest whole model (template, Dynare or a model of equations) the picker takes, or ""."""
    wanted = [k for k in _kinds(source) if k in _MODEL_KINDS]
    if not wanted:
        return ""
    names = [n for n, o in session.objects.items() if set(wanted) & {c.__name__ for c in type(o).__mro__}]
    return names[-1] if names else ""


def _elsewhere_hint(session: Any, source: str) -> tuple[str, Any, str] | None:
    """When the session has a template model a picker of equations cannot take: what to use instead."""
    kinds = set(_kinds(source))
    if kinds & {"Template", "DynareModel"} or not kinds & {"Equation", "Model", "EViewsXmlModel"}:
        return None  # it takes templates, or it is not a picker of equations and models
    templates = [n for n, o in session.objects.items() if type(o).__name__ in ("Template", "DynareModel")]
    if not templates:
        return None
    from ..commands import COMMANDS

    solver = next((i for i in COMMANDS.values() if i.cls.__name__ == "SolveRationalExpectations" and i.menu), None)
    if solver is None:
        return None
    return (f"{templates[-1]} is a template model: solve it with {_menu_path(solver)}", solver, templates[-1])


class EmptyPicker:
    """The line under an object field when the session has nothing it can pick: what is missing, in words, and a
    button to the dialog that makes one (``open_dialog(cls, initial)``, when given)."""

    def __init__(self, spec: Any, session: Any, open_dialog: Any = None) -> None:
        self.spec, self.session, self.open_dialog = spec, session, open_dialog
        self.widget = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.widget)
        row.setContentsMargins(0, 0, 0, 0)
        self.label = QtWidgets.QLabel()
        self.label.setWordWrap(True)
        self.label.setStyleSheet(theme.muted_css())
        self.button = QtWidgets.QPushButton()
        self.button.setAutoDefault(False)
        self.button.clicked.connect(self._go)
        row.addWidget(self.label, 1)
        row.addWidget(self.button, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        self.widget.setVisible(False)
        self.target: Any = None
        self.initial: Any = None

    def update(self, options: list[str]) -> bool:
        """Show the line when there is nothing to pick; True when it shows."""
        empty = not [o for o in options if o] and not self.spec.empty  # (blank has a meaning there: nothing missing)
        self.widget.setVisible(empty)
        if not empty:
            return False
        words = picker_words(self.spec.source or "")
        makers = makers_for(self.spec.source or "")
        hint = _elsewhere_hint(self.session, self.spec.source or "")
        self.target, self.initial = None, None
        if hint is not None:
            text, solver, template = hint
            self.label.setText(f"No {words} yet. {text}" + ("" if text.endswith("…") else "."))
            self.target = solver.cls
            self.initial = template
            self.button.setText(_menu_path(solver).rpartition(" → ")[2])
        elif makers:
            first = makers[0]
            how = " or ".join(dict.fromkeys(_menu_path(m) for m in makers[:2]))
            self.label.setText(f"No {words} yet: make one first ({how}).")
            self.target = first.cls
            self.button.setText(_menu_path(first).rpartition(" → ")[2])
        else:
            self.label.setText(f"No {words} yet: make one first, then come back to this dialog.")
        self.button.setVisible(self.target is not None and self.open_dialog is not None)
        self.button.setToolTip("close this dialog and open the one that makes it")
        return True

    def _go(self) -> None:
        if self.target is not None and self.open_dialog is not None:
            self.open_dialog(self.target, self.initial)


# --- a failed step ---------------------------------------------------------------------------------------------------

def error_box(parent: Any, title: str, exc: BaseException, values: Any = None, show: bool = True) -> Any:
    """The message box for a step that failed: titled after the dialog ("Forecast: could not run"; on a Mac, whose
    message boxes have no title, that line leads the text), the plain message and what to try, and the technical text under Details. Shown over the dialog without blocking (``open``), so the
    dialog stays as it was to correct; ``show`` False builds it without showing it."""
    from .messages import explain, failure_title, titled

    found = explain(exc, values or {})
    box = QtWidgets.QMessageBox(parent)
    box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    heading, text = titled(failure_title(title), found.text)  # on a Mac, which shows no title, the text says it
    box.setWindowTitle(heading)
    box.setText(text)
    if found.hint:
        box.setInformativeText(found.hint)
    box.setDetailedText(found.details)
    box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
    box.explained = found
    if show:
        box.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        box.open()
    return box
