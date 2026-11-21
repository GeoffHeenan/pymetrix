"""What a project may tell an AI assistant, by confidentiality mode: one tested place, so every assistant sends the
same (``docs/design/assistant-plugin.md``, "What core needs", item 4).

    from pymetrix.disclosure import describe_session, to_text

    facts = describe_session(session)          # metadata only: names, kinds, sizes, dates, settings
    prompt_part = to_text(facts)

The modes (``MODES``), a project's choice:

- ``off`` (the default): nothing is sent; an assistant panel asks which mode to use the first time.
- ``metadata_only``: only what ``describe_session`` gives.
- ``full_context``: more may be sent (the steps' code; an object's tables when the assistant asks to see them), but
  the assistant panel shows each send and asks first.
- ``local_model``: sent only to a model on this computer.

``describe_session`` and ``describe_object`` are built from an allow-list, never by taking things out: what they copy
is an object's name and kind, counts and sizes, period labels, and settings that are plain identifiers, whole numbers
or true/false. Data values, numbers with decimals (estimates, calibrations), formulas and equations, file paths,
free text (labels, descriptions, notes, report text, help) and the code of steps never appear: the code of a step is
replaced by the name of its dialog and the names it made. A series or variable name is copied only when it is an
identifier, so text arriving as a name cannot carry a sentence into the prompt.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .errors import PyMetrixError

__all__ = ["DEFAULT_MODE", "MODES", "describe_object", "describe_session", "to_text"]

MODES = ("off", "metadata_only", "full_context", "local_model")
DEFAULT_MODE = "off"
MAX_NAMES = 200  # series or variables listed per object, at most
MAX_STEPS = 50  # the latest steps described

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,63}$")


def _identifier(value: Any) -> str | None:
    """``value`` if it is a plain identifier (``gdp``, ``cov_type``, ``HC1``): no spaces, operators, paths or words."""
    return value if isinstance(value, str) and _IDENT.match(value) else None


def _names(values: Any) -> list[str]:
    out = []
    for v in values:
        name = _identifier(str(v)) if isinstance(v, str) else None
        if name is not None:
            out.append(name)
        if len(out) >= MAX_NAMES:
            break
    return out


def _setting(value: Any) -> Any:
    """A setting kept: true/false, a whole number, an identifier, or a list of those; anything else is left out."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return _identifier(value)
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        kept = _names(value)
        return kept if len(kept) == len(value) and kept else None
    return None


def _settings(mapping: Any) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        try:
            mapping = dict(mapping or {})
        except (TypeError, ValueError):
            return {}
    out = {}
    for key, value in mapping.items():
        name = _identifier(str(key))
        kept = _setting(value)
        if name is not None and kept is not None:
            out[name] = kept
    return out


def _period(value: Any) -> str | None:
    """A period or date label (``2020Q1``, ``2021-03``, ``1999``): kept; anything else left out."""
    text = str(value)
    return text if re.fullmatch(r"\d{4}([-Q/MW]\d{1,2}){0,2}|\d{4}-\d{2}-\d{2}", text) else None


def describe_object(name: str, obj: Any) -> dict[str, Any]:
    """An object in metadata only: its kind and what is safe to say about its shape (see the module's rules)."""
    from .data import Dataset

    kind = type(obj).__name__
    out: dict[str, Any] = {"name": name, "kind": kind}
    if isinstance(obj, Dataset):
        frame = obj.frame
        index = frame.index
        out.update(series=_names(obj.names), n_series=len(obj.names), n_rows=len(frame))
        freq = getattr(index, "freqstr", None)
        if freq:
            out["frequency"] = re.sub(r"[^A-Z]", "", str(freq).split("-")[0])[:3]
        if len(index):
            first, last = _period(index[0]), _period(index[-1])
            if first and last:
                out["span"] = [first, last]
        return out
    provenance = getattr(obj, "provenance", None)
    if provenance is not None:
        sample = getattr(provenance, "sample", None)
        if sample:
            kept = [_period(p) for p in sample]
            if all(kept):
                out["sample"] = kept
        nobs = getattr(provenance, "nobs", None)
        if isinstance(nobs, int):
            out["nobs"] = nobs
        out["method"] = _settings(getattr(provenance, "method", None))
        out["spec"] = _settings(getattr(provenance, "spec", None))
        out["data"] = _names(dict(getattr(provenance, "data_vintages", None) or {}))
        converged = getattr(provenance, "converged", None)
        if isinstance(converged, bool):
            out["converged"] = converged
        out["n_warnings"] = len(getattr(provenance, "warnings", ()) or ())
    panels = getattr(obj, "panels", None)
    if kind == "Figure" and panels is not None:
        out["n_panels"] = len(panels)
        out["series"] = _names(getattr(layer, "label", "") for p in panels for layer in p.layers)
    names = getattr(obj, "names", None)
    if "series" not in out and isinstance(names, (list, tuple)):
        out["names"] = _names(names)
    return {k: v for k, v in out.items() if v not in ({}, [], None)}


def _step(record: Any, full: bool = False) -> dict[str, Any]:
    command = getattr(record, "command", None)
    label = getattr(type(command), "label", None) if command is not None else None
    what = type(command).__name__ if command is not None else "code"
    if getattr(record, "code_step", False):
        what = "typed code"  # its code is not sent: it may hold anything
    step: dict[str, Any] = {"step": what if _identifier(what) or what == "typed code" else "step"}
    if isinstance(label, str) and _identifier(label.replace(" ", "_")):
        step["dialog"] = label
    made = _names(getattr(record, "writes", ()) or ())
    if made:
        step["made"] = made
    step["ok"] = bool(getattr(record, "ok", False))
    if full:  # full_context: the step's code too (formulas, paths, typed code), shown to the user before it is sent
        step["code"] = str(getattr(record, "code", ""))
    return step


def describe_session(session: Any, mode: str = "metadata_only") -> dict[str, Any]:
    """The project as ``mode`` allows: in ``metadata_only`` (and ``local_model``), each object (``describe_object``) and
    the latest steps, by dialog and what they made, never their code. In ``full_context``, the steps' code as well:
    the assistant panel shows it to the user, who sends it or not, before each question."""
    if mode not in ("metadata_only", "local_model", "full_context"):
        raise PyMetrixError(f"nothing is described in mode {mode!r}; the modes that send are metadata_only, local_model and full_context")
    full = mode == "full_context"
    objects = [describe_object(n, o) for n, o in session.objects.items() if _identifier(n)]
    steps = [_step(r, full) for r in list(getattr(session, "history", []))[-MAX_STEPS:]]
    return {"mode": "full_context" if full else "metadata_only", "objects": objects, "steps": steps}


def to_text(facts: dict[str, Any]) -> str:
    """The description as compact JSON, for a prompt."""
    return json.dumps(facts, ensure_ascii=False, separators=(",", ":"), default=str)
