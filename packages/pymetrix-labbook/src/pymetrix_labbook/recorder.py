"""Writing the lab book: a session observer that turns each session event into a chained entry.

The lab book of a session lives in ``session.attachments["labbook"]["labbook.jsonl"]``, so it is saved and reopened
with its project, and kept even when the project is opened without this plugin. Entries record what an auditor needs,
not the data: the step's code, the names and versions it read, and for what it wrote, the type, a fingerprint and the
number of observations.

What is recorded:

- ``start``: the lab book began; ``existing_steps`` counts the history it found (work done before it was installed);
- ``step`` and ``failed``: each step asked for, and each recalculation (``during``); steps replayed while opening a
  project are not recorded again, since they were recorded when first run;
- ``cancelled``, ``undone``, ``removed``: work that leaves no trace in the history;
- ``opened`` and ``saved``: the project file, with the history's length and fingerprint;
- ``brief``, ``brief_skipped`` and ``note``: written by ``pymetrix_labbook.brief`` (the project brief, its
  amendments, and notes), and ``plan`` by ``pymetrix_labbook.plan``, not by session events.

Each entry says who acted (``by``): the person at the machine, or an AI agent working through pymetrix's MCP server.
"""

from __future__ import annotations

import hashlib
import weakref
from typing import Any

from .chain import Chain

__all__ = ["FILE", "KEY", "fingerprint", "lab_book", "observe", "saved_history_fingerprint"]

KEY = "labbook"  # the attachment's key
FILE = "labbook.jsonl"  # its file

_BOOKS: weakref.WeakKeyDictionary[Any, Chain] = weakref.WeakKeyDictionary()


def fingerprint(obj: Any) -> str | None:
    """A short fingerprint of a dataset's values or a result's estimates, for checking a replay; None for objects it
    cannot fingerprint. Estimates are rounded to 10 significant digits, so a replay on another machine matches."""
    import numpy as np
    import pandas as pd

    digest = hashlib.sha256()
    frame = getattr(obj, "frame", None)
    if isinstance(frame, pd.DataFrame):  # a Dataset
        digest.update("|".join(map(str, frame.columns)).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
        return digest.hexdigest()[:16]
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        digest.update(pd.util.hash_pandas_object(obj, index=True).to_numpy().tobytes())
        return digest.hexdigest()[:16]
    params = getattr(obj, "params", None)
    names, values = getattr(params, "names", None), getattr(params, "values", None)
    if names is not None and values is not None:
        try:
            rounded = np.array([float(f"{v:.10g}") for v in np.ravel(np.asarray(values, dtype=float))])
        except (TypeError, ValueError):
            return None
        digest.update("|".join(map(str, names)).encode())
        digest.update(rounded.tobytes())
        return digest.hexdigest()[:16]
    return None


MAX_ESTIMATES = 30  # coefficients kept for a fit, for counting specifications and drawing their curve


def _fit_summary(obj: Any) -> dict[str, Any]:
    """For a fitted model with a dependent variable: its outcome, sample and estimates (value and standard error, to
    8 significant digits). Results, not data: they let the lab book say how many specifications of an outcome were
    tried, and draw them (``pymetrix_labbook.tried``)."""
    outcome = getattr(obj, "dependent", None)
    params = getattr(obj, "params", None)
    names, values = getattr(params, "names", None), getattr(params, "values", None)
    if not isinstance(outcome, str) or names is None or values is None:
        return {}
    out: dict[str, Any] = {"outcome": outcome}
    sample = getattr(getattr(obj, "provenance", None), "sample", None)
    if isinstance(sample, tuple) and len(sample) == 2:
        out["sample"] = f"{sample[0]}-{sample[1]}"
    try:
        se = list(getattr(params, "se", None) if getattr(params, "se", None) is not None else [None] * len(values))
        out["estimates"] = {str(n): [float(f"{float(v):.8g}"), None if e is None else float(f"{float(e):.8g}")]
                            for n, v, e in list(zip(names, values, se))[:MAX_ESTIMATES]}
    except (TypeError, ValueError):
        pass
    return out


def _written(session: Any, record: Any) -> dict[str, Any]:
    out = {}
    for name, version in sorted(record.writes.items()):
        obj = session.objects.get(name) if session.versions.get(name) == version else None
        info: dict[str, Any] = {"version": version}
        if obj is not None:
            info["type"] = type(obj).__name__
            fp = fingerprint(obj)
            if fp:
                info["fingerprint"] = fp
            nobs = getattr(getattr(obj, "provenance", None), "nobs", None)
            if isinstance(nobs, int):
                info["nobs"] = nobs
            info.update(_fit_summary(obj))
        out[name] = info
    return out


def _step(session: Any, record: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"code": record.code, "reads": dict(sorted(record.reads.items())), "writes": _written(session, record)}
    call = getattr(record, "call", None)
    if call is not None and getattr(call, "function", None):
        data["call"] = call.function
    if getattr(record, "typed", False):
        data["typed"] = True  # Python kept as code: replayed only when trusted
    if not record.ok and record.error:
        data["error"] = record.error
    if getattr(record, "proposed_by", ""):  # an assistant proposed it; the person ran it from its dialog (pymetrix API 1.9)
        data["proposed_by"] = record.proposed_by
    from .brief import current_objectives

    linked = current_objectives(session)
    if linked:
        data["objectives"] = list(linked)  # what the step addresses (brief.objective, or the app's ticks)
    return data


def _history_fingerprint(history: Any) -> str:
    return hashlib.sha256("\n".join(r.code for r in history).encode("utf-8")).hexdigest()[:16]


def saved_history_fingerprint(history: Any) -> str:
    """The fingerprint of the steps a project keeps when saved (failed steps are not saved), so the one recorded at a
    save can be compared with the project as reopened."""
    return _history_fingerprint([r for r in history if r.ok or r.skipped or r.pruned or r.deferred])


def lab_book(session: Any, new: tuple[Any, ...] = ()) -> Chain:
    """The lab book of a session: read from its project, or started now (a ``start`` entry). ``new``: the records of
    the event that starts it, which are not work done before it."""
    from pymetrix import __version__ as pymetrix_version

    from ._version import __version__

    book = _BOOKS.get(session)
    if book is not None:
        return book
    blob = session.attachments.get(KEY, {}).get(FILE)
    if blob is not None:
        book = Chain.from_bytes(blob)
    else:
        before = [r for r in session.history if not any(r is n for n in new)]
        book = Chain()
        book.append("start", {"labbook": __version__, "pymetrix": pymetrix_version, "existing_steps": len(before),
                              "history_fingerprint": _history_fingerprint(before)})
    _BOOKS[session] = book
    return book


def _by(event: Any) -> dict[str, str] | None:
    """Who acted, from the event (pymetrix 0.1.0a10 on, extension API 1.5); None from an older pymetrix."""
    actor = getattr(event, "actor", None)
    return actor.as_dict() if actor is not None else None


def observe(event: Any) -> None:
    """The ``SessionObserver``'s handler: one entry for each event (none for steps replayed while opening)."""
    session = event.session
    if event.kind in ("step", "failed") and event.during == "opening":
        return
    book = lab_book(session, tuple(event.records) if event.kind in ("step", "failed") else ())
    when, by = event.time, _by(event)
    if event.kind in ("step", "failed"):
        from .plan import plan_label

        for record in event.records:
            data = _step(session, record)
            label = plan_label(book, record.code)  # planned or exploratory, once a plan is locked
            if label:
                data["plan"] = label
            book.append(event.kind, data, event.during, when, by)
    elif event.kind == "cancelled":
        book.append("cancelled", {"code": event.code, **dict(event.detail)}, event.during, when, by)
    elif event.kind in ("undone", "removed"):
        book.append(event.kind, {"codes": [r.code for r in event.records], **dict(event.detail)}, event.during, when, by)
    elif event.kind == "opened":
        book.append("opened", {"path": event.path, "steps": len(session.history),
                               "history_fingerprint": _history_fingerprint(session.history)}, event.during, when, by)
    elif event.kind == "saving":
        book.append("saved", {"path": event.path, "as_copy": bool(event.detail.get("as_copy")), "steps": len(session.history),
                              "history_fingerprint": saved_history_fingerprint(session.history)}, event.during, when, by)
        session.attachments[KEY] = {FILE: book.to_bytes()}
