"""The project brief, notes and objectives: what the work set out to do, written into the lab book before it is done.

The design is pymetrix's ``docs/design/research-audit.md``, layer 2. A brief gives the rest of the lab book its
meaning: without a stated scope, an auditor can count what was tried but cannot tell whether it was on topic::

    from pymetrix_labbook import brief, note, objective

    brief(session, "Fiscal multipliers in Norland", "How large is the government-spending multiplier?",
          objectives={"O1": "Estimate the multiplier over two years"},
          hypotheses={"H1": {"statement": "The multiplier is below one",
                             "supports": "a two-year cumulative multiplier under 1"}},
          primary_outcomes=["two-year cumulative multiplier"], data={"sources": "PSETS: Norland national accounts"})
    with objective("O1", "H1"):
        session.run(...)                                     # these steps are linked to O1 and H1
    note(session, "Dropped 2020: the survey changed its method", about="O1")
    brief(session, ..., reason="Added H2 after the referee's comment")   # an amendment, never an edit

- **A brief** writes a ``brief`` entry. A later brief is an amendment and needs its ``reason``; the record shows how
  the project changed, and when. ``skip_brief`` records that there is none, and why.
- **A private brief** (``commit_brief``, ``reveal_brief``): only the brief's hash goes into the lab book, to be
  time-stamped outside pymetrix; the brief itself, kept in a file, is revealed at publication and checked against it.
- **Notes** write ``note`` entries: the reasoning a paper lab book holds, linked to an objective or a step.
- **Objectives:** steps run inside ``objective(...)`` are tagged with the objectives and hypotheses they address.
  ``verify`` then reports objectives with no analysis, analyses linked to none (exploratory, which is normal, but
  labelled), hypotheses added to the brief only after results on them existed, and a brief written after the work
  began.
"""

from __future__ import annotations

import contextvars
import getpass
import hashlib
import json
import secrets
import weakref
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .recorder import lab_book

__all__ = ["active_objectives", "brief", "briefs", "commit_brief", "current_brief", "current_objectives", "note", "objective",
           "reveal_brief", "set_active_objectives", "skip_brief"]

_OBJECTIVES: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar("labbook_objectives", default=())
_ACTIVE: weakref.WeakKeyDictionary[Any, tuple[str, ...]] = weakref.WeakKeyDictionary()  # set in the app's panel


def _by(session: Any) -> dict[str, str]:
    """Who is writing: the agent acting for the session, or the person at the machine."""
    actor = getattr(session, "actor", None)
    if actor is not None:
        return actor.as_dict()
    try:
        name = getpass.getuser()
    except Exception:  # noqa: BLE001 - no login name: unknown, not an error
        name = ""
    from pymetrix.commands import events

    return {"kind": "person", "name": name, "via": getattr(events, "INTERFACE", "python")}


def _append(session: Any, event: str, data: dict[str, Any]) -> dict[str, Any]:
    entry = lab_book(session).append(event, data, by=_by(session))
    session.dirty = True  # a brief or a note is work to save
    return entry


def _hypotheses(given: Mapping[str, Any] | None) -> dict[str, dict[str, str]]:
    out = {}
    for key, value in (given or {}).items():
        item = {"statement": value} if isinstance(value, str) else dict(value)
        if not item.get("statement"):
            raise ValueError(f"hypothesis {key!r} needs a statement")
        out[str(key)] = {k: str(v) for k, v in item.items() if k in ("statement", "supports", "contradicts") and v}
    return out


def briefs(session_or_chain: Any) -> list[tuple[int, dict[str, Any]]]:
    """Every version of the brief, as (entry, brief): those written openly, at their own entry, and those committed
    privately and since revealed, at their commitment's entry (when they were written)."""
    from .chain import Chain

    chain = session_or_chain if isinstance(session_or_chain, Chain) else lab_book(session_or_chain)
    out = [(e["seq"], e["data"]) for e in chain.entries if e.get("event") == "brief"]
    out += [(e["data"]["commitment"], e["data"]["brief"]) for e in chain.entries if e.get("event") == "brief_revealed"]
    return sorted(out, key=lambda x: x[0])


def current_brief(session: Any) -> dict[str, Any] | None:
    """The project's brief as it stands (the latest, amendments included; a committed brief once revealed), or None."""
    found = briefs(session)
    return found[-1][1] if found else None


def _has_brief(session: Any) -> bool:
    return bool(briefs(session)) or any(e.get("event") == "brief_commitment" for e in lab_book(session).entries)


def _brief_record(session: Any, title: str, question: str, *, why: str = "", objectives: Mapping[str, str] | None = None,
                  hypotheses: Mapping[str, Any] | None = None, primary_outcomes: Sequence[str] = (),
                  secondary_outcomes: Sequence[str] = (), data: Mapping[str, str] | None = None, methods: str = "",
                  limits: str = "", team: Mapping[str, str] | None = None, funder: str = "", conditions: str = "",
                  reason: str = "") -> dict[str, Any]:
    if not title or not question:
        raise ValueError("a brief needs a title and a research question")
    amending = _has_brief(session)
    if amending and not reason:
        raise ValueError("the project already has a brief: an amendment needs its reason (reason=...)")
    ids = [*map(str, objectives or {}), *map(str, hypotheses or {})]
    if len(ids) != len(set(ids)):
        raise ValueError("objective and hypothesis ids must all differ")
    record: dict[str, Any] = {
        "title": title, "question": question, "why": why,
        "objectives": {str(k): str(v) for k, v in (objectives or {}).items()},
        "hypotheses": _hypotheses(hypotheses),
        "primary_outcomes": list(primary_outcomes), "secondary_outcomes": list(secondary_outcomes),
        "data": {str(k): str(v) for k, v in (data or {}).items()}, "methods": methods, "limits": limits,
        "team": {str(k): str(v) for k, v in (team or {}).items()}, "funder": funder, "conditions": conditions,
    }
    if amending:
        record["reason"] = reason
    return {k: v for k, v in record.items() if v not in ("", [], {})}


def brief(session: Any, title: str, question: str, **fields: Any) -> dict[str, Any]:
    """Record the project brief: its scope and objectives, before the work. Called again, it amends the brief and
    needs ``reason``; the whole brief is restated, so each ``brief`` entry stands on its own.

    Keywords: ``why``; ``objectives``: id -> what it aims at ({"O1": "..."}); ``hypotheses``: id -> statement, or
    {"statement", "supports", "contradicts"}: what would support or contradict it; ``primary_outcomes`` and
    ``secondary_outcomes``; ``data``: the intended sources, vintages, sample period and units; ``methods``: broadly,
    not exact specifications; ``limits``; ``team``: name -> role; ``funder``; ``conditions``; ``reason``.

    To keep the brief private until publication, with its date still provable, use ``commit_brief``."""
    return _append(session, "brief", _brief_record(session, title, question, **fields))


def _digest(payload: dict[str, Any]) -> str:
    from .chain import _canonical

    return hashlib.sha256(_canonical(payload)).hexdigest()


def commit_brief(session: Any, path: str | Path, title: str, question: str, **fields: Any) -> str:
    """Commit to a brief without disclosing it: the brief, with a random salt, is written to ``path`` (keep it
    private, and unchanged), and the lab book records only its hash. Register that hash with a time-stamping service
    (OSF, an RFC 3161 authority, a journal's registered-report system) to prove when it was written; ``reveal_brief``
    discloses it later. Returns the hash. Keywords as ``brief``'s; a later commitment amends and needs ``reason``."""
    record = _brief_record(session, title, question, **fields)
    payload = {"brief": record, "salt": secrets.token_hex(16)}  # the salt stops anyone guessing a short brief
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists: a commitment's file is never overwritten")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    digest = _digest(payload)
    _append(session, "brief_commitment", {"hash": digest, "algorithm": "sha256"})
    return digest


def reveal_brief(session: Any, path: str | Path) -> dict[str, Any]:
    """Disclose a committed brief: the file ``commit_brief`` wrote is checked against the commitment and the brief
    enters the lab book, counted as written when it was committed."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = _digest(payload)
    entries = lab_book(session).entries
    revealed = {e["data"]["commitment"] for e in entries if e.get("event") == "brief_revealed"}
    match = next((e for e in entries if e.get("event") == "brief_commitment" and e["data"].get("hash") == digest), None)
    if match is None:
        raise ValueError(f"{path} matches no commitment in this lab book: it was changed, or belongs to another project")
    if match["seq"] in revealed:
        raise ValueError(f"the brief committed at entry {match['seq']} is already revealed")
    return _append(session, "brief_revealed", {"commitment": match["seq"], "hash": digest, **payload})


def skip_brief(session: Any, reason: str) -> dict[str, Any]:
    """Record that the project has no brief, and why (exploratory work, a teaching exercise...)."""
    if not reason:
        raise ValueError("say why the brief is skipped")
    return _append(session, "brief_skipped", {"reason": reason})


def note(session: Any, text: str, about: str = "", step: int | None = None) -> dict[str, Any]:
    """Record a note: the reasoning behind a choice. ``about``: the objective or hypothesis it concerns; ``step``: the
    lab-book entry (its number) it concerns, such as the step it explains."""
    if not text.strip():
        raise ValueError("a note needs text")
    data: dict[str, Any] = {"text": text}
    if about:
        data["about"] = about
    if step is not None:
        data["step"] = int(step)
    return _append(session, "note", data)


@contextmanager
def objective(*ids: str) -> Iterator[None]:
    """Link the steps run inside the block to these objectives or hypotheses (nested blocks add to the outer ones)."""
    if not ids:
        raise ValueError("name at least one objective or hypothesis")
    token = _OBJECTIVES.set(tuple(dict.fromkeys((*_OBJECTIVES.get(), *map(str, ids)))))
    try:
        yield
    finally:
        _OBJECTIVES.reset(token)


def set_active_objectives(session: Any, ids: Sequence[str]) -> None:
    """Link every step run in ``session`` from now on to these objectives, until changed (the app's Lab book panel:
    its ticks). An empty list ends it."""
    if ids:
        _ACTIVE[session] = tuple(dict.fromkeys(map(str, ids)))
    else:
        _ACTIVE.pop(session, None)


def active_objectives(session: Any) -> tuple[str, ...]:
    """The objectives ticked for ``session`` (``set_active_objectives``)."""
    return _ACTIVE.get(session, ())


def current_objectives(session: Any = None) -> tuple[str, ...]:
    """The objectives and hypotheses the steps now being run are linked to: those of any ``objective(...)`` block and
    those ticked for the session (the recorder reads this)."""
    ticked = _ACTIVE.get(session, ()) if session is not None else ()
    return tuple(dict.fromkeys((*_OBJECTIVES.get(), *ticked)))
