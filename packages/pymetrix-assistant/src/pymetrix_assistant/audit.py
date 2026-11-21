"""The assistant's record in the project (assistant stage 2, step 3): each question, the exact text sent with it, the
answer, the proposals, and what the user did with each (opened and ran it as step N, with the settings changed
first; opened and cancelled; skipped).
It is kept in ``session.attachments["pymetrix-assistant"]["audit.jsonl"]``, one JSON object a line, so it is saved
and reopened with the project, and kept as it is where this plugin is not installed.

    for entry in read(session): ...
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

__all__ = ["ATTACHMENT", "FILE", "append", "as_text", "next_question", "read", "resumable"]

ATTACHMENT = "pymetrix-assistant"
FILE = "audit.jsonl"


def read(session: Any) -> list[dict[str, Any]]:
    """The record's entries, oldest first (lines that cannot be read are left out)."""
    raw = (getattr(session, "attachments", {}) or {}).get(ATTACHMENT, {}).get(FILE, b"")
    out = []
    for line in raw.decode("utf-8", "replace").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            out.append(entry)
    return out


def append(session: Any, kind: str, **data: Any) -> dict[str, Any]:
    """Add an entry (``kind``: asked, failed, shown, opened, skipped), stamped with the time; the project has changed."""
    entry = {"kind": kind, "time": datetime.now(UTC).isoformat(timespec="seconds"), **data}
    files = session.attachments.setdefault(ATTACHMENT, {})
    files[FILE] = files.get(FILE, b"") + (json.dumps(entry, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    session.dirty = True
    return entry


def next_question(session: Any) -> int:
    """The number the next question is given (1, 2, ...)."""
    return 1 + max((int(e.get("question", 0)) for e in read(session) if e.get("kind") in ("asked", "failed")), default=0)


def as_text(entries: list[dict[str, Any]]) -> str:
    """The record, for people: each question with what was sent, the answer and what became of each proposal."""
    parts: list[str] = []
    for e in entries:
        kind, when = e.get("kind"), str(e.get("time", ""))[:19].replace("T", " ")
        q = e.get("question")
        if kind == "asked":
            parts.append(f"── Question {q}, {when} UTC, {e.get('model', '')}, mode {e.get('mode', '')}"
                         + (" (answered by Anthropic's fallback model)" if e.get("fallback") else ""))
            parts.append(f"You: {e.get('text', '')}")
            parts.append(f"Sent with it: {e.get('sent', '')}")
            if e.get("reply"):
                parts.append(f"Answer: {e['reply']}")
            for p in e.get("proposals", []):
                parts.append(f"Proposal {q}.{p.get('n')}: {p.get('command')}: {p.get('why', '')}\n    {p.get('code', '')}")
        elif kind == "failed":
            parts.append(f"── Question {q}, {when} UTC: not answered ({e.get('error', '')})\nYou: {e.get('text', '')}")
        elif kind == "opened":
            ran = f"ran it as step {e['step']}" if e.get("step") else "closed the dialog without running it"
            if e.get("step") and e.get("changed"):
                ran += f", after changing {', '.join(e['changed'])}"
            parts.append(f"Proposal {q}.{e.get('proposal')} ({e.get('command')}), {when} UTC: opened; you {ran}")
        elif kind == "shown":
            what = f"{e.get('object')}" + (f" ({e['section']})" if e.get("section") else "")
            parts.append(f"Question {q}, {when} UTC: the assistant asked to see {what}; "
                         + (f"you sent it:\n{e.get('text', '')}" if e.get("sent") else "you did not send it"))
        elif kind == "skipped":
            parts.append(f"Proposal {q}.{e.get('proposal')} ({e.get('command')}), {when} UTC: skipped")
    return "\n".join(parts) if parts else "Nothing asked in this project yet."


MAX_RESUMED = 20  # questions brought back into a resumed conversation, at most (the latest)


def resumable(entries: list[dict[str, Any]], provider: str, mode: str, full: bool) -> tuple[int | None, list[dict[str, Any]]]:
    """The latest conversation that can be continued as things are now, and its answered questions, oldest first.

    Only one held with the same destination (``provider``: anthropic or local), the same mode and the same full-context
    choice: its answers were made from what that destination was told, so they go back to the same place, never to
    one told less (a local model's answers from full context are not sent to Anthropic in metadata only). Records
    from before conversations were numbered count as one conversation."""
    matching = [e for e in entries if e.get("kind") == "asked" and e.get("provider", "anthropic") == provider
                and e.get("mode") == mode and bool(e.get("full", False)) == bool(full)]
    if not matching:
        return None, []
    latest = matching[-1].get("conversation", 0)
    chosen = [e for e in matching if e.get("conversation", 0) == latest][-MAX_RESUMED:]
    return latest, chosen


def as_turns(entries: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Each question and its answer, the answer with the steps it proposed: the turns a resumed conversation starts
    from."""
    turns = []
    for e in entries:
        answer = str(e.get("reply") or "")
        proposed = [f"- {p.get('command')}: {p.get('why', '')} ({p.get('code', '')})" for p in e.get("proposals", [])]
        if proposed:
            answer = (answer + "\n\n" if answer else "") + "Proposed:\n" + "\n".join(proposed)
        turns.append((str(e.get("text") or ""), answer or "(no answer)"))
    return turns
