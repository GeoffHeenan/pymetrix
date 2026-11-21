"""Checking a project's lab book: deterministic checks, with no AI involved.

``verify(path)`` reads the project file and checks, in order:

1. **The chain:** every entry's hash, its number and its link to the entry before (``Chain.check``).
2. **Coverage:** the project's save log against the lab book. A save made without the lab book active, or more saves
   than the lab book recorded, means work the lab book did not see.
3. **The steps:** the project's steps, as saved, are those the lab book recorded at its last save.
4. **The replay** (``replay=True``): the project is opened, which runs its steps again, and each current result's
   fingerprint is compared with the one the lab book recorded when it was made.
5. **The brief** (``brief_report``): whether one was written before the work, objectives with no analysis, analyses
   linked to no objective, and hypotheses added only after results on them existed.
6. **The plan** (``plan.plan_report``): planned data read before the plan was locked, planned specifications not run,
   exploratory steps since, and changes to the plan.
7. **What was tried** (``tried.outcome_counts``): how many specifications of each outcome were fitted.

What the replay cannot check is listed rather than passed over: typed code not run because it is not trusted, steps
whose plugin is missing, results not computed yet, and objects without a fingerprint.
"""

from __future__ import annotations

import json
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .chain import Chain
from .recorder import FILE, KEY, fingerprint, saved_history_fingerprint

__all__ = ["Verification", "actors", "brief_report", "describe_actor", "verify"]


@dataclass
class Verification:
    """What ``verify`` found: errors (the lab book or the project is not what it should be), warnings (gaps the lab
    book cannot account for) and notes (what could not be checked, and why)."""

    path: str
    entries: int = 0
    head: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)  # what was checked and passed
    actors: dict[str, int] = field(default_factory=dict)  # who took the steps: "agent claude-ai (mcp)" -> steps

    @property
    def ok(self) -> bool:
        """No errors (warnings and notes may remain)."""
        return not self.errors

    def __str__(self) -> str:
        lines = [f"Lab book of {self.path}: {self.entries} entries, head {self.head[:16] or '-'}",
                 "Verified: no errors." if self.ok else f"NOT VERIFIED: {len(self.errors)} error(s)."]
        who = [f"{actor}: {n} step(s)" for actor, n in self.actors.items()]
        for title, items in (("Errors", self.errors), ("Warnings", self.warnings), ("Not checked", self.notes),
                             ("Who acted", who), ("Checked", self.checked)):
            if items:
                lines += ["", f"{title}:"] + [f"  - {x}" for x in items]
        return "\n".join(lines)


def _current_writes(chain: Chain) -> dict[str, dict[str, Any]]:
    """What the lab book says each current object is: the last recorded write of each name, after taking out steps
    later undone (the latest of each undone code) and steps removed."""
    steps: list[dict[str, Any]] = []
    for entry in chain.entries:
        event, data = entry.get("event"), entry.get("data", {})
        if event == "step":
            steps.append(data)
        elif event == "undone":
            for code in reversed(data.get("codes", [])):
                for i in range(len(steps) - 1, -1, -1):
                    if steps[i].get("code") == code:
                        del steps[i]
                        break
        elif event == "removed":
            for code in data.get("codes", []):
                for i, step in enumerate(steps):
                    if step.get("code") == code:
                        del steps[i]
                        break
    writes: dict[str, dict[str, Any]] = {}
    for step in steps:
        writes.update(step.get("writes", {}))
    return writes


def describe_actor(by: dict[str, Any] | None) -> str:
    """One actor as a line reads it: "person geoff (app)", "agent claude-ai 0.1.0 (mcp)"; "not recorded" for entries
    written before the lab book recorded who acted."""
    if not by:
        return "not recorded"
    name = f" {by['name']}" if by.get("name") else ""
    return f"{by.get('kind', '?')}{name} ({by.get('via', '?')})"


def actors(chain: Chain) -> dict[str, int]:
    """How many steps (run or failed) each actor took, in the order they first acted."""
    counts: dict[str, int] = {}
    for entry in chain.entries:
        if entry.get("event") in ("step", "failed"):
            key = describe_actor(entry.get("by"))
            counts[key] = counts.get(key, 0) + 1
    return counts


def brief_report(chain: Chain) -> tuple[list[str], list[str], list[str]]:
    """What the lab book shows about the project brief: (warnings, notes, checked)."""
    warn: list[str] = []
    notes: list[str] = []
    checked: list[str] = []
    from .brief import _digest
    from .brief import briefs as all_briefs

    entries = [e for e in chain.entries if "unreadable" not in e]
    steps = [e for e in entries if e.get("event") == "step"]
    skipped = [e for e in entries if e.get("event") == "brief_skipped"]
    commitments = {e["seq"]: e for e in entries if e.get("event") == "brief_commitment"}
    revealed = [e for e in entries if e.get("event") == "brief_revealed"]
    for r in revealed:  # a revealed brief must be the one committed to
        data = r.get("data", {})
        committed = commitments.get(data.get("commitment"), {}).get("data", {}).get("hash")
        payload = {k: data[k] for k in ("brief", "salt") if k in data}
        if committed is None or _digest(payload) != committed:
            warn.append(f"the brief revealed at entry {r['seq']} does not match the commitment it names")
        else:
            checked.append(f"the brief revealed at entry {r['seq']} matches its commitment at entry {data['commitment']}")
    pending = sorted(set(commitments) - {r["data"].get("commitment") for r in revealed})
    for seq in pending:
        notes.append(f"a private brief was committed at entry {seq} (hash {commitments[seq]['data']['hash'][:16]}…) and "
                     "is not yet revealed: it cannot be checked until it is")
    briefs = [{"seq": seq, "data": data} for seq, data in all_briefs(chain)]
    if not briefs:
        if pending:
            first = pending[0]
            before = [e for e in steps if e["seq"] < first]
            if before:
                warn.append(f"the private brief was committed at entry {first}, after {len(before)} step(s) of work")
            return warn, notes, checked
        if skipped:
            notes.append(f"no project brief: skipped at entry {skipped[0]['seq']} ({skipped[0]['data'].get('reason', '')})")
        else:
            notes.append("no project brief was recorded: objectives and hypotheses cannot be checked")
        return warn, notes, checked
    first = briefs[0]["seq"]
    before = [e for e in steps if e["seq"] < first]
    if before:
        warn.append(f"the brief was written at entry {first}, after {len(before)} step(s) of work")
    else:
        checked.append(f"the brief: written at entry {first}, before any step")
    for amendment in briefs[1:]:
        notes.append(f"the brief was amended at entry {amendment['seq']}: {amendment['data'].get('reason', '')}")
    latest = briefs[-1]["data"]
    known = {**latest.get("objectives", {}), **latest.get("hypotheses", {})}
    tagged: dict[str, int] = {}  # id -> the entry of the first step linked to it
    untagged = 0
    for step in steps:
        ids = step.get("data", {}).get("objectives", [])
        if not ids:
            untagged += 1
        for i in ids:
            tagged.setdefault(i, step["seq"])
    unaddressed = [i for i in known if i not in tagged]
    if unaddressed:
        warn.append(f"objectives or hypotheses with no analysis linked to them: {', '.join(unaddressed)}")
    elif known:
        checked.append(f"every objective and hypothesis in the brief has analysis linked to it ({len(known)})")
    unknown = sorted(i for i in tagged if i not in known)
    if unknown:
        warn.append(f"steps are linked to ids the brief does not name: {', '.join(unknown)}")
    if untagged:
        notes.append(f"{untagged} step(s) are linked to no objective: exploratory work")
    for i, at in sorted(tagged.items(), key=lambda kv: kv[1]):
        added = next((b["seq"] for b in briefs if i in {**b["data"].get("objectives", {}), **b["data"].get("hypotheses", {})}), None)
        if added is not None and added > at:
            warn.append(f"{i} was added to the brief at entry {added}, after a step on it at entry {at}")
    return warn, notes, checked


def verify(path: str | Path, replay: bool = True, trust: bool | None = None) -> Verification:
    """Check a project's lab book (see the module docstring). ``trust`` is passed to ``load_project`` for the replay:
    None runs typed code only if it is trusted."""
    path = Path(path)
    result = Verification(str(path))
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("manifest.json"))
        names = manifest.get("attachments", {}).get(KEY, [])
        blob = z.read(f"attachments/{KEY}/{FILE}") if FILE in names else None
    if blob is None:
        result.errors.append("the project has no lab book: it was never saved with pymetrix-labbook installed")
        return result
    chain = Chain.from_bytes(blob)
    result.entries, result.head = len(chain.entries), chain.head
    result.actors = actors(chain)

    # 1. the chain
    problems = chain.check()
    result.errors += [f"the chain is broken: {p}" for p in problems]
    if not problems:
        result.checked.append(f"the chain: {len(chain.entries)} entries, each hash and link intact")
    start = chain.entries[0] if chain.entries else {}
    if start.get("event") == "start" and start.get("data", {}).get("existing_steps"):
        result.notes.append(f"the lab book began on a project that already had {start['data']['existing_steps']} step(s); "
                            "the work before it is not recorded")

    # the brief (from the lab book alone; check 5)
    brief_warnings, brief_notes, brief_checked = brief_report(chain)
    result.warnings += brief_warnings
    result.notes += brief_notes
    result.checked += brief_checked
    from .plan import plan_report

    plan_warnings, plan_notes, plan_checked = plan_report(chain)
    result.warnings += plan_warnings
    result.notes += plan_notes
    result.checked += plan_checked
    from .tried import outcome_counts

    for outcome, (specs, n) in sorted(outcome_counts(chain).items()):
        result.notes.append(f"{outcome}: {specs} specification(s) fitted ({n} fit(s), kept or not)")

    # 2. coverage: the save log against the lab book's saves
    log = [s for s in manifest.get("save_log", []) if not s.get("copy")]
    first = next((i for i, s in enumerate(log) if KEY in s.get("recorders", {})), None)
    if first is None:
        result.errors.append("the save log has no save with the lab book active")
    else:
        for s in log[first:]:
            if KEY not in s.get("recorders", {}):
                result.warnings.append(f"saved on {s.get('saved', '?')} without the lab book active: work before that save "
                                       "may not be recorded")
        saves = [e for e in chain.entries if e.get("event") == "saved" and not e.get("data", {}).get("as_copy")]
        own = sum(1 for s in log[first:] if KEY in s.get("recorders", {}))
        if own != len(saves):
            result.warnings.append(f"the save log lists {own} save(s) with the lab book active, but the lab book recorded "
                                   f"{len(saves)}")
        elif not result.warnings:
            result.checked.append(f"coverage: every one of the {len(log) - first} save(s) since the lab book began had it active")

    # 3. and 4. the steps, and the replay
    last = chain.entries[-1] if chain.entries else {}
    if last.get("event") != "saved":
        result.errors.append("the lab book does not end with the save that wrote it: entries after it were removed, or it "
                             "was replaced")
    if not replay:
        result.notes.append("the replay was not run (replay=False): steps and results not checked")
        return result
    from pymetrix.commands import load_project

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = load_project(path, trust=trust)
    expected = last.get("data", {}).get("history_fingerprint")
    if expected and saved_history_fingerprint(session.history) != expected:
        result.errors.append("the project's steps are not those the lab book recorded at its last save: the project was "
                             "changed outside pymetrix, or its lab book was copied from another project")
    elif expected:
        result.checked.append(f"the steps: the project's {len(session.history)} step(s) are those recorded at the last save")
    skipped = [r for r in session.history if r.skipped]
    if skipped:
        result.notes.append(f"{len(skipped)} step(s) were not run on replay (typed code not trusted, or a plugin missing): "
                            "their results are not checked")
    if any("not trusted" in str(w.message) for w in caught):
        result.notes.append("typed code was not run, because it is not trusted; run with trust=True after reading it")
    pending = set(session.pending())
    matched, unmatched = 0, []
    for name, info in sorted(_current_writes(chain).items()):
        recorded = info.get("fingerprint")
        if not recorded or name not in session.objects or name in pending:
            continue
        now = fingerprint(session.objects[name])
        if now == recorded:
            matched += 1
        else:
            unmatched.append(name)
    for name in unmatched:
        result.errors.append(f"{name} differs on replay from the result the lab book recorded")
    if matched and not unmatched:
        result.checked.append(f"the replay: {matched} result(s) match the fingerprints recorded when they were made")
    if pending:
        result.notes.append(f"{len(pending)} result(s) not computed on opening (deferred): {', '.join(sorted(pending))}")
    return result
