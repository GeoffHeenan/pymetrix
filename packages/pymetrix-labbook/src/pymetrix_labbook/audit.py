"""The auditor's tools: a read-only MCP server over a project's lab book (research-audit layer 5).

    pymetrix-labbook audit study.pymx                     # an MCP server over stdio, for Claude Desktop, VS Code...

    {"mcpServers": {"audit": {"command": "pymetrix-labbook", "args": ["audit", "C:/review/study.pymx"]}}}

An assistant uses the tools to answer an auditor's questions, citing lab-book entries: were the exclusions decided
before the outcome was first fitted? Does the reported model match the plan? Did the sample change right after a
weak result? How many outcomes were tried before the reported one?

It is separate from the researcher's own assistant (pymetrix's MCP server), so the auditor is independent of the
audited. It reads the project file and never writes it. It starts from the deterministic checks (``verify``) and the
lab book's own record, not from the data: entries hold no data values. Its instructions tell the assistant to flag
patterns for a person to judge, citing entries, and never to give a verdict of its own.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from .chain import Chain
from .recorder import FILE, KEY

__all__ = ["AuditTools", "build_audit_server", "read_lab_book"]

INSTRUCTIONS = """You are helping a person audit a research project from its lab book: an append-only, hash-chained \
record of every step of the work in pymetrix, including steps undone, failed or removed, the project brief, notes and \
any pre-registered plan. The tools only read it.

- Start from verify_report and summary, then look closer with entries, brief, deviations and specifications.
- Cite the lab-book entries (their numbers) behind every statement.
- Flag patterns for the person to judge: work before the brief or plan, data read before a plan was locked, changes \
of sample or specification right after a result, hypotheses added after results on them, many specifications tried \
for the reported outcome. Say what the record shows and what it cannot show.
- Never give a verdict on the research or its authors. Exploratory work is normal; what matters is that it is labelled.
- The lab book proves what happened while it was recorded, not that nothing happened elsewhere."""

MAX_ENTRIES = 200


def read_lab_book(path: str | Path) -> Chain:
    """A project's lab book, read from its file without opening the project (no step is run)."""
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("manifest.json"))
        if FILE not in manifest.get("attachments", {}).get(KEY, []):
            raise ValueError(f"{path} has no lab book: it was never saved with pymetrix-labbook installed")
        return Chain.from_bytes(z.read(f"attachments/{KEY}/{FILE}"))


def _line(entry: dict[str, Any]) -> str:
    from .verify import describe_actor

    data = entry.get("data", {})
    shown = {k: v for k, v in data.items() if k not in ("writes", "reads")}
    writes = data.get("writes", {})
    if writes:
        shown["writes"] = {n: {k: v for k, v in w.items() if k in ("type", "outcome", "sample", "nobs")} for n, w in writes.items()}
    if data.get("reads"):
        shown["reads"] = sorted(data["reads"])
    return (f"{entry.get('seq')} · {entry.get('time', '')} · {entry.get('event')} ({entry.get('during', '')}) · "
            f"{describe_actor(entry.get('by')) if 'by' in entry else 'not recorded'} · "
            f"{json.dumps(shown, ensure_ascii=False, sort_keys=True)}")


class AuditTools:
    """The tools' logic, over one project file (tested without MCP)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        read_lab_book(self.path)  # fail now, not at the first question

    @property
    def chain(self) -> Chain:
        return read_lab_book(self.path)  # read each time: the file may be replaced by a newer save

    def summary(self) -> str:
        from .brief import briefs
        from .plan import current_plan
        from .tried import outcome_counts
        from .verify import actors

        chain = self.chain
        entries = chain.entries
        kinds: dict[str, int] = {}
        for e in entries:
            kinds[e.get("event", "?")] = kinds.get(e.get("event", "?"), 0) + 1
        found = briefs(chain)
        plan = current_plan(chain)
        lines = [f"# Lab book of {self.path.name}", "",
                 f"- Entries: {len(entries)} ({', '.join(f'{k} {n}' for k, n in sorted(kinds.items()))}); head {chain.head[:16]}",
                 f"- From {entries[0].get('time', '?')} to {entries[-1].get('time', '?')}" if entries else "- Empty",
                 "- Who acted: " + ("; ".join(f"{a}: {n} step(s)" for a, n in actors(chain).items()) or "no steps"),
                 f"- Brief: {found[-1][1].get('title', '')} (written at entry {found[0][0]}, {len(found)} version(s))" if found
                 else "- Brief: none" + (" (a private brief is committed, not revealed)"
                                         if any(e.get("event") == "brief_commitment" for e in entries) else ""),
                 f"- Plan: {len(plan['specifications'])} specification(s) locked" if plan else "- Plan: none",
                 "- Outcomes fitted: " + ("; ".join(f"{o}: {k} specification(s), {n} fit(s)"
                                                    for o, (k, n) in sorted(outcome_counts(chain).items())) or "none")]
        return "\n".join(lines)

    def entries(self, kind: str = "", outcome: str = "", first: int = 0, last: int = -1, limit: int = 50) -> str:
        chain = self.chain
        chosen = []
        for e in chain.entries:
            seq = e.get("seq", -1)
            if seq < first or (last >= 0 and seq > last):
                continue
            if kind and e.get("event") != kind:
                continue
            if outcome and not any(w.get("outcome") == outcome for w in e.get("data", {}).get("writes", {}).values()):
                continue
            chosen.append(e)
        limit = max(1, min(int(limit), MAX_ENTRIES))
        shown = chosen[:limit]
        more = f"\n\n({len(chosen) - limit} more: narrow by kind, outcome or range)" if len(chosen) > limit else ""
        return ("\n".join(_line(e) for e in shown) or "No entries match.") + more

    def brief(self) -> str:
        from .brief import briefs

        chain = self.chain
        versions = briefs(chain)
        out = [f"## Version {i + 1}, written at entry {seq}\n\n```json\n{json.dumps(data, ensure_ascii=False, indent=1)}\n```"
               for i, (seq, data) in enumerate(versions)]
        pending = [e["seq"] for e in chain.entries if e.get("event") == "brief_commitment"]
        revealed = {e["data"].get("commitment") for e in chain.entries if e.get("event") == "brief_revealed"}
        out += [f"A private brief committed at entry {s} is not revealed: only its hash is known." for s in pending if s not in revealed]
        skipped = [e for e in chain.entries if e.get("event") == "brief_skipped"]
        out += [f"The brief was skipped at entry {e['seq']}: {e['data'].get('reason', '')}" for e in skipped]
        return "\n\n".join(out) or "No brief was recorded."

    def deviations(self) -> str:
        from .plan import plan_report
        from .verify import brief_report

        chain = self.chain
        out = []
        for title, (warn, notes, checked) in (("Brief", brief_report(chain)), ("Plan", plan_report(chain))):
            if warn or notes or checked:
                out.append(f"## {title}\n" + "\n".join([*(f"- warning: {w}" for w in warn), *(f"- note: {n}" for n in notes),
                                                         *(f"- checked: {c}" for c in checked)]))
        return "\n\n".join(out) or "No brief or plan to compare the work with."

    def specifications(self, outcome: str = "") -> str:
        from .tried import fits, outcome_counts

        chain = self.chain
        if not outcome:
            counts = outcome_counts(chain)
            return "\n".join(f"- {o}: {k} specification(s), {n} fit(s)" for o, (k, n) in sorted(counts.items())) or "No fits recorded."
        rows = fits(chain, outcome)
        if not rows:
            return f"No fit of {outcome} was recorded."
        return "\n".join(f"- entry {f['seq']}: {f['name']} = {f['spec'][0]} · sample {f['sample'] or '?'}"
                         + (" · failed" if f["failed"] else "")
                         + "".join(f" · {t} {v[0]:.4g}" + (f" (se {v[1]:.3g})" if v[1] is not None else "")
                                   for t, v in list(f["estimates"].items())[:6])
                         for f in rows)

    def verify_report(self, replay: bool = False) -> str:
        from .verify import verify

        return str(verify(self.path, replay=replay))


def build_audit_server(path: str | Path) -> Any:
    """The read-only MCP server over the project at ``path``."""
    from pymetrix.adapters import require

    from ._version import __version__

    server_module = require("mcp.server.mcpserver")
    tools = AuditTools(path)
    server = server_module.MCPServer("pymetrix-labbook-audit", title="pymetrix lab-book audit", version=__version__,
                                     instructions=INSTRUCTIONS)
    server.audit_tools = tools

    @server.tool(structured_output=False)
    def summary() -> str:
        """The lab book at a glance: entries by kind, who acted, the brief, the plan and the outcomes fitted."""
        return tools.summary()

    @server.tool(structured_output=False)
    def entries(kind: str = "", outcome: str = "", first: int = 0, last: int = -1, limit: int = 50) -> str:
        """Lab-book entries, one a line, with their number, time, who acted and what they hold. Narrow by ``kind``
        (step, failed, undone, removed, cancelled, brief, note, plan, saved...), by the ``outcome`` a step fitted, or
        by the range of entry numbers ``first`` to ``last``."""
        return tools.entries(kind, outcome, first, last, limit)

    @server.tool(structured_output=False)
    def brief() -> str:
        """Every version of the project brief, each with the entry at which it was written; private or skipped briefs."""
        return tools.brief()

    @server.tool(structured_output=False)
    def deviations() -> str:
        """The work against the brief and the plan: work before them, objectives with no analysis, hypotheses added
        after results, data read before the plan, planned specifications not run, exploratory steps, changes."""
        return tools.deviations()

    @server.tool(structured_output=False)
    def specifications(outcome: str = "") -> str:
        """How many specifications of each outcome were fitted, kept or not; with ``outcome``, each fit of it, with
        its entry, code, sample and estimates."""
        return tools.specifications(outcome)

    @server.tool(structured_output=False)
    def verify_report(replay: bool = False) -> str:
        """The deterministic checks: the chain, the save log, the steps, the brief and plan; with ``replay``, the
        project's steps are run again and each result compared with its recorded fingerprint (slower)."""
        return tools.verify_report(replay)

    return server
