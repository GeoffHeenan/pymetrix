"""The pre-registered analysis plan: exact specifications locked before the data are read, for confirmatory work.

The design is pymetrix's ``docs/design/research-audit.md``, layer 2 (the stricter of its two records)::

    from pymetrix_labbook import lock_plan

    lock_plan(session, specifications=[FitEquation("c ~ y", name="fit"), "var = VAR(['c', 'y'], lags=2).fit(data)"],
              data=["data"], sample="1990Q1-2019Q4", exclusions="none", hypotheses=["H1"],
              primary_outcomes=["the coefficient on y"])

- **Locking** writes a ``plan`` entry whose ``plan_hash`` covers the plan. A plan is locked once: changing it later
  needs a ``reason``, and both plans stay in the chain, so a change is visible and dated.
- **Each later step** is labelled ``planned`` (its code matches a planned specification, whatever the result is named)
  or ``exploratory``. Exploratory work is normal; the point is that it is labelled.
- **``plan_report``** (run by ``verify``) lists the deviations: planned data read before the plan was locked, planned
  specifications never run, exploratory steps since the lock, and changes to the plan with the work done before each.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any

from .chain import Chain
from .recorder import lab_book

__all__ = ["current_plan", "lock_plan", "normalise", "plan_label", "plan_report"]

_ASSIGN = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*(?!=)")


def normalise(code: str) -> str:
    """A step's code without the name it is assigned to, and without spacing, so that
    ``fit = Equation('c ~ y').fit(data)`` and ``fit2 = Equation('c ~ y').fit(data)`` are the same specification."""
    return re.sub(r"\s+", "", _ASSIGN.sub("", code.strip(), count=1))


def _code(spec: Any) -> str:
    if isinstance(spec, str):
        return spec
    to_code = getattr(spec, "to_code", None)
    if to_code is None:
        raise TypeError(f"a specification is a step's code or a pymetrix command, not {type(spec).__name__}")
    return to_code()


def current_plan(session_or_chain: Any) -> dict[str, Any] | None:
    """The plan in force (the latest ``plan`` entry's data), or None if none was locked."""
    chain = session_or_chain if isinstance(session_or_chain, Chain) else lab_book(session_or_chain)
    plans = [e for e in chain.entries if e.get("event") == "plan"]
    return plans[-1]["data"] if plans else None


def lock_plan(session: Any, specifications: Sequence[Any], *, data: Sequence[str] = (), sample: str = "",
              exclusions: str = "", hypotheses: Sequence[str] = (), primary_outcomes: Sequence[str] = (),
              data_request: str = "", reason: str = "") -> dict[str, Any]:
    """Lock the analysis plan: the model specifications (steps' code, or pymetrix commands), the datasets they use
    (``data``, by name), the sample and exclusion rules, the hypotheses tested and the primary outcomes. Locking again
    changes the plan and needs ``reason``."""
    from .brief import _append

    specs = [_code(s).strip() for s in specifications]
    if not specs:
        raise ValueError("a plan needs at least one specification")
    changing = current_plan(session) is not None
    if changing and not reason:
        raise ValueError("a plan is already locked: changing it needs its reason (reason=...)")
    plan: dict[str, Any] = {"specifications": specs, "data": list(data), "sample": sample, "exclusions": exclusions,
                            "hypotheses": list(hypotheses), "primary_outcomes": list(primary_outcomes),
                            "data_request": data_request}
    plan = {k: v for k, v in plan.items() if v not in ("", [])}
    plan["plan_hash"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode("utf-8")).hexdigest()
    if changing:
        plan["reason"] = reason
    return _append(session, "plan", plan)


def plan_label(chain: Chain, code: str) -> str | None:
    """How a step run now relates to the plan in force: "planned", "exploratory", or None when no plan is locked."""
    plan = current_plan(chain)
    if plan is None:
        return None
    return "planned" if normalise(code) in {normalise(s) for s in plan["specifications"]} else "exploratory"


def plan_report(chain: Chain) -> tuple[list[str], list[str], list[str]]:
    """The deviation report: (warnings, notes, checked)."""
    warn: list[str] = []
    notes: list[str] = []
    checked: list[str] = []
    entries = [e for e in chain.entries if "unreadable" not in e]
    plans = [e for e in entries if e.get("event") == "plan"]
    if not plans:
        return warn, notes, checked
    first = plans[0]["seq"]
    steps = [e for e in entries if e.get("event") == "step"]
    planned_data = set(plans[0]["data"].get("data", []))
    early = sorted({(name, e["seq"]) for e in steps if e["seq"] < first for name in e.get("data", {}).get("reads", {})
                    if name in planned_data}, key=lambda x: x[1])
    if early:
        name, at = early[0]
        warn.append(f"the planned data {name} were read at entry {at}, before the plan was locked at entry {first}")
    elif planned_data:
        checked.append(f"the plan: locked at entry {first}, before its data were read")
    else:
        checked.append(f"the plan: locked at entry {first}")
    for change in plans[1:]:
        done = sum(1 for e in steps if first < e["seq"] < change["seq"])
        warn.append(f"the plan was changed at entry {change['seq']}, after {done} step(s) since it was locked: "
                    f"{change['data'].get('reason', '')}")
    latest = plans[-1]["data"]
    after = [e for e in steps if e["seq"] > first]
    run = {normalise(e["data"].get("code", "")) for e in after}
    missing = [s for s in latest["specifications"] if normalise(s) not in run]
    if missing:
        warn.append(f"{len(missing)} planned specification(s) not run: " + "; ".join(missing))
    else:
        checked.append(f"every planned specification was run ({len(latest['specifications'])})")
    exploratory = sum(1 for e in after if e["data"].get("plan") == "exploratory")
    if exploratory:
        notes.append(f"{exploratory} step(s) since the plan was locked are exploratory: not in the plan")
    return warn, notes, checked
