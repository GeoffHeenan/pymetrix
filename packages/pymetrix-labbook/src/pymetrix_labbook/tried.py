"""Counting what was tried: how many specifications of each outcome were fitted, and their specification curve.

The design is pymetrix's ``docs/design/research-audit.md``, layer 3. Forking paths are disclosed from the lab book,
not from the researcher's memory::

    from pymetrix_labbook import specification_curve, tried

    tried(session, "fit")          # 'fit is 1 of 23 specifications fitted for c (31 fits, lab book entries 2-58)'
    specification_curve(session, "c", "y", highlight="fit").savefig("curve.png")

Every fit counts: kept, undone, removed or failed, since the point is what was tried, not what was kept. Two fits are
the same specification when their code is the same apart from the result's name, and their sample is the same.
"""

from __future__ import annotations

from typing import Any

from .chain import Chain
from .plan import normalise
from .recorder import lab_book

__all__ = ["fits", "outcome_counts", "report_note", "specification_curve", "tried"]


def _chain(session_or_chain: Any) -> Chain:
    return session_or_chain if isinstance(session_or_chain, Chain) else lab_book(session_or_chain)


def fits(session_or_chain: Any, outcome: str | None = None) -> list[dict[str, Any]]:
    """Every fit the lab book recorded (of ``outcome``, if given): its entry, name, outcome, sample, estimates, the
    specification it belongs to (``spec``, the code without the result's name, and the sample) and whether it failed."""
    out = []
    for entry in _chain(session_or_chain).entries:
        if entry.get("event") not in ("step", "failed"):
            continue
        data = entry.get("data", {})
        for name, info in data.get("writes", {}).items():
            if "outcome" not in info or (outcome is not None and info["outcome"] != outcome):
                continue
            out.append({"seq": entry["seq"], "name": name, "outcome": info["outcome"], "sample": info.get("sample", ""),
                        "estimates": info.get("estimates", {}), "failed": entry.get("event") == "failed",
                        "spec": (normalise(data.get("code", "")), info.get("sample", ""))})
    return out


def outcome_counts(session_or_chain: Any) -> dict[str, tuple[int, int]]:
    """Outcome -> (distinct specifications, fits)."""
    counts: dict[str, tuple[set, int]] = {}
    for f in fits(session_or_chain):
        specs, n = counts.get(f["outcome"], (set(), 0))
        specs.add(f["spec"])
        counts[f["outcome"]] = (specs, n + 1)
    return {k: (len(v[0]), v[1]) for k, v in counts.items()}


def tried(session: Any, name: str) -> str:
    """One line for a report: where the result ``name`` stands among the specifications fitted for its outcome."""
    mine = [f for f in fits(session) if f["name"] == name and not f["failed"]]
    if not mine:
        raise ValueError(f"the lab book has no fit named {name!r} with an outcome")
    last = mine[-1]
    every = fits(session, last["outcome"])
    specs = list(dict.fromkeys(f["spec"] for f in every))
    seqs = [f["seq"] for f in every]
    span = f"lab book entries {min(seqs)}-{max(seqs)}" if len(seqs) > 1 else f"lab book entry {seqs[0]}"
    return (f"{name} is 1 of {len(specs)} specification{'s' if len(specs) != 1 else ''} fitted for {last['outcome']} "
            f"({len(every)} fit{'s' if len(every) != 1 else ''}, {span})")


def specification_curve(session_or_chain: Any, outcome: str, term: str, highlight: str | None = None) -> Any:
    """A matplotlib figure: the estimate of ``term`` in each specification fitted for ``outcome`` (its latest fit),
    sorted, with 95% intervals; ``highlight`` (a result's name) marks the one reported."""
    from matplotlib.figure import Figure  # no pyplot: nothing global, nothing to close

    latest: dict[Any, dict[str, Any]] = {}
    for f in fits(session_or_chain, outcome):
        if not f["failed"] and term in f["estimates"]:
            latest[f["spec"]] = f
    if not latest:
        raise ValueError(f"no fit of {outcome!r} has a coefficient {term!r}")
    rows = sorted(latest.values(), key=lambda f: f["estimates"][term][0])
    fig = Figure(figsize=(7, 3.6))
    ax = fig.add_subplot()
    for i, f in enumerate(rows):
        value, se = f["estimates"][term]
        mark = highlight is not None and f["name"] == highlight
        colour = "#e07a2f" if mark else "#1b2a2f"
        if se is not None:
            ax.plot([i, i], [value - 1.96 * se, value + 1.96 * se], color=colour, lw=1.5, alpha=0.6)
        ax.plot(i, value, "o", color=colour, ms=7 if mark else 4, zorder=3)
    ax.axhline(0, color="#9aa7a9", lw=0.8)
    ax.set_xticks([])
    ax.set_xlabel(f"{len(rows)} specification(s) of {outcome}, sorted by the estimate")
    ax.set_ylabel(f"coefficient on {term}")
    ax.set_title(f"Specification curve: {term} in models of {outcome}" + (f" ({highlight} highlighted)" if highlight else ""))
    fig.tight_layout()
    return fig


def report_note(session: Any, name: str) -> str | None:
    """The line an exported report carries for the result ``name`` (pymetrix's ``ReportNote``, extension API 1.6):
    ``tried``'s, or None for a result the lab book has no fit of."""
    try:
        return tried(session, name)
    except ValueError:
        return None
