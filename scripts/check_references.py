"""Check the help pages' references against Crossref: run in CI, where the network is open.

    python scripts/check_references.py            # check DOIs; suggest DOIs for references without one
    python scripts/check_references.py --strict   # also fail when a reference has no DOI
    python scripts/check_references.py --metadata # also print Crossref's year, journal, volume, issue and pages

Every reference of a registered method, test, filter, dialog or help topic is checked. A reference with a DOI
(``doi:10.xxxx/...``) must resolve at Crossref to a work whose title matches the citation, else the run fails: a DOI
typed wrong, or one that points at another paper, is caught before it reaches a help page. For a reference without
a DOI, Crossref's best match is listed with a score, as a suggestion to check and add by hand; nothing is added
automatically. A JSTOR stable ID (``jstor:1827924``), given for an article with no DOI, is checked only for its form:
JSTOR turns away automated requests, so whether it opens is not checked. The help renders a DOI as a doi.org link and a reference without one as a search link
(``pymetrix.help.reference_markdown``).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pymetrix import help as hp
from pymetrix.plugins import registry

API = "https://api.crossref.org/works"
AGENT = "pymetrix-reference-check/1.0 (https://github.com/GeoffHeenan/pymetrix)"
SOFTWARE = ("10.32614/cran.",)  # CRAN packages, which the help does not cite (CONTRIBUTING.md, Citations); the R
# Journal shares the prefix 10.32614 but publishes papers, so only the CRAN part is left out
STOP = {"the", "a", "an", "of", "and", "in", "for", "on", "to", "with", "by", "its", "from", "at", "as", "or", "is"}


def references() -> dict[str, list[str]]:
    """Each reference, with the pages that cite it."""
    hp._load_all()
    found: dict[str, list[str]] = {}
    for group in hp.HELP_GROUPS:
        for entry in registry(group).entries():
            for ref in getattr(entry.obj, "references", ()) or ():
                found.setdefault(ref, []).append(f"{group}/{entry.name}")
    for topic in hp.topics():
        for ref in topic.references:
            found.setdefault(ref, []).append(f"topic/{topic.id}")
    return found


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2}


def title_match(citation: str, title: str) -> float:
    """The share of the work's title words that the citation contains (1 = every word)."""
    t = words(title)
    return len(t & words(citation)) / len(t) if t else 0.0


def get(url: str) -> dict | None:
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            time.sleep(2 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Crossref did not answer: {url}")


def metadata(citation: str, work: dict, score: float) -> str:
    """One line, ``META {json}``, with the bibliographic fields of the work Crossref matched: for completing citations
    (issue numbers, pages) from Crossref rather than by hand."""
    issued = (work.get("issued") or {}).get("date-parts") or [[None]]
    return "META " + json.dumps({"citation": citation, "doi": work.get("DOI", ""), "score": round(score, 3),
                                 "title": " ".join(work.get("title", [])), "year": issued[0][0],
                                 "journal": " ".join(work.get("container-title", [])), "volume": work.get("volume", ""),
                                 "issue": work.get("issue", ""), "page": work.get("page", "")}, ensure_ascii=False)


def main(argv: list[str]) -> int:
    strict = "--strict" in argv
    meta = "--metadata" in argv
    refs = references()
    problems, suggestions, checked = [], [], 0
    for citation, pages in sorted(refs.items()):
        found = hp._DOI.search(citation)
        if found:
            doi = found.group(1)
            work = get(f"{API}/{urllib.parse.quote(doi)}")
            title = " ".join((work or {}).get("message", {}).get("title", []))
            score = title_match(citation, title) if title else 0.0
            checked += 1
            if work is None:
                problems.append(f"| {doi} | does not resolve at Crossref | {', '.join(pages)} |")
            elif score < 0.6:
                problems.append(f"| {doi} | is \"{title}\" (title match {score:.0%}) | {', '.join(pages)} |")
            if meta and work is not None:
                print(metadata(citation, work["message"], score))
            continue
        if hp._URL.search(citation):
            continue
        if "jstor:" in citation.lower():
            if not hp._JSTOR.search(citation):
                problems.append(f"| {citation.lower().split('jstor:', 1)[1].split()[0]} | is not a JSTOR stable ID | "
                                f"{', '.join(pages)} |")
            continue
        result = get(f"{API}?rows=3&query.bibliographic={urllib.parse.quote(citation)}")
        items = (result or {}).get("message", {}).get("items", [])
        items = [i for i in items if not i.get("DOI", "").startswith(SOFTWARE)]  # help cites papers, not packages
        if items:
            title = " ".join(items[0].get("title", []))
            suggestions.append((title_match(citation, title), items[0].get("DOI", ""), title, citation))
            if meta:
                print(metadata(citation, items[0], suggestions[-1][0]))
        time.sleep(0.2)  # Crossref asks for restraint
    stable = sum(1 for c in refs if not hp._DOI.search(c) and hp._JSTOR.search(c))
    lines = [f"# Reference check: {len(refs)} references, {checked} with a DOI, {stable} with a JSTOR stable ID", ""]
    if problems:
        lines += ["## DOIs that fail", "", "| DOI or ID | problem | cited by |", "|---|---|---|", *problems, ""]
    if suggestions:
        lines += ["## Suggested DOIs (check each before adding it as `doi:...`)", "", "| match | DOI | Crossref title | citation |",
                  "|---|---|---|---|"]
        lines += [f"| {s:.0%} | {d} | {t} | {c} |" for s, d, t, c in sorted(suggestions, reverse=True)]
    report = "\n".join(lines)
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(report, encoding="utf-8")
    return 1 if problems or (strict and suggestions) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
