"""SDMX dataflows grouped into families by their names, for File → Browse series….

A provider such as the IMF lists hundreds of dataflows, unsorted, many of them parts or releases of a larger dataset:
"Monetary and Financial Statistics (MFS), Interest Rates", "Labor Statistics (LS) 2026 January". :func:`group_dataflows`
sorts them and groups them by **family**, the leading name with its parenthesised abbreviation ("Monetary and
Financial Statistics (MFS)"). What follows the family is the dataflow's label within it: the text after a comma
("Interest Rates"), or a release date ("2026 January"). The rules, a pure function of the names:

- The family ends at the first parenthesised group, after some words, whose text starts with a capital letter or a
  digit (an abbreviation: "(MFS)", "(GFS, COFOG)", "(DOTS (legacy))"); parentheses that do not ("(by partner
  country)") stay part of the family's name. Commas and parentheses inside the group belong to it.
- A family of two or more dataflows is a folder; a family of one, and a name with no family, is a row of its own,
  shown with its full name. A list in which no family has two dataflows is therefore a flat list.
- Folders and rows are sorted by name, ignoring case. Within a family the dataflow with the family's name alone comes
  first, then the others by label (ignoring case and any release date at its end), each label's releases newest
  first. Release dates are read ("2026 January", "January 2026", "2026-01", "2026Q1", "2026") and compared as dates.

Nothing here touches Qt or the network.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = ["DataflowFamily", "GroupedDataflow", "group_dataflows", "release_date", "split_family"]

_MONTHS = {name: i for i, names in enumerate(
    (("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"), ("may",), ("june", "jun"),
     ("july", "jul"), ("august", "aug"), ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"),
     ("december", "dec")), start=1) for name in names}
_MONTH = "(?P<month>" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_YEAR = r"(?P<year>(?:19|20)\d\d)"
_DATES = [re.compile(p, re.IGNORECASE) for p in (
    rf"(?:^|[\s,;:(\-–—])(?P<day>\d{{1,2}})\s+{_MONTH}\s+{_YEAR}\)?$",       # 15 January 2026
    rf"(?:^|[\s,;:(\-–—]){_YEAR}\s+{_MONTH}(?:\s+(?P<day>\d{{1,2}}))?\)?$",  # 2026 January, 2026 January 15
    rf"(?:^|[\s,;:(\-–—]){_MONTH}\s+{_YEAR}\)?$",                            # January 2026
    rf"(?:^|[\s,;:(\-–—]){_YEAR}-(?P<mm>0?[1-9]|1[0-2])\)?$",                  # 2026-01
    rf"(?:^|[\s,;:(\-–—]){_YEAR}\s*-?\s*Q(?P<quarter>[1-4])\)?$",              # 2026Q1, 2026 Q1
    rf"(?:^|[\s,;:(\-–—]){_YEAR}\)?$",                                         # 2026
)]
_LEAD = " \t,;:-–—"


def release_date(text: str) -> tuple[tuple[int, int, int], str] | None:
    """The release date at the end of ``text`` as (year, month, day), with the text before it; None if there is none.
    A year alone is (year, 0, 0) and a quarter its last month, so ``max`` is the newest release."""
    text = " ".join(text.split())
    for pattern in _DATES:
        found = pattern.search(text)
        if found is None:
            continue
        parts = found.groupdict()
        year = int(parts["year"])
        if parts.get("month"):
            month = _MONTHS[parts["month"].lower()]
        elif parts.get("mm"):
            month = int(parts["mm"])
        elif parts.get("quarter"):
            month = 3 * int(parts["quarter"])
        else:
            month = 0
        day = int(parts["day"]) if parts.get("day") else 0
        if day > 31:
            continue
        start = found.start() + (1 if found.group(0)[:1] in _LEAD + "(" else 0)
        return (year, month, day), text[:start].rstrip(_LEAD)
    return None


def _closing(name: str, start: int) -> int:
    """The index of the parenthesis that closes the one at ``start``; -1 if it is never closed."""
    depth = 0
    for i in range(start, len(name)):
        if name[i] == "(":
            depth += 1
        elif name[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def split_family(name: str) -> tuple[str, str]:
    """``(family, rest)`` of a dataflow's name: the leading name with its parenthesised abbreviation, and what follows
    it (without the comma or other separator); ``("", name)`` when the name has no such abbreviation."""
    name = " ".join(name.split())
    i = name.find("(")
    while i >= 0:
        end = _closing(name, i)
        if end < 0:
            break
        inner = name[i + 1:end].strip()
        if name[:i].strip() and (inner[:1].isupper() or inner[:1].isdigit()):
            return name[:end + 1], name[end + 1:].strip(_LEAD)
        i = name.find("(", end + 1)
    return "", name


@dataclass(frozen=True)
class GroupedDataflow:
    """One dataflow: ``code`` as the browser keeps it (``IMF/IMF.STA,CPI``), its full ``name``, the ``label`` it has
    in its family's folder (its full name when it is a row of its own), and its ``release`` date, if any."""

    code: str
    name: str
    label: str
    family: str = ""
    release: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class DataflowFamily:
    """A folder of dataflows (two or more) or, with one, a row of its own: ``title`` is the family's name, or the
    dataflow's full name for a row of its own."""

    title: str
    flows: tuple[GroupedDataflow, ...]

    @property
    def folder(self) -> bool:
        return len(self.flows) > 1


def _member_key(flow: GroupedDataflow) -> tuple:
    if flow.label.casefold() == flow.family.casefold():  # the family's name alone: the base dataflow
        return (0, "", 0, (0, 0, 0), flow.code.casefold())
    dated = release_date(flow.label)
    stem = dated[1] if dated else flow.label
    newest_first = tuple(-x for x in dated[0]) if dated else (0, 0, 0)
    return (1, stem.casefold(), 1 if dated else 0, newest_first, flow.code.casefold())


def group_dataflows(flows: Iterable[tuple[str, str]]) -> list[DataflowFamily]:
    """Group ``(code, name)`` pairs into families, sorted as the module's docstring says. Every dataflow is in the
    result once; a list with no family of two or more comes back as rows of one, sorted by name (a flat list). A
    dataflow with no name is shown by its code."""
    families: dict[tuple[str, str], list[tuple[str, str, str, str]]] = {}
    for code, name in flows:
        name = " ".join(str(name or code).split())
        family, rest = split_family(name)
        key = ("family", family.casefold()) if family else ("name", name.casefold())  # a name alone never joins a family
        families.setdefault(key, []).append((code, name, family, rest))
    out = []
    for (kind, _), members in families.items():
        if kind == "name" or len(members) == 1:
            for code, name, family, _rest in members:
                out.append(DataflowFamily(name, (GroupedDataflow(code, name, name, family, _release(name, family)),)))
            continue
        title = members[0][2]  # as the first of the family is written
        grouped = [GroupedDataflow(code, name, rest or family, title, _release(name, family)) for code, name, family, rest in members]
        out.append(DataflowFamily(title, tuple(sorted(grouped, key=_member_key))))
    return sorted(out, key=lambda f: (f.title.casefold(), f.title, f.flows[0].code.casefold()))


def _release(name: str, family: str) -> tuple[int, int, int] | None:
    rest = name[len(family):] if family and name.startswith(family) else name
    dated = release_date(rest)
    return dated[0] if dated else None
