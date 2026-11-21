"""How a result's summary is laid out as HTML (notebooks, ``to_html``, the HTML report).

A table's notes are of two kinds. "Label: value" lines with a short value (``R-squared: 0.9999``) are statistics:
they are grouped under small headings (Estimation, Fit, Information criteria, Residual checks, Other; the mapping is
``pymetrix.wording.stat_group``), each group a two-column list, labels left and muted, values right-aligned. The
rest (warnings, explanations) stay sentences, after the statistics. The plain text (``str(table)``), CSV and Markdown
keep the notes as they are written.

The HTML uses inline styles only (a notebook's page keeps no ``<style>`` of ours) and degrades in Qt's rich text,
which ignores flex layout: the groups then stack. Colours are the neutral greys below, which read on light and dark
backgrounds; the desktop app draws the same groups with widgets and its palette (``pymetrix.ui.stats_panel``).
"""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

MUTED_COLOUR = "#808080"  # labels and group headings: mid grey, legible on white and on a dark background
STRIPE_COLOUR = "rgba(128, 128, 128, 0.10)"  # every other row of a table, a light shading on either theme
RULE_COLOUR = "rgba(128, 128, 128, 0.55)"  # the line under a table's header row

_LABEL = re.compile(r"[A-Za-z][A-Za-z0-9 .'()\[\]/²_+\-–]*")  # an unknown label: words, no commas, quotes or '='
_NUMBER = re.compile(r"[<>≈~±+\-−]?\s?(\d[\d,]*\.?\d*|\.\d+)(e[+\-]?\d+)?\s?%?(\s*\([^()]*\))?|nan|-?inf", re.IGNORECASE)
_SENTENCE_BREAK = re.compile(r"[.!?]\s+[A-Z]")  # "...90%. Standard errors: ..." holds two sentences

KNOWN_VALUE_MAX = 90  # characters: a known label's value (Method, Sample) may be this long and still be a statistic
SHORT_VALUE_MAX = 24  # an unknown label's value, when not a number: "chi2(4)", "Cholesky", "yes"


@dataclass(frozen=True)
class StatGroup:
    title: str
    items: tuple[tuple[str, str], ...]  # (label, value), in the notes' order


@dataclass(frozen=True)
class SplitNotes:
    groups: tuple[StatGroup, ...]  # in the order of ``wording.STAT_GROUPS``, only those with statistics
    prose: tuple[str, ...]  # the other notes, in their order

    @property
    def headed(self) -> bool:
        """Whether the groups show their headings: only when there is more than one (one heading says nothing)."""
        return len(self.groups) > 1


def is_number_text(value: str) -> bool:
    """Whether a statistic's value reads as a number ("0.9999", "-121.74", "<0.0001", "12%", "3.2 (0.04)")."""
    return bool(_NUMBER.fullmatch(value.strip()))


def parse_stat(note: str) -> tuple[str, str] | None:
    """``(label, value)`` when a note is a statistic ("Label: value" with a short value), else None (prose)."""
    from .wording import STAT_PROSE_LABELS, is_known_stat

    text = str(note).strip()
    if "\n" in text or ":" not in text:
        return None
    label, _, value = text.partition(":")
    label, value = label.strip(), value.strip()
    if not label or not value or label.lower() in STAT_PROSE_LABELS:
        return None
    number = is_number_text(value)
    if not number and (_SENTENCE_BREAK.search(value) or value.endswith((".", ":"))):
        return None
    if is_known_stat(label):
        return (label, value) if len(label) <= 50 and len(value) <= KNOWN_VALUE_MAX else None
    if not _LABEL.fullmatch(label) or len(label) > 40:
        return None
    if number or (len(value) <= SHORT_VALUE_MAX and value.count(" ") <= 3 and "=" not in value):  # not code
        return label, value
    return None


def split_notes(notes: Sequence[Any]) -> SplitNotes:
    """A table's notes as grouped statistics and prose. Every note lands in one or the other, none is dropped."""
    from .wording import STAT_GROUPS, stat_group

    grouped: dict[str, list[tuple[str, str]]] = {g: [] for g in STAT_GROUPS}
    prose = []
    for note in notes:
        stat = parse_stat(str(note))
        if stat is None:
            prose.append(str(note))
        else:
            grouped.setdefault(stat_group(stat[0]), []).append(stat)
    groups = tuple(StatGroup(g, tuple(items)) for g, items in grouped.items() if items)
    return SplitNotes(groups, tuple(prose))


def notes_html(notes: Sequence[Any]) -> str:
    """The notes as HTML: the statistics in groups side by side where there is room (two or three), stacked where
    there is not, then the prose. Empty for no notes."""
    split = split_notes(notes)
    out = []
    if split.groups:
        out.append('<div style="display:flex;flex-wrap:wrap;column-gap:2.5em;row-gap:0.4em;max-width:80em;'
                   'margin:0.8em 0 0.3em 0">')
        for group in split.groups:
            out.append('<div style="flex:0 1 auto;min-width:16em;max-width:26em;margin:0 0 0.5em 0">')
            if split.headed:
                out.append(f'<div style="font-size:85%;font-weight:600;color:{MUTED_COLOUR};margin:0.4em 0 0.2em 0">'
                           f'{html.escape(group.title)}</div>')
            out.append('<table cellspacing="0" cellpadding="0" style="border-collapse:collapse;border:none;'
                       'margin:0;width:100%">')
            for label, value in group.items:
                side = "right" if is_number_text(value) else "left"  # numbers line up on the right, words from the left
                out.append('<tr style="background:transparent">'
                           f'<td style="border:none;padding:1px 1.5em 1px 0;text-align:left;vertical-align:top;white-space:nowrap;'
                           f'color:{MUTED_COLOUR}">{html.escape(label)}</td>'
                           f'<td align="{side}" style="border:none;padding:1px 0;text-align:{side};vertical-align:top;'
                           f'font-variant-numeric:tabular-nums">{html.escape(value)}</td></tr>')
            out.append("</table></div>")
        out.append("</div>")
    out.extend(f'<p style="margin:0.3em 0">{html.escape(p)}</p>' for p in split.prose)
    return "\n".join(out)


def _is_number(v: Any) -> bool:
    import numpy as np

    return isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_))


def table_html(columns: Sequence[str], rows: Sequence[Sequence[str]], numeric: Sequence[bool]) -> str:
    """A data table as HTML: as wide as its content, padded cells, a bold header row over a thin rule, numbers
    right-aligned in tabular figures, every other row lightly shaded. ``rows`` are the cells as text already;
    ``numeric`` says which columns hold numbers."""
    pad = "padding:3px 10px"
    out = ['<table cellspacing="0" cellpadding="3" style="border-collapse:collapse;border:none;margin:0.3em 0 0.4em 0">',
           "<thead><tr>"]
    for j, c in enumerate(columns):
        side = "right" if numeric[j] else "left"
        out.append(f'<th align="{side}" style="{pad};text-align:{side};font-weight:bold;border:none;'
                   f'border-bottom:1px solid {RULE_COLOUR}">{html.escape(c)}</th>')
    out.append("</tr></thead><tbody>")
    for i, row in enumerate(rows):
        shade = f"background-color:{STRIPE_COLOUR};" if i % 2 else "background:transparent;"
        out.append(f'<tr style="{shade}">')
        for j, cell in enumerate(row):
            side = "right" if numeric[j] else "left"
            out.append(f'<td align="{side}" style="{pad};{shade}text-align:{side};border:none;'
                       f'{"font-variant-numeric:tabular-nums;white-space:nowrap" if numeric[j] else ""}">'
                       f"{html.escape(cell)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


_DEGREES = re.compile(r"\(?\s*\d+(\.\d+)?(\s*,\s*\d+(\.\d+)?)*\s*\)?")  # degrees of freedom as text: "4", "(2, 96)"


def is_numberish(v: Any) -> bool:
    """A number, or text that reads as one: a test's df written "4" or "(2, 96)", "<0.0001", "12%"."""
    if _is_number(v):
        return True
    return isinstance(v, str) and bool(v.strip()) and (is_number_text(v) or bool(_DEGREES.fullmatch(v.strip())))


def numeric_columns(rows: Sequence[Sequence[Any]], width: int) -> list[bool]:
    """Which of ``width`` columns hold only numbers, or text that reads as numbers (missing values aside): those are
    right-aligned. A test's value column (statistic, df as text, p-value) is one."""
    flags = []
    for j in range(width):
        present = [r[j] for r in rows if j < len(r) and r[j] is not None and not (isinstance(r[j], float) and r[j] != r[j])]
        flags.append(bool(present) and all(is_numberish(v) for v in present))
    return flags
