"""An EViews program split into statements, each with its lines, its command and its parts.

The splitting is the core EViews reader's (``eviews_blank_comments`` and ``eviews_statements`` in
``pymetrix.extension``): comments become spaces, so every offset stays, and a line ending in `` _`` continues on the
next. A statement is then read into its command word, its options in brackets (``ls(cov=white)``), an object and
method (``x.label(d)``), or an assignment (``y = ...``, ``!i = 1``, ``%s = "a"``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pymetrix.extension import eviews_blank_comments, eviews_statements

__all__ = ["Program", "Statement", "parse_program"]


@dataclass(frozen=True)
class Statement:
    """One statement: ``text`` as written (comments blanked, continued lines joined), from offset ``at``, on lines
    ``line`` to ``last_line`` (from 1). ``kind`` is ``"command"`` (``smpl 1990q1 2000q4``), ``"method"``
    (``x.label(d) GDP``, with ``obj`` and ``method``) or ``"assign"`` (``y = x``, with ``lhs``). ``word`` is the
    command or method in lower case, ``options`` the text in brackets right after it, and ``rest`` (from offset
    ``rest_at``) what follows: the arguments, or an assignment's right-hand side."""

    text: str
    at: int
    line: int
    last_line: int
    kind: str
    word: str = ""
    options: str | None = None
    obj: str = ""
    lhs: str = ""
    rest: str = ""
    rest_at: int = 0


@dataclass(frozen=True)
class Program:
    """A program read: its ``source``, its lines, the statements, and each line's comment (``'`` to the end of the
    line), by line number."""

    source: str
    lines: tuple[str, ...]
    statements: tuple[Statement, ...]
    comments: dict[int, str]


_WORD = re.compile(r"\s*([A-Za-z_@][\w@]*|[!%]\w+|\{[^}]*\}\w*)")
_ASSIGN = re.compile(r"\s*([!%]?[A-Za-z_{][\w{}%!]*(?:\([^()=]*\))?)\s*=(?!=)")


def _options(text: str, pos: int) -> tuple[str | None, int]:
    """Options in brackets starting at ``pos`` (no space before them): the text inside and the position after."""
    if pos >= len(text) or text[pos] != "(":
        return None, pos
    depth = 0
    for i in range(pos, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[pos + 1:i], i + 1
    return text[pos + 1:], len(text)


def _statement(text: str, at: int, line: int, last_line: int) -> Statement:
    lead = len(text) - len(text.lstrip())
    body = text.strip()
    base = at + lead
    m = _ASSIGN.match(body)
    word = _WORD.match(body)
    first = word.group(1).lower() if word else ""
    if m and not (first in _NOT_ASSIGNMENT and word is not None and word.end() < m.end(1)):
        rest_at = m.end()
        rest = body[rest_at:]
        return Statement(text, at, line, last_line, "assign", lhs=m.group(1).strip(), rest=rest.strip(),
                         rest_at=base + rest_at + (len(rest) - len(rest.lstrip())))
    if word is None:
        return Statement(text, at, line, last_line, "command", rest=body, rest_at=base)
    name = word.group(1)
    pos = word.end()
    if pos < len(body) and body[pos] == "." and not name.startswith(("!", "%")):
        method = re.match(r"\.(@?\w+)", body[pos:])
        if method:
            options, after = _options(body, pos + method.end())
            rest = body[after:]
            return Statement(text, at, line, last_line, "method", word=method.group(1).lower(), options=options, obj=name,
                             rest=rest.strip(), rest_at=base + after + (len(rest) - len(rest.lstrip())))
    options, after = _options(body, pos)
    rest = body[after:]
    return Statement(text, at, line, last_line, "command", word=name.lower(), options=options, rest=rest.strip(),
                     rest_at=base + after + (len(rest) - len(rest.lstrip())))


# commands whose arguments may hold an equals sign: never read as assignments
_NOT_ASSIGNMENT = frozenset({"for", "if", "while", "smpl", "sample", "call", "subroutine", "series", "genr", "frml", "scalar",
                             "string", "alpha", "vector", "matrix", "sym", "coef", "equation", "model", "group", "wfcreate",
                             "pagecreate", "wfopen", "copy", "freeze", "graph", "table", "statusline", "include"})


def parse_program(source: str) -> Program:
    """``source`` split into statements; see the module's docstring."""
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    clean = eviews_blank_comments(source)
    lines = tuple(source.split("\n"))
    comments: dict[int, str] = {}
    for number, (raw, blanked) in enumerate(zip(lines, clean.split("\n")), start=1):
        cut = len(blanked.rstrip())
        tail = raw[cut:].strip()
        if tail.startswith("'"):
            comments[number] = tail[1:].strip()
    clean_lines = clean.split("\n")
    statements = []
    for text, at in eviews_statements(clean):
        line = source.count("\n", 0, at) + 1
        last = line
        while last < len(clean_lines) and re.search(r"(^|\s)_\s*$", clean_lines[last - 1]):
            last += 1
        statements.append(_statement(text, at, line, last))
    return Program(source, lines, tuple(statements), comments)
