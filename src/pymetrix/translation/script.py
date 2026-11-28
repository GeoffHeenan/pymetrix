"""The readable script: each line of the program a comment above what it became, blocks as Python blocks (loops stay
loops, an ``if`` block an ``if``), functions defined first, and the lines of an included program in line, under its
name.

``render`` writes the script's body from a front end's outcomes (``Outcome``: a statement, its status, reason and
ops, a block's body), the programs read (``Source``: their lines, statements and comments) and the functions; the
front end writes the head (a docstring and the imports: ``pymetrix_imports`` finds pymetrix's names the body uses).
An outcome's ``kind`` may be ``"include"`` (an included program: its outcomes as ``children``, its name as
``included_file``) or ``"subroutine"`` (a function, defined above).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .ir import Cmd, Code, Def, For, If, Invoke, Jump, Step, While, step_code

__all__ = ["body_lines", "op_code", "pymetrix_imports", "render", "source_comment"]

_PAD = "    "


def op_code(op: Any, returns: tuple[str, ...] = ()) -> str:
    """One op as Python: a statement, or a block's first line (``for i in range(1, 4):``)."""
    if isinstance(op, Code):
        return op.code
    if isinstance(op, Cmd):
        return op.script or op.command.to_code()
    if isinstance(op, Step):
        return step_code(op)
    if isinstance(op, For):
        return f"for {op.var} in {op.iterable}:"
    if isinstance(op, While):
        return f"while {op.cond}:"
    if isinstance(op, If):
        return f"if {op.cond}:"
    if isinstance(op, Def):
        return f"def {op.name}({', '.join(op.params)}):"
    if isinstance(op, Invoke):
        call = f"{op.name}({', '.join(op.args)})"
        if not op.targets or all(t is None for t in op.targets):
            return call
        return f"{', '.join(t or '_' for t in op.targets)} = {call}"
    if isinstance(op, Jump):
        if op.kind == "break":
            return "break"
        if op.kind == "return":
            return f"return {', '.join(returns)}" if returns else "return"
        return "raise SystemExit  # stop: the program ends here"
    raise TypeError(f"no Python for {type(op).__name__}")


def body_lines(body: tuple[Any, ...], indent: int, returns: tuple[str, ...] = ()) -> list[str]:
    """Ops (``(line, op, file)`` entries) as lines of Python, without the program's lines as comments."""
    out: list[str] = []
    pad = _PAD * indent
    for _, op, _ in body:
        if isinstance(op, Def):
            out.append(pad + op_code(op))
            if op.shared:
                out.append(pad + _PAD + f"global {', '.join(op.shared)}")
            out += body_lines(op.body, indent + 1, op.returned) or [pad + _PAD + "pass"]
            if op.returned:
                out.append(pad + _PAD + op_code(Jump("return"), op.returned))
            continue
        for line in op_code(op, returns).split("\n"):
            out.append(pad + line)
        if isinstance(op, (For, While, If)):
            out += body_lines(op.body, indent + 1, returns) or [pad + _PAD + "pass"]
            if isinstance(op, If) and op.orelse:
                out.append(pad + "else:")
                out += body_lines(op.orelse, indent + 1, returns)
    return out


class _Line(str):
    """A line of the script that stands for a line of a program: ``key`` is (file, line)."""

    key: tuple[str, int] | None = None


def _mark(text: str, file: str, line: int) -> str:
    out = _Line(text)
    out.key = (file, line)
    return out


def source_comment(line: int, text: str) -> str:
    """A statement of the program as the script's comment above what it became: ``# 5: series y = x``."""
    return f"# {line}: {' '.join(text.split())}"


class _Writer:
    """Writes outcomes as the script's lines, with the program's comment lines (those with no statement) where they
    are, a cursor per program file (an included program's lines are written in line)."""

    def __init__(self, programs: dict[str, Any], comment: Callable[[int, str], str] = source_comment) -> None:
        self.programs = programs
        self.comment = comment
        self.cursor: dict[str, int] = {}
        self.used: set[tuple[str, int]] = set()
        self.covered: dict[str, set[int]] = {}
        for file, p in programs.items():
            lines: set[int] = set()
            for st in p.statements:
                lines.update(range(st.line, st.last_line + 1))
            self.covered[file] = lines

    def pending(self, file: str, upto: int, pad: str, blanks: bool) -> list[str]:
        """The comment lines (and, at the top level, a blank line) between the cursor and line ``upto``."""
        p = self.programs.get(file)
        out: list[str] = []
        if p is None:
            return out
        start = self.cursor.get(file, 0)
        upto = min(upto, len(p.lines) + 1)
        blank = False
        for number in range(start + 1, upto):
            if (file, number) in self.used or number in self.covered[file]:
                continue
            raw = p.lines[number - 1] if number <= len(p.lines) else ""
            if not raw.strip():
                blank = True
                continue
            if number in p.comments:
                if blank and blanks and out and out[-1] != "":
                    out.append("")
                blank = False
                out.append(_mark(pad + f"# {p.comments[number]}", file, number))
                self.used.add((file, number))
        if blank and blanks:
            out.append("")
        self.cursor[file] = max(start, upto - 1)
        return out

    def _comment(self, o: Any, pad: str = "") -> str:
        return _mark(pad + self.comment(o.statement.line, o.statement.text), o.file, o.statement.line)

    def _own_comment(self, o: Any, pad: str) -> list[str]:
        p = self.programs.get(o.file)
        if p is not None and o.statement.line in p.comments:
            return [pad + f"# {p.comments[o.statement.line]}"]
        return []

    def _end(self, o: Any) -> None:
        last = o.closer.statement.last_line if getattr(o, "closer", None) is not None else o.statement.last_line
        self.cursor[o.file] = max(self.cursor.get(o.file, 0), last)

    def outcome(self, o: Any, indent: int, returns: tuple[str, ...]) -> list[str]:
        """An outcome as lines of the script, its body (a block's) inside it."""
        from .report import NOTE, TRANSLATED

        pad = _PAD * indent
        out: list[str] = self.pending(o.file, o.statement.line, pad, blanks=indent == 0)
        out += self._own_comment(o, pad)
        out.append(self._comment(o, pad))
        self.cursor[o.file] = max(self.cursor.get(o.file, 0), o.statement.last_line)
        kind = getattr(o, "kind", "")
        if kind == "include":
            shown = getattr(o, "included_file", "")
            if o.status in (TRANSLATED, NOTE):
                out.append(pad + f"#   the lines of {shown} follow (its subroutines are defined above)")
                for c in o.children:
                    out += self.outcome(c, indent, returns)
                out += self.pending(shown, 10 ** 9, pad, blanks=False)
                out.append(pad + f"#   the end of {shown}")
            else:
                out.append(pad + f"#   {o.status}: {o.reason}")
            return out
        if kind == "subroutine":
            out.append(pad + ("#   defined above, as a function" if o.status in (TRANSLATED, NOTE) else f"#   {o.status}: {o.reason}"))
            self._end(o)
            return out
        if o.status not in (TRANSLATED, NOTE):
            out.append(pad + f"#   {o.status}: {o.reason}")
            for c in getattr(o, "children", []):
                if c is not None:
                    out += self.outcome(c, indent, returns)
            self._end(o)
            return out
        if o.reason:
            out.append(pad + f"#   note: {o.reason}")
        for op in o.ops:
            for line in op_code(op, returns).split("\n"):
                out.append(pad + line)
        block = [op for op in o.ops if isinstance(op, (For, While, If))]
        if block:
            inner: list[str] = []
            for c in o.children:
                inner += self.outcome(c, indent + 1, returns)
            out += inner + ([] if _has_code(inner) else [pad + _PAD + "pass"])
            if isinstance(block[0], If) and (o.orelse or getattr(o, "else_outcome", None) is not None):
                out.append(pad + "else:")
                inner = []
                if o.else_outcome is not None:
                    inner += self.pending(o.file, o.else_outcome.statement.line, pad + _PAD, blanks=False)
                    inner.append(self._comment(o.else_outcome, pad + _PAD))
                    self.cursor[o.file] = max(self.cursor.get(o.file, 0), o.else_outcome.statement.line)
                for c in o.orelse:
                    inner += self.outcome(c, indent + 1, returns)
                out += inner + ([] if _has_code(inner) else [pad + _PAD + "pass"])
        if getattr(o, "closer", None) is not None:
            out += self.pending(o.file, o.closer.statement.line, pad + _PAD, blanks=False)
            out.append(self._comment(o.closer, pad))
        self._end(o)
        return out

    def function(self, o: Any, d: Def) -> list[str]:
        """A function's outcome as its definition, the program's lines as comments."""
        out: list[str] = []
        where = f" ({o.file})" if o.file else ""
        self.cursor[o.file] = o.statement.last_line
        out += self._own_comment(o, "")
        out.append(_mark(self._comment(o) + where, o.file, o.statement.line))
        if o.reason:
            out.append(f"#   note: {o.reason}")
        out.append(op_code(d))
        if d.shared:
            out.append(_PAD + f"global {', '.join(d.shared)}  # a subroutine that is not local shares the program's names")
        inner: list[str] = []
        for c in o.children:
            inner += self.outcome(c, 1, d.returned)
        out += inner
        if getattr(o, "closer", None) is not None:
            out += self.pending(o.file, o.closer.statement.line, _PAD, blanks=False)
        if d.returned:
            out.append(_PAD + op_code(Jump("return"), d.returned))
        elif not _has_code(inner):
            out.append(_PAD + "pass")
        if getattr(o, "closer", None) is not None:
            out.append(self._comment(o.closer, _PAD))
            for n in range(o.statement.line, o.closer.statement.line + 1):
                self.used.add((o.file, n))
        self.cursor[o.file] = 0
        return out


def _has_code(lines: list[str]) -> bool:
    return any(line.strip() and not line.lstrip().startswith("#") for line in lines)


def render(outcomes: list[Any], functions: list[tuple[Any, Def]], programs: dict[str, Any], top: list[str],
           comment: Callable[[int, str], str] = source_comment) -> tuple[str, dict[tuple[str, int], int]]:
    """The script's body: ``top`` (the arguments), the functions (``(outcome, Def)``), then the program's
    ``outcomes``; and the line of the body (from 0) that stands for each line of a program, by (file, line).
    ``programs`` are the programs read (``Source``), by file ('' the main one); ``comment(line, text)`` writes a
    statement as the comment above what it became."""
    w = _Writer(programs, comment)
    body: list[str] = list(top)
    if top:
        body.append("")
    for o, d in functions:
        body += w.function(o, d)
        body.append("")
    w.cursor = {}
    for o in outcomes:
        body += w.outcome(o, 0, ())
    body += w.pending("", 10 ** 9, "", blanks=False)
    lines: list[str] = []
    for item in body:  # one blank line at most between parts, none at the ends
        for piece in (item.split("\n") if not isinstance(item, _Line) else [item]):
            piece = piece if isinstance(piece, _Line) else piece
            if piece == "" and (not lines or lines[-1] == ""):
                continue
            lines.append(piece)
    while lines and lines[-1] == "":
        lines.pop()
    where: dict[tuple[str, int], int] = {}
    for i, line in enumerate(lines):
        if isinstance(line, _Line) and line.key is not None:
            where.setdefault(line.key, i)
    return "\n".join(lines), where


def pymetrix_imports(text: str) -> list[str]:
    """pymetrix's names the script's code uses (``Equation(``, ``Dataset.read``), outside comments and strings:
    ``Dataset``, ``Matrix`` and ``Sample`` first, then the rest in order."""
    import re

    import pymetrix

    public = set(pymetrix.__all__)
    code = "\n".join(line for line in text.split("\n") if not line.lstrip().startswith("#"))
    # strings and trailing comments out, in one pass from the left: a comment's apostrophe (the program's names)
    # does not open a string, and a # inside a string does not start a comment
    code = re.sub(r'"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'|#[^\n]*', lambda m: "" if m.group().startswith("#") else '""', code)
    found = {m.group(1) for m in re.finditer(r"(?<![\w.])([A-Za-z_]\w*)\s*[.(]", code)}
    order = ["Dataset", "Matrix", "Sample"]
    return [n for n in order if n in found and n in public] + sorted(n for n in found if n in public and n not in order)
