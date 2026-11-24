"""Phase 3: control flow. Loops, ``if`` blocks, subroutines, included programs, ``exitloop``, ``return`` and ``stop``,
the program's arguments (``%0``, ``%1``), names made from the program's variables (``y_{%s}``) and the program's own
values: conditions, loop bounds, and the strings and numbers made with EViews's programming-language functions
(``@word``, ``@wcount``, ``@wlookup``, ``@isobject``, a group's ``@count`` and ``@seriesname``, an equation's ``@r2``).

The program is read into blocks (``build``); the walker (``Flow``) translates them in order with the translator of
phases 1 and 2. A block becomes the same block in the script (``for i in range(1, n + 1):``, ``if r2 > 0.5:``, a
subroutine a function), and the same block in steps mode, which runs it in Python and makes each statement inside
it a recorded step: the History stays flat, as if each step had been made in a dialog. A condition is decided when
it is reached, so it can use a result made just before (``if eq.@r2 > 0.5``).

A block's body is translated once, as the program text is. What a body assigns is a value known only when the
program runs: lines after it use it by name, never a value guessed before. A name made from such a value
(``series y_{%s} = log(x_{%s})``) is an f-string in the script (``f"y_{s}"``) and the name made when the step runs;
which series a page has is no longer checked after it.

A subroutine's body is translated at its first ``call`` (on the page then selected), or after the program when
nothing calls it. EViews passes arguments by reference: a subroutine that assigns a parameter returns it, and the
call gives the value back to the caller's variable (``simend = dateshift(simstart, simend, 23)``). A subroutine
that is not ``local`` shares the program's variables (``global`` in the script); a local one shares nothing, and
the objects it would make are deleted when it returns in EViews, so its data and estimation lines are left out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ir import Code, Def, For, If, Invoke, Jump, Text, Var, While, name_code, quote
from .parse import Program, Statement, parse_program
from .report import ERROR, LEFT, NOTE, TRANSLATED

__all__ = ["Flow", "Node", "PLACEHOLDER", "ProgramValues", "WHILE_CAP", "build", "dynamic_ops", "render_code"]

WHILE_CAP = 10_000  # passes of a while loop in steps mode before it stops with an error (ProgramTranslation.run)

PLACEHOLDER = re.compile(r"pmxrt(\d+)z")  # a program variable's value in a name, while a statement is translated

_OPENERS = {"for": "next", "while": "wend", "if": "endif", "subroutine": "endsub"}
_CLOSERS = {"next": "for", "wend": "while", "endif": "if", "endsub": "subroutine"}
_WORDS = {"for": "for loop", "while": "while loop", "if": "if block", "subroutine": "subroutine"}


# -- blocks ----------------------------------------------------------------------------------------------------------


@dataclass
class Node:
    """A statement, or a block with its body: ``kind`` is ``plain``, ``for``, ``while``, ``if``, ``subroutine``,
    ``include`` (``included``: the included program's nodes) or ``stray`` (a block's end with no start). ``file`` is
    the program the statement is in ('' for the main one)."""

    st: Statement
    kind: str
    file: str = ""
    body: list[Node] = field(default_factory=list)
    orelse: list[Node] = field(default_factory=list)
    else_st: Statement | None = None
    close: Statement | None = None
    included: list[Node] = field(default_factory=list)
    program: Program | None = None  # an included program, read
    error: str = ""


def _blocks(program: Program, file: str) -> list[Node]:
    top: list[Node] = []
    stack: list[tuple[Node, list[Node]]] = []  # an open block and the body being filled
    for st in program.statements:
        word = st.word if st.kind == "command" else ""
        into = stack[-1][1] if stack else top
        if word in _OPENERS:
            node = Node(st, word, file)
            into.append(node)
            stack.append((node, node.body))
        elif word == "else" and stack and stack[-1][0].kind == "if" and stack[-1][0].else_st is None:
            node = stack[-1][0]
            node.else_st = st
            stack[-1] = (node, node.orelse)
        elif word in _CLOSERS and stack and stack[-1][0].kind == _CLOSERS[word]:
            stack.pop()[0].close = st
        elif word in _CLOSERS or word == "else":
            what = f"{word} without its {_CLOSERS.get(word, 'if')}" + (f" (the {_WORDS[stack[-1][0].kind]} open at line "
                                                                      f"{stack[-1][0].st.line} ends with "
                                                                      f"{_OPENERS[stack[-1][0].kind]})" if stack else "")
            into.append(Node(st, "stray", file, error=what))
        else:
            into.append(Node(st, "include" if word == "include" else "plain", file))
    for node, _ in stack:
        node.error = f"the {_WORDS[node.kind]} has no {_OPENERS[node.kind]}"
    return top


def build(program: Program, file: str = "", folder: Path | None = None, seen: tuple[str, ...] = ()) -> list[Node]:
    """The program's blocks, with each included program read (relative to ``folder``) and its blocks inside the
    ``include`` node."""
    nodes = _blocks(program, file)

    def walk(ns: list[Node]) -> None:
        for n in ns:
            if n.kind == "include":
                _include(n, folder, seen)
            walk(n.body)
            walk(n.orelse)

    walk(nodes)
    return nodes


def _include(node: Node, folder: Path | None, seen: tuple[str, ...]) -> None:
    words = re.findall(r'"[^"]*"|\S+', node.st.rest)
    if len(words) != 1:
        node.error = "include names one program: include lib.prg"
        return
    name = words[0].strip('"').replace("\\", "/")
    path = Path(name if name.lower().endswith(".prg") else name + ".prg")
    if not path.is_absolute():
        path = (folder or Path.cwd()) / path
    shown = name if name.lower().endswith(".prg") else name + ".prg"
    if not path.is_file():
        node.error = f"the included program {shown} is not found (looked for {path})"
        return
    key = str(path.resolve()).lower()
    if key in seen:
        node.error = f"{shown} includes itself (through {' -> '.join(Path(s).name for s in seen)})"
        return
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    node.program = parse_program(text)
    node.included = build(node.program, shown, path.parent, (*seen, key))


def _assigned(nodes: list[Node], subs: dict[str, Any], seen: set[str] | None = None) -> set[str]:
    """The program variables (``!x``, ``%s``, scalars and strings) that statements in ``nodes`` may assign, through
    the subroutines they call too."""
    seen = set() if seen is None else seen
    out: set[str] = set()
    for n in nodes:
        st = n.st
        if st.kind == "assign" and st.lhs[:1] in "!%":
            out.add(st.lhs.lower())
        elif st.kind == "assign" and not st.lhs[:1] in "{(":
            out.add(st.lhs.lower())
        elif st.kind == "command" and st.word in ("scalar", "string"):
            m = re.match(r"\s*([A-Za-z_]\w*)", st.rest)
            if m:
                out.add(m.group(1).lower())
        elif st.kind == "command" and st.word == "for":
            m = re.match(r"\s*([!%]\w+)", st.rest)
            if m:
                out.add(m.group(1).lower())
        elif st.kind == "command" and st.word == "call":
            m = re.match(r"\s*(\w+)\s*(?:\((.*)\))?\s*$", st.rest, re.S)
            if m:
                for a in _split_args(m.group(2) or ""):
                    if re.fullmatch(r"\s*[!%]\w+\s*", a):
                        out.add(a.strip().lower())
                sub = subs.get(m.group(1).lower())
                if sub is not None and not sub.local and sub.name not in seen:
                    seen.add(sub.name)
                    out |= _assigned(sub.node.body, subs, seen)
        out |= _assigned(n.body, subs, seen) | _assigned(n.orelse, subs, seen) | _assigned(n.included, subs, seen)
    return out


def _split_args(text: str) -> list[str]:
    """Arguments separated by commas outside brackets and quotes."""
    out, depth, start, quoted = [], 0, 0, False
    for i, ch in enumerate(text):
        if ch == '"':
            quoted = not quoted
        elif not quoted and ch == "(":
            depth += 1
        elif not quoted and ch == ")":
            depth -= 1
        elif not quoted and ch == "," and depth == 0:
            out.append(text[start:i])
            start = i + 1
    if text.strip():
        out.append(text[start:])
    return out


# -- names made as the program runs ---------------------------------------------------------------------------------


def _to_text(s: str, registry: dict[int, str]) -> Any:
    if not PLACEHOLDER.search(s):
        return s
    parts: list[Any] = []
    pos = 0
    for m in PLACEHOLDER.finditer(s):
        if m.start() > pos:
            parts.append(s[pos:m.start()])
        parts.append(Var(registry[int(m.group(1))]))
        pos = m.end()
    if pos < len(s):
        parts.append(s[pos:])
    return Text(tuple(parts))


def _fstring(literal: str, registry: dict[int, str]) -> str:
    """A string literal of the script with the placeholders as an f-string's fields."""
    prefix = ""
    if literal[:1] in "fF":
        prefix, literal = "f", literal[1:]
    quote_ch = literal[0]
    body = literal[1:-1]
    if not prefix:
        body = body.replace("{", "{{").replace("}", "}}")
    body = PLACEHOLDER.sub(lambda m: "{" + registry[int(m.group(1))] + "}", body)
    return f"f{quote_ch}{body}{quote_ch}"


def render_code(code: str, registry: dict[int, str]) -> str:
    """Python with placeholders made into what runs: in a string, an f-string's field (``f"log(x_{s})"``); in a
    name, the object of that name (``globals()[f"eq_{c}"]``)."""
    if not PLACEHOLDER.search(code):
        return code
    out: list[str] = []
    pos = 0

    def names(text: str) -> str:
        def one(m: re.Match[str]) -> str:
            if not PLACEHOLDER.search(m.group(0)) or (m.start() > 0 and text[m.start() - 1] == "."):
                return m.group(0)
            return f"globals()[{_fstring(quote(m.group(0)), registry)}]"

        return re.sub(r"[A-Za-z_]\w*", one, text)

    for m in re.finditer(r'[fF]?"(?:[^"\\]|\\.)*"|[fF]?\'(?:[^\'\\]|\\.)*\'', code):
        out.append(names(code[pos:m.start()]))
        out.append(_fstring(m.group(0), registry) if PLACEHOLDER.search(m.group(0)) else m.group(0))
        pos = m.end()
    out.append(names(code[pos:]))
    return "".join(out)


def _dyn(value: Any, registry: dict[int, str]) -> Any:
    import dataclasses

    from .ir import Arg, Attr, New, RefTo, SampleSpec

    if isinstance(value, str):
        return _to_text(value, registry)
    if isinstance(value, Text):
        parts: list[Any] = []
        for p in value.parts:
            t = _to_text(p, registry) if isinstance(p, str) else p
            parts.extend(t.parts if isinstance(t, Text) else [t])
        return Text(tuple(parts))
    if isinstance(value, SampleSpec):
        return dataclasses.replace(value, start=_dyn(value.start, registry), end=_dyn(value.end, registry),
                                   where=_dyn(value.where, registry))
    if isinstance(value, New):
        return dataclasses.replace(value, args=tuple(_dyn(a, registry) for a in value.args))
    if isinstance(value, Arg):
        return dataclasses.replace(value, value=_dyn(value.value, registry))
    if isinstance(value, RefTo) and isinstance(value.name, str):
        return RefTo(_to_text(value.name, registry))
    if isinstance(value, Attr):
        return value if not PLACEHOLDER.search(value.base) else value  # an attribute of a page: pages are never made so
    if isinstance(value, dict):
        return {_dyn(k, registry): _dyn(v, registry) for k, v in value.items()}
    if isinstance(value, list):
        return [_dyn(v, registry) for v in value]
    if isinstance(value, tuple):
        return tuple(_dyn(v, registry) for v in value)
    return value


def dynamic_ops(ops: list[Any], registry: dict[int, str]) -> list[Any]:
    """A statement's ops with the placeholders of the program's values made into names made as the program runs."""
    import dataclasses

    from .ir import Cmd, Step

    out: list[Any] = []
    for op in ops:
        if isinstance(op, Step):
            out.append(dataclasses.replace(
                op, receiver=_to_text(op.receiver, registry) if isinstance(op.receiver, str) else _dyn(op.receiver, registry),
                output=_to_text(op.output, registry), callee=render_code(op.callee, registry),
                args=tuple(_dyn(a, registry) for a in op.args)))
        elif isinstance(op, Code):
            out.append(Code(render_code(op.code, registry), tuple(_to_text(n, registry) for n in op.names)))
        elif isinstance(op, Cmd):
            fields = {f.name: _to_text(getattr(op.command, f.name), registry) for f in dataclasses.fields(op.command)
                      if isinstance(getattr(op.command, f.name), str) and PLACEHOLDER.search(getattr(op.command, f.name))}
            if fields:
                out.append(dataclasses.replace(op, fields=fields, script=render_code(op.command.to_code(), registry)))
            else:
                out.append(op)
        else:
            out.append(op)
    return out


# -- the program's values ---------------------------------------------------------------------------------------------

_TOKEN = re.compile(r"""\s*(?:
    (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<str>"[^"]*")
  | (?P<member>[A-Za-z_]\w*\.@\w+)
  | (?P<name>[!%]\w+|@?[A-Za-z_]\w*)
  | (?P<op><>|<=|>=|[-+*/^=<>(),])
)""", re.X)

# EViews's string and word functions: Python for the arguments' code (each a (code, kind) pair)
_STRING_FUNCTIONS = {
    "@lower": (1, lambda a: f"{_m(a[0])}.lower()", "str"), "@upper": (1, lambda a: f"{_m(a[0])}.upper()", "str"),
    "@trim": (1, lambda a: f"{_m(a[0])}.strip()", "str"), "@ltrim": (1, lambda a: f"{_m(a[0])}.lstrip()", "str"),
    "@rtrim": (1, lambda a: f"{_m(a[0])}.rstrip()", "str"), "@len": (1, lambda a: f"len({a[0][0]})", "num"),
    "@length": (1, lambda a: f"len({a[0][0]})", "num"),
    "@replace": (3, lambda a: f"{_m(a[0])}.replace({a[1][0]}, {a[2][0]})", "str"),
    "@left": (2, lambda a: f"{_m(a[0])}[:{_int(a[1])}]", "str"), "@right": (2, lambda a: f"{_m(a[0])}[-{_int(a[1], True)}:]", "str"),
    "@mid": ((2, 3), lambda a: f"{_m(a[0])}[{_int(a[1])} - 1:{_int(a[1])} - 1 + {_int(a[2])}]" if len(a) == 3
             else f"{_m(a[0])}[{_int(a[1])} - 1:]", "str"),
    "@instr": (2, lambda a: f"({_m(a[0])}.find({a[1][0]}) + 1)", "num"),
    "@str": (1, lambda a: f"str({a[0][0]})", "str"), "@val": (1, lambda a: f"float({a[0][0]})", "num"),
    "@wcount": (1, lambda a: f"len({_m(a[0])}.split())", "num"),
    "@word": (2, lambda a: f"{_m(a[0])}.split()[{_int(a[1])} - 1]", "str"),
    "@wfind": (2, lambda a: f"({_m(a[0])}.split().index({a[1][0]}) + 1 if {a[1][0]} in {_m(a[0])}.split() else 0)", "num"),
    "@wfindnc": (2, lambda a: f"({_m(a[0])}.lower().split().index({_m(a[1])}.lower()) + 1 "
                              f"if {_m(a[1])}.lower() in {_m(a[0])}.lower().split() else 0)", "num"),
    "@abs": (1, lambda a: f"abs({a[0][0]})", "num"), "@sqrt": (1, lambda a: f"{a[0][0]} ** 0.5", "num"),
    "@floor": (1, lambda a: f"float(int({a[0][0]} // 1))", "num"),
    "@ceiling": (1, lambda a: f"float(-int(-({a[0][0]}) // 1))", "num"),
}
_NOTES = {"@str": "@str gives Python's text for the number, which may differ from EViews's in the digits shown (verify)",
          "@word": "@word past the last word is an error in Python, an empty string in EViews",
          "@val": "@val of a text that is not a number is an error in Python, NA in EViews"}
_STATISTICS = {"@mean": "mean", "@sum": "sum", "@min": "min", "@max": "max", "@stdev": "std", "@median": "median",
               "@obs": "count"}


def _m(arg: tuple[str, str, int]) -> str:
    """An argument as the receiver of a method: bracketed unless it is a name or a literal."""
    code = arg[0]
    return code if re.fullmatch(r"[A-Za-z_][\w.]*|\"[^\"]*\"|globals\(\)\[[^\]]*\]", code) else f"({code})"


def _int(arg: tuple[str, str, int], parens: bool = False) -> str:
    code = arg[0]
    if re.fullmatch(r"\d+", code):
        return code
    return f"int({code})"


class _Values:
    """A recursive-descent reader of an EViews program expression into Python: (code, kind, precedence), with
    ``static`` false once it uses something known only when the program runs (an object, the data)."""

    def __init__(self, tr: Any, text: str, at: int) -> None:
        self.tr, self.text, self.at = tr, text, at
        self.tokens: list[tuple[str, str]] = []
        pos = 0
        while pos < len(text):
            if not text[pos:].strip():
                break
            m = _TOKEN.match(text, pos)
            if m is None or m.end() == pos:
                raise _error(f"{text[pos:].strip()[:20]!r} does not read here")
            kind = m.lastgroup or "op"
            self.tokens.append((kind, m.group(kind)))
            pos = m.end()
        self.i = 0
        self.notes: list[str] = []
        self.static = True

    def peek(self) -> tuple[str, str]:
        return self.tokens[self.i] if self.i < len(self.tokens) else ("end", "")

    def take(self) -> tuple[str, str]:
        tok = self.peek()
        self.i += 1
        return tok

    def expect(self, op: str) -> None:
        kind, value = self.take()
        if value != op:
            raise _error(f"expected {op!r}" + (f", not {value!r}" if value else " at the end"))

    def word(self, w: str) -> bool:
        kind, value = self.peek()
        return kind == "name" and value.lower() == w

    def read(self) -> tuple[str, str, int]:
        out = self.or_()
        if self.peek()[0] != "end":
            raise _error(f"{self.peek()[1]!r} does not read here")
        return out

    @staticmethod
    def wrap(part: tuple[str, str, int], prec: int) -> str:
        return part[0] if part[2] >= prec else f"({part[0]})"

    def or_(self) -> tuple[str, str, int]:
        left = self.and_()
        while self.word("or"):
            self.take()
            right = self.and_()
            left = (f"{self.wrap(left, 1)} or {self.wrap(right, 1)}", "bool", 1)
        return left

    def and_(self) -> tuple[str, str, int]:
        left = self.not_()
        while self.word("and"):
            self.take()
            right = self.not_()
            left = (f"{self.wrap(left, 2)} and {self.wrap(right, 2)}", "bool", 2)
        return left

    def not_(self) -> tuple[str, str, int]:
        if self.word("not"):
            self.take()
            inner = self.not_()
            return (f"not {self.wrap(inner, 3)}", "bool", 3)
        return self.cmp()

    def cmp(self) -> tuple[str, str, int]:
        left = self.add()
        kind, value = self.peek()
        if value in ("=", "<>", "<", ">", "<=", ">="):
            self.take()
            right = self.add()
            op = {"=": "==", "<>": "!="}.get(value, value)
            if left[1] == "str" and right[1] == "str" and op not in ("==", "!="):
                raise _left("ordering strings (<, >) is not translated: = and <> are")
            return (f"{self.wrap(left, 5)} {op} {self.wrap(right, 5)}", "bool", 4)
        return left

    def add(self) -> tuple[str, str, int]:
        left = self.mul()
        while self.peek()[1] in ("+", "-"):
            op = self.take()[1]
            right = self.mul()
            kind = "str" if left[1] == "str" or right[1] == "str" else "num"
            if kind == "str" and op == "-":
                raise _error("a string minus a string")
            left = (f"{self.wrap(left, 5)} {op} {self.wrap(right, 6)}", kind, 5)
        return left

    def mul(self) -> tuple[str, str, int]:
        left = self.neg()
        while self.peek()[1] in ("*", "/"):
            op = self.take()[1]
            right = self.neg()
            left = (f"{self.wrap(left, 6)} {op} {self.wrap(right, 7)}", "num", 6)
        return left

    def neg(self) -> tuple[str, str, int]:
        if self.peek()[1] == "-":
            self.take()
            inner = self.neg()
            return (f"-{self.wrap(inner, 7)}", "num", 7)
        if self.peek()[1] == "+":
            self.take()
            return self.neg()
        return self.pow_()

    def pow_(self) -> tuple[str, str, int]:
        base = self.atom()
        if self.peek()[1] == "^":
            self.take()
            exp = self.neg()
            return (f"{self.wrap(base, 9)} ** {self.wrap(exp, 7)}", "num", 8)
        return base

    def args(self) -> list[tuple[str, str, int]]:
        out: list[tuple[str, str, int]] = []
        self.expect("(")
        if self.peek()[1] == ")":
            self.take()
            return out
        while True:
            out.append(self.or_())
            if self.peek()[1] == ",":
                self.take()
                continue
            self.expect(")")
            return out

    def raw_args(self) -> str:
        """The text of a call's brackets, as written (a series expression for a statistic)."""
        depth = 0
        start = None
        while True:
            kind, value = self.take()
            if kind == "end":
                raise _error("a bracket is not closed")
            if value == "(":
                depth += 1
                if depth == 1:
                    start = self.i
                    continue
            elif value == ")":
                depth -= 1
                if depth == 0:
                    return " ".join(v for _, v in self.tokens[start:self.i - 1])

    def atom(self) -> tuple[str, str, int]:
        kind, value = self.take()
        if kind == "num":
            text = "0" + value if value.startswith(".") else value
            return (text, "num", 9)
        if kind == "str":
            return (quote(value[1:-1]), "str", 9)
        if value == "(":
            inner = self.or_()
            self.expect(")")
            return (inner[0], inner[1], 9) if inner[2] >= 9 else (f"({inner[0]})", inner[1], 9)
        if kind == "member":
            obj, _, member = value.partition(".")
            index = self.args() if self.peek()[1] == "(" else []
            self.static = False
            code, out_kind, note = self.tr.value_member(obj, member.lower(), index)
            if note and note not in self.notes:
                self.notes.append(note)
            return (code, out_kind, 9)
        if kind == "name":
            low = value.lower()
            if low.startswith("@"):
                return self.function(low)
            if self.peek()[1] == "(" and not low.startswith(("!", "%")):
                if low == "c" or low in self.tr._vectors():
                    index = self.args()
                    self.static = False
                    code, note = self.tr.coefficient_code(low, index)
                    if note and note not in self.notes:
                        self.notes.append(note)
                    return (code, "num", 9)
                raise _left(f"{value}(...) is not translated in a program's value: a function EViews's programs use is "
                            "written with @")
            if low in ("na",):
                return ('float("nan")', "num", 9)
            if PLACEHOLDER.fullmatch(low):
                raise _left("an object named by a variable ({%x}) as a value: use one of its data members ({%x}.@count)")
            code, out_kind, known = self.tr.value_name(value)
            if not known:
                self.static = False
            return (code, out_kind, 9)
        if kind == "end":
            raise _error("the expression ends too soon")
        raise _error(f"{value!r} does not read here")

    def function(self, name: str) -> tuple[str, str, int]:
        if name == "@obssmpl":
            page = self.tr._page()
            self.static = False
            return (f"int({page.var}.sample_mask().sum())", "num", 9)
        if name in _STATISTICS:
            text = self.raw_args()
            page = self.tr._page()
            inner = self.tr.expression(text, self.at, page)
            self.static = False
            fn = _STATISTICS[name]
            note = f"{name} over the page's sample when the line runs, missing values skipped"
            if note not in self.notes:
                self.notes.append(note)
            return (f"float({page.var}.eval({inner.code()})[{page.var}.sample_mask()].{fn}())", "num", 9)
        if name == "@isobject":
            args = self.args()
            if len(args) != 1:
                raise _error("@isobject takes one name")
            page = self.tr._page()
            self.static = False
            arg = _m(args[0])
            return (f"({arg}.lower() in {page.var}.names or {arg}.lower() in globals())", "bool", 9)
        if name == "@wlookup":
            args = self.args()
            if not 1 <= len(args) <= 2:
                raise _error("@wlookup takes a pattern, and a type")
            if len(args) == 2 and args[1][0].lower() not in ('"series"',):
                raise _left("@wlookup with a type other than series is not translated: the page's series are")
            page = self.tr._page()
            self.static = False
            note = "@wlookup looks among the page's series (Dataset.series with the pattern)"
            if note not in self.notes:
                self.notes.append(note)
            return (f'" ".join({page.var}.series({_m(args[0])}.lower()).names)', "str", 9)
        if name in _STRING_FUNCTIONS:
            n, make, out_kind = _STRING_FUNCTIONS[name]
            args = self.args()
            allowed = n if isinstance(n, tuple) else (n,)
            if len(args) not in allowed:
                shown = " or ".join(map(str, allowed))
                raise _error(f"{name} takes {shown} argument{'s' if max(allowed) > 1 else ''}, not {len(args)}")
            if name in _NOTES and _NOTES[name] not in self.notes:
                self.notes.append(_NOTES[name])
            return (make(args), out_kind, 9)
        if name in ("@log", "@exp"):
            args = self.args()
            if len(args) != 1:
                raise _error(f"{name} takes one number")
            raise _left(f"{name} of a value known when the program runs: pymetrix's scripts import only pymetrix (use "
                        "a series' log, or work the number out first)")
        from .left import GAP_FUNCTIONS, PROGRAM_FUNCTIONS

        if name in ("@dateval", "@dateadd", "@datestr", "@datediff", "@strdate", "@dtoo", "@otod"):
            raise _left(f"{name}: EViews's date numbers are not translated (a gap, docs/known-gaps.md): write the dates, "
                        "or give them as the program's arguments")
        if name in ("@uiprompt", "@uidialog", "@uiedit", "@uilist", "@uiradio"):
            raise _left("interactive prompts; in pymetrix: the dialog asks for the program's arguments")
        if name in PROGRAM_FUNCTIONS or name in GAP_FUNCTIONS:
            raise _left(f"{name} is not translated in a program's values (a gap, docs/known-gaps.md): @word, @wcount, "
                        "@wfind, @wlookup, @isobject, @lower, @upper, @replace, @trim, @left, @right, @mid, @instr, "
                        "@len, @str, @val and the sample's statistics are")
        raise _left(f"{name} is not an EViews function this translator knows")


def _error(reason: str) -> Exception:
    from .data import _error as e

    return e(reason)


def _left(reason: str) -> Exception:
    from .data import _left as e

    return e(reason)


_SAFE = {"len": len, "float": float, "int": int, "str": str, "abs": abs}


class ProgramValues:
    """The translator's handling of the program's own values (mixed into ``Translator``)."""

    def program_value(self, text: str, at: int = 0) -> tuple[str, Any, str, list[str]]:
        """An EViews program expression as Python: its code, its value where known before the program runs (else
        ``_UNKNOWN``), its kind (``num``, ``str`` or ``bool``) and notes."""
        from .data import _UNKNOWN

        text, _ = self._braces(text)
        reader = _Values(self, text, at)
        code, kind, _ = reader.read()
        value: Any = _UNKNOWN
        if reader.static:
            known = {v.name: v.value for v in self.py.values() if v.value is not _UNKNOWN and v.kind in ("scalar", "control", "string")}
            try:
                value = eval(code, {"__builtins__": _SAFE}, known)  # noqa: S307 - Python made here from the program's own values
            except Exception:  # noqa: BLE001 - a name with no value yet, an index past the end: worked out when it runs
                value = _UNKNOWN
            if isinstance(value, bool):
                value = float(value)
        return code, value, kind, reader.notes

    def value_name(self, name: str) -> tuple[str, str, bool]:
        """A name in a program expression: a scalar, a string or a control variable (``!i``, ``%s``)."""
        from .data import _UNKNOWN

        key = name.lower()
        v = self.py.get(key)
        if v is None and key[:1] in "!%":
            v = self.runtime_variable(key)
        if v is None:
            if key in self.unset or self.lost:
                raise self._missing(name)
            page = self.current
            if page is not None and page.series is not None and key in page.series:
                raise _left(f"{name} is a series: a program's value takes a number (scalar m = @mean({name}) first)")
            raise _error(f"{name} is not a scalar, a string or a variable of the program")
        if v.kind in ("scalar", "control"):
            return v.name, "num", v.value is not _UNKNOWN
        if v.kind == "string":
            return v.name, "str", v.value is not _UNKNOWN
        raise _left(f"{name} is a {v.kind}: as a program's value, use one of its data members ({name}.@count)")

    def value_member(self, obj: str, member: str, index: list[tuple[str, str, int]]) -> tuple[str, str, str]:
        """``obj.@member(index)``: a group's ``@count``, ``@seriesname(i)`` and ``@members``; an equation's values
        (``@r2``, ``@coefs(i)``) through phase 2; for an object named by a variable, by what the member says it is."""
        key = obj.lower()
        dynamic = PLACEHOLDER.search(key) is not None
        v = self.py.get(key)
        if v is not None and v.kind == "group" or dynamic and member in ("@count", "@seriesname", "@members"):
            ref = name_code(_to_text(key, self.dynamic)) if dynamic else v.name
            note = f"{obj} is taken to be a group, as {member} is a group's" if dynamic else ""
            if member == "@count":
                return f"len({ref}.names)", "num", note
            if member == "@members":
                return f'" ".join({ref}.names)', "str", note
            if member == "@seriesname":
                if len(index) != 1:
                    raise _error("@seriesname takes the series' position")
                return f"{ref}.names[{_int(index[0])} - 1]", "str", note
            raise _left(f"{obj}.{member} is not translated: a group's @count, @seriesname and @members are")
        if dynamic:
            fit = self.fits.get(key)
            if fit is None:
                raise _left(f"{obj}.{member}: an object named by a variable that no line translated so far estimates")
        if index and len(index) != 1:
            raise _error(f"{member} takes one index")
        i: int | None = None
        if index:
            try:
                i = int(index[0][0])
            except ValueError:
                raise _left(f"{obj}.{member} with an index known when the program runs is not translated: a number is") from None
        code, note = self.member_code(obj, member, i)
        if dynamic:
            code = render_code(code, self.dynamic)
        return code, "num", note

    def coefficient_code(self, vector: str, index: list[tuple[str, str, int]]) -> tuple[str, str]:
        if len(index) != 1 or not re.fullmatch(r"\d+", index[0][0]):
            raise _left(f"{vector}(...) with an index known when the program runs is not translated: a number is")
        _, codes, notes = self.members_in(f"{vector}({index[0][0]})")
        return next(iter(codes.values())), "; ".join(notes)

    def runtime_variable(self, key: str) -> Any:
        """Inside a block or a subroutine: a program variable (``!x``, ``%s``) no line before has set, taken as set
        when the program runs (by the caller, or an earlier pass of a loop)."""
        from .data import _UNKNOWN, _Py

        if not self.runtime or key[:1] not in "!%":
            return None
        name, _ = self._name(key, f"{'control' if key[0] == '!' else 'string'} {key}")
        v = _Py(name, "control" if key[0] == "!" else "string", _UNKNOWN)
        self.py[key] = v
        return v

    def placeholder(self, key: str) -> str:
        """The placeholder of a program variable's value in a name, while a statement is translated."""
        name = self.py[key].name
        for i, n in self.dynamic.items():
            if n == name:
                return f"pmxrt{i}z"
        i = len(self.dynamic)
        self.dynamic[i] = name
        return f"pmxrt{i}z"


# -- the walker ------------------------------------------------------------------------------------------------------


@dataclass
class _Sub:
    name: str  # lower case
    node: Node
    params: list[tuple[str, str, str]] = field(default_factory=list)  # (type, EViews name, key in py)
    local: bool = False
    var: str = ""  # the function's Python name
    pnames: list[str] = field(default_factory=list)  # the parameters' Python names
    returns: list[int] = field(default_factory=list)  # positions of the parameters the body assigns
    done: bool = False
    translating: bool = False
    error: str = ""
    outcome: Any = None


_DOLLAR = ("a name with $ in it (_$_x) is not translated: pymetrix's names are Python names; rename the object in the "
           "program (a gap, docs/known-gaps.md)")

_PARAM_TYPES = ("string", "scalar", "series", "group", "equation", "vector", "matrix", "sym", "coef", "model", "sample",
                "table", "graph", "svector", "rowvector", "var", "system", "alpha", "text", "spool", "pool", "valmap",
                "frml", "link", "logl", "sspace")


class Flow:
    """Translates the program's blocks in order (see the module's docstring)."""

    def __init__(self, tr: Any, nodes: list[Node]) -> None:
        self.tr = tr
        self.nodes = nodes
        self.subs: dict[str, _Sub] = {}
        self.loops = 0  # loops around the statement being translated
        self.sub: _Sub | None = None  # the subroutine being translated
        self._collect(nodes)

    def _collect(self, nodes: list[Node]) -> None:
        from .data import Outcome

        for n in nodes:
            if n.kind == "subroutine" and not n.error:
                sub = self._signature(n)
                sub.outcome = Outcome(n.st, TRANSLATED, "", file=n.file, kind="subroutine")
                if sub.name in self.subs:
                    sub.error = f"a second subroutine {sub.name} (the first is at line {self.subs[sub.name].node.st.line}" + \
                                (f" of {self.subs[sub.name].node.file}" if self.subs[sub.name].node.file else "") + ")"
                else:
                    self.subs[sub.name] = sub
                    continue
                self.subs.setdefault(f"{sub.name}#{n.st.line}{n.file}", sub)
            self._collect(n.included)

    def _signature(self, n: Node) -> _Sub:
        m = re.fullmatch(r"\s*(local\s+)?([A-Za-z_]\w*)\s*(?:\((.*)\))?\s*", n.st.rest, re.S | re.I)
        if m is None:
            sub = _Sub("?", n, error="subroutine names the subroutine: subroutine f(string %a, scalar !b)")
            return sub
        sub = _Sub(m.group(2).lower(), n, local=bool(m.group(1)))
        for part in _split_args(m.group(3) or ""):
            words = part.split()
            if len(words) != 2 or words[0].lower() not in _PARAM_TYPES:
                sub.error = f"the parameter {part.strip()!r} does not read: a type and a name (string %a, scalar !b)"
                break
            kind, pname = words[0].lower(), words[1]
            if kind not in ("string", "scalar"):
                sub.error = (f"a parameter of type {kind} ({pname}): object arguments are not translated (a gap, "
                             "docs/known-gaps.md); string and scalar parameters are")
                break
            sub.params.append((kind, pname, pname.lower()))
        return sub

    # -- walking --------------------------------------------------------------------------------------------------

    def walk(self, nodes: list[Node]) -> list[Any]:
        out = []
        for n in nodes:
            out.extend(self.node(n))
        return out

    def node(self, n: Node) -> list[Any]:
        from .data import Outcome

        tr = self.tr
        if n.error:
            tr.forget()
            return [self._outcome(n.st, ERROR, n.error, n.file), *self._inside(n, ERROR, f"inside a block that does not read (line {n.st.line})")]
        if n.kind == "plain":
            return [self.plain(n)]
        if n.kind == "include":
            shown = self._shown(n)
            count = sum(1 for line in n.program.lines if line.strip()) if n.program is not None else 0
            o = Outcome(n.st, TRANSLATED, f"the {count} lines of {shown} follow, under its name; its subroutines are "
                        "functions of the script", file=n.file, kind="include")
            o.program, o.included_file = n.program, shown
            o.children = self.walk(n.included)
            return [o]
        if n.kind == "subroutine":
            sub = self._sub_of(n)
            if sub.outcome is None:
                sub.outcome = Outcome(n.st, TRANSLATED, "", file=n.file, kind="subroutine")
            o = sub.outcome
            if sub.error:
                o.status, o.reason = ERROR if "does not read" in sub.error or "a second" in sub.error else LEFT, sub.error
                o.children = self._inside_list(n.body, LEFT, f"inside the subroutine {sub.name} (line {n.st.line}), which is "
                                               "left out")
                o.closer = self._outcome(n.close, LEFT, f"the end of the subroutine {sub.name}, which is left out", n.file)
            return [o]
        return [self.block(n)]

    @staticmethod
    def _shown(n: Node) -> str:
        words = re.findall(r'"[^"]*"|\S+', n.st.rest)
        name = words[0].strip('"') if words else "?"
        return name if name.lower().endswith(".prg") else name + ".prg"

    def _sub_of(self, n: Node) -> _Sub:
        for sub in self.subs.values():
            if sub.node is n:
                return sub
        return self._signature(n)

    def _outcome(self, st: Statement | None, status: str, reason: str, file: str) -> Any:
        from .data import Outcome

        if st is None:
            return None
        return Outcome(st, status, reason, file=file)

    def _inside_list(self, nodes: list[Node], status: str, reason: str) -> list[Any]:
        out = []
        for c in nodes:
            o = self._outcome(c.st, status if c.kind != "plain" or status != ERROR else LEFT, reason, c.file)
            self.tr.inside_block(c.st)
            o.children = self._inside_list(c.body, status, reason)
            o.orelse = self._inside_list(c.orelse, status, reason)
            o.else_outcome = self._outcome(c.else_st, LEFT, reason, c.file)
            o.closer = self._outcome(c.close, LEFT, reason, c.file)
            if c.kind == "include":
                o.children = self._inside_list(c.included, status, reason)
            out.append(o)
        return out

    def _inside(self, n: Node, status: str, reason: str) -> list[Any]:
        """The statements of a block that is left out (or does not read), each saying so."""
        out = self._inside_list(n.body, LEFT, reason)
        if n.else_st is not None:
            out.append(self._outcome(n.else_st, LEFT, reason, n.file))
        out += self._inside_list(n.orelse, LEFT, reason)
        if n.close is not None:
            out.append(self._outcome(n.close, LEFT, reason, n.file))
        return out

    def _runtime(self) -> Any:
        tr = self.tr

        class _Ctx:
            def __enter__(self_inner) -> None:
                tr.runtime += 1

            def __exit__(self_inner, *exc: Any) -> None:
                tr.runtime -= 1

        return _Ctx()

    def _unknown(self, keys: set[str]) -> None:
        """Program variables a block assigns: their values are known only when the program runs."""
        from .data import _UNKNOWN

        for k in keys:
            v = self.tr.py.get(k)
            if v is not None and v.kind in ("scalar", "control", "string"):
                v.value = _UNKNOWN

    # -- statements -------------------------------------------------------------------------------------------------

    _NOT_IN_BLOCKS = {"wfopen": "opening the workfile", "wfcreate": "making a page", "pagecreate": "making a page",
                      "pageselect": "selecting a page", "model": "making a model", "system": "making a system",
                      "sspace": "a state-space object"}

    def plain(self, n: Node) -> Any:
        from .data import Outcome

        tr = self.tr
        st = n.st
        word = st.word if st.kind == "command" else ""
        if word in ("exitloop", "return", "stop"):
            return self.jump(n)
        if word == "call":
            return self.call(n)
        if tr.runtime and (word in self._NOT_IN_BLOCKS or st.kind == "method" and st.obj.lower() in tr.models and st.word in ("append", "merge")):
            what = self._NOT_IN_BLOCKS.get(word, "building a model")
            o = Outcome(st, LEFT, f"{what} inside a loop, an if block or a subroutine is not translated: pages and models are "
                        "followed as the program's text runs, so make them before its control flow", file=n.file)
            tr.inside_block(st)
            tr.forget()
            return o
        if self.sub is not None and self.sub.local and word not in ("scalar", "string") and st.kind != "assign" \
                and not (st.kind == "assign" and st.lhs[:1] in "!%"):
            o = Outcome(st, LEFT, f"inside the local subroutine {self.sub.name}: the objects a local subroutine makes are "
                        "deleted when it returns in EViews, so its data and estimation lines are not translated (its "
                        "strings, numbers and control flow are)", file=n.file)
            tr.inside_block(st)
            return o
        if self.sub is not None and self.sub.local and st.kind == "assign" and st.lhs[:1] not in "!%" \
                and st.lhs.lower() not in tr.py:
            o = Outcome(st, LEFT, f"inside the local subroutine {self.sub.name}: a series it makes is deleted when it returns in "
                        "EViews; not translated", file=n.file)
            tr.inside_block(st)
            return o
        try:
            o = tr.statement(st)
        except Exception as exc:  # noqa: BLE001 - a line the translator cannot take is an error row, not a lost translation
            tr.forget()
            o = Outcome(st, ERROR, f"the translator failed on this line ({type(exc).__name__}: {exc}); please report it")
        o.file = n.file
        return o

    def jump(self, n: Node) -> Any:
        from .data import Outcome

        word = n.st.word
        if n.st.rest.strip():
            return Outcome(n.st, ERROR, f"{word} takes nothing after it", file=n.file)
        if word == "exitloop":
            if not self.loops:
                return Outcome(n.st, ERROR, "exitloop outside a loop", file=n.file)
            o = Outcome(n.st, TRANSLATED, "", [Jump("break")], file=n.file)
            return o
        if word == "return":
            if self.sub is None:
                o = Outcome(n.st, NOTE, "return outside a subroutine ends the program, as stop does", [Jump("stop")], file=n.file)
                return o
            return Outcome(n.st, TRANSLATED, "", [Jump("return")], file=n.file)
        return Outcome(n.st, TRANSLATED, "", [Jump("stop")], file=n.file)

    def call(self, n: Node) -> Any:
        from .data import Outcome, _Out

        tr = self.tr
        st = n.st
        m = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*(?:\((.*)\))?\s*", st.rest, re.S)
        if m is None:
            return Outcome(st, ERROR, "call names the subroutine: call f(%a, 1)", file=n.file)
        name = m.group(1).lower()
        sub = self.subs.get(name)
        if "$" in re.sub(r'"[^"]*"', "", st.text):
            tr._unset(st)
            if sub is not None:
                self._unknown(_assigned([sub.node], self.subs))
            tr.forget()
            return Outcome(st, LEFT, _DOLLAR, file=n.file)
        if sub is None:
            status = ERROR if not tr.lost else LEFT
            tr.forget()
            return Outcome(st, status, f"no subroutine {m.group(1)} in the program or the programs it includes", file=n.file)
        if sub.error:
            self._unknown(_assigned([sub.node], self.subs))
            tr.forget()
            return Outcome(st, LEFT, f"the subroutine {sub.name} is left out: {sub.error}", file=n.file)
        args = _split_args(m.group(2) or "")
        if len(args) != len(sub.params):
            return Outcome(st, ERROR, f"{sub.name} takes {len(sub.params)} argument{'s' if len(sub.params) != 1 else ''}, "
                           f"not {len(args)}", file=n.file)
        codes: list[str] = []
        notes: list[str] = []
        targets: list[str | None] = []
        try:
            for (kind, pname, _), arg in zip(sub.params, args):
                text = arg.strip()
                plain_var = re.fullmatch(r"[!%]\w+", text) is not None
                if plain_var and text.lower() not in tr.py and text.lower() not in tr.unset and not tr.lost and not tr.runtime:
                    # an argument the subroutine sets (EViews makes the variable): it starts empty
                    var, _ = tr._name(text.lower(), f"{'string' if text[0] == '%' else 'control'} {text.lower()}")
                    from .data import _UNKNOWN, _Py

                    tr.py[text.lower()] = _Py(var, "string" if text[0] == "%" else "control", _UNKNOWN)
                    codes.append('""' if text[0] == "%" else 'float("nan")')
                    notes.append(f"{text} is set by {sub.name}")
                else:
                    code, _, vkind, vnotes = tr.program_value(text, st.rest_at)
                    if kind == "string" and vkind != "str":
                        raise _left(f"{pname} is a string parameter: {text} is not a string")
                    if kind == "scalar" and vkind == "str":
                        raise _left(f"{pname} is a scalar parameter: {text} is a string")
                    codes.append(code)
                    notes += [x for x in vnotes if x not in notes]
                targets.append(tr.py[text.lower()].name if plain_var and text.lower() in tr.py else None)
        except _Out as out:
            tr._unset(st)
            self._unknown(_assigned([sub.node], self.subs))
            tr.forget()
            return Outcome(st, out.status, out.reason, file=n.file)
        self.translate_sub(sub)
        if sub.error:
            self._unknown(_assigned([sub.node], self.subs))
            tr.forget()
            return Outcome(st, LEFT, f"the subroutine {sub.name} is left out: {sub.error}", file=n.file)
        back = tuple(targets[i] for i in sub.returns)
        # what the subroutine assigns, through its parameters and (not local) the program's variables
        changed = {a.strip().lower() for a, i in zip(args, range(len(args))) if i in sub.returns}
        if not sub.local:
            changed |= _assigned(sub.node.body, self.subs)
        self._unknown(changed)
        if any(t is None for t in back):
            notes.append("an argument that is not a variable does not take back the value the subroutine gives its "
                         "parameter, as in EViews")
        return Outcome(st, NOTE if notes else TRANSLATED, "; ".join(notes), [Invoke(sub.var, tuple(codes), back)], file=n.file)

    def translate_sub(self, sub: _Sub) -> None:
        """The subroutine's body, translated once: at its first call, or after the program."""
        from .data import _UNKNOWN, _Py

        if sub.done or sub.translating:
            return
        tr = self.tr
        sub.translating = True
        if not sub.var:
            sub.var, _ = tr._name(sub.name, f"subroutine {sub.name}")
        saved_py = dict(tr.py) if sub.local else {k: v for k, v in tr.py.items()}
        saved = (self.sub, self.loops)
        self.sub, self.loops = sub, 0
        params: list[str] = []
        for kind, pname, key in sub.params:
            var, _ = tr._name(pname, f"parameter {sub.name}.{key}")
            params.append(var)
            tr.py[key] = _Py(var, "string" if kind == "string" else "control" if pname.startswith("!") else "scalar", _UNKNOWN)
        if sub.local:  # a local subroutine sees only its parameters
            for k in list(tr.py):
                if k not in {key for _, _, key in sub.params}:
                    del tr.py[k]
        sub.pnames = params
        assigned = _assigned(sub.node.body, self.subs)
        sub.returns = [i for i, (_, _, key) in enumerate(sub.params) if key in assigned]
        self._unknown(assigned)
        with self._runtime():
            sub.outcome.children = self.walk(sub.node.body)
        sub.outcome.closer = self._outcome(sub.node.close, TRANSLATED, f"the end of the subroutine {sub.name}", sub.node.file)
        names = ", ".join(params)
        sub.outcome.reason = "; ".join(x for x in (
            f"a function: def {sub.var}({names})" + (" (local: it shares none of the program's variables)" if sub.local else ""),
            f"it returns {', '.join(params[i] for i in sub.returns)}, which EViews passes back to the caller's variables"
            if sub.returns else "") if x)
        sub.outcome.status = NOTE
        sub.outcome.ops = [Def(sub.var, tuple(params), local=sub.local)]  # the report's Python: the function's first line
        # the parameters are the function's own: the program's variables of those names come back
        for _, _, key in sub.params:
            if key in saved_py:
                tr.py[key] = saved_py[key]
            else:
                tr.py.pop(key, None)
        if sub.local:
            tr.py.clear()
            tr.py.update(saved_py)
        self.sub, self.loops = saved
        sub.translating = False
        sub.done = True

    def finish(self) -> None:
        """The subroutines nothing calls: translated after the program, as it stands at its end."""
        for sub in list(self.subs.values()):
            if not sub.done and not sub.error and sub.outcome is not None:
                self.translate_sub(sub)
                sub.outcome.reason += "; nothing calls it: translated after the program"

    # -- blocks ---------------------------------------------------------------------------------------------------

    def _left_block(self, n: Node, status: str, reason: str) -> list[Any]:
        from .data import Outcome

        self.tr.forget()
        self._unknown(_assigned([n], self.subs))
        o = Outcome(n.st, status, reason, file=n.file)
        o.children = self._inside(n, LEFT, f"inside the {_WORDS[n.kind]} at line {n.st.line}, which is left out")
        return [o]

    def block(self, n: Node) -> Any:
        from .data import _Out

        tr = self.tr
        try:
            if "$" in re.sub(r'"[^"]*"', "", n.st.text):
                raise _left(_DOLLAR)
            if n.kind == "for":
                header, note = self.for_header(n.st)
            elif n.kind == "while":
                code, _, kind, notes = tr.program_value(n.st.rest, n.st.rest_at)
                header, note = While(code), "; ".join(notes)
            else:
                text = re.sub(r"(?i)\s+then\s*$", "", n.st.rest)
                if text == n.st.rest:
                    raise _error("if ... then: then ends the condition")
                code, _, kind, notes = tr.program_value(text, n.st.rest_at)
                header, note = If(code), "; ".join(notes)
        except _Out as out:
            return self._left_block(n, out.status, out.reason)[0]
        return self._body(n, header, note)

    def _body(self, n: Node, header: Any, note: str) -> Any:
        from .data import Outcome

        tr = self.tr
        assigned = _assigned([n], self.subs)
        self._unknown(assigned)
        o = Outcome(n.st, NOTE if note else TRANSLATED, note, [header], file=n.file, kind=n.kind)
        loop = n.kind in ("for", "while")
        self.loops += loop
        with self._runtime():
            if n.kind == "if":
                before = {k: (set(p.series) if p.series is not None else None, p.sample) for k, p in tr.pages.items()}
                o.children = self.walk(n.body)
                after = {k: (set(p.series) if p.series is not None else None, p.sample) for k, p in tr.pages.items()}
                for k, p in tr.pages.items():
                    if k in before:
                        p.series = set(before[k][0]) if before[k][0] is not None else None
                        p.sample = before[k][1]
                o.else_outcome = self._outcome(n.else_st, TRANSLATED, f"the else branch of the if block at line {n.st.line}", n.file)
                o.orelse = self.walk(n.orelse)
                for k, p in tr.pages.items():
                    if k not in after:
                        continue
                    one, sample = after[k]
                    p.series = None if one is None or p.series is None else p.series | one
                    if sample != p.sample:
                        p.sample = None
                        tr.sample_unknown.add(p.var)
            else:
                o.children = self.walk(n.body)
        self.loops -= loop
        self._unknown(assigned)
        o.closer = self._outcome(n.close, TRANSLATED, f"the end of the {_WORDS[n.kind]} at line {n.st.line}", n.file)
        if n.kind == "while":
            o.reason = "; ".join(x for x in (o.reason, f"steps mode stops after {WHILE_CAP:,} passes (run(while_cap=...))") if x)
            o.status = NOTE
        return o

    def for_header(self, st: Statement) -> tuple[Any, str]:
        from .data import _UNKNOWN, _Py

        tr = self.tr
        text = st.rest
        m = re.fullmatch(r"\s*(![A-Za-z_]\w*)\s*=\s*(.+?)\s+to\s+(.+?)(?:\s+step\s+(.+?))?\s*", text, re.I | re.S)
        notes: list[str] = []
        if m:
            key = m.group(1).lower()
            bounds = []
            for part in (m.group(2), m.group(3), m.group(4)):
                if part is None:
                    bounds.append(None)
                    continue
                code, value, kind, ns = tr.program_value(part, st.rest_at)
                if kind == "str":
                    raise _error(f"{part.strip()} is a string, not a number")
                notes += [x for x in ns if x not in notes]
                bounds.append((code, value))
            v = tr.py.get(key)
            if v is None:
                var, _ = tr._name(key, f"control {key}")
                tr.py[key] = _Py(var, "control", _UNKNOWN)
            elif v.kind not in ("control", "scalar"):
                raise _error(f"{m.group(1)} is a {v.kind}, not a number")
            var = tr.py[key].name
            start, end, step = bounds

            def as_int(b: tuple[str, Any]) -> str:
                code, value = b
                if re.fullmatch(r"-?\d+", code):
                    return code
                if value is not _UNKNOWN and isinstance(value, (int, float)) and float(value).is_integer():
                    return f"int({code})" if not re.fullmatch(r"len\(.*\)", code) else code
                return code if re.fullmatch(r"len\(.*\)", code) else f"int({code})"

            s, e = as_int(start), as_int(end)
            if step is None:
                stop = str(int(e) + 1) if re.fullmatch(r"-?\d+", e) else f"{e} + 1"
                return For(var, f"range({s}, {stop})"), "; ".join(notes)
            k = as_int(step)
            if re.fullmatch(r"-?\d+", k):
                if int(k) == 0:
                    raise _error("a for loop's step is 0")
                d = 1 if int(k) > 0 else -1
                stop = str(int(e) + d) if re.fullmatch(r"-?\d+", e) else f"{e} {'+' if d > 0 else '-'} 1"
                return For(var, f"range({s}, {stop}, {k})"), "; ".join(notes)
            return For(var, f"range({s}, {e} + (1 if {k} > 0 else -1), {k})"), "; ".join(notes)
        m = re.fullmatch(r"\s*(%[A-Za-z_]\w*)\s+(.*)", text, re.S)
        if m:
            key = m.group(1).lower()
            words = re.findall(r'"[^"]*"|\S+', m.group(2))
            if words and re.fullmatch(r"%\w+", words[0]) and len(words) > 1 and all(re.fullmatch(r"%\w+", w) for w in words[:2]) \
                    and words[1].lower() not in tr.py:
                raise _left("a for loop over several string variables at once (for %a %b ...) is not translated")
            items: list[str] = []
            for w in words:
                inner = re.fullmatch(r"\{?(%\w+)\}?", w)
                if inner and (w.startswith("{") == w.endswith("}")):
                    vkey = inner.group(1).lower()
                    v = tr.py.get(vkey) or tr.runtime_variable(vkey)
                    if v is None:
                        raise tr._missing(inner.group(1))
                    if v.kind != "string":
                        raise _error(f"{inner.group(1)} is a {v.kind}, not a string")
                    if v.value is not _UNKNOWN and not tr.runtime:
                        items += [quote(x) for x in str(v.value).split()]
                    else:
                        items.append(f"*{v.name}.split()")
                elif "{" in w or "!" in w or "%" in w:
                    raise _left(f"{w}: a for loop's item made from a variable is not translated; a string variable of names "
                                "({%names}) is")
                else:
                    items.append(quote(w.strip('"')))
            if not items:
                raise _error("a for loop over strings names them: for %s a b c")
            v = tr.py.get(key)
            if v is None:
                var, _ = tr._name(key, f"string {key}")
                tr.py[key] = _Py(var, "string", _UNKNOWN)
            elif v.kind != "string":
                raise _error(f"{m.group(1)} is a {v.kind}, not a string")
            var = tr.py[key].name
            if len(items) == 1 and items[0].startswith("*"):
                return For(var, items[0][1:]), "; ".join(notes)
            return For(var, "[" + ", ".join(items) + "]"), "; ".join(notes)
        raise _error("a for loop reads: for !i = 1 to n [step k], or for %s a b c")


def ops_of(outcomes: list[Any], file_of: Any = None) -> tuple[Any, ...]:
    """The steps of translated outcomes, blocks with their bodies: (line, op, file) entries."""
    import dataclasses

    out: list[Any] = []
    for o in outcomes:
        if o is None:
            continue
        if getattr(o, "kind", "") == "include":
            out.extend(ops_of(o.children))
            continue
        if getattr(o, "kind", "") == "subroutine":
            continue  # functions are collected apart (functions_of)
        if o.status not in (TRANSLATED, NOTE):
            continue
        for op in o.ops:
            if isinstance(op, (For, While)):
                op = dataclasses.replace(op, body=ops_of(o.children))
            elif isinstance(op, If):
                op = dataclasses.replace(op, body=ops_of(o.children), orelse=ops_of(o.orelse))
            out.append((o.statement.line, op, o.file))
    return tuple(out)


def functions_of(flow: Flow) -> list[tuple[int, Def, str]]:
    """The subroutines translated, as functions: (line, Def, file), in the order they are defined."""
    out = []
    seen = set()
    for sub in flow.subs.values():
        if id(sub) in seen or sub.error or not sub.done or sub.outcome is None:
            continue
        seen.add(id(sub))
        body = ops_of(sub.outcome.children)
        shared = _shared(body, sub) if not sub.local else ()
        out.append((sub.node.st.line, Def(sub.var, tuple(sub.pnames), tuple(sub.pnames[i] for i in sub.returns), shared,
                                          sub.local, body), sub.node.file))
    return out


def _shared(body: tuple[Any, ...], sub: _Sub) -> tuple[str, ...]:
    """The names a (not local) subroutine's body assigns other than its parameters: the program's (``global``)."""
    from .script import body_lines

    names: list[str] = []
    for line in body_lines(body, 0):
        m = re.match(r"\s*(?:for\s+([A-Za-z_]\w*)\s+in\b|((?:[A-Za-z_]\w*\s*,\s*)*[A-Za-z_]\w*)\s*=(?!=))", line)
        if m:
            for n in re.split(r"\s*,\s*", (m.group(1) or m.group(2)).strip()):
                if n and n not in sub.pnames and n not in names and n != "_":
                    names.append(n)
    return tuple(names)

