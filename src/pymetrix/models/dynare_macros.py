"""Dynare's macro processor (``@#`` directives), as a text pass before a ``.mod`` file is read.

    text, lines = expand_macros(source, base=Path("model.mod").parent)

``read_mod`` runs it on every file with ``@#`` directives or ``@{...}`` substitutions; the model editor's Dynare
check does too, and reports a macro error at the line of its directive. The subset read, as Dynare 6 reads it:

- ``@#define x = expr``: numbers, strings (``"..."``), booleans (``true``, ``false``), arrays ``[1, 2, 3]``, tuples
  ``(1, "a")``, ranges ``1:5`` and ``1:2:9``, comprehensions ``[i^2 for i in 1:3 when i != 2]``; operators ``+ - * / ^``
  (``+`` joins strings and arrays), comparisons, ``&&``, ``||``, ``!``, ``in``, indexing ``a[2]`` and ``a[1:2]``
  (from 1), and the functions ``length``, ``isempty``, ``defined``, ``exp``, ``ln``/``log``, ``log10``, ``sqrt``,
  ``abs``, ``floor``, ``ceil``, ``round``, ``trunc``, ``mod``, ``max``, ``min``, ``sin``, ``cos``, ``tan``, ``sum``,
  ``gamma``, ``erf``, ``erfc``, ``normpdf`` and ``normcdf`` (``(x)`` or ``(x, mu, sigma)``); and macro functions
  ``@#define f(a, b) = expr``.
- Operations on arrays, as Dynare 6: union ``a | b`` (``a``, then the elements of ``b`` not in it), intersection
  ``a & b`` (the elements of ``b`` that are in ``a``), difference ``a - b`` (the elements of ``a`` not in ``b``),
  Cartesian product ``a * b`` and power ``a ^ n`` (arrays of tuples). ``&`` binds tighter than ``|``, and both
  looser than ``+`` and ``-``. Type tests ``isreal``, ``isstring``, ``isboolean``, ``istuple``, ``isarray``, and
  casts ``(real)``, ``(string)``, ``(bool)``, ``(tuple)``, ``(array)``. Equality is of type and value
  (``1 == true`` is false).
- ``@{expr}`` anywhere in the text: the value written out (integers without a decimal point, strings without quotes).
- ``@#if``, ``@#ifdef``, ``@#ifndef``, ``@#elseif``, ``@#else``, ``@#endif``; ``@#for x in expr``, ``@#for (a, b) in
  expr``, with ``when cond``, to ``@#endfor``.
- ``@#include "file"`` (relative to the model file's folder, then the folders of ``@#includepath``); ``@#echo``
  (a warning) and ``@#error`` (an error); ``@#echomacrovars`` (a warning listing the macro variables); ``@#line
  "file" n`` is accepted and changes nothing here (in Dynare it renames the place its error messages give: pymetrix
  keeps reporting the line of the model file, where the editor shows it).
- A directive is a line of its own; a line ending in ``\\`` continues on the next.

Anything else after ``@#`` is refused with its name and line, as are unclosed blocks and includes not found.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .solvers import ModelError

__all__ = ["MacroError", "expand_macros"]

DIRECTIVES = ("define", "if", "ifdef", "ifndef", "elseif", "else", "endif", "for", "endfor", "include", "includepath",
              "echo", "error", "echomacrovars", "line")
CASTS = ("real", "string", "bool", "tuple", "array")


class MacroError(ModelError):
    """A problem in the macro-processor directives, at ``line`` of the model file (1 for the first)."""

    def __init__(self, message: str, line: int = 0) -> None:
        super().__init__(message)
        self.line = line


# ---------------------------------------------------------------------------------------------
# expressions

_TOKEN = re.compile(r"""\s*(?:
    (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<str>"[^"]*"|'[^']*')
  | (?P<name>[A-Za-z_]\w*)
  | (?P<op>\|\||&&|==|!=|<=|>=|[<>!+\-*/^:()\[\],|&])
)""", re.X)


def _tokens(text: str, line: int) -> list[tuple[str, Any]]:
    out, pos = [], 0
    while pos < len(text):
        if not text[pos:].strip():
            break
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise MacroError(f"line {line}: cannot read the macro expression at {text[pos:].strip()[:20]!r}", line)
        kind = m.lastgroup
        value = m.group(kind)
        out.append((kind, float(value) if kind == "num" else value[1:-1] if kind == "str" else value))
        pos = m.end()
    out.append(("end", None))
    return out


class _Parser:
    """A Pratt parser for the macro language: expressions as nested tuples."""

    def __init__(self, text: str, line: int) -> None:
        self.toks, self.i, self.line, self.text = _tokens(text, line), 0, line, text

    def peek(self) -> tuple[str, Any]:
        return self.toks[self.i]

    def take(self) -> tuple[str, Any]:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def expect(self, op: str) -> None:
        kind, value = self.take()
        if value != op or kind not in ("op", "name"):
            raise MacroError(f"line {self.line}: expected {op!r} in the macro expression {self.text.strip()!r}", self.line)

    def at(self, *ops: str) -> bool:
        kind, value = self.peek()
        return kind in ("op", "name") and value in ops

    def done(self) -> None:
        if self.peek()[0] != "end":
            raise MacroError(f"line {self.line}: unexpected {self.peek()[1]!r} in the macro expression {self.text.strip()!r}", self.line)

    def expr(self) -> Any:
        return self.binary(0)

    _LEVELS = (("||",), ("&&",), ("==", "!="), ("<", ">", "<=", ">="), ("in",), (":",), ("|",), ("&",), ("+", "-"), ("*", "/"))

    def binary(self, level: int) -> Any:
        if level == len(self._LEVELS):
            return self.unary()
        ops = self._LEVELS[level]
        left = self.binary(level + 1)
        if ops == (":",):  # a range: a:b or a:step:b
            if self.at(":"):
                self.take()
                middle = self.binary(level + 1)
                if self.at(":"):
                    self.take()
                    return ("range", left, middle, self.binary(level + 1))
                return ("range", left, None, middle)
            return left
        if ops == ("in",):
            if self.at("in"):
                self.take()
                return ("bin", "in", left, self.binary(level + 1))
            return left
        while self.at(*ops):
            op = self.take()[1]
            left = ("bin", op, left, self.binary(level + 1))
        return left

    def unary(self) -> Any:
        if self.at("-", "+", "!"):
            op = self.take()[1]
            return ("unary", op, self.unary())
        base = self.postfix()
        if self.at("^"):
            self.take()
            return ("bin", "^", base, self.unary())
        return base

    def postfix(self) -> Any:
        node = self.primary()
        while self.at("["):
            self.take()
            index = self.expr()
            self.expect("]")
            node = ("index", node, index)
        return node

    def primary(self) -> Any:
        kind, value = self.take()
        if kind == "num":
            return ("lit", value)
        if kind == "str":
            return ("lit", value)
        if kind == "name":
            if value in ("true", "false"):
                return ("lit", value == "true")
            if self.at("("):
                self.take()
                args = [] if self.at(")") else self._list(")")
                self.expect(")")
                return ("call", value, args)
            return ("name", value)
        if value == "(":
            nxt, after = self.peek(), self.toks[self.i + 1] if self.i + 1 < len(self.toks) else ("end", None)
            if nxt[0] == "name" and nxt[1] in CASTS and after == ("op", ")"):  # a cast: (real)"3.5"
                self.i += 2
                return ("cast", nxt[1], self.unary())
            first = self.expr()
            if self.at(","):
                self.take()
                rest = [] if self.at(")") else self._list(")")  # (5,): a tuple of one
                self.expect(")")
                return ("tuple", [first, *rest])
            self.expect(")")
            return first
        if value == "[":
            if self.at("]"):
                self.take()
                return ("array", [])
            first = self.expr()
            if self.at("for"):  # [expr for x in array when cond]
                self.take()
                names = self._loop_names()
                self.expect("in")
                source = self.expr()
                cond = None
                if self.at("when"):
                    self.take()
                    cond = self.expr()
                self.expect("]")
                return ("comprehension", first, names, source, cond)
            items = [first]
            if self.at(","):
                self.take()
                items += self._list("]")
            self.expect("]")
            return ("array", items)
        raise MacroError(f"line {self.line}: unexpected {value!r} in the macro expression {self.text.strip()!r}", self.line)

    def _list(self, closing: str) -> list[Any]:
        items = [self.expr()]
        while self.at(","):
            self.take()
            items.append(self.expr())
        return items

    def _loop_names(self) -> list[str]:
        if self.at("("):
            self.take()
            names = [self._name()]
            while self.at(","):
                self.take()
                names.append(self._name())
            self.expect(")")
            return names
        return [self._name()]

    def _name(self) -> str:
        kind, value = self.take()
        if kind != "name":
            raise MacroError(f"line {self.line}: expected a name in {self.text.strip()!r}", self.line)
        return value


@dataclass(frozen=True)
class _Function:
    params: tuple[str, ...]
    body: Any


def _fn(f: Any, n: int | None = 1) -> Any:
    return (f, n)


_FUNCS: dict[str, tuple[Any, int | None]] = {
    "exp": _fn(math.exp), "ln": _fn(math.log), "log": _fn(math.log), "log10": _fn(math.log10), "sqrt": _fn(math.sqrt),
    "abs": _fn(abs), "floor": _fn(lambda x: float(math.floor(x))), "ceil": _fn(lambda x: float(math.ceil(x))),
    "round": _fn(lambda x: float(math.floor(x + 0.5)) if x >= 0 else -float(math.floor(-x + 0.5))),
    "trunc": _fn(lambda x: float(math.trunc(x))), "sin": _fn(math.sin), "cos": _fn(math.cos), "tan": _fn(math.tan),
    "mod": _fn(math.fmod, 2), "max": _fn(max, None), "min": _fn(min, None), "gamma": _fn(math.gamma), "erf": _fn(math.erf),
    "erfc": _fn(math.erfc),
}


def _truth(value: Any, line: int, what: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value != 0
    raise MacroError(f"line {line}: {what} is {_text(value, quote=True)}, not a boolean or a number", line)


def _number(value: Any, line: int, op: str) -> float:
    if isinstance(value, bool) or not isinstance(value, float):
        raise MacroError(f"line {line}: {op} needs numbers, not {_text(value, quote=True)}", line)
    return value


def _eval(node: Any, env: dict[str, Any], line: int) -> Any:
    kind = node[0]
    if kind == "lit":
        return node[1]
    if kind == "name":
        if node[1] not in env or isinstance(env[node[1]], _Function):
            raise MacroError(f"line {line}: the macro variable {node[1]} is not defined (@#define {node[1]} = ...)", line)
        return env[node[1]]
    if kind == "array":
        return [_eval(n, env, line) for n in node[1]]
    if kind == "tuple":
        return tuple(_eval(n, env, line) for n in node[1])
    if kind == "range":
        low = _number(_eval(node[1], env, line), line, "a range")
        step = 1.0 if node[2] is None else _number(_eval(node[2], env, line), line, "a range")
        high = _number(_eval(node[3], env, line), line, "a range")
        if step == 0:
            raise MacroError(f"line {line}: a range with a step of 0", line)
        count = int(math.floor((high - low) / step + 1e-10)) + 1
        return [low + i * step for i in range(max(count, 0))]
    if kind == "comprehension":
        _, body, names, source, cond = node
        out = []
        for item in _iterable(_eval(source, env, line), line):
            inner = {**env, **_bind(names, item, line)}
            if cond is None or _truth(_eval(cond, inner, line), line, "the condition after when"):
                out.append(_eval(body, inner, line))
        return out
    if kind == "unary":
        value = _eval(node[2], env, line)
        if node[1] == "!":
            return not _truth(value, line, "the operand of !")
        value = _number(value, line, f"unary {node[1]}")
        return -value if node[1] == "-" else value
    if kind == "index":
        return _index(_eval(node[1], env, line), _eval(node[2], env, line), line)
    if kind == "call":
        return _call(node[1], node[2], env, line)
    if kind == "cast":
        return _cast(node[1], _eval(node[2], env, line), line)
    op, a = node[1], _eval(node[2], env, line)
    if op in ("&&", "||"):  # short-circuit, as Dynare
        left = _truth(a, line, f"the left of {op}")
        if (op == "&&" and not left) or (op == "||" and left):
            return left
        return _truth(_eval(node[3], env, line), line, f"the right of {op}")
    b = _eval(node[3], env, line)
    if op == "in":
        return any(_same(a, x) for x in _iterable(b, line))
    if op == "==":
        return _same(a, b)
    if op == "!=":
        return not _same(a, b)
    if op in ("|", "&") or (op in ("-", "*", "^") and isinstance(a, list)):
        return _set_operation(op, a, b, line)
    if op in ("<", ">", "<=", ">="):
        if type(a) is not type(b) or not isinstance(a, (float, str)):
            raise MacroError(f"line {line}: {op} compares two numbers or two strings, not {_text(a, True)} and {_text(b, True)}", line)
        return {"<": a < b, ">": a > b, "<=": a <= b, ">=": a >= b}[op]
    if op == "+":
        if isinstance(a, str) and isinstance(b, str):
            return a + b
        if isinstance(a, list) and isinstance(b, list):
            return a + b
        return _number(a, line, "+") + _number(b, line, "+")
    x, y = _number(a, line, op), _number(b, line, op)
    if op == "-":
        return x - y
    if op == "*":
        return x * y
    if op == "/":
        if y == 0:
            raise MacroError(f"line {line}: division by zero in a macro expression", line)
        return x / y
    return x ** y


def _same(a: Any, b: Any) -> bool:
    """Equality as Dynare's: of type and value (``1 == true`` is false), element by element in arrays and tuples."""
    if isinstance(a, (list, tuple)) and type(a) is type(b):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _set_operation(op: str, a: Any, b: Any, line: int) -> list[Any]:
    """Dynare 6's operations on arrays: union, intersection, difference, Cartesian product and power."""
    if op == "^":
        n = _number(b, line, "^ on an array")
        if not float(n).is_integer():
            raise MacroError(f"line {line}: the power of an array is a whole number, not {_text(n)}", line)
        out = list(a)
        for _ in range(int(n) - 1):
            out = _set_operation("*", out, a, line)
        return out
    if not isinstance(a, list) or not isinstance(b, list):
        raise MacroError(f"line {line}: {op} takes two arrays, not {_text(a, True)} and {_text(b, True)}", line)
    if op == "|":
        out = list(a)
        for x in b:
            if not any(_same(x, y) for y in out):
                out.append(x)
        return out
    if op == "&":
        return [x for x in b if any(_same(x, y) for y in a)]
    if op == "-":
        return [x for x in a if not any(_same(x, y) for y in b)]
    flat = lambda x: x if isinstance(x, tuple) else (x,)  # noqa: E731  (tuples are spliced: (a*b)*c gives triples)
    return [flat(x) + flat(y) for x in a for y in b]


def _cast(kind: str, value: Any, line: int) -> Any:
    """``(real)``, ``(string)``, ``(bool)``, ``(tuple)``, ``(array)`` as Dynare 6 casts: an array or tuple of one element
    to a real or a boolean as that element; strings to reals by reading them, to booleans if "true" or "false"."""
    def fail() -> MacroError:
        return MacroError(f"line {line}: ({kind}) cannot cast {_text(value, True)}", line)

    if kind in ("real", "bool") and isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise fail()
        value = value[0]
    if kind == "real":
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, float):
            return value
        try:
            return float(str(value).strip())
        except ValueError:
            raise fail() from None
    if kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, float):
            return value != 0
        if isinstance(value, str) and value.strip() in ("true", "false"):
            return value.strip() == "true"
        raise fail()
    if kind == "string":
        return _text(value)
    if kind == "tuple":
        return tuple(value) if isinstance(value, (list, tuple)) else (value,)
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _iterable(value: Any, line: int) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    raise MacroError(f"line {line}: {_text(value, True)} is not an array to loop over", line)


def _bind(names: list[str], item: Any, line: int) -> dict[str, Any]:
    if len(names) == 1:
        return {names[0]: item}
    if not isinstance(item, (tuple, list)) or len(item) != len(names):
        raise MacroError(f"line {line}: ({', '.join(names)}) takes tuples of {len(names)}, not {_text(item, True)}", line)
    return dict(zip(names, item))


def _index(value: Any, index: Any, line: int) -> Any:
    if not isinstance(value, (list, tuple, str)):
        raise MacroError(f"line {line}: {_text(value, True)} cannot be indexed", line)
    positions = index if isinstance(index, list) else [index]
    out = []
    for p in positions:
        p = _number(p, line, "an index")
        if not float(p).is_integer() or not 1 <= p <= len(value):
            raise MacroError(f"line {line}: index {_text(p)} is outside 1 to {len(value)}", line)
        out.append(value[int(p) - 1])
    if isinstance(index, list):
        return "".join(out) if isinstance(value, str) else out
    return out[0]


def _call(name: str, args: list[Any], env: dict[str, Any], line: int) -> Any:
    if name == "defined":
        if len(args) != 1 or args[0][0] != "name":
            raise MacroError(f"line {line}: defined(name) takes the name of a macro variable", line)
        return args[0][1] in env
    values = [_eval(a, env, line) for a in args]
    fn = env.get(name)
    if isinstance(fn, _Function):
        if len(values) != len(fn.params):
            raise MacroError(f"line {line}: the macro function {name} takes {len(fn.params)} arguments, not {len(values)}", line)
        return _eval(fn.body, {**env, **dict(zip(fn.params, values))}, line)
    tests = {"isreal": float, "isstring": str, "isboolean": bool, "istuple": tuple, "isarray": list}
    if name in tests:
        if len(values) != 1:
            raise MacroError(f"line {line}: {name} takes one argument", line)
        return type(values[0]) is tests[name]
    if name == "sum":
        if len(values) != 1 or not isinstance(values[0], list):
            raise MacroError(f"line {line}: sum takes one array of numbers", line)
        return float(sum(_number(v, line, "sum") for v in values[0]))
    if name in ("normpdf", "normcdf"):
        if len(values) not in (1, 3):
            raise MacroError(f"line {line}: {name} takes (x) or (x, mu, sigma)", line)
        x, mu, sigma = [_number(v, line, name) for v in values] + ([0.0, 1.0] if len(values) == 1 else [])
        z = (x - mu) / sigma
        if name == "normcdf":
            return 0.5 * math.erfc(-z / math.sqrt(2.0))
        return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))
    if name in ("length", "isempty"):
        if len(values) != 1 or not isinstance(values[0], (list, tuple, str)):
            raise MacroError(f"line {line}: {name} takes one array or string", line)
        return float(len(values[0])) if name == "length" else len(values[0]) == 0
    if name not in _FUNCS:
        raise MacroError(f"line {line}: {name} is not a macro function", line)
    f, n = _FUNCS[name]
    if name in ("max", "min") and len(values) == 1 and isinstance(values[0], list):
        values = values[0]
    if n is not None and len(values) != n:
        raise MacroError(f"line {line}: {name} takes {n} argument{'s' if n > 1 else ''}", line)
    try:
        return float(f(*[_number(v, line, name) for v in values]))
    except (ValueError, OverflowError) as exc:
        raise MacroError(f"line {line}: {name}: {exc}", line) from None


def _text(value: Any, quote: bool = False) -> str:
    """A value as ``@{...}`` writes it, as Dynare 6: numbers to 15 significant digits (``%.15g``), strings without
    quotes, also inside arrays and tuples. ``quote``: strings in quotes, for messages."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.15g}"
    if isinstance(value, str):
        return f'"{value}"' if quote else value
    if isinstance(value, list):
        return "[" + ", ".join(_text(v, quote) for v in value) + "]"
    if isinstance(value, tuple):
        return "(" + ", ".join(_text(v, quote) for v in value) + ")"
    return str(value)


def evaluate(text: str, env: dict[str, Any] | None = None, line: int = 0) -> Any:
    """A macro expression's value (numbers as floats, arrays as lists)."""
    p = _Parser(text, line)
    node = p.expr()
    p.done()
    return _eval(node, dict(env or {}), line)


# ---------------------------------------------------------------------------------------------
# directives

_DIRECTIVE = re.compile(r"^\s*@#\s*([A-Za-z_]\w*)?(.*)$")


def _lines(text: str) -> list[tuple[str, int]]:
    """The lines with their numbers, directive lines ending in ``\\`` joined to the next."""
    out: list[tuple[str, int]] = []
    raw = text.split("\n")
    i = 0
    while i < len(raw):
        line, number = raw[i], i + 1
        if _DIRECTIVE.match(line):
            while line.rstrip().endswith("\\") and i + 1 < len(raw):
                i += 1
                line = line.rstrip()[:-1] + " " + raw[i]
        out.append((line, number))
        i += 1
    return out


def _parse(lines: list[tuple[str, int]], origin: int | None) -> list[Any]:
    """The directives as a tree: ("text", line, n), ("if", [(kind, arg, body, n)], else_body), ("for", head, body, n),
    and leaves ("define" | "include" | ..., arg, n). ``origin`` is the line of the include, for an included file."""
    root: list[Any] = []
    stack: list[tuple[str, Any, int]] = []  # (kind, node, line opened)

    def where(n: int) -> int:
        return origin if origin is not None else n

    def current() -> list[Any]:
        if not stack:
            return root
        kind, node, _ = stack[-1]
        if kind == "if" and node[2] is None:  # in an @#if or @#elseif branch; else in the @#else or the @#for
            return node[1][-1][2]
        return node[2]

    for text, n in lines:
        m = _DIRECTIVE.match(text)
        if not m:
            current().append(("text", text, where(n)))
            continue
        name, arg = m.group(1) or "", m.group(2).strip()
        at = where(n)
        if name not in DIRECTIVES:
            raise MacroError(f"line {at}: unknown macro directive @#{name or arg[:10]} (Dynare's preprocessor reads "
                             f"{', '.join('@#' + d for d in DIRECTIVES)})", at)
        if name in ("if", "ifdef", "ifndef"):
            node = ["if", [(name, arg, [], at)], None]
            current().append(node)
            stack.append(("if", node, at))
        elif name in ("elseif", "else"):
            if not stack or stack[-1][0] != "if" or stack[-1][1][2] is not None:
                raise MacroError(f"line {at}: @#{name} without an open @#if", at)
            node = stack[-1][1]
            if name == "elseif":
                node[1].append(("elseif", arg, [], at))
            else:
                node[2] = []
        elif name == "endif":
            if not stack or stack[-1][0] != "if":
                raise MacroError(f"line {at}: @#endif without an open @#if"
                                 + (f" (the @#for of line {stack[-1][2]} is still open)" if stack else ""), at)
            stack.pop()
        elif name == "for":
            node = ["for", (arg, at), []]
            current().append(node)
            stack.append(("for", node, at))
        elif name == "endfor":
            if not stack or stack[-1][0] != "for":
                raise MacroError(f"line {at}: @#endfor without an open @#for"
                                 + (f" (the @#if of line {stack[-1][2]} is still open)" if stack else ""), at)
            stack.pop()
        else:
            current().append((name, arg, at))
    if stack:
        kind, _, opened = stack[-1]
        closing = "@#endif" if kind == "if" else "@#endfor"
        raise MacroError(f"line {opened}: the @#{kind} of line {opened} is not closed (no {closing})", opened)
    return root


@dataclass
class _State:
    env: dict[str, Any]
    base: Path | None
    paths: list[Path]
    out: list[str]
    lines: list[int]
    depth: int = 0


_SUBST = re.compile(r"@\{")


def _substitute(text: str, env: dict[str, Any], line: int) -> str:
    out, pos = [], 0
    for m in _SUBST.finditer(text):
        if m.start() < pos:
            continue
        close = text.find("}", m.end())
        if close < 0:
            raise MacroError(f"line {line}: @{{ without its closing }}", line)
        out.append(text[pos:m.start()])
        out.append(_text(evaluate(text[m.end():close], env, line)))
        pos = close + 1
    out.append(text[pos:])
    return "".join(out)


def _run(nodes: list[Any], st: _State) -> None:
    for node in nodes:
        kind = node[0]
        if kind == "text":
            st.out.append(_substitute(node[1], st.env, node[2]))
            st.lines.append(node[2])
        elif kind == "if":
            chosen = node[2]
            for which, arg, body, at in node[1]:
                if which in ("ifdef", "ifndef"):
                    if not re.fullmatch(r"[A-Za-z_]\w*", arg):
                        raise MacroError(f"line {at}: @#{which} takes the name of a macro variable, not {arg!r}", at)
                    test = (arg in st.env) == (which == "ifdef")
                else:
                    test = _truth(evaluate(arg, st.env, at), at, f"the condition of @#{which}")
                if test:
                    chosen = body
                    break
            _run(chosen or [], st)
        elif kind == "for":
            arg, at = node[1]
            m = re.match(r"^\s*(\(\s*[A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*\s*\)|[A-Za-z_]\w*)\s+in\s+(.*)$", arg)
            if not m:
                raise MacroError(f"line {at}: @#for reads 'x in array' or '(a, b) in array', not {arg!r}", at)
            names = re.findall(r"[A-Za-z_]\w*", m.group(1))
            p = _Parser(m.group(2), at)
            source = p.expr()
            cond = None
            if p.at("when"):
                p.take()
                cond = p.expr()
            p.done()
            saved = {n: st.env[n] for n in names if n in st.env}
            for item in _iterable(_eval(source, st.env, at), at):
                st.env.update(_bind(names, item, at))
                if cond is None or _truth(_eval(cond, st.env, at), at, "the condition after when"):
                    _run(node[2], st)
            for n in names:
                st.env.pop(n, None)
            st.env.update(saved)
        elif kind == "define":
            arg, at = node[1], node[2]
            m = re.match(r"^([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?\s*=\s*(.+)$", arg, re.S)
            if not m:
                raise MacroError(f"line {at}: @#define reads 'name = expression' (or 'f(a, b) = expression'), not {arg!r}", at)
            if m.group(2) is not None:
                params = tuple(p.strip() for p in m.group(2).split(",") if p.strip())
                p = _Parser(m.group(3), at)
                body = p.expr()
                p.done()
                st.env[m.group(1)] = _Function(params, body)
            else:
                st.env[m.group(1)] = evaluate(m.group(3), st.env, at)
        elif kind in ("include", "includepath"):
            arg, at = node[1], node[2]
            value = evaluate(arg, st.env, at)
            if not isinstance(value, str):
                raise MacroError(f"line {at}: @#{kind} takes a file name in quotes, not {_text(value, True)}", at)
            if kind == "includepath":
                folder = Path(value)
                st.paths.append(folder if folder.is_absolute() or st.base is None else st.base / folder)
                continue
            if st.base is None and not Path(value).is_absolute():
                raise MacroError(f"line {at}: @#include \"{value}\" needs the model read from its file (read_mod(path)), "
                                 "so that the included file can be found next to it", at)
            candidates = [Path(value)] if Path(value).is_absolute() else [d / value for d in [st.base, *st.paths] if d is not None]
            found = next((c for c in candidates if c.is_file()), None)
            if found is None:
                raise MacroError(f"line {at}: the included file {value} was not found (looked in "
                                 f"{', '.join(str(c.parent) for c in candidates)})", at)
            if st.depth > 50:
                raise MacroError(f"line {at}: includes nested more than 50 deep (does {value} include itself?)", at)
            included = _blank_comments(found.read_text(encoding="utf-8"))
            st.depth += 1
            _run(_parse(_lines(included), at), st)
            st.depth -= 1
        elif kind == "line":  # @#line "file" n: where Dynare's messages say they are; the model file's lines are kept
            if not re.fullmatch(r'"[^"]*"\s+\d+', node[1]):
                raise MacroError(f'line {node[2]}: @#line reads \'"file" n\', not {node[1]!r}', node[2])
        elif kind == "echo":
            warnings.warn(f"@#echo (line {node[2]}): {_text(evaluate(node[1], st.env, node[2]))}", UserWarning, stacklevel=4)
        elif kind == "error":
            raise MacroError(f"line {node[2]}: @#error: {_text(evaluate(node[1], st.env, node[2]))}", node[2])
        elif kind == "echomacrovars":
            shown = {k: v for k, v in st.env.items() if not isinstance(v, _Function)}
            warnings.warn("@#echomacrovars: " + ", ".join(f"{k} = {_text(v, True)}" for k, v in shown.items()), UserWarning, stacklevel=4)


def _blank_comments(text: str) -> str:
    """Comments (``//``, ``/* */``, ``%`` at the start of a line) as spaces, so lines stay where they were."""
    def blank(m: re.Match) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))
    return re.sub(r"(?ms)/\*.*?\*/|//[^\n]*|^[ \t]*%[^\n]*", blank, text)


def has_macros(text: str) -> bool:
    """Whether the text has macro directives or ``@{...}`` substitutions."""
    return "@#" in text or "@{" in text


def expand_macros(text: str, base: str | Path | None = None, defines: dict[str, Any] | None = None) -> tuple[str, list[int]]:
    """The text with its macro directives carried out, and for each line of the result the line of ``text`` it
    comes from (for an included file's lines, the line of its ``@#include``). ``base``: the folder included files are
    looked for in (the model file's); None for text without a file, where ``@#include`` is refused. ``defines``:
    macro variables set beforehand (as ``dynare model.mod -DN=2``). Comments are blanked first."""
    st = _State({k: float(v) if isinstance(v, int) and not isinstance(v, bool) else v for k, v in (defines or {}).items()},
                Path(base) if base is not None else None, [], [], [])
    _run(_parse(_lines(_blank_comments(text)), None), st)
    for i, out in enumerate(st.out):
        if "@#" in out:
            raise MacroError(f"line {st.lines[i]}: a macro directive must start its line: {out.strip()[:40]!r}", st.lines[i])
    return "\n".join(st.out), st.lines
