"""A translation's intermediate representation: what each statement of a program in another language becomes, from
which both the readable script (``script.py``) and the steps (``steps.py``) are made, so the two cannot drift apart.

- ``Text``: a pymetrix expression or condition as text, with the program's scalars in it by name. The script writes
  it as an f-string (``f"x*({s})"``); the steps put in the value the scalar has when the step runs.
- ``Code``: Python that is not a step: a scalar, a string, a matrix. The script has it as it is; steps mode runs the
  same text.
- ``Step``: a recorded call of pymetrix's API (``make_call``), with the variable it is made on and the one it makes.
- ``Cmd``: a recorded command that is not a call (``AddToDataset``: a fit's residuals or a forecast added to a page).

A step's values may also be ``New`` (a pymetrix object made from its arguments: ``Equation("y ~ x")``, ``OLS(cov="white")``),
``Attr`` (an attribute of a variable's value: ``m.model``), ``RefTo`` (an object of the session by its name, kept by
reference in the record: ``Scenario(adjust=af)``), ``Py`` (a Python expression worked out when the step runs:
``main.sample``) and ``Spread`` (a string variable's words, spread over a list: ``*names.split()``).

Control flow (``For``, ``While``, ``If``, ``Def``, ``Invoke``, ``Jump``) holds bodies of the same ops: the script
writes them as Python blocks, and steps mode runs them in Python, each statement inside a step. A name made from the
program's variables (EViews's ``y_{%s}``) is a ``Text`` wherever a name goes: an f-string in the script, the name
made when the step runs. An object so named is ``globals()[f"eq_{c}"]`` in the script.

Nothing here is about one language: the EViews plugin's translator makes these ops, and so can a front end for
another (docs/design/r-translator.md, phase 0).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["Arg", "Attr", "Cmd", "Code", "Def", "For", "If", "Invoke", "Jump", "New", "Op", "Py", "RefTo", "SampleSpec", "Spread",
           "Step", "Text", "Var", "While", "literal", "name_code", "quote", "resolved", "step_code"]


@dataclass(frozen=True)
class Text:
    """Text with scalars in it: ``parts`` are pieces of text and ``Var(name)`` for a Python variable's value."""

    parts: tuple[Any, ...]

    @classmethod
    def plain(cls, text: str) -> Text:
        return cls((text,))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parts if isinstance(p, Var))

    def code(self) -> str:
        """The text as Python: a string literal, or an f-string where it holds variables."""
        if not self.names:
            return quote("".join(self.parts))
        body = "".join(p.replace("{", "{{").replace("}", "}}") if isinstance(p, str) else "{" + p.name + "}" for p in self.parts)
        return "f" + quote(body)

    def value(self, env: Mapping[str, Any]) -> str:
        """The text with each variable's value, as the f-string gives it."""
        return "".join(p if isinstance(p, str) else format(env[p.name]) for p in self.parts)

    def __str__(self) -> str:
        return "".join(p if isinstance(p, str) else "{" + p.name + "}" for p in self.parts)


@dataclass(frozen=True)
class Var:
    """A Python variable's value inside a ``Text``."""

    name: str


def quote(text: str) -> str:
    """Text as a Python string literal, in double quotes."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"' if "\n" not in text else repr(text)


@dataclass(frozen=True)
class SampleSpec:
    """A ``Sample`` to make: its ends (pymetrix periods, None for an open end) and its condition."""

    start: str | None = None
    end: str | None = None
    where: Text | None = None

    def code(self) -> str:
        args = []
        if self.start is not None and self.end is not None:
            args.append(f"({literal(self.start)}, {literal(self.end)})")
        elif self.start is not None:
            args.append(f"start={literal(self.start)}")
        elif self.end is not None:
            args.append(f"end={literal(self.end)}")
        if self.where is not None:
            args.append(f"where={self.where.code()}")
        return f"Sample({', '.join(args)})"

    def value(self, env: Mapping[str, Any]) -> Any:
        from pymetrix import Sample

        start, end = (x.value(env) if isinstance(x, Text) else x for x in (self.start, self.end))
        return Sample(start=start, end=end, where=self.where.value(env) if self.where is not None else None)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(n for x in (self.start, self.end, self.where) if isinstance(x, Text) for n in x.names)


class Op:
    """What a statement becomes."""


@dataclass(frozen=True)
class New:
    """A pymetrix object made from its arguments: ``callee`` is its class or function as the script names it
    (``Equation``, ``OLS``, ``Scenario``), found in ``pymetrix``; ``args`` as a ``Step``'s."""

    callee: str
    args: tuple[Any, ...] = ()

    def code(self) -> str:
        return f"{self.callee}({_arguments(self.args)})"

    def value(self, env: Mapping[str, Any]) -> Any:
        import pymetrix

        positional, named = [], {}
        for a in self.args:
            v = resolved(a.value, env)
            if a.how == "pos":
                positional.append(v)
            elif a.how == "star":
                positional.extend(v)
            elif a.how == "varkw":
                named.update(v)
            else:
                named[a.name] = v
        return getattr(pymetrix, self.callee)(*positional, **named)


@dataclass(frozen=True)
class Attr:
    """An attribute of a variable's value: ``m.model``."""

    base: str
    attr: str

    def code(self) -> str:
        return f"{self.base}.{self.attr}"


@dataclass(frozen=True)
class RefTo:
    """An object of the session by its name, recorded by reference (``Ref``), as the script names it (a ``Text`` for a
    name made when the program runs)."""

    name: Any


@dataclass(frozen=True)
class Py:
    """A Python expression worked out when the step runs (``main.sample``), with the program's variables and the
    session's objects; the script has it as it is."""

    code: str


@dataclass(frozen=True)
class Spread:
    """The words of a string variable, spread over a list (``*names.split()``), as EViews's ``group g {%names}``."""

    name: str


@dataclass(frozen=True)
class Code(Op):
    """Python that is not a step (a scalar, a string, a matrix): ``code`` sets ``names`` (a ``Text`` for a name made
    when the program runs)."""

    code: str
    names: tuple[Any, ...] = ()


@dataclass(frozen=True)
class Arg:
    """One argument of a ``Step``: the parameter's ``name``, its ``value``, and ``how`` it is given: ``"pos"`` by
    position in the script (by name to ``make_call``), ``"star"`` a tuple spread over ``*name``, ``"kw"`` by name,
    ``"varkw"`` a mapping spread over ``**name``."""

    name: str
    value: Any
    how: str = "kw"


@dataclass(frozen=True)
class Cmd(Op):
    """A recorded command that is not a call: ``command`` (an ``AddToDataset``, say), whose ``to_code()`` is its line
    of the script, and the variables it ``uses``."""

    command: Any
    uses: tuple[str, ...] = ()
    fields: Any = None  # {field: Text}: the command's names made when the program runs
    script: str = ""  # its line of the script, when it is not command.to_code()


@dataclass(frozen=True)
class Step(Op):
    """A call of pymetrix's API, recorded in steps mode: ``function`` (its stable ID, ``pymetrix:dataset.with_series``)
    on the variable ``receiver`` (None for a class method), making ``output``; ``callee`` is how the script calls it
    (``data.with_series``, ``Dataset.read``). A value is plain (text, a number, a tuple), a ``Text``, a ``SampleSpec``
    or a ``Var`` (a Python variable's value: a named sample)."""

    function: str
    receiver: Any  # a variable's name (str or Text), None, or a value: New, Attr, Var
    output: Any  # a variable's name (str, or Text when it is made as the program runs)
    callee: str
    args: tuple[Arg, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        """The Python variables the step's values use."""
        out: list[str] = []

        def walk(v: Any) -> None:
            if isinstance(v, Var):
                out.append(v.name)
            elif isinstance(v, RefTo):
                out.extend(v.name.names if isinstance(v.name, Text) else (v.name,))
            elif isinstance(v, Attr):
                out.append(v.base)
            elif isinstance(v, New):
                for a in v.args:
                    walk(a.value)
            elif isinstance(v, (Text, SampleSpec)):
                out.extend(v.names)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    walk(x)
            elif isinstance(v, Mapping):
                for x in v.values():
                    walk(x)
        if isinstance(self.receiver, str):
            out.append(self.receiver)
        elif isinstance(self.receiver, Text):
            out.extend(self.receiver.names)
        elif self.receiver is not None:
            walk(self.receiver)
        for a in self.args:
            walk(a.value)
        return tuple(out)


def literal(value: Any) -> str:
    """A value as Python, in the script."""
    if isinstance(value, (Text, SampleSpec, New, Attr)):
        return value.code()
    if isinstance(value, Var):
        return value.name
    if isinstance(value, RefTo):
        return name_code(value.name)
    if isinstance(value, Py):
        return value.code
    if isinstance(value, Spread):
        return f"*{value.name}.split()"
    if isinstance(value, str):
        return quote(value)
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{literal(k)}: {literal(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        inner = ", ".join(literal(v) for v in value)
        return f"[{inner}]" if isinstance(value, list) else f"({inner}{',' if len(value) == 1 else ''})"
    raise TypeError(f"no Python for {type(value).__name__}")


def resolved(value: Any, env: Mapping[str, Any]) -> Any:
    """A value with its variables' values put in: what steps mode passes to ``make_call``."""
    if isinstance(value, (Text, SampleSpec)):
        return value.value(env)
    if isinstance(value, Var):
        return env[value.name]
    if isinstance(value, New):
        return value.value(env)
    if isinstance(value, Attr):
        return getattr(env[value.base], value.attr)
    if isinstance(value, RefTo):
        from pymetrix.extension import Ref

        return Ref(value.name.value(env) if isinstance(value.name, Text) else value.name)
    if isinstance(value, Py):
        import pymetrix

        return eval(value.code, {n: getattr(pymetrix, n) for n in pymetrix.__all__ if hasattr(pymetrix, n)}, dict(env))  # noqa: S307
    if isinstance(value, Mapping):
        return {resolved(k, env): resolved(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [resolved(v, env) for v in value]
    if isinstance(value, tuple):
        out: list[Any] = []
        for v in value:
            if isinstance(v, Spread):
                out.extend(str(env[v.name]).split())
            else:
                out.append(resolved(v, env))
        return tuple(out)
    return value


def name_code(name: Any) -> str:
    """A variable's name in the script: as it is, or ``globals()[f"eq_{c}"]`` for a name made as the program runs."""
    return f"globals()[{name.code()}]" if isinstance(name, Text) else name


def step_code(step: Step) -> str:
    """A step as one line of the script: ``output = callee(arguments)``."""
    return f"{name_code(step.output)} = {step.callee}({_arguments(step.args)})"


def _arguments(args: tuple[Any, ...]) -> str:
    import keyword

    parts: list[str] = []
    for a in args:
        if a.how == "pos":
            parts.append(literal(a.value))
        elif a.how == "star":
            if isinstance(a.value, (Attr, Py)):
                parts.append(f"*{literal(a.value)}")
            else:
                parts.extend(literal(v) for v in a.value)
        elif a.how == "varkw":
            for k, v in a.value.items():
                if isinstance(k, str) and k.isidentifier() and not keyword.iskeyword(k):
                    parts.append(f"{k}={literal(v)}")
                else:
                    parts.append(f"**{{{literal(k)}: {literal(v)}}}")
        else:
            parts.append(f"{a.name}={literal(a.value)}")
    return ", ".join(parts)


# -- control flow -----------------------------------------------------------------------------------------------------
# A body is a tuple of (line, op, file) entries, as ProgramTranslation.ops is.


@dataclass(frozen=True)
class For(Op):
    """``for var in iterable:`` (``range(1, n + 1)``, ``["a", "b"]``, ``names.split()``)."""

    var: str
    iterable: str
    body: tuple[Any, ...] = ()


@dataclass(frozen=True)
class While(Op):
    """``while cond:``; steps mode stops after a number of passes the user can change."""

    cond: str
    body: tuple[Any, ...] = ()


@dataclass(frozen=True)
class If(Op):
    """``if cond: ... else: ...``, decided when it runs."""

    cond: str
    body: tuple[Any, ...] = ()
    orelse: tuple[Any, ...] = ()


@dataclass(frozen=True)
class Def(Op):
    """A function: its parameters; ``returns``, the parameters it assigns, given back to the caller's variables (as
    EViews passes arguments by reference); or ``result``, Python for the one value it returns (an R function's last
    expression or ``return(x)``), worked out in the function's own variables when it ends or reaches a ``return``;
    ``shared``, the names it assigns at the program's level (``global``); ``local``, whether it sees none of the
    program's names. A function has ``returns`` or a ``result``, not both."""

    name: str
    params: tuple[str, ...]
    returns: tuple[str, ...] = ()
    shared: tuple[str, ...] = ()
    local: bool = False
    body: tuple[Any, ...] = ()
    result: str = ""

    def __post_init__(self) -> None:
        if self.result and self.returns:
            raise ValueError(f"function {self.name}: returns or a result, not both")

    @property
    def returned(self) -> tuple[str, ...]:
        """What its ``return`` gives back, as Python: the result, or the parameters it assigns."""
        return (self.result,) if self.result else self.returns


@dataclass(frozen=True)
class Invoke(Op):
    """A call of a ``Def`` (EViews's ``call f(...)``): the arguments as Python, and ``targets``: for each parameter the
    function assigns, the caller's variable that takes its value back (None where the argument is not a variable);
    for a function with a ``result``, the one variable that takes it (or none)."""

    name: str
    args: tuple[str, ...] = ()
    targets: tuple[Any, ...] = ()


@dataclass(frozen=True)
class Jump(Op):
    """``exitloop`` (break), ``return`` and ``stop`` (the program ends)."""

    kind: str
