"""The translation's intermediate representation: what each EViews statement becomes, from which both the script
(``script.py``) and the steps (``steps.py``) are made, so the two cannot drift apart.

- ``Text``: a pymetrix expression or condition as text, with the program's scalars in it by name. The script writes
  it as an f-string (``f"x*({s})"``); the steps put in the value the scalar has when the step runs.
- ``Code``: Python that is not a step: a scalar, a string, a matrix. The script has it as it is; steps mode runs the
  same text.
- ``Step``: a recorded call of pymetrix's API (``make_call``), with the variable it is made on and the one it makes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["Arg", "Code", "Op", "SampleSpec", "Step", "Text", "Var", "literal", "quote", "resolved", "step_code"]


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
            args.append(f"({quote(self.start)}, {quote(self.end)})")
        elif self.start is not None:
            args.append(f"start={quote(self.start)}")
        elif self.end is not None:
            args.append(f"end={quote(self.end)}")
        if self.where is not None:
            args.append(f"where={self.where.code()}")
        return f"Sample({', '.join(args)})"

    def value(self, env: Mapping[str, Any]) -> Any:
        from pymetrix import Sample

        return Sample(start=self.start, end=self.end, where=self.where.value(env) if self.where is not None else None)

    @property
    def names(self) -> tuple[str, ...]:
        return self.where.names if self.where is not None else ()


class Op:
    """What a statement becomes."""


@dataclass(frozen=True)
class Code(Op):
    """Python that is not a step (a scalar, a string, a matrix): ``code`` sets ``names``."""

    code: str
    names: tuple[str, ...] = ()


@dataclass(frozen=True)
class Arg:
    """One argument of a ``Step``: the parameter's ``name``, its ``value``, and ``how`` it is given: ``"pos"`` by
    position in the script (by name to ``make_call``), ``"star"`` a tuple spread over ``*name``, ``"kw"`` by name,
    ``"varkw"`` a mapping spread over ``**name``."""

    name: str
    value: Any
    how: str = "kw"


@dataclass(frozen=True)
class Step(Op):
    """A call of pymetrix's API, recorded in steps mode: ``function`` (its stable ID, ``pymetrix:dataset.with_series``)
    on the variable ``receiver`` (None for a class method), making ``output``; ``callee`` is how the script calls it
    (``data.with_series``, ``Dataset.read``). A value is plain (text, a number, a tuple), a ``Text``, a ``SampleSpec``
    or a ``Var`` (a Python variable's value: a named sample)."""

    function: str
    receiver: str | None
    output: str
    callee: str
    args: tuple[Arg, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        """The Python variables the step's values use."""
        out: list[str] = []

        def walk(v: Any) -> None:
            if isinstance(v, Var):
                out.append(v.name)
            elif isinstance(v, (Text, SampleSpec)):
                out.extend(v.names)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    walk(x)
            elif isinstance(v, Mapping):
                for x in v.values():
                    walk(x)
        for a in self.args:
            walk(a.value)
        return tuple(out)


def literal(value: Any) -> str:
    """A value as Python, in the script."""
    if isinstance(value, (Text, SampleSpec)):
        return value.code()
    if isinstance(value, Var):
        return value.name
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
    if isinstance(value, Mapping):
        return {k: resolved(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [resolved(v, env) for v in value]
    if isinstance(value, tuple):
        return tuple(resolved(v, env) for v in value)
    return value


def step_code(step: Step) -> str:
    """A step as one line of the script: ``output = callee(arguments)``."""
    import keyword

    parts: list[str] = []
    for a in step.args:
        if a.how == "pos":
            parts.append(literal(a.value))
        elif a.how == "star":
            parts.extend(literal(v) for v in a.value)
        elif a.how == "varkw":
            for k, v in a.value.items():
                if k.isidentifier() and not keyword.iskeyword(k):
                    parts.append(f"{k}={literal(v)}")
                else:
                    parts.append(f"**{{{literal(k)}: {literal(v)}}}")
        else:
            parts.append(f"{a.name}={literal(a.value)}")
    return f"{step.output} = {step.callee}({', '.join(parts)})"
