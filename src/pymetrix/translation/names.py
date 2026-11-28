"""Python names for a program's objects: a name the script cannot use as it is (a Python or pymetrix name) gets
``_v``, and a name already taken by something else a number (``x_2``), each with a note for the report."""

from __future__ import annotations

from collections.abc import Collection, MutableMapping

__all__ = ["allocate_name", "reserved_names"]


def reserved_names(*extra: str) -> set[str]:
    """The names a script's variables must not take: pymetrix's public names, Python's builtins, and ``extra``."""
    import builtins

    import pymetrix

    return set(getattr(pymetrix, "__all__", ())) | {n for n in dir(builtins) if not n.startswith("_")} | set(extra)


def allocate_name(source: str, base: str, what: str, taken: MutableMapping[str, str],
                  reserved: Collection[str] = ()) -> tuple[str, str]:
    """A Python variable for the program's object ``source``, from ``base`` (already a Python identifier, as the front
    end makes it): ``base_v`` when ``base`` is ``reserved``, then ``_2``, ``_3`` ... while the name is ``taken`` by
    something other than ``what`` (a description, ``"series x"``). Records it in ``taken``; returns the name and a
    note for the report when it is not ``base`` ('' otherwise)."""
    note = ""
    if base in reserved:
        note = f"{source} is called {base}_v ({base} is a Python or pymetrix name)"
        base = f"{base}_v"
    name, i = base, 2
    while name in taken and taken[name] != what:
        name, i = f"{base}_{i}", i + 1
    if name != base:
        note = f"{source} is called {name} ({base} is taken)"
    taken[name] = what
    return name, note
