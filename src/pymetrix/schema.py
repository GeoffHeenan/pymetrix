"""JSON Schemas for commands, settings and API calls: what a UI, an AI tool or a project reader can rely on.

    json_schema(FitEquation)          # a JSON Schema (draft 2020-12) of the command's fields
    json_schema(OLS)                  # of a settings class, as ``codec.encode`` writes it
    catalogue()                       # every command and API call: id, schema version, plugin, help, schema

Schemas describe values as ``pymetrix.codec`` encodes them, so a value that validates against a schema
decodes to the object. They are generated from the classes' type hints, ``Widget`` metadata and defaults, so
they cannot drift from the code. ``CATALOGUE_VERSION`` changes only when the layout of the catalogue does.
"""

from __future__ import annotations

import dataclasses
import inspect
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

from .codec import SerialisationError, encode, type_name

__all__ = ["CATALOGUE_VERSION", "catalogue", "json_schema", "type_schema"]

CATALOGUE_VERSION = 1
_DRAFT = "https://json-schema.org/draft/2020-12/schema"


def json_schema(obj: type | Callable[..., Any]) -> dict[str, Any]:
    """The JSON Schema of a dataclass's fields (a command or settings class) or of a function's arguments."""
    defs: dict[str, Any] = {}
    body = _dataclass_schema(obj, defs, top=True) if isinstance(obj, type) else _signature_schema(obj, defs)
    out = {"$schema": _DRAFT, **body}
    if defs:
        out["$defs"] = defs
    return out


def type_schema(hint: Any, defs: dict[str, Any] | None = None) -> dict[str, Any]:
    """The schema of one type hint, as the codec encodes values of that type."""
    return _type(hint, {} if defs is None else defs)


def _signature_schema(fn: Callable[..., Any], defs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(fn)
    hints = _hints(fn)
    props, required = {}, []
    for name, p in sig.parameters.items():
        if name in ("self", "cls") or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        prop = _type(hints.get(name, Any), defs)
        if p.default is p.empty:
            required.append(name)
        else:
            _default(prop, p.default)
        props[name] = prop
    out: dict[str, Any] = {"type": "object", "properties": props, "additionalProperties": False}
    if required:
        out["required"] = required
    doc = inspect.getdoc(fn)
    if doc:
        out["description"] = doc.split("\n\n")[0]
    return out


def _hints(fn: Callable[..., Any]) -> dict[str, Any]:
    """The function's type hints; one at a time when some name only exists for type checkers (``Dataset``), so the
    others are still described (an unresolvable one describes any value)."""
    try:
        return get_type_hints(fn, include_extras=True)
    except Exception:  # noqa: BLE001 - see below
        pass
    namespace = {**vars(typing), **getattr(fn, "__globals__", {})}
    out = {}
    for name, annotation in getattr(fn, "__annotations__", {}).items():
        try:
            out[name] = eval(annotation, namespace) if isinstance(annotation, str) else annotation  # noqa: S307 - our own annotations
        except Exception:  # noqa: BLE001
            out[name] = Any
    return out


def _dataclass_schema(cls: type, defs: dict[str, Any], top: bool = False) -> dict[str, Any]:
    from .forms import Widget

    hints = get_type_hints(cls, include_extras=True)
    props: dict[str, Any] = {}
    required = []
    if not top:
        props["$type"] = {"const": type_name(cls)}
        required.append("$type")
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        hint = hints[f.name]
        widget = None
        if get_origin(hint) is Annotated:
            hint, *extras = get_args(hint)
            widget = next((e for e in extras if isinstance(e, Widget)), None)
        prop = _type(hint, defs)
        if widget is not None:
            if widget.help:
                prop["description"] = widget.help
            if widget.min is not None and prop.get("type") in ("integer", "number"):
                prop["minimum"] = widget.min
            if widget.max is not None and prop.get("type") in ("integer", "number"):
                prop["maximum"] = widget.max
        if f.default is not dataclasses.MISSING:
            _default(prop, f.default)
        elif f.default_factory is not dataclasses.MISSING:
            _default(prop, f.default_factory())
        else:
            required.append(f.name)
        props[f.name] = prop
    out: dict[str, Any] = {"type": "object", "properties": props, "additionalProperties": False}
    if required:
        out["required"] = required
    doc = inspect.getdoc(cls)
    if doc and not doc.startswith(f"{cls.__name__}("):  # dataclasses' generated docstring says nothing
        out["description"] = doc.split("\n\n")[0]
    return out


def _default(prop: dict[str, Any], value: Any) -> None:
    try:
        prop["default"] = encode(value)
    except SerialisationError:
        pass


def _type(hint: Any, defs: dict[str, Any]) -> dict[str, Any]:
    from .codec import _CODECS

    if hint is Any or hint is object or isinstance(hint, typing.TypeVar):
        return {}
    origin = get_origin(hint)
    if origin is Annotated:
        return _type(get_args(hint)[0], defs)
    if hint is None or hint is type(None):
        return {"type": "null"}
    if origin is Literal:
        values = list(get_args(hint))
        return {"enum": values}
    if origin in (Union, types.UnionType):
        options = [_type(a, defs) for a in get_args(hint)]
        options = [o for i, o in enumerate(options) if o not in options[:i]]
        return {} if {} in options else {"anyOf": options}
    if hint is bool:
        return {"type": "boolean"}
    if hint is int:
        return {"type": "integer"}
    if hint is float:
        return {"anyOf": [{"type": "number"}, {"type": "object", "properties": {"$float": {"enum": ["nan", "inf", "-inf"]}},
                                               "required": ["$float"]}]}
    if hint is str:
        return {"type": "string"}
    if origin is tuple:
        args = get_args(hint)
        if len(args) == 2 and args[1] is Ellipsis:
            items: dict[str, Any] = {"type": "array", "items": _type(args[0], defs)}
        elif args:
            items = {"type": "array", "prefixItems": [_type(a, defs) for a in args], "minItems": len(args), "maxItems": len(args)}
        else:
            items = {"type": "array"}
        return {"type": "object", "properties": {"$tuple": items}, "required": ["$tuple"]}
    if origin in (list, Sequence, typing.Sequence) or hint in (list, Sequence):
        args = get_args(hint)
        array = {"type": "array", "items": _type(args[0], defs)} if args else {"type": "array"}
        if origin is list or hint is list:
            return array
        return {"anyOf": [array, {"type": "object", "properties": {"$tuple": array}, "required": ["$tuple"]}]}  # list or tuple
    if origin in (dict, Mapping, typing.Mapping) or hint in (dict, Mapping):
        args = get_args(hint)
        return {"type": "object", "additionalProperties": _type(args[1], defs)} if len(args) == 2 else {"type": "object"}
    if isinstance(hint, type):
        if hint in _CODECS:
            return {"type": "object", "properties": {"$type": {"const": _CODECS[hint][0]}, "value": {}}, "required": ["$type", "value"]}
        if dataclasses.is_dataclass(hint):
            key = type_name(hint).replace(":", ".")
            if key not in defs:
                defs[key] = {}  # placeholder: recursive types refer to themselves
                defs[key] = _dataclass_schema(hint, defs)
            return {"$ref": f"#/$defs/{key}"}
    return {}  # a type the schema cannot describe: any value the codec accepts


def catalogue() -> dict[str, Any]:
    """Every installed command and API call, with its stable ID, schema version, plugin, menu, help and JSON Schema."""
    from .commands.registry import COMMAND_IDS

    entries = []
    for cid, info in sorted(COMMAND_IDS.items()):
        doc = inspect.getdoc(info.cls) or ""
        entries.append({"id": cid, "kind": "command", "version": info.cls.schema_version, "plugin": info.plugin,
                        "plugin_version": info.plugin_version, "menu": info.menu, "help": doc.split("\n\n")[0],
                        "schema": json_schema(info.cls)})
    try:
        from .calls import API_CALLS
    except ImportError:  # before the call layer is loaded
        API_CALLS = {}
    for cid, spec in sorted(API_CALLS.items()):
        doc = inspect.getdoc(spec.function) or ""
        entries.append({"id": cid, "kind": "call", "version": spec.version, "plugin": spec.plugin, "plugin_version": spec.plugin_version,
                        "menu": None, "help": doc.split("\n\n")[0], "schema": spec.schema()})
    return {"catalogue_version": CATALOGUE_VERSION, "entries": entries}
