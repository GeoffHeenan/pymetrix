"""A typed JSON codec for specifications, settings, commands and calls: what can be saved, sent and read back.

    value = encode(OLS(cov="hac", hac_lags=4))
    # {"$type": "pymetrix:OLS", "cov": "hac", "hac_lags": 4, "start": None}
    decode(value) == OLS(cov="hac", hac_lags=4)           # True: every encoded value reads back equal
    json.dumps(value)                                      # plain JSON, for projects, UI and AI tools

What is encoded:

- ``None``, ``bool``, ``int``, ``str`` and finite ``float`` as themselves; ``nan`` and infinities as
  ``{"$float": "nan"}``;
- lists as lists, tuples as ``{"$tuple": [...]}``, sets as ``{"$set": [...]}`` (sorted when possible);
- dictionaries with string keys as objects (a key starting with ``$`` is escaped as ``{"$dict": [[k, v], ...]}``,
  as are dictionaries with other keys);
- dates, datetimes, pandas periods, enums and NumPy scalars and arrays, each tagged;
- dataclasses as ``{"$type": name, field: value, ...}`` with every init field, defaults included, so a
  decoded object never depends on defaults that may change;
- ``Ref(name)``, a reference to a named object in a session, as ``{"$ref": name}``. ``encode(obj, names)`` also
  writes the named objects inside a setting by name (the equations in ``System([ce, ie])``), and ``Fields`` describes
  a setting that holds references, which cannot be built until they are resolved.

Anything else raises ``SerialisationError``: nothing is ever written as its ``repr``. Data (``Dataset``) and
results are not settings: pass them by reference.

Type names are ``"pymetrix:OLS"`` for classes exported by ``pymetrix``, else ``"module.path:QualName"``. Decoding
imports only from ``pymetrix``, from packages that registered extensions, and from packages added with
``allow_module``, so a project file cannot make pymetrix import arbitrary code. When a class moves or is renamed,
add the old name to ``TYPE_ALIASES``; when a field is renamed, add it to ``FIELD_ALIASES``. Classes whose constructor does not take their fields (such as ``Cluster``)
register an encoder and decoder with ``register_codec``.

``encode``, ``decode``, ``Ref`` and ``SerialisationError`` are part of the public interface for programs that write and
read projects (docs/guide.md, "Reading and writing projects from programs"): public, provisional before 1.0.
"""

from __future__ import annotations

import dataclasses
import importlib
import math
import threading
from collections.abc import Callable, Mapping
from datetime import date, datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from .errors import PyMetrixError

__all__ = ["CODEC_VERSION", "Fields", "Ref", "SerialisationError", "TYPE_ALIASES", "allow_module", "decode", "encode", "register_codec",
           "type_name"]

CODEC_VERSION = 1  # raise if the tags themselves change; old values must still decode


class SerialisationError(PyMetrixError, TypeError):
    """A value cannot be encoded as JSON, or an encoded value cannot be read back."""


@dataclasses.dataclass(frozen=True)
class Ref:
    """A reference to an object by its name in a session (a dataset, a fitted equation...), not a copy of it."""

    name: str

    def __repr__(self) -> str:
        return self.name


@dataclasses.dataclass(frozen=True)
class Fields:
    """A settings object of class ``cls`` given by its fields, some of them references (``Ref``): it encodes as the
    object would, and decodes to it once the references are resolved (``System`` with ``Ref("ce")`` equations).
    Give fields as the class keeps them (tuples for its tuple fields); missing ones take their defaults."""

    cls: type
    fields: Mapping[str, Any]


TYPE_ALIASES: dict[str, str] = {}  # old type name -> current, for classes that moved or were renamed
FIELD_ALIASES: dict[str, dict[str, str]] = {  # type name -> {old field name: current}, for renamed fields
    "pymetrix:Scenario": {"target": "hold"},
    "pymetrix:BMA": {"target": "outcome"},
    "pymetrix:MaxShare": {"target": "variable"},
    "pymetrix.figures:StyleCall": {"target": "kind"},
    "pymetrix.models.model:ModelEquation": {"target": "dependent"},
    "pymetrix.models.system:SystemEquation": {"target": "dependent"},
}
_CODECS: dict[type, tuple[str, Callable[[Any], Any], Callable[[Any], Any]]] = {}  # cls -> (name, encode, decode)
_DECODERS: dict[str, Callable[[Any], Any]] = {}
CODE_OF: dict[type, Callable[[Any], str]] = {}  # cls -> the Python that rebuilds an instance, for generated code
_ALLOWED: set[str] = {"pymetrix"}


def allow_module(package: str) -> None:
    """Let ``decode`` import classes from ``package`` (its top-level name), e.g. a plugin not using entry points."""
    _ALLOWED.add(package.split(".")[0])


def register_codec(cls: type, name: str, to_value: Callable[[Any], Any], from_value: Callable[[Any], Any],
                   code: Callable[[Any], str] | None = None) -> None:
    """Encode instances of ``cls`` as ``{"$type": name, "value": to_value(obj)}`` and decode them with ``from_value``.
    For classes whose constructor does not take their fields. ``code(obj)`` is the Python generated code writes for
    one (default: its ``repr``)."""
    _CODECS[cls] = (name, to_value, from_value)
    _DECODERS[name] = from_value
    if code is not None:
        CODE_OF[cls] = code


def type_name(cls: type) -> str:
    """The stable name ``decode`` finds ``cls`` by: ``"pymetrix:OLS"`` when ``pymetrix`` exports it."""
    if cls in _CODECS:
        return _CODECS[cls][0]
    import pymetrix

    if pymetrix.__dict__.get(cls.__name__) is cls:
        return f"pymetrix:{cls.__name__}"
    return f"{cls.__module__}:{cls.__qualname__}"


# -- encoding ---------------------------------------------------------------------------------------------------------


class _Names(threading.local):
    names: Mapping[int, str] = {}


_NAMES = _Names()


def encode(obj: Any, names: Mapping[int, str] | None = None) -> Any:
    """A JSON-ready value that ``decode`` turns back into an equal object; see the module docstring. ``names`` maps
    ``id(object)`` to a name: those objects, inside settings, are written as references (``{"$ref": name}``)."""
    previous, _NAMES.names = _NAMES.names, names or {}
    try:
        return _encode(obj, ())
    finally:
        _NAMES.names = previous


def _encode(obj: Any, path: tuple[str, ...]) -> Any:
    if isinstance(obj, Enum):  # before str and int: enums such as Measure are also strings
        return {"$enum": type_name(type(obj)), "value": _encode(obj.value, path)}
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, np.bool_):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else {"$float": repr(value)}
    if isinstance(obj, np.bool_):
        return bool(obj)
    if type(obj) in _CODECS:
        name, to_value, _ = _CODECS[type(obj)]
        return {"$type": name, "value": _encode(to_value(obj), (*path, name))}
    if isinstance(obj, Ref):
        return {"$ref": obj.name}
    if isinstance(obj, pd.Period):
        return {"$period": str(obj), "freq": obj.freqstr}
    if isinstance(obj, datetime):  # before date: a datetime is a date
        return {"$datetime": obj.isoformat()}
    if isinstance(obj, date):
        return {"$date": obj.isoformat()}
    if path and not isinstance(obj, tuple) and id(obj) in _NAMES.names:
        return {"$ref": _NAMES.names[id(obj)]}  # a named object inside a setting or a list: by name
    if _is_data_or_result(obj):
        raise SerialisationError(f"{_where(path)}a {type(obj).__name__} is data or a result, not a setting: pass it by name (Ref)")
    if isinstance(obj, Fields):
        given = dict(obj.fields)
        unknown = sorted(set(given) - {f.name for f in dataclasses.fields(obj.cls) if f.init})
        if unknown:
            raise SerialisationError(f"{_where(path)}{obj.cls.__name__} has no field {', '.join(unknown)}")
        out = {"$type": type_name(obj.cls)}
        for f in dataclasses.fields(obj.cls):
            if f.init:
                value = given[f.name] if f.name in given else (f.default if f.default is not dataclasses.MISSING else f.default_factory())  # type: ignore[misc]
                out[f.name] = _encode(value, (*path, f.name))
        return out
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {"$type": type_name(type(obj))}
        for f in dataclasses.fields(obj):
            if f.init:
                out[f.name] = _encode(getattr(obj, f.name), (*path, f.name))
        return out
    if isinstance(obj, list):
        return [_encode(v, (*path, str(i))) for i, v in enumerate(obj)]
    if isinstance(obj, tuple):
        return {"$tuple": [_encode(v, (*path, str(i))) for i, v in enumerate(obj)]}
    if isinstance(obj, (set, frozenset)):
        items = [_encode(v, path) for v in obj]
        try:
            items = sorted(items)
        except TypeError:  # mixed or unorderable values: their order is not part of the value
            pass
        return {"$frozenset" if isinstance(obj, frozenset) else "$set": items}
    if isinstance(obj, Mapping):
        if all(isinstance(k, str) and not k.startswith("$") for k in obj):
            return {k: _encode(v, (*path, k)) for k, v in obj.items()}
        return {"$dict": [[_encode(k, path), _encode(v, (*path, str(k)))] for k, v in obj.items()]}
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind not in "biuf":
            raise SerialisationError(f"{_where(path)}an array of {obj.dtype} cannot be saved; use numbers")
        return {"$array": [_encode(v, path) for v in obj.ravel().tolist()], "dtype": str(obj.dtype), "shape": list(obj.shape)}
    raise SerialisationError(f"{_where(path)}a {type(obj).__module__}.{type(obj).__qualname__} cannot be saved as a setting")


def _is_data_or_result(obj: Any) -> bool:
    return isinstance(getattr(obj, "frame", None), pd.DataFrame) or "provenance" in getattr(type(obj), "__dataclass_fields__", {}) \
        or isinstance(obj, (pd.DataFrame, pd.Series))


def _where(path: tuple[str, ...]) -> str:
    return f"{'.'.join(path)}: " if path else ""


# -- decoding ---------------------------------------------------------------------------------------------------------


def decode(value: Any, refs: Mapping[str, Any] | Callable[[str], Any] | None = None) -> Any:
    """The object ``encode`` made ``value`` from. ``refs`` resolves ``{"$ref": name}`` (a mapping such as a session's
    objects, or a function); without it, references decode as ``Ref(name)``."""
    return _decode(value, refs)


def _decode(value: Any, refs: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_decode(v, refs) for v in value]
    if not isinstance(value, Mapping):
        raise SerialisationError(f"cannot read a {type(value).__name__} as a saved value")
    if "$type" in value:
        return _decode_object(value, refs)
    if "$ref" in value:
        name = value["$ref"]
        if refs is None:
            return Ref(name)
        try:
            return refs(name) if callable(refs) else refs[name]
        except KeyError:
            raise SerialisationError(f"no object named {name!r}") from None
    if "$tuple" in value:
        return tuple(_decode(v, refs) for v in value["$tuple"])
    if "$set" in value:
        return {_decode(v, refs) for v in value["$set"]}
    if "$frozenset" in value:
        return frozenset(_decode(v, refs) for v in value["$frozenset"])
    if "$dict" in value:
        return {_decode(k, refs): _decode(v, refs) for k, v in value["$dict"]}
    if "$float" in value:
        return float(value["$float"])
    if "$date" in value:
        return date.fromisoformat(value["$date"])
    if "$datetime" in value:
        return datetime.fromisoformat(value["$datetime"])
    if "$period" in value:
        return pd.Period(value["$period"], freq=value["freq"])
    if "$enum" in value:
        return _find(value["$enum"])(_decode(value["value"], refs))
    if "$array" in value:
        return np.asarray(value["$array"], dtype=value["dtype"]).reshape(value["shape"])
    unknown = [k for k in value if k.startswith("$")]
    if unknown:
        raise SerialisationError(f"unknown tag {unknown[0]!r}: saved by a newer pymetrix?")
    return {k: _decode(v, refs) for k, v in value.items()}


def _decode_object(value: Mapping[str, Any], refs: Any) -> Any:
    name = TYPE_ALIASES.get(value["$type"], value["$type"])
    if name in _DECODERS:
        return _DECODERS[name](_decode(value["value"], refs))
    cls = _find(name)
    if not dataclasses.is_dataclass(cls):
        raise SerialisationError(f"{name} is not a settings class")
    known = {f.name for f in dataclasses.fields(cls) if f.init}
    renamed = FIELD_ALIASES.get(name, {})
    kwargs = {renamed.get(k, k): _decode(v, refs) for k, v in value.items() if k != "$type"}
    upgrade = getattr(cls, "_saved_fields", None)  # a class may say what fields missing from older saves meant
    if callable(upgrade):
        kwargs = upgrade(kwargs)
    unknown = sorted(set(kwargs) - known)
    if unknown:
        raise SerialisationError(f"{name} has no field {', '.join(unknown)}: saved by a different version?")
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise SerialisationError(f"cannot rebuild {name}: {exc}") from exc


def _find(name: str) -> type:
    module, sep, qualname = name.partition(":")
    if not sep or not qualname:
        raise SerialisationError(f"{name!r} is not a type name (module:Class)")
    if module.split(".")[0] not in _allowed():
        raise SerialisationError(f"{name}: classes from {module.split('.')[0]!r} are not read from saved files "
                                 "(it is not pymetrix or an installed plugin)")
    try:
        obj: Any = importlib.import_module(module)
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError):
        raise SerialisationError(f"{name} is not installed (or has moved: see codec.TYPE_ALIASES)") from None
    if not isinstance(obj, type):
        raise SerialisationError(f"{name} is not a class")
    return obj


def _allowed() -> set[str]:
    from .plugins import GROUPS, registry

    found = set(_ALLOWED)
    for group in GROUPS:
        for entry in registry(group).entries():
            module = getattr(entry.obj, "__module__", "") or ""
            if module:
                found.add(module.split(".")[0])
    return found


def _builtin_codecs() -> None:
    from .data.sample import Sample
    from .estimation.covariance import Cluster

    register_codec(Cluster, "pymetrix:Cluster", lambda c: list(c.levels), lambda v: Cluster(*v),
                   code=lambda c: f"Cluster({', '.join(map(repr, c.levels))})")
    register_codec(Sample, "pymetrix:Sample", lambda s: s.to_dict(), Sample.from_dict)


_builtin_codecs()
