"""The ``@api`` decorator and call recording, without the command layer (so core modules can import it).

See ``pymetrix.calls`` for what a recorded call is.
"""

from __future__ import annotations

import functools
import inspect
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .errors import SpecificationError

if TYPE_CHECKING:
    from .calls import Call

__all__ = ["API_CALLS", "CallSpec", "Recorder", "api", "call_migration", "recording", "renamed_arguments"]

_CALL_ID = re.compile(r"^[a-z][a-z0-9_]*:[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")


@dataclass(frozen=True)
class CallSpec:
    """What ``@api`` knows about a public function: its ID, schema version, plugin and where it lives."""

    id: str
    version: int
    function: Callable[..., Any]  # the undecorated function
    plugin: str
    plugin_version: str
    show: Callable[[Any], Any] | None = None  # arguments always written in generated code, even at their default

    @property
    def kind(self) -> str:
        """``"method"`` (first parameter ``self``), ``"classmethod"`` (``cls``) or ``"function"``."""
        first = next(iter(self.signature.parameters), "")
        return {"self": "method", "cls": "classmethod"}.get(first, "function")

    @functools.cached_property
    def signature(self) -> inspect.Signature:
        return inspect.signature(self.function)

    @property
    def owner(self) -> Any:
        """The class a method or classmethod belongs to (None for functions)."""
        if self.kind == "function":
            return None
        import importlib

        obj: Any = importlib.import_module(self.function.__module__)
        for part in self.function.__qualname__.split(".")[:-1]:
            obj = getattr(obj, part)
        return obj

    @property
    def name(self) -> str:
        return self.function.__name__

    def schema(self) -> dict[str, Any]:
        from .schema import json_schema

        return json_schema(self.function)


API_CALLS: dict[str, CallSpec] = {}
_MIGRATIONS: dict[tuple[str, int], Callable[[dict[str, Any]], dict[str, Any]]] = {}


def _snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


def api(id: str | None = None, version: int = 1, show: Callable[[Any], Any] | None = None,
        as_dataset: bool = True) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Make a public function or method a recorded call (see the module docstring). ``id`` defaults to
    ``plugin:class.method`` in lower case (``pymetrix:equation.fit``). ``show(arguments)`` names arguments that
    generated code writes even at their default (``vintage='latest'``). ``as_dataset=False``: a ``data`` argument
    given as plain pandas is passed on as it is, for functions that take arrays rather than a ``Dataset``."""

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        from .plugins import current_plugin

        plugin = current_plugin()
        cid = id or f"{plugin.prefix.replace('-', '_')}:{'.'.join(_snake(p) for p in fn.__qualname__.split('.'))}"
        if not _CALL_ID.match(cid):
            raise SpecificationError(f"call ID {cid!r} must look like 'plugin:name' or 'plugin:class.method' (lower case)")
        known = API_CALLS.get(cid)
        if known is not None and (known.function.__module__, known.function.__qualname__) != (fn.__module__, fn.__qualname__):
            raise SpecificationError(f"call ID {cid!r} is already used by {known.function.__module__}.{known.function.__qualname__}")
        spec = CallSpec(cid, version, fn, plugin.name, plugin.version, show)
        API_CALLS[cid] = spec
        names = list(inspect.signature(fn).parameters)
        data_at = names.index("data") if "data" in names and as_dataset else None  # plain DataFrames are accepted wherever data is

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if data_at is not None:
                args, kwargs = _coerce_data(args, kwargs, data_at)
            recorder = _STATE.recorder
            if recorder is None or _STATE.depth:
                return fn(*args, **kwargs)
            _STATE.depth += 1
            try:
                call = recorder.make(spec, args, kwargs)
                result = fn(*args, **kwargs)
            finally:
                _STATE.depth -= 1
            recorder.done(call, result)
            return result

        wrapper.__api__ = spec  # type: ignore[attr-defined]
        return wrapper

    return decorate


def _coerce_data(args: tuple[Any, ...], kwargs: dict[str, Any], at: int) -> tuple[tuple[Any, ...], dict[str, Any]]:
    import pandas as pd

    if len(args) > at and isinstance(args[at], (pd.DataFrame, pd.Series)):
        from .data.coerce import as_dataset

        args = (*args[:at], as_dataset(args[at]), *args[at + 1:])
    elif isinstance(kwargs.get("data"), (pd.DataFrame, pd.Series)):
        from .data.coerce import as_dataset

        kwargs = {**kwargs, "data": as_dataset(kwargs["data"])}
    return args, kwargs


def call_migration(call_id: str, from_version: int):
    """Register a function turning a call's arguments saved at ``from_version`` into those of the next version."""

    def decorate(fn: Callable[[dict[str, Any]], dict[str, Any]]) -> Callable[[dict[str, Any]], dict[str, Any]]:
        _MIGRATIONS[(call_id, from_version)] = fn
        return fn

    return decorate


class _BothNames(SpecificationError, TypeError):
    """An argument given by both its old and its new name: a SpecificationError, and still the TypeError it was."""


def renamed_arguments(**renames: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Accept keyword arguments by their old names (old=new), with a DeprecationWarning; both names at once raise a
    SpecificationError (also a TypeError). Put it above ``@api``, so
    the call is recorded under the new names; saved calls with the old names need a ``call_migration``. For a
    settings class, wrap its ``__init__`` and add the old field names to ``codec.FIELD_ALIASES``."""

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        name = fn.__qualname__.removesuffix(".__init__")  # Scenario(), FittedVAR.proxy_svar()

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for old, new in renames.items():
                if old in kwargs:
                    import warnings

                    if new in kwargs:
                        raise _BothNames(f"{name}() got both {old!r} and {new!r}; {old!r} is now called {new!r}")
                    warnings.warn(f"argument {old!r} of {name}() is now called {new!r}", DeprecationWarning, stacklevel=2)
                    kwargs[new] = kwargs.pop(old)
            return fn(*args, **kwargs)

        return wrapper

    return decorate


# -- recording --------------------------------------------------------------------------------------------------------


class _State(threading.local):
    def __init__(self) -> None:
        self.recorder: Recorder | None = None
        self.depth = 0  # inside a recorded call: calls it makes are part of it, not steps of their own


_STATE = _State()


class Recorder:
    """Collects the outermost ``@api`` calls made while a session records. ``failed`` says a call could not be
    recorded (an argument that is neither a setting nor a named object), so the code is kept as typed code."""

    def __init__(self, objects: Mapping[str, Any] | None = None, source: str = "script") -> None:
        self.objects = objects if objects is not None else {}
        self.source = source
        self.calls: list[tuple[Call, Any]] = []
        self.failed = False
        self._made: dict[int, Call] = {}  # results of recorded calls not yet used, by identity: a chain uses them

    def _names(self) -> dict[int, str]:
        return {id(v): k for k, v in self.objects.items()}

    def make(self, spec: CallSpec, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Call | None:
        try:
            bound = spec.signature.bind(*args, **kwargs)
        except TypeError:
            return None  # the call itself will raise
        bound.apply_defaults()
        values = dict(bound.arguments)
        names = self._names()

        def nested(value: Any) -> Any:  # a named object is used by name, even if a call in this cell made it
            return None if id(value) in names else self._chained(value)
        try:
            target = None
            if spec.kind != "function":
                first = values.pop(next(iter(spec.signature.parameters)))  # self, or cls (which the call's ID names)
                target = nested(first) or _encode_argument(first, names) if spec.kind == "method" else None
            arguments = {k: nested(v) or _encode_argument(v, names) for k, v in values.items()}
        except Exception:  # noqa: BLE001 - SerialisationError: an argument that cannot be recorded
            self.failed = True
            return None
        from .calls import Call

        return Call(spec.id, spec.version, target, arguments, source=self.source)

    def done(self, call: Call | None, result: Any) -> None:
        if call is not None:
            self.calls.append((call, result))
            self._made[id(result)] = call

    def _chained(self, value: Any) -> Any:
        """A value that an earlier call just returned (``data.over(...).fit(...)``): that call, nested in this one."""
        call = self._made.pop(id(value), None) if not isinstance(value, (bool, int, float, str, type(None))) else None
        if call is None:
            return None
        self.calls = [(c, r) for c, r in self.calls if c is not call]
        return _nested_call(call)


def _nested_call(call: Call) -> dict[str, Any]:
    """``call`` as the value of an argument or receiver of another call: ``{"$call": ...}``."""
    return {"$call": {"function": call.function, "version": call.version, "receiver": call.receiver, "arguments": dict(call.arguments)}}


def _encode_argument(value: Any, names: Mapping[int, str]) -> Any:
    """Named objects (datasets, results, a named equation or sample) by name, as the code that uses them names
    them; other settings and specifications by value. Numbers, text and tuples are always values."""
    from .codec import Ref, SerialisationError, encode

    if isinstance(value, Ref):
        return {"$ref": value.name}
    name = None if value is None or isinstance(value, (bool, int, float, str, tuple)) else names.get(id(value))
    if name is not None:
        return {"$ref": name}
    try:
        return encode(value, names)
    except SerialisationError:
        raise SerialisationError(f"a {type(value).__name__} that is not a named object cannot be recorded") from None


class recording:
    """``with recording(session.objects) as rec:`` collects the ``@api`` calls made inside the block in ``rec.calls``."""

    def __init__(self, objects: Mapping[str, Any] | None = None, source: str = "script") -> None:
        self.recorder = Recorder(objects, source)
        self._previous: tuple[Recorder | None, int] | None = None

    def __enter__(self) -> Recorder:
        self._previous = (_STATE.recorder, _STATE.depth)
        _STATE.recorder, _STATE.depth = self.recorder, 0
        return self.recorder

    def __exit__(self, *exc: Any) -> None:
        assert self._previous is not None
        _STATE.recorder, _STATE.depth = self._previous


class _quiet:
    """Run pymetrix functions without recording them (a call being replayed records itself, not its parts)."""

    def __enter__(self) -> None:
        _STATE.depth += 1

    def __exit__(self, *exc: Any) -> None:
        _STATE.depth -= 1


