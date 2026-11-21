"""API calls as commands: the public function *is* the command, so scripts and dialogs record the same step.

Public functions and methods carry ``@api``::

    class Equation:
        @api("pymetrix:equation.fit")
        def fit(self, data, span=None, method=None, ...): ...

Outside a session nothing happens: ``api`` only checks whether a session is recording. Inside one (typed code
in the Immediate or Code window, ``Session.run_code``, or a dialog), each outermost call of an ``@api`` function
is recorded as a ``Call``: the function's stable ID and schema version, its receiver and every argument (defaults
applied, so the record shows the effective options), encoded by ``pymetrix.codec``. Datasets and results are
recorded by name (``{"$ref": "data"}``); settings and specifications by value.

A ``Call`` is a ``Command``: the session runs it, records it, saves it in projects and replays it, and its
``to_code()`` is the one canonical Python for that step::

    fit = Equation('c ~ y + c[-1]').fit(data, method=OLS(cov='hac'))

Dialogs build their ``Call`` (``to_call``) and run it, so the code a dialog writes and the code a user types
give the same record. ``API_CALLS`` lists every call by ID; ``schema.catalogue()`` includes them. When an
argument changes in a way old projects would not survive, raise the call's ``version`` and register a
``call_migration``, as for commands.

``Call`` (with ``invoke`` and ``to_code``), ``make_call``, ``call_of`` and ``call_record`` are a public interface for
programs that write and read projects, with ``pymetrix.codec``'s JSON and ``Session.attachments`` (docs/guide.md,
"Reading and writing projects from programs"): public, provisional before 1.0.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from ._api import _MIGRATIONS, API_CALLS, CallSpec, Recorder, _quiet, api, call_migration, recording, renamed_arguments
from .codec import _CODECS, CODE_OF, TYPE_ALIASES, Ref, SerialisationError, decode, encode
from .commands.registry import Command, MigrationError, Widget, command, migration
from .errors import SpecificationError
from .results import deprecated_alias

if TYPE_CHECKING:
    from .commands.session import Session

__all__ = ["API_CALLS", "Call", "CallSpec", "api", "as_calls", "call_migration", "call_of", "call_record", "make_call", "method_call", "python_code",
           "recording"]

VALUE = "pymetrix:value"
MATRIX_EXPRESSION = "pymetrix:matrix.expression"  # pymetrix.matrix_expr: a matrix step, written as its expression


@api(VALUE)
def _value(value: Any) -> Any:
    """A setting kept under a name (``ce = Equation('c ~ y')``): the step that typed code defining a setting, and the
    Define equation and New sample dialogs, record. Its code is the setting itself."""
    return value


# -- the command ------------------------------------------------------------------------------------------------------


@command(menu=None, id="pymetrix:call")
@dataclass(frozen=True)
class Call(Command):
    """A call of a public pymetrix function, recorded from typed code or built by a dialog."""

    label: ClassVar[str] = "Python"
    schema_version: ClassVar[int] = 2  # 2: target became receiver

    function: Annotated[str, Widget("text", help="the call's stable ID, e.g. pymetrix:equation.fit")]
    version: Annotated[int, Widget("number", min=1, help="the call's schema version when it was recorded")] = 1
    receiver: Annotated[Any, Widget("hidden")] = None  # the object a method is called on, encoded (None for functions)
    arguments: Annotated[Mapping[str, Any], Widget("hidden")] = field(default_factory=dict)  # every argument, encoded
    output: Annotated[str, Widget("text", help="the name the result is kept under; empty to keep none")] = ""
    source: Annotated[str, Widget("hidden")] = field(default="script", compare=False)  # immediate, editor, script or a dialog's ID

    def with_output(self, name: str) -> Call:
        return dataclasses.replace(self, output=name)

    @property
    def spec(self) -> CallSpec:
        spec = API_CALLS.get(self.function)
        if spec is None:
            raise MigrationError(f"no call {self.function!r} is installed")
        return spec

    def _current(self) -> dict[str, Any]:
        """The arguments brought up to the installed function's version."""
        target = self.spec.version
        if self.version > target:
            raise MigrationError(f"{self.function} was saved by a newer {self.spec.plugin} (call version {self.version}; installed: "
                                 f"{target}); upgrade {self.spec.plugin}")
        out = dict(self.arguments)
        for v in range(self.version, target):
            step = _MIGRATIONS.get((self.function, v))
            if step is None:
                raise MigrationError(f"{self.function}: no migration from version {v} to {v + 1}")
            out = step(dict(out))
        known = set(self.spec.signature.parameters)
        unknown = sorted(set(out) - known)
        if unknown:
            raise MigrationError(f"{self.function} has no argument {', '.join(unknown)}")
        return out

    def refs(self) -> set[str]:
        """The named objects the call uses."""
        found: set[str] = set()

        def walk(v: Any) -> None:
            if isinstance(v, Mapping):
                if "$ref" in v and len(v) == 1:
                    found.add(v["$ref"])
                for x in v.values():
                    walk(x)
            elif isinstance(v, list):
                for x in v:
                    walk(x)

        walk(self.receiver)
        walk(dict(self.arguments))
        return found

    def reads(self, session: Session, versions: dict[str, int]) -> set[str]:
        return self.refs() & set(versions)

    def validate(self, session: Session) -> list[str]:
        if self.function not in API_CALLS:
            return [f"no call {self.function!r} is installed"]
        missing = sorted(n for n in self.refs() if n not in session.objects)
        return [f"no object named {n!r}" for n in missing]

    def invoke(self, objects: Mapping[str, Any]) -> Any:
        """Run the call against the named ``objects`` and return its result (nothing is stored). Calls nested in it
        (a chain such as ``data.over(...).fit(...)``) run first."""
        spec = self.spec
        local = dict(objects)

        def run_nested(inner: Mapping[str, Any]) -> str:
            token = f"__call_{len(local)}"
            local[token] = _inner_call(inner).invoke(local)
            return token

        receiver, args_in = _nested(self.receiver, run_nested), _nested(self._current(), run_nested)
        objects = local
        arguments = {k: decode(v, objects) for k, v in args_in.items()}
        args: list[Any] = []
        if spec.kind == "method":
            args.append(decode(receiver, objects))
        elif spec.kind == "classmethod":
            args.append(spec.owner)
        kwargs: dict[str, Any] = {}
        # with values for *args, the arguments before it go by position too
        star = any(p.kind is p.VAR_POSITIONAL and arguments.get(n) for n, p in spec.signature.parameters.items())
        for name, p in spec.signature.parameters.items():
            if name not in arguments:
                continue
            value = arguments[name]
            if p.kind is p.POSITIONAL_ONLY or (star and p.kind is p.POSITIONAL_OR_KEYWORD):
                args.append(value)
            elif p.kind is p.VAR_POSITIONAL:
                args.extend(value)
            elif p.kind is p.VAR_KEYWORD:
                kwargs.update(value)
            else:
                kwargs[name] = value
        with _quiet():
            return spec.function(*args, **kwargs)

    def run(self, session: Session) -> Any:
        result = self.invoke(session.objects)
        if self.output:
            session.objects[self.output] = result
        return result

    def to_code(self) -> str:
        def code_of(inner: Mapping[str, Any]) -> str:  # a nested call is written in place: data.over(...).fit(...)
            return _inner_call(inner).to_code()

        spec = self.spec
        arguments = _nested(self._current(), code_of)
        receiver = _nested(self.receiver, code_of)
        if self.function == VALUE:  # a setting kept under a name: ce = Equation('c ~ y')
            code = _saved_code(arguments["value"], top=True)
            return f"{self.output} = {code}" if self.output else code
        if self.function == MATRIX_EXPRESSION:  # a matrix step: its code is the expression it ran (b = X.T @ y)
            code = str(decode(arguments["expression"])).strip()
            return f"{self.output} = {code}" if self.output else code
        always =set(spec.show(arguments) or ()) if spec.show else set()
        parts: list[str] = []
        positional = True
        params = list(spec.signature.parameters.values())[0 if spec.kind == "function" else 1:]
        # the first argument is given by position when it is required, or when nothing is (with_sample(...))
        loose = not any(p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params)
        for name, p in spec.signature.parameters.items():
            if name not in arguments:
                continue
            value = arguments[name]
            if p.kind is p.VAR_POSITIONAL:
                parts.extend(python_code(v, top=True) for v in decode(value))
                continue
            if p.kind is p.VAR_KEYWORD:
                parts.extend(f"{k}={_saved_code(v, top=True)}" for k, v in value.items())
                continue
            if p.default is not p.empty and _same(value, p.default) and name not in always:
                positional = False  # later arguments are named, so leaving this one out is safe
                continue
            # with nothing required, a first number is named (solve(order=2)), anything else is not (example('sample_macro'))
            loose = loose and not isinstance(value, (bool, int, float))
            if positional and (p.default is p.empty or loose) and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                parts.append(_saved_code(value, top=True))
                loose = False
            else:
                positional = False
                parts.append(f"{name}={_saved_code(value, top=True)}")
        if spec.kind == "method":
            callee = f"{_receiver_code(receiver)}.{spec.name}"
        elif spec.kind == "classmethod":
            callee = f"{spec.owner.__name__}.{spec.name}"
        else:
            callee = _function_name(spec)
        code = f"{callee}({', '.join(parts)})"
        return f"{self.output} = {code}" if self.output else code

    def same_as(self, other: Call) -> bool:
        """The same call (function, version, receiver, arguments and output), wherever it was recorded."""
        return (self.function, self.version, self.receiver, dict(self.arguments), self.output) == \
            (other.function, other.version, other.receiver, dict(other.arguments), other.output)

    def __setstate__(self, state: dict[str, Any]) -> None:  # pickled before the rename
        if "target" in state:
            state["receiver"] = state.pop("target")
        self.__dict__.update(state)

    target = deprecated_alias("receiver", "target", why="the object a method is called on; target is its old name")


Call.__init__ = renamed_arguments(target="receiver")(Call.__init__)  # type: ignore[method-assign]


@migration("pymetrix:call", 1)
def _call_receiver(fields: dict[str, Any]) -> dict[str, Any]:  # schema 2: target became receiver
    if "target" in fields:
        fields["receiver"] = fields.pop("target")
    return fields


# -- the call that made a result -------------------------------------------------------------------------------------

_MADE: dict[int, tuple[Any, Call]] = {}  # id(result) -> (a weak reference to it, the call a session step made it with)


def _remember_call(result: Any, call: Call) -> None:
    """Note that a session step made ``result`` with ``call`` (``call_of`` gives it back while the result lives)."""
    import weakref

    key = id(result)
    try:
        ref = weakref.ref(result, lambda _ref, key=key: _MADE.pop(key, None) if _MADE.get(key, (None,))[0] is _ref else None)
    except TypeError:  # numbers, text and other values that cannot be referred to weakly: not results
        return
    _MADE[key] = (ref, call)


def call_of(result: Any) -> Call | None:
    """The call a session step made ``result`` with (a dialog's, or typed code recorded as calls), or None: made
    outside a session, by typed code kept as code, or not by a step. ``save_result`` saves it in the provenance."""
    found = _MADE.get(id(result))
    return found[1] if found is not None and found[0]() is result else None


def call_record(call: Call) -> dict[str, Any]:
    """A call as plain JSON (``function``, ``version``, ``receiver``, ``arguments``, ``output``): ``Call(**record)``
    rebuilds it, and ``Call(**record).invoke(objects)`` runs it again on the named objects."""
    return {"function": call.function, "version": call.version, "receiver": call.receiver, "arguments": dict(call.arguments),
            "output": call.output}


def _inner_call(inner: Mapping[str, Any]) -> Call:
    """A call nested in another (``{"$call": ...}``), including one saved before ``target`` became ``receiver``."""
    receiver = inner["receiver"] if "receiver" in inner else inner.get("target")
    return Call(inner["function"], inner.get("version", 1), receiver, inner.get("arguments", {}))


def _nested(value: Any, replace: Any) -> Any:
    """``value`` with each ``{"$call": ...}`` replaced by ``{"$ref": replace(call)}``."""
    if isinstance(value, Mapping):
        if "$call" in value and len(value) == 1:
            return {"$ref": replace(value["$call"])}
        return {k: _nested(v, replace) for k, v in value.items()}
    if isinstance(value, list):
        return [_nested(v, replace) for v in value]
    return value


def make_call(call_id: str, receiver: Any = None, /, *, output: str = "", recorded_by: str = "", **arguments: Any) -> Call:
    """The ``Call`` of ``call_id`` on ``receiver`` (for methods) with ``arguments``, defaults applied, as recording the
    call would give: how dialogs build the call they run. Name workspace objects with ``Ref("data")``; a ``Call`` as
    the receiver, an argument or a value of a mapping argument is nested in this one (``data.series(...).fit(...)``,
    ``nowcast.combine({"bridge": bridge.nowcast(...)})``). A ``*args`` parameter is given
    by name, as a tuple."""
    from ._api import _encode_argument, _nested_call

    spec = API_CALLS[call_id]
    params = list(spec.signature.parameters.values())
    leading = [receiver] if spec.kind == "method" else ([spec.owner] if spec.kind == "classmethod" else [])
    positional: list[Any] = []
    for p in params:
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.name in arguments and any(
                q.kind is q.VAR_POSITIONAL and q.name in arguments for q in params):
            positional.append(arguments.pop(p.name))  # before *args, so these go by position too
        elif p.kind is p.VAR_POSITIONAL:
            positional.extend(arguments.pop(p.name, ()))
    bound = spec.signature.bind(*leading, *positional, **arguments)
    bound.apply_defaults()
    values = dict(bound.arguments)
    if spec.kind != "function":
        values.pop(params[0].name)

    def encoded(value: Any) -> Any:
        if isinstance(value, Mapping) and any(isinstance(v, Call) for v in value.values()):  # {"bridge": bridge.nowcast(...)}
            return {k: encoded(v) for k, v in value.items()}
        return _nested_call(value) if isinstance(value, Call) else _encode_argument(value, {})

    encoded_receiver = encoded(receiver) if spec.kind == "method" else None
    return Call(call_id, spec.version, encoded_receiver, {k: encoded(v) for k, v in values.items()}, output=output,
                source=recorded_by)


def method_call(cls: type, method: str, receiver: Any, /, *, output: str = "", recorded_by: str = "", **arguments: Any) -> Call:
    """The call of ``cls``'s recorded ``method`` on ``receiver`` (a ``Ref``, or a ``Call`` to chain on): how dialogs that act
    on whatever object they are given (Forecast: ``fit.forecast(...)``) build theirs, ``cls`` being that object's class."""
    spec = getattr(getattr(cls, method, None), "__api__", None)
    if spec is None:
        raise SpecificationError(f"{cls.__name__}.{method} is not a recorded call")
    return make_call(spec.id, receiver, output=output, recorded_by=recorded_by, **arguments)


def _same(value: Any, default: Any) -> bool:
    try:
        return value == encode(default)
    except SerialisationError:
        return False


def _class_name(cls: type) -> str:
    """How scripts (``from pymetrix import *``) name a settings class: ``OLS``, or ``priors.Gamma`` for one that
    pymetrix exports only through a module."""
    import types

    import pymetrix

    if getattr(pymetrix, cls.__name__, None) is cls:
        return cls.__name__
    for name in pymetrix.__all__:  # what `from pymetrix import *` gives a script
        module = getattr(pymetrix, name, None)
        if isinstance(module, types.ModuleType) and getattr(module, cls.__name__, None) is cls:
            return f"{name}.{cls.__name__}"
    return cls.__name__


def _function_name(spec: CallSpec) -> str:
    """How scripts (``from pymetrix import *``) name a function: ``sweep``, or ``templates.rbc`` for one that pymetrix
    exports only through a module."""
    import types

    import pymetrix

    if getattr(pymetrix, spec.name, None) is not None and getattr(getattr(pymetrix, spec.name), "__api__", None) is spec:
        return spec.name
    for name in pymetrix.__all__:  # what `from pymetrix import *` gives a script
        module = getattr(pymetrix, name, None)
        if isinstance(module, types.ModuleType) and getattr(getattr(module, spec.name, None), "__api__", None) is spec:
            return f"{name}.{spec.name}"
    return spec.name


def _receiver_code(value: Any) -> str:
    code = _saved_code(value)
    return code if "$ref" in value or code.endswith(")") and not code.startswith("(") else f"({code})"


def _holds_ref(value: Any) -> bool:
    if isinstance(value, Mapping):
        return ("$ref" in value and len(value) == 1) or any(_holds_ref(v) for v in value.values())
    return isinstance(value, list) and any(_holds_ref(v) for v in value)


def _saved_code(value: Any, top: bool = False) -> str:
    """``python_code`` of a saved value. A setting holding references (``System([ce, ie])``) cannot be built
    without the objects they name, so it is written from its saved fields."""
    if not (isinstance(value, Mapping) and "$type" in value and _holds_ref(value)):
        return python_code(decode(value), top=top)
    from .codec import _find

    cls = _find(TYPE_ALIASES.get(value["$type"], value["$type"]))
    shown: list[str] = []
    first = bool(getattr(cls, "code_positional", 0))
    for f in dataclasses.fields(cls):
        if not f.init or f.name not in value:
            continue
        saved = value[f.name]
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            shown.append(_saved_code(saved) if first else f"{f.name}={_saved_code(saved)}")
            first = False
            continue
        default = f.default if f.default is not dataclasses.MISSING else f.default_factory()  # type: ignore[misc]
        if _same(saved, default):
            first = False
            continue
        shown.append(_saved_code(saved) if first else f"{f.name}={_saved_code(saved)}")
        first = False
    return f"{_class_name(cls)}({', '.join(shown)})"


def python_code(obj: Any, top: bool = False) -> str:
    """Python that rebuilds ``obj``: settings dataclasses show their fields by name, those with defaults only where
    they differ (specifications marked ``code_positional`` give their first field by position); named objects are
    their names. Tuples are written as tuples
    when they are arguments of a call (``top``), and as lists inside settings, which turn lists into tuples."""
    if isinstance(obj, Ref):
        return obj.name
    if isinstance(obj, Enum):
        return f"{type(obj).__name__}.{obj.name}"
    if type(obj) in _CODECS:
        code = CODE_OF.get(type(obj))
        return code(obj) if code is not None else repr(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        shown: list[str] = []
        first = bool(getattr(type(obj), "code_positional", 0))  # specifications: Equation('c ~ y'), VAR(['c', 'y'], ...)
        for f in dataclasses.fields(obj):
            if not f.init:
                continue
            value = getattr(obj, f.name)
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                shown.append(python_code(value) if first else f"{f.name}={python_code(value)}")
                first = False
                continue
            default = f.default if f.default is not dataclasses.MISSING else f.default_factory()  # type: ignore[misc]
            try:
                if encode(value) == encode(default):
                    first = False
                    continue
            except SerialisationError:
                pass
            shown.append(python_code(value) if first else f"{f.name}={python_code(value)}")
            first = False
        return f"{_class_name(type(obj))}({', '.join(shown)})"
    if isinstance(obj, tuple) and top:
        return f"({python_code(obj[0])},)" if len(obj) == 1 else f"({', '.join(python_code(v) for v in obj)})"
    if isinstance(obj, (list, tuple)):
        return f"[{', '.join(python_code(v) for v in obj)}]"
    if isinstance(obj, (set, frozenset)):
        return f"{{{', '.join(python_code(v) for v in sorted(obj, key=repr))}}}" if obj else "set()"
    if isinstance(obj, Mapping):
        return f"{{{', '.join(f'{python_code(k)}: {python_code(v, top)}' for k, v in obj.items())}}}"  # an argument's dict holds arguments
    if isinstance(obj, float) and not math.isfinite(obj):
        return f"float({str(obj)!r})"
    return repr(obj)


# -- typed code as calls ----------------------------------------------------------------------------------------------


def as_calls(code: str, recorder: Recorder, namespace: Mapping[str, Any], result: Any) -> list[Call] | None:
    """The calls that typed ``code`` consists of, or None when it is more than calls.

    Code counts as calls when every statement is ``name = <call>`` (or, last, a bare ``<call>``), each statement
    made exactly one recorded call, and each name holds that call's result. Anything else (imports, loops, helper
    variables, arguments that are neither settings nor named objects) stays typed code, run and recorded as typed.
    """
    import ast

    if recorder.failed:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    body = tree.body
    if not body:
        return None
    defined = [_setting_defined(stmt, namespace) for stmt in body]
    if len(body) - sum(d is not None for d in defined) != len(recorder.calls):
        return None  # a setting built from a call's result (or a statement that is not a call) is more than calls
    calls = iter(recorder.calls)
    names = {id(v): k for k, v in namespace.items()}
    out = []
    for i, (stmt, setting) in enumerate(zip(body, defined)):
        if setting is not None:
            name, value = setting
            try:
                encoded = encode(value, {k: v for k, v in names.items() if v != name})
            except SerialisationError:
                return None
            out.append(Call(VALUE, 1, None, {"value": encoded}, output=name, source=recorder.source))
            continue
        call, value = next(calls)
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name) \
                and isinstance(stmt.value, ast.Call):
            name = stmt.targets[0].id
            if namespace.get(name) is not value:
                return None
            out.append(call.with_output(name))
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and i == len(body) - 1 and value is result:
            out.append(call)
        else:
            return None
    return out


def _setting_defined(stmt: Any, namespace: Mapping[str, Any]) -> tuple[str, Any] | None:
    """``(name, value)`` when ``stmt`` is ``name = Setting(...)``: a settings class (a dataclass the codec saves, not data
    or a result) called directly, its instance now held under ``name``."""
    import ast

    if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Call)):
        return None
    func: Any = stmt.value.func
    path: list[str] = []
    while isinstance(func, ast.Attribute):
        path.insert(0, func.attr)
        func = func.value
    if not isinstance(func, ast.Name) or func.id not in namespace:
        return None
    cls: Any = namespace[func.id]
    for part in path:
        cls = getattr(cls, part, None)
    name = stmt.targets[0].id
    value = namespace.get(name)
    if not (isinstance(cls, type) and isinstance(value, cls) and (dataclasses.is_dataclass(cls) or cls in _CODECS)):
        return None
    from .codec import _is_data_or_result

    return None if _is_data_or_result(value) else (name, value)
