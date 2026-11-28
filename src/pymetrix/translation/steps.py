"""Steps mode: a translation run in a session, each ``Step`` a recorded call (``make_call``) and each ``Cmd`` a recorded
command, as a dialog's step is. Scalars, strings and matrices, and the values read from results (EViews's
``eq.@coefs(2)``), are Python values, made by running the script's own code for them (``Code``).

Control flow runs here, in Python: a loop's body runs once per pass, an ``if`` block's condition is worked out when
it is reached (with the results made before it), and a function's body runs at each call, with its parameters;
the session records only the steps, so its History is flat. A function that is not ``local`` shares the program's
variables; what it assigns to a parameter in ``returns`` goes back to the caller's variable (as EViews passes
arguments by reference), and its ``result``, when it has one, to the call's target. ``stop`` ends the run (not an
error); a ``while`` loop that makes more passes than ``while_cap`` stops with an error at its line.
"""

from __future__ import annotations

import dataclasses
from collections import ChainMap
from typing import Any

from .ir import Cmd, Code, Def, For, If, Invoke, Jump, Step, Text, While, resolved

__all__ = ["WHILE_CAP", "StepFailed", "run_steps", "step_call"]

WHILE_CAP = 10_000  # passes of a while loop in steps mode before it stops with an error (Translation.while_cap)
# make_call's own keywords and the named parameters of the calls a step makes: a series with one of these names goes
# into the call's **keywords afterwards
_TAKEN = frozenset({"output", "recorded_by", "within", "source", "vintage", "name", "self", "cls"})


def step_call(step: Step, env: Any, recorded_by: str = "translation") -> Any:
    """The ``Call`` of a step, with the program's values (``env``) put into its expressions; ``recorded_by`` is the
    steps' source (``Translation.recorded_by``)."""
    from pymetrix.extension import Ref, make_call

    kwargs: dict[str, Any] = {}
    late: dict[str, str] = {}  # placeholder -> the real name, for **kwargs keys that make_call cannot take
    spread = ""
    order: list[str] = []
    for a in step.args:
        value = resolved(a.value, env)
        if a.how != "varkw":
            kwargs[a.name] = value
            continue
        spread = a.name
        for key, v in value.items():
            order.append(key)
            if key in _TAKEN or not key.isidentifier():
                placeholder = f"pmx_series_{len(late)}"
                late[placeholder] = key
                kwargs[placeholder] = v
            else:
                kwargs[key] = v
    if isinstance(step.receiver, str):
        receiver = Ref(step.receiver)
    elif isinstance(step.receiver, Text):
        receiver = Ref(step.receiver.value(env))
    else:
        receiver = resolved(step.receiver, env) if step.receiver is not None else None
    output = step.output.value(env) if isinstance(step.output, Text) else step.output
    call = make_call(step.function, receiver, output=output, recorded_by=recorded_by, **kwargs)
    if late:
        args = dict(call.arguments)
        inner = dict(args[spread])
        for placeholder, key in late.items():
            inner[key] = inner.pop(placeholder)
        args[spread] = {k: inner[k] for k in order}
        call = dataclasses.replace(call, arguments=args)
    return call


class StepFailed(Exception):
    """A step that failed in steps mode: ``line`` and ``file`` (an included program's name, or '') say where."""

    def __init__(self, message: str, line: int, file: str) -> None:
        super().__init__(message)
        self.line, self.file = line, file


class _Break(Exception):
    pass


class _Return(Exception):
    pass


class _Stop(Exception):
    pass


@dataclasses.dataclass
class _Scope:
    """The variables a statement sees: a function's own (its parameters; everything, for a local one) over the
    program's."""

    own: dict[str, Any]
    program: dict[str, Any]
    local: bool = False

    def names(self) -> ChainMap:
        return ChainMap(self.own, self.program) if not self.local else ChainMap(self.own)

    def set(self, name: str, value: Any) -> None:
        if self.local or name in self.own:
            self.own[name] = value
        else:
            self.program[name] = value


class _Runner:
    def __init__(self, translation: Any, session: Any, runner: Any, while_cap: int) -> None:
        import pymetrix

        self.t = translation
        self.session = session
        self.runner = runner or session.run
        self.while_cap = while_cap
        self.functions: dict[str, Def] = {d.name: d for _, d, _ in translation.functions}
        self.public = {n: getattr(pymetrix, n) for n in pymetrix.__all__ if hasattr(pymetrix, n)}
        self.public.update(translation.step_names())  # what the script imports besides pymetrix

    def namespace(self, scope: _Scope) -> dict[str, Any]:
        return {**self.public, **self.session.objects, **scope.names()}

    def value(self, code: str, scope: _Scope) -> Any:
        return eval(code, self.namespace(scope))  # noqa: S307 - the translation's own Python for a condition or a bound

    def body(self, ops: Any, scope: _Scope) -> None:
        for line, op, file in ops:
            try:
                self.op(op, scope)
            except (_Break, _Return, _Stop, StepFailed):
                raise
            except Exception as exc:
                where = f"line {line} of {file or self.t.program or 'the program'}" if line else (self.t.program or "the program")
                raise StepFailed(f"{where}: {exc}", line, file) from exc

    def op(self, op: Any, scope: _Scope) -> None:
        if isinstance(op, Code):
            namespace = self.namespace(scope)
            exec(op.code, namespace)  # noqa: S102 - the translation's own code for a scalar, string or matrix
            for name in op.names:
                key = name.value(ChainMap(scope.names(), self.session.objects)) if isinstance(name, Text) else name
                scope.set(key, namespace[key])
        elif isinstance(op, Step):
            self.runner(step_call(op, ChainMap(scope.names(), self.session.objects), self.t.recorded_by))
        elif isinstance(op, Cmd):
            command = op.command
            if op.fields:
                env = ChainMap(scope.names(), self.session.objects)
                command = dataclasses.replace(command, **{k: v.value(env) for k, v in op.fields.items()})
            self.runner(command)
        elif isinstance(op, For):
            for item in self.value(op.iterable, scope):
                scope.set(op.var, item)
                try:
                    self.body(op.body, scope)
                except _Break:
                    break
        elif isinstance(op, While):
            passes = 0
            while self.value(op.cond, scope):
                passes += 1
                if passes > self.while_cap:
                    raise RuntimeError(f"the while loop made {self.while_cap:,} passes and has not ended: stopped (raise "
                                       "while_cap if it needs more)")
                try:
                    self.body(op.body, scope)
                except _Break:
                    break
        elif isinstance(op, If):
            self.body(op.body if self.value(op.cond, scope) else op.orelse, scope)
        elif isinstance(op, Invoke):
            d = self.functions[op.name]
            values = [self.value(code, scope) for code in op.args]
            inner = _Scope(dict(zip(d.params, values)), scope.program, d.local)
            try:
                self.body(d.body, inner)
            except _Return:
                pass
            values = [self.value(d.result, inner)] if d.result else [inner.own[name] for name in d.returns]
            for target, value in zip(op.targets, values):
                if target is not None:
                    scope.set(target, value)
        elif isinstance(op, Jump):
            raise {"break": _Break, "return": _Return, "stop": _Stop}[op.kind]()
        else:
            raise TypeError(f"no step for {type(op).__name__}")


def run_steps(translation: Any, session: Any = None, attach: bool = True, *, args: Any = None, while_cap: int | None = None,
              runner: Any = None) -> dict[str, Any]:
    """``Translation.run``: the translation's ops run in ``session``, each step through ``runner`` (``session.run``
    when None)."""
    from pymetrix.commands import Session
    from pymetrix.errors import PyMetrixError

    session = session if session is not None else Session()
    for name, value in translation.objects.items():
        if name not in session.objects:
            session.objects[name] = value
    env: dict[str, Any] = {}
    if args is not None:
        values = args.split() if isinstance(args, str) else [str(a) for a in args]
        names = list(translation.arguments)
        for i, name in enumerate(names):
            env[name] = values[i] if i < len(values) else ""
    cap = getattr(translation, "while_cap", WHILE_CAP) if while_cap is None else int(while_cap)
    run = _Runner(translation, session, runner, cap)
    ops = [entry for entry in translation.ops if not (args is not None and isinstance(entry[1], Code)
                                                     and entry[1].names and entry[1].names[0] in translation.arguments)]
    try:
        run.body(ops, _Scope({}, env))
    except _Stop:
        pass
    except (_Break, _Return):
        pass
    except StepFailed as exc:
        cause = exc.__cause__
        try:
            placed = type(cause)(str(exc)) if cause is not None else PyMetrixError(str(exc))
        except Exception:  # noqa: BLE001 - an error class that takes other arguments
            placed = PyMetrixError(str(exc))
        try:
            placed.line, placed.file = exc.line, exc.file  # type: ignore[attr-defined]
        except AttributeError:  # an error class that takes no attributes
            placed = PyMetrixError(str(exc))
            placed.line, placed.file = exc.line, exc.file  # type: ignore[attr-defined]
        raise placed from cause
    if attach:
        stem = translation.program.rsplit(".", 1)[0] if translation.program else "program"
        files = session.attachments.setdefault(translation.attachments, {})
        files[f"{stem}.py"] = translation.script.encode("utf-8")
        files[f"{stem}.txt"] = translation.report.to_text().encode("utf-8")
    return {**env, "session": session}
