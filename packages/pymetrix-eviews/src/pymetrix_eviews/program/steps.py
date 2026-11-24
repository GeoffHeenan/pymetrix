"""Steps mode: a translation run in a session, each data or estimation statement a recorded call (``make_call``) or
command, as a dialog's step is. Scalars, strings and matrices, and the values read from results (``eq.@coefs(2)``),
are Python values, made by running the script's own code for them."""

from __future__ import annotations

import dataclasses
from typing import Any

from collections import ChainMap

from .ir import Cmd, Code, Step, resolved

__all__ = ["ATTACHMENTS", "run_steps", "step_call"]

ATTACHMENTS = "pymetrix-eviews"  # session.attachments[ATTACHMENTS]["<program>.py"]: the script, saved with the project
RECORDED_BY = "eviews:program"  # the steps' source, as a dialog's ID is for its steps
# make_call's own keywords and the named parameters of the calls a step makes: a series with one of these names goes
# into the call's **keywords afterwards
_TAKEN = frozenset({"output", "recorded_by", "within", "source", "vintage", "name", "self", "cls"})


def step_call(step: Step, env: dict[str, Any]) -> Any:
    """The ``Call`` of a step, with the program's values (``env``) put into its expressions."""
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
    else:
        receiver = resolved(step.receiver, env) if step.receiver is not None else None
    call = make_call(step.function, receiver, output=step.output, recorded_by=RECORDED_BY, **kwargs)
    if late:
        args = dict(call.arguments)
        inner = dict(args[spread])
        for placeholder, key in late.items():
            inner[key] = inner.pop(placeholder)
        args[spread] = {k: inner[k] for k in order}
        call = dataclasses.replace(call, arguments=args)
    return call


def _namespace(session: Any, env: dict[str, Any]) -> dict[str, Any]:
    import pymetrix

    return {**{n: getattr(pymetrix, n) for n in pymetrix.__all__ if hasattr(pymetrix, n)}, **session.objects, **env}


def run_steps(translation: Any, session: Any = None, attach: bool = True) -> dict[str, Any]:
    from pymetrix.commands import Session
    from pymetrix.errors import PyMetrixError

    session = session if session is not None else Session()
    env: dict[str, Any] = {}
    for line, op in translation.ops:
        try:
            if isinstance(op, Code):
                namespace = _namespace(session, env)
                exec(op.code, namespace)  # noqa: S102 - the translation's own code for a scalar, string or matrix
                for name in op.names:
                    env[name] = namespace[name]
            elif isinstance(op, Step):
                session.run(step_call(op, ChainMap(env, session.objects)))
            elif isinstance(op, Cmd):
                session.run(op.command)
        except Exception as exc:
            message = f"line {line} of {translation.program or 'the program'}: {exc}"
            try:
                placed = type(exc)(message)
            except Exception:  # noqa: BLE001 - an error class that takes other arguments
                placed = PyMetrixError(message)
            raise placed from exc
    if attach:
        stem = translation.program.rsplit(".", 1)[0] if translation.program else "program"
        files = session.attachments.setdefault(ATTACHMENTS, {})
        files[f"{stem}.py"] = translation.script.encode("utf-8")
        files[f"{stem}.txt"] = translation.report.to_text().encode("utf-8")
    return {**env, "session": session}
