"""Backend adapters: how a pymetrix setting becomes a library argument, and what comes back.

Every wrapped estimator declares an ``AdapterSpec``: data, not code, saying

- ``inputs``: how pymetrix's arrays reach the library (``"y" -> "dependent"``);
- ``options``: each pymetrix setting and the library argument it becomes (``Option("fuller", "fuller")``), or
  how pymetrix uses it itself (``Option("endog", None, "selects the endogenous columns")``);
- ``fixed``: arguments pymetrix sets on the library's behalf, with the reason (``maxiter=5000``);
  ``backend_options`` may override them;
- ``outputs``: library attributes and the pymetrix fields they fill;
- ``unsupported``: settings this library cannot honour, refused with a clear error rather than ignored.

The estimator runs the library through ``Adapter.run(spec, settings)``::

    with LINEARMODELS.run(LIML_SPEC, self) as run:
        model = run.call("model", iv.IVLIML, y, exog, endog, inst, fuller=self.fuller)
        res = run.call("fit", model.fit, cov_type=cov_type, debiased=self.debiased)
    run.record      # BackendCall: library, version, the options each stage was given, warnings, messages

``run.call`` merges the fixed arguments, the mapped ones and the user's ``backend_options`` for that stage
(keys ``"maxiter"`` go to the spec's ``default_stage``; ``"model.xxx"`` or ``"fit.xxx"`` choose the stage). A
``backend_options`` key that a pymetrix setting already controls is an error naming that setting. Warnings
raised by the library are captured (and still shown), so they reach the result's provenance.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..errors import SpecificationError

__all__ = ["BackendOptions", "Adapter", "AdapterSpec", "BackendCall", "Option", "Options", "Run", "check_backend_options", "options_of", "with_backend"]


_Value = str | int | float | bool | None
BackendOptions = Mapping[str, _Value | Sequence[_Value]]  # the type of every ``backend_options`` field: JSON-like values


class Options(Mapping[str, Any]):
    """``backend_options``: a read-only, hashable mapping of extra library arguments.

    Built from a mapping, or from text as typed in a form (``"maxiter=500, tol=1e-8"``)."""

    __slots__ = ("_items",)

    def __init__(self, values: Mapping[str, Any] | str | None = None) -> None:
        if isinstance(values, Options):
            items = dict(values._items)
        elif isinstance(values, str):
            import re

            from ..forms import parse_options

            dotted = re.sub(r"\b(model|fit)\.(?=[A-Za-z_]\w*\s*=)", r"\1__stage__", values)  # fit.maxiter=5: not Python syntax
            args, items = parse_options(dotted)
            items = {k.replace("__stage__", "."): v for k, v in items.items()}
            if len(args) == 1 and isinstance(args[0], dict) and not items:  # "{'maxiter': 500}"
                items = dict(args[0])
            elif args:
                raise SpecificationError(f"backend options are name=value pairs, e.g. maxiter=500: {values!r}")
        else:
            items = dict(values or {})
        bad = [k for k in items if not isinstance(k, str) or not k.replace(".", "_").isidentifier()]
        if bad:
            raise SpecificationError(f"backend option names must be argument names: {bad}")
        self._items = dict(sorted(items.items()))

    def __getitem__(self, key: str) -> Any:
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(repr(sorted(self._items.items())))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self._items) == dict(other)

    def __repr__(self) -> str:
        return repr(self._items)


def options_of(obj: Any) -> Options:
    """The ``backend_options`` of a settings object (empty when it has none)."""
    return Options(getattr(obj, "backend_options", None))


def check_backend_options(obj: Any) -> None:
    """For ``__post_init__``: turn ``backend_options`` into ``Options`` and check its names."""
    object.__setattr__(obj, "backend_options", Options(getattr(obj, "backend_options", None)))


@dataclass(frozen=True)
class Option:
    """A pymetrix setting and the library argument it becomes (``argument=None``: pymetrix uses it itself)."""

    field: str
    argument: str | None
    note: str = ""
    stage: str = ""  # "model" or "fit"; empty: the spec's default stage


@dataclass(frozen=True)
class AdapterSpec:
    """How one pymetrix estimator maps onto one library call (see the module docstring)."""

    wraps: str  # the library's class or function, e.g. "linearmodels.iv.IVLIML"
    backend: str  # "linearmodels"
    inputs: Mapping[str, str] = field(default_factory=dict)
    options: tuple[Option, ...] = ()
    fixed: Mapping[str, tuple[Any, str]] = field(default_factory=dict)  # "stage.argument" or "argument" -> (value, reason)
    outputs: Mapping[str, str] = field(default_factory=dict)
    unsupported: tuple[tuple[Callable[[Any], bool], str], ...] = ()  # (applies to these settings?, why not)
    default_stage: str = "fit"
    stages: tuple[str, ...] = ("model", "fit")
    notes: str = ""

    def mapped(self, stage: str) -> dict[str, str]:
        """Library argument -> pymetrix setting, for the arguments pymetrix controls at ``stage``."""
        return {o.argument: o.field for o in self.options if o.argument and (o.stage or self.default_stage) == stage}

    def fixed_for(self, stage: str) -> dict[str, Any]:
        out = {}
        for key, (value, _) in self.fixed.items():
            st, _, arg = key.rpartition(".")
            if (st or self.default_stage) == stage:
                out[arg] = value
        return out

    def check(self, settings: Any) -> None:
        """Refuse settings this library cannot honour."""
        for applies, why in self.unsupported:
            if applies(settings):
                raise SpecificationError(f"{type(settings).__name__} ({self.wraps}): {why}")

    def fields(self) -> set[str]:
        """The pymetrix settings the spec accounts for (mapped, used by pymetrix, or refused when set)."""
        return {o.field for o in self.options}

    def describe(self) -> str:
        """The mapping as Markdown, for help pages and the adapters design note."""
        rows = [f"**{self.wraps}** ({self.backend})", ""]
        if self.inputs:
            rows += ["| data | library argument |", "|---|---|", *[f"| {k} | `{v}` |" for k, v in self.inputs.items()], ""]
        rows += ["| pymetrix setting | library argument | note |", "|---|---|---|"]
        rows += [f"| `{o.field}` | {f'`{o.stage or self.default_stage}.{o.argument}`' if o.argument else '(pymetrix)'} | {o.note} |"
                 for o in self.options]
        rows.append(f"| `backend_options` | any other argument (default stage: {self.default_stage}) | merged last; checked for clashes |")
        if self.fixed:
            rows += ["", "| fixed by pymetrix | value | why (override with backend_options) |", "|---|---|---|"]
            rows += [f"| `{k}` | `{v!r}` | {why} |" for k, (v, why) in self.fixed.items()]
        if self.outputs:
            rows += ["", "| library output | pymetrix field |", "|---|---|", *[f"| `{k}` | {v} |" for k, v in self.outputs.items()]]
        if self.unsupported:
            rows += ["", "Not supported: " + "; ".join(why for _, why in self.unsupported) + "."]
        if self.notes:
            rows += ["", self.notes]
        return "\n".join(rows)


@dataclass(frozen=True)
class BackendCall:
    """What a library was given and said: its name and version, the arguments of each stage, warnings, messages."""

    backend: str
    version: str
    wraps: str
    options: Mapping[str, Mapping[str, Any]]  # stage -> argument -> value as recorded (data arguments by type and shape)
    backend_options: Mapping[str, Any]
    warnings: tuple[str, ...] = ()
    messages: tuple[str, ...] = ()  # convergence and optimiser messages the library returned

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.backend, "version": self.version, "wraps": self.wraps}

    @property
    def label(self) -> str:
        """``"linearmodels 7.1"``: what ``provenance.method["backend"]`` has always held."""
        return f"{self.backend} {self.version}".strip()


@dataclass(frozen=True)
class Adapter:
    """One library: its import name, version and how pymetrix calls it."""

    name: str  # the library ("statsmodels")
    module: str = ""  # top-level module, if different from the name

    @property
    def version(self) -> str:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(self.module or self.name)
        except PackageNotFoundError:
            return ""

    def run(self, spec: AdapterSpec, settings: Any = None) -> Run:
        if spec.backend != self.name:
            raise SpecificationError(f"{spec.wraps} is a {spec.backend} spec, not {self.name}")
        if settings is not None:
            spec.check(settings)
        return Run(self, spec, options_of(settings))


class Run:
    """One use of a library by one estimator: ``with adapter.run(spec, settings) as run: run.call(...)``."""

    def __init__(self, adapter: Adapter, spec: AdapterSpec, backend_options: Options) -> None:
        self.adapter, self.spec, self.backend_options = adapter, spec, backend_options
        self.options: dict[str, dict[str, Any]] = {}
        self.warnings: list[str] = []
        self.messages: list[str] = []
        self._stack: contextlib.ExitStack | None = None
        unknown = sorted({k.partition(".")[0] for k in backend_options if "." in k} - set(spec.stages))
        if unknown:
            raise SpecificationError(f"backend options for {spec.wraps} go to {' or '.join(spec.stages)}, not {', '.join(unknown)}")

    def __enter__(self) -> Run:
        from .._capture import capture

        self._stack = contextlib.ExitStack()
        self.warnings = self._stack.enter_context(capture())
        return self

    def __exit__(self, *exc: Any) -> None:
        assert self._stack is not None
        self._stack.close()

    def _user(self, stage: str) -> dict[str, Any]:
        out = {}
        for key, value in self.backend_options.items():
            st, _, arg = key.rpartition(".")
            if (st or self.spec.default_stage) == stage:
                out[arg] = value
        return out

    def call(self, stage: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call the library at ``stage`` with the mapped ``kwargs``, the spec's fixed arguments and the user's
        ``backend_options`` for that stage, and record what it was given."""
        user = self._user(stage)
        mapped = self.spec.mapped(stage)
        clash = sorted(set(user) & (set(kwargs) - set(self.spec.fixed_for(stage))))
        if clash:
            named = [f"{a} (set it with {mapped[a]}=...)" if a in mapped else a for a in clash]
            raise SpecificationError(f"backend options {', '.join(named)} are set by pymetrix for {self.spec.wraps}; "
                                     "use the pymetrix setting instead")
        merged = {**self.spec.fixed_for(stage), **kwargs, **user}
        self.options[stage] = {k: _recorded(v) for k, v in merged.items()}
        return fn(*args, **merged)

    def message(self, text: str) -> None:
        """Record a message the library returned (a convergence flag, an optimiser's message)."""
        if text and text not in self.messages:
            self.messages.append(text)

    @property
    def record(self) -> BackendCall:
        return BackendCall(self.adapter.name, self.adapter.version, self.spec.wraps, dict(self.options), dict(self.backend_options),
                           tuple(self.warnings), tuple(self.messages))


def _recorded(value: Any) -> Any:
    """A library argument as provenance keeps it: settings as values, data by type and shape."""
    import numpy as np
    import pandas as pd

    if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
        return f"<{type(value).__name__} {'x'.join(map(str, getattr(value, 'shape', ())))}>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _recorded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_recorded(v) for v in value]
    return f"<{type(value).__name__}>"


def with_backend(provenance: Any, record: BackendCall, plugin: str = "") -> Any:
    """``provenance`` with what the library was given and said: its name and version (also as the familiar
    ``method["backend"]`` label), the arguments of each stage, the backend options, and its warnings and messages."""
    from dataclasses import replace

    method = {**provenance.method, "backend": record.label, **({"plugin": plugin} if plugin else {})}
    warnings = (*provenance.warnings, *(w for w in record.warnings if w not in provenance.warnings))
    return replace(provenance, method=method, backend=record.to_dict(),
                   options={"stages": dict(record.options), "backend_options": dict(record.backend_options)},
                   warnings=warnings, messages=(*provenance.messages, *record.messages))


def to_axes(values: Any, index: Any, name: str | None = None, freq: str | None = None) -> Any:
    """A library's output on pymetrix's axes (``index``: the rows it was estimated on).

    A ``DatetimeIndex`` is turned into periods and aligned by label; any other library index (a ``RangeIndex``,
    or an array without one) is placed by position, after checking that the lengths agree. Raises rather than
    misaligning."""
    import numpy as np
    import pandas as pd

    index = index if isinstance(index, pd.Index) else pd.Index(index)
    if isinstance(values, (pd.Series, pd.DataFrame)) and isinstance(values.index, pd.DatetimeIndex) and isinstance(index, pd.PeriodIndex):
        out = values.copy()
        out.index = values.index.to_period(freq or index.freqstr).rename(index.name)
        return out.reindex(index) if name is None else out.reindex(index).rename(name)
    data = values.to_numpy() if isinstance(values, (pd.Series, pd.DataFrame)) else np.asarray(values)
    if len(data) != len(index):
        raise SpecificationError(f"the library returned {len(data)} rows for {len(index)} periods; pymetrix cannot place them")
    if data.ndim == 2:
        columns = values.columns if isinstance(values, pd.DataFrame) else None
        return pd.DataFrame(data, index=index, columns=columns)
    return pd.Series(data, index=index, name=name if name is not None else getattr(values, "name", None))
