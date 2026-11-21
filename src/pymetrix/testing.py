"""Conformance kit: checks every extension runs in its own test suite (Extension API v1).

Use from pytest::

    from pymetrix.testing import check_entry, check_result, crosscheck

    def test_liml_conforms():
        check_entry(registry("methods").get("LIML"))

Each check raises ``AssertionError`` with a message saying what to fix.
``check_plugin(PLUGIN)`` runs the declarative checks on everything a plugin
registers; numerical checks (``check_result``, ``check_project``,
``check_determinism``, ``crosscheck``) need data and are written per plugin.
"""

from __future__ import annotations

import collections.abc as cabc
import json
import types
import typing
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, get_args, get_origin, get_type_hints

import numpy as np

from ._api import renamed_arguments
from ._serial import to_jsonable
from .forms import build, form_spec, option_string
from .plugins import Entry, PluginInfo, installed, registry

__all__ = [
    "check_api_reference", "check_axes", "check_course", "check_determinism", "check_entry", "check_form", "check_help", "check_plugin",
    "check_project", "check_result", "check_roundtrip", "crosscheck", "fingerprint", "sample_settings", "write_api_reference",
]

DECLARATIVE_GROUPS = ("methods", "models", "variances", "tests", "filters", "solvers")
_SCALARS = (str, int, float, bool, type(None))


# -- forms ---------------------------------------------------------------------


def _allowed(tp: Any) -> bool:
    """Types a setting may have, so it can appear in a form and be saved in a project."""
    origin = get_origin(tp)
    if origin is Annotated:
        return _allowed(get_args(tp)[0])
    if tp in _SCALARS:
        return True
    if origin is Literal:
        return all(isinstance(a, _SCALARS) for a in get_args(tp))
    if origin in (typing.Union, types.UnionType):
        return all(_allowed(a) for a in get_args(tp))
    if origin in (cabc.Sequence, list, tuple):
        args = [a for a in get_args(tp) if a is not Ellipsis]
        return all(_allowed(a) and a not in (dict,) for a in args)
    if origin in (cabc.Mapping, dict):
        key, value = get_args(tp)
        return key is str and _allowed(value)
    return isinstance(tp, type) and is_dataclass(tp)  # nested settings such as Cluster(...)


def sample_settings(cls: type) -> dict[str, Any]:
    """Placeholder values for required settings, so a default instance can be built."""
    hints = get_type_hints(cls, include_extras=True)
    out: dict[str, Any] = {}
    for f in fields(cls):
        if not f.init or f.default is not MISSING or f.default_factory is not MISSING:
            continue
        tp = hints[f.name]
        tp = get_args(tp)[0] if get_origin(tp) is Annotated else tp
        text = str(tp)
        if "Sequence" in text or "list" in text or "tuple" in text:
            out[f.name] = ("x", "z") if "str" in text else (1,)
        elif tp is int or "int" in text:
            out[f.name] = 1
        elif tp is float:
            out[f.name] = 1.0
        else:
            out[f.name] = "x"
    return out


def check_form(cls: type, *, help_required: bool = True) -> None:
    """The settings make a form: allowed types, one line of help each, defaults that build,
    and settings text that rebuilds the same object."""
    assert is_dataclass(cls), f"{cls.__name__}: settings must be a (frozen) dataclass"
    assert getattr(cls, "__dataclass_params__").frozen, f"{cls.__name__}: make the dataclass frozen"
    hints = get_type_hints(cls, include_extras=True)
    specs = form_spec(cls)
    shown = {s.name for s in specs}
    for f in fields(cls):
        if f.init and f.name in shown:
            assert _allowed(hints[f.name]), (f"{cls.__name__}.{f.name}: type {hints[f.name]} cannot be shown in a form or saved; "
                                            "use str, int, float, bool, Literal, sequences of these, Mapping[str, ...] or a settings dataclass "
                                            "(or Widget('hidden') for settings made in code only)")
    if help_required:
        missing = [s.name for s in specs if not s.help and s.kind != "options"]
        assert not missing, f"{cls.__name__}: give these settings one line of help (Widget(help=...)): {missing}"
    for s in specs:
        if s.show_if is not None:
            assert s.show_if[0] in shown, f"{cls.__name__}.{s.name}: show_if names {s.show_if[0]!r}, which is not a setting"
        if s.kind == "number" and s.default is not None:
            low, high, _ = s.bounds()
            assert low <= s.default <= high, f"{cls.__name__}.{s.name}: default {s.default} is outside min/max"
    required = sample_settings(cls)
    try:
        obj = cls(**required)
    except Exception as exc:
        raise AssertionError(f"{cls.__name__}: the defaults do not build an object: {exc}") from exc
    # settings text, as the dialog writes it, rebuilds the object
    optional = [s for s in specs if not s.required and s.kind != "options"]
    text = option_string(optional, {s.name: getattr(obj, s.name) for s in optional})
    assert text == "", f"{cls.__name__}: default settings should give empty settings text, got {text!r}"
    entry = Entry("check", cls.__name__.lower(), cls.__name__, cls, PluginInfo("check", ""))
    rebuilt = build(entry, "", **required)
    assert to_jsonable(rebuilt) == to_jsonable(obj), f"{cls.__name__}: settings text does not rebuild the object"


# -- help ----------------------------------------------------------------------


@renamed_arguments(target="obj")
def check_help(obj: Any, *, background_required: bool = False) -> None:
    """Help at three levels: usage (docstring and settings help), reference (``wraps``
    resolves when the library is installed) and background (``topics`` exist; with
    ``background_required`` at least one topic or reference, as for contrib plugins)."""
    from .help import _TOPICS, help_page, resolve

    page = help_page(obj)
    name = page.title
    assert page.summary, f"{name}: add a docstring whose first line says what it does"
    assert len(page.summary) <= 200, f"{name}: keep the docstring's first line to one sentence"
    if page.wraps:
        entry = obj if isinstance(obj, Entry) else None
        lib = page.wraps.split(".")[0]
        if entry.available if entry is not None else installed(lib):
            try:
                resolve(page.wraps)
            except (ImportError, AttributeError) as exc:
                raise AssertionError(f"{name}: wraps={page.wraps!r} does not resolve: {exc}") from exc
    if page.docs:
        assert page.docs.startswith(("https://", "http://")), f"{name}: docs must be a URL"
    unknown = [t for t in page.topics if t not in _TOPICS]
    assert not unknown, f"{name}: unknown help topics {unknown}; add them with pymetrix.help.add_topic"
    if background_required:
        assert page.topics or page.references, f"{name}: link background help (topics=...) or cite references=..."


# -- reproducibility -----------------------------------------------------------


def _namespace(obj: Any) -> dict[str, Any]:
    import importlib

    import pymetrix

    ns: dict[str, Any] = {k: getattr(pymetrix, k) for k in pymetrix.__all__}
    for cls in {type(obj), *(type(v) for v in vars(obj).values() if is_dataclass(v))} if is_dataclass(obj) else {type(obj)}:
        ns[cls.__name__] = cls
        ns.update({k: v for k, v in vars(importlib.import_module(cls.__module__)).items() if isinstance(v, type)})
    return ns


def check_roundtrip(obj: Any) -> None:
    """The generated Python and the project-file JSON both rebuild an equal object."""
    from .commands.builtin import _code

    code = _code(obj)
    try:
        rebuilt = eval(code, _namespace(obj))  # noqa: S307 - our own generated code
    except Exception as exc:
        raise AssertionError(f"{type(obj).__name__}: generated code {code!r} does not run: {exc}") from exc
    assert to_jsonable(rebuilt) == to_jsonable(obj), f"{type(obj).__name__}: {code!r} rebuilds a different object"
    data = json.loads(json.dumps(to_jsonable(obj)))  # must be JSON
    assert data == to_jsonable(rebuilt)


def fingerprint(obj: Any) -> list[tuple[str, str]]:
    """What an object window shows, as text: equal fingerprints mean equal results."""
    from .views import view_sections

    out = []
    for title, table in view_sections(obj):
        if title == "Provenance":
            continue  # carries the creation time
        out.append((title, str(table)))
    return out


def check_result(result: Any, *, backend: str | None = None, plugin: str | None = None) -> None:
    """A core result type, provenance naming the backend (and plugin), and views that render."""
    from .data import Dataset
    from .estimation import FittedEquation
    from .output import Table
    from .stats import TestResult

    known = (FittedEquation, TestResult, Dataset, Table)
    views = registry("views")
    assert isinstance(result, known) or views.of(result) is not None, (
        f"{type(result).__name__} is not a core result type; register its views with pymetrix.views.register_view")
    prov = getattr(result, "provenance", None)
    if backend is not None or plugin is not None:
        assert prov is not None, f"{type(result).__name__}: record provenance"
        method = dict(prov.method)
        if backend is not None:
            assert str(method.get("backend", "")).startswith(backend), f"provenance.method['backend'] should name {backend!r}, got {method.get('backend')!r}"
        if plugin is not None:
            assert str(method.get("plugin", "")).startswith(plugin), f"provenance.method['plugin'] should name {plugin!r}, got {method.get('plugin')!r}"
    shown = fingerprint(result)
    assert shown, f"{type(result).__name__}: no views"


_LIBRARIES = ("statsmodels", "linearmodels", "arch", "sklearn")
_VERBS = ("resid", "fitted", "factors", "smoothed", "filtered", "endog")  # public results that are not dataclass fields


def check_axes(result: Any) -> None:
    """One time-series abstraction (object-model item 6): the user-facing series and tables of a result use
    pymetrix's axes. Fails on a ``DatetimeIndex`` (a library's dates), on an unnamed ``RangeIndex`` over time data
    (positions where periods belong), and on a library object outside ``backend_result`` (and, for results that need
    one, its documented field)."""
    import dataclasses

    import pandas as pd

    from .data import Dataset

    needs = getattr(type(result), "needs_backend", False)
    raw_fields = set(getattr(type(result), "_backend_fields", ())) | {"design"}
    names = [f.name for f in dataclasses.fields(result)] if dataclasses.is_dataclass(result) else []
    names += [v for v in _VERBS if v not in names and hasattr(type(result), v)]
    for name in names:
        if name.startswith("_") or name in raw_fields:
            continue
        try:
            value = getattr(result, name)
        except Exception:  # noqa: BLE001 - a verb that needs more than the result (data, a library object) is not checked
            continue
        where = f"{type(result).__name__}.{name}"
        if isinstance(value, Dataset):
            value = value.frame
        if isinstance(value, (pd.Series, pd.DataFrame)):
            _check_index(value, where)
        elif type(value).__module__.split(".")[0] in _LIBRARIES and not needs:
            raise AssertionError(f"{where} is a {type(value).__module__}.{type(value).__name__}: keep library objects in backend_result")
    extra = getattr(result, "extra", None)
    if isinstance(extra, dict):
        for key, value in extra.items():
            if type(value).__module__.split(".")[0] in _LIBRARIES:
                raise AssertionError(f"{type(result).__name__}.extra[{key!r}] is a library object: keep it in backend_result")
            if isinstance(value, (pd.Series, pd.DataFrame)):
                _check_index(value, f"{type(result).__name__}.extra[{key!r}]")


def _check_index(value: Any, where: str) -> None:
    """No library dates; and no bare positions where labels belong: a Series, or a table of numbers only, on an
    unnamed RangeIndex. (Long tables whose columns say what each row is, as tidy tables do, are fine.)"""
    import pandas as pd

    index = value.index
    levels = [index.get_level_values(i) for i in range(index.nlevels)] if isinstance(index, pd.MultiIndex) else [index]
    for level in levels:
        assert not isinstance(level, pd.DatetimeIndex), f"{where} has a library's DatetimeIndex: put it on pymetrix's periods"
    numeric_only = isinstance(value, pd.Series) or all(pd.api.types.is_numeric_dtype(t) for t in value.dtypes)
    if isinstance(index, pd.RangeIndex) and index.name is None and len(index) > 1 and numeric_only:
        raise AssertionError(f"{where} has an unnamed RangeIndex: name what the rows are (periods, lag, horizon...)")


def check_determinism(make: Any) -> None:
    """Calling ``make()`` twice gives the same result (randomness only through a seed)."""
    first, second = fingerprint(make()), fingerprint(make())
    assert first == second, "two runs with the same settings and seed differ"


def check_project(session: Any, command: Any, directory: str | Path) -> None:
    """Run a command, save the project, reopen it: the result is identical and the command's code round-trips.

    A step slow enough to count as expensive (on a busy machine, any) is deferred on opening in Manual mode unless its
    result was saved with the project; it is computed here, as Compute would, before the results are compared."""
    from .commands import load_project, save_project

    result = session.run(command)
    name = getattr(command, "name", None) or getattr(command, "data", None)  # e.g. Filter adds to its dataset
    path = save_project(session, Path(directory) / "conformance.pymx")
    reopened = load_project(path)
    reopened.compute()  # nothing to do unless a step was deferred
    if name is not None:
        assert fingerprint(reopened[name]) == fingerprint(result), f"{type(command).__name__}: the reopened project gives a different result"
    assert reopened.script() == session.script(), "the reopened project's script differs"


def crosscheck(ours: Any, reference: Any, *, rtol: float = 1e-6, atol: float = 1e-10, what: str = "") -> None:
    """Our numbers agree with the wrapped library called directly."""
    np.testing.assert_allclose(np.asarray(ours, dtype=float), np.asarray(reference, dtype=float), rtol=rtol, atol=atol,
                               err_msg=f"cross-check failed{': ' + what if what else ''}")


# -- whole entries and plugins ---------------------------------------------------


def check_entry(entry: Entry, *, background_required: bool = False) -> None:
    """The declarative checks for one registered extension."""
    cls = entry.obj
    assert entry.label, f"{entry.qualified}: needs a label"
    if entry.group in DECLARATIVE_GROUPS:
        check_form(cls)
        check_help(entry, background_required=background_required)
        if entry.available:
            obj = entry.build(**sample_settings(cls))
            check_roundtrip(obj)
    if entry.group == "methods":
        assert entry.kind in ("ls", "iv", "ml", "panel") or entry.kind, f"{entry.qualified}: declare kind"
        assert callable(getattr(cls, "estimate", None)), f"{entry.qualified}: methods implement estimate(eq, data, *, span, where, weights, missing)"
    if entry.group == "variances":
        assert callable(getattr(cls, "fit", None)), f"{entry.qualified}: variances implement fit(method, eq, data, design)"
    if entry.group == "tests":
        assert cls.name and cls.null and cls.applies_to, f"{entry.qualified}: tests declare name, null and applies_to"
    if entry.group == "filters":
        assert callable(getattr(cls, "trend", None)) or callable(getattr(cls, "cycle", None)), f"{entry.qualified}: filters implement trend or cycle"
        assert callable(getattr(cls, "describe", None)), f"{entry.qualified}: filters implement describe(f)"
    if entry.group == "solvers":
        assert callable(getattr(cls, "solve_period", None)), f"{entry.qualified}: solvers implement solve_period(system)"
    if entry.group == "commands":
        check_form(cls)
        from .commands.registry import COMMAND_IDS, MIGRATIONS

        assert COMMAND_IDS.get(cls.command_id, None) is not None and COMMAND_IDS[cls.command_id].cls is cls, \
            f"{entry.qualified}: register commands with @command so they get a stable ID for projects"
        missing = [v for v in range(1, cls.schema_version) if (cls.command_id, v) not in MIGRATIONS]
        assert not missing, f"{cls.command_id}: schema {cls.schema_version} needs migrations from {missing} (pymetrix.commands.registry.migration)"


def check_plugin(info: PluginInfo, *, contrib: bool = False) -> list[Entry]:
    """Metadata, then ``check_entry`` on everything the plugin registers. ``contrib``
    applies the contrib-tier standard (background help required)."""
    assert info.supported, f"{info.name}: API {info.api} is not supported"
    assert info.version, f"{info.name}: give a version"
    assert info.license, f"{info.name}: declare a licence (SPDX identifier, e.g. BSD-3-Clause)"
    if contrib:
        assert info.maintainer, f"{info.name}: contrib plugins name a maintainer"
    entries = [e for g in DECLARATIVE_GROUPS for e in registry(g).entries() if e.plugin.name == info.name]
    assert entries, f"{info.name}: registers nothing"
    for entry in entries:
        check_entry(entry, background_required=contrib)
    return entries


# -- course material ---------------------------------------------------------------


def check_course(path: str | Path) -> Any:
    """Course material loads, every note's links match a help page, and every linked file exists.

    Instructors can run this on a course folder before giving it to students::

        python -c "from pymetrix.testing import check_course; check_course('my_course')"
    """
    from .help import load_course

    course = load_course(path)
    assert course.notes, f"{path}: no notes (.md) or notebooks (.ipynb) found"
    assert not course.problems, f"{course.name}: " + "; ".join(course.problems)
    unlinked = [n.path.name for n in course.notes if not n.links]
    assert not unlinked, f"{course.name}: these notes link to no help page (add links: [...] to their front matter): {unlinked}"
    return course


# -- a course notebook's reference to the pymetrix it uses ----------------------------------------------

REFERENCE_TAG = "pymetrix-reference"
_PUBLIC = ("pymetrix", "pymetrix.stats", "pymetrix.commands", "pymetrix.toolkit", "pymetrix.help", "pymetrix.testing")
_REFERENCE_INTRO = ("## pymetrix used in this lab\n\n"
                    "Every pymetrix function, class and method this lab calls, with its arguments and the first line of its "
                    "documentation. For the full documentation, type the name and a question mark in a cell (`backtest?`), "
                    "or press Shift+Tab inside the brackets of a call.")


def _code_cells(nb: dict) -> list[tuple[int, str]]:
    return [(i, "".join(c.get("source", []))) for i, c in enumerate(nb.get("cells", []))
            if c.get("cell_type") == "code" and REFERENCE_TAG not in c.get("metadata", {}).get("tags", [])]


def _run_traced(path: Path) -> list[str]:
    """Run a notebook's code cells in a scratch folder, recording each pymetrix function, class and method that the
    cells call directly, as ``"module:qualname"``."""
    import os
    import sys
    import tempfile

    nb = json.loads(path.read_text(encoding="utf-8"))
    marker = f"<{path.name} cell "
    used: dict[str, None] = {}

    def record(frame: Any, event: str, arg: Any) -> None:
        if event != "call" or frame.f_back is None or not frame.f_back.f_code.co_filename.startswith(marker):
            return
        module = frame.f_globals.get("__name__", "")
        code = frame.f_code
        if code.co_name == "__init__" and "self" in frame.f_locals:  # a class made (dataclass __init__s are generated)
            cls = type(frame.f_locals["self"])
            module, qualname = cls.__module__, cls.__qualname__
        elif module == "pymetrix._api" and code.co_name == "wrapper":  # @api or @renamed_arguments: the function it wraps
            local = frame.f_locals
            fn = local["spec"].function if "spec" in local else local["fn"]
            fn = getattr(fn, "__api__", None) and fn.__api__.function or fn  # @renamed_arguments above @api
            module, qualname = fn.__module__, fn.__qualname__.removesuffix(".__init__")  # a class made through a renaming __init__
        else:
            qualname = getattr(code, "co_qualname", code.co_name)
        if (module == "pymetrix" or module.startswith("pymetrix.")) and "<" not in qualname \
                and not any(part.startswith("_") for part in qualname.split(".")):
            used.setdefault(f"{module}:{qualname}", None)

    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:  # pymetrix draws without pyplot; switching an imported pyplot's backend
        matplotlib.use("Agg")  # can fail (matplotlib 3.8 with IPython 9), and is not needed
    cwd, ns = os.getcwd(), {"__name__": "__main__"}
    with tempfile.TemporaryDirectory() as folder:
        os.chdir(folder)
        sys.setprofile(record)
        try:
            for i, source in _code_cells(nb):
                exec(compile(source, f"{marker}{i}>", "exec"), ns)  # noqa: S102 - the instructor's own notebook
        finally:
            sys.setprofile(None)
            os.chdir(cwd)
    return list(used)


def _resolve(entry: str) -> Any:
    import importlib

    module, _, qualname = entry.partition(":")
    obj: Any = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part) if not isinstance(obj, type) else obj.__dict__.get(part, getattr(obj, part))
    return obj


def _first_sentence(obj: Any) -> str:
    import inspect
    import re

    doc = inspect.getdoc(obj.fget if isinstance(obj, property) else obj) or ""
    if isinstance(obj, type) and doc.startswith(f"{obj.__name__}("):
        doc = ""  # the signature a dataclass writes when it has no docstring
    paragraph = " ".join(doc.split("\n\n")[0].split())
    end = re.search(r"(?<!\be\.g)(?<!\bi\.e)(?<!\betc)\.(\s|$)", paragraph)
    sentence = paragraph[: end.start() + 1] if end else paragraph
    return sentence.replace("``", "`").replace("|", "\\|")


def _signature(obj: Any) -> str:
    import inspect

    if isinstance(obj, property):
        return ""
    if isinstance(obj, (classmethod, staticmethod)):
        obj = obj.__func__
    try:
        params = inspect.signature(obj).parameters.values()
    except (TypeError, ValueError):
        return "(...)"
    names = [("*" if p.kind is p.VAR_POSITIONAL else "**" if p.kind is p.VAR_KEYWORD else "") + p.name
             for p in params if p.name not in ("self", "cls")]
    return f"({', '.join(names)})"


def _home(obj: Any, name: str) -> str:
    """Where to import a function or class from: the first public pymetrix module that has it."""
    import importlib

    for module in _PUBLIC:
        if getattr(importlib.import_module(module), name, None) is obj:
            return module
    return obj.__module__


def _reference_markdown(entries: list[str]) -> str:
    groups: dict[str, list[str]] = {}
    methods: list[str] = []
    for entry in sorted(entries, key=lambda e: e.partition(":")[2].lower()):
        obj, qualname = _resolve(entry), entry.partition(":")[2]
        row = f"| `{qualname}{_signature(obj)}` | {_first_sentence(obj)} |"
        if "." in qualname:
            methods.append(row)
        else:
            groups.setdefault(_home(obj, qualname), []).append(row)
    parts = [_REFERENCE_INTRO]
    for module in sorted(groups, key=lambda m: (m != "pymetrix", m)):
        parts.append(f"### From `{module}`\n\n| | |\n|---|---|\n" + "\n".join(groups[module]))
    if methods:
        parts.append("### Methods and properties of the results\n\n| | |\n|---|---|\n" + "\n".join(methods))
    return "\n\n".join(parts)


def _reference_cell(nb: dict) -> dict | None:
    return next((c for c in nb.get("cells", []) if REFERENCE_TAG in c.get("metadata", {}).get("tags", [])), None)


def write_api_reference(notebook: str | Path) -> bool:
    """Run a course notebook and write, as its last cell, a reference to every pymetrix function, class and method
    it calls, from their signatures and docstrings. Returns whether the notebook changed.

    Instructors run it after editing a lab; ``check_api_reference`` then says when it needs running again::

        python -c "from pymetrix.testing import write_api_reference; write_api_reference('my_lab.ipynb')"
    """
    path = Path(notebook)
    nb = json.loads(path.read_text(encoding="utf-8"))
    entries = sorted(_run_traced(path))
    source = [line + "\n" for line in _reference_markdown(entries).split("\n")]
    source[-1] = source[-1].rstrip("\n")
    new = {"cell_type": "markdown", "metadata": {"tags": [REFERENCE_TAG], "pymetrix": {"reference": entries}}, "source": source}
    cell = _reference_cell(nb)
    if cell == new:
        return False
    if cell is None:
        nb["cells"].append(new)
    else:
        nb["cells"][nb["cells"].index(cell)] = new
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return True


def check_api_reference(notebook: str | Path) -> None:
    """A course notebook's reference cell (``write_api_reference``) is current: every entry still exists, its
    arguments and first line match the code, and every name the notebook imports from pymetrix is listed."""
    import ast

    path = Path(notebook)
    nb = json.loads(path.read_text(encoding="utf-8"))
    cell = _reference_cell(nb)
    assert cell is not None, f"{path.name}: no reference cell; run pymetrix.testing.write_api_reference on it"
    entries = cell["metadata"].get("pymetrix", {}).get("reference", [])
    try:
        expected = _reference_markdown(entries)
    except (ImportError, AttributeError) as exc:
        raise AssertionError(f"{path.name}: the reference lists something pymetrix no longer has ({exc}); rerun write_api_reference") from exc
    assert "".join(cell["source"]) == expected, f"{path.name}: the reference is out of date with the docstrings; rerun write_api_reference"
    undocumented = [e for e in entries if not _first_sentence(_resolve(e))]
    assert not undocumented, f"{path.name}: these have no docstring to describe them: {undocumented}"
    listed = {e.partition(":")[2].partition(".")[0] for e in entries}
    imported = {alias.asname or alias.name for _, source in _code_cells(nb) for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("pymetrix") for alias in node.names}
    missing = sorted(imported - listed)
    assert not missing, f"{path.name}: imports {missing} from pymetrix but the reference does not list them; rerun write_api_reference"
