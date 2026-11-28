"""Session: UI-owned workspace state. The core never sees it."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from ..errors import NotFoundError, PyMetrixError, SpecificationError
from .registry import Command, FieldSpec

__all__ = ["CommandError", "CommandRecord", "Session"]

SCRIPT_HEADER = "from pymetrix import *\n"


class CommandError(PyMetrixError, ValueError):
    """A dialog step whose fields do not validate."""


_MISSING = object()


def _named_objects(cmd: Command, session: Session) -> set[str]:
    """The workspace objects a dialog command names in its fields (e.g. data="data", fit="fit"); its ``name``
    field is where the result goes, not an input."""
    from dataclasses import fields as dc_fields

    names: set[str] = set()
    for f in dc_fields(cmd):  # type: ignore[arg-type]
        if f.name == "name":
            continue
        value = getattr(cmd, f.name)
        if isinstance(value, str):
            names |= {part.strip() for part in value.split(",") if part.strip() in session.objects}
    return names


def result_items(obj: Any) -> list[str]:
    """Series an object can contribute to a dataset (residuals, fitted values, columns...)."""
    from ..data import Dataset
    from ..estimation import FittedEquation

    if isinstance(obj, FittedEquation):
        items = [k for k in ("resid", "fitted") if getattr(obj, k) is not None]
        return items + [f"extra:{k}" for k in ("std_resid", "volatility", "structural_resid") if k in obj.extra]
    if isinstance(obj, Dataset):
        return [f"column:{c}" for c in obj.names]
    if type(obj).__name__ == "FittedStateSpace":  # smoothed (two-sided) and filtered (real-time) states
        states = [s for s in obj.model.states]
        return [f"smoothed:{s}" for s in states] + [f"filtered:{s}" for s in states]
    if hasattr(obj, "resid") and isinstance(getattr(obj, "resid"), Dataset):  # VARs and VECMs
        return [f"resid:{v}" for v in obj.resid.names]
    if type(obj).__name__ == "FittedPCA":  # the components
        return [f"scores:{c}" for c in obj.loadings.columns]
    if type(obj).__name__ == "FittedFactorModel":  # regression (Thomson) and Bartlett factor scores
        return [f"{how}:{c}" for how in ("regression", "bartlett") for c in obj.loadings.columns]
    return []


def result_series(obj: Any, item: str) -> Any:
    """The series named by ``item`` from ``result_items``."""
    kind, _, key = item.partition(":")
    if not key:
        return getattr(obj, item)
    if kind == "extra":
        return obj.extra[key]
    if kind == "column":
        return obj[key]
    if kind == "resid":
        return obj.resid[key]
    if kind in ("smoothed", "filtered"):
        return getattr(obj, kind)[key]
    if kind == "scores":
        return obj.scores[key]
    if kind in ("regression", "bartlett"):
        return obj.scores(kind)[key]
    raise NotFoundError(f"{item!r} is not available from this object")


def result_code(source: str, item: str) -> str:
    """Python for the series named by ``item``."""
    kind, _, key = item.partition(":")
    if not key:
        return f"{source}.{item}"
    return {"extra": f"{source}.extra[{key!r}]", "column": f"{source}[{key!r}]", "resid": f"{source}.resid[{key!r}]",
            "smoothed": f"{source}.smoothed[{key!r}]", "filtered": f"{source}.filtered[{key!r}]",
            "scores": f"{source}.scores[{key!r}]", "regression": f"{source}.scores()[{key!r}]",
            "bartlett": f"{source}.scores('bartlett')[{key!r}]"}[kind]


def standard_modules() -> dict[str, Any]:
    """Modules ready in the Immediate and Code windows, under their usual short names."""
    import numpy
    import pandas
    import scipy

    return {"np": numpy, "pd": pandas, "sp": scipy}


_IMPORTS = (("np", "import numpy as np"), ("pd", "import pandas as pd"), ("sp", "import scipy as sp"))


def _standard_imports(code: str) -> str:
    """Import lines for the short module names a script uses (np., pd., sp.)."""
    import re

    return "".join(f"{line}\n" for name, line in _IMPORTS if re.search(rf"(?<![\w.]){name}\.", code))


def _plugin_imports(code: str) -> str:
    """``from package import Class`` for registered classes outside pymetrix's namespace that ``code`` uses."""
    import re

    import pymetrix

    from ..plugins import GROUPS, registry

    used = set(re.findall(r"\b([A-Z]\w*)\(", code))
    lines = set()
    for group in GROUPS:
        for entry in registry(group).entries():
            cls = entry.obj
            name = getattr(cls, "__name__", "")
            core = pymetrix.__dict__.get(name)  # its own names only: a plugin's, found lazily (X13), still needs importing
            # a plugin's class named as one of pymetrix's own (finance's FactorModel result) does not displace it
            ours = isinstance(core, type) and core.__module__.split(".")[0] == "pymetrix"
            if isinstance(cls, type) and name in used and core is not cls and not ours:
                lines.add(f"from {cls.__module__} import {name}")
    return "".join(f"{line}\n" for line in sorted(lines))


@dataclass(frozen=True)
class CommandRecord:
    """One step of the session: a dialog command or typed code, what it read and what it wrote.

    ``reads`` and ``writes`` map object names to their versions (see ``Session.versions``), so
    the session can tell when a result was computed from an object that has since changed.
    """

    command: Command
    code: str
    ok: bool
    error: str | None = None
    created: datetime = field(default_factory=datetime.now)
    reads: dict[str, int] = field(default_factory=dict)
    writes: dict[str, int] = field(default_factory=dict)
    series: dict[str, frozenset[str]] = field(default_factory=dict)  # dataset -> the series the step used, when known
    skipped: bool = False  # kept from a project but not run (its typed code was not trusted); saved again
    seconds: float = 0.0  # how long the step took to run
    pruned: bool = False  # a dead end (nothing current depends on it): kept in the history, not run on opening
    deferred: bool = False  # an expensive step not computed yet (opened in Manual mode); Compute runs it
    outputs: tuple[str, ...] = ()  # the objects the step writes, as saved (for pruned and deferred steps)
    cautions: tuple[str, ...] = ()  # results it used that were out of date, frozen with changed inputs, or not computed
    call: Any = None  # the API call the step made (a pymetrix.calls.Call): a dialog's, or typed code recorded as one
    random_generator: str = field(default_factory=lambda: _current_generator())  # the bit generator its draws used (pymetrix.set_random_generator)
    proposed_by: str = ""  # who proposed the step the user then ran from its dialog (an AI assistant: "Claude"); see Session.proposing
    note: str = ""  # a deferred step: why it may not compute where it stands (an object it uses is missing there)
    inputs: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)  # a deferred step: the objects it reads, as at its place

    @property
    def typed(self) -> bool:
        """Code typed in the Immediate or Code window and kept as code (not recognised as API calls)."""
        return type(self.command).__name__ == "RunCode"

    @property
    def code_step(self) -> bool:
        """Typed code: kept as code, or recorded as an API call (a ``Call`` step), rather than a dialog."""
        return self.typed or type(self.command).__name__ == "Call"

    @property
    def replayable(self) -> bool:
        return self.ok and getattr(self.command, "replayable", True)


def _retry_key(record: CommandRecord) -> tuple[str, Any] | None:
    """What makes a later step a retry of a failed one: the same dialog (command id) on the same target, the object it
    makes (``name``) or, for a dialog that makes none, the object it works on (``source``, else ``data``). Typed
    code has none: an edited line is a different step."""
    if record.code_step:
        return None
    cmd = record.command
    for f in ("name", "source", "data"):
        value = getattr(cmd, f, None)
        if isinstance(value, str) and value:
            return (getattr(cmd, "command_id", "") or type(cmd).__name__, (f, value))
    return None


def _current_generator() -> str:
    from .._random import get_random_generator

    return get_random_generator()


_HIDDEN = {"In", "Out", "exit", "quit", "get_ipython", "open"}


def series_fingerprint(obj: Any) -> dict[str, int] | None:
    """For a dataset, a hash of its rows, its active sample and each series, so a change can be told apart from an
    addition: adding residuals to ``data`` does not make a fit of ``c ~ y`` out of date, but setting a new sample
    does (estimates use the active sample unless they give their own)."""
    from ..data import Dataset

    if not isinstance(obj, Dataset):
        return None
    import pandas as pd

    import hashlib
    import json

    frame = obj.frame
    out = {"(rows)": int(pd.util.hash_pandas_object(frame.index.to_frame(index=False), index=False).sum())}
    sample = obj.meta.sample
    text = json.dumps(sample.to_dict() if hasattr(sample, "to_dict") else sample, sort_keys=True, default=str)
    out["(sample)"] = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    for name in frame.columns:
        out[str(name)] = int(pd.util.hash_pandas_object(frame[name], index=False).sum())
    return out


def is_workspace_value(name: str, value: Any) -> bool:
    """Whether a name in the code namespace is a workspace object (shown in the Objects panel):
    not private, not a module, function or class, not one of pymetrix's own names."""
    import types

    import pymetrix

    if name.startswith("_") or name in _HIDDEN:
        return False
    if isinstance(value, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, type)):
        return False
    return getattr(pymetrix, name, _MISSING) is not value


UNDO_STEPS = 20  # how many dialog steps Undo can take back


def load_target(cmd: Any) -> str | None:
    """The dataset a step reads from a file or a data service (Load data, Load from connector, a recorded
    ``Dataset.read``), or None for any other step."""
    kind = type(cmd).__name__
    if kind in ("LoadData", "LoadFromConnector"):
        return getattr(cmd, "name", None)
    if kind == "Call" and getattr(cmd, "function", "") == "pymetrix:dataset.read":
        return getattr(cmd, "output", None) or None
    return None


class Session:
    def __init__(self) -> None:
        self.objects: dict[str, Any] = {}
        self.history: list[CommandRecord] = []
        self.initial: dict[str, Any] = {}  # objects placed in the session directly, as first seen by a command
        self.loaded: dict[str, Any] = {}  # datasets as read by Load data and Load from connector steps (projects keep them)
        self.opened: dict[str, bytes] = {}  # saved results as read by Open result commands, by path: projects keep a copy
        self.made: set[str] = set()  # names whose current objects commands produced
        self.path: Any = None  # project file, once saved or opened
        self.demo: str | None = None  # the demo it was opened from (pymetrix.demo), for the title and Reload demo
        self.dirty = False
        self.courses: list[str] = []  # course-material folders saved with the project (see pymetrix.help)
        self.versions: dict[str, int] = {}  # name -> version, raised whenever the object is replaced
        self._ids: dict[str, int] = {}
        self._series: dict[int, dict[str, int]] = {}  # version -> series fingerprint, for datasets
        self._counter = 0
        self.mode = "manual"  # calculation: "manual" marks results out of date; "automatic" recomputes them at once
        self.current_cautions: list[str] = []  # while a step runs: the cautions about what it uses (reports print them)
        self.frozen: set[str] = set()  # results never recomputed until unfrozen
        self.working_sets: dict[str, str] = {}  # dataset -> patterns or a selection's name: the series the app offers
        self.report_outline: list[dict[str, Any]] = []  # the report being put together: results' tabs, headings, text
        self._recomputing = False
        self.recalculation_stopped = False  # the last step stands, but Cancel stopped the recalculation after it
        self._undo: list[dict[str, Any]] = []  # the state before each of the last dialog steps (see ``undo``)
        self._redo: list[dict[str, Any]] = []  # the steps undone, last undone last, to run again (see ``redo``)
        self._redoing = False
        self.missing_plugins: dict[str, str] = {}  # when opened: plugins its steps need that are not here -> why
        self.record_calls = True  # typed code made only of pymetrix API calls is recorded as those calls (pymetrix.calls)
        self._code_recording: Any = None  # while the Immediate window runs a cell (begin_code ... end_code)
        self.attachments: dict[str, dict[str, bytes]] = {}  # plugins' own files, saved in the project: key -> name -> bytes
        self.save_log: list[dict[str, Any]] = []  # each save of its project: when, by which pymetrix, with which recorders
        self._opening = False  # while a project's steps are replayed (see commands.events: "opening")
        self.actor: Any = None  # who is acting, when not the person at the machine (commands.events.acting)
        self._proposal: tuple[type, str] | None = None  # the dialog step being proposed, and by whom (``proposing``)
        import pymetrix

        # where typed code runs: pymetrix's names, NumPy, pandas and SciPy, the workspace objects, and the user's own functions
        self.namespace: dict[str, Any] = {k: getattr(pymetrix, k) for k in pymetrix.__all__}
        self.namespace.update(standard_modules())

    def attach_course(self, folder: Any) -> None:
        """Save a folder of course material (notes, notebooks, slides) with this session's project."""
        from pathlib import Path

        from ..help import load_course

        path = str(Path(folder).expanduser().resolve())
        load_course(path)
        if path not in self.courses:
            self.courses.append(path)
            self.dirty = True

    def detach_course(self, folder: Any) -> None:
        from pathlib import Path

        path = str(Path(folder).expanduser().resolve())
        if path in self.courses:
            self.courses.remove(path)
            self.dirty = True

    def __getitem__(self, name: str) -> Any:
        try:
            return self.objects[name]
        except KeyError:
            raise NotFoundError(f"no object named {name!r} in the session; have {sorted(self.objects)}") from None

    def of_type(self, type_names: str) -> list[str]:
        """Names of objects whose type (or a base type) is one of the comma-separated names."""
        wanted = {t.strip() for t in type_names.split(",")}
        if "Reportable" in wanted:  # anything with views of its own: what a report can add
            from ..views import reportable

            return sorted(n for n, o in self.objects.items() if reportable(o))
        return sorted(
            n for n, o in self.objects.items() if wanted & {c.__name__ for c in type(o).__mro__}
        )

    def choices(self, spec: FieldSpec, values: dict[str, Any], scoped: bool = True) -> list[str]:
        """Options for a session-dependent form field; ``values`` holds the form's current entries. An optional
        single choice (default empty) starts with a blank, so a dropdown does not quietly pick the first option.
        A dataset's series are those of its working set, if it has one, unless ``scoped=False``."""
        options = self._choices(spec, values)
        if scoped and spec.kind == "series":
            shown = self.working_set(values.get(spec.depends_on or "", ""))
            if shown is not None:
                keep = set(shown)
                options = [o for o in options if o in keep]
        if not spec.multiple and (spec.empty or (not spec.required and spec.default in ("", None))) and "" not in options:
            options = ["", *options]
        return options

    def _choices(self, spec: FieldSpec, values: dict[str, Any]) -> list[str]:
        if spec.kind == "object" and spec.source:
            return self.of_type(spec.source)
        if spec.kind == "test":
            from ..stats.base import all_tests, available_tests

            target = self.objects.get(values.get(spec.depends_on or "", ""))
            return available_tests(target) if target is not None else all_tests()
        if spec.kind == "series":
            from ..data import Dataset

            data = self.objects.get(values.get(spec.depends_on or "", ""))
            return list(data.names) if isinstance(data, Dataset) else []
        if spec.kind in ("level", "entity"):
            from ..data import Dataset

            data = self.objects.get(values.get(spec.depends_on or "", ""))
            if not isinstance(data, Dataset):
                return []
            if spec.kind == "level":
                return [a.name for a in data.axes] + [n for n in data.names if not pd_numeric(data, n)] + [n for n in data.names if pd_numeric(data, n)]
            if len(data.entity_axes) != 1:
                return []
            return [str(v) for v in data.frame.index.get_level_values(data.entity_axes[0].name).unique()]
        if spec.kind in ("method", "extension"):
            from ..plugins import registry

            labels = registry(spec.source or "methods").labels()
            return ([str(spec.default)] if spec.default and str(spec.default) not in labels else []) + labels
        if spec.kind == "column":
            from ..io import columns

            path = values.get(spec.depends_on or "", "")
            blank = [] if spec.multiple else [""]  # a list to tick (Load data's columns of a large file) has no blank
            return blank + columns(path) if path else blank
        if spec.kind == "dataset":
            from ..datasets import available

            return list(available())
        if spec.kind == "named":  # report templates (plugins' too), themes (those added at runtime too), data connectors
            from ..io import connectors
            from ..report import report_templates
            from ..themes import themes

            return list({"report_templates": report_templates, "themes": themes, "connectors": connectors}[spec.source or ""]())
        if spec.kind == "item":
            return result_items(self.objects.get(values.get(spec.depends_on or "", "")))
        if spec.kind == "shock":  # a solution's or a template's shocks, in the model's order
            shocks = getattr(self.objects.get(values.get(spec.depends_on or "", "")), "shocks", None)
            return [str(s) for s in shocks] if isinstance(shocks, (tuple, list)) else []
        if spec.kind == "stylestep":  # a figure's recorded style calls: "0: line('gc', color='C3')"
            from ..views import figure_of

            try:
                figure = figure_of(self.objects[values.get(spec.depends_on or "", "")], values.get("tab") or "Graph")
            except Exception:
                return [""]
            return ["", *(f"{k}: {code}" for k, code in enumerate(figure.style_steps()))]
        if spec.kind == "axesmethod":  # the Axes methods fig.call records
            from ..figure_properties import axes_methods

            return axes_methods()
        if spec.kind == "figurepart":  # the series and the panels of a figure, for its property editor
            from ..views import figure_of

            try:
                figure = figure_of(self.objects[values.get(spec.depends_on or "", "")], values.get("tab") or "Graph")
            except Exception:
                return [""]
            series = self._choices(type(spec)(**{**spec.__dict__, "kind": "layer"}), values)
            panels = []
            for i, panel in enumerate(figure.panels):
                panels.append(f"panel {i}" + (f": {panel.title}" if panel.title else "") + " (axes)")
                if any(getattr(layer, "axis", "left") == "right" for layer in panel.layers):
                    panels.append(f"panel {i} right (axes)")
            return [*series, *panels]
        if spec.kind == "layer":  # the series a figure (or a result's figure) draws: "gdp (line)", "i (bars)"
            from ..figures import Band, Bars, Line
            from ..views import figure_of

            try:
                figure = figure_of(self.objects[values.get(spec.depends_on or "", "")], values.get("tab") or "Graph")
            except Exception:
                return [""]
            from ..figures import Ecdf, Estimates, Line3d

            kinds = {Line: "line", Bars: "bars", Band: "band", Estimates: "line", Ecdf: "line", Line3d: "line"}
            seen = {f"{layer.label} ({kinds[type(layer)]})": None for p in figure.panels for layer in p.drawn()
                    if type(layer) in kinds and getattr(layer, "label", "")}
            return ["", *seen]
        return [str(c) for c in spec.choices or ()]

    def options_form(self, spec: FieldSpec, values: dict[str, Any]) -> list[FieldSpec]:
        """Fields of an "options" sub-form: the settings of the extension chosen in field
        ``spec.depends_on`` (a test, method, variance, filter...), less those the main form sets itself."""
        from ..forms import settings_form
        from ..plugins import registry

        if spec.source == "figure-properties":  # matplotlib's properties of the figure part chosen
            from ..figure_properties import property_form
            from .figures import _part

            try:
                return property_form(_part(str(values.get(spec.depends_on or "", "")))[0])
            except ValueError:
                return []
        group = {"method": "methods", None: "tests", "": "tests"}.get(spec.source, spec.source)
        entry = registry(group).find(str(values.get(spec.depends_on or "", "")))  # type: ignore[arg-type]
        if entry is None:
            return []
        return settings_form(entry, set(spec.exclude))

    # -- running and recording ----------------------------------------------------------

    def _emit(self, kind: str, **fields: Any) -> None:
        """Tell the session observers (pymetrix.commands.events) what happened."""
        from .events import emit

        during = "opening" if self._opening else "recalculation" if self._recomputing else "work"
        emit(self, kind, during=during, **fields)

    def _snapshot(self) -> dict[str, int]:
        """Current versions, after noting any objects added or replaced outside commands."""
        for name, obj in self.objects.items():
            if self._ids.get(name) != id(obj):
                self._counter += 1
                self.versions[name], self._ids[name] = self._counter, id(obj)
                fingerprint = series_fingerprint(obj)
                if fingerprint is not None:
                    self._series[self._counter] = fingerprint
        for name in [n for n in self.versions if n not in self.objects]:
            del self.versions[name], self._ids[name]
        return dict(self.versions)

    def cautions(self, names: Any) -> list[str]:
        """Warnings about the named results: out of date, frozen while their inputs changed, or not computed."""
        names = set(names)
        stale, frozen, pending = self.stale(), self.stale(frozen=True), self.pending()
        out = []
        for n in sorted(names):
            if n in pending:
                out.append(f"{n} is not computed yet")
            elif n in stale:
                out.append(f"{n} is out of date: {stale[n]}")
            elif n in frozen:
                out.append(f"{n} is frozen, and its inputs have changed since: {frozen[n]}")
        return out

    @contextmanager
    def proposing(self, command_cls: type, by: str) -> Iterator[None]:
        """Mark the next ``command_cls`` step run inside the block as proposed by ``by`` (``CommandRecord.proposed_by``):
        an assistant's proposal, opened in its dialog for the user to run (extension API 1.9). Saved with the project,
        shown in History and noted in the script."""
        by = " ".join(str(by).split())[:60]
        if not by:
            raise SpecificationError("proposing: say who proposed the step")
        before = self._proposal
        self._proposal = (command_cls, by)
        try:
            yield
        finally:
            self._proposal = before

    def _record(self, cmd: Command, code: str, ok: bool, error: str | None, before: dict[str, int], reads: set[str],
                seconds: float = 0.0, cautions: tuple[str, ...] = (), call: Any = None) -> CommandRecord:
        after = self._snapshot()
        writes = {n: v for n, v in after.items() if before.get(n) != v}
        series: dict[str, frozenset[str]] = {}
        if type(cmd).__name__ != "RunCode":  # dialog steps know which series they use; typed code does not
            from .builtin import series_used

            used = series_used(cmd)
            if used is not None and getattr(cmd, "data", None) in reads:
                series[cmd.data] = used  # type: ignore[attr-defined]
        if call is None or not ok:
            call = cmd if type(cmd).__name__ == "Call" else (_dialog_call(cmd, self) if ok else None)
        proposed_by = ""
        if self._proposal is not None and isinstance(cmd, self._proposal[0]) and not self._recomputing:
            proposed_by = self._proposal[1]  # the proposed step itself, not a recalculation after it
            self._proposal = None
        record = CommandRecord(cmd, code, ok, error, reads={n: before[n] for n in reads if n in before}, writes=writes, series=series,
                               seconds=seconds, outputs=tuple(sorted(writes)), cautions=cautions, call=call, proposed_by=proposed_by)
        self.history.append(record)
        if ok:
            self.made |= set(writes)
            self.dirty = True
            _note_call(self, call)
        self._emit("step" if ok else "failed", records=(record,))
        return record

    def _remember_initial(self) -> None:
        for name, obj in self.objects.items():
            if name not in self.made and name not in self.initial:
                self.initial[name] = obj

    def _form_dataset(self, values: dict[str, Any]) -> Any:
        """The dataset a form works on: its ``data`` field, else the first dataset it names."""
        from ..data import Dataset

        names = [values.get("data")] + [v for v in values.values() if isinstance(v, str)]
        for name in names:
            obj = self.objects.get(name) if isinstance(name, str) else None
            if isinstance(obj, Dataset) and obj.time_axis is not None:
                return obj
        return None

    def period_choices(self, values: dict[str, Any]) -> list[str]:
        """The periods of the dataset a form works on, for period fields: what their spinners step through."""
        import pandas as pd

        data = self._form_dataset(values)
        if data is None:
            return []
        periods = pd.PeriodIndex(data.frame.index.get_level_values(data.time_axis.name)).unique().sort_values()
        return [str(p) for p in periods]

    def sample_bounds(self, values: dict[str, Any]) -> tuple[str, str] | None:
        """The first and last periods of the active sample of the dataset a form works on (None without one):
        what empty start and end fields stand for."""
        import pandas as pd

        data = self._form_dataset(values)
        if data is None or data.meta.sample is None:
            return None
        periods = pd.PeriodIndex(data.frame.index.get_level_values(data.time_axis.name))[data.sample_mask()]
        return (str(periods.min()), str(periods.max())) if len(periods) else None

    def run(self, cmd: Command) -> Any:
        """Run a command and record it as a step (with the automatic recalculation after it, in Automatic mode). A
        dialog step can be undone (``undo``); typed code cannot, since it may change objects in place."""
        if self._recomputing:
            return self._run(cmd)
        if not self._redoing:  # a new step: what was undone can no longer be redone
            self._redo.clear()
        if type(cmd).__name__ == "RunCode":
            self._undo.clear()
            return self._run(cmd)
        state = self._undo_state()
        result = self._run(cmd)
        self._undo.append({"state": state, "after": self._snapshot(), "step": self.history[len(state["history"])],
                           "frozen": set(self.frozen)})
        del self._undo[:-UNDO_STEPS]
        return result

    def add(self, cmd: Command, deferred: bool = False, outputs: Any = None) -> Any:
        """Add a step to the session. ``deferred=False`` (the default) runs it now, exactly as ``run`` does, and returns
        its result. ``deferred=True`` records it without running it, as Manual mode defers a slow step when a project
        is opened: the step is pending (``pending()``) until ``compute()`` (the app's Compute) runs it in its place;
        History lists it as not computed, the script marks it, and a saved project keeps it deferred. It returns None.

        ``outputs`` names the objects the step will make, when they cannot be told from it: a call's ``output``
        (a dialog's call too) or a dialog's ``name`` otherwise; for typed code, the names it assigns. A deferred step
        is checked as far as it can be without running it (its code, its span, and the objects it uses, unless it
        uses results of other pending steps), and can be undone. What it uses is told from the command (the objects
        its fields name, its call's refs, the names typed code reads) and recorded as its ``reads``, as they are now:
        the steps that make them are not dead ends, and ``compute`` runs it against them as they are here, even when
        a later step makes one of them again."""
        if not deferred:
            if outputs is not None:
                raise SpecificationError("outputs is for a deferred step: a step run now records what it makes")
            return self.run(cmd)
        made = _deferred_outputs(cmd, self, outputs)
        call = cmd if type(cmd).__name__ == "Call" else _dialog_call(cmd, self)
        code = call.to_code() if call is not None else cmd.to_code()
        _check_code(cmd, code)
        record = _deferred(self, CommandRecord(cmd, code, ok=False, outputs=made, call=call))
        errors = span_errors(cmd) + ([record.note] if record.note else [])  # the note: what it uses must exist now
        if errors:
            raise CommandError("; ".join(errors))
        if not self._redoing:
            self._redo.clear()
        state = self._undo_state()
        self.history.append(record)
        self.dirty = True
        self._undo.append({"state": state, "after": self._snapshot(), "step": record, "frozen": set(self.frozen)})
        del self._undo[:-UNDO_STEPS]
        self._emit("step", records=(record,), detail={"deferred": True})
        return None

    def _undo_state(self) -> dict[str, Any]:
        return {"objects": dict(self.objects), "history": list(self.history), "versions": dict(self.versions),
                "ids": dict(self._ids), "made": set(self.made), "loaded": dict(self.loaded), "opened": dict(self.opened),
                "initial": dict(self.initial), "frozen": set(self.frozen)}

    def undo_blocked(self) -> str | None:
        """Why the last step cannot be undone, or None if it can."""
        if not self._undo:
            return "there is no step to undo (typed code, and opening a project, start afresh)"
        entry = self._undo[-1]
        if self._snapshot() != entry["after"]:
            return "objects have changed since the last step (in typed code or by hand), so it cannot be undone"
        if self.frozen != entry["frozen"]:
            return "results were frozen or unfrozen since the last step, so it cannot be undone"
        kept = entry["state"]["history"]
        n = len(kept)
        if len(self.history) <= n or self.history[n] is not entry["step"] or any(a is not b for a, b in zip(self.history, kept)):
            return "the history has changed since the last step (cleaned up, or steps removed), so it cannot be undone"
        return None

    def undo_changes(self) -> set[str]:
        """The objects Undo would remove or replace (empty if it cannot undo)."""
        if self.undo_blocked() is not None:
            return set()
        before = self._undo[-1]["state"]["objects"]
        return {n for n, v in self.objects.items() if before.get(n, _MISSING) is not v}

    def undo_label(self) -> str:
        """The step Undo would take back, as its code's first line ("" if none)."""
        if self.undo_blocked() is not None:
            return ""
        step = self.history[len(self._undo[-1]["state"]["history"])]
        line = step.code.strip().splitlines()[0] if step.code.strip() else type(step.command).__name__
        return line if len(line) <= 60 else line[:57] + "…"

    def undo(self) -> CommandRecord:
        """Take back the last dialog step: the objects as they were before it, and its record (with the recalculation
        after it) out of the history. Only while the objects are as the step left them; the last 20 steps."""
        reason = self.undo_blocked()
        if reason is not None:
            raise CommandError(reason)
        state = self._undo.pop()["state"]
        undone = self.history[len(state["history"])]
        removed = tuple(self.history[len(state["history"]):])  # the step, and the recalculation after it
        self.objects.clear()
        self.objects.update(state["objects"])
        self.history[:] = state["history"]
        self.versions, self._ids = dict(state["versions"]), dict(state["ids"])
        self.made, self.frozen = set(state["made"]), set(state["frozen"])
        self.loaded, self.opened, self.initial = dict(state["loaded"]), dict(state["opened"]), dict(state["initial"])
        self.dirty = True
        self.push()
        self._redo.append({"command": undone.command, "after": self._snapshot(), "frozen": set(self.frozen),
                           "history": list(self.history)})
        del self._redo[:-UNDO_STEPS]
        self._emit("undone", records=removed)
        return undone

    def redo_blocked(self) -> str | None:
        """Why the last step undone cannot be run again, or None if it can."""
        if not self._redo:
            return "there is no step to redo (a new step, typed code or opening a project start afresh)"
        entry = self._redo[-1]
        if self._snapshot() != entry["after"] or self.frozen != entry["frozen"]:
            return "objects have changed since the undo (in typed code or by hand), so the step cannot be redone"
        if len(self.history) != len(entry["history"]) or any(a is not b for a, b in zip(self.history, entry["history"])):
            return "the history has changed since the undo, so the step cannot be redone"
        return None

    def redo_label(self) -> str:
        """The step Redo would run again, as its code's first line ("" if none)."""
        if self.redo_blocked() is not None:
            return ""
        command = self._redo[-1]["command"]
        line = (_dialog_code(command, self) or "").strip().splitlines()
        text = line[0] if line else type(command).__name__
        return text if len(text) <= 60 else text[:57] + "…"

    def redo(self) -> CommandRecord:
        """Run the last step undone again, as it was recorded (with the recalculation after it): what Undo took back,
        in order, while the objects are as Undo left them. A new step clears what can be redone. Observers hear of it
        as a step, as of any step run."""
        reason = self.redo_blocked()
        if reason is not None:
            raise CommandError(reason)
        entry = self._redo.pop()
        steps = len(self.history)
        self._redoing = True
        try:
            self.run(entry["command"])
        except Exception:
            self._redo.append(entry)  # still there to redo, once whatever stopped it is fixed
            raise
        finally:
            self._redoing = False
        if self._redo:  # the step undone before this one now follows from where the redo left the session
            self._redo[-1].update(after=self._snapshot(), frozen=set(self.frozen), history=list(self.history))
        return self.history[steps]

    def _run(self, cmd: Command) -> Any:
        errors = span_errors(cmd) + cmd.validate(self)
        if errors:
            raise CommandError("; ".join(errors))
        call = _dialog_call_before(cmd, self)  # a call built from the session: as it is before the step
        code = call.to_code() if call is not None else cmd.to_code()  # as _dialog_code
        _check_code(cmd, code)  # a step that cannot be written as Python is neither run nor recorded
        self._remember_initial()
        before = self._snapshot()
        reads = cmd.reads(self, before) if hasattr(cmd, "reads") else _named_objects(cmd, self)
        from time import perf_counter

        cautions = tuple(self.cautions(reads)) if not self._recomputing else ()
        self.current_cautions = list(cautions)
        started = perf_counter()
        recorder = None
        try:
            if self.record_calls and type(cmd).__name__ == "RunCode" and getattr(cmd, "replayable", False):
                from .._api import recording

                with recording(self.namespace, source=cmd.source) as recorder:  # type: ignore[attr-defined]
                    result = cmd.run(self)
            else:
                result = cmd.run(self)
        except Exception as exc:
            from ..progress import Cancelled

            if not isinstance(exc, Cancelled):  # a cancelled step leaves no trace in the history: as if never run
                self._record(cmd, code, False, str(exc), before, reads, perf_counter() - started, cautions)
            else:  # but observers hear of it
                self._emit("cancelled", code=code, detail={"command": type(cmd).__name__})
            raise
        finally:
            self.current_cautions = []
        target = load_target(cmd)
        if target is not None and target in self.objects:
            self.loaded[target] = self.objects[target]
        self._note_reads([cmd])
        calls = self._typed_calls(cmd.code, recorder, result, before) if recorder is not None else None  # type: ignore[attr-defined]
        if calls:
            self._record_calls(calls, before, perf_counter() - started, cautions)
        else:
            self._record(cmd, code, True, None, before, reads, perf_counter() - started, cautions, call)
        self.push()
        if not self._recomputing:
            self.recalculation_stopped = False
        try:
            self._after_change()
        except Exception as exc:
            from ..progress import Cancelled

            if not isinstance(exc, Cancelled):
                raise
            self.recalculation_stopped = True  # the step is recorded; what depends on it stays out of date
        return result

    def _load_step(self, name: str) -> CommandRecord | None:
        """The step that last read ``name`` from its source (Load data, Load from connector or ``Dataset.read``)."""
        return next((r for r in reversed(self.history) if r.ok and not r.pruned and load_target(r.command) == name), None)

    def refreshable(self) -> list[str]:
        """The datasets File → Refresh data… can read again: those a load step read from a file or a data service."""
        return sorted(n for n in self.loaded if n in self.objects and self._load_step(n) is not None)

    def refresh(self, *names: str) -> list[CommandRecord]:
        """Read datasets again from their sources (all those ``refreshable`` lists, or the ones named): File → Refresh
        data…. A project keeps the data as it was read, so opening it gives the same numbers whatever the source holds
        by then; refreshing takes the source's current data (a load pinned to a vintage reads that vintage again).

        Each load step runs again, recorded as a new step, and so do the later steps that changed the dataset in place
        (series generated or added to it), in order. What was computed from the dataset is then out of date: Automatic
        mode recomputes it, Manual mode marks it. Typed code that changed the dataset is not run again: the refresh is
        refused, naming the step, so the change can be made again by hand after refreshing. Returns the steps run."""
        wanted = list(names) or self.refreshable()
        plans: list[tuple[CommandRecord, list[CommandRecord]]] = []
        for name in wanted:
            step = self._load_step(name)
            if step is None or name not in self.objects:
                raise SpecificationError(f"{name} was not read from a file or a data service, so it cannot be refreshed"
                                         + (f" (refreshable: {', '.join(self.refreshable())})" if self.refreshable() else ""))
            later = [r for r in self.history[self.history.index(step) + 1:]
                     if r.ok and not r.pruned and name in r.writes and name in r.reads]
            typed = [r for r in later if type(r.command).__name__ == "RunCode"]
            if typed:
                raise SpecificationError(f"{name} was changed by typed code after it was read ({typed[0].code.strip().splitlines()[0]!r}), "
                                         "which refreshing would not run again: refresh it in code, or make the change again after "
                                         "refreshing")
            plans.append((step, later))
        start = len(self.history)
        for step, later in plans:
            self.run(step.command)
            for r in later:
                self.run(r.command)
        return self.history[start:]

    def _typed_calls(self, code: str, recorder: Any, result: Any, before: dict[str, int]) -> list[Any] | None:
        """The API calls typed code consists of (see ``pymetrix.calls.as_calls``), if it can be replayed as them: each
        result kept under a name used once, and each object a call uses existing before it."""
        from ..calls import as_calls

        calls = as_calls(code, recorder, self.namespace, result)
        if not calls:
            return None
        outputs = [c.output for c in calls if c.output]
        if len(outputs) != len(set(outputs)) or any(o not in self.objects for o in outputs):
            return None
        known = set(before)
        for c in calls:
            if not c.refs() <= known:
                return None
            known |= {c.output} if c.output else set()
        return calls

    def _note_reads(self, commands: list[Any]) -> None:
        """Data read by a typed ``Dataset.read`` call is kept like Load data's: projects save the data as read, so
        reopening one uses the same data, whatever the source holds by then (its vintage request says when)."""
        for c in commands:
            if type(c).__name__ == "Call" and c.function == "pymetrix:dataset.read" and c.output in self.objects:
                self.loaded[c.output] = self.objects[c.output]

    def _record_calls(self, calls: list[Any], before: dict[str, int], seconds: float, cautions: tuple[str, ...]) -> None:
        """Record typed code as the API calls it made, one step each."""
        self._note_reads(calls)
        after = self._snapshot()
        local = dict(before)
        start = len(self.history)
        for c in calls:
            writes = {c.output: after[c.output]} if c.output in after else {}
            record = CommandRecord(c, c.to_code(), True, reads={n: local[n] for n in c.refs() if n in local}, writes=writes,
                                   seconds=seconds / len(calls), outputs=tuple(writes), cautions=cautions, call=c)
            self.history.append(record)
            local.update(writes)
            self.made |= set(writes)
            _note_call(self, c)
        self.dirty = True
        self._emit("step", records=tuple(self.history[start:]))

    def restore(self, cmd: Command, objects: dict[str, Any], seconds: float = 0.0) -> CommandRecord:
        """Record a step whose results are already known (from a project's results cache) without running it."""
        self._remember_initial()
        before = self._snapshot()
        reads = cmd.reads(self, before) if hasattr(cmd, "reads") else _named_objects(cmd, self)
        self.objects.update(objects)
        record = self._record(cmd, cmd.to_code(), True, None, before, reads, seconds)
        self.push()
        return record

    # -- calculation: automatic or manual, frozen results, dead ends and deferred steps ----------

    def set_mode(self, mode: str) -> None:
        """``"automatic"``: out-of-date results are recomputed after every step, in dependency order (frozen ones
        excepted); ``"manual"``: they are marked out of date until ``recompute()`` (the app's Recalculate)."""
        if mode not in ("automatic", "manual"):
            raise SpecificationError("the calculation mode is 'automatic' or 'manual'")
        self.mode = mode
        self.dirty = True
        self._after_change()

    def freeze(self, *names: str) -> None:
        """Keep these results as they are: never recomputed, even when their inputs change, until unfrozen."""
        unknown = [n for n in names if n not in self.objects]
        if unknown:
            raise NotFoundError(f"no objects named {unknown}")
        self.frozen |= set(names)
        self.dirty = True

    # -- the report outline: figures, tables and results collected from every workspace ------------------

    def add_to_report(self, source: str, tab: str | None = None, caption: str = "") -> dict[str, Any]:
        """Put a result in the report outline: one of its tabs (a figure or a table), or, without ``tab``, its
        summary and graph. Collecting is not a step: exporting the outline is (the Report panel's Export…)."""
        from ..views import titles

        if source not in self.objects:
            raise NotFoundError(f"no object named {source!r}")
        if tab is not None and tab not in titles(self.objects[source]):
            raise NotFoundError(f"{source} has no tab {tab!r}; its tabs are {', '.join(titles(self.objects[source]))}")
        item = {"kind": "item", "source": source, "tab": tab, "caption": caption}
        self.report_outline.append(item)
        self.dirty = True
        return item

    def add_report_heading(self, text: str, level: int = 2) -> None:
        if level not in (1, 2, 3):
            raise SpecificationError("a heading's level is 1, 2 or 3")
        self.report_outline.append({"kind": "heading", "text": str(text), "level": int(level)})
        self.dirty = True

    def add_report_text(self, text: str) -> None:
        self.report_outline.append({"kind": "text", "text": str(text)})
        self.dirty = True

    def move_report_item(self, index: int, to: int) -> None:
        item = self.report_outline.pop(index)
        self.report_outline.insert(max(0, min(to, len(self.report_outline))), item)
        self.dirty = True

    def remove_report_item(self, index: int) -> None:
        self.report_outline.pop(index)
        self.dirty = True

    # -- working sets: the series of a large dataset the app's lists show ---------------------------------

    def set_working_set(self, dataset: str, spec: str | None) -> list[str]:
        """Show only some of a dataset's series in the app: in every dialog's series lists, the spreadsheet and the
        dataset's window. ``spec`` is patterns (``"gdp_*, cpi*, u"``) or the name of a selection; None or "" shows
        every series again. A view, not a step: commands still record the series they use by name, so scripts run
        the same. Returns the series it shows."""
        if not spec or not str(spec).strip():
            self.working_sets.pop(dataset, None)
            self.dirty = True
            return list(self[dataset].names) if dataset in self.objects else []
        names = self._resolve_working_set(dataset, str(spec).strip())
        if not names:
            raise SpecificationError(f"{spec!r} matches no series of {dataset}")
        self.working_sets[dataset] = str(spec).strip()
        self.dirty = True
        return names

    def working_set(self, dataset: str) -> list[str] | None:
        """The series a dataset's working set shows (None: no working set, every series)."""
        spec = self.working_sets.get(dataset)
        if not spec or dataset not in self.objects:
            return None
        try:
            return self._resolve_working_set(dataset, spec)
        except Exception:  # noqa: BLE001 - a selection removed or a pattern gone stale: show everything
            return None

    def _resolve_working_set(self, dataset: str, spec: str) -> list[str]:
        from ..data import Dataset
        from ..selection import Selection

        data = self.objects.get(dataset)
        if not isinstance(data, Dataset):
            raise SpecificationError(f"no dataset named {dataset!r}")
        chosen = self.objects.get(spec)
        if isinstance(chosen, Selection):  # a selection: its series, those the dataset has
            return [n for n in chosen if n in data.names]
        patterns = [p.strip() for p in spec.split(",") if p.strip()]
        found = set(data.series(*patterns))  # names and patterns, as Data > Select series reads them
        return [n for n in data.names if n in found]

    def unfreeze(self, *names: str) -> None:
        """Let these results (all, without names) be recomputed again."""
        self.frozen -= set(names or self.frozen)
        self.dirty = True
        self._after_change()

    def _after_change(self) -> None:
        if self.mode == "automatic" and not self._recomputing:
            self.recompute()

    def needed(self) -> list[bool]:
        """For each step of the history, whether the current objects depend on it. Typed code, steps that write
        nothing (exports, saved rounds) and deferred steps always count; pruned steps never do."""
        wanted = set(self._snapshot().values())
        keep = [False] * len(self.history)
        for i in range(len(self.history) - 1, -1, -1):
            r = self.history[i]
            if r.pruned or not (r.ok or r.skipped or r.deferred):
                continue
            if r.typed or r.skipped or r.deferred or not r.writes or any(v in wanted for v in r.writes.values()):
                keep[i] = True
                wanted |= set(r.reads.values())
        return keep

    def dead_ends(self) -> list[CommandRecord]:
        """Steps nothing current depends on: results later replaced or deleted, and the steps only they used."""
        return [r for r, k in zip(self.history, self.needed()) if not k and (r.ok or r.pruned)]

    def prune(self) -> list[CommandRecord]:
        """Remove the dead ends from the history (and so from the script and the saved project)."""
        dead = {id(r) for r in self.dead_ends()}
        removed = [r for r in self.history if id(r) in dead]
        self.history = [r for r in self.history if id(r) not in dead]
        if removed:
            self.dirty = True
            self._emit("removed", records=tuple(removed), detail={"reason": "dead ends cleaned up"})
        return removed

    def pending(self) -> dict[str, CommandRecord]:
        """Results not computed yet (deferred when the project was opened, or added with ``add(cmd, deferred=True)``):
        name -> the step that makes it."""
        return {n: r for r in self.history if r.deferred for n in r.outputs}

    def compute(self, names: Any = None) -> list[CommandRecord]:
        """Run deferred steps, in history order: all, or those that make (or lead to) the named results. Each runs
        against the objects it reads as they were at its place (``CommandRecord.inputs``), as if it had run there."""
        from ..output import quiet_show

        done: list[CommandRecord] = []
        steps = [i for i, r in enumerate(self.history) if r.deferred]
        if names is not None:  # the deferred steps the named results need, back through their inputs
            need, chosen = set(names), set()
            produced = [n for j in steps for n in self.history[j].outputs]
            for i in reversed(steps):
                r = self.history[i]
                if set(r.outputs) & need:
                    chosen.add(i)
                    need |= _mentioned(r, produced)
            steps = [i for i in steps if i in chosen]
        with quiet_show():
            for i in steps:
                self._recomputing = True
                try:
                    with _generator_of(self.history[i]), self._inputs_at(self.history[i]):  # as at its place
                        self.run(self.history[i].command)
                finally:
                    self._recomputing = False
                done.append(self.history.pop())
                self.history[i] = done[-1]
        self.push()
        self._after_change()
        return done

    @contextmanager
    def _inputs_at(self, record: CommandRecord) -> Iterator[None]:
        """Run a deferred step against the objects it reads as they were at its place in the history: those a later
        step has replaced since are put back for the step, then the current ones again (unless the step writes them)."""
        current = self._snapshot()
        swapped = {n: (self.objects.get(n, _MISSING), current.get(n), self._ids.get(n)) for n in record.inputs
                   if n in record.reads and current.get(n) != record.reads[n]}
        for n in swapped:
            self.objects[n] = record.inputs[n]
            self.versions[n], self._ids[n] = record.reads[n], id(record.inputs[n])
        try:
            yield
        finally:
            written = set(self.history[-1].writes) if self.history and self.history[-1].command is record.command else set()
            for n, (obj, version, ident) in swapped.items():
                if n in written:
                    continue
                if obj is _MISSING:
                    self.objects.pop(n, None)
                    self.versions.pop(n, None)
                    self._ids.pop(n, None)
                else:
                    self.objects[n], self.versions[n], self._ids[n] = obj, version, ident  # type: ignore[assignment]

    # typed code: the Immediate window runs cells itself and reports them here

    def push(self) -> None:
        """Copy the workspace objects into the code namespace (after a dialog command changed them)."""
        for name in [n for n, v in self.namespace.items() if is_workspace_value(n, v) and n not in self.objects]:
            del self.namespace[name]
        self.namespace.update(self.objects)

    def pull(self) -> None:
        """Take objects created, replaced or deleted by typed code into the workspace."""
        current = {n: v for n, v in self.namespace.items() if is_workspace_value(n, v)}
        for name in [n for n in self.objects if n not in current]:
            del self.objects[name]
        for name, value in current.items():
            if self.objects.get(name, _MISSING) is not value:
                self.objects[name] = value

    def begin_code(self) -> dict[str, int]:
        """Call before typed code runs elsewhere (the Immediate window); pass the result to ``end_code``."""
        self._remember_initial()
        self.push()
        if self.record_calls:
            from .._api import recording

            self._code_recording = recording(self.namespace, source="immediate")
            self._code_recording.__enter__()
        return self._snapshot()

    def end_code(self, before: dict[str, int], code: str, ok: bool, error: str | None = None, source: str = "immediate") -> CommandRecord:
        """Record typed code that has already run in ``self.namespace``."""
        from .code import RunCode

        self._redo.clear()
        self._undo.clear()  # typed code may change objects in place: the steps before it cannot be undone

        recorder = None
        if self._code_recording is not None:
            recorder = self._code_recording.recorder
            self._code_recording.__exit__(None, None, None)
            self._code_recording = None
        self.pull()
        cmd = RunCode(code, source)  # type: ignore[arg-type]
        calls = None
        if ok and recorder is not None and cmd.replayable:
            from dataclasses import replace

            recorder.calls = [(replace(c, source=source), v) for c, v in recorder.calls]
            last = recorder.calls[-1][1] if recorder.calls else None
            calls = self._typed_calls(code, recorder, last, before)
        if calls:
            self._record_calls(calls, before, 0.0, ())
            self._after_change()
            return self.history[-1]
        record = self._record(cmd, cmd.to_code(), ok, error, before, cmd.reads(self, before))
        if ok:
            self._after_change()
        return record

    def run_code(self, code: str, source: str = "script") -> Any:
        """Run Python in the session's namespace and record it, as the Immediate window does."""
        from .code import RunCode

        return self.run(RunCode(code, source))

    def resolved_failures(self) -> dict[int, int]:
        """Failed dialog steps that a later step of the same dialog on the same target ran successfully: the failed
        step's place in the history -> the place of the step that resolved it. They stay in the history (greyed) but
        are no longer problems; nothing else changes (failed steps are never saved or replayed). Undoing the
        successful step makes the failure count again."""
        out: dict[int, int] = {}
        later: dict[tuple[str, Any], int] = {}  # retry key -> the earliest later successful step
        for i in range(len(self.history) - 1, -1, -1):
            record = self.history[i]
            key = _retry_key(record)
            if key is None:
                continue
            if record.ok:
                later[key] = i
            elif not record.skipped and key in later:
                out[i] = later[key]
        return out

    # -- stale results ----------------------------------------------------------------------

    def producer(self, name: str) -> CommandRecord | None:
        """The step that made the current version of ``name`` (None for objects placed directly)."""
        version = self._snapshot().get(name)
        return next((r for r in reversed(self.history) if r.ok and r.writes.get(name) == version), None)

    def stale(self, frozen: bool = False) -> dict[str, str]:
        """Objects computed from inputs that have changed since: name -> reason. Frozen results count as up to
        date (so nothing is recomputed because of them); ``frozen=True`` lists instead the frozen results whose
        inputs have changed."""
        versions = self._snapshot()
        memo: dict[str, str | None] = {}

        def reason(name: str, seen: frozenset[str]) -> str | None:
            if name in memo:
                return memo[name]
            if name in self.frozen and seen:
                return None  # a frozen input is taken as it is
            record = self.producer(name)
            why = None
            if record is not None:
                for used, version in record.reads.items():
                    if used in record.writes:
                        continue  # updated by the same step, e.g. adding a series to data
                    if versions.get(used) != version:
                        why = self._change(used, version, versions.get(used), name, record.series.get(used))
                        if why:
                            break
                    if used not in seen and reason(used, seen | {name}):
                        why = f"it depends on {used}, which is out of date"
                        break
            memo[name] = why
            return why

        found = {n: r for n in self.objects if (r := reason(n, frozenset()))}
        return {n: r for n, r in found.items() if (n in self.frozen) == frozen}

    def _change(self, used: str, then: int, now: int | None, name: str, only: frozenset[str] | None = None) -> str | None:
        """Why ``used`` counts as changed for ``name`` (None: only series were added to a dataset, or
        only series the step did not use, when ``only`` lists those it did)."""
        if now is None:
            return f"{used} no longer exists"
        old, new = self._series.get(then), self._series.get(now)
        if old is None or new is None:
            return f"{used} has changed since {name} was computed"
        if old["(rows)"] != new["(rows)"]:
            return f"the periods or entities of {used} have changed since {name} was computed"
        if old.get("(sample)") != new.get("(sample)"):
            return f"the active sample of {used} has changed since {name} was computed"
        changed = [s for s in old if s not in ("(rows)", "(sample)") and old[s] != new.get(s) and (only is None or s in only)]
        if not changed:
            return None
        listed = ", ".join(changed[:3]) + (" …" if len(changed) > 3 else "")
        return f"{listed} in {used} {'has' if len(changed) == 1 else 'have'} changed since {name} was computed"

    def recompute(self, names: Any = None) -> list[CommandRecord]:
        """Re-run, in history order, the steps that produced stale objects (all, or those named)."""
        from ..output import quiet_show

        done: list[CommandRecord] = []
        wanted = None if names is None else set(names)
        self._recomputing = True
        try:
            with quiet_show():
                return self._recompute_steps(wanted, done)
        finally:
            self._recomputing = False

    def _recompute_steps(self, wanted: set[str] | None, done: list[CommandRecord]) -> list[CommandRecord]:
        for _ in range(len(self.history) + 1):
            targets = [n for n in self.stale() if wanted is None or n in wanted]
            producers = [r for n in targets if (r := self.producer(n)) is not None]
            if not producers:
                break
            first = min(producers, key=self.history.index)
            with _generator_of(first):  # the generator the result was first made with
                self.run(first.command)
            done.append(self.history[-1])
        return done

    def script(self) -> str:
        """Python that reproduces the session's successful commands, importing any plugin classes it uses. Steps run
        with a random generator other than the default are preceded by ``set_random_generator(...)``."""
        lines, using = [], "pcg64"
        for r in self.history:
            if not ((r.ok or r.skipped or r.pruned or r.deferred) and r.code.strip()):
                continue
            if r.random_generator != using:
                lines.append(f"set_random_generator({r.random_generator!r})  # the random generator these steps used")
                using = r.random_generator
            if r.proposed_by:
                lines.append(f"# proposed by {r.proposed_by} (an assistant); run from its dialog")
            if r.deferred:
                lines.append(f"# not computed yet in the session (deferred): {', '.join(r.outputs) or 'the step'}")
            lines.append(r.code if (r.ok or r.pruned or r.deferred) else _not_run(r.code))
        body = "\n".join(lines)
        return _standard_imports(body) + SCRIPT_HEADER + _plugin_imports(body) + _engine_line() + "\n" + body + "\n"


def _note_call(session: Session, call: Any) -> None:
    """Remember the call that made a step's result (``pymetrix.calls.call_of``), for its saved provenance."""
    output = getattr(call, "output", "") if type(call).__name__ == "Call" else ""
    if output and output in session.objects:
        from ..calls import _remember_call

        _remember_call(session.objects[output], call)


def _deferred_outputs(cmd: Any, session: Session, given: Any) -> tuple[str, ...]:
    """The objects a step added without running will make: ``given``, else its call's output (a dialog's call too),
    a dialog's ``name``, or the names typed code assigns at its top level."""
    if given is not None:
        names = (given,) if isinstance(given, str) else tuple(given)
    elif type(cmd).__name__ == "Call":
        names = (cmd.output,) if cmd.output else ()
    elif type(cmd).__name__ == "RunCode":
        import ast

        try:
            body = ast.parse(cmd.code).body
        except SyntaxError:
            body = []
        found: dict[str, None] = {}
        for stmt in body:
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target] if isinstance(stmt, (ast.AnnAssign, ast.AugAssign)) else []
            for t in targets:
                for node in ast.walk(t):
                    if isinstance(node, ast.Name):
                        found[node.id] = None
        names = tuple(found)
    else:
        call = _dialog_call(cmd, session)
        name = call.output if call is not None and call.output else getattr(cmd, "name", "")
        names = (name,) if isinstance(name, str) and name else ()
    if not names or not all(isinstance(n, str) and n.isidentifier() for n in names):
        raise SpecificationError("a deferred step must say which objects it makes: give outputs=['name', ...]")
    return tuple(dict.fromkeys(names))


def _deferred(session: Session, record: CommandRecord) -> CommandRecord:
    """A step recorded without running, deferred: what it reads, told from the command since it has not run (the
    objects it says it reads, those its fields name and its call's refs), as they are now, at its place, so the steps
    that make them are not dead ends and ``compute`` runs it against them; and a note when an object it names is
    missing (other than results of pending steps), so it may not compute."""
    cmd = record.command
    call = record.call if record.call is not None else cmd if type(cmd).__name__ == "Call" else _dialog_call(cmd, session)
    versions = session._snapshot()
    used = set(cmd.reads(session, versions)) if hasattr(cmd, "reads") else _named_objects(cmd, session)
    if call is not None and call is not cmd:
        used |= call.refs()
    waiting = sorted(session.pending())
    note = "" if waiting and _mentioned(record, waiting) else _missing_inputs(cmd, call, session)
    return replace(record, ok=False, deferred=True, call=call, note=note, reads={n: versions[n] for n in used if n in versions},
                   inputs={n: session.objects[n] for n in used if n in session.objects})


def _missing_inputs(cmd: Any, call: Any, session: Session) -> str:
    """Why a step not run cannot run where it stands: the objects it names that do not exist there ("" if none)."""
    problems: list[str] = []
    try:
        problems += cmd.validate(session)
        if call is None and getattr(cmd, "to_call", None) is not None and _takes_session(cmd.to_call):
            cmd.to_call(session)  # a dialog whose call is built from the objects it names: they must exist
    except Exception as exc:  # noqa: BLE001 - the reason is the note
        problems.append(str(exc))
    if call is not None and call is not cmd:
        problems += [f"no object named {n!r}" for n in sorted(call.refs()) if n not in session.objects]
    return "; ".join(dict.fromkeys(problems))


def _generator_of(record: CommandRecord) -> Any:
    """A context in which the step's random draws use the generator it was run with."""
    from .._random import random_generator

    return random_generator(record.random_generator)


def _dialog_call(cmd: Any, session: Session | None = None) -> Any:
    """The API call a dialog command makes, for dialogs built on calls (None for the others). Dialogs acting on
    whatever object they are given (Forecast: ``fit.forecast(...)``) take the session, to find that object's class."""
    to_call = getattr(cmd, "to_call", None)
    if to_call is None:
        return None
    try:
        return to_call(session) if _takes_session(to_call) else to_call()
    except Exception:  # noqa: BLE001 - a call that cannot be described is not recorded with the step
        return None


def _takes_session(to_call: Any) -> bool:
    import inspect

    return bool(inspect.signature(to_call).parameters)


def _dialog_call_before(cmd: Any, session: Session) -> Any:
    """The call of a dialog whose call is built from the session, as it stands before the step runs (Generate series
    for each ``ip_*``: the series matched then, not those the step adds). None for the others: theirs is made after."""
    to_call = getattr(cmd, "to_call", None)
    return _dialog_call(cmd, session) if to_call is not None and _takes_session(to_call) else None


def _check_code(cmd: Any, code: str) -> None:
    """Refuse a dialog step whose code is not Python (``forecast = .forecast(...)``: an object field left empty), before
    it runs: such a step could never be replayed, so it must not reach the history. Typed code is the user's own."""
    if type(cmd).__name__ == "RunCode" or not code:
        return
    import ast

    try:
        ast.parse(code)
    except SyntaxError:
        raise CommandError("the step could not be written as Python, so it was not run: a field it needs is empty or "
                           f"not valid ({code.strip().splitlines()[0][:80] if code.strip() else type(cmd).__name__})") from None


def _dialog_code(cmd: Any, session: Session) -> str:
    """A dialog's code: its call's, when the call needs the session to build; else its own ``to_code``."""
    to_call = getattr(cmd, "to_call", None)
    if to_call is not None and _takes_session(to_call):
        call = _dialog_call(cmd, session)
        if call is not None:
            return call.to_code()
    return cmd.to_code()


def _engine_line() -> str:
    """The Kalman filter engine, when the session chose one: replayed scripts then filter the same way."""
    from .. import statespace

    if statespace.ENGINE == "auto":
        return ""
    return f"import pymetrix.statespace\npymetrix.statespace.ENGINE = {statespace.ENGINE!r}  # the filter this session used\n"


def _mentioned(record: CommandRecord, names: list[str]) -> set[str]:
    """Which of ``names`` a step uses: named in a dialog's fields, or appearing in typed code."""
    import re
    from dataclasses import fields as dc_fields

    text = " ".join(str(getattr(record.command, f.name)) for f in dc_fields(record.command))  # type: ignore[arg-type]
    return {n for n in names if re.search(rf"(?<![\w.]){re.escape(n)}(?!\w)", text)}


def span_errors(cmd: Any) -> list[str]:
    """A step's start must not be after its end (when both are periods of the same frequency)."""
    import pandas as pd

    start, end = getattr(cmd, "start", None), getattr(cmd, "end", None)
    if not (isinstance(start, str) and isinstance(end, str) and start.strip() and end.strip()):
        return []
    try:
        a, b = pd.Period(start.strip()), pd.Period(end.strip())
    except (ValueError, TypeError):
        return []
    if a.freq != b.freq:
        return []
    return [f"the start ({start.strip()}) is after the end ({end.strip()})"] if a > b else []


def _not_run(code: str) -> str:
    lines = "\n".join(f"# {line}" for line in code.splitlines())
    return f"# Not run when the project was opened (its typed code is not trusted):\n{lines}"


def pd_numeric(data: Any, name: str) -> bool:
    """True for a numeric series (categories such as region are offered first for grouping)."""
    import pandas as pd

    return bool(pd.api.types.is_numeric_dtype(data.frame[name].dtype))
