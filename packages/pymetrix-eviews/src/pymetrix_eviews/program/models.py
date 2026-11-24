"""Phase 2: models. ``model m``, ``m.append``, ``m.merge eq``, ``@identity``, ``@add`` and ``@innov`` are collected into
one text, which the core reader reads (``read_eviews(text, data=..., params=...)``) before the model is first used;
its refusals are passed into the report at their lines. ``m.solve``, ``m.solveopt``, ``m.addassign``, ``m.addinit``,
``m.scenario``, ``m.override`` and ``m.exclude`` become ``Model.solve`` with a solver and a ``Scenario``, the add
factors ``Model.add_factors``, and the solution's endogenous series are added to the page under the scenario's
alias (``y_0``), as EViews writes them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .estimation import _commas, _options, _value
from .ir import Arg, Attr, Cmd, New, Op, RefTo, SampleSpec, Step, Var

__all__ = ["Models"]


@dataclass
class _Scenario:
    name: str
    alias: str
    overrides: list[str] = field(default_factory=list)
    excludes: list[str] = field(default_factory=list)


@dataclass
class _Model:
    var: str
    eviews: str
    page: Any
    lines: list[tuple[int, str]] = field(default_factory=list)  # (program line, the line of the model's text)
    scalars: dict[str, int] = field(default_factory=dict)  # the program's scalars it uses -> the line declaring them
    merged: dict[str, int] = field(default_factory=dict)  # equation -> line
    dirty: bool = True
    read: bool = False
    refused: str = ""
    endogenous: list[str] = field(default_factory=list)
    equations: list[str] = field(default_factory=list)  # names of its equations, for add factors
    identities: set[str] = field(default_factory=set)
    solver: Any = None
    solver_note: str = ""
    scenarios: dict[str, _Scenario] = field(default_factory=dict)
    active: str = "baseline"
    factors: str = ""  # the variable holding its add factors (addinit), if any
    solving: str = ""  # the variable holding the Model read (m_model = m.model), which solves
    assigned: bool = False
    building: list[Any] = field(default_factory=list)  # the outcomes of the lines that build it (set by the translator)
    given: bool = False  # given to the translation (objects=), not built by the program


class Models:
    """The phase 2 handlers for models, mixed into the ``Translator`` (``data.py``)."""

    def _cmd_model(self, st: Any) -> tuple[list[Op], str]:
        from .data import _error

        m = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*", st.rest)
        if m is None:
            raise _error("model names the model: model m")
        key = m.group(1).lower()
        var, note = self._name(m.group(1), f"model {key}")
        page = self.current  # a model's text needs no page; solving it does
        model = _Model(var, m.group(1), page)
        model.scenarios["baseline"] = _Scenario("baseline", "_0")
        self.models[key] = model
        self.types[key] = "model"
        model.lines.append((st.line, f"model {key}"))
        self.building.append((key, st.line))
        return [], "; ".join(x for x in ("the model's text is collected and read (read_eviews) before its first use", note) if x)

    def _model_method(self, st: Any, key: str) -> tuple[list[Op], str]:
        from .data import _error, _left

        model = self.models[key]
        word = st.word
        if model.given and word in ("append", "merge"):
            raise _left(f"{model.eviews} is given to the translation (objects=): its equations are not changed here")
        if model.refused and word not in ("append", "merge"):
            raise _left(f"the model {model.eviews} is not read: {model.refused}")
        if word == "append":
            text = st.rest.strip()
            if not text:
                raise _error("append takes a line of the model: m.append y = c(1) + x")
            self._model_scalars(model, text)
            model.lines.append((st.line, f"{key}.append {text}"))
            model.dirty = True
            self.building.append((key, st.line))
            return [], "collected into the model's text"
        if word == "merge":
            name = st.rest.strip().lower()
            fit = self.fits.get(name)
            if fit is None:
                raise _left(f"{st.rest.strip()} is not an equation estimated by a line translated so far")
            if fit.kind != "equation" or fit.method != "ls" or fit.method == "ardl":
                raise _left(f"{name} is estimated by {fit.method}: the model reader re-reads least-squares equations only; "
                            "add the fitted equation to a Model in pymetrix (Model([...]))")
            if any(t in fit.spec.lower() for t in ("ar(", "ma(")):
                raise _left(f"{name} has ARMA terms: the model reader does not read them")
            if model.page is None:
                model.page = fit.page
            if fit.page is not model.page:
                raise _left(f"{name} is estimated on another page")
            sample = fit.sample
            if isinstance(sample, Var) or (isinstance(sample, SampleSpec) and sample.where is not None):
                raise _left(f"{name} is estimated over a named sample or a sample with a condition: the model reader takes two dates")
            if isinstance(sample, SampleSpec) and (sample.start or sample.end):
                model.lines.append((st.line, f"smpl {_eviews_period(sample.start) or '@first'} {_eviews_period(sample.end) or '@last'}"))
            else:
                model.lines.append((st.line, "smpl @all"))
            model.lines.append((st.line, f"equation {name}.ls {' '.join(fit.spec.split())}"))
            model.lines.append((st.line, f"{key}.merge {name}"))
            model.merged[name] = st.line
            model.dirty = True
            self.building.append((key, st.line))
            return [], (f"collected: the model reader fits {name} again by least squares, on its sample, as line "
                        f"{self._fit_line.get(name, '?')} does")
        if word == "solveopt":
            note = self._solver(model, st.options)
            return [], "; ".join(x for x in ("the solver for the model's later solves (no step)", note) if x)
        if word == "solve":
            return self._solve(st, model)
        if word == "addassign":
            return self._addassign(st, model)
        if word == "addinit":
            return self._addinit(st, model)
        if word == "scenario":
            return self._scenario(st, model)
        if word in ("override", "exclude"):
            names = [w.lower() for w in st.rest.split()]
            if not names:
                raise _error(f"{word} names the variables: m.{word} x")
            if st.options:
                raise _left(f"{word}({st.options}) is not translated")
            scen = model.scenarios[model.active]
            if scen.name == "baseline" and word == "override":
                raise _left("an override in the baseline scenario is not translated: EViews ignores overrides there (verify)")
            kept = scen.overrides if word == "override" else scen.excludes
            for n in names:
                if n not in kept:
                    kept.append(n)
            return [], f"{', '.join(names)} {'take their values from the alias series' if word == 'override' else 'held at their data values'} in the scenario {scen.name} (no step)"
        if word == "makegroup":
            return self._makegroup(st, model)
        raise _left(f"{model.eviews}.{word} is not translated: append, merge, solve, solveopt, addassign, addinit, scenario, "
                    "override and exclude are")

    def _makegroup(self, st: Any, model: _Model) -> tuple[list[Op], str]:
        """``m.makegroup(a,n) g @endog``: a group (a ``Selection`` of the page) of the model's endogenous or exogenous
        variables."""
        from .data import _Py, _left

        named, flags = _options(st.options)
        unknown = [f for f in flags if f.lower() not in ("a", "n")] + sorted(named)
        if unknown:
            raise _left(f"makegroup({st.options}) is not translated: the options a and n are")
        words = st.rest.split()
        if len(words) != 2 or words[1].lower() not in ("@endog", "@exog"):
            raise _left("makegroup is translated for one group of all the endogenous or all the exogenous variables: "
                        "m.makegroup g @endog")
        page = self._page()
        ops = self._ensure_read(model)
        which = "endogenous" if words[1].lower() == "@endog" else "exogenous"
        var, note = self._name(words[0], f"group {words[0].lower()}")
        self.py[words[0].lower()] = _Py(var, "group", members=list(model.endogenous) if which == "endogenous" else [], page=page.var)
        ops.append(Step("pymetrix:dataset.series", page.var, var, f"{page.var}.series",
                        (Arg("patterns", Attr(model.solving, which), "star"),)))
        notes = [f"the model's {which} variables, by their names in the data (the baseline's series)"]
        if flags:
            notes.append(f"makegroup's options {', '.join(flags)}: the group holds the baseline's names (verify what a and n do)")
        return ops, "; ".join(x for x in (note, *notes) if x)

    def _model_scalars(self, model: _Model, text: str) -> None:
        for name in re.findall(r"(?<![\w.@!%])([A-Za-z_]\w*)(?!\w)(?!\s*\()", text):
            key = name.lower()
            v = self.py.get(key)
            if v is not None and v.kind == "scalar" and key not in model.scalars:
                model.scalars[key] = 0

    # -- reading ----------------------------------------------------------------------------------------------------

    def _text(self, model: _Model, check: bool) -> tuple[str, list[int]]:
        """The model's text, and the program's line of each of its lines. ``check``: the scalars with a value for the
        checking only (their values are given with params= when read)."""
        lines: list[str] = []
        where: list[int] = []
        first = model.lines[0][0]
        for key in model.scalars:
            v = self.py[key]
            value = v.value
            if check and not (isinstance(value, float) and math.isnan(value)):
                lines.append(f"scalar {key} = 1")
            else:
                lines.append(f"scalar {key}")
            where.append(self._declared.get(key, first))
        for line, text in model.lines:
            lines.append(text)
            where.append(line)
        return "\n".join(lines) + "\n", where

    def _ensure_read(self, model: _Model) -> list[Op]:
        """The step reading the model, when its text changed since it was last read; refused with the reader's problems."""
        from pymetrix.models import ModelSource

        from .data import _left

        if model.refused:
            raise _left(f"the model {model.eviews} is not read: {model.refused}")
        if not model.dirty:
            return []
        text, where = self._text(model, True)
        check = ModelSource(text, "eviews").check()
        errors = [d for d in check.diagnostics if d.severity == "error"]
        if errors:
            parts = []
            for d in errors:
                line = where[d.line - 1] if 0 < d.line <= len(where) else where[0]
                self.refusals.setdefault(line, []).append(f"the model reader: {d.message}")
                parts.append(f"line {line}: {d.message}")
            model.refused = "; ".join(parts)
            raise _left(f"the model {model.eviews} is not read: {model.refused}")
        model.endogenous = [s.name for s in check.symbols if s.role == "endogenous"]
        model.dirty = False
        model.read = True
        text, _ = self._text(model, False)
        args: list[Arg] = [Arg("source", text, "pos")]
        if model.scalars:
            args.append(Arg("params", {k: Var(self.py[k].name) for k in model.scalars}))
        if model.merged:
            args.append(Arg("data", RefTo(model.page.var)))
        from pymetrix.commands import RunCode

        if not model.solving:
            model.solving, _ = self._name(f"{model.var}_model", f"model {model.var}_model")
        return [Step("pymetrix:read_eviews", None, model.var, "read_eviews", tuple(args)),
                Cmd(RunCode(code=f"{model.solving} = {model.var}.model", source="script"), (model.var,))]

    def finish_models(self) -> list[tuple[int, list[Op], str]]:
        """At the end of the program: each model built but never used is read at its last building line; the reader's
        problems are kept for their lines (``refusals``)."""
        from .data import _Out

        out = []
        for key, model in self.models.items():
            if model.dirty and not model.refused:
                last = max(line for k, line in self.building if k == key)
                try:
                    ops = self._ensure_read(model)
                except _Out:
                    continue
                out.append((last, ops, "the model is read here, after its last line: nothing later uses it"))
        return out

    # -- solving ----------------------------------------------------------------------------------------------------

    def _solver(self, model: _Model, options: str | None) -> str:
        from .data import _left

        named, flags = _options(options)
        if flags:
            raise _left(f"solveopt's options {', '.join(flags)} are not translated")
        notes: list[str] = []
        kind = named.pop("o", "")
        args: list[Arg] = []
        if kind == "g":
            cls = "GaussSeidel"
        elif kind in ("n", ""):
            cls = "Newton"
        elif kind == "b":
            cls = "Newton"
            notes.append("o=b (Broyden): pymetrix's Newton; the solution is the same where both converge, the iterations differ")
        else:
            raise _left(f"solveopt(o={kind}) is not translated: g, n and b are")
        if "m" in named:
            args.append(Arg("max_iter", _value(named.pop("m"), "m")))
        if "c" in named:
            args.append(Arg("tol", _value(named.pop("c"), "c", float)))
            args.append(Arg("convergence", "step"))
            notes.append("c= is EViews's convergence on the relative change of the variables: pymetrix's convergence='step' (verify)")
        for key in ("s", "d"):
            if key in named:
                value = named.pop(key).lower()
                if (key, value) not in (("s", "d"), ("d", "d")):
                    raise _left(f"solveopt({key}={value}) is not translated: deterministic (s=d), dynamic (d=d) solutions are")
        unmapped = sorted(named)
        if unmapped:
            notes.append(f"not mapped: {', '.join(f'{k}={named[k]}' for k in unmapped)}; check what they do in EViews's "
                         "documentation of solveopt (g= rounding digits, z= the zero threshold, verify) before comparing solutions")
        model.solver = New(cls, tuple(args)) if (cls != "Newton" or args) else None
        model.solver_note = "; ".join(notes)
        return model.solver_note

    def _solve(self, st: Any, model: _Model) -> tuple[list[Op], str]:
        from pymetrix.commands import AddToDataset

        from .data import _left

        notes: list[str] = []
        if st.options:
            note = self._solver(model, st.options)
            if note:
                notes.append(note)
        elif model.solver_note:
            notes.append(model.solver_note)
        if st.rest.strip():
            raise _left("solve with arguments is not translated")
        page = model.page = model.page or self._page()
        if self._page() is not page:
            raise _left(f"the model {model.eviews} works on page {page.eviews}; select it first")
        span = self._span(page, "solve")
        ops = self._ensure_read(model)
        scen = model.scenarios[model.active]
        data: Any = RefTo(page.var)
        if scen.overrides:
            missing = [f"{v}{scen.alias}" for v in scen.overrides if page.series is not None and f"{v}{scen.alias}" not in page.series]
            if missing:
                raise _left(f"the override series {', '.join(missing)} are not in the page")
            name, _ = self._name(f"{page.var}{scen.alias}", f"data {page.var}{scen.alias}")
            ops.append(Step("pymetrix:dataset.with_series", page.var, name, f"{page.var}.with_series",
                            (Arg("exprs", {v: f"{v}{scen.alias}" for v in scen.overrides}, "varkw"),)))
            data = RefTo(name)
            notes.append(f"the overridden series take the alias series' values ({', '.join(f'{v}{scen.alias}' for v in scen.overrides)}) "
                         "in every period the solution reads, as EViews (verify)")
        innovs = [m.group(1).lower() for _, text in model.lines for m in re.finditer(r"@innov\s+([A-Za-z_]\w*)", text, re.I)]
        zero = {f"{v}_innov": "0" for v in innovs if page.series is None or f"{v}_innov" not in page.series}
        if zero:
            base = data.name
            name, _ = self._name(f"{page.var}_solve", f"data {page.var}_solve")
            ops.append(Step("pymetrix:dataset.with_series", base, name, f"{base}.with_series", (Arg("exprs", zero, "varkw"),)))
            data = RefTo(name)
            notes.append(f"a deterministic solve: the innovations ({', '.join(zero)}) are zero")
        sargs: list[Arg] = []
        if model.factors:
            sargs.append(Arg("adjust", RefTo(model.factors)))
        if scen.excludes:
            sargs.append(Arg("fix", {v: None for v in scen.excludes}))
        if sargs or scen.name != "baseline":
            sargs.append(Arg("name", scen.name))
        args = [Arg("data", data, "pos"), Arg("span", span, "pos")]
        if sargs:
            args.append(Arg("scenario", New("Scenario", tuple(sargs))))
        if model.solver is not None:
            args.append(Arg("solver", model.solver))
        elif not model.solver_note:
            notes.append("pymetrix's Newton: EViews's default solver differs by version (verify); the solution is the same "
                         "where both converge")
        out, _ = self._name(f"{model.var}{scen.alias}", f"solution {model.var}{scen.alias}")
        ops.append(Step("pymetrix:model.solve", model.solving, out, f"{model.solving}.solve", tuple(args)))
        for v in model.endogenous:
            series = f"{v}{scen.alias}"
            ops.append(Cmd(AddToDataset(source=out, item=f"column:{v}", data=page.var, series_name=series, replace=True), (out,)))
            if page.series is not None:
                page.series.add(series)
        notes.append(f"solved over {span[0]} to {span[1]}; the endogenous series are added to the page as {scen.alias} "
                     "(outside the span they hold the data's values; verify what EViews keeps there)")
        if model.factors:
            notes.append("with the add factors of addinit")
        return ops, "; ".join(notes)

    def _equations_named(self, model: _Model, words: list[str]) -> str:
        from .data import _left

        if [w.lower() for w in words] in (["@all"], ["@stochastic"], ["@stoch"]):
            return "all"
        raise _left("add factors for some of the equations are not translated: @all is (Scenario(adjust={...}) takes the ones wanted)")

    def _addassign(self, st: Any, model: _Model) -> tuple[list[Op], str]:
        from .data import _left

        named, flags = _options(st.options)
        if named or flags:
            raise _left(f"addassign({st.options}) is not translated: EViews's default add factor (on the equation) is")
        self._equations_named(model, st.rest.split())
        if model.merged:
            raise _left("add factors on a model with merged estimated equations are not translated: the core reader adds them "
                        "to the model's text equations only")
        model.assigned = True
        return [], ("add factors for every equation: pymetrix keeps them as a Scenario's adjust=, set by addinit "
                    "(no series named _a are made; no step)")

    def _addinit(self, st: Any, model: _Model) -> tuple[list[Op], str]:
        from .data import _left

        named, flags = _options(st.options)
        if not model.assigned:
            raise _left("addinit before addassign: the model has no add factors")
        v = named.pop("v", "n").lower()
        if v != "n" or named or flags:
            raise _left(f"addinit({st.options}) is not translated: v=n (the add factors that make the equations hold at the "
                        "data) is")
        self._equations_named(model, st.rest.split())
        page = model.page = model.page or self._page()
        span = self._span(page, "addinit")
        ops = self._ensure_read(model)
        out, _ = self._name(f"{model.var}_addfactors", f"add factors {model.var}")
        ops.append(Step("pymetrix:model.add_factors", model.solving, out, f"{model.solving}.add_factors",
                        (Arg("data", RefTo(page.var), "pos"), Arg("span", span))))
        model.factors = out
        note = (f"the add factors that make each equation hold at the data over {span[0]} to {span[1]}; later solves use them "
                "(Scenario(adjust=...)) in every scenario (verify which scenarios use add factors in EViews)")
        if "v" not in (st.options or ""):
            note += "; v=n, EViews's default (verify)"
        return ops, note

    def _scenario(self, st: Any, model: _Model) -> tuple[list[Op], str]:
        from .data import _error, _left

        name_m = re.fullmatch(r'\s*"([^"]+)"\s*', st.rest)
        if name_m is None:
            raise _error('scenario names the scenario in quotes: m.scenario "s1"')
        name = name_m.group(1)
        parts = _commas(st.options)
        create = False
        alias = None
        for p in parts:
            low = p.lower()
            if low == "n":
                create = True
            elif low.startswith("a="):
                alias = p[2:].strip()
                if not alias.startswith("_"):
                    alias = "_" + alias
            else:
                raise _left(f"scenario({p}) is not translated: n (a new scenario) and a= (its alias) are")
        key = name.lower()
        if key in ("baseline",) and create:
            raise _left("a new scenario named Baseline is not translated")
        if create:
            if key in model.scenarios:
                raise _error(f"the scenario {name} exists already")
            n = len(model.scenarios)
            model.scenarios[key] = _Scenario(name, alias or f"_{n}")
            note = f"a new scenario {name}, alias {model.scenarios[key].alias}" + ("" if alias else " (EViews's default alias, verify)")
        else:
            if key in ("baseline", "actuals"):
                key = "baseline"
            if key not in model.scenarios:
                raise _error(f"no scenario {name}: m.scenario(n) \"{name}\" makes one")
            if alias:
                model.scenarios[key].alias = alias
            note = f"the scenario {name} is used by later solves"
        model.active = key
        if key == "baseline" and name.lower() == "actuals":
            raise _left("solving into the actual series (the Actuals scenario) is not translated: the page's series would be "
                        "replaced; solve a scenario and use its alias series")
        return [], note + " (no step)"


def _eviews_period(period: str | None) -> str | None:
    if period is None:
        return None
    m = re.fullmatch(r"(\d{4})Q(\d)", period)
    if m:
        return f"{m.group(1)}q{m.group(2)}"
    m = re.fullmatch(r"(\d{4})-(\d{2})", period)
    if m:
        return f"{m.group(1)}m{int(m.group(2))}"
    return period

