"""Models written out as text: pymetrix's syntax, Dynare's (.mod), IRIS's (.model) or EViews's (a program, or a
model's text: ``form="text"``), from any model pymetrix built.

    t = export_model(nk, "dynare")    # a Translation: the text, what did not carry across, and the check
    t.text; t.notes; t.difference     # the largest gap between the two models' responses and steady states
    ModelSource(text, "iris").export("dynare")   # from text in one syntax to another
    algebraic_equations(nk)           # each equation in algebraic notation: yₜ = yₜ₊₁ − (1 / σ) (iₜ − πₜ₊₁) + eyₜ
    write_latex(nk)                   # the model as a LaTeX align block, each equation tagged with its name

The text is generated from the model, not from any text it was read from: it means the same (the check below), but
comments, layout, ``!for`` loops and model-local variables are not reproduced, and an equation the reader rearranged
is written as rearranged. Each translation is read back in and solved, and its responses to each shock and its steady
state compared with the original's (``difference``, ``checked``), so a translation that does not hold says so.

What each syntax gets: pymetrix's, every section; Dynare's, the variables (with ``long_name`` descriptions), shocks with
their sizes and correlations (``corr`` in the shocks block, and in ``estimated_params`` for ``corr_<a>_<b>`` specs), parameter values, ``initval`` guesses and the measurement equations as equations for the observed series,
their errors as shocks, the series in ``varobs``; IRIS's, the declaration blocks with descriptions, parameter values
in ``!parameters`` and shock sizes as ``std_<shock>``, measurement equations with their errors as measurement shocks,
and reported series as ``!reporting_equations``. The shocks' correlations are written in each: ``corr a, b = r`` in
pymetrix's ``shocks`` (with their estimation settings), Dynare's ``corr`` and IRIS's ``corr_<a>__<b>`` parameters. A
template's known-date regimes are written in pymetrix's ``regimes`` section; Dynare and IRIS have no place for them (a
note says so). EViews's: ``model m`` and an ``m.append`` line for each equation (``@identity`` for those without
coefficients, a shock that is the last term added to one equation as its ``@innov``), the coefficients ``c(i)`` and
scalars with their values, constant series for exogenous values and guesses; names EViews would read differently
(case, reserved words such as ``c``) renamed, and noted, with what EViews's model has no place for.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._api import api
from ..errors import SpecificationError
from ..output import Table
from .solvers import ModelError

__all__ = ["AlgebraicEquation", "Translation", "algebraic_equations", "export_model", "write_latex"]

_TARGETS = ("pymetrix", "dynare", "iris", "eviews")


@dataclass(frozen=True, eq=False)
class Translation:
    """A model written in a syntax: the text, notes on what did not carry across, and the check against the original."""

    text: str
    syntax: str
    notes: tuple[str, ...] = ()
    difference: float | None = None  # the largest gap in responses and steady states, written against original
    checked: bool = False  # read back in and solved; False when it could not be (the notes say why)

    def summary(self) -> Table:
        rows = [[n] for n in self.notes]
        check = (f"Read back and solved: responses and steady state within {self.difference:.2g} of the original"
                 if self.checked and self.difference is not None else "Not checked against the original (see the notes)")
        return Table(["note"], rows, title=f"Written in {self.syntax}'s syntax", notes=[check])

    def __repr__(self) -> str:
        state = f"checked, difference {self.difference:.2g}" if self.checked and self.difference is not None else "not checked"
        return f"<Translation to {self.syntax}: {len(self.text.splitlines())} lines, {len(self.notes)} notes, {state}>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


@dataclass
class _Pieces:
    """What a written model holds, whatever it was built from."""

    model: Any
    endogenous: list[str]
    shocks: list[str]
    exogenous: dict[str, float]  # deterministic exogenous variables and their steady-state values
    params: dict[str, float]
    shock_sd: dict[str, float]
    equations: list[tuple[str, str, Any, Any, str]]  # label, dependent, lhs Expr, rhs Expr, separator
    measurement: list[tuple[str, Any, float, str]] = field(default_factory=list)  # observable, Expr, error sd, label
    measurement_params: dict[str, float] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    guess: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    transforms: dict[str, str] = field(default_factory=dict)  # observable: its transform of raw data
    reported: list[tuple[str, Any, str]] = field(default_factory=list)  # name, Expr, label
    specs: dict[str, Any] = field(default_factory=dict)  # Param metadata of parameters, sd_<shock> and corr_<a>_<b>
    shock_corr: dict[tuple[str, str], float] = field(default_factory=dict)  # correlations of the shocks, in the shocks' order
    regimes: tuple[Any, ...] = ()  # known-date regimes (a template's)


def _writable(item: Any) -> None:
    if not hasattr(item, "text") or type(item).__name__ != "Equation":
        raise ModelError(f"{type(item).__name__} equations (estimated, or fitted systems) are not written yet: write the model "
                         "before estimating its equations, or with their coefficients as parameters")
    if not item.is_identity and not item.param_names:
        raise ModelError(f"{item.text!r} is in linear form, to be estimated: give its coefficients names (params=) to write it")


def _pieces(obj: Any) -> _Pieces:
    from ..expr.parser import parse_equation, parse_expression
    from ..metadata import Role
    from .dynare import DynareModel
    from .model import Model
    from .templates import Template

    notes: list[str] = []
    if isinstance(obj, Template):
        model, shocks, shock_sd = obj.model, list(obj.shocks), dict(obj.shock_sd)
        measurement, labels, guess = obj.measurement, dict(obj.labels), dict(obj.guess)
    elif isinstance(obj, DynareModel):
        model, shocks, shock_sd = obj.model, list(obj.exogenous), dict(obj.shock_sd)
        measurement, labels, guess = None, dict(obj.labels), {k: v for k, v in obj.guess.items()}
        if obj.estimated:
            notes.append("estimated_params (priors, bounds, initial values) are not written: the model is, at its calibration")
        if obj.varobs:
            notes.append(f"varobs ({', '.join(obj.varobs)}) is not written as measurement equations: the model's variables are observed as they are")
        if obj.commands:
            notes.append(f"commands are not written: {'; '.join(obj.commands[:4])}{' ...' if len(obj.commands) > 4 else ''}")
        if getattr(obj, "loglinear", False):
            notes.append("loglinear: written in the logs of the variables, as the model was read (exp(y) for each)")
    elif isinstance(obj, Model):
        model, measurement, labels, guess = obj, None, {}, {}
        shocks = [n for n, r in (obj.roles or {}).items() if r == Role.SHOCK] or list(obj.compile().shocks)
        shock_sd = {s: 1.0 for s in shocks}
        notes.append("a Model alone has no shock sizes: each written as 1")
    else:
        raise TypeError(f"export writes a template, a Dynare model or a Model, not {type(obj).__name__}")
    params = {p.name: float(p.value) for p in model.parameters if p.value is not None}
    exogenous = {x: 0.0 for x in model.exogenous if x not in shocks}
    for x in exogenous:
        exogenous[x] = float(guess.pop(x, 0.0))
    equations = []
    for item in model.equations:
        _writable(item)
        lhs, sep, rhs = parse_equation(item.text, item.param_names)
        equations.append((getattr(item, "name", None) or "", item.dependent, lhs, rhs, sep))
    rows = []
    mparams: dict[str, float] = {}
    if measurement is not None:
        mparams = {k: float(v) for k, v in measurement.params.items()}
        for text in measurement.equations:
            obs, _, expr = text.partition("=")
            obs = obs.strip()
            rows.append((obs, parse_expression(expr.strip(), list(mparams)), float(measurement.errors.get(obs, 0.0)),
                         dict(measurement.labels).get(obs, "")))
        if getattr(measurement, "trends", None):
            notes.append("the measurement's deterministic trends are not written")
    out = _Pieces(model, list(model.endogenous), shocks, exogenous, params, shock_sd, equations, rows, mparams, labels, guess, notes)
    out.specs = {q.name: q.spec for q in model.compile().parameters if q.spec is not None and q.name in params}
    from .covariance import corr_pairs, normalise

    out.specs.update({s.name: s for s in getattr(model, "specs", ())
                      if (s.name.startswith("sd_") and s.name[3:] in shocks) or s.name in corr_pairs(shocks)})
    out.shock_corr = {k: v for k, v in normalise(shocks, getattr(obj, "shock_corr", None)).items() if v != 0}
    out.regimes = tuple(getattr(obj, "regimes", ()) or ()) if isinstance(obj, Template) else ()
    if measurement is not None:
        out.transforms = dict(getattr(measurement, "transforms", {}) or {})
        out.reported = [(r.observable, r.rhs, dict(measurement.labels).get(r.observable, "")) for r in getattr(measurement, "reports", ())]
    return out


_SHARED_FUNCTIONS = frozenset({"normcdf", "normpdf", "erf"})  # Dynare's (and IRIS's) among pymetrix's distribution functions


def _expand(expr: Any) -> Any:
    """d, dlog and pch written out with lags; comparisons and logical operators refused (no other syntax has them)."""
    from ..expr.functions import FUNCTIONS
    from ..expr.nodes import BinOp, Call, Name, Num

    def shifted(e: Any, k: int) -> Any:
        return e.map(lambda n: Name(n.id, n.lag + k) if isinstance(n, Name) else None)

    def rewrite(n: Any) -> Any:
        if isinstance(n, Call) and n.fn in ("d", "dlog", "pch") and len(n.args) == 1:
            inner = _expand(n.args[0])
            if n.fn == "d":
                return BinOp("-", inner, shifted(inner, -1), True)
            if n.fn == "dlog":
                return BinOp("-", Call("log", (inner,)), Call("log", (shifted(inner, -1),)), True)
            return BinOp("*", Num(100.0), BinOp("-", BinOp("/", inner, shifted(inner, -1)), Num(1.0), True))
        if isinstance(n, BinOp) and n.op not in ("+", "-", "*", "/", "^"):
            raise ModelError(f"the operator {n.op!r} has no equivalent in Dynare or IRIS")
        if isinstance(n, BinOp) and n.source:
            # the parser has already written d(x), dlog(x) and pch(x) out; the source only keeps them printing as typed
            return BinOp(n.op, _expand(n.left), _expand(n.right), n.grouped, "")
        if isinstance(n, Call) and n.fn in ("step", "not"):
            raise ModelError(f"{n.fn}() has no equivalent in Dynare or IRIS")
        if isinstance(n, Call) and n.fn in ("floor", "ceil", "round", "isna", "fillna", "iif", "na"):
            what = "the missing value na" if n.fn == "na" else f"{n.fn}()"
            raise ModelError(f"{what} has no equivalent in Dynare's model block (and is not written for IRIS either): "
                             "write it out in other terms to export; log10() is written")
        if isinstance(n, Call) and n.fn in FUNCTIONS and n.fn not in _SHARED_FUNCTIONS:
            raise ModelError(f"{n.fn}() has no equivalent in Dynare or IRIS, which have only normcdf, normpdf and erf "
                             "among the distribution and moving-window functions: write it out in other terms to export")
        return None

    return expr.map(rewrite)


def _text(expr: Any, syntax: str) -> str:
    out = str(_expand(expr) if syntax != "pymetrix" else expr)
    if syntax == "dynare":
        return re.sub(r"\b([A-Za-z_]\w*)\[([+-]\d+)\]", r"\1(\2)", out)
    if syntax == "iris":
        return re.sub(r"\b([A-Za-z_]\w*)\[([+-]\d+)\]", r"\1{\2}", out)
    return out


def _number(v: float) -> str:
    return repr(float(v)) if not float(v).is_integer() else str(int(v)) if abs(v) < 1e15 else repr(float(v))


def _quote(text: str, syntax: str) -> str:
    return f'"{text}"' if text and '"' not in text else ""


_DYNARE_SHAPES = {"Normal": "normal_pdf", "Beta": "beta_pdf", "Gamma": "gamma_pdf", "InvGamma1": "inv_gamma1_pdf",
                  "InvGamma": "inv_gamma2_pdf", "Uniform": "uniform_pdf", "GeneralisedBeta": "beta_pdf",
                  "ShiftedGamma": "gamma_pdf", "ShiftedInvGamma1": "inv_gamma1_pdf", "ShiftedInvGamma2": "inv_gamma2_pdf"}
# the generalised priors in pymetrix's syntax: the shape with its third (and fourth) number, beta(0.6, 0.1, -0.5, 1)
_SOURCE_SHAPES = {"InvGamma": "invgamma2", "GeneralisedBeta": "beta", "ShiftedGamma": "gamma", "ShiftedInvGamma1": "invgamma1",
                  "ShiftedInvGamma2": "invgamma2"}


def _options_text(spec: Any) -> str:
    """A spec's estimation settings in pymetrix's syntax: ``  bounds 0 1  prior beta(0.7, 0.1)  start 0.8  fixed``."""
    if spec is None:
        return ""
    parts = []
    if tuple(spec.bounds) != (-np.inf, np.inf):
        parts.append(f"bounds {_number(spec.bounds[0]) if np.isfinite(spec.bounds[0]) else '-inf'} "
                     f"{_number(spec.bounds[1]) if np.isfinite(spec.bounds[1]) else 'inf'}")
    if spec.prior is not None:
        name = _SOURCE_SHAPES.get(type(spec.prior).__name__, type(spec.prior).__name__.lower())
        parts.append(f"prior {name}({', '.join(_number(v) if np.isfinite(v) else repr(float(v)) for v in vars(spec.prior).values())})")
    if spec.start is not None:
        parts.append(f"start {_number(spec.start)}")
    if spec.fixed:
        parts.append("fixed")
    if spec.symbol:
        parts.append(f"symbol {spec.symbol}")
    return "".join(f"  {x}" for x in parts)


def _label(p: _Pieces, name: str) -> str:
    """A name's description: an IRIS ``std_<shock>`` its size's (``sd_<shock>``), ``corr_<a>__<b>`` its correlation's."""
    key = f"sd_{name[4:]}" if name.startswith("std_") else name.replace("__", "_", 1) if name.startswith("corr_") else name
    spec = p.specs.get(key)
    return (spec.label if spec is not None and spec.label else "") or p.labels.get(name, "")


def _regime_line(r: Any) -> str:
    """A regime as a line of pymetrix's ``regimes`` section."""
    head = (f"{r.start}..{r.end}" if r.end else r.start) + (f" {r.name}" if r.name else "") + ("" if r.expected else " surprise")
    parts = [", ".join(f"{k} = {_number(v)}" for k, v in r.params.items())] if r.params else []
    parts += [f"{label}: {text}" for label, text in r.equations.items()]
    return f"    {head}: {'; '.join(parts)}" + (f"  {_quote(r.description, 'pymetrix')}" if getattr(r, "description", "") else "")


def _write_pymetrix(p: _Pieces, origin: str) -> str:
    lines = [f"# written by pymetrix{origin}"]
    if p.params or p.measurement_params:
        lines.append("parameters")
        for name, value in {**p.params, **{k: v for k, v in p.measurement_params.items() if k not in p.params}}.items():
            lines.append(f"    {name} = {_number(value)}{_options_text(p.specs.get(name))}"
                         + (f"    {_quote(_label(p, name), 'pymetrix')}" if _label(p, name) else ""))
    if p.shocks:
        lines.append("shocks")
        for s in p.shocks:
            lines.append(f"    {s} = {_number(p.shock_sd.get(s, 1.0))}{_options_text(p.specs.get(f'sd_{s}'))}"
                         + (f"    {_quote(p.labels[s], 'pymetrix')}" if p.labels.get(s) else ""))
        for (a, b), r in p.shock_corr.items():
            label = _label(p, f"corr_{a}_{b}")
            lines.append(f"    corr {a}, {b} = {_number(r)}{_options_text(p.specs.get(f'corr_{a}_{b}'))}"
                         + (f"    {_quote(label, 'pymetrix')}" if label else ""))
    if p.exogenous:
        lines.append("exogenous")
        for x, v in p.exogenous.items():
            lines.append(f"    {x} = {_number(v)}" + (f"    {_quote(p.labels[x], 'pymetrix')}" if p.labels.get(x) else ""))
    extra = [v for v in p.endogenous if p.guess.get(v) or p.labels.get(v)]
    if extra:
        lines.append("variables")
        for v in extra:
            lines.append(f"    {v} = {_number(p.guess.get(v, 0.0))}" + (f"    {_quote(p.labels[v], 'pymetrix')}" if p.labels.get(v) else ""))
    lines.append("equations")
    for label, dependent, lhs, rhs, _ in p.equations:
        lines.append(f"    {label + ': ' if label and label != dependent else ''}{_text(lhs, 'pymetrix')} = {_text(rhs, 'pymetrix')}")
    if p.measurement:
        lines.append("measurement")
        for obs, expr, sd, label in p.measurement:
            lines.append(f"    {obs} = {_text(expr, 'pymetrix')}" + (f"  from {p.transforms[obs]}" if obs in p.transforms else "")
                         + (f"  error {_number(sd)}" if sd else "") + (f"  {_quote(label, 'pymetrix')}" if label else ""))
    if p.reported:
        lines.append("reported")
        for name, expr, label in p.reported:
            lines.append(f"    {name} = {_text(expr, 'pymetrix')}" + (f"  {_quote(label, 'pymetrix')}" if label else ""))
    if p.regimes:
        lines.append("regimes")
        lines += [_regime_line(r) for r in p.regimes]
    return "\n".join(lines) + "\n"


def _write_dynare(p: _Pieces, origin: str) -> str:
    errors = {obs: f"e_{obs}" for obs, _, sd, _ in p.measurement if sd}
    lines = [f"// written by pymetrix{origin}"]

    def declared(names: Sequence[str]) -> str:
        return " ".join(f"{n} (long_name='{p.labels[n]}')" if p.labels.get(n) and "'" not in p.labels[n] else n for n in names)

    lines.append(f"var {declared(p.endogenous + [obs for obs, _, _, _ in p.measurement])};")
    lines.append(f"varexo {declared(p.shocks + list(errors.values()) + list(p.exogenous))};")
    params = {**p.params, **{k: v for k, v in p.measurement_params.items() if k not in p.params}}
    if params:
        from ..notation import _given

        def decorated(name: str) -> str:
            spec, label = p.specs.get(name), _label(p, name)
            tex = f" ${_given(spec.symbol, 'latex')}$" if spec is not None and spec.symbol else ""
            return name + tex + (f" (long_name='{label}')" if label and "'" not in label else "")

        lines.append(f"parameters {' '.join(decorated(n) for n in params)};")
        lines += [f"{name} = {_number(value)};" for name, value in params.items()]
    lines.append("model;")
    for label, _, lhs, rhs, _ in p.equations:
        lines.append(f"  {'[name=' + repr(label) + '] ' if label else ''}{_text(lhs, 'dynare')} = {_text(rhs, 'dynare')};")
    for obs, expr, _, _ in p.measurement:
        lines.append(f"  {obs} = {_text(expr, 'dynare')}{' + ' + errors[obs] if obs in errors else ''};")
    lines.append("end;")
    initial = {**{v: g for v, g in p.guess.items() if v in p.endogenous}, **p.exogenous}
    if initial:
        lines.append("initval;")
        lines += [f"  {v} = {_number(g)};" for v, g in initial.items()]
        lines.append("end;")
    sizes = [(s, p.shock_sd.get(s, 1.0)) for s in p.shocks] + [(errors[obs], sd) for obs, _, sd, _ in p.measurement if sd]
    if sizes:
        lines.append("shocks;")
        lines += [f"  var {s}; stderr {_number(sd)};" for s, sd in sizes]
        lines += [f"  corr {a}, {b} = {_number(r)};" for (a, b), r in p.shock_corr.items()]
        lines.append("end;")
    if p.measurement:
        lines.append(f"varobs {' '.join(obs for obs, _, _, _ in p.measurement)};")
    estimated = _estimated_params(p)
    if estimated:
        lines += ["estimated_params;", *estimated, "end;"]
    return "\n".join(lines) + "\n"


def _estimated_params(p: _Pieces) -> list[str]:
    """Dynare's ``estimated_params`` lines for the specs with a prior or bounds (fixed ones are left out, as Dynare
    leaves out what is not estimated): ``name, init, low, high, shape_pdf, mean, sd;``."""
    from .covariance import corr_pairs

    pairs = corr_pairs(p.shocks)
    out = []
    for name, spec in p.specs.items():
        if spec.fixed or (spec.prior is None and tuple(spec.bounds) == (-np.inf, np.inf)):
            continue
        if name in pairs:  # corr e1, e2, ...: the correlation of two shocks
            target, value = f"corr {pairs[name][0]}, {pairs[name][1]}", p.shock_corr.get(pairs[name], 0.0)
        else:
            target = f"stderr {name[3:]}" if name.startswith("sd_") else name
            value = p.shock_sd.get(name[3:], 1.0) if name.startswith("sd_") else p.params.get(name, 0.0)
        init = _number(spec.start if spec.start is not None else value)
        low, high = ("" if not np.isfinite(b) else _number(b) for b in spec.bounds)
        fields = [target]
        if spec.prior is None:
            fields += [init, low or "-inf", high or "inf"]
        else:
            shape = _DYNARE_SHAPES.get(type(spec.prior).__name__)
            if shape is None:
                continue
            numbers = [_number(v) if np.isfinite(v) else "inf" for v in vars(spec.prior).values()]
            fields += ([init, low, high] if spec.start is not None or low or high else []) + [shape]
            fields += ["", "", *numbers] if shape == "uniform_pdf" else numbers
        out.append(f"  {', '.join(fields)};")
    return out


def _write_iris(p: _Pieces, origin: str) -> str:
    if p.exogenous:
        raise ModelError(f"deterministic exogenous variables ({', '.join(p.exogenous)}) have no place in an IRIS model's "
                         "transition equations: make them shocks or variables first")
    errors = {obs: f"e_{obs}" for obs, _, sd, _ in p.measurement if sd}
    lines = [f"% written by pymetrix{origin}"]

    def block(keyword: str, names: Sequence[str]) -> None:
        if names:
            lines.append(keyword)
            lines.extend(f"    {_quote(p.labels.get(n, ''), 'iris') + ' ' if p.labels.get(n) else ''}{n}" for n in names)

    block("!transition_variables", p.endogenous)
    block("!transition_shocks", p.shocks)
    params = {**p.params, **{k: v for k, v in p.measurement_params.items() if k not in p.params},
              **{f"std_{s}": p.shock_sd.get(s, 1.0) for s in p.shocks},
              **{f"std_{errors[obs]}": sd for obs, _, sd, _ in p.measurement if sd},
              **{f"corr_{a}__{b}": r for (a, b), r in p.shock_corr.items()}}  # IRIS's names for correlations
    if params:
        lines.append("!parameters")
        lines += [f"    {_quote(_label(p, name), 'iris') + ' ' if _label(p, name) else ''}{name} = {_number(value)}"
                  for name, value in params.items()]
    lines.append("!transition_equations")
    for label, _, lhs, rhs, _ in p.equations:
        lines.append(f"    {_quote(label, 'iris') + ' ' if label else ''}{_text(lhs, 'iris')} = {_text(rhs, 'iris')};")
    if p.measurement:
        block("!measurement_variables", [obs for obs, _, _, _ in p.measurement])
        if errors:
            lines.append("!measurement_shocks")
            lines += [f"    {e}" for e in errors.values()]
        lines.append("!measurement_equations")
        for obs, expr, _, _ in p.measurement:
            lines.append(f"    {obs} = {_text(expr, 'iris')}{' + ' + errors[obs] if obs in errors else ''};")
    if p.reported:
        lines.append("!reporting_equations")
        lines += [f"    {_quote(label, 'iris') + ' ' if label else ''}{name} = {_text(expr, 'iris')};" for name, expr, label in p.reported]
    return "\n".join(lines) + "\n"


def _write_eviews(p: _Pieces, origin: str, form: str = "program") -> tuple[str, dict[str, str], dict[str, float]]:
    """EViews syntax: a program (``model m``, ``m.append`` for each equation, the parameters as ``scalar`` and ``c(i)``
    assignments, exogenous values and guesses as constant series) or the model's text alone (``form="text"``: the
    values in a comment). A shock that is the last term added to one equation with coefficients is written as its
    ``@innov`` (read back as ``<variable>_innov``); other shocks are series. Also returns the names changed (lower case,
    reserved names; such shocks) and the parameter values by their EViews reader names."""
    from ..expr.nodes import BinOp, Name
    from .eviews import eviews_expression, eviews_names

    coefficient = re.compile(r"c_\d+")
    rename = eviews_names([*p.endogenous, *p.shocks, *p.exogenous, *(q for q in p.params if not coefficient.fullmatch(q))])
    params = set(p.params)
    innov: dict[int, tuple[str, float]] = {}
    for s in p.shocks:
        places = [i for i, (_, _, lhs, rhs, _) in enumerate(p.equations) if s in lhs.variables() | rhs.variables()]
        if len(places) != 1:
            continue
        i = places[0]
        _, dependent, _, rhs, _ = p.equations[i]
        uses = [q for q in params if q in p.equations[i][2].params() | rhs.params() or q in rhs.variables()]
        if uses and isinstance(rhs, BinOp) and rhs.op == "+" and not rhs.grouped and isinstance(rhs.right, Name) and rhs.right.id == s \
                and rhs.right.lag == 0 and s not in rhs.left.variables() and i not in innov:
            innov[i] = (s, p.shock_sd.get(s, 1.0))
            rename[s] = f"{rename.get(dependent, dependent)}_innov"
    lines = []
    for i, (_, dependent, lhs, rhs, _) in enumerate(p.equations):
        body = rhs.left if i in innov else rhs
        text = f"{eviews_expression(lhs, params, rename)} = {eviews_expression(body, params, rename)}"
        has_params = bool(lhs.params() | body.params()) or any(q in body.variables() | lhs.variables() for q in params)
        lines.append(text if has_params or i in innov else f"@identity {text}")
        if i in innov:
            lines.append(f"@innov {rename.get(dependent, dependent)} {_number(innov[i][1])}")
    values = {(q if coefficient.fullmatch(q) else rename.get(q, q)): v for q, v in p.params.items()}
    head = [f"' written by pymetrix{origin}"]
    if form == "text":
        given = ", ".join(f"{q.replace('_', '(', 1) + ')' if coefficient.fullmatch(q) else q} = {_number(v)}" for q, v in values.items())
        head += [f"' the coefficients' and scalars' values: {given}"] if given else []
        return "\n".join([*head, *lines]) + "\n", rename, values
    out = list(head)
    for q, v in values.items():
        out.append(f"{'c(' + q[2:] + ')' if coefficient.fullmatch(q) else 'scalar ' + q} = {_number(v)}")
    constants = {**{x: v for x, v in p.exogenous.items() if v}, **{x: v for x, v in p.guess.items() if v and x in p.endogenous}}
    out += [f"series {rename.get(x, x)} = {_number(v)}" for x, v in constants.items()]
    out.append("model m")
    out += [f"m.append {line}" for line in lines]
    return "\n".join(out) + "\n", rename, values


def _solution(obj: Any) -> Any:
    return obj.solve() if hasattr(obj, "solve") and not type(obj).__name__ == "Model" else obj.linearize().solve()


def _compare(original: Any, written: Any, shocks: Sequence[str], periods: int = 20, renamed: Mapping[str, str] | None = None) -> float:
    """The largest gap between the two models' steady states and responses to each shock (unit size), on the
    original's own variables and the shocks they share (``renamed``: names the written model gives them)."""
    renamed = renamed or {}
    a, b = _solution(original), _solution(written)
    names = [v for v in a.model.observed if renamed.get(v, v) in b.model.observed]
    common = [s for s in shocks if s in a.shocks and renamed.get(s, s) in b.shocks]
    gap = 0.0
    if common:
        ra, rb = a.irf(periods), b.irf(periods)
        for s in common:
            for v in names:
                x = ra.values[:, list(ra.responses).index(v), list(ra.shocks).index(s)]
                y = rb.values[:, list(rb.responses).index(renamed.get(v, v)), list(rb.shocks).index(renamed.get(s, s))]
                gap = max(gap, float(np.max(np.abs(x - y))))
    sa, sb = dict(a.model.steady.values), dict(b.model.steady.values)
    for v in names:
        gap = max(gap, abs(float(sa[v]) - float(sb[renamed.get(v, v)])))
    return gap


@api("pymetrix:export_model")
def export_model(obj: Any, syntax: str, origin: str = "", form: str = "program") -> Translation:
    """``obj`` (a template, a Dynare model or a Model) written in ``syntax`` ("pymetrix", "dynare", "iris" or
    "eviews"): see the module docstring. ``origin`` is said in the first comment line (where the model came from);
    ``form`` (EViews only) is "program" (a .prg: ``model m``, ``m.append`` lines and the values) or "text" (the model's
    text alone, its values in a comment)."""
    from .source import ModelSource

    if syntax not in _TARGETS:
        raise SpecificationError(f"syntax is one of {', '.join(_TARGETS)}, not {syntax!r}")
    p = _pieces(obj)
    notes = list(p.notes)
    if any("max" in str(r) or "min" in str(r) for _, _, _, r, _ in p.equations) and syntax != "pymetrix":
        notes.append("max and min are written as functions: pymetrix solves them as occasionally binding constraints (occbin); "
                     + ("Dynare does so only with occbin_constraints, written separately" if syntax == "dynare" else
                        "IRIS's own tools treat them differently"))
    if p.measurement and syntax == "dynare":
        notes.append("measurement equations are written as equations for the observed series (in var and varobs), their errors as shocks")
    if p.regimes and syntax != "pymetrix":
        notes.append(f"the regimes ({', '.join(r.label for r in p.regimes)}) are not written: "
                     + ("Dynare has no known-date regimes in a model file (its OccBin constraints are occasionally binding "
                        "ones, by conditions, not dates)" if syntax == "dynare" else "IRIS model files have no known-date regimes")
                     + ": give them to solve, estimate or forecast (regimes=...)")
    if p.guess and syntax == "iris":
        notes.append("steady-state guesses are not written: give them to read_iris(guess=...)")
    if syntax == "iris" and any(s.prior is not None or tuple(s.bounds) != (-np.inf, np.inf) or s.fixed for s in p.specs.values()):
        notes.append("priors, bounds and fixed parameters are not written: IRIS keeps them with the estimation, not in the model file")
    if syntax == "iris" and any(s.symbol for s in p.specs.values()):
        notes.append("parameter symbols are not written: IRIS model files have no place for them")
    if p.transforms and syntax != "pymetrix":
        notes.append(f"the transforms from raw data ({', '.join(f'{o} from {t}' for o, t in p.transforms.items())}) are not written: "
                     "give the data already transformed")
    if p.reported and syntax == "dynare":
        notes.append(f"the reported series ({', '.join(name for name, _, _ in p.reported)}) are not written: Dynare has no equivalent")
    if form not in ("program", "text"):
        raise SpecificationError("form is 'program' or 'text'")
    renamed: dict[str, str] = {}
    values: dict[str, float] = {}
    if syntax == "eviews":
        text, renamed, values = _write_eviews(p, f" from {origin}" if origin else "", form)
        notes += _eviews_notes(p, renamed, form)
    else:
        writer = {"pymetrix": _write_pymetrix, "dynare": _write_dynare, "iris": _write_iris}[syntax]
        text = writer(p, f" from {origin}" if origin else "")
    if syntax in ("dynare", "iris") and not p.shocks:
        notes.append("the model has no shocks (a simulation model, as EViews's): its exogenous series are written as "
                     + ("varexo, to be given paths (Dynare's perfect-foresight simul), not drawn by stoch_simul" if syntax == "dynare"
                        else "exogenous variables"))
    difference, checked = None, False
    try:
        if syntax == "eviews" and form == "text":
            from .eviews import read_eviews

            written = read_eviews(text, params=values, name="written")
        else:
            written = ModelSource(text, syntax, "written").build()
        difference, checked = _compare(obj, written, p.shocks, renamed=renamed), True
        if difference > 1e-8:
            notes.append(f"the written model differs from the original by {difference:.3g} (responses or steady state): check it")
    except (ModelError, SpecificationError, ValueError, np.linalg.LinAlgError) as exc:
        notes.append(f"not checked: the written text could not be built and solved again ({exc})")
    return Translation(text, syntax, tuple(notes), difference, checked)


def _eviews_notes(p: _Pieces, renamed: Mapping[str, str], form: str) -> list[str]:
    """What an EViews model's text or program does not carry."""
    notes = []
    as_series = [s for s in p.shocks if not renamed.get(s, "").endswith("_innov")]
    if as_series:
        notes.append(f"the shocks {', '.join(as_series)} are written as exogenous series (EViews models take their shocks as "
                     "series, and only an equation's own innovation, @innov, as a size): their sizes are not written")
    if p.shock_corr:
        notes.append("the shocks' correlations are not written: EViews's model text has no place for them")
    changed = {k: v for k, v in renamed.items() if not v.endswith("_innov")}
    if changed:
        notes.append("names changed for EViews (it ignores case; c, d, log and other names are reserved): "
                     + ", ".join(f"{k} as {v}" for k, v in changed.items()))
    if p.measurement:
        notes.append("the measurement equations are not written: an EViews model has no measurement map")
    if p.reported:
        notes.append(f"the reported series ({', '.join(name for name, _, _ in p.reported)}) are not written")
    if p.transforms:
        notes.append("the transforms from raw data are not written")
    if p.regimes:
        notes.append(f"the regimes ({', '.join(r.label for r in p.regimes)}) are not written: EViews's model text has no known-date regimes")
    if any(s.prior is not None or tuple(s.bounds) != (-np.inf, np.inf) or s.fixed or s.symbol for s in p.specs.values()):
        notes.append("priors, bounds, fixed parameters and symbols are not written: an EViews model has no place for them")
    if p.labels:
        notes.append("descriptions are not written: an EViews model's text has none")
    if form == "text" and (any(p.exogenous.values()) or any(v for k, v in p.guess.items() if k in p.endogenous)):
        notes.append("exogenous values and starting guesses are not written in the model's text (the program form writes them "
                     "as constant series)")
    return notes


def translate(source: Any, syntax: str) -> Translation:
    """A ``ModelSource`` in another syntax: its model built, then written (the text's comments and layout do not carry
    across: notes say so)."""
    built = source.build()
    out = export_model(built, syntax, origin=f"{source.syntax}'s syntax")
    lost = []
    if re.search(r"(^|\s)(#|//|%)", source.text) or source.syntax == "eviews" and re.search(r"'", source.text):
        lost.append("comments are not carried across: the text is generated from the model")
    if source.syntax == "iris" and re.search(r"!for|!substitutions|!if", source.text):
        lost.append("!for loops, !substitutions and !if are written out in full")
    if source.syntax == "iris" and "!log_variables" in source.text:
        lost.append("log variables (!log_variables) are written as exp(x), x being the variable's log, as the model was read")
    if source.syntax == "dynare" and re.search(r"^\s*#", source.text, re.M):
        lost.append("model-local variables (#) are substituted into the equations")
    return Translation(out.text, out.syntax, tuple(lost) + out.notes, out.difference, out.checked)


def _labels_of(obj: Any) -> Mapping[str, str]:
    return getattr(obj, "labels", {}) or {}


@dataclass(frozen=True)
class AlgebraicEquation:
    """One equation of a model in algebraic notation (``pymetrix.notation``)."""

    name: str  # the equation's name, or the variable it determines
    variable: str  # the variable it determines, or the observed series of a measurement equation
    text: str
    kind: str = "equation"  # or "measurement"


def algebraic_equations(obj: Any, style: str = "unicode") -> tuple[AlgebraicEquation, ...]:
    """The equations of ``obj`` (a template, a Dynare model or a Model with named parameters) in algebraic notation,
    ``style`` "unicode" or "latex": time as a subscript, Greek names as letters, ``d(x)`` as Δx; then the measurement
    equations, with ηₜ for a measurement error. As the model was read: an IRIS log variable is ``exp`` of its log."""
    from ..expr.nodes import Name
    from ..notation import expression

    if style not in ("unicode", "latex"):
        raise SpecificationError("style is 'unicode' or 'latex'")
    p = _pieces(obj)
    given = {q.name: q.spec.symbol for q in p.model.parameters if q.spec is not None and q.spec.symbol}  # Param(symbol=)
    out = [AlgebraicEquation(name or dep, dep, f"{expression(lhs, style, parameters=given)} = {expression(rhs, style, parameters=given)}")
           for name, dep, lhs, rhs, _ in p.equations]
    eta = "\\eta_t" if style == "latex" else "ηₜ"
    for obs, expr, error, _ in p.measurement:
        text = f"{expression(Name(obs, 0), style)} = {expression(expr, style, parameters=given)}" + (f" + {eta}" if error else "")
        out.append(AlgebraicEquation(obs, obs, text, "measurement"))
    return tuple(out)


def write_latex(obj: Any) -> str:
    """The model's equations as LaTeX ``align`` blocks (the measurement equations in a second one), each tagged with
    its name: for a paper or slides (amsmath)."""

    def tag(name: str) -> str:
        return "\\tag{" + re.sub(r"([_&%$#{}])", r"\\\1", name) + "}"

    def block(rows: Sequence[AlgebraicEquation]) -> list[str]:
        lines = ["\\begin{align}"]
        for i, eq in enumerate(rows):
            lhs, _, rhs = eq.text.partition(" = ")
            lines.append(f"  {lhs} &= {rhs} {tag(eq.name)}" + (" \\\\" if i < len(rows) - 1 else ""))
        return [*lines, "\\end{align}"]

    rows = algebraic_equations(obj, "latex")
    model = [r for r in rows if r.kind == "equation"]
    measured = [r for r in rows if r.kind == "measurement"]
    name = getattr(obj, "name", "") or "model"
    out = [f"% {name}: written by pymetrix", *block(model)]
    if measured:
        out += ["% measurement equations", *block(measured)]
    return "\n".join(out) + "\n"
