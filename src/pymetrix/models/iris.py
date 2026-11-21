"""Reading IRIS model files (``.model``): a model, its measurement equations and shocks, as a template ready to use.

    nk = read_iris("qpm.model", params={"b1": 0.7, "std_eps_y": 0.5, ...})
    nk.solve(); nk.system(); nk.estimate(data)      # a Template: everything a template model can do
    nk.labels["y_gap"]                              # the descriptions given in the file

Read: the declaration blocks ``!transition_variables``, ``!transition_shocks``, ``!measurement_variables``,
``!measurement_shocks``, ``!parameters`` (names, optionally ``name = value``), ``!exogenous_variables`` and
``!log_variables``; the equation blocks ``!transition_equations``, ``!measurement_equations`` and
``!reporting_equations``; quoted descriptions before names and equations; ``%`` and ``%{ ... %}`` comments; time subscripts ``x{-1}``, ``x{+1}``;
``diff(x)`` and ``difflog(x)``; steady-state versions after ``!!`` (the dynamic part is used); ``=#``;
``!substitutions`` (``$name$``); and the preprocessor's ``!for ?x = a, b !do ... !end`` loops. Parameter values
come from the file (numbers, or expressions in parameters given before) or from ``params=`` (which wins); shock
sizes from parameters named ``std_<shock>`` (1 if absent), as IRIS names them, and their correlations from
``corr_<shock>__<shock>`` (two transition shocks; 0 if absent), into the template's ``shock_corr``.

Parameter descriptions become the model's parameter metadata (``Param(name, label=...)``, ``model.spec(name)``), and
those of ``std_<shock>`` the label of the shock size ``sd_<shock>`` (of a measurement shock, ``sd_<series>`` of the series it
enters); ``labels`` keeps every description as well.

Reporting equations (``!reporting_equations``, ``name = expression;`` in model variables, their lags and parameters)
become the measurement map's reported series (``reported``): shown with smoothed histories, forecasts and each draw of a
stochastic forecast, never fitted to data. A description before one is its label; one may use those before it (which
are put in, as IRIS works them out in order). A file with reporting equations but no measurement equations gets a map
that only reports: it gives no state-space form (nothing is observed).

Measurement shocks become measurement errors of the series they enter. Log variables are written as ``exp(x)``
in every equation, so that ``x`` is the variable's log: the first-order solution is then in log deviations, as
IRIS's log-linearisation gives (steady-state guesses for them are taken in levels and logged).

Steady-state references ``&x`` are the steady-state value of ``x`` (its level, for a log variable): the steady state
is found with ``&x`` read as ``x``, which is what it equals there, and the dynamic model then holds it as a fixed
parameter ``ss_x`` (estimation leaves it alone; read the file again for other parameter values). ``!if``,
``!elseif``, ``!else`` and ``!end`` choose text by conditions on parameter values (from ``params=`` and the numbers
assigned in the file), as IRIS's preprocessor does; MATLAB's ``==``, ``~=``, ``&&``, ``||`` and ``~`` are understood.

``!dtrends`` (``obs += g*ttrend``) become the measurement map's deterministic trends, added to the observed series by
the filter, the likelihood and forecasts; ``ttrend`` counts periods from ``trend_base=`` (IRIS's ``baseYear``, e.g.
``"2020Q1"``), which must be given when it is used. Their parameters are held at the values read. Ignored with a
note: ``!links``, ``!autoswaps``, ``!preprocessor`` and ``!postprocessor``. The file is read as text: no IRIS
code is used or needed.

The QPM toolkit's dialect (the IRIS-style files of ``qpm_model_toolkit``) is read too:
section names in any case, with hyphens (``!transition-variables``), its aliases (``!variables-transition``,
``!dynamic-equations``, ``!assignments`` ...) and ``!variables:transition`` form; ``//`` and ``#`` comments;
equation names (``is_curve: y = ...``), kept as the equation's description; ``!modelname``, added to the
description; and ``!parameter-values`` (``name = value;`` for parameters declared in ``!parameters``). Its
``!exogenous-variables``, which it reads as shocks, keeps IRIS's meaning here. The model editor's check
(``ModelSource(text, "iris").check()``) finds comments, sections, declarations and equation names with the same
functions, so it takes what ``read_iris`` takes.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .._api import api
from .safe_eval import safe_eval
from .solvers import ModelError

__all__ = ["read_iris"]

_DECLARATIONS = {"!transition_variables", "!variables", "!transition_shocks", "!shocks", "!measurement_variables",
                 "!measurement_shocks", "!parameters", "!exogenous_variables", "!log_variables"}
_IGNORED = {"!links", "!autoswaps", "!autoswaps_simulate", "!autoswaps_steady", "!steady_autoswaps", "!preprocessor",
            "!postprocessor", "!comments"}
_UNSUPPORTED: dict[str, str] = {}
# the QPM toolkit's names for IRIS's sections (its !exogenous_variables, which it reads as shocks, keeps IRIS's meaning)
_ALIASES = {"model_name": "modelname", "endogenous_variables": "transition_variables",
            "variables_transition": "transition_variables", "variables_measurement": "measurement_variables",
            "shocks_transition": "transition_shocks", "shocks_measurement": "measurement_shocks",
            "parameters_values": "parameter_values", "assignments": "parameter_values",
            "dynamic_equations": "transition_equations", "equations_transition": "transition_equations",
            "equations_measurement": "measurement_equations"}
_KNOWN = _DECLARATIONS | _IGNORED | {"!modelname", "!parameter_values", "!transition_equations", "!equations",
                                     "!measurement_equations", "!reporting_equations", "!dtrends", "!substitutions", "!all_but"}


def _strip_comments(text: str, blank: bool = False) -> str:
    """``%{ ... %}`` blocks, then line comments (``%``, and the toolkit's ``//`` and ``#``, but not IRIS's ``=#``);
    a comment marker inside a quoted description is text. ``blank=True`` puts spaces in their place (newlines kept),
    so that every place in the text is where it was (the editor's check)."""
    if blank:
        text = re.sub(r"%\{.*?%\}", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)
    else:
        text = re.sub(r"%\{.*?%\}", " ", text, flags=re.S)
    return re.sub(r"\"[^\"\n]*\"|(?:%|//|(?<!=)#)[^\n]*",
                  lambda m: m.group(0) if m.group(0)[0] == '"' else (" " * len(m.group(0)) if blank else ""), text)


def _expand_for(text: str) -> str:
    """``!for ?x = a, b !do body !end`` (or ``!for a, b !do`` with ``?`` as the control name): the body once for
    each value, the control name replaced; nested loops from the innermost out."""
    loop = re.compile(r"!for\s+(?:(\?[A-Za-z_]\w*|[A-Za-z_]\w*)\s*=)?\s*(.*?)\s*!do(.*?)!end", re.S)
    while "!for" in text:
        first_end = text.find("!end")
        begin = text.rfind("!for", 0, first_end) if first_end >= 0 else -1
        m = loop.match(text, begin) if begin >= 0 else None
        if m is None:
            raise ModelError("a !for loop is not closed with !do ... !end")
        name, values, body = m.group(1) or "?", m.group(2), m.group(3)
        items = [v for v in re.split(r"[,\s]+", values.strip().strip("<>")) if v]
        pattern = re.compile(re.escape(name) + r"(?![A-Za-z0-9])" if name.startswith("?") else rf"\b{re.escape(name)}\b")
        expanded = " ".join(pattern.sub(item, body) for item in items)
        text = text[:m.start()] + expanded + text[m.end():]
    return text


def _condition(text: str, values: Mapping[str, float]) -> bool:
    """An !if condition, MATLAB style (``==``, ``~=``, ``&&``, ``||``, ``~``), evaluated with the parameter values."""
    expr = text.strip().replace("~=", "!=").replace("&&", " and ").replace("||", " or ")
    expr = re.sub(r"~(?!=)", " not ", expr).replace("^", "**")
    names = set(re.findall(r"\b[A-Za-z_]\w*\b", expr)) - {"and", "or", "not", "true", "false", "True", "False"}
    missing = sorted(n for n in names if n not in values)
    if missing:
        raise ModelError(f"cannot evaluate the !if condition {text.strip()!r}: give {', '.join(missing)} with params={{...}} "
                         "or as numbers in the file")
    scope = {**{k: float(v) for k, v in values.items()}, "true": True, "false": False, "True": True, "False": False}
    try:
        return bool(safe_eval(expr, scope, conditions=True))  # checked: values, comparisons and logic only
    except Exception as exc:
        raise ModelError(f"cannot evaluate the !if condition {text.strip()!r}: {exc}") from None


def _resolve_if(text: str, params: Mapping[str, float], values: Mapping[str, float] | None = None) -> str:
    """``!if cond ... !elseif cond ... !else ... !end``: the branch whose condition holds, outermost blocks first, so
    that conditions inside branches not taken are never evaluated. Conditions use ``params`` and the numbers the file
    assigns (``name = 0.5``)."""
    if values is None:
        found = re.finditer(r"\b([A-Za-z_]\w*)\s*=\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\b", text)
        values = {m.group(1): float(m.group(2)) for m in found}
        values.update({k: float(v) for k, v in params.items()})
    token = re.compile(r"!(if|elseif|else|end)\b")
    out, pos = [], 0
    while True:
        m = token.search(text, pos)
        while m is not None and m.group(1) != "if":
            raise ModelError(f"!{m.group(1)} without an !if before it")
        if m is None:
            out.append(text[pos:])
            return "".join(out)
        out.append(text[pos:m.start()])
        depth, branches, mark, kind = 1, [], m.end(), "if"
        scan = m.end()
        while depth:
            t = token.search(text, scan)
            if t is None:
                raise ModelError("an !if block is not closed with !end")
            word = t.group(1)
            if word == "if":
                depth += 1
            elif word == "end":
                depth -= 1
            if depth == 1 and word in ("elseif", "else") or depth == 0:
                branches.append((kind, text[mark:t.start()]))
                kind, mark = word, t.end()
            scan = t.end()
        chosen = ""
        for kind, body in branches:
            if kind == "else":
                chosen = body
                break
            condition, _, rest = body.partition("\n")
            if _condition(condition, values):
                chosen = rest
                break
        out.append(_resolve_if(chosen, params, values))
        pos = scan


def _keyword(raw: str) -> str:
    """A section keyword in IRIS's spelling: case and hyphens ignored (``!Transition-Variables``), the toolkit's
    aliases and ``!variables:transition`` form mapped to IRIS's names."""
    word = re.sub(r"\s+", "", raw[1:]).lower().replace("-", "_").rstrip(":")
    head, _, tail = word.partition(":")
    for candidate in ((f"{head}_{tail}", f"{tail}_{head}") if tail else (word,)):
        candidate = _ALIASES.get(candidate, candidate)
        if f"!{candidate}" in _KNOWN:
            return f"!{candidate}"
    return f"!{word}"


_SECTION = re.compile(r"![A-Za-z][\w\-]*(?:[ \t]*:[ \t]*(?:transition|measurement|values)\b)?:?", re.I)
# a toolkit equation name, "is_curve: y = ..." (not ":=")
_EQUATION_NAME = re.compile(r"([A-Za-z_]\w*)\s*:(?!=)\s*")


def _sections(text: str) -> list[tuple[str, str, int, int, int]]:
    """The file's sections, in order: (keyword in IRIS's spelling, keyword as written, its start, the body's start and
    end), places in ``text``. Shared by ``read_iris`` and the editor's check (``ModelSource(text, "iris")``)."""
    marks = list(_SECTION.finditer(text))
    return [(_keyword(m.group(0).strip()), m.group(0).strip(), m.start(), m.end(),
             marks[k + 1].start() if k + 1 < len(marks) else len(text)) for k, m in enumerate(marks)]


def _blocks(text: str) -> list[tuple[str, str]]:
    """The file's sections: (keyword, body), in order."""
    return [(keyword, text[start:end]) for keyword, _, _, start, end in _sections(text)]


def _declared_at(body: str) -> list[tuple[str, str, str | None, int]]:
    """(name, label, value, the name's place in ``body``) for each name in a declaration block. A description goes
    before its name (IRIS), or straight after it on the same line with no comma between (the QPM toolkit,
    ``y "Output gap"``)."""
    out: list[tuple[str, str, str | None, int]] = []
    pending_label = ""
    last_end = -1
    for m in re.finditer(r"\"[^\"]*\"|'[^']*'|[A-Za-z_]\w*\s*=\s*[^,;\n\"']+|[A-Za-z_]\w*", body):
        tok = m.group(0)
        if tok[0] in "\"'":
            if out and not out[-1][1] and last_end >= 0 and re.fullmatch(r"[ \t]*", body[last_end:m.start()]):
                out[-1] = (out[-1][0], tok[1:-1], out[-1][2], out[-1][3])
            else:
                pending_label = tok[1:-1]
            last_end = -1
            continue
        last_end = m.end()
        name, eq, value = tok.partition("=")
        out.append((name.strip(), pending_label, value.strip() if eq else None, m.start()))
        pending_label = ""
    return out


def _declared(body: str) -> list[tuple[str, str, str | None]]:
    """(name, label, value) for each name in a declaration block: see ``_declared_at``."""
    return [(name, label, value) for name, label, value, _ in _declared_at(body)]


def _equations(body: str) -> list[tuple[str, str]]:
    """(label, equation) for each equation in a block; the dynamic part of ``dynamic !! steady``."""
    out = []
    for raw in body.split(";"):
        text = " ".join(raw.split())
        if not text:
            continue
        label = ""
        m = re.match(r"^(\"[^\"]*\"|'[^']*')\s*(.*)$", text)
        if m:
            label, text = m.group(1)[1:-1], m.group(2)
        m = _EQUATION_NAME.match(text)  # the toolkit's equation names, "is_curve: y = ..."
        if m:
            label, text = label or m.group(1), text[m.end():]
        text = text.split("!!")[0].strip().replace("=#", "=")
        if text:
            out.append((label, text))
    return out


def _to_pymetrix(expr: str, log_vars: set[str], steady_refs: str = "error") -> str:
    """IRIS expression syntax to pymetrix's: x{-1} -> x[-1], diff/difflog expanded, log variables as exp(x);
    ``&x`` as ``x`` (``steady_refs="variable"``, for the steady state) or as the parameter ``ss_x``."""
    if "&" in expr:
        if steady_refs == "error":
            raise ModelError(f"steady-state references (&x) are not read here: {expr!r}")
        expr = re.sub(r"&\s*([A-Za-z_]\w*)", (lambda m: m.group(1)) if steady_refs == "variable" else (lambda m: f"ss_{m.group(1)}"), expr)
    expr = re.sub(r"\{\s*([+-]?\s*\d+)\s*\}", lambda m: f"[{int(m.group(1).replace(' ', '')):+d}]" if int(m.group(1).replace(" ", "")) else "", expr)

    def expand(fn: str, text: str) -> str:
        pattern = re.compile(rf"\b{fn}\(\s*([A-Za-z_]\w*)((?:\[[+-]\d+\])?)\s*\)")

        def lagged(name: str, lag: str) -> str:
            k = int(lag[1:-1]) if lag else 0
            return f"{name}[{k - 1:+d}]" if k - 1 else name

        def repl(m: re.Match) -> str:
            name, lag = m.group(1), m.group(2)
            here = f"{name}{lag}"
            if fn == "diff":
                return f"({here} - {lagged(name, lag)})"
            return f"(log({here}) - log({lagged(name, lag)}))"

        return pattern.sub(repl, text)

    expr = expand("difflog", expand("diff", expr))
    for v in sorted(log_vars, key=len, reverse=True):
        expr = re.sub(rf"\b{re.escape(v)}\b((?:\[[+-]\d+\])?)", lambda m, v=v: f"exp({v}{m.group(1)})", expr)
    return expr


@api("pymetrix:read_iris")
def read_iris(source: str | Path, params: Mapping[str, float] | None = None, guess: Mapping[str, float] | None = None,
              name: str | None = None, trend_base: str | None = None) -> Any:
    """Read an IRIS model file (a path, or the file's text) into a ``Template``: see the module docstring."""
    from ..estimation import Equation
    from .dynare import _evaluate, _match, _single_current
    from ..metadata import Param
    from .measurement import MeasurementMap
    from .model import Model
    from .templates import Template

    path = Path(source) if not isinstance(source, str) or ("\n" not in source and "!" not in source) else None
    text = path.read_text(encoding="utf-8") if path is not None else str(source)
    text = _expand_for(_strip_comments(text))
    if "!if" in text:
        text = _resolve_if(text, params or {})
    declared: dict[str, list[tuple[str, str, str | None]]] = {}
    transition, measurement, reporting, ignored, dtrends = [], [], [], [], []
    substitutions: dict[str, str] = {}
    assigned: list[tuple[str, str]] = []
    title = ""
    log_all_but = False
    previous = ""
    for keyword, body in _blocks(text):
        if keyword == "!all_but":
            if previous != "!log_variables":
                raise ModelError("!all_but must follow !log_variables")
            log_all_but = True
            declared.setdefault("!log_variables", []).extend(_declared(body))
            continue
        previous = keyword
        if keyword in _UNSUPPORTED:
            raise ModelError(f"{_UNSUPPORTED[keyword]} are not read yet")
        if keyword in _IGNORED:
            ignored.append(keyword)
        elif keyword == "!modelname":
            title = " ".join(body.split()).strip("\"'")
        elif keyword == "!parameter_values":
            for _, item in _equations(body):
                key, sep, value = item.partition("=")
                if not sep:
                    raise ModelError(f"a parameter value must be name = value: {item!r}")
                assigned.append((key.strip(), value.strip()))
        elif keyword in _DECLARATIONS:
            declared.setdefault(keyword.replace("!variables", "!transition_variables").replace("!shocks", "!transition_shocks"), []).extend(_declared(body))
        elif keyword in ("!transition_equations", "!equations"):
            transition += _equations(body)
        elif keyword == "!measurement_equations":
            measurement += _equations(body)
        elif keyword == "!reporting_equations":
            reporting += _equations(body)
        elif keyword == "!dtrends":
            dtrends += _equations(body)
        elif keyword == "!substitutions":
            for _, item in _equations(body):
                key, sep, value = item.partition("=")
                if not sep:
                    raise ModelError(f"a substitution must be name = expression: {item!r}")
                substitutions[key.strip()] = value.strip()
        else:
            raise ModelError(f"unknown section {keyword}")

    def substitute(eq: str) -> str:  # $name$ -> (expression), substitutions within substitutions too
        for _ in range(len(substitutions) + 1):
            eq = re.sub(r"\$(\w+)\$", lambda m: f"({substitutions[m.group(1)]})" if m.group(1) in substitutions else m.group(0), eq)
        unknown = re.search(r"\$\w+\$", eq)
        if unknown:
            raise ModelError(f"no substitution for {unknown.group(0)}")
        return eq

    transition = [(label, substitute(eq)) for label, eq in transition]
    measurement = [(label, substitute(eq)) for label, eq in measurement]
    reporting = [(label, substitute(eq)) for label, eq in reporting]

    def names(block: str) -> list[str]:
        return [n for n, _, _ in declared.get(block, [])]

    endo, shocks = names("!transition_variables"), names("!transition_shocks")
    observables, errors = names("!measurement_variables"), names("!measurement_shocks")
    exogenous = names("!exogenous_variables")
    labels = {n: lab for block in declared.values() for n, lab, _ in block if lab}
    log_block = declared.get("!log_variables", [])
    if log_all_but:
        excluded = {n for n, _, _ in log_block}
        log_vars = {v for v in endo if v not in excluded}
    else:
        log_vars = set(names("!log_variables"))
    values: dict[str, float] = {}
    for n, _, v in declared.get("!parameters", []):
        if v is not None:  # a number or an expression in earlier parameters
            values[n] = _evaluate(v.strip(), values, n)
    undeclared = [n for n, _ in assigned if n not in names("!parameters")]
    if undeclared:
        raise ModelError(f"!parameter_values gives {undeclared}, which are not in !parameters")
    for n, v in assigned:
        values[n] = _evaluate(v, values, n)
    values.update({k: float(v) for k, v in (params or {}).items()})
    param_names = names("!parameters")
    if not transition:
        raise ModelError("no !transition_equations")
    if len(transition) != len(endo):
        raise ModelError(f"{len(transition)} transition equations for {len(endo)} transition variables")
    correlations = _correlations(param_names, shocks, errors, values)
    model_params = [p for p in param_names if not p.startswith("std_") and p not in correlations]
    missing = [p for p in model_params if p not in values]
    if missing:
        raise ModelError(f"no values for the parameters {missing}: give them in the file or with params={{...}}")

    refs = sorted({m for _, eq in transition + measurement + reporting for m in re.findall(r"&\s*([A-Za-z_]\w*)", eq)})
    unknown_refs = [r for r in refs if r not in endo]
    if unknown_refs:
        raise ModelError(f"steady-state references to {unknown_refs}, which are not transition variables")

    def described(name: str) -> str:
        return next((lab for n, lab, _ in declared.get("!parameters", []) if n == name and lab), "")

    specs = [Param(p, label=described(p)) for p in model_params if described(p)]
    specs += [Param(f"sd_{s}", label=described(f"std_{s}")) for s in shocks if described(f"std_{s}")]
    shock_corr = {pair: r for pair, r in correlations.values() if r != 0}
    specs += [Param(f"corr_{a}_{b}", label=described(n)) for n, ((a, b), _) in correlations.items() if described(n)]

    def build(steady_refs: str, extra: Mapping[str, float]) -> Any:
        names = [*model_params, *extra]
        converted = []
        for _, eq in transition:
            lhs, sep, rhs = _to_pymetrix(eq, log_vars, steady_refs).partition("=")
            if not sep:
                lhs, rhs = lhs, "0"
            used = [p for p in names if re.search(rf"\b{re.escape(p)}\b", lhs + " " + rhs)]
            converted.append((lhs.strip(), rhs.strip(), used))
        targets = _match(converted, endo)
        eqs = []
        for (lhs, rhs, used), target in zip(converted, targets):
            natural = _single_current(lhs, endo + shocks + exogenous)
            body = (lhs, rhs) if natural == target else (target, f"{target} + ({lhs}) - ({rhs})")
            text_eq = f"{body[0]} {'~' if used else '='} {body[1]}"
            eqs.append(Equation(text_eq, params=used) if used else Equation(text_eq))
        everything = {**values, **extra}
        used_names = {p for e in eqs for p in e.param_names}
        return Model(eqs, params={p: everything[p] for p in names if p in used_names} or None,
                     specs=[s for s in specs if s.name in used_names or s.name.startswith(("sd_", "corr_"))])

    start_guess = {k: (math.log(float(v)) if k in log_vars else float(v)) for k, v in (guess or {}).items()}
    for v in log_vars:
        start_guess.setdefault(v, 0.0)
    steady_values: dict[str, float] = {}
    if refs:  # the steady state, where &x is x; then &x as fixed parameters ss_x
        found = build("variable", {}).steady_state(guess=start_guess or None)
        steady_values = {f"ss_{r}": (math.exp(found[r]) if r in log_vars else float(found[r])) for r in refs}
    model = build("parameter", steady_values)
    if steady_values:
        model = Model(model.equations, params=model.parameters.fix(*[n for n in steady_values if n in model.parameters.names]),
                      specs=model.specs)
    shock_sd = {s: float(values.get(f"std_{s}", 1.0)) for s in shocks}

    trends: dict[str, str] = {}
    for _, item in dtrends:  # obs += expression in parameters and ttrend; several for one series add up
        m = re.fullmatch(r"([A-Za-z_]\w*)\s*\+=\s*(.+)", item.strip())
        if not m or m.group(1) not in observables:
            raise ModelError(f"!dtrends: write each as measurement_variable += expression, not {item!r}")
        term = _to_pymetrix(m.group(2).strip(), set())
        trends[m.group(1)] = f"{trends[m.group(1)]} + ({term})" if m.group(1) in trends else f"({term})"
    if trends and not measurement:
        raise ModelError("!dtrends needs !measurement_equations for the series it adds trends to")
    reported: list[str] = []
    report_labels: dict[str, str] = {}
    defined: dict[str, str] = {}  # reported name -> its expression, for the reporting equations after it
    taken = set(endo) | set(shocks) | set(exogenous) | set(observables) | set(errors) | set(param_names)
    for label, eq in reporting:
        lhs, sep, rhs = eq.partition("=")
        name_r = lhs.strip()
        if not sep or not re.fullmatch(r"[A-Za-z_]\w*", name_r):
            raise ModelError(f"a reporting equation is name = expression: {eq!r}")
        if name_r in taken or name_r in defined:
            raise ModelError(f"the reporting equation for {name_r} needs a name of its own ({name_r} is already "
                             f"{'reported' if name_r in defined else 'declared'})")
        rhs = rhs.strip()
        for prior, expr in defined.items():
            if re.search(rf"\b{re.escape(prior)}\s*\{{", rhs):
                raise ModelError(f"the reporting equation for {name_r} uses {prior} at another period: write it out")
            rhs = re.sub(rf"\b{re.escape(prior)}\b", f"({expr})", rhs)
        defined[name_r] = rhs
        reported.append(f"{name_r} = {_to_pymetrix(rhs, log_vars, 'parameter')}")
        if label:
            report_labels[name_r] = label
    measurement_map = None
    error_specs: dict[str, Any] = {}
    if measurement or reported:
        texts, sizes = [], {}
        for _, eq in measurement:
            lhs, sep, rhs = eq.partition("=")
            observable = lhs.strip()
            if not sep or observable not in observables:
                raise ModelError(f"a measurement equation must have a measurement variable on the left: {eq!r}")
            entering = [e for e in errors if re.search(rf"\b{re.escape(e)}\b", rhs)]
            for e in entering:  # a measurement shock is the error of the series it enters
                rhs = re.sub(rf"\b{re.escape(e)}\b", "0", rhs)
                sizes[observable] = float(values.get(f"std_{e}", 1.0))
                if described(f"std_{e}"):
                    error_specs[f"sd_{observable}"] = Param(f"sd_{observable}", label=described(f"std_{e}"))
            texts.append(f"{observable} = {_to_pymetrix(rhs.strip(), log_vars, 'parameter')}")
        known = {**values, **steady_values}
        searched = [*texts, *reported, *trends.values()]
        used_params = {p: known[p] for p in [*model_params, *steady_values] if any(re.search(rf"\b{re.escape(p)}\b", t) for t in searched)}
        measurement_map = MeasurementMap(texts, params=used_params, errors=sizes,
                                         labels={**{o: labels[o] for o in observables if o in labels}, **report_labels},
                                         trends=trends, trend_base=trend_base, reported=reported)
        strangers = sorted(set().union(*(r.rhs.variables() for r in measurement_map.reports)) - {*endo, *exogenous}
                           if measurement_map.reports else ())
        if strangers:
            raise ModelError(f"the reporting equations use {', '.join(strangers)}, which are not variables of the model")
        if error_specs:  # the measurement errors' descriptions, with the model's metadata
            model = Model(model.equations, params=model.params, specs=[*model.specs, *error_specs.values()])
    start = dict(start_guess)  # logs for log variables (level 1 by default)
    description = f"Read from an IRIS model file{f' ({path.name})' if path else ''}{f': {title}' if title else ''}"
    if ignored:
        description += f"; not used: {', '.join(sorted(set(ignored)))}"
    if log_vars:
        description += f"; log variables (the model's values are their logs): {', '.join(sorted(log_vars))}"
    return Template(name or (path.stem if path else "iris"), model, tuple(shocks), shock_sd, measurement_map, start, description,
                    labels=labels, shock_corr=shock_corr)


def _correlations(param_names: list[str], shocks: list[str], errors: list[str], values: Mapping[str, float]
                  ) -> dict[str, tuple[tuple[str, str], float]]:
    """IRIS's correlation parameters, ``corr_<a>__<b>`` for two transition shocks: ``{name: ((a, b), r)}``, the pair in
    the shocks' order (0 when not given a value); checked: -1 < r < 1 and a valid correlation matrix together.
    Correlations of measurement shocks are refused (not read)."""
    from .covariance import correlation_matrix

    out: dict[str, tuple[tuple[str, str], float]] = {}
    for n in param_names:
        parts = n[5:].split("__") if n.startswith("corr_") else []
        if len(parts) != 2 or not any(q in shocks or q in errors for q in parts):
            continue  # an ordinary parameter
        a, b = parts
        if a in errors or b in errors:
            raise ModelError(f"'{n}': correlations of measurement shocks are not read (only of !transition_shocks)")
        unknown = [q for q in parts if q not in shocks]
        if unknown:
            raise ModelError(f"'{n}' names {unknown[0]}, which is not a transition shock")
        if a == b:
            raise ModelError(f"'{n}': a correlation is between two different shocks (std_{a} for its size)")
        r = float(values.get(n, 0.0))
        if not -1 < r < 1:
            raise ModelError(f"'{n}' is {r:g}: a correlation is between -1 and 1")
        out[n] = ((a, b) if shocks.index(a) < shocks.index(b) else (b, a), r)
    try:
        correlation_matrix(shocks, {pair: r for pair, r in out.values()})
    except ModelError as exc:
        raise ModelError(f"{exc} ('{next(iter(out))}' and the others)") from None
    return out
