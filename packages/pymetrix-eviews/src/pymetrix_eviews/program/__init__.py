"""EViews programs (``.prg``) translated into pymetrix: a readable script, a report with a row per line, and the same
work as recorded steps (design note, docs/design/eviews-plugin.md, section 2).

    from pymetrix_eviews import translate_program

    t = translate_program("C:/work/macro.prg", bundle="C:/work/macro-export/macro.evbundle")
    print(t.script)              # Python that uses pymetrix's public API, each EViews line above what it became
    t.report.summary()           # a Table: line, status, EViews, Python, reason
    out = t.run(session)         # steps mode: each data statement a recorded step in the session

Phase 1 is data work: workfiles and pages, samples, series, formulas, groups, labels, renaming and deleting, scalars,
strings and matrices. Phase 2 is estimation, tests and models: equations by the common methods, VARs and VECs,
systems, tests, forecasts, the values read from results, and models collected, read by the core reader and solved.
Phase 3 is control flow: ``for`` loops (``step``, lists of strings), ``while`` loops, ``if`` blocks decided as the
program runs, subroutines (typed arguments, ``local``) and ``call``, included programs, ``exitloop``, ``return``,
``stop``, the program's arguments (``%0``, ``%1``) and names made from its variables (``{%s}``); the script keeps
them as Python blocks and functions, and steps mode runs them in Python, each statement inside a recorded step.
What has no pymetrix equivalent (graphs, tables, display) is left out with the reason: every line has a row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .control import WHILE_CAP, Flow, build, functions_of, ops_of
from .data import Outcome, Translator, _Page, _Py, _UNKNOWN
from .ir import Code, quote
from .parse import Program, parse_program
from .report import LEFT, NOTE, STATUSES, TRANSLATED, ProgramTranslation, ReportRow, TranslationReport
from .script import render

__all__ = ["STATUSES", "WHILE_CAP", "ProgramTranslation", "ReportRow", "TranslationReport", "parse_program", "translate_program"]


def _source(source: Any) -> tuple[str, str, Path | None]:
    """The program's text, its file name ('' for text given directly) and its folder (for included programs)."""
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and source.strip().lower().endswith(".prg")):
        path = Path(source)
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1252")
        return text, path.name, path.parent
    return str(source), "", None


def translate_program(source: Any, bundle: Any = None, *, read_options: dict[str, Any] | None = None,
                      args: Any = (), objects: Any = None, folder: Any = None) -> ProgramTranslation:
    """An EViews program translated: ``.script`` (readable Python), ``.report`` (a row per line: translated,
    translated with a note, left out or error, with the reason) and ``.run(session)`` (steps mode).

    ``source`` is the program's text, or the path of a ``.prg`` file. ``bundle`` is what its ``wfopen`` opens: the
    bundle the export program wrote (``.evbundle``), or a CSV or Excel file the user saved from the workfile in
    EViews; ``read_options`` are ``Dataset.read``'s options for that file (``time="obs", freq="Q"``). Without it,
    ``wfopen`` is an error saying how to make one; a program that starts with ``wfcreate`` needs none.

    ``args`` are the program's arguments (``%0``, ``%1``, ...: what follows the program's name in ``exec prog.prg
    us 2020q1``), as strings or one string of words; they are the script's parameters (``arg0 = "us"`` at its top).
    ``objects`` are objects the program uses but does not make, by their EViews names: in EViews an add-in, a
    database or another program makes them. A ``Model`` (or a model read, whose ``.model`` is one) is solved by the
    program's model lines; a ``Dataset`` is a page. The script expects them under those names, and steps mode puts
    them in the session. ``folder`` is where included programs are looked for (default: the program's own folder,
    or the current one for text)."""
    text, name, own_folder = _source(source)
    program = parse_program(text)
    translator = Translator(None if bundle is None else str(bundle), read_options)
    given = _given(translator, objects)
    top, arguments = _arguments(translator, args, text)
    nodes = build(program, "", Path(folder) if folder is not None else own_folder)
    flow = Flow(translator, nodes)
    outcomes = flow.walk(nodes)
    flow.finish()
    _finish(translator, [o for o in _flat(outcomes) if not o.file])
    rows = _rows(program, outcomes, name)
    ops = tuple((0, op, "") for op in top) + ops_of(outcomes)
    functions = functions_of(flow)
    script = _script(program, outcomes, functions, flow, name, translator.uses_bundle, top, given)
    pages = {p.eviews: p.var for p in translator.pages.values()}
    return ProgramTranslation(script, TranslationReport(tuple(rows), name), name, ops, pages, tuple(functions), dict(given),
                              arguments)


def _given(translator: Translator, objects: Any) -> dict[str, Any]:
    """The objects given to the translation, registered: models as models read, Datasets as pages."""
    import keyword

    from pymetrix import Dataset
    from pymetrix.errors import SpecificationError
    from pymetrix.models import Model

    from .models import _Model, _Scenario

    out: dict[str, Any] = {}
    for key, value in dict(objects or {}).items():
        var = str(key).lower()
        if not var.isidentifier() or keyword.iskeyword(var) or var in translator.public:
            raise SpecificationError(f"objects: {key!r} is not a name the script can use (a Python word or a pymetrix name)")
        translator.taken[var] = f"given {var}"
        model = value if isinstance(value, Model) else getattr(value, "model", None)
        if isinstance(model, Model):
            m = _Model(var, var, None, dirty=False, read=True, endogenous=list(model.endogenous), solving=var if model is value else "")
            if not m.solving:
                m.solving, _ = translator._name(f"{var}_model", f"model {var}_model")
                m.given_code = f"{m.solving} = {var}.model"
            m.scenarios["baseline"] = _Scenario("baseline", "_0")
            m.given = True
            translator.models[var] = m
            translator.types[var] = "model"
        elif isinstance(value, Dataset):
            t = value.time_axis
            freq = None
            if t is not None:
                code = str(t.freq).upper()
                freq = "Y" if code.startswith(("Y", "A")) else code[:1] if code[:1] in ("Q", "M", "W", "D") else None
            periods = value.frame.index.get_level_values(t.name) if t is not None else None
            page = _Page(var, var, freq, set(value.names), first=str(periods.min()) if periods is not None and len(periods) else None,
                         last=str(periods.max()) if periods is not None and len(periods) else None)
            translator.pages[var] = page
            translator.current = translator.current or page
        else:
            raise SpecificationError(f"objects: {key!r} is a {type(value).__name__}: models and Datasets are taken")
        out[var] = value
    return out


def _arguments(translator: Translator, args: Any, text: str) -> tuple[list[Any], dict[str, str]]:
    """The program's arguments as the script's first lines (``arg0 = "us"``), and %0, %1 ... registered as strings
    whose values are known when the program runs (a parameter of the script)."""
    import re

    if isinstance(args, str):
        args = args.split()
    values = [str(a) for a in (args or ())]
    used = [int(m.group(1)) for m in re.finditer(r"%(\d+)\b", text)]
    count = max([len(values), *(u + 1 for u in used)]) if (values or used) else 0
    ops: list[Any] = []
    arguments: dict[str, str] = {}
    for i in range(count):
        var, _ = translator._name(f"arg{i}", f"argument %{i}")
        value = values[i] if i < len(values) else ""
        translator.py[f"%{i}"] = _Py(var, "string", _UNKNOWN)
        ops.append(Code(f"{var} = {quote(value)}  # the program's argument %{i}" + ("" if i < len(values) else ": not given"), (var,)))
        arguments[var] = value
    return ops, arguments


def _flat(outcomes: list[Any]) -> list[Any]:
    out = []
    for o in outcomes:
        if o is None:
            continue
        out.append(o)
        out += _flat(o.children)
        if o.else_outcome is not None:
            out.append(o.else_outcome)
        out += _flat(o.orelse)
        if o.closer is not None:
            out.append(o.closer)
    return out


def _finish(translator: Translator, outcomes: list[Outcome]) -> None:
    """Models read at their last line when nothing used them; the model reader's problems at their lines; the lines
    that build a model it refused, left out saying why."""
    by_line = {o.statement.line: o for o in outcomes}

    def add(o: Outcome, note: str) -> None:
        o.reason = "; ".join(x for x in (o.reason, note) if x)

    for line, ops, note in translator.finish_models():
        o = by_line[line]
        o.ops.extend(ops)
        add(o, note)
    for key, line in translator.building:
        model = translator.models[key]
        o = by_line.get(line)
        if o is not None and model.refused and o.status in (TRANSLATED, NOTE):
            o.status, o.reason, o.ops = LEFT, f"the model {model.eviews} is not read: {model.refused}", []
    for line, problems in translator.refusals.items():
        o = by_line.get(line)
        if o is None or o.status == LEFT and o.reason.startswith("the model"):
            continue
        add(o, "; ".join(problems))
        if o.status == TRANSLATED:
            o.status = NOTE


def _python(outcome: Outcome) -> str:
    from .script import op_code

    return "\n".join(op_code(op) for op in outcome.ops)


def _rows(program: Program, outcomes: list[Any], name: str, file: str = "") -> list[ReportRow]:
    """A row per line with text of the program (``file`` '' for the main one), and after an ``include`` line, the
    rows of the included program under its name."""
    by_line: dict[int, tuple[Outcome, bool]] = {}
    includes: dict[int, Outcome] = {}
    for o in _flat(outcomes):
        if o.file != file:
            continue
        if o.kind == "include" and o.status in (TRANSLATED, NOTE):
            includes[o.statement.line] = o
        for line in range(o.statement.line, o.statement.last_line + 1):
            by_line.setdefault(line, (o, line == o.statement.line))
    shown = file or name
    rows = []
    for number, raw in enumerate(program.lines, start=1):
        if not raw.strip():
            continue
        found = by_line.get(number)
        if found is None:
            if number in program.comments:
                rows.append(ReportRow(number, TRANSLATED, raw, "# " + program.comments[number], "a comment", shown))
            else:
                rows.append(ReportRow(number, TRANSLATED, raw, "", "", shown))
            continue
        o, first = found
        if first:
            rows.append(ReportRow(number, o.status, raw, _python(o), o.reason, shown))
        else:
            rows.append(ReportRow(number, o.status, raw, "", f"continues line {o.statement.line}", shown))
        inc = includes.get(number) if first else None
        if inc is not None and inc.children is not None:
            sub_program = _program_of(inc)
            if sub_program is not None:
                rows += _rows(sub_program, inc.children, name, _file_of(inc))
    return rows


def _program_of(o: Outcome) -> Program | None:
    return getattr(o, "program", None)


def _file_of(o: Outcome) -> str:
    return getattr(o, "included_file", "")


def _imports(text: str) -> list[str]:
    """pymetrix's names the script's code uses (``Equation(``, ``Dataset.read``, ``read_eviews(``), outside comments."""
    import re

    import pymetrix

    public = set(pymetrix.__all__)
    code = "\n".join(line for line in text.split("\n") if not line.lstrip().startswith("#"))
    code = re.sub(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'', '""', code)
    found = {m.group(1) for m in re.finditer(r"(?<![\w.])([A-Za-z_]\w*)\s*[.(]", code)}
    order = ["Dataset", "Matrix", "Sample"]
    return [n for n in order if n in found and n in public] + sorted(n for n in found if n in public and n not in order)


def _script(program: Program, outcomes: list[Any], functions: list[Any], flow: Flow, name: str, uses_bundle: bool,
            top: list[Any], given: dict[str, Any]) -> str:
    programs: dict[str, Program] = {"": program}
    for o in _flat(outcomes):
        if o.kind == "include" and getattr(o, "program", None) is not None:
            programs[o.included_file] = o.program

    pairs = []
    for _, d, _ in functions:
        sub = next((x for x in flow.subs.values() if x.var == d.name and x.outcome is not None), None)
        if sub is not None:
            pairs.append((sub.outcome, d))
    text = render(outcomes, pairs, programs, [op.code for op in top])
    used = _imports(text)
    title = f"{name}: an EViews program" if name else "An EViews program"
    head = [f'"""{title}, translated by pymetrix-eviews (translate_program).',
            "",
            "Each EViews line is a comment above what it became; lines left out say why. The translation report has a row",
            'for every line."""', ""]
    if used:
        head.append(f"from pymetrix import {', '.join(used)}")
    if uses_bundle:
        head.append("import pymetrix_eviews  # noqa: F401 - reads the workfile's bundle (.evbundle)")
    if given:
        kinds = ", ".join(f"{k} (a {type(v).__name__})" for k, v in given.items())
        head += ["", "# The objects the program uses but does not make (in EViews an add-in, a database or another program makes",
                 f"# them): give them before running this script, under these names: {kinds}"]
    return "\n".join(head) + "\n\n" + text + "\n"
