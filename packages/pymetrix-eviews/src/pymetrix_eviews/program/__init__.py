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

The parts that are not about EViews (the intermediate representation, the report, the script writer, steps mode and
the panel) are core's ``pymetrix.translation`` (extension API 1.11); this package reads EViews and says what each
statement becomes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pymetrix.translation import flatten, pymetrix_imports, render, report_rows

from .control import WHILE_CAP, Flow, build, functions_of, ops_of
from .data import Outcome, Translator, _Page, _Py, _UNKNOWN
from .ir import Code, quote
from .parse import Program, parse_program
from .report import LANGUAGE, LEFT, NOTE, STATUSES, TRANSLATED, ProgramTranslation, ReportRow, TranslationReport

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
    script, where = _script(program, outcomes, functions, flow, name, translator.uses_bundle, top, given)
    where = {(name if not f else f, line): n for (f, line), n in where.items()}
    pages = {p.eviews: p.var for p in translator.pages.values()}
    sources = {name: program.source}
    for o in _flat(outcomes):
        if o.kind == "include" and getattr(o, "program", None) is not None:
            sources.setdefault(o.included_file, o.program.source)
    return ProgramTranslation(script, TranslationReport(tuple(rows), name, LANGUAGE), name, ops, tuple(functions), dict(given),
                              arguments, sources, where, pages=pages)


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


_flat = flatten  # every outcome, a block's body, else branch and end included (core's)


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


def _rows(program: Program, outcomes: list[Any], name: str, file: str = "") -> list[ReportRow]:
    """A row per line with text of the program (``file`` '' for the main one), and after an ``include`` line, the
    rows of the included program under its name (core's ``report_rows``)."""
    return report_rows(program, outcomes, name, file, ReportRow)


_imports = pymetrix_imports  # pymetrix's names the script's code uses, outside comments and strings (core's)


def _script(program: Program, outcomes: list[Any], functions: list[Any], flow: Flow, name: str, uses_bundle: bool,
            top: list[Any], given: dict[str, Any]) -> tuple[str, dict[tuple[str, int], int]]:
    programs: dict[str, Program] = {"": program}
    for o in _flat(outcomes):
        if o.kind == "include" and getattr(o, "program", None) is not None:
            programs[o.included_file] = o.program

    pairs = []
    for _, d, _ in functions:
        sub = next((x for x in flow.subs.values() if x.var == d.name and x.outcome is not None), None)
        if sub is not None:
            pairs.append((sub.outcome, d))
    text, where = render(outcomes, pairs, programs, [op.code for op in top])
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
    if any("eviews_dates." in line for line in text.split("\n") if not line.lstrip().startswith("#")):
        head.append("import pymetrix_eviews.dates as eviews_dates  # EViews's date numbers (@dateval, @datestr, ...)")
    if given:
        kinds = ", ".join(f"{k} (a {type(v).__name__})" for k, v in given.items())
        head += ["", "# The objects the program uses but does not make (in EViews an add-in, a database or another program makes",
                 f"# them): give them before running this script, under these names: {kinds}"]
    first = len(head) + 2  # the body starts after the head and a blank line (lines counted from 1)
    return "\n".join(head) + "\n\n" + text + "\n", {k: first + i for k, i in where.items()}
