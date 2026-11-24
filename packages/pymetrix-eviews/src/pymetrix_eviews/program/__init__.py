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
Control flow (phase 3) is left out with that reason, and so is what has no pymetrix equivalent (graphs, tables,
display): every line has a row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .data import Outcome, Translator, is_closer, is_opener
from .ir import Cmd, Code, step_code
from .left import PHASE3, PHASE3_WORDS
from .parse import Program, parse_program
from .report import ERROR, LEFT, NOTE, STATUSES, TRANSLATED, ProgramTranslation, ReportRow, TranslationReport

__all__ = ["STATUSES", "ProgramTranslation", "ReportRow", "TranslationReport", "parse_program", "translate_program"]


def _source(source: Any) -> tuple[str, str]:
    """The program's text and its file name ('' for text given directly)."""
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and source.strip().lower().endswith(".prg")):
        path = Path(source)
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1252")
        return text, path.name
    return str(source), ""


def translate_program(source: Any, bundle: Any = None, *, read_options: dict[str, Any] | None = None) -> ProgramTranslation:
    """An EViews program translated: ``.script`` (readable Python), ``.report`` (a row per line: translated,
    translated with a note, left out or error, with the reason) and ``.run(session)`` (steps mode).

    ``source`` is the program's text, or the path of a ``.prg`` file. ``bundle`` is what its ``wfopen`` opens: the
    bundle the export program wrote (``.evbundle``), or a CSV or Excel file the user saved from the workfile in
    EViews; ``read_options`` are ``Dataset.read``'s options for that file (``time="obs", freq="Q"``). Without it,
    ``wfopen`` is an error saying how to make one; a program that starts with ``wfcreate`` needs none."""
    text, name = _source(source)
    program = parse_program(text)
    translator = Translator(None if bundle is None else str(bundle), read_options)
    outcomes = _translate(program, translator)
    _finish(translator, outcomes)
    rows = _rows(program, outcomes, name)
    ops = [(o.statement.line, op) for o in outcomes for op in o.ops]
    script = _script(program, outcomes, name, translator.uses_bundle)
    pages = {p.eviews: p.var for p in translator.pages.values()}
    return ProgramTranslation(script, TranslationReport(tuple(rows), name), name, ops, pages)


def _translate(program: Program, translator: Translator) -> list[Outcome]:
    out: list[Outcome] = []
    stack: list[tuple[str, int]] = []
    for st in program.statements:
        opener, closer = is_opener(st), is_closer(st)
        if stack or opener or closer:
            if opener and not stack:
                reason = f"{PHASE3_WORDS[opener]}: {PHASE3}"
            elif stack:
                word, line = stack[0]
                reason = f"inside {PHASE3_WORDS[word]} (line {line}): {PHASE3}"
            else:
                out.append(Outcome(st, ERROR, f"{st.word} without the start of its block"))
                continue
            translator.forget()
            translator.inside_block(st)
            if opener:
                stack.append((opener, st.line))
            elif closer and stack:
                stack.pop()
            out.append(Outcome(st, LEFT, reason))
            continue
        out.append(translator.statement(st))
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
        o = by_line[line]
        if model.refused and o.status in (TRANSLATED, NOTE):
            o.status, o.reason, o.ops = LEFT, f"the model {model.eviews} is not read: {model.refused}", []
    for line, problems in translator.refusals.items():
        o = by_line.get(line)
        if o is None or o.status == LEFT and o.reason.startswith("the model"):
            continue
        add(o, "; ".join(problems))
        if o.status == TRANSLATED:
            o.status = NOTE


def _python(outcome: Outcome) -> str:
    return "\n".join(op.code if isinstance(op, Code) else op.command.to_code() if isinstance(op, Cmd) else step_code(op)
                     for op in outcome.ops)


def _rows(program: Program, outcomes: list[Outcome], name: str) -> list[ReportRow]:
    by_line: dict[int, tuple[Outcome, bool]] = {}
    for o in outcomes:
        for line in range(o.statement.line, o.statement.last_line + 1):
            by_line.setdefault(line, (o, line == o.statement.line))
    rows = []
    for number, raw in enumerate(program.lines, start=1):
        if not raw.strip():
            continue
        found = by_line.get(number)
        if found is None:
            if number in program.comments:
                rows.append(ReportRow(number, TRANSLATED, raw, "# " + program.comments[number], "a comment", name))
            else:
                rows.append(ReportRow(number, TRANSLATED, raw, "", "", name))
            continue
        o, first = found
        if first:
            rows.append(ReportRow(number, o.status, raw, _python(o), o.reason, name))
        else:
            rows.append(ReportRow(number, o.status, raw, "", f"continues line {o.statement.line}", name))
    return rows


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


def _script(program: Program, outcomes: list[Outcome], name: str, uses_bundle: bool) -> str:
    starts = {o.statement.line: o for o in outcomes}
    covered = {line for o in outcomes for line in range(o.statement.line, o.statement.last_line + 1)}
    body: list[str] = []
    for number, raw in enumerate(program.lines, start=1):
        if number in starts:
            o = starts[number]
            if number in program.comments:
                body.append(f"# {program.comments[number]}")
            eviews = " ".join(o.statement.text.split())
            body.append(f"# {number}: {eviews}")
            if o.status in (TRANSLATED, NOTE):
                if o.reason:
                    body.append(f"#   note: {o.reason}")
                if o.ops:
                    body.append(_python(o))
            else:
                body.append(f"#   {o.status}: {o.reason}")
        elif number in covered:
            continue
        elif number in program.comments:
            body.append(f"# {program.comments[number]}")
        elif not raw.strip():
            if body and body[-1] != "":
                body.append("")
    text = "\n".join(body).strip("\n")
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
    return "\n".join(head) + "\n\n" + text + "\n"

