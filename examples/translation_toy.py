"""A translator for a toy language of two statements, written on ``pymetrix.translation`` (extension API 1.11).

The language has ``years first last`` (a new annual dataset) and ``let name = formula`` (a series made from a
formula), with ``--`` starting a comment. ``print`` is understood but left out, as display only; anything else is
an error. A real front end (the EViews plugin, an R one) has a parser and many more statements, but it fits
together the same way:

1. read the program into a ``Source``: its lines, its statements (``Statement``: text, first and last line) and
   its comments;
2. give each statement an ``Outcome``: a status, a reason and the ops it becomes (``Step``, a recorded call of
   pymetrix's API); a statement that does not translate raises ``left(reason)`` or ``error(reason)``;
3. let core make the rest: ``render`` writes the script, ``report_rows`` the report (a row per line with text), and
   a ``Translation`` subclass runs the same ops as recorded steps.

    t = translate_toy("years 2001 2005\\nlet t = obsnum()\\nlet y = 2 * t + 1  -- a line\\nprint y\\n")
    print(t.script)            # Python, each toy line a comment above what it became
    t.report.summary()         # a row per line, with its status and the reason
    t.run(session)             # the same work as recorded steps

Run it from the repository folder:  python examples/translation_toy.py
"""

from __future__ import annotations

import re

from pymetrix.translation import (NOTE, TRANSLATED, Arg, Outcome, Source, Statement, Step, Translation, TranslationReport,
                                  Untranslated, error, left, left_reason, pymetrix_imports, render, report_rows)


class ToyTranslation(Translation):
    """A toy program translated: its report's column is "Toy", its steps are recorded as ``toy:program`` and its
    script and report kept in the project under ``toy-translator``."""

    language = "Toy"
    recorded_by = "toy:program"
    attachments = "toy-translator"


def read_toy(text: str) -> Source:
    """The program read: a statement per line, ``--`` to the end of a line a comment."""
    lines = tuple(text.split("\n"))
    statements, comments = [], {}
    for number, raw in enumerate(lines, start=1):
        code, _, comment = raw.partition("--")
        if comment.strip():
            comments[number] = comment.strip()
        if code.strip():
            statements.append(Statement(code.strip(), number, number))
    return Source(text, lines, tuple(statements), comments)


def translate_statement(st: Statement) -> tuple[list[Step], str]:
    """What one statement becomes, and a note ('' for none); raises ``left`` or ``error`` when it does not."""
    word, _, rest = st.text.partition(" ")
    if word == "years":
        ends = rest.split()
        if len(ends) != 2 or not all(e.isdigit() for e in ends):
            raise error("years takes two years: years 2001 2005")
        return [Step("pymetrix:dataset.empty", None, "data", "Dataset.empty",
                     (Arg("freq", "Y", "pos"), Arg("start", ends[0], "pos"), Arg("end", ends[1], "pos")))], ""
    if word == "let":
        name, equals, formula = rest.partition("=")
        name = name.strip()
        if not equals or not re.fullmatch(r"[A-Za-z_]\w*", name) or not formula.strip():
            raise error("let takes a name and a formula: let y = 2 * x")
        python = name.lower()
        note = f"{name} is called {python}" if python != name else ""
        return [Step("pymetrix:dataset.with_series", "data", "data", "data.with_series",
                     (Arg("exprs", {python: formula.strip()}, "varkw"),))], note
    if word == "print":
        raise left(left_reason("display only", "print(data) in a script; the Data window in the app"))
    raise error(f"{word!r} is not a statement of the toy language: years and let are")


def translate_toy(text: str, name: str = "") -> ToyTranslation:
    """A toy program translated: ``.script``, ``.report`` and ``.run(session)``, from one representation."""
    program = read_toy(text)
    outcomes = []
    for st in program.statements:
        try:
            ops, note = translate_statement(st)
        except Untranslated as out:
            outcomes.append(Outcome(st, out.status, out.reason))
            continue
        outcomes.append(Outcome(st, NOTE if note else TRANSLATED, note, ops))
    body, where = render(outcomes, [], {"": program}, [])
    head = [f'"""{name or "A toy program"}, translated by the toy translator."""', "",
            f"from pymetrix import {', '.join(pymetrix_imports(body))}", ""]
    script = "\n".join([*head, body]) + "\n"
    first = len(head) + 1  # the script's line (from 1) of the body's first
    rows = report_rows(program, outcomes, name)
    return ToyTranslation(script, TranslationReport(tuple(rows), name, ToyTranslation.language), name,
                          ops=tuple((o.statement.line, op, "") for o in outcomes for op in o.ops),
                          sources={name: text},
                          script_lines={(f or name, line): first + i for (f, line), i in where.items()})


if __name__ == "__main__":
    t = translate_toy("years 2001 2005\nlet t = obsnum()  -- 1, 2, ...\nlet Y = 2 * t + 1\nprint y\n", "toy.txt")
    print(t.script)
    print(t.report.to_text())
    print(t.run()["session"]["data"].frame)
