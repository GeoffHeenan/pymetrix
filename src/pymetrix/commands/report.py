"""The report dialog: workspace objects to an Excel, Word, HTML or Markdown report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def report_notes(rep: Any, session: Session, names: list[str]) -> Any:
    """Add the plugins' notes (extension API 1.6, the ``report_notes`` group) on the results ``names``: one section per
    note's heading, a line per result. A note that fails is left out, with a warning; it never stops the report."""
    import warnings

    from ..extension import ReportNote
    from ..plugins import load, registry

    load("report_notes")
    sections: dict[str, list[str]] = {}
    for entry in registry("report_notes").entries(available=True):
        spec = entry.obj
        if not isinstance(spec, ReportNote):
            continue
        for name in names:
            try:
                line = spec.note(session, name)
            except Exception as exc:  # noqa: BLE001 - a plugin's note must not stop the report
                warnings.warn(f"the report note {entry.qualified} failed on {name}: {exc}", stacklevel=2)
                continue
            if line:
                sections.setdefault(spec.heading, []).append(str(line))
    for heading, lines in sections.items():
        rep.heading(heading, 1)
        for line in lines:
            rep.text(line)
    return rep


@command(menu="File/Export report...")
@dataclass(frozen=True)
class ExportReport(Command):
    """Write results to a report: Excel (each table on its own sheet), Word, HTML or Markdown (in order).

    Each result adds its summary and its graph, under a heading with its name; Sections all adds every tab of its
    window. A template chooses the views instead: estimation_summary (summary, graph, diagnostics), forecast_round
    (fan charts, paths, judgement, accuracy) or technical_appendix (everything). The theme sets how figures look and
    the decimals in tables; the sample limits the dates drawn in figures. An appendix lists where the numbers come
    from: data vintages, samples, methods and the pymetrix version. Plugins' notes (Advanced) add a line about each
    result from a plugin, such as the lab book's count of the specifications tried. Word reports need python-docx
    (pip install pymetrix[report]).
    """

    topics: ClassVar[tuple[str, ...]] = ("reports",)

    objects: Annotated[str, Widget("object", source="Reportable", multiple=True, show_if=("use_outline", (False,)), help="results to include, in order")]
    path: Annotated[str, Widget("savefile", source="Excel (*.xlsx);;Word (*.docx);;Web page (*.html);;Markdown (*.md)",
                                help="the file to write: .xlsx, .docx, .html or .md")] = "report.xlsx"
    title: Annotated[str, Widget("text", help="the heading at the top of the report (and the Word document's title)")] = "Report"
    author: Annotated[str, Widget("text", help="the author shown under the title, with the date; empty = none")] = ""
    template: Annotated[str, Widget("named", source="report_templates", show_if=("use_outline", (False,)), empty="summary and graph",
                                    help="which views of each result; empty = summary and graph")] = ""
    theme: Annotated[str, Widget("named", advanced=True, source="themes", empty="the session's default",
                                 help="how figures look and the decimals in tables; empty = the session's default")] = ""
    start: Annotated[str, Widget("period", group="Dates in figures")] = ""
    end: Annotated[str, Widget("period", group="Dates in figures")] = ""
    all_sections: Annotated[bool, Widget("checkbox", advanced=True, show_if=("use_outline", (False,)),
                                         help="every tab of each result's window, not just its summary and graph")] = False
    provenance: Annotated[bool, Widget("checkbox", advanced=True, help="add the appendix of data vintages and methods")] = True
    plugin_notes: Annotated[bool, Widget("checkbox", advanced=True,
                                         help="add the plugins' notes on each result, such as the lab book's count of "
                                              "the specifications tried; they come from this session's record, so the "
                                              "recorded script does not write them again")] = False
    use_outline: Annotated[bool, Widget("checkbox", help="the report outline (the Report panel's items, headings and text) "
                                                         "instead of results")] = False
    date: Annotated[str, Widget("text", advanced=True, help="the date printed under the title; empty = today")] = ""
    figure_width: Annotated[float, Widget("number", min=2.0, max=20.0, step=0.5, advanced=True,
                                          help="the width figures are drawn at, in inches (their height keeps their shape)")] = 6.0
    outline: Annotated[str, Widget("hidden", help="the report outline, as JSON: taken from the Report panel when it opens this")] = ""

    @classmethod
    def from_outline(cls, session: Session, **fields: Any) -> ExportReport:
        """The dialog on the session's report outline (the Report panel's Export…)."""
        import json

        return cls(**{"objects": "", "path": "report.docx", **fields, "use_outline": True, "outline": json.dumps(session.report_outline)})

    def _items(self) -> list[dict[str, Any]]:
        import json

        return list(json.loads(self.outline)) if self.outline else []

    def _span(self) -> tuple[str | None, str | None] | None:
        return (self.start or None, self.end or None) if self.start or self.end else None

    def validate(self, session: Session) -> list[str]:
        if self.use_outline and not self.outline:
            return ["export the outline from the Report panel (its Export…), which gives this dialog the outline"]
        if self.use_outline:
            items = self._items()
            missing = sorted({i["source"] for i in items if i["kind"] == "item" and i["source"] not in session.objects})
            errors = [f"no object named {n!r} (it was added to the report, then removed or renamed)" for n in missing]
            if not self.path.lower().endswith((".xlsx", ".docx", ".html", ".htm", ".md")):
                errors.append("save the report as .docx, .html, .md or .xlsx")
            return errors + ([] if any(i["kind"] == "item" for i in items) else
                             ["the report outline is empty: use Add to report on a figure, a table or a result"])
        missing = [n for n in _names(self.objects) if n not in session.objects]
        errors = [f"no object named {n!r}" for n in missing]
        if not self.path.lower().endswith((".xlsx", ".docx", ".html", ".htm", ".md")):
            errors.append("save the report as .xlsx, .docx, .html or .md")
        if self.template and self.all_sections:
            errors.append("choose a template or all sections, not both")
        return errors + ([] if _names(self.objects) else ["choose what to put in the report"])

    def run(self, session: Session) -> Any:
        from ..report import Report

        if self.use_outline:  # the outline's items, headings and text, in its order
            rep = Report(self.title, author=self.author, provenance=self.provenance, theme=self.theme or None, span=self._span(), **self._extra())
            for item in self._items():
                if item["kind"] == "heading":
                    rep.heading(item["text"], item.get("level", 2))
                elif item["kind"] == "text":
                    rep.text(item["text"])
                else:
                    rep.add(session[item["source"]], caption=item.get("caption", ""), sections=[item["tab"]] if item.get("tab") else None)
            return self._cautioned(rep, session).save(self.path)
        if self.template:
            rep = Report.from_template(self.template, {n: session[n] for n in _names(self.objects)}, title=self.title, author=self.author,
                                       theme=self.theme or None, span=self._span(), date=self.date.strip() or None)
            rep.provenance = self.provenance
            if "figure_size" in self._extra():
                rep.figure_size = self._extra()["figure_size"]
            return self._cautioned(rep, session).save(self.path)
        rep = Report(self.title, author=self.author, provenance=self.provenance, theme=self.theme or None, span=self._span(), **self._extra())
        for name in _names(self.objects):
            rep.heading(name, 2)
            rep.add(session[name], sections="all" if self.all_sections else None)
        return self._cautioned(rep, session).save(self.path)

    def _extra(self) -> dict[str, Any]:
        """The date and figure size, when not the defaults."""
        out: dict[str, Any] = {"date": self.date.strip()} if self.date.strip() else {}
        if self.figure_width != 6.0:
            out["figure_size"] = (self.figure_width, None)
        return out

    def _cautioned(self, rep: Any, session: Session) -> Any:
        """Results that were out of date (or frozen while their inputs changed) are said so at the top; with
        ``plugin_notes``, the plugins' notes on the reported results (``ReportNote``) go in sections at the end."""
        for caution in session.current_cautions:
            rep.caution(caution)
        if not self.plugin_notes:
            return rep
        names = ([i["source"] for i in self._items() if i["kind"] == "item"] if self.use_outline else _names(self.objects))
        return report_notes(rep, session, list(dict.fromkeys(names)))

    def to_code(self) -> str:
        code = self._code()
        if self.plugin_notes:  # they come from the session's record, which a script does not have
            code = "# the plugins' notes in this report came from the session's record; this script does not add them\n" + code
        return code

    def _code(self) -> str:
        options = ([f"author={self.author!r}"] if self.author else []) + ([f"theme={self.theme!r}"] if self.theme else []) \
            + ([f"span={self._span()!r}"] if self._span() else [])
        extra = self._extra()
        if self.template and not self.use_outline:  # the outline wins: run ignores template and all_sections with it
            results = "{" + ", ".join(f"{n!r}: {n}" for n in _names(self.objects)) + "}"
            dated = [f"date={extra['date']!r}"] if "date" in extra else []
            lines = [f"report = Report.from_template({self.template!r}, {results}, title={self.title!r}{''.join(', ' + o for o in options + dated)})"]
            lines += [] if self.provenance else ["report.provenance = False"]
            lines += [f"report.figure_size = {extra['figure_size']!r}"] if "figure_size" in extra else []
            lines.append(f"report.save({self.path!r})")
            return "\n".join(lines)
        args = [repr(self.title)] + options + ([] if self.provenance else ["provenance=False"]) + [f"{k}={v!r}" for k, v in extra.items()]
        sections = ', sections="all"' if self.all_sections else ""
        lines = [f"report = Report({', '.join(args)})"]
        if self.use_outline:  # every item written out, so the report is written again without the app
            for item in self._items():
                if item["kind"] == "heading":
                    lines.append(f"report.heading({item['text']!r}, {item.get('level', 2)})")
                elif item["kind"] == "text":
                    lines.append(f"report.text({item['text']!r})")
                else:
                    caption = f", caption={item['caption']!r}" if item.get("caption") else ""
                    tab = f", sections=[{item['tab']!r}]" if item.get("tab") else ""
                    lines.append(f"report.add({item['source']}{caption}{tab})")
            lines.append(f"report.save({self.path!r})")
            return "\n".join(lines)
        lines += [f"report.heading({n!r}, 2).add({n}{sections})" for n in _names(self.objects)]
        lines.append(f"report.save({self.path!r})")
        return "\n".join(lines)

