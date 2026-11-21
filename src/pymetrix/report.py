"""Reports: tables, figures and text, written to Excel, Word, HTML or Markdown.

    rep = Report("Quarterly forecast, 2026Q3", author="Forecasting team", theme="policy")
    rep.heading("Consumption")
    rep.text("The consumption function is stable; residuals show no serial correlation.")
    rep.add(fit)                       # a result: its tables and graph, as the app shows them
    rep.add(round_.plot(data), caption="GDP growth, fan chart")
    rep.add(fit.test("breusch_godfrey", lags=4))
    rep.save("forecast.docx")          # or .xlsx, .html, .md

    rep = Report.from_template("estimation_summary", {"Consumption": fit, "Investment": inv}, title="Equations")
    report_templates()                 # estimation_summary, forecast_round, technical_appendix, policy_brief

Anything the app can show can go in a report: a ``Table``, a ``Figure``, a test result,
or any result object (estimates, VARs, forecasts, rounds, backtests...), which adds the
same tables and graph as its window. Each result's provenance (data vintages, sample,
method, package version) is listed in an appendix, so a report says where every number
came from. Excel puts each table on its own sheet, with a contents sheet; Word (needs
``python-docx``: ``pip install pymetrix[report]``), HTML and Markdown follow the report's order;
Markdown writes its figures as PNG files in a folder beside it.

A report is built from results' views, which are built from their named fields, never from
printed text. ``theme`` sets how figures look and the decimals in tables, ``span`` limits the
dates in figures. A template is a named list of which views to include for each result.
"""

from __future__ import annotations

import base64
import html
import io
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .errors import NotFoundError, SpecificationError
from .output import Table, _fmt

__all__ = ["Report", "ReportTemplate", "report_templates"]


@dataclass(frozen=True)
class ReportTemplate:
    """A named report layout: which views of each result to include, in order, and whether to add provenance."""

    name: str
    description: str
    sections: tuple[str, ...] | str  # view titles in order (those a result has), or "all"
    provenance: bool = True
    version: int = 1


_TEMPLATES = {t.name: t for t in (
    ReportTemplate("estimation_summary", "Estimates and their checks: the summary, the graph and the diagnostics of each result.",
                   ("Summary", "Graph", "Diagnostics", "Long run", "Bounds test")),
    ReportTemplate("forecast_round", "A forecast round: fan charts, the paths, judgement and accuracy.",
                   ("Graph", "Paths", "Judgement", "Accuracy", "Summary")),
    ReportTemplate("technical_appendix", "Everything: every view of every result, with provenance.", "all"),
    ReportTemplate("policy_brief", "Policy scenarios against a baseline: who is compared with what, how far each scenario "
                   "departs from the baseline, the loss of each (when one is set), the paths and the departures.",
                   ("Scenarios", "Impact", "Loss", "Graph", "Less the baseline")),
)}


def _all_templates() -> dict[str, ReportTemplate]:
    """The built-in templates and those plugins register (``register("report_templates", ReportTemplate(...))``)."""
    from .plugins import load, registry

    load("report_templates")
    out = dict(_TEMPLATES)
    for entry in registry("report_templates").entries(available=True):
        if isinstance(entry.obj, ReportTemplate) and entry.obj.name not in out:
            out[entry.obj.name] = entry.obj
    return out


def report_templates() -> list[str]:
    """The names of the report templates, the built-in ones and plugins'."""
    return list(_all_templates())


@dataclass
class _Block:
    kind: str  # heading, text, table, figure
    value: Any
    caption: str = ""
    level: int = 1


@dataclass
class Report:
    """A document of headings, text, tables and figures, saved as .xlsx, .docx, .html or .md."""

    title: str
    author: str = ""
    date: str = field(default_factory=lambda: date.today().isoformat())
    provenance: bool = True  # an appendix with each result's provenance
    theme: str | None = None  # figures' look and tables' decimals; None: the session's default
    span: tuple[Any, Any] | None = None  # the dates drawn in figures
    template: str = ""  # the template it was made from, if any
    figure_size: Any = (6.0, None)  # figures drawn at the size they appear (inches, mm or a preset; None: the height keeps the shape)
    blocks: list[_Block] = field(default_factory=list)
    _sources: list[tuple[str, Any]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        from .themes import get_theme

        get_theme(self.theme)  # an unknown theme fails now

    @classmethod
    def from_template(cls, template: str, results: Any, title: str = "", author: str = "", theme: str | None = None,
                      span: tuple[Any, Any] | None = None, date: str | None = None) -> Report:
        """A report laid out by a named template (``report_templates()``). ``results`` maps headings to results (or is
        a list of results); each gets a heading and the template's views that it has."""
        templates = _all_templates()
        if template not in templates:
            raise NotFoundError(f"no report template {template!r}; templates: {', '.join(templates)}")
        t = templates[template]
        items = list(results.items()) if isinstance(results, dict) else [(type(r).__name__, r) for r in results]
        if not items:
            raise SpecificationError("give the results to report")
        rep = cls(title or template.replace("_", " ").capitalize(), author=author, provenance=t.provenance, theme=theme, span=span,
                  template=template, **({"date": date} if date else {}))
        from .views import view_sections

        for heading, obj in items:
            if isinstance(obj, (Table, pd.DataFrame)) or type(obj).__name__ == "Figure":
                rep.heading(heading, 2).add(obj)
                continue
            have = [title for title, _ in view_sections(obj)]
            wanted = "all" if t.sections == "all" else [s for s in t.sections if s in have]
            if not wanted:
                raise SpecificationError(f"the {template} template has nothing to show for {heading!r}: it shows {', '.join(t.sections)}; "
                                 f"{heading!r} has {', '.join(have)}")
            rep.heading(heading, 2).add(obj, caption=heading, sections=wanted)
        return rep

    @property
    def digits(self) -> int:
        from .themes import get_theme

        return get_theme(self.theme).digits

    def heading(self, text: str, level: int = 1) -> Report:
        if level not in (1, 2, 3):
            raise SpecificationError("heading level must be 1, 2 or 3")
        self.blocks.append(_Block("heading", text, level=level))
        return self

    def caution(self, text: str) -> Report:
        """A warning at the top of the report (before any results), e.g. that a result in it is out of date."""
        at = sum(1 for b in self.blocks if b.kind == "text" and str(b.value).startswith("Caution:"))
        self.blocks.insert(at, _Block("text", f"Caution: {' '.join(text.split())}"))
        return self

    def text(self, text: str) -> Report:
        """A paragraph; blank lines separate paragraphs."""
        for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
            self.blocks.append(_Block("text", " ".join(para.split())))
        return self

    def add(self, obj: Any, caption: str = "", sections: Sequence[str] | str | None = None) -> Report:
        """Add a table, a figure, or a result. A result adds its summary and its graph; ``sections`` picks
        others by the titles of its window's tabs, in that order (e.g. ``["Summary", "Diagnostics"]``), or ``"all"``."""
        from .figures import Figure

        if isinstance(obj, Table):
            self.blocks.append(_Block("table", obj, caption or obj.title))
        elif isinstance(obj, Figure):
            self.blocks.append(_Block("figure", obj, caption or obj.title))
        elif isinstance(obj, pd.DataFrame):
            self.blocks.append(_Block("table", Table.from_frame(obj), caption))
        else:
            from .views import view_sections

            parts = view_sections(obj)
            if sections is None:  # the summary and the graph (one figure: a result whose first view is its graph adds no other)
                parts = parts[:1] + ([] if isinstance(parts[0][1], Figure) else [p for p in parts[1:] if isinstance(p[1], Figure)][:1])
            elif sections != "all":
                wanted = [sections] if isinstance(sections, str) else list(sections)
                missing = [x for x in wanted if x not in [t for t, _ in parts]]
                if missing:
                    raise NotFoundError(f"no section {missing}; sections: {[t for t, _ in parts]}")
                parts = [next(p for p in parts if p[0] == w) for w in wanted]  # in the order asked for
            for i, (title, value) in enumerate(parts):
                own = getattr(value, "title", "") or title
                label = (caption if i == 0 else f"{caption}: {title}") if caption else own
                self.blocks.append(_Block("figure" if isinstance(value, Figure) else "table", value, label))
            prov = getattr(obj, "provenance", None)
            if prov is not None:
                self._sources.append((caption or type(obj).__name__, prov))
        return self

    # -- writing ----------------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        writer = {".xlsx": self._excel, ".docx": self._word, ".html": self._html, ".htm": self._html, ".md": self._markdown}.get(path.suffix.lower())
        if writer is None:
            raise SpecificationError("save a report as .xlsx, .docx, .html or .md")
        writer(path)
        return path

    def _appendix(self) -> Table | None:
        if not self.provenance or not self._sources:
            return None
        from .views import sample_text

        rows = []
        for name, p in self._sources:
            vintages = ", ".join(f"{k}: {v.released}" for k, v in (p.data_vintages or {}).items() if v is not None) or "not recorded"
            sample = sample_text(p)  # a backtest's or rolling fit's: what its re-estimates ran over, and on what
            method = p.method.get("type", "") if isinstance(p.method, dict) else str(p.method)
            rows.append([name, method, sample, p.nobs, vintages, p.package_version])
        return Table(["result", "method", "sample", "observations", "data vintages (release dates)", "pymetrix"], rows,
                     title="Where the numbers come from")

    @staticmethod
    def _frame(t: Table) -> pd.DataFrame:
        return t._flat()

    def _png(self, fig: Any) -> bytes:
        """The figure drawn at ``figure_size``, so its text prints at its size in points."""
        if hasattr(fig, "with_options"):
            fig = fig.with_options(theme=fig.theme or self.theme, span=fig.span or self.span)
        if hasattr(fig, "png"):
            return fig.png(dpi=200, size=self.figure_size) if self.figure_size is not None else fig.png()
        return fig._repr_png_()

    def _width(self) -> float:
        """The figures' width in inches, as they appear."""
        from .figures import _size_inches

        return _size_inches(self.figure_size, (7.5, 3.6))[0] if self.figure_size is not None else 6.0

    @staticmethod
    def _alt(fig: Any, caption: str) -> str:
        """Alt text: what the figure shows, in words (its title, panels and series)."""
        describe = getattr(fig, "describe", None)
        return describe(technical=False) if callable(describe) else caption

    def _excel(self, path: Path) -> None:
        try:
            from openpyxl import Workbook
            from openpyxl.drawing.image import Image
            from openpyxl.styles import Alignment, Font
            from openpyxl.worksheet.hyperlink import Hyperlink
        except ImportError as exc:
            raise ImportError("Excel reports need openpyxl: pip install pymetrix[report]") from exc

        wb = Workbook()
        contents = wb.active
        contents.title = "Contents"
        contents["A1"] = self.title
        contents["A1"].font = Font(bold=True, size=14)
        contents["A2"] = " · ".join(x for x in (self.author, self.date) if x)
        row = 4
        used: set[str] = {"Contents"}
        number = 0
        blocks = list(self.blocks)
        appendix = self._appendix()
        if appendix is not None:
            blocks += [_Block("heading", "Appendix", level=1), _Block("table", appendix, "Provenance")]
        for block in blocks:
            if block.kind == "heading":
                contents.cell(row, 1, block.value).font = Font(bold=True, size=13 - block.level)
                row += 1
            elif block.kind == "text":
                cell = contents.cell(row, 1, block.value)
                cell.alignment = Alignment(wrap_text=True, vertical="top")
                contents.row_dimensions[row].height = 15 * (1 + len(block.value) // 110)
                row += 1
            else:
                number += 1
                name = self._sheet_name(number, block.caption or getattr(block.value, "title", "") or block.kind, used)
                ws = wb.create_sheet(name)
                link = contents.cell(row, 1, f"{number}. {block.caption or getattr(block.value, 'title', '') or block.kind}")
                link.hyperlink = Hyperlink(ref=link.coordinate, location=f"'{name}'!A1")
                link.font = Font(color="0563C1", underline="single")
                row += 1
                if block.kind == "table":
                    self._excel_table(ws, block.value, Font)
                else:
                    ws["A1"] = block.caption or block.value.title
                    ws["A1"].font = Font(bold=True)
                    img = Image(io.BytesIO(self._png(block.value)))
                    img.anchor = "A3"
                    ws.add_image(img)
        contents.column_dimensions["A"].width = 110
        wb.save(path)

    @staticmethod
    def _sheet_name(number: int, title: str, used: set[str]) -> str:
        bad = '[]:*?/\\'
        clean = "".join(c for c in title if c not in bad).strip() or "Table"
        name = f"{number} {clean}"[:31]
        while name in used:
            name = name[:28] + f" {len(used)}"
        used.add(name)
        return name

    def _excel_table(self, ws: Any, t: Table, Font: Any) -> None:
        r = 1
        if t.title:
            ws.cell(r, 1, t.title).font = Font(bold=True)
            r += 2
        frame = self._frame(t)
        for c, name in enumerate(frame.columns, 1):
            ws.cell(r, c, str(name)).font = Font(bold=True)
        for values in frame.itertuples(index=False):
            r += 1
            for c, v in enumerate(values, 1):
                if isinstance(v, (float, np.floating)):
                    if np.isfinite(v):
                        cell = ws.cell(r, c, float(v))
                        cell.number_format = ("0." + "0" * self.digits if self.digits else "0") if abs(v) < 1e6 else "0.000E+00"
                elif isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                    ws.cell(r, c, int(v))
                elif v is not None:
                    ws.cell(r, c, str(v))
        for note in t.notes:
            r += 1
            ws.cell(r + 1, 1, str(note)).font = Font(italic=True, size=9)
        for c, name in enumerate(frame.columns, 1):
            width = max([len(str(name))] + [len(f"{v:.4f}" if isinstance(v, float) else str(v)) for v in frame.iloc[:, c - 1]]) + 2
            ws.column_dimensions[ws.cell(1, c).column_letter].width = min(max(width, 8), 60)

    def _word(self, path: Path) -> None:
        try:
            import docx
            from docx.shared import Inches, Pt
        except ImportError as exc:
            raise ImportError("Word reports need python-docx: pip install pymetrix[report]") from exc

        doc = docx.Document()
        doc.core_properties.title = self.title
        doc.core_properties.author = self.author
        doc.add_heading(self.title, level=0)
        if self.author or self.date:
            doc.add_paragraph(" · ".join(x for x in (self.author, self.date) if x)).runs[0].italic = True
        blocks = list(self.blocks)
        appendix = self._appendix()
        if appendix is not None:
            blocks += [_Block("heading", "Appendix: where the numbers come from", level=1), _Block("table", appendix)]
        tables = figures = 0
        for block in blocks:
            if block.kind == "heading":
                doc.add_heading(block.value, level=block.level)
            elif block.kind == "text":
                doc.add_paragraph(block.value)
            elif block.kind == "table":
                tables += 1
                t: Table = block.value
                caption = block.caption or t.title
                if caption:
                    p = doc.add_paragraph()
                    p.add_run(f"Table {tables}. {caption}").bold = True
                frame = self._frame(t)
                table = doc.add_table(rows=1 + len(frame), cols=len(frame.columns))
                table.style = "Light List Accent 1" if "Light List Accent 1" in [s.name for s in doc.styles] else "Table Grid"
                for c, name in enumerate(frame.columns):
                    table.cell(0, c).text = str(name)
                for r, values in enumerate(frame.itertuples(index=False), 1):
                    for c, v in enumerate(values):
                        cell = table.cell(r, c)
                        cell.text = _fmt(v, self.digits)
                        if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                            cell.paragraphs[0].alignment = 2  # right
                for row in table.rows:
                    for cell in row.cells:
                        for run in cell.paragraphs[0].runs:
                            run.font.size = Pt(8.5)
                for note in t.notes:
                    run = doc.add_paragraph().add_run(str(note))
                    run.italic, run.font.size = True, Pt(8)
            else:
                figures += 1
                picture = doc.add_picture(io.BytesIO(self._png(block.value)), width=Inches(self._width()))
                picture._inline.docPr.set("descr", self._alt(block.value, block.caption))  # alt text, for screen readers
                picture._inline.docPr.set("title", block.caption or block.value.title or "Figure")
                caption = block.caption or block.value.title
                if caption:
                    p = doc.add_paragraph()
                    p.add_run(f"Figure {figures}. {caption}").italic = True
        doc.save(str(path))

    def _html(self, path: Path) -> None:
        parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(self.title)}</title>",
                 "<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem;color:#222}"
                 "table{border-collapse:collapse;margin:.5rem 0 1rem;font-size:.85rem}th,td{padding:.2rem .6rem;border-bottom:1px solid #ddd}"
                 "th{text-align:left;border-bottom:2px solid #888}td.n{text-align:right;font-variant-numeric:tabular-nums}"
                 "figure{margin:1rem 0}figcaption,.note{font-size:.8rem;color:#555}img{max-width:100%}</style></head><body>",
                 f"<h1>{html.escape(self.title)}</h1>"]
        if self.author or self.date:
            parts.append(f"<p class='note'>{html.escape(' · '.join(x for x in (self.author, self.date) if x))}</p>")
        blocks = list(self.blocks)
        appendix = self._appendix()
        if appendix is not None:
            blocks += [_Block("heading", "Appendix: where the numbers come from", level=2), _Block("table", appendix)]
        for block in blocks:
            if block.kind == "heading":
                parts.append(f"<h{block.level + 1}>{html.escape(block.value)}</h{block.level + 1}>")
            elif block.kind == "text":
                parts.append(f"<p>{html.escape(block.value)}</p>")
            elif block.kind == "table":
                parts.append(self._html_table(block.value, block.caption))
            else:
                data = base64.b64encode(self._png(block.value)).decode()
                caption = html.escape(block.caption or block.value.title)
                alt = html.escape(self._alt(block.value, block.caption), quote=True)
                parts.append(f"<figure><img alt=\"{alt}\" src='data:image/png;base64,{data}'><figcaption>{caption}</figcaption></figure>")
        parts.append("</body></html>")
        path.write_text("\n".join(parts), encoding="utf-8")

    def _html_table(self, t: Table, caption: str) -> str:
        from .summary_layout import notes_html

        frame = self._frame(t)
        head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
        body = []
        for values in frame.itertuples(index=False):
            cells = "".join(f"<td class='n'>{_fmt(v, self.digits)}</td>" if isinstance(v, (int, float, np.number)) and not isinstance(v, bool)
                            else f"<td>{html.escape(_fmt(v, self.digits))}</td>" for v in values)
            body.append(f"<tr>{cells}</tr>")
        title = html.escape(caption or t.title)
        notes = f"<div class='note'>{notes_html(t.notes)}</div>" if t.notes else ""  # statistics grouped, as in a notebook
        return f"<p><b>{title}</b></p><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>{notes}"

    def _markdown(self, path: Path) -> None:
        """Markdown, with figures as PNG files in ``<name>_files`` beside it."""
        folder = path.with_name(f"{path.stem}_files")
        lines = [f"# {self.title}", ""]
        if self.author or self.date:
            lines += [f"*{' · '.join(x for x in (self.author, self.date) if x)}*", ""]
        blocks = list(self.blocks)
        appendix = self._appendix()
        if appendix is not None:
            blocks += [_Block("heading", "Appendix: where the numbers come from", level=1), _Block("table", appendix)]
        tables = figures = 0
        for block in blocks:
            if block.kind == "heading":
                lines += ["#" * (block.level + 1) + " " + block.value, ""]
            elif block.kind == "text":
                lines += [block.value, ""]
            elif block.kind == "table":
                tables += 1
                lines += self._markdown_table(block.value, block.caption, tables)
            else:
                figures += 1
                folder.mkdir(parents=True, exist_ok=True)
                name = f"figure-{figures}.png"
                (folder / name).write_bytes(self._png(block.value))
                caption = block.caption or block.value.title
                alt = self._alt(block.value, caption).replace("[", "(").replace("]", ")")
                lines += [f"![{alt}]({folder.name}/{name})", "", f"*Figure {figures}. {caption}*", ""]
        path.write_text("\n".join(lines), encoding="utf-8")

    def _markdown_table(self, t: Table, caption: str, number: int) -> list[str]:
        out = [f"**Table {number}. {caption or t.title}**", ""] if (caption or t.title) else []
        return out + [t.to_markdown(self.digits), ""]
