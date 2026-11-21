"""Reports to Excel, Word and HTML."""

import re

import pytest

pytest.importorskip("matplotlib")

from pymetrix import VAR, Dataset, Equation, Report  # noqa: E402
from pymetrix.output import Table  # noqa: E402


@pytest.fixture(scope="module")
def report():
    data = Dataset.example("sample_macro").with_series(ly="100*log(y)", lc="100*log(c)")
    fit = Equation("c ~ y + c[-1]").fit(data)
    var = VAR(["ly", "lc"], lags=2).fit(data)
    rep = Report("Quarterly forecast", author="Forecasting team", date="2026-10-12")
    rep.heading("Consumption").text("The consumption function is stable.\n\nResiduals are white noise.")
    rep.add(fit, caption="Consumption function")
    rep.add(fit.test("breusch_godfrey", lags=4))
    rep.heading("VAR", 2).add(var.irf(8, draws=0), caption="Impulse responses")
    rep.add(Table(["a", "b"], [[1, 2.5]], title="A small table"))
    return rep, fit


def test_what_goes_in(report):
    rep, fit = report
    kinds = [(b.kind, b.caption) for b in rep.blocks]
    assert kinds[:3] == [("heading", ""), ("text", ""), ("text", "")]
    assert ("table", "Consumption function") in kinds and ("figure", "Consumption function: Graph") in kinds
    assert ("table", "Breusch-Godfrey LM") in kinds and ("table", "A small table") in kinds
    assert not any("Residuals" in c for _, c in kinds)  # a result adds its summary and graph, not every tab
    every = Report("x").add(fit, sections="all")
    assert len(every.blocks) > 5
    chosen = Report("x").add(fit, sections=["Diagnostics"])
    assert [b.caption for b in chosen.blocks] == [chosen.blocks[0].value.title]
    with pytest.raises(KeyError, match="no section"):
        Report("x").add(fit, sections=["Nothing"])
    with pytest.raises(ValueError, match="xlsx, .docx, .html or .md"):
        rep.save("report.pdf")


def test_excel(report, tmp_path):
    load_workbook = pytest.importorskip("openpyxl").load_workbook  # the [excel] extra

    rep, fit = report
    wb = load_workbook(rep.save(tmp_path / "r.xlsx"))
    assert wb.sheetnames[0] == "Contents" and wb.sheetnames[1] == "1 Consumption function" and wb.sheetnames[-1].endswith("Provenance")
    contents = wb["Contents"]
    assert contents["A1"].value == "Quarterly forecast" and contents["A2"].value == "Forecasting team · 2026-10-12"
    links = [c.hyperlink.location for c in contents["A"] if c.hyperlink]
    assert links[0] == "'1 Consumption function'!A1"
    table = wb["1 Consumption function"]
    rows = list(table.iter_rows(values_only=True))
    header = next(r for r in rows if r and r[1] == "coef")
    const = next(r for r in rows if r and r[0] == "const")
    assert header[:3] == (None, "coef", "std err") and const[1] == pytest.approx(fit.params["const"])  # numbers stay numbers
    assert len(wb["2 Consumption function Graph"]._images) == 1
    provenance = list(wb[wb.sheetnames[-1]].iter_rows(values_only=True))
    assert any(r and r[0] == "Consumption function" and r[1] == "OLS" for r in provenance)


def test_html(report, tmp_path):
    rep, _ = report
    text = rep.save(tmp_path / "r.html").read_text(encoding="utf-8")
    assert "<h1>Quarterly forecast</h1>" in text and "<h2>Consumption</h2>" in text and "<h3>VAR</h3>" in text
    assert len(re.findall(r"<img alt=", text)) == 2 and "data:image/png;base64," in text
    assert "Where the numbers come from" in text and "<td class='n'>" in text


def test_word(report, tmp_path):
    docx = pytest.importorskip("docx")
    rep, _ = report
    doc = docx.Document(str(rep.save(tmp_path / "r.docx")))
    text = [p.text for p in doc.paragraphs]
    assert text[0] == "Quarterly forecast" and "Table 1. Consumption function" in text and "Figure 1. Consumption function: Graph" in text
    assert len(doc.inline_shapes) == 2 and len(doc.tables) == 4  # summary, test, small table, provenance
    assert doc.core_properties.author == "Forecasting team"
    assert doc.tables[0].cell(0, 1).text == "coef"


def test_dialog(tmp_path):
    from pymetrix.commands import ExportReport, FitEquation, Session

    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    s.run(FitEquation("c ~ y", name="fit"))
    path = str(tmp_path / "dialog.html")
    s.run(ExportReport("fit", path=path, title="From the dialog"))
    assert "report = Report('From the dialog')\nreport.heading('fit', 2).add(fit)\nreport.save(" in s.script()
    assert "<h1>From the dialog</h1>" in (tmp_path / "dialog.html").read_text(encoding="utf-8")
    assert ExportReport("fit", path="r.pdf").validate(s) == ["save the report as .xlsx, .docx, .html or .md"]


def test_missing_writers_say_which_extra_installs_them(monkeypatch, tmp_path):
    import sys

    from pymetrix import Report

    monkeypatch.setitem(sys.modules, "openpyxl", None)  # as if not installed
    with pytest.raises(ImportError, match=r"pip install pymetrix\[report\]"):
        Report("x").save(tmp_path / "r.xlsx")
