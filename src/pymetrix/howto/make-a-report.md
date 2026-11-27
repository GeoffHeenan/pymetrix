---
summary: collect results, headings and text in the report outline as you work, and write them out as HTML, Word, Excel or Markdown, with where each number came from.
audience: everyone
order: 11
---

# Make a report

Put results into a report as you work: a result's summary and graph, a chosen table, headings and paragraphs, in the
order you want. Then write it out as HTML, Word, Excel or Markdown, with an appendix that says where each number came
from. Writing the report is a recorded step, so the script writes it again.

## Before you start

- Results to report. This page reports the error-correction equation for consumption of [Estimate an equation and
  test its residuals](estimate-and-test-an-equation.md): its summary and graph, and the table of its residuals'
  tests.
- The format. HTML and Markdown need nothing more; Excel puts each table on a sheet of its own, with the numbers as
  numbers; Word needs python-docx (`pip install pymetrix[report]`).

## In the app

1. File → Load example data…, with Example dataset: sample_macro and Save as: data. Estimate → Equation…, with
   Formula: `dlog(c) ~ dlog(y) + log(c)[-1] + log(y)[-1]` and Save as: ecm.
2. In the result's window, Add to report (in its corner) puts its summary and graph in the report outline. Choose
   the Diagnostics view, and Add to report under its table puts that table in too.
3. View → Report shows the Report panel: the outline, an item a line. Heading… adds a heading at the end (The
   consumption function); Up and Down move the selected item, so put the heading first. Text… adds a paragraph:
   Consumption returns to its long-run ratio to output. Double-click the Diagnostics item to give it a caption:
   Tests of the residuals.
4. Export… opens File → Export report… on the outline, with Use the report outline ticked. Write to:
   consumption.html, Report title: Consumption and Author: Forecasting team. Provenance appendix (under Advanced
   settings) is ticked: the appendix lists the data, samples and methods behind each result.
5. Without the outline, File → Export report… takes Results (each with its summary and graph, or the views a
   template names under Views: estimation_summary, forecast_round, technical_appendix) and writes them in the order
   listed.

## In code

The step Export… records writes each item out, so the report is made again without the app:

```python
from pymetrix import Dataset, Equation, Report

data = Dataset.example("sample_macro")
ecm = Equation("dlog(c) ~ dlog(y) + log(c)[-1] + log(y)[-1]").fit(data)
report = Report("Consumption", author="Forecasting team")
report.heading("The consumption function", 2)
report.add(ecm)                                                   # its summary and graph
report.add(ecm, caption="Tests of the residuals", sections=["Diagnostics"])
report.text("Consumption returns to its long-run ratio to output.")
report.save("consumption.html")                                   # or .docx, .xlsx, .md
```

The outline, as the app keeps it in the session, and its export:

```python
from pymetrix.commands import ExportReport, FitEquation, LoadExample, Session

session = Session()
session.run(LoadExample("sample_macro", name="data"))
session.run(FitEquation("dlog(c) ~ dlog(y) + log(c)[-1] + log(y)[-1]", name="ecm"))
session.add_report_heading("The consumption function")
session.add_to_report("ecm")                                      # Add to report, in the window's corner
session.add_to_report("ecm", "Diagnostics", caption="Tests of the residuals")
session.add_report_text("Consumption returns to its long-run ratio to output.")
session.run(ExportReport.from_outline(session, path="outline.html", title="Consumption", author="Forecasting team"))
```

## Check the result

- The file's headings: Consumption (the title), The consumption function, and Appendix: where the numbers come
  from. The two HTML files, from code and from the outline, have the same headings, captions and tables.
- The tables: the coefficients as the Summary view shows them (-0.75 on the lagged log of consumption), and the
  Diagnostics table with Breusch-Godfrey, Breusch-Pagan, Jarque-Bera and RESET, under its caption.
- The appendix, a row for each result: its method (OLS), its sample (2000Q2 to 2024Q4, 99 observations), the data's
  vintages where they are recorded, and the pymetrix version.
- View → Script: the export step as the code above, `report.save('outline.html')` last.

## Pitfalls

- **Word needs python-docx.** Without it, a `.docx` report is refused with what to install; HTML, Markdown and
  Excel need nothing more.
- **Putting the outline together is not a step.** Adding, moving and captioning items are not in History or the
  script; the outline is saved with the project, and the export step writes it out in full.
- **A report shows results as they are now.** A result that is out of date (its data changed after it was made) is
  reported as it is, with a caution at the top of the report: recalculate first (Calculate → Recalculate).
- **The date is today's.** Each report is dated when it is written; give Date (under Advanced settings) to fix it.
- **Markdown writes its figures as files.** They go in a folder beside the `.md` file: send both.

## See also

- [Reports](pymetrix:topic/reports), in the help.
- [Run a forecast round: baseline, judgement, scenarios, report](run-a-forecast-round.md): a round's report from
  its template.
- [The user guide's Reports](../../../docs/guide.md#reports),
  [Templates, themes and chart options](../../../docs/guide.md#templates-themes-and-chart-options) and
  [The report outline: collect as you go](../../../docs/guide.md#the-report-outline-collect-as-you-go).
