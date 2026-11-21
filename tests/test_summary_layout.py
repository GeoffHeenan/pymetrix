"""Summary statistics in a result's HTML and its window: "Label: value" notes grouped under headings (Estimation, Fit,
Information criteria, Residual checks, Other), the other notes kept as sentences after them, nothing lost; the plain
text, CSV and Markdown as before."""

from __future__ import annotations

import html
import os
import warnings

import pytest

from pymetrix import Dataset, Equation
from pymetrix.output import Table
from pymetrix.summary_layout import notes_html, parse_stat, split_notes
from pymetrix.wording import STAT_GROUPS, stat_group


@pytest.fixture(scope="module")
def macro():
    return Dataset.example("sample_macro")


@pytest.fixture(scope="module")
def ols(macro):
    return Equation("c ~ y").fit(macro)


def _shown(note: str, out: str) -> bool:
    """A note is in the HTML: whole (prose), or its label and its value (a statistic)."""
    stat = parse_stat(note)
    if stat is None:
        return html.escape(note) in out
    return all(f">{html.escape(part)}</td>" in out for part in stat)


def _results(macro):
    import pymetrix as pm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = Equation("c ~ y").fit(macro)
        yield "FittedEquation", fit
        yield "HAC", Equation("c ~ y + c[-1]").fit(macro, method=pm.OLS(cov="hac"))
        yield "TestResult", fit.test("breusch_godfrey", lags=4)
        yield "ADF", pm.stats.run_test(macro, "adf", variable="c")
        yield "VAR", pm.VAR(["c", "y"], lags=2).fit(macro)
        yield "System", pm.System([Equation("c ~ y"), Equation("i ~ y")]).fit(macro, method=pm.SUR())
        yield "ARIMA", pm.SARIMA("c", (1, 1, 0)).fit(macro)
        yield "GARCH", Equation("100*dlog(c) ~ 1", variance=pm.GARCH()).fit(macro)


def test_every_note_still_appears_in_the_html(macro):
    from pymetrix.views import view_sections

    checked = 0
    for kind, obj in _results(macro):
        for title, part in view_sections(obj):
            if isinstance(part, Table) and part.notes:
                out = part._repr_html_()
                missing = [n for n in part.notes if not _shown(str(n), out)]
                assert not missing, (kind, title, missing)
                checked += 1
        assert "<" in obj._repr_html_(), kind
    assert checked >= 8


def test_the_ols_statistics_are_grouped(ols):
    split = split_notes(ols.summary().notes)
    groups = {g.title: [label for label, _ in g.items] for g in split.groups}
    assert groups == {
        "Estimation": ["Method", "Sample"],
        "Fit": ["R-squared", "Adjusted R-squared", "S.E. of regression", "Log likelihood"],
        "Information criteria": ["Akaike info criterion", "Schwarz criterion", "Hannan-Quinn criterion"],
        "Residual checks": ["Durbin-Watson"],
    }
    assert [g.title for g in split.groups] == [g for g in STAT_GROUPS if g in groups]  # in the headings' order
    assert split.prose == () and split.headed
    out = ols._repr_html_()
    positions = [out.index(f">{title}</div>") for title in groups]
    assert positions == sorted(positions)  # each heading shown, in order
    assert "Durbin-Watson" in out.split(">Residual checks</div>")[1]


@pytest.mark.parametrize(("label", "group"), [
    ("R-squared", "Fit"), ("r-squared (within)", "Fit"), ("Adjusted R-squared", "Fit"), ("LOG LIKELIHOOD", "Fit"),
    ("S.E. of regression", "Fit"), ("Standard errors", "Estimation"), ("aic", "Information criteria"),
    ("Hannan-Quinn criterion", "Information criteria"), ("Durbin-Watson", "Residual checks"),
    ("Hansen J p-value", "Residual checks"), ("nobs", "Estimation"), ("First-stage F (x)", "Estimation"),
    ("Flux capacitance", "Other"), ("k", "Other"),
])
def test_labels_map_to_groups_case_insensitively_by_prefix(label, group):
    assert stat_group(label) == group


def test_an_unknown_label_lands_in_other_and_is_kept():
    split = split_notes(["Method: OLS", "Flux capacitance: 1.21", "R-squared: 0.5"])
    assert [g.title for g in split.groups] == ["Estimation", "Fit", "Other"]
    assert split.groups[-1].items == (("Flux capacitance", "1.21"),)
    out = notes_html(["Method: OLS", "Flux capacitance: 1.21", "R-squared: 0.5"])
    assert ">Other</div>" in out and ">Flux capacitance</td>" in out and ">1.21</td>" in out


def test_sentences_stay_prose_after_the_statistics():
    notes = ["Warnings: the optimiser stopped early",
             "Null hypothesis: no serial correlation up to the given lag",
             "Standard errors from the observed information, treating the uniforms as known.",
             "Bands: 68% and 90%. Standard errors: Newey-West with h+1 lags",
             "Log likelihood -279.0294; AIC 564.0588; BIC 571.8744",
             "Column 'a, b[-h]': the correlation of a's residual with b's h periods earlier",
             "100 rows, 4 series: 1 note",
             "Information criteria: AIC -5.96278, BIC -5.68254, HQ -5.84898, FPE 0.00257294",
             "R-squared: 0.9",
             "Distribution: chi2"]
    split = split_notes(notes)
    assert list(split.prose) == notes[:-2]
    assert [g.items for g in split.groups] == [(("R-squared", "0.9"),), (("Distribution", "chi2"),)]
    out = notes_html(notes)
    for prose in notes[:-2]:
        assert f">{html.escape(prose)}</p>" in out
    assert out.index("R-squared") < out.index("Warnings")  # the statistics first


def test_one_group_shows_no_heading():
    out = notes_html(["Null hypothesis: a unit root", "Distribution: Dickey-Fuller"])
    assert "Residual checks" not in out and ">Distribution</td>" in out and "Null hypothesis: a unit root" in out


def test_long_or_numeric_values():
    assert parse_stat("Sample: 2000Q1 to 2024Q4, 100 observations (0 dropped)") == (
        "Sample", "2000Q1 to 2024Q4, 100 observations (0 dropped)")
    assert parse_stat("Method: OLS; HAC standard errors (Newey-West, Bartlett kernel, 3 lags)")[0] == "Method"
    assert parse_stat("Breusch-Pagan test of a diagonal covariance: 5.273 (df 1, p = 0.0217)")[1].startswith("5.273")
    assert parse_stat("Loss weights: zdivgr 1, delrff 1") == ("Loss weights", "zdivgr 1, delrff 1")
    assert parse_stat("Made by: fit = Equation(...)") is None  # code, not a statistic
    assert parse_stat("Prior: the returns implied by the market weights, pi = delta S w with delta = 2.5") is None


def test_the_html_table_fits_its_content_with_numbers_right_aligned(ols):
    out = ols._repr_html_()
    table = out[out.index("<table"):out.index("</table>")]
    assert "width:100%" not in table and "border-collapse:collapse" in table
    assert "font-weight:bold" in table and 'text-align:left' in table  # the row labels
    assert table.count('<td align="right"') == 2 * 4  # every coefficient, std err, t and p
    assert "background-color:rgba(128, 128, 128, 0.10)" in table  # every other row shaded
    assert "#" not in table.replace("&#", "")  # no hard-coded colour in the table itself (named constants only)


def test_plain_text_csv_and_markdown_are_unchanged():
    """As the main branch writes them before this change (recorded from it)."""
    table = Table(["coef", "p"], [[1.5, 0.01], [-2.25, 0.5]], index=["a", "b"], title="y ~ a + b",
                  notes=["Method: OLS", "R-squared: 0.9000", "Warnings: few observations"])
    assert str(table) == ("y ~ a + b\n      coef       p\n------------------\na   1.5000  0.0100\nb  -2.2500  0.5000\n"
                          "Method: OLS\nR-squared: 0.9000\nWarnings: few observations")
    assert table.to_markdown() == ("|  | coef | p |\n|---|---:|---:|\n| a | 1.5000 | 0.0100 |\n| b | -2.2500 | 0.5000 |\n\n"
                                   "*Method: OLS*  \n*R-squared: 0.9000*  \n*Warnings: few observations*  ")
    assert table.to_csv() == ",coef,p\na,1.5,0.01\nb,-2.25,0.5\n"


def test_the_fitted_equation_text_keeps_its_notes_line_by_line(ols):
    text = str(ols)
    notes = list(ols.summary().notes)
    assert text.splitlines()[-len(notes):] == notes


def test_the_html_report_groups_the_statistics(ols, tmp_path):
    from pymetrix.report import Report

    report = Report("Fits")
    report.add(ols.summary())
    path = tmp_path / "fits.html"
    report.save(str(path))
    out = path.read_text(encoding="utf-8")
    assert ">Information criteria</div>" in out and ">Durbin-Watson</td>" in out


# ---------------------------------------------------------------------------------------------- the window

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_the_window_shows_every_note_grouped_in_columns_that_follow_the_width(ols):
    from pymetrix.ui.stats_panel import COLUMN_GAP, LABEL_GAP, StatsPanel

    _app()
    notes = list(ols.summary().notes) + ["Warnings: a sentence that stays a sentence."]
    panel = StatsPanel(notes)
    assert [g.title for g in panel.groups] == ["Estimation", "Fit", "Information criteria", "Residual checks"]
    assert panel.text().splitlines() == [n for n in notes if parse_stat(n)] + [notes[-1]]  # every note, word for word
    # three columns from the width they need in the font as drawn (each column its widest group), not a fixed width:
    # about 840 pixels with a usual 9-point UI font, 1220 where every glyph is drawn as a box (Windows CI, no fonts)
    three = sum(max(g.natural for g in panel.groups[c::3]) for c in range(3)) + 2 * COLUMN_GAP
    for width, columns in ((three, 3), (three - 1, 2), (260, 1)):
        panel.resize(width, 400)
        panel.show()
        _app().processEvents()
        assert panel.columns == columns, width
    # a narrow window wraps the statistics rather than scroll sideways: the panel is as narrow as its widest group's
    # longest label word, the gap and its longest value word (or the group's heading), each in the font it is drawn in,
    # not as wide as the Sample line. A few pixels allow for the labels' own margins.
    def widest_word(label):
        return max(label.fontMetrics().horizontalAdvance(word) for word in label.text().split())

    needed = max(max(widest_word(g.heading) if g.heading is not None else 0,
                     max(widest_word(name) for name, _ in g.rows) + LABEL_GAP + max(widest_word(v) for _, v in g.rows))
                 for g in panel.groups)
    least = panel.minimumSizeHint().width()
    assert least <= needed + 8, (least, needed)
    sample = next(f"{name.text()}: {shown.text()}" for g in panel.groups for name, shown in g.rows if name.text() == "Sample")
    assert least < 0.6 * panel.fontMetrics().horizontalAdvance(sample), (least, sample)
    heading = panel.groups[0].heading
    from pymetrix.ui import theme

    # muted by the theme's secondary-text colour, derived from the palette and readable on it (4.5:1)
    assert heading.text() == "Estimation" and theme.muted_css(heading.palette()) in heading.styleSheet()
    narrowest = min(panel.groups, key=lambda g: g.natural)  # well short of the 420 pixels a group wraps beyond
    before = narrowest.natural
    font = panel.font()
    font.setPointSizeF(font.pointSizeF() * 2)
    panel.setFont(font)  # zoomed in: measured again
    _app().processEvents()
    assert narrowest.natural > 1.5 * before
    assert narrowest.rows[0][1].font().pointSizeF() == pytest.approx(font.pointSizeF())
    panel.close()


def test_the_summary_view_of_a_fit_uses_the_panel(ols):
    from pymetrix.ui.qt import object_view
    from pymetrix.ui.stats_panel import StatsPanel

    _app()
    view = object_view(ols, name="fit", code="fit = ...")
    assert any("Durbin-Watson: " in panel.text() for panel in view.findChildren(StatsPanel))
    grid = view.grids["Summary"]
    assert grid.alternatingRowColors() and grid.horizontalHeader().font().bold()
    view.close()


# ------------------------------------------------------------- estimators that wrote their statistics in one note

def test_arima_and_var_write_one_statistic_a_note_and_group_like_ols(macro):
    import pymetrix as pm

    arima = pm.SARIMA("c", (1, 1, 0)).fit(macro)
    s = arima.stats
    notes = list(arima.summary().notes)
    assert f"Log likelihood: {s['loglike']:.4f}" in notes and f"Akaike info criterion: {s['aic']:.4f}" in notes
    assert f"Schwarz criterion: {s['bic']:.4f}" in notes
    assert str(arima.summary()).splitlines()[-len(notes):] == notes  # its text: one statistic a line
    groups = {g.title: [label for label, _ in g.items] for g in split_notes(notes).groups}
    assert groups["Information criteria"] == ["Akaike info criterion", "Schwarz criterion"]
    assert groups["Fit"] == ["Log likelihood"]

    var = pm.VAR(["c", "y"], lags=2).fit(macro)
    notes = list(var.summary().notes)
    e = var.estimate
    assert f"Akaike info criterion: {e.criteria['aic']:.6g}" in notes and f"Log likelihood: {e.loglike:.3f}" in notes
    assert any(n.startswith("Largest root: ") for n in notes)
    groups = {g.title: [label for label, _ in g.items] for g in split_notes(notes).groups}
    assert groups["Information criteria"][:3] == ["Akaike info criterion", "Schwarz criterion", "Hannan-Quinn criterion"]


def test_a_test_results_values_are_right_aligned(ols):
    result = ols.test("breusch_godfrey", lags=4)
    out = result._repr_html_()
    table = out[out.index("<table"):out.index("</table>")]
    assert table.count('<td align="right"') == 3  # statistic, df (written as text) and p-value
    assert '<th align="right"' in table
    from pymetrix.summary_layout import is_numberish

    assert is_numberish("4") and is_numberish("(2, 96)") and is_numberish(0.5) and not is_numberish("chi2")
    assert not is_numberish(True) and not is_numberish("")


def test_a_test_results_values_are_right_aligned_in_the_window(ols):
    from PySide6.QtCore import Qt as QtNS

    from pymetrix.ui.qt import object_view

    _app()
    view = object_view(ols.test("breusch_godfrey", lags=4), name="bg", code="bg = ...")
    grid = next(iter(view.grids.values()))
    column = [grid.horizontalHeaderItem(c).text() for c in range(grid.columnCount())].index("value")
    for r in range(grid.rowCount()):
        assert grid.item(r, column).textAlignment() & QtNS.AlignmentFlag.AlignRight, grid.item(r, 0).text()
    view.close()
