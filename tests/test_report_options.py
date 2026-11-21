"""Report and chart options: themes, date spans, subtitles and footnotes, templates and Markdown."""

import re
import numpy as np
import pytest

pytest.importorskip("matplotlib")

from pymetrix import VAR, Dataset, Equation, ForecastRound, Report, get_theme, report_templates, set_theme, themes  # noqa: E402
from pymetrix.figures import PALETTE  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return Dataset.example("sample_macro").with_series(ly="100*log(y)", lc="100*log(c)")


@pytest.fixture(scope="module")
def fit(data):
    return Equation("c ~ y + c[-1]").fit(data)


@pytest.fixture(scope="module")
def round_(data):
    var = VAR(["ly", "lc"], lags=1).fit(data)
    return ForecastRound.from_forecast("baseline", var.forecast(data, ("2025Q1", "2026Q4")), data=data)


def test_themes():
    assert themes() == ["default", "publication", "teaching", "policy", "monochrome"]
    assert get_theme().name == "default" and get_theme("policy").digits == 2
    with pytest.raises(KeyError, match="no theme 'loud'"):
        get_theme("loud")
    previous = set_theme("teaching")
    try:
        assert get_theme().name == "teaching" and previous == "default"
    finally:
        set_theme(previous)
    assert get_theme("policy").with_options(digits=1).digits == 1 and get_theme("policy").digits == 2


def test_figure_options_change_the_look_not_the_numbers(round_, data):
    fig = round_.plot(data)
    clipped = fig.with_options(span=("2025Q3", None), subtitle="Log levels, times 100", footnote="Source: sample data.")
    full, short = fig.data(), clipped.data()
    assert short.index.min().strftime("%YQ%q") == "2025Q3" and len(short) < len(full)
    assert np.allclose(short["ly: forecast"].dropna(), full["ly: forecast"].loc["2025Q3":].dropna())  # same numbers, fewer dates
    drawn = clipped.draw()
    assert "Log levels, times 100" in drawn._suptitle.get_text() and drawn._supxlabel.get_text() == "Source: sample data."
    with pytest.raises(TypeError, match="no figure option colour"):
        fig.with_options(colour="red")
    with pytest.raises(KeyError, match="no theme"):
        fig.with_options(theme="loud")

    def forecast_line(figure):
        ax = figure.draw().axes[0]
        return next(line for line in ax.get_lines() if line.get_label() == "forecast")

    assert forecast_line(fig).get_color() == PALETTE[0]
    assert forecast_line(fig.with_options(theme="policy")).get_color() == get_theme("policy").palette[0]  # the theme's own palette
    mono = fig.with_options(theme="monochrome").draw().axes[0]
    assert {line.get_color() for line in mono.get_lines()} <= set(get_theme("monochrome").palette) | {"#444444"}


def test_markdown_and_digits(fit, tmp_path):
    rep = Report("Consumption", author="Team", date="2026-10-11", theme="policy").add(fit, caption="Consumption function")
    path = rep.save(tmp_path / "r.md")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Consumption\n\n*Team · 2026-10-11*") and "**Table 1. Consumption function**" in text
    assert f"| y | {fit.params['y']:.2f} |" in text  # from the coefficients, at the theme's two decimals
    assert re.search(r"!\[[^\]]+\]\(r_files/figure-1\.png\)", text) and "*Figure 1. Consumption function: Graph*" in text  # alt text: what it shows
    assert (tmp_path / "r_files" / "figure-1.png").stat().st_size > 1000
    assert "Where the numbers come from" in text and "|---|" in text.replace("---:", "---")
    html = Report("x", theme="publication").add(fit).save(tmp_path / "r.html").read_text(encoding="utf-8")
    assert f">{fit.params['y']:.3f}<" in html


def test_templates(fit, round_, data):
    assert report_templates()[:3] == ["estimation_summary", "forecast_round", "technical_appendix"]  # then plugins', as finance_summary
    est = Report.from_template("estimation_summary", {"Consumption": fit}, title="Equations")
    assert [b.caption for b in est.blocks if b.kind != "heading"] == ["Consumption", "Consumption: Graph", "Consumption: Diagnostics"]
    assert est.template == "estimation_summary" and est.title == "Equations"
    forecast = Report.from_template("forecast_round", {"Baseline": round_}, theme="policy", span=("2020Q1", None))
    assert [b.kind for b in forecast.blocks] == ["heading", "figure", "table"] and forecast.span == ("2020Q1", None)
    everything = Report.from_template("technical_appendix", [fit])
    assert sum(b.kind != "heading" for b in everything.blocks) == len(__import__("pymetrix.views", fromlist=["titles"]).titles(fit))
    with pytest.raises(KeyError, match="no report template 'brief'"):
        Report.from_template("brief", [fit])
    with pytest.raises(ValueError, match="nothing to show for 'LM test'"):
        Report.from_template("forecast_round", {"LM test": fit.test("breusch_godfrey", lags=2)})


def test_the_dialog_records_template_and_theme(fit, data, tmp_path):
    from pymetrix.commands import ExportReport, FitEquation, Session, load_project, save_project

    s = Session()
    s.objects["data"] = data
    s.run(FitEquation("c ~ y + c[-1]", name="cons"))
    path = (tmp_path / "dialog.md").as_posix()
    s.run(ExportReport("cons", path=path, title="Equations", template="estimation_summary", theme="policy", start="2010Q1"))
    code = "report = Report.from_template('estimation_summary', {'cons': cons}, title='Equations', theme='policy', span=('2010Q1', None))"
    assert code in s.script()
    first = (tmp_path / "dialog.md").read_text(encoding="utf-8")
    reopened = load_project(save_project(s, tmp_path / "p.pymx"))
    assert reopened.script() == s.script()
    namespace = {"data": data}
    exec(s.script(), namespace)  # the generated code writes the same report
    assert (tmp_path / "dialog.md").read_text(encoding="utf-8") == first
    assert ExportReport("cons", template="forecast_round", all_sections=True).validate(s) == ["choose a template or all sections, not both"]


def test_the_dialogs_offer_plugins_templates_and_added_themes(fit, data, tmp_path, monkeypatch):
    from pymetrix import Theme, plugins
    from pymetrix.commands import ExportReport, FitEquation, Session, form_spec
    from pymetrix.commands.figures import FigureOptions
    from pymetrix.report import ReportTemplate
    from pymetrix.themes import _THEMES

    group = plugins.registry("report_templates")
    monkeypatch.setattr(group, "_entries", dict(group._entries))  # the plugin's entry goes when the test ends
    plugins.register("report_templates", ReportTemplate("coefficients_only", "Just the estimates.", ("Summary",)), name="coefficients_only")
    monkeypatch.setitem(_THEMES, "house", Theme("house", digits=1))
    s = Session()
    s.objects["data"] = data
    s.run(FitEquation("c ~ y + c[-1]", name="cons"))
    specs = {spec.name: spec for spec in form_spec(ExportReport)}
    templates, report_themes = s.choices(specs["template"], {}), s.choices(specs["theme"], {})
    assert templates[:4] == ["", "estimation_summary", "forecast_round", "technical_appendix"]  # then plugins', as finance_summary
    assert templates[-1] == "coefficients_only" and templates == ["", *report_templates()]
    assert report_themes[:6] == ["", "default", "publication", "teaching", "policy", "monochrome"]
    assert report_themes[-1] == "house" and report_themes == ["", *themes()]
    figure_theme = next(spec for spec in form_spec(FigureOptions) if spec.name == "theme")
    assert s.choices(figure_theme, {}) == report_themes
    assert specs["template"].display("") == "(summary and graph)" and specs["theme"].display("") == "(the session's default)"
    path = (tmp_path / "plugin.md").as_posix()
    s.run(ExportReport("cons", path=path, template="coefficients_only", theme="house"))
    assert "report = Report.from_template('coefficients_only', {'cons': cons}, title='Report', theme='house')" in s.script()
    text = (tmp_path / "plugin.md").read_text(encoding="utf-8")
    assert f"| y | {fit.params['y']:.1f} |" in text and "Figure" not in text  # the summary alone (no graph), at the house theme's one decimal
