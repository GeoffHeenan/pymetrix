"""Selections and loops, show(), statement-by-statement runs, and d()/dlog() regressors."""

import os
from pathlib import Path

import numpy as np
import pytest

import pymetrix
from pymetrix import HP, OLS, Dataset, Equation, Results, Selection, show
from pymetrix.commands import DeriveEach, FitEach, SelectSeries, Session, TestEach, load_project, save_project
from pymetrix.output import quiet_show, set_show_handler

DATA = Path(__file__).parents[1] / "examples" / "sample_macro.parquet"


@pytest.fixture
def data():
    return Dataset.read(str(DATA))


def test_selections_combine_and_template(data):
    d = data.filter(["c", "y"], HP()).with_info("c", units="USD bn").with_info("i", units="USD bn")
    assert list(d.series("c", "i", "g")) == ["c", "i", "g"]
    assert list(d.series("*_cycle")) == ["c_cycle", "y_cycle"]
    assert list(d.series(units="USD bn")) == ["c", "i"]
    assert list(d.series(derived=True)) == ["c_trend", "c_cycle", "y_trend", "y_cycle"]
    assert list(d.series(where=lambda v: v.name.endswith("trend"))) == ["c_trend", "y_trend"]
    with pytest.raises(KeyError, match="no series 'nope'"):
        d.series("nope")
    g = d.series("c", "i")
    assert list(g + ["g", "c"]) == ["c", "i", "g"] and list((g + ["g"]) - ["i"]) == ["c", "g"]
    assert list(g.format("d{s}")) == ["dc", "di"] and g.formulas("d{s}", "dlog({s})") == {"dc": "dlog(c)", "di": "dlog(i)"}
    assert [s for s in g] == ["c", "i"] and "c" in g and g[0] == "c" and isinstance(g[:1], Selection)


def test_loops_return_results_side_by_side(data):
    g = data.series("c", "i", "g")
    fits = g.fit("dlog({s}) ~ dlog({s})[-1] + dlog(y)", data, method=OLS(cov="white"))
    assert isinstance(fits, Results) and list(fits) == ["c", "i", "g"]
    single = Equation("dlog(i) ~ dlog(i)[-1] + dlog(y)").fit(data, method=OLS(cov="white"))
    assert np.allclose(fits["i"].params.values, single.params.values)
    table = fits.coefficients().to_frame()
    assert list(table.columns)[-3:] == ["c", "i", "g"]
    assert "dlog({s})[-1]" in str(fits.coefficients()) and "dlog(y)" in str(fits.coefficients())  # rows line up
    assert "r2" in str(fits.stats()) and fits.summary().title == "Coefficients"
    tests = g.test(data, "adf", lags=2)
    assert "ADF" in tests.tests().title and np.isclose(tests["c"].statistic, data.test("adf", "c", lags=2).statistic)
    means = (g + ["nope"]).map(lambda s: float(data[s].mean()), errors="collect")
    assert list(means) == ["c", "i", "g"] and "nope" in means.failed and "nope" in str(means.failures())
    with pytest.raises(KeyError):
        (g + ["nope"]).map(lambda s: data[s])


def test_selection_dialogs_record_replay_and_save(data, tmp_path):
    s = Session()
    s.objects["data"] = data
    s.run(SelectSeries(series="c, i", patterns="g*", name="spend"))
    s.run(DeriveEach("d{s}", "dlog({s})", selection="spend"))
    s.run(FitEach("d{s} ~ d{s}[-1]", selection="spend", cov="hac", start="2001Q1"))
    s.run(TestEach("adf", selection="spend", options="lags=2"))
    script = s.script()
    assert "spend = data.series('c', 'i', 'g*')" in script
    assert "data = data.derive(dc='dlog(c)', di='dlog(i)', dg='dlog(g)')" in script  # the recipes, written out
    assert "fits = spend.fit('d{s} ~ d{s}[-1]', data, method=OLS(cov='hac'), span=('2001Q1', None))" in script
    assert "tests = spend.test(data, 'adf', lags=2)" in script
    assert set(s["data"].derived) == {"dc", "di", "dg"}
    reopened = load_project(save_project(s, tmp_path / "selections.pymx"))
    assert np.allclose(reopened["fits"]["g"].params.values, s["fits"]["g"].params.values)
    ns = {"data": data}
    exec(script, ns)
    assert np.allclose(ns["fits"]["c"].params.values, s["fits"]["c"].params.values)
    assert FitEach("dc ~ dc[-1]", selection="spend").validate(s) == ["the formula needs {s} where the series goes"]
    assert DeriveEach("dx", "dlog({s})", selection="spend").validate(s)[0].startswith("the name template needs {s}")
    # or tick the series in the dialog itself, without making a selection first
    s.run(TestEach("adf", series="c, i", name="ticked"))
    assert "ticked = data.series('c', 'i').test(data, 'adf')" in s.script() and list(s["ticked"]) == ["c", "i"]
    s.run(FitEach("{s} ~ y", series="c, g", name="fits2"))
    assert "fits2 = data.series('c', 'g').fit('{s} ~ y', data, method=OLS())" in s.script()
    assert TestEach("adf").validate(s) == ["tick the series to loop over, or choose a selection"]
    assert TestEach("adf", series="c", selection="spend").validate(s) == ["tick series or choose a selection, not both"]


def test_show_prints_routes_and_stays_quiet(data, capsys, monkeypatch):
    # a console opened by an earlier test leaves an IPython session behind; a plain script has none
    monkeypatch.setattr("IPython.get_ipython", lambda: None)
    fit = Equation("c ~ y").fit(data)
    show(fit)
    assert "const" in capsys.readouterr().out  # no app, no notebook: the summary is printed
    seen = []
    previous = set_show_handler(lambda obj, title: seen.append((obj, title)))
    try:
        show(fit, "my fit")
        with quiet_show():
            show(fit)
        assert seen == [(fit, "my fit")]
    finally:
        set_show_handler(previous)
    assert pymetrix.show is show


def test_show_is_quiet_while_replaying(data, tmp_path):
    s = Session()
    s.objects["data"] = data
    s.run_code("fit = Equation('c ~ y').fit(data)\nshow(fit)")
    seen = []
    previous = set_show_handler(lambda obj, title: seen.append(obj))
    try:
        load_project(save_project(s, tmp_path / "shows.pymx"))
        s.objects["data"] = data.with_series(y=data["y"] * 2)
        s.recompute()
    finally:
        set_show_handler(previous)
    assert seen == []


def test_statements_split_scripts_into_steps():
    from pymetrix.ui.editor_qt import statements  # needs PySide6

    code = "a = 1\n\n# a comment\nfor s in 'ab':\n    print(s)\n@staticmethod\ndef f():\n    return 1\nb = (2 +\n     3)\n"
    parts = statements(code, first=10)
    assert [line for _, line in parts] == [10, 13, 15, 18]
    assert parts[1][0] == "for s in 'ab':\n    print(s)" and parts[2][0].startswith("@staticmethod") and parts[3][0] == "b = (2 +\n     3)"
    assert statements("x = (", 1) == [("x = (", 1)]  # does not parse: one piece, for Python's error


def test_difference_functions_as_regressors(data):
    eq = Equation("dlog(c) ~ dlog(y) + dlog(c)[-1] + (g - i)")
    assert [t.label for t in eq.terms] == ["dlog(y)", "dlog(c)[-1]", "g - i"]
    fit = eq.fit(data)
    lagged = data.with_series(dc="dlog(c)", dy="dlog(y)", gi="g - i").with_series(dc1="dc[-1]")
    ref = Equation("dc ~ dy + dc1 + gi").fit(lagged)
    assert np.allclose(fit.params.values, ref.params.values)
    with pytest.raises(Exception, match="only '- 1'"):
        Equation("c ~ y - g")


QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_code_window_runs_one_step_per_statement(data):
    pytest.importorskip("qtconsole")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import main_window

    s = Session()
    s.objects["data"] = data
    w = main_window(s)
    w.show()
    try:
        editor = w.open_editor(text="g = data.series('c', 'i')\nfor x in g:\n    pass\nfits = g.fit('{s} ~ {s}[-1]', data)\nshow(fits)\nbad = 1 / 0\nnever = 1\n")
        editor.run_all()
        for _ in range(100):
            app.processEvents()
        steps = [(r.code.splitlines()[0], r.ok) for r in s.history]
        assert steps == [("g = data.series('c', 'i')", True), ("for x in g:", True), ("fits = g.fit('{s} ~ {s}[-1]', data)", True),
                         ("show(fits)", True), ("bad = 1 / 0", False)]
        assert "never" not in s.objects and editor.editor.error_line == 6
        assert any("fits" in sub.windowTitle() for sub in w.mdi.subWindowList())  # show() opened its window
    finally:
        editor.editor.document().setModified(False)
        s.dirty = False
        w.close()


def test_old_names_still_work(data, tmp_path):
    """data.group() and projects saved with DefineGroup and group= fields still run."""
    import json
    import zipfile

    assert list(data.group("c", "i")) == list(data.series("c", "i"))
    s = Session()
    s.objects["data"] = data
    s.run(SelectSeries(series="c, i", name="spend"))
    s.run(FitEach("{s} ~ y", selection="spend"))
    path = save_project(s, tmp_path / "new.pymx")
    old = tmp_path / "old.pymx"
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(old, "w") as dst:
        for item in src.namelist():
            content = src.read(item)
            if item == "commands.json":
                steps = json.loads(content)
                for step in steps:
                    step.pop("id"), step.pop("version")  # format 1: steps by class name only
                    if step["command"] == "SelectSeries":
                        step["command"] = "DefineGroup"
                    if "selection" in step["fields"]:
                        step["fields"]["group"] = step["fields"].pop("selection")
                content = json.dumps(steps).encode()
            dst.writestr(item, content)
    reopened = load_project(old)
    assert np.allclose(reopened["fits"]["i"].params.values, s["fits"]["i"].params.values)
