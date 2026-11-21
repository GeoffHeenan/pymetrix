"""Demo polish: table renderers, result fields, the data dictionary, demo projects, and the window affordances."""

import os

import numpy as np
import pytest

from pymetrix import Dataset, Equation, demo, demos
from pymetrix.output import Table
from pymetrix.views import dataset_status, fields, view


@pytest.fixture(scope="module")
def data():
    return Dataset.example("sample_macro")


def test_tables_render_as_text_without_reading_display_text():
    t = Table(["coef", "label"], [[1.234567, "a|b"], [float("nan"), "c"]], index=["x", "y"], index_name="term", notes=["a note"])
    assert t.to_csv() == "term,coef,label\nx,1.234567,a|b\ny,,c\n"  # full precision; missing values blank
    assert t.to_csv(sep="\t", header=False) == "x\t1.234567\ta|b\ny\t\tc\n"
    md = t.to_markdown(digits=2).splitlines()
    assert md[:4] == ["| term | coef | label |", "|---|---:|---|", "| x | 1.23 | a\\|b |", "| y |  | c |"]
    assert md[-1] == "*a note*  "
    repeated = Table(["coef", "coef"], [[1.0, 2.0]], index=["x"])  # e.g. a VAR's equations side by side
    assert repeated.to_markdown().splitlines()[1] == "|---|---:|---:|"


def test_fields_list_what_a_result_holds(data):
    fit = Equation("c ~ y + c[-1]").fit(data)
    t = fields(fit, "cons", code="cons = Equation('c ~ y + c[-1]').fit(data)")
    rows = dict(zip(t.index, t.rows))
    assert rows["cons.params"][:2] == ["PointEstimate", "3"] and "c[-1]" in rows["cons.params"][2]
    assert rows["cons.provenance.warnings"][0] == "tuple" and rows["cons.provenance.nobs"][2] == "99"
    assert rows["cons.resid"][:2] == ["Series", "99"]
    assert t.notes == ["Made by: cons = Equation('c ~ y + c[-1]').fit(data)"]
    for python in t.index:  # every row is Python that works
        assert eval(python, {"cons": fit}) is not None or python.endswith(("seed", "random_generator", "converged", "weights", "resolved", "sample_definition", "options", "backend_result"))


def test_data_dictionary_and_status(data):
    gappy = data.with_series(z=data["y"].where(data["y"].index.year != 2010))
    status = dataset_status(gappy)
    assert status == "5 series · quarterly · 2000Q1 to 2024Q4 · 100 periods · 4 missing values"
    meta = view(gappy, "Variables").to_frame()
    assert meta.loc["z", ["observations", "first", "last", "gaps"]].tolist() == [96, "2000Q1", "2024Q4", 4]
    assert view(gappy, "Data").title == status


def test_demo_projects(tmp_path):
    from pymetrix.commands import load_project, save_project

    assert list(demos()) == ["forecast_round", "consumption", "nk_estimation", "policy_and_nonlinear"]
    s = demo("forecast_round")
    assert sorted(s.objects) == ["data", "fc", "forecast_round", "var"] and len(s["forecast_round"].adjustments) == 2
    assert "forecast_round.adjust('ly', {'2025Q1': -0.004, '2025Q2': -0.002}" in s.script()
    reopened = load_project(save_project(s, tmp_path / "demo.pymx"))
    assert reopened.script() == s.script()
    assert np.allclose(reopened["forecast_round"].paths["ly"], s["forecast_round"].paths["ly"])
    assert demo("consumption")["dynamic"].params.names == ("const", "y", "c[-1]")
    with pytest.raises(KeyError, match="no demo 'x'"):
        demo("x")


def test_the_modelling_demos():
    s = demo("nk_estimation")
    assert {"nk", "data", "estimated", "posterior"} <= set(s.objects)
    post, mode = s["posterior"], np.asarray(s["estimated"].params.values, dtype=float)
    assert post.draws.shape == (2, 1000, 3)
    assert np.all(np.abs(post.draws.mean(axis=(0, 1)) - mode) < 0.5 * post.draws.std(axis=(0, 1)))  # posterior near the mode
    p = demo("policy_and_nonlinear")
    assert p["commitment"].loss() < p["discretion"].loss()  # commitment does better than discretion
    assert p["best_rule"].converged and p["best_rule"].loss > p["commitment"].loss()  # a simple rule cannot beat commitment
    floor = p["nk_zlb"].model.params["i_floor"]
    assert p["zlb_episode"].frame["i"].min() == pytest.approx(floor) and p["zlb_episode"].binding.to_numpy().any()
    assert p["runs"].points == ((1.2,), (1.5,), (2.0,), (3.0,))


@pytest.fixture
def window():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("qtconsole")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import main_window

    w = main_window(demo("consumption"))
    w.show()
    yield w
    w.state["session"].dirty = False
    w.close()


def test_result_windows_copy_and_fields(window):
    window.show_object("dynamic")
    tabs = window.mdi.activeSubWindow().widget()
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    assert titles[-1] == "Python access" and "Provenance" in titles
    texts = tabs.copy_texts["Coefficients"]
    assert texts["python"] == "from pymetrix.views import view\nview(dynamic, 'Coefficients').to_frame()"
    assert texts["markdown"].startswith("| ") and texts["values"].count("\n") == 3 and "\t" in texts["excel"]
    assert "Made by: dynamic = Equation('c ~ y + c[-1]').fit(data, method=OLS())" in tabs.copy_texts["Python access"]["markdown"]
    window.show_object("data")
    data_tabs = window.mdi.activeSubWindow().widget()
    assert "Python access" not in [data_tabs.tabText(i) for i in range(data_tabs.count())]  # data, not a result


def test_filter_zoom_and_demo_menu(window, monkeypatch):
    window.object_filter.setText("fitted")
    visible = [window.objects.topLevelItem(i).text(0) for i in range(window.objects.topLevelItemCount())
               if not window.objects.topLevelItem(i).isHidden()]
    assert visible == ["dynamic", "static"]  # matched by type
    window.object_filter.setText("")
    window.show_object("dynamic")
    sub = window.mdi.activeSubWindow()
    before = sub.widget().font().pointSizeF()
    window.zoom(1)
    window.zoom(1)
    assert sub.widget().font().pointSizeF() == pytest.approx(before * 1.3)
    window.zoom(0)
    assert sub.widget().font().pointSizeF() == pytest.approx(before)
    from pymetrix.ui import demo_chooser

    monkeypatch.setattr(demo_chooser, "choose_demo", lambda parent=None: demo_chooser.demo_entries()[0][0])  # the first demo
    window.state["session"].dirty = False
    window.open_demo()
    assert "forecast_round" in window.state["session"].objects


def test_the_app_command_line():
    pytest.importorskip("PySide6")
    from pymetrix.ui.qt import _arguments

    args = _arguments(["--demo", "forecast_round", "--maximized", "-platform", "offscreen"])  # Qt's own options pass through
    assert (args.demo, args.maximized, args.project) == ("forecast_round", True, None)
    assert _arguments(["work.pymx"]).project == "work.pymx"


def test_the_codespace_configuration():
    import json
    import re
    from pathlib import Path

    root = Path(__file__).parents[1]
    config = json.loads(re.sub(r"(?m)^\s*//.*$", "", (root / ".devcontainer" / "devcontainer.json").read_text(encoding="utf-8")))
    assert config["forwardPorts"] == [6080] and "desktop-lite" in next(iter(config["features"]))
    for script in ("setup.sh", "start-app.sh"):
        text = (root / ".devcontainer" / script).read_bytes()
        assert b"\r\n" not in text, f"{script} must have LF line endings to run on Linux"
    assert "--demo" in (root / ".devcontainer" / "start-app.sh").read_text(encoding="utf-8")


def test_calculation_menu_frozen_and_not_computed_results(window):
    session = window.state["session"]
    items = lambda: {window.objects.topLevelItem(i).text(0): window.objects.topLevelItem(i).text(2)  # noqa: E731
                     for i in range(window.objects.topLevelItemCount())}
    labels = [a.text().replace("&", "") for a in window.calculate_menu.actions() if a.text()]
    assert labels[:3] == ["Recalculate automatically", "Recalculate on request (F9)", "Recalculate"] and "Clean up history…" in labels
    assert window.calculate_menu.actions()[1].isChecked()  # Manual, as sessions start
    session.run_code("data = data.with_series(c=data['c'] * 1.01)")
    window.refresh()
    assert items()["dynamic"] == "out of date"
    session.freeze("dynamic")
    window.refresh()
    assert items()["dynamic"] == "frozen; inputs changed"
    session.unfreeze("dynamic")
    window.calculate_menu.actions()[0].trigger()  # Automatic: recomputed at once
    assert session.mode == "automatic" and items()["dynamic"] == ""
    window.show_object("data")
    tabs = window.mdi.activeSubWindow().widget()
    assert tabs.tabText(0) == "Overview"
    assert "name\tdata" in tabs.copy_texts["Overview"]["excel"]
