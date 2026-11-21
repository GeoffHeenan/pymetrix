"""Projects, adding results to datasets, structured result views, and the desktop window's new parts."""

import importlib.util
import os
import shutil
import zipfile

import numpy as np
import pytest

from pymetrix import Dataset
from pymetrix.commands import (
    AddToDataset,
    FitEquation,
    FitVAR,
    ImpulseResponse,
    LoadData,
    RunTest,
    Session,
    load_project,
    save_project,
)
from pymetrix.ui.render import sections


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "macro.parquet"
    shutil.copy("examples/sample_macro.parquet", path)
    return path


def test_project_round_trip_survives_a_deleted_source(source, tmp_path):
    s = Session()
    s.run(LoadData(str(source)))
    s.run(FitEquation("c ~ y + c[-1]", name="cons"))
    s.run(RunTest(source="cons", test="breusch_godfrey", options="lags=2"))
    s.run(AddToDataset("cons", "resid", series_name="u"))
    s.run(FitVAR("c, i, y", lags="1"))
    s.run(ImpulseResponse("var", periods=6, draws=50))
    path = save_project(s, tmp_path / "demo")
    assert path.suffix == ".pymx" and not s.dirty
    with zipfile.ZipFile(path) as z:
        assert {"manifest.json", "commands.json", "session.py", "data/data.parquet"} <= set(z.namelist())
    source.unlink()  # the project carries its own copy of the data
    s2 = load_project(path)
    assert sorted(s2.objects) == sorted(s.objects) and s2.path == path and not s2.dirty
    assert "u" in s2["data"] and np.allclose(s2["data"]["u"], s["data"]["u"], equal_nan=True)  # first period: no lag
    assert np.allclose(s2["cons"].params.values, s["cons"].params.values)
    assert np.isclose(s2["test"].statistic, s["test"].statistic)
    assert np.allclose(s2["irf"].upper, s["irf"].upper)  # bootstrap bands reproduce: fixed seed
    assert s2.script() == s.script()


def test_datasets_placed_directly_are_saved_as_they_started(ts, tmp_path):
    s = Session()
    s.objects["data"] = ts
    s.run(FitEquation("y ~ x + z", name="fit"))
    s.run(AddToDataset("fit", "fitted", series_name="yhat"))
    path = save_project(s, tmp_path / "placed.pymx")
    s2 = load_project(path)
    assert "yhat" in s2["data"] and np.allclose(s2["fit"].params.values, s["fit"].params.values)
    assert "yhat" not in s2.initial["data"]  # the embedded copy is the dataset before the commands ran


def test_add_results_items_and_code(ts):
    s = Session()
    s.objects["data"] = ts
    s.run(FitEquation("y ~ x + z", name="fit"))
    from pymetrix.commands.session import result_items

    assert result_items(s["fit"]) == ["resid", "fitted"]
    assert result_items(ts)[:2] == ["column:y", "column:x"]
    s.run(AddToDataset("fit", "resid"))
    assert "data = data.with_series(fit_resid=fit.resid)" in s.script()
    assert np.allclose(s["data"]["fit_resid"], s["fit"].resid)


def test_structured_sections(ts):
    from pymetrix import VAR, Equation

    fit = Equation("y ~ x + z").fit(ts)
    titles = [t for t, _ in sections(fit)]
    graph = ["Graph"] if importlib.util.find_spec("matplotlib") else []  # graphs need matplotlib
    qq = ["Q-Q plot"] if graph else []
    correlogram = ["Correlogram", *(["Correlogram graph"] if graph else [])]  # a time series' residuals
    assert titles == ["Summary", *graph, "Equation", "Coefficients", "Covariance", "Statistics", "Residuals", "Diagnostics", *qq,
                      *correlogram, "Provenance"]
    coefs = dict(sections(fit))["Coefficients"].to_frame()
    assert np.allclose(coefs["coef"], fit.params.values)
    assert [t for t, _ in sections(ts)] == ["Data", *graph, "Variables"]
    var = VAR(["y", "x"], lags=1).fit(ts)
    assert [t for t, _ in sections(var.irf(4, draws=0))] == [*graph, *(["Heatmap"] if graph else []), "Selected horizons", "All horizons"]
    assert dict(sections(fit.test("jarque_bera")))["Result"].rows[0][0] == "statistic"


def test_window_tabs_window_picker_and_file_picker(ts):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import command_dialog, main_window

    s = Session()
    s.objects["data"] = ts
    s.run(FitEquation("y ~ x + z", name="fit"))
    w = main_window(s)
    w.show_object("data")
    w.show_object("fit")
    tabs = w.mdi.activeSubWindow().widget()
    assert tabs.tabText(0) == "Summary" and "Coefficients" in tabs.tables
    w.list_windows()
    listed = [a.text() for a in w.view_menu.actions() if a.isCheckable() and " — " in a.text()]
    assert listed == ["data — Dataset", "fit — Fitted equation"]
    assert "untitled *" in w.windowTitle()
    dialog = command_dialog(LoadData, s)
    dialog.editors["path"].path.setText("examples/sample_macro.parquet")
    choices = [dialog.editors["time"].widget.itemText(i) for i in range(dialog.editors["time"].widget.count())]
    assert choices[:2] == ["(none)", "time"]  # a blank choice says so and dialog.editors["path"].browse.text() == "Browse…"
    w.switch(Session())
    assert not w.mdi.subWindowList()
    w.close()  # also hands show() back to printing
