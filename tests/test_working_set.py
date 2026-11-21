"""Large datasets in the app: a working set limits the series shown, and every series list can be searched."""

import os

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset
from pymetrix.commands import PlotSeries, SelectSeries, Session, form_spec, load_project, save_project
from pymetrix.errors import SpecificationError


@pytest.fixture
def wide():
    """A dataset of 400 series: gdp_, cpi_ and u_ for many countries."""
    index = pd.period_range("2000Q1", periods=20, freq="Q")
    rng = np.random.default_rng(0)
    columns = {f"{kind}_{k:03d}": rng.normal(size=20) for kind in ("gdp", "cpi") for k in range(150)}
    columns.update({f"u_{k:03d}": rng.normal(size=20) for k in range(100)})
    return Dataset.from_frame(pd.DataFrame(columns, index=index))


def _series_field(cls, name="series"):
    return next(f for f in form_spec(cls) if f.name == name)


def test_a_working_set_limits_the_series_offered(wide, tmp_path):
    s = Session()
    s.objects["data"] = wide
    field = _series_field(PlotSeries)
    assert len(s.choices(field, {"data": "data"})) == 400
    shown = s.set_working_set("data", "gdp_00*, u_001")
    assert shown == [f"gdp_{k:03d}" for k in range(10)] + ["u_001"]
    assert s.choices(field, {"data": "data"}) == shown  # every series list: the working set's
    assert len(s.choices(field, {"data": "data"}, scoped=False)) == 400  # Show all
    s.run(SelectSeries("cpi_01*", name="prices"))
    assert s.set_working_set("data", "prices") == [f"cpi_{k:03d}" for k in range(10, 20)]  # or a selection
    with pytest.raises(SpecificationError, match="matches no series"):
        s.set_working_set("data", "nothing_*")
    assert s.working_sets["data"] == "prices"  # the old one kept
    reopened = load_project(save_project(s, tmp_path / "wide.pymx"))  # saved with the project
    assert reopened.working_sets == {"data": "prices"} and reopened.working_set("data") == s.working_set("data")
    s.run(PlotSeries("gdp_001", name="fig"))
    assert s.history[-1].code == "fig = data.plot('gdp_001')"  # commands name their series: scripts run the same
    s.set_working_set("data", None)
    assert s.working_set("data") is None and len(s.choices(field, {"data": "data"})) == 400
    s.working_sets["data"] = "gone"  # a selection since removed: every series, not an error
    assert s.working_set("data") is None


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_series_lists_filter_count_and_show_the_working_set(app, wide):
    from pymetrix.ui.qt import command_dialog

    s = Session()
    s.objects["data"] = wide
    s.set_working_set("data", "gdp_*")
    dialog = command_dialog(PlotSeries, s, initial=PlotSeries("gdp_001, u_005"))
    series = dialog.editors["series"]
    listed = [series.widget.item(i).text() for i in range(series.widget.count())]
    assert len(listed) == 151 and listed[-1] == "u_005"  # the working set, and a ticked series outside it kept
    assert series.scope_row.isVisibleTo(dialog) and series.scope_note.text() == "150 of 400 series (working set: gdp_*)"
    assert series.count.text() == "2 ticked of 151"
    series.search.setText("gdp_01")  # the filter
    visible = [series.widget.item(i).text() for i in range(series.widget.count()) if not series.widget.item(i).isHidden()]
    assert visible == [f"gdp_{k:03d}" for k in range(10, 20)] and series.count.text() == "10 shown, 2 ticked of 151"
    series.search.setText("")
    series.ticked_only.setChecked(True)  # what is ticked
    assert [series.widget.item(i).text() for i in range(series.widget.count()) if not series.widget.item(i).isHidden()] == ["gdp_001", "u_005"]
    series.scope_button.click()  # Show all
    assert series.widget.count() == 400 and series.scope_button.text() == "Working set only"
    assert series.get() == "gdp_001, u_005"  # the ticks kept throughout
    total = dialog.editors["total"]  # a single series: a dropdown that narrows as it is typed in
    assert total.widget.isEditable() and total.widget.completer().filterMode().name == "MatchContains"


def test_the_spreadsheet_and_the_dataset_window_show_the_working_set(app, wide, monkeypatch):
    from pymetrix.ui.qt import main_window
    from pymetrix.ui.sheet_qt import SheetWindow

    s = Session()
    s.objects["data"] = wide
    s.set_working_set("data", "u_*")
    sheet = SheetWindow("data", lambda: s["data"], lambda cmd: bool(s.run(cmd)) or True,
                        working_set=lambda: (s.working_set("data"), s.working_sets["data"]) if s.working_set("data") else None)
    assert len(sheet.model.columns) == 100 and sheet.status.text().strip() == "20 periods × 100 series of 400 (working set: u_*)"
    sheet.scope_action.setChecked(False)  # every series
    assert len(sheet.model.columns) == 400
    sheet.scope_action.setChecked(True)
    sheet.table.setCurrentIndex(sheet.model.index(0, 0))
    sheet.edit_line.line.setText("9")
    sheet.edit_line.line.textEdited.emit("9")
    assert sheet.edit_line.accept() and sheet.apply()  # editing works on the series shown
    assert s["data"]["u_000"].iloc[0] == 9.0

    window = main_window(s)
    try:
        from PySide6.QtWidgets import QLabel

        window.show_object("data")
        header = window.mdi.activeSubWindow().widget().header.text()
        assert header.strip().startswith("100 of 400 series (working set: u_*)")
        assert window.apply_working_set("data", None)  # cleared: the window shows every series again
        texts = [w.text() for w in window.findChildren(QLabel)]
        assert any("many series: right-click it in the workspace for a working set" in t for t in texts)  # and suggests one
        from PySide6.QtWidgets import QMessageBox

        said = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: said.append(a[2]))
        assert not window.apply_working_set("data", "zzz_*")  # a pattern that matches nothing: said, and nothing changed
        from pymetrix.ui.messages import titled

        # titled: on a Mac, which shows no message box title, "Working set" leads the text
        assert said == [titled("Working set", "'zzz_*' matches no series of data")[1]] and "data" not in s.working_sets
    finally:
        s.dirty = False
        window.close()  # nothing of this test's left for a later collection to finalize, maybe off the GUI thread
        sheet.close()
        window.close()
