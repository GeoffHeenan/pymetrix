"""Editing data: Dataset.edit/rename/drop/with_info, the steps that record them, and the spreadsheet window."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymetrix import HP, Dataset
from pymetrix.commands import (
    DeriveSeries,
    DropSeries,
    EditData,
    FitEquation,
    FreezeSeries,
    GenerateSeries,
    RenameSeries,
    SeriesInfo,
    Session,
    load_project,
    save_project,
)

DATA = Path(__file__).parents[1] / "examples" / "sample_macro.parquet"


@pytest.fixture
def data():
    return Dataset.read(str(DATA)).derive(share="c / y").filter("c", HP())


@pytest.fixture
def panel():
    rows = [{"id": k, "t": p, "y": float(i + (10 if k == "b" else 0))} for k in "ab" for i, p in enumerate(pd.period_range("2000Q1", periods=6, freq="Q"))]
    return Dataset.from_frame(pd.DataFrame(rows), time="t", entity="id").derive(dy="y - y[-1]")


def test_edit_flags_cells_and_recomputes(data, tmp_path):
    edited = data.edit({"c": {"2000Q1": 500.0}, "g": {"2000Q2": None}})
    assert edited["c"].loc["2000Q1"] == 500.0 and np.isnan(edited["g"].loc["2000Q2"])
    assert edited.flag("c").loc["2000Q1"] == "edited" and edited.flag("c").loc["2000Q2"] == "" and edited.flag("y").loc["2000Q1"] == ""
    assert edited["share"].loc["2000Q1"] == pytest.approx(500.0 / data["y"].loc["2000Q1"])
    assert edited.variables["c"].vintage == data.variables["c"].vintage  # an edit is not a new release
    with pytest.raises(ValueError, match="share is derived"):
        data.edit({"share": {"2000Q1": 1.0}})
    with pytest.raises(ValueError, match="c_trend .* could not be recomputed"):
        data.edit({"c": {"2000Q2": None}})  # a gap inside the sample: the HP trend cannot follow
    with pytest.raises(KeyError, match="no row"):
        data.edit({"c": {"1800Q1": 1.0}})
    path = tmp_path / "edited.parquet"
    edited.write(str(path))
    assert Dataset.read(str(path)).flag("c").loc["2000Q1"] == "edited"


def test_panel_rows_are_keyed_by_entity_and_period(panel):
    edited = panel.edit({"y": {("a", "2000Q3"): 100.0}})
    assert edited.frame.loc[("a", pd.Period("2000Q3", "Q")), "dy"] == 100.0 - 1.0
    assert edited.frame.loc[("b", pd.Period("2000Q3", "Q")), "y"] == 12.0
    with pytest.raises(KeyError, match="one value per axis"):
        panel.edit({"y": {"2000Q3": 1.0}})


def test_rename_drop_and_info(data):
    renamed = data.rename(c="cons")
    assert "cons" in renamed and "c" not in renamed
    assert renamed.derived["share"].expr == "cons/y" and renamed.derived["c_trend"].of == "cons"
    assert np.allclose(renamed.with_series(cons=renamed["cons"] * 2)["share"], data["share"] * 2)
    with pytest.raises(ValueError, match="already a series"):
        data.rename(c="y")
    with pytest.raises(ValueError, match="derived from c"):
        data.drop("c")
    assert data.drop("g").names == ["c", "i", "y", "share", "c_trend", "c_cycle"]
    described = data.with_info("c", units="USD bn", measure="flow", scale=9, label="Consumption")
    var = described.variables["c"]
    assert (var.units, var.measure.value, var.scale, var.label) == ("USD bn", "flow", 9, "Consumption")
    with pytest.raises(ValueError, match="cannot set recipe"):
        data.with_info("c", recipe=None)


def test_steps_record_replay_and_save(data, tmp_path):
    s = Session()
    s.objects["data"] = data
    s.run(FitEquation("share ~ g", name="fit"))
    s.run(EditData("data", json.dumps([["c", ["2000Q1"], 500.0], ["g", ["2000Q3"], None]])))
    s.run(RenameSeries("data", "i", "inv"))
    s.run(SeriesInfo("data", "c", json.dumps({"units": "USD bn"})))
    s.run(GenerateSeries("lc", "log(c)"))
    s.run(GenerateSeries("dlc", "dlog(c)", keep_updated=True))
    s.run(FreezeSeries("data", "share"))
    s.run(DropSeries("data", "lc"))
    script = s.script()
    assert "data = data.edit({'c': {'2000Q1': 500.0}, 'g': {'2000Q3': None}})" in script
    assert "data = data.rename(i='inv')" in script and "data = data.with_info('c', units='USD bn')" in script
    assert "data = data.with_series(lc='log(c)')" in script and "data = data.derive(dlc='dlog(c)')" in script
    assert "data = data.freeze('share')" in script and "data = data.drop('lc')" in script
    assert s.stale() == {"fit": "g, share in data have changed since fit was computed"}
    reopened = load_project(save_project(s, tmp_path / "edits.pymx"))
    assert reopened["data"].names == s["data"].names and reopened["data"].flag("c").loc["2000Q1"] == "edited"
    ns = {"data": data}
    exec(script, ns)
    assert np.allclose(ns["data"]["c"], s["data"]["c"])


QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _sheet(session, name="data"):
    from pymetrix.ui.sheet_qt import SheetWindow

    def run(cmd):
        try:
            session.run(cmd)
        except Exception:
            return False
        sheet.sync()
        return True

    sheet = SheetWindow(name, lambda: session[name], run)
    return sheet


def test_sheet_edits_preview_undo_and_apply(app, data):
    from PySide6.QtCore import Qt

    s = Session()
    s.objects["data"] = data
    sheet = _sheet(s)
    m = sheet.model
    c, share = m.columns.index("c"), m.columns.index("share")
    assert not m.flags(m.index(0, share)) & Qt.ItemFlag.ItemIsEditable and m.flags(m.index(0, c)) & Qt.ItemFlag.ItemIsEditable
    sheet.set_cells({(0, c): "500"}, "Edit cell")
    assert m.pending and m.data(m.index(0, share)) == f"{500 / data['y'].iloc[0]:.4f}"  # derived values preview the edit
    sheet.undo.undo()
    assert not m.pending and m.data(m.index(0, share)) == f"{data['share'].iloc[0]:.4f}"
    sheet.undo.redo()
    assert sheet.set_cells({(0, share): "1"}, "Edit cell") == ["share is derived; freeze it to edit"]
    sheet.set_cells({(0, c): repr(float(data["c"].iloc[0]))}, "Edit cell")  # typing the original value back clears the edit
    assert not m.pending
    sheet.table.setCurrentIndex(m.index(1, c))
    assert sheet.paste("1,234.5\tNA\n2") == []
    assert len(m.pending) == 3 and m.pending[("i", m.label(1))] is None
    assert sheet.apply() and not m.pending and s.history[-1].command.__class__.__name__ == "EditData"
    assert s["data"]["c"].iloc[1] == 1234.5 and s["data"].flag("c").iloc[1] == "edited"
    sheet.table.selectionModel().select(m.index(1, c), sheet.table.selectionModel().SelectionFlag.Select)
    assert sheet.copy() == "1234.5"


def test_sheet_follows_and_protects_pending_edits(app, data):
    s = Session()
    s.objects["data"] = data
    sheet = _sheet(s)
    sheet.set_cells({(0, 0): "500"}, "Edit cell")
    s.run(GenerateSeries("z", "c * 2"))  # the dataset changes elsewhere while edits are pending
    sheet.sync()
    assert not sheet.banner.isHidden() and sheet.model.pending and "z" in sheet.model.columns
    assert sheet.apply() and s["data"]["c"].iloc[0] == 500.0 and "z" in s["data"]
    s.objects.pop("data")
    assert not sheet.sync()


def test_sheet_panels_and_series_info(app, panel):
    s = Session()
    s.objects["data"] = panel
    sheet = _sheet(s)
    assert sheet.entity_box is not None and sheet.model.rowCount() == 12
    sheet.entity_box.setCurrentText("b")
    assert sheet.model.rowCount() == 6 and sheet.model.headerData(0, __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.Orientation.Vertical) == "2000Q1"
    sheet.set_cells({(2, 0): "50"}, "Edit cell")
    assert sheet.apply() and s["data"].frame.loc[("b", pd.Period("2000Q3", "Q")), "y"] == 50.0
    sheet.info.show_series("y")
    sheet.info.editors["units"].setText("index")
    sheet.info.editors["measure"].setCurrentIndex(sheet.info.editors["measure"].findData("index"))  # shown as "Index"
    assert sheet.info.changes() == {"units": "index", "measure": "index"}
    sheet.info.save()
    assert s["data"].variables["y"].units == "index" and s.history[-1].code == "data = data.with_info('y', units='index', measure='index')"


def test_the_edit_line_shows_and_changes_the_selected_cell(app, data):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    s = Session()
    s.objects["data"] = data
    sheet = _sheet(s)
    m, line = sheet.model, sheet.edit_line
    assert not line.line.isEnabled() and line.where.text() == ""  # no cell yet
    c, share = m.columns.index("c"), m.columns.index("share")
    sheet.table.setCurrentIndex(m.index(2, c))
    assert line.line.text() == repr(float(data["c"].iloc[2])) and line.where.text() == f"c · {m.label(2)}"  # in full
    assert not line.accept_button.isEnabled() and not line.cancel_button.isEnabled()
    QTest.keyClicks(line.line, "x")  # typing: ✓ and ✗ come on
    assert line.editing() and line.accept_button.isEnabled() and line.cancel_button.isEnabled()
    line.cancel_button.click()  # ✗: the cell's value back
    assert line.line.text() == repr(float(data["c"].iloc[2])) and not m.pending
    line.line.setText("abc")
    line.line.textEdited.emit("abc")
    assert not line.accept() and "not a number" in line.line.toolTip() and not m.pending  # left for correcting
    line.line.setText("512.25")
    line.line.textEdited.emit("512.25")
    line.accept_button.click()  # ✓: a pending edit, as typing in the cell makes
    assert m.pending[("c", m.label(2))] == 512.25 and sheet.table.currentIndex().row() == 2
    assert m.data(m.index(2, share)) == f"{512.25 / data['y'].iloc[2]:.4f}"  # derived series preview it
    sheet.undo.undo()  # the line follows the data
    assert not m.pending and line.line.text() == repr(float(data["c"].iloc[2]))
    line.line.setText("600")
    line.line.textEdited.emit("600")
    QTest.keyClick(line.line, Qt.Key.Key_Return)  # Enter accepts and moves down, as in a spreadsheet
    assert m.pending[("c", m.label(2))] == 600.0 and sheet.table.currentIndex().row() == 3
    assert line.line.text() == repr(float(data["c"].iloc[3]))
    QTest.keyClicks(line.line, "9")
    QTest.keyClick(line.line, Qt.Key.Key_Escape)  # Esc puts it back
    assert line.line.text() == repr(float(data["c"].iloc[3]))
    sheet.table.setCurrentIndex(m.index(0, share))  # a derived series: shown, not editable
    assert line.line.isReadOnly() and "freeze it to edit" in line.line.toolTip()
    sheet.table.setCurrentIndex(m.index(4, m.columns.index("i")))  # a series nothing is derived from
    line.line.setText("")  # empty: a missing value
    line.line.textEdited.emit("")
    assert sheet.apply()  # Apply takes the value in the line too
    assert np.isnan(s["data"]["i"].iloc[4]) and s["data"]["c"].iloc[2] == 600.0
