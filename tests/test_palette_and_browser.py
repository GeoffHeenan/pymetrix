"""Find anything (the command palette) and the series browser of a dataset's window."""

import os

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset
from pymetrix.commands import Session
from pymetrix.ui.palette import PaletteEntry, rank


def _entries(log):
    titles = ["Data → Transform → Seasonal adjustment…", "Data → Sample → Set sample…", "Estimate → VARs and local projections → VAR…",
              "Estimate → VARs and local projections → VAR lag selection…", "Test → Run test…", "File → Save project"]
    out = [PaletteEntry(t, "Menu", lambda t=t: log.append(t)) for t in titles]
    out.append(PaletteEntry("data", "Object", lambda: log.append("open data"), "Dataset"))
    out.append(PaletteEntry("Seasonal adjustment (X-13)", "Help", lambda: log.append("help"), "Topics"))
    return out


def test_the_palette_finds_by_a_few_letters_of_any_words():
    entries = _entries([])
    assert [e.title for e in rank(entries, "seas adj")][:2] == ["Data → Transform → Seasonal adjustment…", "Seasonal adjustment (X-13)"]
    assert rank(entries, "var")[0].title == "Estimate → VARs and local projections → VAR…"  # the item named so, first
    assert rank(entries, "var lag")[0].title.endswith("VAR lag selection…")
    assert [e.kind for e in rank(entries, "data")][:1] == ["Object"]  # the object named data, before menus containing it
    assert rank(entries, "dataset")[0].title == "data"  # an object's type is searched too
    assert rank(entries, "zzz") == [] and len(rank(entries, "")) == len(entries)


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_the_palette_dialog_runs_what_is_chosen(app):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from pymetrix.ui.palette import palette_dialog

    log: list[str] = []
    dialog = palette_dialog(lambda: _entries(log))
    QTest.keyClicks(dialog.search, "set sam")
    assert dialog.found.count() == 1 and dialog.found.item(0).text().startswith("Data → Sample → Set sample…")
    QTest.keyClick(dialog.search, Qt.Key.Key_Return)
    assert log == ["Data → Sample → Set sample…"] and dialog.result() == dialog.DialogCode.Accepted
    dialog = palette_dialog(lambda: _entries(log))
    QTest.keyClicks(dialog.search, "var")
    QTest.keyClick(dialog.search, Qt.Key.Key_Down)  # the arrows choose another
    QTest.keyClick(dialog.search, Qt.Key.Key_Return)
    assert log[-1].endswith("VAR lag selection…")


@pytest.fixture
def wide():
    index = pd.period_range("2000Q1", periods=12, freq="Q")
    rng = np.random.default_rng(1)
    data = Dataset.from_frame(pd.DataFrame({f"gdp_{k}": rng.normal(size=12) for k in range(5)} | {"cpi": rng.normal(size=12)}, index=index))
    data = data.with_info("cpi", units="index", label="Consumer prices")
    frame = data.frame.copy()
    frame.loc[frame.index[:4], "gdp_3"] = np.nan
    return Dataset.from_frame(frame).with_info("cpi", units="index", label="Consumer prices")


def test_the_series_browser_filters_sorts_and_acts_on_the_series_picked(app, wide):
    from PySide6.QtCore import QItemSelectionModel, Qt

    from pymetrix.ui.qt import object_view

    done: dict[str, list[str]] = {}
    actions = {k: (lambda names, k=k: done.__setitem__(k, names)) for k in ("plot", "sheet", "select", "working_set")}
    tabs = object_view(wide, name="data", series_actions=actions)
    browser = tabs.browser
    assert browser.count.text() == "6 of 6 series: select series to act on them" and not any(b.isEnabled() for b in browser.buttons.values())
    browser.search.setText("consumer")  # any column: the label
    assert browser.proxy.rowCount() == 1 and browser.proxy.index(0, 0).data() == "cpi"
    browser.search.setText("gdp")
    assert browser.count.text() == "5 of 6 series: select series to act on them"
    columns = [browser.proxy.headerData(c, Qt.Orientation.Horizontal) for c in range(browser.proxy.columnCount())]
    observations = columns.index("Observations")  # headers in words: Source key, not source_key
    browser.view.sortByColumn(observations, Qt.SortOrder.AscendingOrder)  # sorted as numbers
    assert browser.proxy.index(0, 0).data() == "gdp_3" and browser.proxy.index(0, observations).data() == 8
    selection = browser.view.selectionModel()
    for row in (0, 1):
        selection.select(browser.proxy.index(row, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
    assert browser.count.text() == "5 of 6 series, 2 picked" and all(b.isEnabled() for b in browser.buttons.values())
    for key, button in browser.buttons.items():
        button.click()
        assert done[key] == ["gdp_3", browser.proxy.index(1, 0).data()]
    assert [b.text() for b in browser.buttons.values()] == ["Plot…", "Edit in spreadsheet", "New selection…", "Use as working set"]


def test_the_window_finds_anything_and_acts_on_picked_series(app, wide, monkeypatch):
    from pymetrix.ui import qt
    from pymetrix.ui.qt import main_window

    s = Session()
    s.objects["data"] = wide
    window = main_window(s)
    try:
        entries = window.palette_entries()
        titles = {e.title for e in entries}
        assert "Data → Transform → Seasonal adjustment…" in titles and "data" in titles and "View → Find anything…" not in titles
        assert any(e.kind == "Help" for e in entries)
        assert rank(entries, "seas adj")[0].title == "Data → Transform → Seasonal adjustment…"
        opened = []
        monkeypatch.setattr(qt, "command_dialog", lambda cls, session, parent=None, initial=None: opened.append(initial) or
                            type("Shut", (), {"exec": lambda self: False})())
        rank(entries, "data")[0].run()  # an object: its window
        actions = window.mdi.subWindowList()[-1].widget().browser.buttons
        assert set(actions) == {"plot", "sheet", "select", "generate", "working_set"}
        sub = window.mdi.subWindowList()[-1].widget()
        from PySide6.QtCore import QItemSelectionModel

        sub.browser.view.selectionModel().select(sub.browser.proxy.index(0, 0), QItemSelectionModel.SelectionFlag.Select
                                                 | QItemSelectionModel.SelectionFlag.Rows)
        picked = sub.browser.picked()
        sub.browser.buttons["plot"].click()
        assert type(opened[-1]).__name__ == "PlotSeries" and opened[-1].series == ", ".join(picked)
        sub.browser.buttons["select"].click()
        assert type(opened[-1]).__name__ == "SelectSeries" and opened[-1].name == "selection"
        window.mdi.subWindowList()[-1].widget().browser.buttons["sheet"].click()  # the spreadsheet: only those series
        sheet = next(w for w in window.findChildren(__import__("pymetrix.ui.sheet_qt", fromlist=["SheetWindow"]).SheetWindow))
        assert sheet.model.columns == picked and sheet.scope_action.text() == "Picked series only"
        assert "(1 picked)" in sheet.status.text()
    finally:
        s.dirty = False
        window.close()
