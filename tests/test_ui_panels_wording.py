"""The main window's panels, status bar and result windows as a first-time reviewer reads them: one empty-state
pattern inside each view, columns that leave room for the code, a status bar that follows the window in front,
plurals, and words in place of registry names and Python reprs."""

import os

import pytest

from pymetrix import Dataset
from pymetrix.commands import FitEquation, Session


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def data():
    return Dataset.example("sample_macro")


def _close(window):
    window.state["session"].dirty = False
    window.close()


def test_plural():
    from pymetrix.ui.messages import plural

    assert plural(1, "equation") == "1 equation" and plural(2, "equation") == "2 equations"
    assert plural(0, "series", "series") == "0 series" and plural(1200, "step") == "1,200 steps"


def test_empty_panels_say_so_inside_their_views(app):
    from pymetrix.ui.messages import EMPTY_STATES
    from pymetrix.ui.qt import main_window

    window = main_window(Session())
    try:
        window.resize(1200, 800)
        window.show()
        app.processEvents()
        empty = window.objects_empty
        assert empty.parent() is window.objects.viewport() and empty.text() == EMPTY_STATES["objects"]
        assert empty.isVisible() and empty.geometry().top() == 0
        assert window.objects.isColumnHidden(2)  # Status, while no result has one
        first = window.history.topLevelItem(0)
        assert first.text(0) == EMPTY_STATES["history"] and first.toolTip(0) == EMPTY_STATES["history_tip"]
        assert not window.history.isColumnHidden(0)  # the empty line spans the row from column 0: shown while empty
        assert all(text.startswith("No ") for key, text in EMPTY_STATES.items() if not key.endswith("_tip"))
        labels = window.status_labels
        assert labels["dataset"].text() == "No dataset" and not labels["sample"].isVisible()  # no empty cell
    finally:
        _close(window)


def test_history_leaves_room_for_the_code_and_names_the_menu(app, data):
    from pymetrix.ui.qt import main_window

    s = Session()
    s.objects["data"] = data
    s.run(FitEquation("c ~ y", name="fit"))
    window = main_window(s)
    try:
        history = window.history
        assert history.isColumnHidden(0) and history.isColumnHidden(4) and not history.isColumnHidden(2)
        item = history.topLevelItem(0)
        assert item.text(1) == "Equation" and item.toolTip(1).startswith("Estimate → Equation…, step 1 at ")
    finally:
        _close(window)


def test_status_bar_follows_the_window_in_front_and_gives_a_message_room(app, data, first_run):
    # first_run (tests/conftest.py): no arrangement saved by an earlier window of this test process (a test that docks
    # the Guide panel, another that widens Objects and History: together they left the workspace 711 pixels of a
    # 1300-pixel window on Windows CI)
    from pymetrix.ui.qt import main_window

    s = Session()
    s.objects["data"] = data
    s.run(FitEquation("c ~ y", name="fit"))
    s.run(FitEquation("c ~ y + i", name="fit2"))
    window = main_window(s)
    try:
        window.resize(1300, 800)
        window.show()
        app.processEvents()
        window.objects.setCurrentItem(None)
        window.show_object("fit")
        app.processEvents()
        assert window.status_labels["selected"].text() == "fit (Fitted equation)"
        sub = next(w for w in window.mdi.subWindowList() if w.windowTitle().startswith("fit "))
        assert sub.width() > 680  # most of the workspace, not a fixed 680 by 460
        window.show_object("fit2")
        app.processEvents()
        assert window.objects.currentItem().text(0) == "fit2"
        assert len(window.status_labels["last"].text()) <= len("Last step ✓: ") + 40
        window.statusBar().showMessage("Something to read in full")
        assert not window.status_labels["last"].isVisible()
        window.statusBar().clearMessage()
        assert window.status_labels["last"].isVisible()
    finally:
        _close(window)


def test_objects_status_column_shows_when_a_result_needs_attention(app, data):
    from pymetrix.commands import GenerateSeries
    from pymetrix.ui.qt import main_window

    s = Session()
    s.objects["data"] = data
    s.run(FitEquation("c ~ y", name="fit"))
    s.set_mode("manual")
    window = main_window(s)
    try:
        assert window.objects.isColumnHidden(2)
        s.run(GenerateSeries("y", "y * 1.01", replace=True))
        window.refresh()
        assert not window.objects.isColumnHidden(2)
        from PySide6.QtCore import Qt

        item = window.objects.findItems("fit", Qt.MatchFlag.MatchExactly, 0)[0]
        assert item.text(2) == "out of date"
    finally:
        _close(window)


def test_tables_are_as_wide_as_their_headers(app, data):
    from PySide6 import QtWidgets

    from pymetrix.ui.qt import object_view

    from pymetrix.var import VAR

    var = VAR(["c", "i", "y"], lags=2).fit(data)
    view = object_view(var, name="var")
    view.resize(900, 500)
    view.show()
    view.select("Roots")
    QtWidgets.QApplication.processEvents()
    grid = view.grids["Roots"]
    header = grid.horizontalHeader()
    for c in range(grid.columnCount()):
        text = grid.horizontalHeaderItem(c).text()
        assert grid.columnWidth(c) >= header.fontMetrics().horizontalAdvance(text), text
        assert grid.horizontalHeaderItem(c).toolTip() == text
    summary = str(var.summary())
    assert "Akaike info criterion (total): " in summary and "AIC_TOTAL" not in summary and "Final prediction error: 0.0000" not in summary
    view.close()


def test_result_windows_use_plain_words(app, data):
    from pymetrix.ui.qt import FIELDS_VIEW, object_view

    from pymetrix import Equation

    eq = Equation("c ~ y").fit(data)
    view = object_view(eq, name="fit", code="fit = Equation('c ~ y').fit(data)", add_to_report=lambda tab: None)
    assert view.made_by.text() == "Copy the step's code"
    assert view.add_result_to_report.text() == "Add result to report"
    assert FIELDS_VIEW == "Python access" and view.titles()[-1] == "Python access"
    view.close()


def test_model_editor_counts_in_words(app):
    from pymetrix.ui.messages import EMPTY_STATES
    from pymetrix.ui.model_editor_qt import ModelEditorWindow

    window = ModelEditorWindow(lambda c: True, text="parameters\n    rho = 0.5\nshocks\n    e = 1\nequations\n    y = rho*y[-1] + e\n",
                               name="m")
    assert "1 equation for 1 endogenous variable;" in window.status.text() and "0 errors" in window.status.text()
    if not window.last.diagnostics:
        assert window.problems.topLevelItem(0).text(0) == EMPTY_STATES["model_problems"]
    window.close()


def test_plugins_say_what_they_provide_in_words():
    from pymetrix.ui.plugins_qt import group_words

    assert group_words("report_notes", 1) == "1 report note" and group_words("panels", 2) == "2 panels"
    assert group_words("observers", 1) == "1 step observer" and group_words("io") == "Data readers and writers"


def test_help_contents_follow_the_page_and_settings_read_as_words(app):
    from pymetrix import help as hp
    from pymetrix.ui.help_qt import help_window

    sections = hp.contents()
    title, url = sections[1][1][0]
    window = help_window(None)
    window.view.go(url)
    current = window.tree.currentItem()
    assert current is not None and current.text(0) == title and current.parent().isExpanded()
    window.close()
    commands = dict(sections)["Commands"]
    page = hp.page_for_link(next(u for t, u in commands if t == "Equation"))
    assert "| Setting | Default |" in page
    rows = [line for line in page.splitlines() if line.startswith("| ") and "Setting" not in line and "---" not in line]
    data_row = next((r for r in rows if r.startswith(("| Data ", "| data "))), None)
    if data_row is not None:
        assert "'data'" not in data_row
    assert all(not r.split("|")[3].strip()[:1].islower() for r in rows if len(r.split("|")) > 3 and r.split("|")[3].strip())


def test_a_dataset_counts_periods_not_rows(data):
    from pymetrix.views import dataset_status

    assert "100 periods" in dataset_status(data) and "rows" not in dataset_status(data)


def test_the_window_has_the_app_icon(app):
    from pymetrix.ui.qt import main_window

    window = main_window(Session())
    try:
        assert not app.windowIcon().isNull()
    finally:
        _close(window)
