"""The report outline: figures, tables and results collected from every workspace, ordered, then exported."""

import os

import pytest

from pymetrix import Dataset
from pymetrix.commands import ExportReport, FitEquation, Session, load_project, save_project
from pymetrix.errors import NotFoundError


@pytest.fixture
def session():
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    s.run(FitEquation("dlog(c) ~ dlog(y)", name="fit"))  # a step, so a saved project rebuilds it
    return s


def test_the_outline_collects_orders_and_exports_as_code(session, tmp_path):
    s = session
    s.add_to_report("fit", "Graph", caption="Consumption fit")
    s.add_to_report("fit", "Coefficients")
    s.add_report_heading("Results")
    s.add_report_text("Consumption follows income.")
    s.move_report_item(2, 0)  # the heading first
    s.add_to_report("fit")  # the whole result: its summary and graph
    assert [i["kind"] for i in s.report_outline] == ["heading", "item", "item", "text", "item"]
    with pytest.raises(NotFoundError, match="no tab 'Nothing'"):
        s.add_to_report("fit", "Nothing")
    with pytest.raises(NotFoundError, match="no object named 'gone'"):
        s.add_to_report("gone")
    s.remove_report_item(4)
    path = tmp_path / "outline.html"
    command = ExportReport.from_outline(s, path=str(path), title="Consumption")
    assert command.validate(s) == []
    s.run(command)
    html = path.read_text(encoding="utf-8")
    assert html.index("Results") < html.index("Consumption fit") < html.index("Consumption follows income.")
    assert s.history[-1].code == (f"report = Report('Consumption')\n"
                                  "report.heading('Results', 2)\n"
                                  "report.add(fit, caption='Consumption fit', sections=['Graph'])\n"
                                  "report.add(fit, sections=['Coefficients'])\n"
                                  "report.text('Consumption follows income.')\n"
                                  f"report.save({str(path)!r})")
    replayed = tmp_path / "replayed.html"
    ns = {"fit": s["fit"]}
    exec("from pymetrix import *\n" + s.history[-1].code.replace(repr(str(path)), repr(str(replayed))), ns)  # the report again, without the app
    assert replayed.read_text(encoding="utf-8").count("<figure>") == html.count("<figure>") == 1
    reopened = load_project(save_project(s, tmp_path / "with_outline.pymx"))
    assert reopened.report_outline == s.report_outline  # the outline is saved with the project
    empty = Session()
    assert ExportReport.from_outline(empty).validate(empty) == ["the report outline is empty: use Add to report on a figure, a table or a result"]
    s.objects.pop("fit")
    assert "no object named 'fit'" in ExportReport.from_outline(s).validate(s)[0]


@pytest.fixture
def app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_the_report_panel_orders_edits_and_exports(app, session, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    from pymetrix.ui.report_qt import report_panel

    exported = []
    panel = report_panel(lambda: session, lambda: exported.append(list(session.report_outline)))
    assert panel.hint.isVisibleTo(panel) and not panel.buttons["export"].isEnabled()
    session.add_to_report("fit", "Graph")
    session.add_to_report("fit", "Coefficients")
    panel.refresh()
    assert [panel.listing.item(i).text() for i in range(2)] == ["fit — Graph", "fit — Coefficients"]
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Findings", True))
    panel.buttons["heading"].click()
    assert panel.listing.item(2).text() == "## Findings"
    panel.buttons["up"].click()  # the heading, one up
    assert [i["kind"] for i in session.report_outline] == ["item", "heading", "item"]
    panel.listing.setCurrentRow(0)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("The fit", True))
    panel.edit(0)  # a caption, by double-click
    assert session.report_outline[0]["caption"] == "The fit" and panel.listing.item(0).text() == "fit — Graph  “The fit”"
    panel.listing.setCurrentRow(2)
    panel.buttons["remove"].click()
    assert len(session.report_outline) == 2
    panel.buttons["export"].click()
    assert exported and exported[-1] == session.report_outline


def test_add_to_report_from_figures_tables_and_results_in_the_window(app, session, monkeypatch):
    from PySide6.QtWidgets import QPushButton

    from pymetrix.ui import qt
    from pymetrix.ui.qt import main_window

    window = main_window(session)
    try:
        window.show_object("fit")
        tabs = window.mdi.subWindowList()[-1].widget()
        graph = tabs.widget([tabs.tabText(i) for i in range(tabs.count())].index("Graph"))
        graph.add_to_report.click()  # under the figure
        coefficients = tabs.widget([tabs.tabText(i) for i in range(tabs.count())].index("Coefficients"))
        next(b for b in coefficients.findChildren(QPushButton) if b.text() == "Add to report").click()  # under the table
        tabs.add_result_to_report.click()  # the result: its summary and graph
        assert [(i["source"], i["tab"]) for i in session.report_outline] == [("fit", "Graph"), ("fit", "Coefficients"), ("fit", None)]
        assert window.report_view.listing.count() == 3 and "Added fit to the report (3 items" in window.statusBar().currentMessage()
        opened = []
        monkeypatch.setattr(qt, "command_dialog", lambda cls, s, parent=None, initial=None: opened.append(initial) or
                            type("Shut", (), {"exec": lambda self: False})())
        window.report_view.buttons["export"].click()
        assert isinstance(opened[-1], ExportReport) and '"tab": "Coefficients"' in opened[-1].outline
        window.switch_workspace("report")
        from PySide6.QtWidgets import QDockWidget

        dock = next(d for d in window.findChildren(QDockWidget) if d.windowTitle() == "Report")
        assert dock.isVisibleTo(window)
        window.switch_workspace("all")
    finally:
        session.dirty = False
        window.close()
        from PySide6.QtCore import QSettings

        QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope, "pymetrix", "desktop").remove("workspace")


def test_the_export_dialog_keeps_the_outline_it_was_opened_on(app, session):
    from pymetrix.ui.qt import command_dialog

    session.add_to_report("fit", "Graph")
    dialog = command_dialog(ExportReport, session, initial=ExportReport.from_outline(session))
    assert not dialog.row_visible("objects") and not dialog.row_visible("template")  # the outline chooses the items
    values = dialog.editors  # the hidden outline is carried to the command, not dropped
    from dataclasses import fields

    assert "outline" not in values and "outline" in {f.name for f in fields(ExportReport)}
