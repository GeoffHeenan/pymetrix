"""The Guide panel in the app: Open fills in the step's dialog, Show me runs it, ticks follow the history (Undo
takes one back), the demo a case study starts from (opened through the app's ``open_demo``, extension API 1.7, or a
hint where the app has none), the introduction before step 1 and the stages, and Lab 3 end to end through Show me."""

import os

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pymetrix_guide import PLUGIN, course_folders, register
from pymetrix_guide.app import create_panel, stage_strip, step_markdown
from PySide6.QtCore import Qt

from pymetrix import Dataset, demo as pymetrix_demo, help as pymetrix_help
from pymetrix.commands import FitEquation, GenerateSeries, Session
from pymetrix.extension import AppInterface

register()

CONSUMPTION = '''---
title: Consumption, guided
demo: consumption
---

Fit a dynamic consumption function.

```pymetrix-step
title: Fit the consumption function
command: FitEquation
settings: {formula: "c ~ y + c[-1]", name: fit}
check: object fit is FittedEquation
```

Add log output.

```pymetrix-step
title: Log output
command: GenerateSeries
settings: {series: ly, formula: "log(y)"}
check: series ly in data
```

Well done.
'''


INTRODUCED = '''---
title: Consumption, introduced
summary: Two consumption functions, and why
stages: [Data, Model, Report]
---

# Consumption, introduced

## Overview

How an applied study goes.

## This scenario

The data and the question.

---

Add log output.

```pymetrix-step
title: Log output
stage: Data
command: GenerateSeries
settings: {series: ly, formula: "log(y)"}
check: series ly in data
why: The model is in logs.
look: ly is in data.
```

```pymetrix-step
title: Fit the consumption function
stage: Model
command: FitEquation
settings: {formula: "c ~ y + c[-1]", name: fit}
check: object fit is FittedEquation
why: The dynamic function.
```

Well done.
'''


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def course(tmp_path):
    folder = tmp_path / "econ101"
    folder.mkdir()
    (folder / "consumption.md").write_text(CONSUMPTION, encoding="utf-8")
    (folder / "introduced.md").write_text(INTRODUCED, encoding="utf-8")
    (folder / "notes.md").write_text("# Notes\n\nNot a case study.\n", encoding="utf-8")
    pymetrix_help.load_course(folder)
    yield folder
    pymetrix_help.unload_course(folder)


def _panel(session, opened=None, demo=None):
    messages = []
    opened = [] if opened is None else opened

    def run(cmd):
        result = session.run(cmd)
        interface._changed()  # as the app does after each step
        return result

    interface = AppInterface(lambda: session, lambda: None, lambda cls, initial: opened.append((cls, initial)), run,
                             messages.append, _open_demo=demo)  # no demo: open_demo raises NotSupportedError
    widget = create_panel(interface)
    widget.session, widget.interface, widget.messages, widget.opened = session, interface, messages, opened
    return widget


def _ticks(panel):
    return [panel.steps.item(i).checkState() == Qt.CheckState.Checked for i in range(panel.steps.count())]


def test_the_panel_lists_case_studies_from_the_loaded_courses(app, course):
    panel = _panel(Session())
    titles = [panel.choice.itemText(i) for i in range(panel.choice.count())]
    assert "Consumption, guided" in titles and "Lab 3, guided. A quarterly forecast round · introductory" in titles
    assert "Notes" not in titles


def test_the_introduction_comes_first_and_the_stages_show_where_a_step_is(app, course):
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    panel = _panel(s)
    panel.choose("Consumption, introduced")
    assert panel.summary.isVisibleTo(panel) and panel.summary.text() == "Two consumption functions, and why"
    assert (panel.page, panel.index) == (0, 0) and panel.step() is None
    assert panel.progress.text() == "Introduction, page 1 of 2 · then 2 steps · 0 done"
    assert panel.text.toPlainText().startswith("Overview") and "How an applied study goes." in panel.text.toPlainText()
    assert panel.check_label.text() == "Next: step 1, Log output."
    assert not panel.back_button.isEnabled() and panel.next_button.isEnabled()
    assert not panel.open_button.isEnabled() and not panel.show_button.isEnabled()
    assert panel.stages.isVisibleTo(panel) and "<b>" not in panel.stages.text()  # the whole workflow, none current
    assert panel.steps.currentRow() == -1 and panel.steps.item(0).toolTip() == "Stage: Data"
    panel.next_button.click()
    assert panel.page == 1 and "The data and the question." in panel.text.toPlainText() and panel.back_button.isEnabled()
    panel.next_button.click()  # off the introduction: step 1
    assert (panel.page, panel.index) == (None, 0) and panel.progress.text() == "Stage: Data — step 1 of 2 · 0 done"
    assert "Add log output." in panel.text.toPlainText() and "This scenario" not in panel.text.toPlainText()
    assert "Why this step. The model is in logs." in panel.text.toPlainText() and "What to look for. ly is in data." in panel.text.toPlainText()
    assert "<b>Data</b>" in panel.stages.text() and '<span style="color: #8a8a8a">Model</span>' in panel.stages.text()
    assert panel.show_button.isEnabled() and panel.back_button.isEnabled()
    panel.show_button.click()
    panel.next_button.click()
    assert panel.progress.text() == "Stage: Model — step 2 of 2 · 1 done" and "<b>Model</b>" in panel.stages.text()
    assert "What to look for" not in panel.text.toPlainText()  # a step without look: no such part
    panel.back_button.click()
    panel.back_button.click()  # from step 1 back to the introduction's last page
    assert panel.page == 1 and panel.progress.text().startswith("Introduction, page 2 of 2")
    panel.steps.itemClicked.emit(panel.steps.item(1))  # the list goes straight to a step
    assert (panel.page, panel.index) == (None, 1)
    panel.go(2)
    assert panel.progress.text().endswith("finished · 1 of 2 steps done") and "<b>" not in panel.stages.text()
    panel.restart_button.click()  # Start over: the overview
    assert (panel.page, panel.index) == (0, 0) and panel.messages[-1].startswith("Guide: back to the start")
    panel.go(1)
    panel.choose("Consumption, guided")  # no introduction, no stages: step 1 first, as before
    assert panel.page is None and panel.progress.text() == "Step 1 of 2 · 1 done" and not panel.stages.isVisibleTo(panel)
    assert not panel.summary.isVisibleTo(panel) and not panel.back_button.isEnabled()
    panel.choose("Consumption, introduced")  # each case study keeps its place
    assert (panel.page, panel.index) == (None, 1)


def test_the_stage_strip():
    from pymetrix_guide import parse

    study = parse(INTRODUCED)
    assert stage_strip(study, "Model") == 'Workflow: Data › <b>Model</b> › <span style="color: #8a8a8a">Report</span>'
    assert stage_strip(study) == stage_strip(study, finished=True) == "Workflow: Data › Model › Report"
    assert stage_strip(parse(CONSUMPTION)) == ""


def test_open_fills_the_dialog_and_show_me_runs_the_step(app, course):
    from pymetrix.ui.qt import command_dialog

    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    panel = _panel(s)
    panel.choose("Consumption, guided")
    assert panel.progress.text() == "Step 1 of 2 · 0 done" and _ticks(panel) == [False, False]
    assert "Fit a dynamic consumption function." in panel.text.toPlainText() and "To do: fit is a FittedEquation" in panel.check_label.text()
    panel.open_button.click()
    (cls, initial), = panel.opened
    assert cls is FitEquation and initial == FitEquation("c ~ y + c[-1]", name="fit")
    dialog = command_dialog(cls, s, initial=initial)  # what the app opens: the dialog filled in
    assert dialog.editors["formula"].get() == "c ~ y + c[-1]" and dialog.editors["name"].get() == "fit"
    assert not s.history  # nothing runs until the trainee presses OK
    panel.show_button.click()
    assert s.history[-1].command == FitEquation("c ~ y + c[-1]", name="fit") and _ticks(panel) == [True, False]
    assert panel.check_label.text().startswith("✓ Done") and panel.progress.text() == "Step 1 of 2 · 1 done"
    assert "fit = Equation('c ~ y + c[-1]')" in s.script()  # an ordinary recorded step


def test_the_tick_follows_the_history_and_undo_takes_it_back(app, course):
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    panel = _panel(s)
    panel.choose("Consumption, guided")
    panel.next_button.click()
    assert panel.progress.text() == "Step 2 of 2 · 0 done" and panel.back_button.isEnabled()
    panel.interface.run(FitEquation("c ~ y + c[-1]", name="fit"))  # the trainee's own step, changed or not, ticks it
    assert _ticks(panel) == [True, False]
    s.undo()
    panel.interface._changed()  # the app tells its panels after Undo
    assert _ticks(panel) == [False, False]
    s.redo()
    panel.interface._changed()  # and after Redo: the tick comes back
    assert _ticks(panel) == [True, False]
    s.undo()
    panel.interface._changed()
    panel.next_button.click()
    assert panel.progress.text().endswith("finished · 0 of 2 steps done") and "Well done." in panel.text.toPlainText()
    assert not panel.next_button.isEnabled() and not panel.show_button.isEnabled()
    panel.steps.itemClicked.emit(panel.steps.item(0))
    assert panel.index == 0


def test_a_step_that_fails_says_so(app, course):
    panel = _panel(Session())  # no data
    panel.choose("Consumption, guided")
    panel.show_button.click()
    assert "did not run" in panel.check_label.text() and panel.messages[-1].startswith("Guide: the step did not run")


def test_a_case_study_offers_its_demo_and_start_over_reloads_it(app, course):
    import pymetrix

    opened = []
    sessions = [Session()]

    def open_demo(name):  # as the app's: the demo replaces the session, and the panels hear of it
        opened.append(name)
        sessions[0] = pymetrix.demo(name)
        interface._changed()

    messages = []
    interface = AppInterface(lambda: sessions[0], lambda: None, lambda cls, initial: None, lambda c: sessions[0].run(c), messages.append,
                             _open_demo=open_demo)  # extension API 1.7
    panel = create_panel(interface)
    panel.choose("Consumption, guided")
    assert panel.start_bar.isVisibleTo(panel) and "consumption" in panel.start_text.text()
    panel.demo_button.click()
    assert opened == ["consumption"] and sessions[0].demo == "consumption" and not panel.start_bar.isVisibleTo(panel)
    first = sessions[0]
    panel.go(1)
    panel.restart_button.click()  # Start over: step 1, and the demo as it first was
    assert opened == ["consumption", "consumption"] and panel.index == 0
    assert sessions[0] is not first and sessions[0].demo == "consumption"  # built afresh, as it first was
    assert not panel.start_bar.isVisibleTo(panel) and not messages


def test_without_the_apps_open_demo_it_says_where_to_find_it(app, course):
    panel = _panel(Session())  # an interface without a way to open demos: open_demo raises NotSupportedError
    panel.choose("Consumption, guided")
    panel.demo_button.click()
    assert panel.messages[-1] == "Guide: open the demo with File → Open demo project… (consumption)"


def test_an_app_before_extension_api_1_6_gets_the_menu_hint(app, course):
    class OlderInterface:  # extension API 1.6 or earlier: no open_demo at all
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name == "open_demo":
                raise AttributeError(name)
            return getattr(self._inner, name)

    s = pymetrix_demo("consumption")
    messages = []
    inner = AppInterface(lambda: s, lambda: None, lambda cls, initial: None, s.run, messages.append)
    panel = create_panel(OlderInterface(inner))
    panel.choose("Consumption, guided")
    assert not panel.start_bar.isVisibleTo(panel)  # the demo is open already
    panel.restart_button.click()
    assert panel.index == 0 and messages[-1] == "Guide: open the demo with File → Reload demo project (consumption)"


def test_a_demo_the_app_cannot_open_is_said_not_raised(app, course):
    from pymetrix.errors import NotFoundError

    def open_demo(name):
        raise NotFoundError(f"no demo {name!r}; demos: forecast_round")

    panel = _panel(Session(), demo=open_demo)
    panel.choose("Consumption, guided")
    panel.demo_button.click()
    assert panel.messages[-1] == "Guide: the demo consumption did not open: no demo 'consumption'; demos: forecast_round"


def test_the_guide_in_the_app_opens_its_demo_and_start_over_reloads_it(app, course, monkeypatch):
    """Through the real app: the Guide panel's offer and Start over open the demo with File → Open demo project's
    code, which asks to save unsaved work first."""
    from pymetrix.ui.qt import main_window

    asked = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        lambda *a, **k: asked.append(a[1]) or QtWidgets.QMessageBox.StandardButton.Discard)
    s = Session()
    window = main_window(s)
    try:
        panel = window.plugin_docks["Guide"].widget()
        panel.choose("Consumption, guided")
        assert panel.start_bar.isVisibleTo(panel)
        panel.demo_button.click()  # nothing unsaved: no question
        first = window.state["session"]
        assert not asked and first is not s and first.demo == "consumption" and not panel.start_bar.isVisibleTo(panel)
        first.dirty = True  # the trainee's work, unsaved: Start over asks first
        panel.go(1)
        panel.restart_button.click()
        assert asked == ["Unsaved work"] and panel.index == 0
        assert window.state["session"] is not first and window.state["session"].demo == "consumption"
    finally:
        window.state["session"].dirty = False
        window.close()


def test_lab_3_runs_end_to_end_through_show_me(app, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the forecast store and the report are written next to the project
    s = Session()
    panel = _panel(s)
    panel.choose("Lab 3, guided. A quarterly forecast round")
    n = len(panel.study.steps)
    assert panel.page == 0 and panel.text.toPlainText().startswith("Overview: the forecast workflow")
    for _ in panel.study.intro:  # the introduction's pages, then step 1
        assert not panel.show_button.isEnabled()
        panel.next_button.click()
    assert panel.progress.text() == "Stage: Data — step 1 of 19 · 0 done"
    for i in range(n):
        assert panel.index == i and not _ticks(panel)[i]
        panel.show_button.click()
        assert _ticks(panel)[i], (panel.study.steps[i].title, panel.check_label.text())
        assert panel.progress.text().startswith(f"Stage: {panel.study.steps[i].stage} — step {i + 1} of {n}")
        panel.next_button.click()
    assert all(_ticks(panel)) and panel.progress.text().endswith(f"finished · {n} of {n} steps done")
    assert len(s.history) == n and all(r.ok for r in s.history)
    assert (tmp_path / "lab3_report.html").exists() and (tmp_path / "forecasts").is_dir()
    appendix = (tmp_path / "lab3_report.html").read_text(encoding="utf-8")  # the backtest's row: its origins, not the first two
    assert ("origins 2015Q1 to 2022Q4 (32 re-estimates), estimated on 2000Q3 to 2015Q1 at the first and 2000Q3 to 2022Q4 at the last"
            in appendix) and "2015Q1 to 2015Q2" not in appendix
    assert "forecast_round" in s.script()
    assert "## Finished" in step_markdown(panel.study, n)


def test_the_app_docks_the_guide(app):
    from pymetrix.ui.qt import main_window

    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    window = main_window(s)
    try:
        dock = window.plugin_docks["Guide"]
        assert dock.toolTip() == f"from the plugin {PLUGIN.name}"
        panel = dock.widget()
        panel.choose("Lab 3, guided. A quarterly forecast round")
        panel.go(1)
        assert panel.progress.text() == "Stage: Data — step 2 of 19 · 1 done"  # the data are there already
        window.app_interface.run(GenerateSeries("ly", "log(y)"))  # the app's runner: the panel follows
        assert panel.progress.text() == "Stage: Data — step 2 of 19 · 2 done" and panel.check_label.text().startswith("✓ Done")
        window.undo_step()  # Edit → Undo
        assert panel.progress.text() == "Stage: Data — step 2 of 19 · 1 done" and panel.check_label.text().startswith("To do")
        window.redo_step()  # Edit → Redo
        assert panel.progress.text() == "Stage: Data — step 2 of 19 · 2 done" and panel.check_label.text().startswith("✓ Done")
    finally:
        s.dirty = False
        window.close()


def test_the_guide_dock_fits_when_small(app, look):
    """Made small, the Guide keeps its widgets apart and scrolls what does not fit (the checks of
    tests/test_window_fit.py), floating or docked in a small app window, whatever the font and style: no narrower than
    the case-study choice and its reload button (pinned at the top), never held wider by what scrolls."""
    import importlib.util
    from pathlib import Path

    from pymetrix.ui.qt import main_window

    path = Path(__file__).resolve().parents[3] / "tests" / "_fit_check.py"
    if not path.exists():
        pytest.skip("the app's window checks are not here (an installed pymetrix-guide)")
    spec = importlib.util.spec_from_file_location("_guide_fit_check", path)
    fc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fc)
    from PySide6.QtCore import QSettings

    settings = QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope, "pymetrix", "desktop")
    kept = {key: settings.value(key) for key in settings.allKeys()}  # put back afterwards: closed, the window saves
    # its arrangement, the Guide docked, which later windows of this test process would otherwise open with
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    window = main_window(s)
    try:
        window.show()
        dock = window.plugin_docks["Guide"]
        dock.setFloating(True)  # sized on its own, as a trainee may leave it
        dock.show()
        panel = dock.widget()
        panel.choose("Lab 3, guided. A quarterly forecast round")
        top = panel.layout().itemAt(0).layout()  # the choice and the reload button: the panel's least width

        def held_by_the_choice_only():
            """The dock as narrow as asked (230, the last of the sizes checked) or as the choice and its button allow
            (with this font: 12 characters of the choice, wider than 230 where every character is a 12-pixel box, as on
            Windows CI), never wider: the panel's least width is the pinned row's, not the part that scrolls, nor the
            buttons."""
            assert panel.minimumSizeHint().width() == top.minimumSize().width()
            assert dock.size().width() == max(230, dock.minimumSizeHint().width())

        for page in range(len(panel.study.intro)):  # the introduction's pages: long text, which scrolls
            panel.go_intro(page)
            assert fc.check_sizes(dock, resize=dock.resize) == [], f"introduction page {page + 1}"
            held_by_the_choice_only()
            assert panel.scroll.verticalScrollBar().isVisible()
        for index in (0, 5, 7, len(panel.study.steps)):
            panel.go(index)
            assert fc.check_sizes(dock, resize=dock.resize) == [], index
        held_by_the_choice_only()
        assert panel.scroll.verticalScrollBar().isVisible()  # scrolls, not squashed
        dock.setFloating(False)  # docked again, in an app window made small
        window.resize(640, 400)
        app.processEvents()
        assert window.width() <= 640 and fc.problems(dock) == []
    finally:
        s.dirty = False
        window.close()
        settings.clear()
        for key, value in kept.items():
            settings.setValue(key, value)


def test_the_plugin_conforms():
    """The conformance kit's metadata checks (``check_plugin`` checks declarative extensions, methods, tests and
    the like, beyond these; the guide has none: it registers a panel and a course), and the course check."""
    from pymetrix.testing import check_course

    from pymetrix import plugins

    assert PLUGIN.supported and PLUGIN.version and PLUGIN.license == "BSD-3-Clause" and PLUGIN.maintainer
    assert any(e.plugin.name == PLUGIN.name for e in plugins.registry("panels").entries())
    assert PLUGIN.name in plugins.FIRST_PARTY  # the app does not stop to ask about it
    for folder in course_folders():
        check_course(folder)


def test_only_the_public_surface():
    """The plugin imports from pymetrix only through the extension API and the stable modules (and the help's
    course material, ``pymetrix.help``, which the courses group is built on)."""
    import ast
    from pathlib import Path

    from pymetrix.extension import STABLE

    bad = []
    for f in sorted((Path(__file__).resolve().parents[1] / "src" / "pymetrix_guide").rglob("*.py")):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 and node.module.split(".")[0] == "pymetrix":
                if node.module not in STABLE:
                    bad.append(f"{f.name}: from {node.module}")
                bad += [f"{f.name}: {node.module}.{a.name}" for a in node.names if a.name.startswith("_")]
            elif isinstance(node, ast.Import):
                bad += [f"{f.name}: import {a.name}" for a in node.names if a.name.split(".")[0] == "pymetrix" and a.name not in STABLE]
    assert not bad, "imports outside the public surface:\n" + "\n".join(bad)


def test_a_deleted_panel_is_not_refreshed(app, course):
    """The app (or an app before extension API 1.7, which keeps calling it) refreshing a Guide panel Qt has deleted:
    nothing touches its widgets."""
    import shiboken6

    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    panel = _panel(s)
    refresh, interface = panel.refresh, panel.interface
    shiboken6.delete(panel)
    refresh()  # no crash, no RuntimeError
    interface._changed()
    assert interface._listeners == []
