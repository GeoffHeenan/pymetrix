"""Extension API 1.1: a plugin adds a workspace, lays out its own menu, docks a panel that works the app through its
interface, and gives its results a stage and next steps; the app says which plugin a dialog or step comes from,
lists plugins that could not load, and checks a plugin on request."""

import os
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import pytest

from pymetrix import Dataset, plugins
from pymetrix.commands import GenerateSeries, Session
from pymetrix.commands.registry import COMMAND_IDS, COMMANDS, Command, Widget, command

DEMO = plugins.PluginInfo("pymetrix-demo", "0.2", license="MIT", maintainer="Ann Example", description="a demonstration")


@pytest.fixture
def app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class DemoResult:
    """A plugin's result that says its stage and next steps."""

    stage = "forecast"
    next_steps = (("Plot it…", "PlotSeries", "data", "explore"),)


@pytest.fixture
def demo_plugin(monkeypatch, tmp_path):
    from pymetrix.ui.extend import AppPanel, MenuLayout
    from pymetrix.ui.workspaces import Workspace

    monkeypatch.setenv("PYMETRIX_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(plugins, "_STATE", None)
    saved = ({g: dict(r._entries) for g, r in plugins._REGISTRIES.items()}, dict(plugins._PLUGINS), dict(plugins._PROBLEMS))
    seen: dict[str, Any] = {}
    with plugins.plugin_context(DEMO):
        @command(menu="Demo/Double...")
        @dataclass(frozen=True)
        class DemoDouble(Command):
            """Double a series (a demonstration dialog from a plugin)."""

            series: Annotated[str, Widget("text", help="the series")] = "y"
            name: str = "data"

            def run(self, session: Any) -> Any:
                return session.run(GenerateSeries(f"{self.series}2", f"2 * {self.series}"))

            def to_code(self) -> str:
                return f"# double {self.series}"

        @command(menu="Demo/Halve...")
        @dataclass(frozen=True)
        class DemoHalve(Command):
            """Halve a series."""

            series: Annotated[str, Widget("text", help="the series")] = "y"

            def run(self, session: Any) -> Any:
                return None

            def to_code(self) -> str:
                return "# halve"

        plugins.register("workspaces", Workspace("demo", "Demo", ("Demo",), ("Objects", "History", "Demo panel"), ("DemoDouble",),
                                                 "a plugin's own stage"), name="demo")
        plugins.register("menus", MenuLayout("Demo", ["DemoHalve", None, "DemoDouble"]), name="demo_menu")

        def make(app_interface):
            from PySide6 import QtWidgets

            seen["app"] = app_interface
            seen["changes"] = 0
            app_interface.on_history_changed(lambda: seen.__setitem__("changes", seen["changes"] + 1))
            return QtWidgets.QLabel("demo")

        plugins.register("panels", AppPanel("Demo panel", make, area="right"), name="demo_panel")
    yield {"double": DemoDouble, "halve": DemoHalve, "seen": seen}
    for cls in (DemoDouble, DemoHalve):
        COMMANDS.pop(cls.__name__, None)
        for cid in [c for c, i in COMMAND_IDS.items() if i.cls is cls]:
            COMMAND_IDS.pop(cid)
    for g, entries in saved[0].items():
        plugins._REGISTRIES[g]._entries = entries
    plugins._PLUGINS.clear(), plugins._PLUGINS.update(saved[1])
    plugins._PROBLEMS.clear(), plugins._PROBLEMS.update(saved[2])
    plugins._STATE = None


def test_a_plugins_results_say_their_stage_and_next_steps():
    from pymetrix.ui.next_steps import steps_for
    from pymetrix.ui.workspaces import stage_of

    assert stage_of(DemoResult()) == "forecast"
    steps = steps_for(DemoResult())
    assert [(label, cls.__name__, field, ws) for label, cls, field, ws in steps] == [("Plot it…", "PlotSeries", "data", "explore")]


def test_a_plugins_menu_is_laid_out_as_it_says(demo_plugin):
    from pymetrix.commands.menus import menu_entries

    infos = [i for i in COMMANDS.values() if i.menu and i.menu.startswith("Demo/")]
    entries = menu_entries("Demo", infos)
    assert [(e.kind, e.label) for e in entries] == [("command", "Halve…"), ("separator", ""), ("command", "Double…")]


def test_the_app_takes_a_plugins_workspace_panel_and_marks(app, demo_plugin):
    from PySide6.QtWidgets import QDockWidget

    from pymetrix.ui.qt import command_dialog, main_window

    plugins._PROBLEMS["pymetrix-broken"] = "could not be imported: No module named 'broken'"
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    window = main_window(s)
    try:
        assert "demo" in window.workspace_actions and window.workspace_actions["demo"].text() == "Demo"
        dock = window.plugin_docks["Demo panel"]
        assert isinstance(dock, QDockWidget) and dock.toolTip() == "from the plugin pymetrix-demo"
        window.switch_workspace("demo")  # its workspace shows its panel
        assert dock.isVisibleTo(window)
        window.switch_workspace("all")
        app_interface = demo_plugin["seen"]["app"]
        assert app_interface.session is s and app_interface.selected() in (None, "data")
        before = demo_plugin["seen"]["changes"]
        app_interface.run(GenerateSeries("z", "y + 1"))  # through the app's runner: recorded, undoable, refreshed
        assert "z" in s.objects["data"].names and s.undo_label().startswith("data = data.with_series(z=")
        assert demo_plugin["seen"]["changes"] > before
        assert any(where == "plugin pymetrix-broken" and "could not load" in what for where, what in window.problem_list)
        dialog = command_dialog(demo_plugin["double"], s)
        assert dialog.plugin_mark.text() == "From the plugin pymetrix-demo 0.2 (MIT)"
        assert not hasattr(command_dialog(GenerateSeries, s), "plugin_mark")  # pymetrix's own: no mark
        s.run(demo_plugin["double"]("y"))
        window.refresh()
        from pymetrix.ui.history_qt import HistoryPanel

        history = window.findChild(HistoryPanel)
        history.show_session(s)
        texts = [history.topLevelItem(i).text(1) for i in range(history.topLevelItemCount())]
        assert any(t.endswith("· demo") for t in texts)  # the step says which plugin it is from
    finally:
        s.dirty = False
        window.close()


def test_check_plugin_reports(demo_plugin):
    from pymetrix.ui.plugins_qt import check_report

    assert check_report("pymetrix-nowhere").endswith("nothing to check.")
    report = check_report("pymetrix-demo")
    assert report.startswith(("Passed", "A check failed", "Checking stopped"))


def test_open_demo_is_part_of_the_panel_interface_with_a_default():
    """Extension API 1.7: ``AppInterface.open_demo(name)``; an interface made without a way to open demos (as a
    plugin's own tests make one) says so with ``NotSupportedError``, an ``AttributeError``."""
    from pymetrix.errors import NotSupportedError
    from pymetrix.extension import AppInterface

    assert tuple(map(int, plugins.API.split("."))) >= (1, 7) and callable(AppInterface.open_demo)
    bare = AppInterface(lambda: None, lambda: None, lambda cls, initial: None, lambda c: None, lambda text: None)
    with pytest.raises(NotSupportedError, match="Open demo project"):
        bare.open_demo("forecast_round")
    opened = []
    wired = AppInterface(lambda: None, lambda: None, lambda cls, initial: None, lambda c: None, lambda text: None,
                         _open_demo=opened.append)
    wired.open_demo("forecast_round")
    assert opened == ["forecast_round"]


def test_a_panel_opens_a_demo_through_the_app_which_asks_to_save_first(app, monkeypatch):
    from PySide6 import QtWidgets

    from pymetrix.errors import NotFoundError
    from pymetrix.ui.qt import main_window

    asked = []
    answer = [QtWidgets.QMessageBox.StandardButton.Cancel]
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: asked.append(a[1]) or answer[0])
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    window = main_window(s)
    try:
        app_interface = window.app_interface
        changes = []
        app_interface.on_history_changed(lambda: changes.append(1))
        with pytest.raises(NotFoundError, match="demos: .*forecast_round"):
            app_interface.open_demo("no_such_demo")
        assert not asked and window.state["session"] is s  # an unknown name: no prompt, nothing changes
        s.dirty = True  # unsaved work: the app asks, and Cancel keeps it
        app_interface.open_demo("forecast_round")
        assert asked == ["Unsaved work"] and window.state["session"] is s
        answer[0] = QtWidgets.QMessageBox.StandardButton.Discard
        app_interface.open_demo("forecast_round")
        opened = window.state["session"]
        assert len(asked) == 2 and opened is not s and opened.demo == "forecast_round" and "forecast_round" in opened.objects
        assert app_interface.session is opened and changes  # panels follow the new session
        opened.dirty = False  # no unsaved work: no prompt
        app_interface.open_demo("forecast_round")
        assert len(asked) == 2 and window.state["session"] is not opened and window.state["session"].demo == "forecast_round"
    finally:
        s.dirty = False
        window.state["session"].dirty = False
        window.close()


def test_a_panel_in_an_unknown_area_is_a_specification_error():
    from pymetrix.errors import SpecificationError
    from pymetrix.extension import AppPanel

    with pytest.raises(SpecificationError, match="left, right or bottom"):
        AppPanel("Somewhere", lambda app: None, area="top")


def test_a_deleted_panels_refresh_is_dropped_not_called(app):
    """A panel's ``refresh`` (a method of a Qt object) given to ``on_history_changed`` is called while the panel
    lives; once Qt deletes the panel it is dropped, never called on its deleted widgets."""
    import shiboken6
    from PySide6 import QtWidgets

    from pymetrix.extension import AppInterface

    class Panel(QtWidgets.QWidget):
        calls = 0

        def refresh(self):
            type(self).calls += 1
            self.setToolTip("refreshed")  # touches the widget: a crash once it is deleted

    interface = AppInterface(lambda: None, lambda: None, lambda cls, initial: None, lambda c: None, lambda text: None)
    panel = Panel()
    interface.on_history_changed(panel.refresh)
    interface._changed()
    assert Panel.calls == 1
    shiboken6.delete(panel)
    assert interface._listeners == []
    interface._changed()
    assert Panel.calls == 1


def test_a_closed_windows_panels_do_not_follow_code_run_in_a_later_window(app):
    """Every window's console shares IPython's one shell: a closed window's console takes its hooks off it, and its
    panels stop following the history, so code run in a later window does not refresh the closed window's panels
    (with Qt having deleted their widgets, that refresh crashed the process: tests/test_windows.py on CI)."""
    pytest.importorskip("qtconsole")
    from pymetrix.ui.qt import main_window

    def opened():
        window = main_window(Session())
        window.show()
        app.processEvents()
        return window

    def run(window, code):
        window.send_immediate(code)
        for _ in range(3):
            app.processEvents()

    first = opened()
    calls = []
    first.app_interface.on_history_changed(lambda: calls.append(1))
    run(first, "a = 1")
    assert calls and first.state["session"]["a"] == 1
    first.state["session"].dirty = False
    first.close()
    shell = first.immediate.shell
    del calls[:]
    second = opened()
    try:
        run(second, "b = 2")
        assert second.state["session"]["b"] == 2
        assert calls == []  # the closed window's panels left alone
        assert first.immediate._post not in shell.events.callbacks["post_run_cell"]
        assert [c.__self__ for c in shell.events.callbacks["post_run_cell"] if hasattr(c, "__self__")] == [second.immediate]
        first.app_interface.on_history_changed(lambda: calls.append(1))  # too late: its window has closed
        run(second, "c = 3")
        assert calls == []
    finally:
        second.state["session"].dirty = False
        second.close()
