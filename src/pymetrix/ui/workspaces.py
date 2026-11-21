"""Workspaces: the app arranged for one stage of the work at a time.

Projects run through stages, Explore (look at the data), Estimate, Model, Forecast and Report, but not in a line:
a revision sends you back to the data, forecasting draws on estimation and models, and reporting goes on throughout.
So a workspace is a lens, not a gate: it shows the menus, panels and shortcuts of its stage and hides the rest,
switching is instant and keeps the session as it is, everything stays one search away (Find anything, Ctrl+K), and
All shows the whole app. Code is a way of working rather than a stage: the Code toggle adds the Script, Immediate and
Problems panels to any workspace. Each workspace remembers its own arrangement of panels.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["ALWAYS_MENUS", "CODE_DOCKS", "WORKSPACES", "Workspace", "all_workspaces", "install_workspaces", "keep_shortcuts", "stage_of"]

ALWAYS_MENUS = ("File", "Edit", "Data", "View", "Calculate", "Help")  # every workspace starts from data and keeps these
# (Edit: Undo and Redo apply to every stage's steps)
CODE_DOCKS = ("Script", "Immediate", "Problems")


from ..extension import Workspace  # defined without the user interface, so plugins can register one on import
from .keys import key_text


WORKSPACES: tuple[Workspace, ...] = (
    Workspace("all", "All", None, None, (), "the whole app"),
    Workspace("explore", "Explore", ("Test",), ("Objects", "History"),
              ("LoadData", "@spreadsheet", "PlotSeries", "CheckData", "SelectSeries", "RunTest", "Correlogram", "SeasonalAdjust",
               "Filter"),
              "look at the data first: load, check, plot, distributions, stationarity and seasonality"),
    Workspace("estimate", "Estimate", ("Estimate", "Test"), ("Objects", "History", "Problems"),
              ("FitEquation", "FitVAR", "FitLocalProjection", "RunTest", "CoefficientPlot", "Forecast"),
              "specify, fit, test and compare econometric models"),
    Workspace("model", "Model", ("Model",), ("Objects", "History", "Problems"),
              ("DefineEquation", "LoadTemplate", "CheckModel", "SolveModel", "EstimateModel", "ParameterSweep", "CompareScenarios"),
              "structural and semi-structural models: define, solve, estimate, simulate"),
    Workspace("forecast", "Forecast", ("Forecast", "Estimate"), ("Objects", "History"),
              ("MakeRound", "AdjustRound", "SaveRound", "CompareRounds", "ConditionalForecast", "CombineForecasts", "RunBacktest"),
              "forecast rounds, judgement, scenarios and evaluation"),
    Workspace("report", "Report", (), ("Objects", "Report", "History"),
              ("ExportReport", "PlotSeries", "@save_result", "@export_script"),
              "figures, tables, reports and exports"),
)


# which stage a result belongs to, by its type (or a base type): its badge on the stage bar when it needs attention
_STAGES = {
    "explore": ("Dataset", "Selection"),
    "model": ("FittedModel", "Template", "DynareModel", "Model", "ModelSystem", "Solution", "LinearSolution", "StateSpaceSystem",
              "Decomposition", "SweepResult", "ScenarioSet", "SecondOrderSolution", "ParticleFilter", "OptimalPolicy", "OSRResult", "Posterior",
              "Identification", "OccBinPath", "ModelCheck", "Translation", "EViewsXmlModel", "OptimalControl"),
    "forecast": ("ForecastRound", "Backtest", "ForecastStore"),
    "report": ("Figure", "Table"),
}


def all_workspaces() -> tuple[Workspace, ...]:
    """The core workspaces, with plugins' (registered under ``workspaces``) before Report."""
    from ..plugins import registry

    extra = tuple(e.obj for e in registry("workspaces").entries(available=True) if isinstance(e.obj, Workspace)
                  and e.obj.key not in {w.key for w in WORKSPACES})
    return WORKSPACES[:-1] + extra + WORKSPACES[-1:]


def workspace_numbers(workspaces: tuple[Workspace, ...]) -> dict[str, int]:
    """Each workspace's number, its key Ctrl+number: the core's 1 to 6 (All, Explore, Estimate, Model, Forecast,
    Report) whatever plugins are installed, as the docs say; plugins' workspaces 7 on, in their order on the bar."""
    core = [w.key for w in WORKSPACES]
    out = {key: i for i, key in enumerate(core, start=1)}
    for ws in workspaces:
        if ws.key not in out:
            out[ws.key] = len(out) + 1
    return out


def stage_of(obj: Any) -> str:
    """The stage a result belongs to: its class's ``stage`` (a plugin's result), else explore, estimate, model, forecast
    or report by type (estimation, by default)."""
    own = getattr(type(obj), "stage", None)
    if isinstance(own, str) and own:
        return own
    names = [c.__name__ for c in type(obj).__mro__]
    for stage, kinds in _STAGES.items():
        if any(n in kinds for n in names):
            return stage
    return "estimate"


_MENU_STAGES = {"Data": "explore", "Estimate": "estimate", "Test": "estimate", "Model": "model", "Forecast": "forecast"}
_COMMAND_STAGES = {"LoadData": "explore", "LoadExample": "explore",  # File menu steps that make data
                   "Forecast": "estimate"}  # in the Forecast menu, but its result (a Forecast) is the estimation stage's


def stage_of_command(cmd: Any) -> str:
    """The stage of the result a step makes, from where its dialog sits in the menus (for results not computed yet)."""
    from ..commands import COMMANDS

    if type(cmd).__name__ in _COMMAND_STAGES:
        return _COMMAND_STAGES[type(cmd).__name__]
    menu = next((i.menu for i in COMMANDS.values() if i.cls is type(cmd) and i.menu), "")
    return _MENU_STAGES.get(menu.partition("/")[0], "estimate")


def keep_shortcuts(window: Any) -> list[Any]:
    """Add every menu action that has a shortcut to the window itself, so its keys work whichever menus the workspace
    shows (an action only in a hidden menu does not answer its keys). A key sequence two actions share is left to the
    menus, where at most one of them is shown: on the window both would answer and neither would run. Returns the
    actions added."""
    found: dict[str, list[Any]] = {}

    def walk(menu: Any) -> None:
        for action in menu.actions():
            if action.menu() is not None and action.menu() is not menu:
                walk(action.menu())
            elif not action.isSeparator():
                for keys in action.shortcuts():
                    if not keys.isEmpty():
                        found.setdefault(keys.toString(), [])
                        if action not in found[keys.toString()]:
                            found[keys.toString()].append(action)

    for top in window.menuBar().actions():
        if top.menu() is not None:
            walk(top.menu())
    shared = {id(a) for owners in found.values() if len(owners) > 1 for a in owners}
    added: list[Any] = []
    for owners in found.values():
        for action in owners:
            if id(action) not in shared and action not in added and action not in window.actions():
                window.addAction(action)
                added.append(action)
    return added


def install_workspaces(window: Any, menus: dict[str, Any], docks: dict[str, Any], view_menu: Any, settings: Any,
                       run_command: Callable[[type], Any], app_actions: dict[str, tuple[str, Callable[[], Any]]]) -> Any:
    """Add the stage bar, the tools bar and View → Workspace to the main window; returns ``switch(key)``.

    ``menus``: every top menu by title; ``docks``: the panels by title; ``app_actions``: key -> (label, callable) for
    the tools that are not commands. Plugins' workspaces (``workspaces``) join the core ones."""
    workspaces = all_workspaces()
    from PySide6 import QtWidgets
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QAction, QActionGroup, QKeySequence

    from ..commands import COMMANDS

    by_class = {i.cls.__name__: i for i in COMMANDS.values() if i.menu}
    stages = QtWidgets.QToolBar("Workspaces", window)
    stages.setObjectName("workspaces")
    stages.setMovable(False)
    window.addToolBar(Qt.ToolBarArea.TopToolBarArea, stages)
    tools = QtWidgets.QToolBar("Workspace tools", window)
    tools.setObjectName("workspace_tools")
    window.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)  # the stage's tools on a row of their own, under the stages
    window.addToolBar(Qt.ToolBarArea.TopToolBarArea, tools)
    group = QActionGroup(window)
    group.setExclusive(True)
    submenu = QtWidgets.QMenu("Workspace", view_menu)
    view_menu.addMenu(submenu)
    actions: dict[str, Any] = {}
    numbers = workspace_numbers(workspaces)
    for ws in workspaces:
        number = numbers[ws.key]
        action = QAction(ws.title, window)
        action.setCheckable(True)
        if number <= 9:
            action.setShortcut(QKeySequence(f"Ctrl+{number}"))
        action.setToolTip(f"{ws.title}: {ws.tip}" + (f" ({key_text(f'Ctrl+{number}')})" if number <= 9 else ""))
        action.triggered.connect(lambda _=False, k=ws.key: switch(k))
        group.addAction(action)
        stages.addAction(action)
        submenu.addAction(action)
        actions[ws.key] = action
    stages.addSeparator()
    submenu.addSeparator()
    code = QAction("Code", window)
    code.setCheckable(True)
    code.setShortcut(QKeySequence("Ctrl+`"))
    code.setToolTip(f"the Script, Immediate and Problems panels, in any workspace ({key_text('Ctrl+`')})")
    code.toggled.connect(lambda on: (show_code(on, toggled=True), settings.setValue("workspace/code", on)))
    stages.addAction(code)
    submenu.addAction(code)
    current = {"key": None}

    def tool_actions(ws: Workspace) -> None:
        from ..wording import menu_label

        old = tools.actions()
        tools.clear()
        for action in old:
            action.deleteLater()
        for name in ws.tools:
            # each action made here and then added: the QAction that addAction(text) returns can come back as the
            # wrapper of an object deleted at the same address (a toolbar's layout item), which has no setToolTip
            if name.startswith("@"):
                label, run = app_actions[name[1:]]
                action = QAction(label, tools)
                action.triggered.connect(lambda _=False, r=run: r())
            elif name in by_class:
                info = by_class[name]
                action = QAction(menu_label(info.menu).rstrip(".…"), tools)
                action.setToolTip(f"{info.menu.partition('/')[0]} → {menu_label(info.menu)}")
                action.triggered.connect(lambda _=False, c=info.cls: run_command(c))
            else:
                continue
            tools.addAction(action)
        tools.setVisible(bool(ws.tools))

    before_code: dict[str, bool] = {}  # the code panels shown in All before Code was turned on

    def show_code(on: bool, toggled: bool = False) -> None:
        """Show the code panels (on), or only those the workspace has. In All, which has every panel, switching to
        it keeps them as they were; turning Code off there puts back those it showed before Code was turned on (the
        Immediate panel of a first run), not none."""
        ws = next(w for w in workspaces if w.key == current["key"]) if current["key"] else workspaces[0]
        if on and toggled:
            before_code.clear()
            before_code.update({t: not docks[t].isHidden() for t in CODE_DOCKS if t in docks})
        for title in CODE_DOCKS:
            if title in docks:
                if ws.docks is None:
                    kept = before_code.get(title, False) if toggled else not docks[title].isHidden()
                else:
                    kept = False
                docks[title].setVisible(on or (ws.docks is not None and title in ws.docks) or kept)
        if on and "Script" in docks:
            docks["Script"].raise_()

    def switch(key: str, remember: bool = True) -> None:
        """Show the workspace ``key``: its menus, panels (as last arranged there) and tools."""
        ws = next(w for w in workspaces if w.key == key)
        if current["key"] is not None and remember:  # leaving: keep this workspace's arrangement
            settings.setValue(f"workspace/{current['key']}/state", window.saveState())
            # and whether it was still the default one, its panels sized to the window (possibly not yet: fitting
            # waits for the event loop), not by the user: then it is fitted again when it comes back
            default = hasattr(window, "layout_fit") and window.layout_fit["pending"]
            settings.setValue(f"workspace/{current['key']}/default", bool(default))
        current["key"] = key
        for title, menu in menus.items():
            menu.menuAction().setVisible(ws.menus is None or title in ALWAYS_MENUS or title in ws.menus)
        saved = settings.value(f"workspace/{key}/state")
        if saved is not None:
            window.restoreState(saved)
        elif ws.docks is not None:
            for title, dock in docks.items():
                dock.setVisible(title in ws.docks)
        stages.setVisible(True)
        tool_actions(ws)
        show_code(code.isChecked())
        if hasattr(window, "layout_fit"):  # the workspace as the user arranged it, or its panels at their default sizes
            if saved is None or str(settings.value(f"workspace/{key}/default", False)).lower() == "true":
                window.fit_default_layout()  # the panels it shows, at their default sizes for this window
            else:
                window.layout_fit["pending"] = False
        actions[key].setChecked(True)
        settings.setValue("workspace/current", key)
        window.workspace = key

    def reset() -> None:
        """Forget the current workspace's arrangement: its panels as it first shows them."""
        settings.remove(f"workspace/{current['key']}/state")
        settings.remove(f"workspace/{current['key']}/default")
        switch(current["key"], remember=False)

    start = settings.value("workspace/current", "all")
    code.blockSignals(True)
    code.setChecked(str(settings.value("workspace/code", False)).lower() == "true")
    code.blockSignals(False)
    switch(start if start in actions else "all", remember=False)
    badges: dict[str, list[str]] = {}

    def update_badges(attention: dict[str, list[str]]) -> None:
        """Show, on each stage's button, how many of its results need attention (out of date, not computed); All
        counts them all. The names are in the tooltip."""
        badges.clear()
        badges.update(attention)
        everything = sorted({n for names in attention.values() for n in names})
        for ws in workspaces:
            names = everything if ws.key == "all" else sorted(attention.get(ws.key, []))
            action = actions[ws.key]
            action.setText(f"{ws.title} · {len(names)}" if names else ws.title)
            number = numbers[ws.key]
            action.setToolTip(f"{ws.title}: {ws.tip}" + (f" ({key_text(f'Ctrl+{number}')})" if number <= 9 else "")
                              + (f"\n{len(names)} need attention (out of date or not computed): {', '.join(names[:8])}"
                                 f"{'…' if len(names) > 8 else ''}" if names else ""))

    def switch_from_bar(key: str) -> None:
        """A stage chosen on the bar: with results needing attention, the Problems panel says which."""
        switch(key)
        names = badges.get(key, []) if key != "all" else [n for v in badges.values() for n in v]
        if names and "Problems" in docks:
            docks["Problems"].setVisible(True)
            docks["Problems"].raise_()

    for key, action in actions.items():
        action.triggered.disconnect()
        action.triggered.connect(lambda _=False, k=key: switch_from_bar(k))
    keep_shortcuts(window)
    window.workspace_actions, window.code_action, window.workspace_tools = actions, code, tools
    window.switch_workspace, window.reset_workspace, window.update_badges = switch, reset, update_badges
    return switch
